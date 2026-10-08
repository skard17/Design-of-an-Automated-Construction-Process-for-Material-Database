#!/usr/bin/env python3
"""Independently adjudicate generated schema leaves absent from a manual reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import evaluate_superconductivity_three_stage as evaluator
import prompt_quality_code_agent as prompt_agent
import run_artifact_guard


ALLOWED_CLASSIFICATIONS = {
    "useful_supplement",
    "reasonable_specialization",
    "redundant",
    "workflow_artifact",
    "misbound",
    "unclear_requires_review",
}
PIPELINE = "generated_schema_extras_adjudication_fresh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build_prompt(
    manual_paths: list[str],
    missing_manual_paths: list[str],
    extra_paths: list[str],
) -> str:
    contract = {
        "task": "automated materials database construction from scientific literature",
        "reference": "human-authored task-domain materials database schema",
        "classifications": {
            "useful_supplement": "a broadly useful materials-database leaf missing from the manual schema",
            "reasonable_specialization": "a defensible task-specific refinement that adds nonredundant information",
            "redundant": "duplicates information already represented by the manual schema or another generated leaf",
            "workflow_artifact": "instruction, review, agent, routing, or implementation metadata rather than scientific data",
            "misbound": "scientific information placed under the wrong entity, hierarchy, or cardinality",
            "unclear_requires_review": "insufficient information for a reliable classification",
        },
        "rules": [
            "judge every extra path independently",
            "do not award utility merely because a path is more generic",
            "distinguish scientific provenance/evidence from workflow metadata",
            "return exactly one item per requested path",
        ],
    }
    return (
        "You are an independent schema-quality adjudicator. You did not generate any candidate schema. "
        "Return JSON only with an items array; each item must contain system_path, classification, and rationale.\n\n"
        f"ADJUDICATION_CONTRACT:\n{json.dumps(contract, ensure_ascii=False, indent=2)}\n\n"
        f"MANUAL_REFERENCE_LEAVES:\n{json.dumps(manual_paths, ensure_ascii=False)}\n\n"
        f"MANUAL_LEAVES_NOT_COVERED_BY_THIS_SYSTEM:\n{json.dumps(missing_manual_paths, ensure_ascii=False)}\n\n"
        f"SYSTEM_ONLY_LEAVES_TO_CLASSIFY:\n{json.dumps(extra_paths, ensure_ascii=False)}"
    )


def validate_items(payload: Any, expected_paths: list[str]) -> list[dict[str, str]]:
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("adjudication response must contain an items list")
    by_path: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("every adjudication item must be an object")
        path = str(item.get("system_path") or "")
        classification = str(item.get("classification") or "")
        rationale = str(item.get("rationale") or "").strip()
        if path not in expected_paths:
            raise ValueError(f"unexpected system_path: {path}")
        if path in by_path:
            raise ValueError(f"duplicate system_path: {path}")
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise ValueError(f"invalid classification for {path}: {classification}")
        if not rationale:
            raise ValueError(f"missing rationale for {path}")
        by_path[path] = {
            "system_path": path,
            "classification": classification,
            "rationale": rationale,
        }
    missing = [path for path in expected_paths if path not in by_path]
    if missing:
        raise ValueError(f"missing adjudications: {missing}")
    return [by_path[path] for path in expected_paths]


def split_retry_batch(batch: list[str], retry_batch_size: int) -> list[list[str]]:
    if retry_batch_size <= 0 or retry_batch_size >= len(batch):
        return [batch]
    return [
        batch[index : index + retry_batch_size]
        for index in range(0, len(batch), retry_batch_size)
    ]


def _run_reserved(
    args: argparse.Namespace,
    run_identity: dict[str, Any],
) -> dict[str, Any]:
    api_key = os.getenv("CODE_AGENT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CODE_AGENT_API_KEY is not configured")
    report_path = Path(args.comparison_report).resolve()
    manual_path = Path(args.manual_schema).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manual_paths = evaluator.flatten_schema(
        json.loads(manual_path.read_text(encoding="utf-8"))
    )
    stage = next(
        (item for item in report.get("stages") or [] if item.get("stage_id") == args.stage_id),
        None,
    )
    if stage is None:
        raise ValueError(f"stage not found: {args.stage_id}")
    field_report = (stage.get("schema") or {}).get("manual_json_238") or {}
    extras = [
        str(item.get("system_path"))
        for item in field_report.get("system_extras") or []
        if item.get("system_path")
    ]
    missing = [str(path) for path in field_report.get("missing_gold_paths") or []]
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    batches = [
        extras[index : index + args.batch_size]
        for index in range(0, len(extras), args.batch_size)
    ]
    def process(index: int, batch: list[str]) -> dict[str, Any]:
        digest = hashlib.sha256("\n".join(batch).encode("utf-8")).hexdigest()[:16]
        checkpoint = run_dir / f"batch-{index:03d}-{digest}.json"
        parts = split_retry_batch(batch, args.retry_batch_size)
        combined_items: list[dict[str, str]] = []
        highest_attempt = 0
        for part_index, part in enumerate(parts, start=1):
            part_digest = hashlib.sha256("\n".join(part).encode("utf-8")).hexdigest()[:16]
            part_checkpoint = run_dir / (
                f"batch-{index:03d}-{digest}-part-{part_index:03d}-{part_digest}.json"
            )
            part_items: list[dict[str, str]] = []
            last_error = ""
            for attempt in range(1, args.max_attempts + 1):
                highest_attempt = max(highest_attempt, attempt)
                try:
                    text = prompt_agent.litellm_chat(
                        base_url=args.base_url,
                        api_key=api_key,
                        model=args.model,
                        prompt=build_prompt(manual_paths, missing, part),
                        temperature=0,
                        max_tokens=args.max_tokens,
                    )
                    part_items = validate_items(prompt_agent.parse_llm_json(text), part)
                    if len(parts) > 1:
                        run_artifact_guard.atomic_write_json(
                            part_checkpoint,
                            {"items": part_items},
                            run_identity=run_identity,
                        )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
            if not part_items:
                return {
                    "index": index,
                    "items": [],
                    "paths": batch,
                    "error": f"sub-batch {part_index}/{len(parts)} failed: {last_error}",
                }
            combined_items.extend(part_items)
        items = validate_items({"items": combined_items}, batch)
        run_artifact_guard.atomic_write_json(
            checkpoint,
            {"items": items},
            run_identity=run_identity,
        )
        return {
            "index": index,
            "items": items,
            "checkpoint": str(checkpoint),
            "attempt": highest_attempt,
            "subbatch_count": len(parts),
        }

    completed_batches: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, index, batch): index
            for index, batch in enumerate(batches, start=1)
        }
        for future in as_completed(futures):
            completed_batches.append(future.result())
    completed_batches.sort(key=lambda item: item["index"])
    failures = [
        {
            "batch": item["index"],
            "paths": item.get("paths") or [],
            "error": item.get("error") or "unknown failure",
        }
        for item in completed_batches
        if not item.get("items")
    ]
    completed = [
        entry for item in completed_batches for entry in item.get("items") or []
    ]
    by_path = {item["system_path"]: item for item in completed}
    ordered = [by_path[path] for path in extras if path in by_path]
    result = {
        "schema_version": "materials-db.independent-extra-field-adjudication.v1",
        "run_identity": run_identity,
        "stage_id": args.stage_id,
        "comparison_report": str(report_path),
        "comparison_report_sha256": sha256(report_path),
        "manual_schema": str(manual_path),
        "manual_schema_sha256": sha256(manual_path),
        "manual_leaf_count": len(manual_paths),
        "extra_count": len(extras),
        "adjudicated_count": len(ordered),
        "failure_count": len(failures),
        "failures": failures,
        "batch_manifests": [
            {
                key: item[key]
                for key in (
                    "index",
                    "checkpoint",
                    "attempt",
                    "subbatch_count",
                )
                if key in item
            }
            for item in completed_batches
        ],
        "items": ordered,
    }
    run_artifact_guard.atomic_write_json(
        Path(args.output).resolve(),
        result,
        run_identity=run_identity,
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output).resolve()
    run_dir = Path(args.run_dir).resolve()
    input_identity = run_artifact_guard.build_input_identity(
        PIPELINE,
        {
            "stage_id": args.stage_id,
            "model": args.model,
            "base_url": args.base_url,
            "batch_size": args.batch_size,
            "retry_batch_size": args.retry_batch_size,
            "max_attempts": args.max_attempts,
            "max_tokens": args.max_tokens,
            "workers": args.workers,
        },
        [args.comparison_report, args.manual_schema],
    )
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline=PIPELINE,
        output_path=output_path,
        input_identity=input_identity,
        artifact_paths=[output_path, run_dir],
    )
    try:
        result = _run_reserved(args, run_identity)
    except BaseException as exc:
        run_artifact_guard.update_run_status(
            run_identity,
            "interrupted",
            error_type=type(exc).__name__,
            error_message=str(exc)[:1000],
        )
        raise
    complete = result["failure_count"] == 0 and result["extra_count"] == result["adjudicated_count"]
    run_artifact_guard.update_run_status(
        run_identity,
        "completed" if complete else "needs_review",
        output_status="success" if complete else "failed",
        failure_count=result["failure_count"],
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-report", required=True)
    parser.add_argument("--manual-schema", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=384000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--retry-batch-size",
        type=int,
        default=0,
        help="Split each fresh original batch into smaller sub-batches of this size.",
    )
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("extra_count", "adjudicated_count", "failure_count")
            }
        )
    )
    return 0 if result["failure_count"] == 0 and result["extra_count"] == result["adjudicated_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
