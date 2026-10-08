#!/usr/bin/env python3
"""Independently adjudicate one-to-one semantic schema correspondence."""

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
    "semantic_equivalent",
    "partial_broader_system",
    "partial_narrower_system",
    "missing",
}
PIPELINE = "schema_correspondence_adjudication_fresh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build_prompt(manual_paths: list[str], system_paths: list[str]) -> str:
    contract = {
        "task": "evaluate an automatically designed materials-literature database schema",
        "reference_domain": "task-domain materials database represented by the provided manual leaves",
        "classifications": {
            "semantic_equivalent": "the system leaf explicitly stores the same atomic information as this manual leaf",
            "partial_broader_system": "the system leaf is a generic container or EAV slot that could hold the manual information but does not explicitly name and preserve it",
            "partial_narrower_system": "the system leaf stores only a narrower special case of the manual information",
            "missing": "no system leaf represents the manual information",
        },
        "anti_inflation_rules": [
            "use semantic_equivalent only for an explicit atomic field, never merely because a generic property_type/value, observation_result, condition, evidence, or free-text field could hold the value",
            "one system leaf may be semantic_equivalent to at most one manual leaf",
            "a generic conditions field cannot replace explicit temperature, pressure, field, direction, angle, current, frequency, or criterion leaves",
            "match value, unit, qualifier, evidence, and raw-data leaves separately; a value leaf does not cover its unit or conditions",
            "different names such as Tc and transition_temperature are equivalent when their atomic storage meaning is the same",
            "array-versus-scalar cardinality may still be semantically equivalent but must not be hidden; the deterministic evaluator records that structural conflict separately",
            "return exactly one item for every requested manual_path",
        ],
        "output": {
            "items": [
                {
                    "manual_path": "exact requested path",
                    "classification": "one allowed classification",
                    "system_path": "exact candidate path or null for missing",
                    "rationale": "short concrete reason",
                }
            ]
        },
    }
    return (
        "You are an independent schema-correspondence adjudicator. You did not generate the candidate "
        "schema and must resist coverage inflation. Return JSON only.\n\n"
        f"CONTRACT:\n{json.dumps(contract, ensure_ascii=False, indent=2)}\n\n"
        f"SYSTEM_EXPLICIT_LEAVES:\n{json.dumps(system_paths, ensure_ascii=False)}\n\n"
        f"MANUAL_LEAVES_TO_CLASSIFY:\n{json.dumps(manual_paths, ensure_ascii=False)}"
    )


def validate_items(
    payload: Any,
    expected_manual_paths: list[str],
    system_paths: list[str],
) -> list[dict[str, str]]:
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("correspondence response must contain an items list")
    expected = set(expected_manual_paths)
    system = set(system_paths)
    by_manual: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("every correspondence item must be an object")
        manual_path = str(item.get("manual_path") or "")
        classification = str(item.get("classification") or "")
        system_path = str(item.get("system_path") or "")
        rationale = str(item.get("rationale") or "").strip()
        if manual_path not in expected:
            raise ValueError(f"unexpected manual_path: {manual_path}")
        if manual_path in by_manual:
            raise ValueError(f"duplicate manual_path: {manual_path}")
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise ValueError(f"invalid classification for {manual_path}: {classification}")
        if classification == "missing":
            system_path = ""
        elif system_path not in system:
            raise ValueError(f"invalid system_path for {manual_path}: {system_path}")
        if not rationale:
            raise ValueError(f"missing rationale for {manual_path}")
        by_manual[manual_path] = {
            "manual_path": manual_path,
            "classification": classification,
            "system_path": system_path,
            "rationale": rationale,
        }
    missing = [path for path in expected_manual_paths if path not in by_manual]
    if missing:
        raise ValueError(f"missing manual adjudications: {missing}")
    return [by_manual[path] for path in expected_manual_paths]


def demote_duplicate_equivalents(items: list[dict[str, str]]) -> list[dict[str, str]]:
    claims: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        if item["classification"] == "semantic_equivalent":
            claims.setdefault(item["system_path"], []).append(index)
    result = [dict(item) for item in items]
    for system_path, indexes in claims.items():
        if len(indexes) <= 1:
            continue
        for index in indexes:
            result[index]["classification"] = "partial_broader_system"
            result[index]["rationale"] = (
                f"{result[index]['rationale']} Demoted because generated leaf {system_path!r} "
                "was claimed as equivalent by multiple manual leaves."
            )
    return result


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
    manual_paths = evaluator.flatten_schema(json.loads(manual_path.read_text(encoding="utf-8")))
    stage = next(
        (item for item in report.get("stages") or [] if item.get("stage_id") == args.stage_id),
        None,
    )
    if stage is None:
        raise ValueError(f"stage not found: {args.stage_id}")
    system_inventory = (stage.get("schema") or {}).get("system_inventory") or {}
    system_paths = evaluator.comparison_field_paths(system_inventory)
    if not system_paths:
        raise ValueError(f"stage has no explicit system leaves: {args.stage_id}")

    batches = [
        manual_paths[index : index + args.batch_size]
        for index in range(0, len(manual_paths), args.batch_size)
    ]
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

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
            last_error = ""
            part_items: list[dict[str, str]] = []
            for attempt in range(1, args.max_attempts + 1):
                highest_attempt = max(highest_attempt, attempt)
                try:
                    text = prompt_agent.litellm_chat(
                        base_url=args.base_url,
                        api_key=api_key,
                        model=args.model,
                        prompt=build_prompt(part, system_paths),
                        temperature=0,
                        max_tokens=args.max_tokens,
                    )
                    part_items = validate_items(
                        prompt_agent.parse_llm_json(text), part, system_paths
                    )
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
                    "error": f"sub-batch {part_index}/{len(parts)} failed: {last_error}",
                }
            combined_items.extend(part_items)
        items = validate_items({"items": combined_items}, batch, system_paths)
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

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, index, batch): index
            for index, batch in enumerate(batches, start=1)
        }
        for future in as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda item: item["index"])
    failures = [
        {"batch": item["index"], "error": item.get("error") or "unknown failure"}
        for item in completed
        if not item.get("items")
    ]
    raw_items = [entry for item in completed for entry in item.get("items") or []]
    items = demote_duplicate_equivalents(raw_items)
    by_manual = {item["manual_path"]: item for item in items}
    ordered = [by_manual[path] for path in manual_paths if path in by_manual]
    result = {
        "schema_version": "materials-db.independent-schema-correspondence.v1",
        "run_identity": run_identity,
        "stage_id": args.stage_id,
        "comparison_report": str(report_path),
        "comparison_report_sha256": sha256(report_path),
        "manual_schema": str(manual_path),
        "manual_schema_sha256": sha256(manual_path),
        "manual_leaf_count": len(manual_paths),
        "system_leaf_count": len(system_paths),
        "adjudicated_manual_leaf_count": len(ordered),
        "semantic_equivalent_count": sum(
            item["classification"] == "semantic_equivalent" for item in ordered
        ),
        "partial_count": sum(item["classification"].startswith("partial_") for item in ordered),
        "missing_count": sum(item["classification"] == "missing" for item in ordered),
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
            for item in completed
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
    complete = (
        result["failure_count"] == 0
        and result["manual_leaf_count"] == result["adjudicated_manual_leaf_count"]
    )
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
    parser.add_argument("--batch-size", type=int, default=24)
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
                for key in (
                    "manual_leaf_count",
                    "adjudicated_manual_leaf_count",
                    "semantic_equivalent_count",
                    "partial_count",
                    "missing_count",
                    "failure_count",
                )
            }
        )
    )
    return 0 if not result["failure_count"] and result["manual_leaf_count"] == result["adjudicated_manual_leaf_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
