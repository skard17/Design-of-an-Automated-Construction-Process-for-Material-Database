#!/usr/bin/env python3
"""Run the frozen legacy targeted prompt in a fresh isolated namespace."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

import prompt_quality_code_agent as prompt_agent
import run_artifact_guard


PIPELINE = "legacy_targeted_extraction_fresh"


def batch_checkpoint_path(
    run_dir: Path,
    paper_id: str,
    batch_targets: list[dict[str, Any]],
) -> Path:
    target_ids = [str(item["target_id"]) for item in batch_targets]
    digest = hashlib.sha256("\n".join(target_ids).encode("utf-8")).hexdigest()[:16]
    paper_digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:12]
    batch_dir = run_dir / "_b" / paper_digest
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir / f"batch-{digest}.json"


def load_legacy_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("frozen_legacy_targeted_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import legacy runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_reserved(
    args: argparse.Namespace,
    run_identity: dict[str, Any],
) -> dict[str, Any]:
    api_key = os.getenv("CODE_AGENT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CODE_AGENT_API_KEY is not configured")
    legacy = load_legacy_runner(Path(args.legacy_runner).resolve())
    target_payload = json.loads(Path(args.targets).resolve().read_text(encoding="utf-8"))
    targets = target_payload["targets"]
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        by_paper.setdefault(str(target["paper_id"]), []).append(target)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = Path(args.pack_dir).resolve()

    def process(paper_id: str, paper_targets: list[dict[str, Any]]) -> dict[str, Any]:
        output_path = run_dir / f"{paper_id}.json"
        evidence = (pack_dir / f"{paper_id}.md").read_text(encoding="utf-8", errors="replace")
        batch_size = args.max_targets_per_call or len(paper_targets)
        batches = [paper_targets[index : index + batch_size] for index in range(0, len(paper_targets), batch_size)]
        facts: list[dict[str, Any]] = []
        batch_attempts = []
        batch_failures = []
        last_error = ""
        raw_text = ""
        for batch_index, batch_targets in enumerate(batches, start=1):
            checkpoint_path = batch_checkpoint_path(run_dir, paper_id, batch_targets)
            prompt = legacy.build_prompt(paper_id, batch_targets, evidence)
            for attempt in range(1, args.max_attempts + 1):
                try:
                    raw_text = prompt_agent.litellm_chat(
                        base_url=args.base_url,
                        api_key=api_key,
                        model=args.model,
                        prompt=prompt,
                        temperature=0,
                        max_tokens=args.max_tokens,
                    )
                    batch_facts = legacy.validate_response(
                        prompt_agent.parse_llm_json(raw_text), batch_targets
                    )
                    facts.extend(batch_facts)
                    checkpoint = {
                        "paper_id": paper_id,
                        "target_ids": [str(item["target_id"]) for item in batch_targets],
                        "facts": batch_facts,
                    }
                    run_artifact_guard.atomic_write_json(
                        checkpoint_path,
                        checkpoint,
                        run_identity=run_identity,
                    )
                    batch_attempts.append(
                        {
                            "batch": batch_index,
                            "attempt": attempt,
                            "checkpoint": str(checkpoint_path),
                        }
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
            else:
                failure = {
                    "batch": batch_index,
                    "target_ids": [str(item["target_id"]) for item in batch_targets],
                    "error": last_error,
                }
                batch_failures.append(failure)
                run_artifact_guard.atomic_write_json(
                    checkpoint_path.with_suffix(".failure.json"),
                    failure,
                    run_identity=run_identity,
                )
        if facts and not batch_failures:
            facts = legacy.validate_response({"facts": facts}, paper_targets)
            stored = {
                "paper_id": paper_id,
                "batch_attempts": batch_attempts,
                "batch_count": len(batches),
                "facts": facts,
            }
            run_artifact_guard.atomic_write_json(
                output_path,
                stored,
                run_identity=run_identity,
            )
            return {
                "paper_id": paper_id,
                "facts": facts,
                "path": str(output_path),
                "batch_attempts": batch_attempts,
            }
        failure = {
            "paper_id": paper_id,
            "target_count": len(paper_targets),
            "error": last_error,
            "raw_text": raw_text,
            "partial_prediction_count": len(facts),
            "partial_facts": facts,
            "batch_attempts": batch_attempts,
            "batch_failures": batch_failures,
        }
        run_artifact_guard.atomic_write_json(
            output_path,
            failure,
            run_identity=run_identity,
        )
        return {
            "paper_id": paper_id,
            "failure": failure,
            "path": str(output_path),
            "batch_attempts": batch_attempts,
        }

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, paper_id, paper_targets): paper_id
            for paper_id, paper_targets in sorted(by_paper.items())
        }
        for future in as_completed(futures):
            completed.append(future.result())

    completed.sort(key=lambda item: item["paper_id"])
    facts = [fact for item in completed for fact in item.get("facts") or []]
    failures = [item["failure"] for item in completed if item.get("failure")]
    result = {
        "source": "frozen_legacy_answer_free_targeted_extraction_parallel_scheduler",
        "run_identity": run_identity,
        "legacy_runner": str(Path(args.legacy_runner).resolve()),
        "target_manifest": str(Path(args.targets).resolve()),
        "target_count": len(targets),
        "prediction_count": len(facts),
        "failure_count": len(failures),
        "failures": failures,
        "facts": facts,
        "paper_manifests": [
            {
                key: item[key]
                for key in ("paper_id", "path", "batch_attempts")
                if key in item
            }
            for item in completed
        ],
        "workers": args.workers,
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
            "model": args.model,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "max_attempts": args.max_attempts,
            "workers": args.workers,
            "max_targets_per_call": args.max_targets_per_call,
        },
        [args.legacy_runner, args.targets, args.pack_dir],
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
    complete = not result["failures"] and result["prediction_count"] == result["target_count"]
    run_artifact_guard.update_run_status(
        run_identity,
        "completed" if complete else "needs_review",
        output_status="success" if complete else "failed",
        failure_count=result["failure_count"],
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-runner", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=384000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-targets-per-call", type=int, default=0)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps({key: result[key] for key in ("target_count", "prediction_count", "failure_count")}))
    return 0 if not result["failures"] and result["prediction_count"] == result["target_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
