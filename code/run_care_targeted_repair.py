#!/usr/bin/env python3
"""Run an answer-free CARE component review over targeted extraction results."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import prompt_quality_code_agent as prompt_agent
import run_artifact_guard
import run_protocol_targeted_extraction as validator
from target_extraction_semantics import enrich_targets


COMPONENT_REVIEW = {
    "schema": "Does the candidate answer the precise concept contract rather than a neighboring property?",
    "binding": "Is the value bound to the exact record_key, sample, phase, orientation, and condition?",
    "evidence": "Does the selected SOURCE_LINE directly support the value and every qualifier?",
    "normalization": "Is the value the smallest supported scalar/string with the unit separated and no semantic rewriting?",
    "provenance": "Do result-status, method, model, criterion, and other qualifiers reflect the source exactly?",
}

PIPELINE = "care_targeted_repair_fresh"


def build_review_prompt(
    paper_id: str,
    targets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    evidence: str,
) -> str:
    return f"""
You are the CARE extraction-repair supervisor in a materials-database workflow whose only data
source is parsed scientific-literature Markdown. The candidate facts were produced without answer
values. Review them using the five component counterfactual checks below. Repair a fact only when
the supplied evidence supports the repair. No expected answers or evaluator feedback are provided.

Component counterfactual checks:
{json.dumps(COMPONENT_REVIEW, ensure_ascii=False, indent=2)}

Rules:
- Return JSON only with top-level keys "facts" and "diagnoses".
- Return exactly one fact for every target_id and no extras.
- Copy target_id exactly; preserve the target's exact qualifier keys.
- Each final value and qualifier must be directly supported by its selected SOURCE_LINE.
- Prefer a direct line that contains the value, unit, record binding, and requested qualifiers.
- Do not copy broad paper-title phrases into identity fields when an exact entity is requested.
- Do not replace provenance/status with confidence language.
- If the candidate is already correct and directly supported, preserve it.
- diagnoses may name responsible components but must not contain expected or gold answers.

Paper: {paper_id}
Target semantic contracts:
{json.dumps(targets, ensure_ascii=False, indent=2)}

Candidate facts:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Required fact shape:
{{"target_id":"id","value":"source-supported value","unit":null,
  "qualifiers":{{"requested_key":"source-supported value"}},"evidence_line":1}}

Evidence pack:
{evidence}
""".strip()


def _run_reserved(
    args: argparse.Namespace,
    run_identity: dict[str, Any],
) -> dict[str, Any]:
    api_key = os.getenv("CODE_AGENT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CODE_AGENT_API_KEY is not configured")
    target_payload = json.loads(Path(args.targets).resolve().read_text(encoding="utf-8"))
    metric = json.loads(Path(args.concept_contract).resolve().read_text(encoding="utf-8"))
    targets = enrich_targets(target_payload["targets"], metric)
    seed_payload = json.loads(Path(args.seed_predictions).resolve().read_text(encoding="utf-8"))
    seed_facts = seed_payload.get("facts") or seed_payload.get("predictions") or []
    seed_by_id = {str(item.get("target_id") or ""): item for item in seed_facts}
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        by_paper.setdefault(str(target["paper_id"]), []).append(target)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    def process(paper_id: str, paper_targets: list[dict[str, Any]]) -> dict[str, Any]:
        pack_path = Path(args.pack_dir).resolve() / f"{paper_id}.md"
        evidence = pack_path.read_text(encoding="utf-8", errors="replace")
        lines = validator.source_lines(evidence)
        output_path = run_dir / f"{paper_id}.json"
        batch_size = args.max_targets_per_call or len(paper_targets)
        batches = [paper_targets[index : index + batch_size] for index in range(0, len(paper_targets), batch_size)]
        canonical: list[dict[str, Any]] = []
        diagnoses = []
        batch_attempts = []
        last_error = ""
        for batch_index, batch_targets in enumerate(batches, start=1):
            candidates = [seed_by_id[str(item["target_id"])] for item in batch_targets]
            prompt = build_review_prompt(paper_id, batch_targets, candidates, evidence)
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
                    parsed = prompt_agent.parse_llm_json(raw_text)
                    canonical.extend(validator.validate_response(parsed, batch_targets, lines))
                    diagnoses.extend(parsed.get("diagnoses") or [])
                    batch_attempts.append({"batch": batch_index, "attempt": attempt})
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
            else:
                canonical = []
                break
        if canonical:
            canonical = validator.validate_response({"facts": canonical}, paper_targets, lines)
            stored = {
                "paper_id": paper_id,
                "batch_count": len(batches),
                "batch_attempts": batch_attempts,
                "mode": "care_guided_component_review_without_gold_answers",
                "components_checked": list(COMPONENT_REVIEW),
                "facts": canonical,
                "diagnoses": diagnoses,
            }
            run_artifact_guard.atomic_write_json(
                output_path,
                stored,
                run_identity=run_identity,
            )
            return {
                "paper_id": paper_id,
                "path": str(output_path),
                "facts": canonical,
                "batch_attempts": batch_attempts,
            }
        failure = {
            "paper_id": paper_id,
            "target_count": len(paper_targets),
            "error": last_error,
            "supervisor_decision": {
                "action": "hold_failed_paper",
                "reason": "CARE provider review failed after bounded attempts",
                "retry_count": args.max_attempts,
            },
        }
        run_artifact_guard.atomic_write_json(
            output_path,
            failure,
            run_identity=run_identity,
        )
        return {"paper_id": paper_id, "path": str(output_path), "failure": failure}

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, paper_id, paper_targets): paper_id
            for paper_id, paper_targets in sorted(by_paper.items())
        }
        for future in as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda item: item["paper_id"])
    all_facts = [fact for item in completed for fact in item.get("facts") or []]
    failures = [item["failure"] for item in completed if item.get("failure")]
    manifests = [
        {key: item[key] for key in ("paper_id", "path", "batch_attempts") if key in item}
        for item in completed
    ]
    complete = not failures and len(all_facts) == len(targets)

    result = {
        "source": "care_guided_answer_free_targeted_repair",
        "run_identity": run_identity,
        "seed_predictions": str(Path(args.seed_predictions).resolve()),
        "target_count": len(targets),
        "prediction_count": len(all_facts),
        "failure_count": len(failures),
        "seed_fallback_paper_count": 0,
        "seed_fallback_target_count": 0,
        "seed_fallback_disabled": True,
        "failures": failures,
        "facts": all_facts,
        "paper_manifests": manifests,
        "care_component_contract": COMPONENT_REVIEW,
        "gold_answers_available_to_repair": False,
        "supervisor_decision": {
            "action": "release_to_evaluation" if complete else "hold_for_review",
            "status": "accepted" if complete else "needs_review",
            "failure_count": len(failures),
        },
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
            "seed_is_candidate_only": True,
            "seed_fallback": False,
        },
        [args.targets, args.concept_contract, args.seed_predictions, args.pack_dir],
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
    parser.add_argument("--targets", required=True)
    parser.add_argument("--concept-contract", required=True)
    parser.add_argument("--seed-predictions", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=384000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-targets-per-call", type=int, default=10)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps({key: result[key] for key in ("target_count", "prediction_count", "failure_count")}))
    return 0 if not result["failures"] and result["prediction_count"] == result["target_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
