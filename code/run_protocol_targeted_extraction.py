#!/usr/bin/env python3
"""Run answer-free targeted extraction under the materials-agent protocol.

This benchmark runner is domain-neutral. A target manifest supplies concept and
record identities but never expected values. The supervisor validates response
shape, evidence-line provenance, and protocol messages before accepting a paper.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import materials_agent_protocol as protocol
import prompt_quality_code_agent as prompt_agent
import run_artifact_guard
from target_extraction_semantics import enrich_targets


SOURCE_LINE_PATTERN = re.compile(r"<!--\s*SOURCE_LINE:\s*(\d+)\s*-->")
PIPELINE = "protocol_targeted_extraction_fresh"


def extraction_protocol_message(
    paper_id: str,
    targets: list[dict[str, Any]],
    evidence_path: Path,
) -> dict[str, Any]:
    return protocol.make_message(
        sender="step9_extraction_supervisor",
        receiver="targeted_materials_extractor",
        phase="targeted_scientific_value_extraction",
        status="approved_to_execute",
        payload_refs={
            "paper_id": paper_id,
            "target_manifest_digest": protocol.stable_digest(targets),
            "evidence_pack": str(evidence_path.resolve()),
        },
        decision={
            "target_count": len(targets),
            "answer_values_present_in_manifest": False,
            "outside_knowledge_allowed": False,
            "one_output_per_target_required": True,
            "evidence_line_required": True,
            "unresolved_policy": "return an explicit unresolved object only when direct evidence is absent",
        },
        requested_actions=[
            "extract one source-supported value per target",
            "preserve measurement conditions and requested qualifiers",
            "bind every value to a SOURCE_LINE marker",
        ],
        next_route="step9_extraction_supervisor_validation",
    )


def build_prompt(
    paper_id: str,
    targets: list[dict[str, Any]],
    evidence: str,
    message: dict[str, Any],
) -> str:
    return f"""
You are the targeted extraction worker in a supervisor-managed materials-database workflow.
The source is parsed scientific-literature Markdown. Extract one answer for every target using
only the supplied evidence pack. The target manifest contains no answer values.

Structured task and handoff protocol:
{json.dumps(message, ensure_ascii=False, indent=2)}

Execution rules:
- Return JSON only with top-level key "facts".
- Return exactly one object per target_id and no extra objects.
- Copy target_id exactly.
- value must be the smallest source-supported scalar or concise scientific string.
- Put a measurement unit only in unit, never duplicate it in value.
- evidence_line must be the SOURCE_LINE marker containing the direct evidence.
- qualifiers must contain exactly the requested qualifier_keys and must be evidence-supported.
- Preserve criterion, conditions, orientation, sample identity, uncertainty, and scientific origin
  when the target requests them.
- Do not infer from general knowledge, references, or neighboring materials.
- These benchmark targets are evidence-backed; do not emit null or invent an unresolved state.

Paper: {paper_id}
Targets:
{json.dumps(targets, ensure_ascii=False, indent=2)}

Required response shape:
{{
  "facts": [
    {{
      "target_id": "exact target id",
      "value": "source-supported value",
      "unit": null,
      "qualifiers": {{"requested_key": "source-supported value"}},
      "evidence_line": 1
    }}
  ]
}}

Evidence pack:
{evidence}
""".strip()


def build_supervisor_prompt(
    paper_id: str,
    targets: list[dict[str, Any]],
    evidence: str,
    message: dict[str, Any],
    worker_facts: list[dict[str, Any]],
    seed_facts: list[dict[str, Any]],
) -> str:
    return f"""
You are the extraction supervisor in a structured materials-database workflow. Adjudicate the
worker candidate and, when present, the prior CARE-reviewed candidate using only the supplied
scientific-literature evidence. The candidates are not gold answers.

Structured supervisor protocol:
{json.dumps(message, ensure_ascii=False, indent=2)}

Rules:
- Return JSON only with top-level key "facts".
- Return exactly one fact per target_id and preserve the exact qualifier keys.
- Select or repair the candidate whose value, unit, record binding, qualifiers, and SOURCE_LINE
  are jointly supported by direct evidence.
- A provenance/status qualifier describes measured, reported, fitted, calculated, estimated,
  derived, predicted, or author-interpreted origin; it is not a confidence label.
- For identities, return the exact named entity, not a paper-title category or material family.
- Prefer a direct source line containing the value and its defining conditions. Do not infer.
- Preserve a prior candidate when the worker candidate is less specific or less directly supported.

Paper: {paper_id}
Target semantic contracts:
{json.dumps(targets, ensure_ascii=False, indent=2)}

Prior CARE-reviewed candidates:
{json.dumps(seed_facts, ensure_ascii=False, indent=2)}

Current protocol-worker candidates:
{json.dumps(worker_facts, ensure_ascii=False, indent=2)}

Evidence pack:
{evidence}
""".strip()


def source_lines(evidence: str) -> set[int]:
    return {int(match.group(1)) for match in SOURCE_LINE_PATTERN.finditer(evidence)}


def validate_response(
    payload: dict[str, Any],
    targets: list[dict[str, Any]],
    valid_source_lines: set[int],
) -> list[dict[str, Any]]:
    facts = payload.get("facts")
    if not isinstance(facts, list):
        raise ValueError("response facts must be a list")
    target_by_id = {str(item["target_id"]): item for item in targets}
    seen: set[str] = set()
    canonical = []
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("every response fact must be an object")
        target_id = str(fact.get("target_id") or "")
        if target_id not in target_by_id:
            raise ValueError(f"unexpected target_id: {target_id}")
        if target_id in seen:
            raise ValueError(f"duplicate target_id: {target_id}")
        seen.add(target_id)
        target = target_by_id[target_id]
        qualifiers = fact.get("qualifiers")
        if not isinstance(qualifiers, dict):
            raise ValueError(f"qualifiers must be an object for {target_id}")
        expected_keys = set(target.get("qualifier_keys") or [])
        if set(qualifiers) != expected_keys:
            raise ValueError(
                f"qualifier keys mismatch for {target_id}: expected "
                f"{sorted(expected_keys)}, got {sorted(qualifiers)}"
            )
        evidence_line = fact.get("evidence_line")
        if not isinstance(evidence_line, int) or evidence_line not in valid_source_lines:
            raise ValueError(f"invalid or absent SOURCE_LINE for {target_id}: {evidence_line}")
        if fact.get("value") is None:
            raise ValueError(f"null value for evidence-backed target {target_id}")
        canonical.append(
            {
                "target_id": target_id,
                "paper_id": target["paper_id"],
                "record_key": target["record_key"],
                "concept_id": target["concept_id"],
                "value": fact["value"],
                "unit": fact.get("unit"),
                "qualifiers": qualifiers,
                "evidence_line": evidence_line,
            }
        )
    missing = sorted(set(target_by_id) - seen)
    if missing:
        raise ValueError(f"missing target_ids: {missing}")
    return canonical


def _run_reserved(
    args: argparse.Namespace,
    run_identity: dict[str, Any],
) -> dict[str, Any]:
    api_key = os.getenv("CODE_AGENT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CODE_AGENT_API_KEY is not configured")
    target_path = Path(args.targets).resolve()
    target_payload = json.loads(target_path.read_text(encoding="utf-8"))
    metric_contract = (
        json.loads(Path(args.concept_contract).resolve().read_text(encoding="utf-8"))
        if args.concept_contract
        else None
    )
    targets = enrich_targets(target_payload["targets"], metric_contract)
    seed_by_id: dict[str, dict[str, Any]] = {}
    if args.seed_predictions:
        seed_payload = json.loads(Path(args.seed_predictions).resolve().read_text(encoding="utf-8"))
        seed_facts = seed_payload.get("facts") or seed_payload.get("predictions") or []
        seed_by_id = {str(item.get("target_id") or ""): item for item in seed_facts}
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        by_paper.setdefault(str(target["paper_id"]), []).append(target)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    def process(paper_id: str, paper_targets: list[dict[str, Any]]) -> dict[str, Any]:
        output_path = run_dir / f"{paper_id}.json"
        pack_path = Path(args.pack_dir).resolve() / f"{paper_id}.md"
        evidence = pack_path.read_text(encoding="utf-8", errors="replace")
        lines = source_lines(evidence)
        batch_size = args.max_targets_per_call or len(paper_targets)
        batches = [
            paper_targets[index : index + batch_size]
            for index in range(0, len(paper_targets), batch_size)
        ]
        canonical: list[dict[str, Any]] = []
        worker_facts: list[dict[str, Any]] = []
        seed_facts: list[dict[str, Any]] = []
        protocol_messages: list[dict[str, Any]] = []
        batch_attempts: list[dict[str, int]] = []
        supervisor_reviewed_target_count = 0
        last_error = ""
        for batch_index, batch_targets in enumerate(batches, start=1):
            message = extraction_protocol_message(paper_id, batch_targets, pack_path)
            message_validation = protocol.validate_message(message)
            if message_validation:
                raise ValueError(f"invalid protocol message: {message_validation}")
            prompt = build_prompt(paper_id, batch_targets, evidence, message)
            batch_seed = [
                seed_by_id[str(item["target_id"])]
                for item in batch_targets
                if str(item["target_id"]) in seed_by_id
            ]
            supervisor_message = None
            for attempt in range(1, args.max_attempts + 1):
                supervisor_message = None
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
                    batch_worker = validate_response(parsed, batch_targets, lines)
                    batch_final = batch_worker
                    if args.supervisor_review:
                        supervisor_message = protocol.make_message(
                            sender="step9_extraction_supervisor",
                            receiver="step9_extraction_adjudicator",
                            phase="scientific_evidence_adjudication",
                            status="approved_to_review",
                            payload_refs={
                                "paper_id": paper_id,
                                "worker_candidate_digest": protocol.stable_digest(batch_worker),
                                "seed_candidate_digest": protocol.stable_digest(batch_seed),
                                "evidence_pack": str(pack_path.resolve()),
                            },
                            decision={
                                "gold_answers_available": False,
                                "choose_or_repair_from_direct_evidence": True,
                                "preserve_better_prior_candidate": bool(batch_seed),
                            },
                            requested_actions=[
                                "adjudicate value, unit, binding, qualifiers, and evidence jointly",
                                "return one final fact per target",
                            ],
                            next_route="step9_aggregation",
                        )
                        review_prompt = build_supervisor_prompt(
                            paper_id,
                            batch_targets,
                            evidence,
                            supervisor_message,
                            batch_worker,
                            batch_seed,
                        )
                        review_text = prompt_agent.litellm_chat(
                            base_url=args.base_url,
                            api_key=api_key,
                            model=args.model,
                            prompt=review_prompt,
                            temperature=0,
                            max_tokens=args.max_tokens,
                        )
                        batch_final = validate_response(
                            prompt_agent.parse_llm_json(review_text), batch_targets, lines
                        )
                    canonical.extend(batch_final)
                    worker_facts.extend(batch_worker)
                    seed_facts.extend(batch_seed)
                    protocol_messages.extend(
                        item for item in (message, supervisor_message) if item is not None
                    )
                    batch_attempts.append({"batch": batch_index, "attempt": attempt})
                    if args.supervisor_review:
                        supervisor_reviewed_target_count += len(batch_final)
                    break
                except Exception as exc:  # noqa: BLE001 - preserve per-batch diagnostics
                    last_error = str(exc)
            else:
                canonical = []
                break

        if canonical:
            canonical = validate_response({"facts": canonical}, paper_targets, lines)
            acceptance = protocol.make_message(
                sender="step9_extraction_supervisor",
                receiver="step9_aggregation",
                phase="targeted_scientific_value_extraction",
                status="accepted",
                payload_refs={
                    "paper_id": paper_id,
                    "prediction_digest": protocol.stable_digest(canonical),
                },
                decision={
                    "target_count": len(paper_targets),
                    "prediction_count": len(canonical),
                    "batch_count": len(batches),
                    "json_valid": True,
                    "target_contract_valid": True,
                    "source_line_contract_valid": True,
                    "supervisor_reviewed_target_count": supervisor_reviewed_target_count,
                    "seed_fallback_target_count": 0,
                },
                produced_artifacts=[str(output_path)],
                next_route="step9_aggregation",
            )
            protocol_messages.append(acceptance)
            stored = {
                "paper_id": paper_id,
                "batch_count": len(batches),
                "batch_attempts": batch_attempts,
                "facts": canonical,
                "worker_facts": worker_facts,
                "seed_facts": seed_facts,
                "supervisor_reviewed_target_count": supervisor_reviewed_target_count,
                "seed_fallback_target_count": 0,
                "protocol_messages": protocol_messages,
                "protocol_validation": protocol.validate_message_list(protocol_messages),
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
                "attempts": batch_attempts,
                "protocol_validation": stored["protocol_validation"],
                "supervisor_reviewed_target_count": supervisor_reviewed_target_count,
                "seed_fallback_target_count": 0,
            }

        failure = {
            "paper_id": paper_id,
            "target_count": len(paper_targets),
            "error": last_error,
            "supervisor_decision": {
                "action": "hold_failed_paper",
                "reason": "provider or supervisor validation failed after bounded attempts",
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
    paper_manifests = [
        {
            key: item[key]
            for key in ("paper_id", "path", "attempts", "protocol_validation")
            if key in item
        }
        for item in completed
    ]
    for manifest, item in zip(paper_manifests, completed):
        for key in (
            "supervisor_reviewed_target_count",
            "seed_fallback_target_count",
        ):
            if key in item:
                manifest[key] = item[key]
    supervisor_reviewed_target_count = sum(
        int(item.get("supervisor_reviewed_target_count") or 0) for item in completed
    )
    seed_fallback_target_count = sum(
        int(item.get("seed_fallback_target_count") or 0) for item in completed
    )

    final_message = protocol.make_message(
        sender="step9_aggregation",
        receiver="evaluation_supervisor",
        phase="targeted_scientific_value_extraction",
        status="complete" if not failures and len(all_facts) == len(targets) else "needs_review",
        payload_refs={
            "target_manifest": str(target_path),
            "target_manifest_digest": protocol.stable_digest(targets),
        },
        decision={
            "target_count": len(targets),
            "prediction_count": len(all_facts),
            "failure_count": len(failures),
            "supervisor_reviewed_target_count": supervisor_reviewed_target_count,
            "seed_fallback_target_count": seed_fallback_target_count,
        },
        produced_artifacts=[str(Path(args.output).resolve())],
        next_route="scientific_value_evaluation",
    )
    result = {
        "source": "answer_free_protocol_targeted_materials_extraction",
        "run_identity": run_identity,
        "target_manifest": str(target_path),
        "concept_contract": str(Path(args.concept_contract).resolve()) if args.concept_contract else None,
        "seed_predictions": str(Path(args.seed_predictions).resolve()) if args.seed_predictions else None,
        "supervisor_review": bool(args.supervisor_review),
        "target_count": len(targets),
        "prediction_count": len(all_facts),
        "failure_count": len(failures),
        "supervisor_reviewed_target_count": supervisor_reviewed_target_count,
        "seed_fallback_paper_count": 0,
        "seed_fallback_target_count": 0,
        "seed_fallback_disabled": True,
        "failures": failures,
        "facts": all_facts,
        "paper_manifests": paper_manifests,
        "protocol_messages": [final_message],
        "protocol_validation": protocol.validate_message_list([final_message]),
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
    input_paths = [args.targets, args.pack_dir]
    if args.concept_contract:
        input_paths.append(args.concept_contract)
    if args.seed_predictions:
        input_paths.append(args.seed_predictions)
    input_identity = run_artifact_guard.build_input_identity(
        PIPELINE,
        {
            "model": args.model,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "max_attempts": args.max_attempts,
            "workers": args.workers,
            "max_targets_per_call": args.max_targets_per_call,
            "supervisor_review": bool(args.supervisor_review),
            "seed_is_candidate_only": True,
            "seed_fallback": False,
        },
        input_paths,
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
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concept-contract")
    parser.add_argument("--seed-predictions")
    parser.add_argument("--supervisor-review", action="store_true")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=384000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-targets-per-call", type=int, default=10)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("target_count", "prediction_count", "failure_count")
            }
        )
    )
    return 0 if not result["failures"] and result["prediction_count"] == result["target_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
