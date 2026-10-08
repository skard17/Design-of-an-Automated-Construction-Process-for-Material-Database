#!/usr/bin/env python3
"""Apply a deterministic, answer-free normalization gate to extracted facts.

The gate is a structured-protocol supervisor module. It never receives gold
answers and never changes the extracted scientific value or evidence line. It
only maps common scientific qualifier variants into documented database forms.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import evaluate_superconductivity_three_stage as evaluator
import materials_agent_protocol as protocol
import run_artifact_guard
import run_protocol_targeted_extraction as validator
from target_extraction_semantics import enrich_targets


NORMALIZATION_RULES = {
    "temperature_zero": "zero-temperature, T->0, and T=0 are stored as '0 K'",
    "ambient_pressure": "ambient pressure is stored as 'ambient'",
    "transition_onset": "Tc_onset spelling variants are stored as 'Tc onset'",
    "normal_state_fraction": "fractions of normal-state resistance/resistivity are stored as '<percent>% Rn'",
    "model_suffix": "a redundant trailing word 'model' is removed",
    "uncertainty_magnitude": "a scalar uncertainty drops the plus/minus sign and is stored numerically",
    "interpretive_status": "suggested, indicated, interpreted, and author-inference variants use 'author interpretation'",
    "epitaxial_film": "epitaxial film is stored as 'epitaxial thin film'",
}

PIPELINE = "protocol_normalization_gate_fresh"


def _number(value: float) -> int | float:
    rounded = round(value)
    return int(rounded) if math.isclose(value, rounded, abs_tol=1e-12) else value


def canonicalize_qualifier(key: str, value: Any) -> tuple[Any, str | None]:
    if value is None or isinstance(value, bool):
        return value, None
    name = str(key).casefold()
    phrase = evaluator._scientific_phrase(value)

    if name == "temperature" and phrase == "0 k":
        return "0 K", "temperature_zero"
    if name == "pressure" and phrase == "ambient":
        return "ambient", "ambient_pressure"
    if name == "criterion":
        fraction = evaluator._criterion_fraction(value)
        if fraction is not None:
            percent = _number(fraction * 100)
            return f"{percent}% Rn", "normal_state_fraction"
        if phrase == "tc onset":
            return "Tc onset", "transition_onset"
    if name == "model":
        raw = str(value).strip()
        trimmed = re.sub(r"\s+model\s*$", "", raw, flags=re.IGNORECASE).strip()
        if trimmed != raw:
            return trimmed, "model_suffix"
    if name == "uncertainty":
        match = re.fullmatch(r"\s*[±+\-]?\s*(\d+(?:\.\d+)?)\s*", str(value))
        if match:
            return _number(float(match.group(1))), "uncertainty_magnitude"
    if name == "result_status" and phrase in {
        "author inference",
        "author interpretation",
        "indicated",
        "interpreted",
        "suggested",
    }:
        return "author interpretation", "interpretive_status"
    if name == "sample_form" and phrase == "epitaxial film":
        return "epitaxial thin film", "epitaxial_film"
    return value, None


def normalize_fact(fact: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = deepcopy(fact)
    changes = []
    qualifiers = normalized.get("qualifiers") or {}
    for key, before in list(qualifiers.items()):
        after, rule = canonicalize_qualifier(key, before)
        if rule and after != before:
            qualifiers[key] = after
            changes.append(
                {
                    "target_id": normalized.get("target_id"),
                    "qualifier": key,
                    "before": before,
                    "after": after,
                    "rule": rule,
                }
            )
    normalized["qualifiers"] = qualifiers
    return normalized, changes


def _run_reserved(
    args: argparse.Namespace,
    run_identity: dict[str, Any],
) -> dict[str, Any]:
    target_payload = json.loads(Path(args.targets).resolve().read_text(encoding="utf-8"))
    metric = (
        json.loads(Path(args.concept_contract).resolve().read_text(encoding="utf-8"))
        if args.concept_contract
        else None
    )
    targets = enrich_targets(target_payload["targets"], metric)
    target_by_id = {str(item["target_id"]): item for item in targets}
    seed_path = Path(args.seed_predictions).resolve()
    seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
    seed_facts = seed_payload.get("facts") or seed_payload.get("predictions") or []
    seed_by_id = {str(item.get("target_id") or ""): item for item in seed_facts}
    missing = sorted(set(target_by_id) - set(seed_by_id))
    if missing:
        raise ValueError(f"seed predictions are missing target ids: {missing}")

    request_message = protocol.make_message(
        sender="step9_extraction_supervisor",
        receiver="protocol_normalization_gate",
        phase="scientific_qualifier_normalization",
        status="approved_to_execute",
        payload_refs={
            "target_manifest": str(Path(args.targets).resolve()),
            "seed_prediction_digest": protocol.stable_digest(seed_facts),
        },
        decision={
            "gold_answers_available": False,
            "scientific_values_mutable": False,
            "evidence_lines_mutable": False,
            "normalization_rules": NORMALIZATION_RULES,
        },
        requested_actions=[
            "normalize only documented qualifier variants",
            "preserve values, units, target identities, and evidence lines",
            "record every changed qualifier and rule",
        ],
        next_route="protocol_normalization_supervisor_validation",
    )
    request_errors = protocol.validate_message(request_message)
    if request_errors:
        raise ValueError(f"invalid normalization request message: {request_errors}")

    normalized_facts = []
    changes = []
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        target_id = str(target["target_id"])
        normalized, fact_changes = normalize_fact(seed_by_id[target_id])
        normalized_facts.append(normalized)
        changes.extend(fact_changes)
        by_paper.setdefault(str(target["paper_id"]), []).append(target)

    pack_dir = Path(args.pack_dir).resolve()
    normalized_by_id = {str(item["target_id"]): item for item in normalized_facts}
    for paper_id, paper_targets in by_paper.items():
        evidence = (pack_dir / f"{paper_id}.md").read_text(
            encoding="utf-8", errors="replace"
        )
        paper_facts = [normalized_by_id[str(item["target_id"])] for item in paper_targets]
        validator.validate_response(
            {"facts": paper_facts}, paper_targets, validator.source_lines(evidence)
        )

    acceptance = protocol.make_message(
        sender="protocol_normalization_supervisor",
        receiver="evaluation_supervisor",
        phase="scientific_qualifier_normalization",
        status="accepted",
        payload_refs={
            "input_message_id": request_message["message_id"],
            "prediction_digest": protocol.stable_digest(normalized_facts),
        },
        decision={
            "target_count": len(targets),
            "prediction_count": len(normalized_facts),
            "change_count": len(changes),
            "target_contract_valid": True,
            "source_line_contract_valid": True,
            "gold_answers_available": False,
        },
        produced_artifacts=[str(Path(args.output).resolve())],
        next_route="scientific_value_evaluation",
    )
    messages = [request_message, acceptance]
    result = {
        "source": "answer_free_structured_protocol_normalization_gate",
        "run_identity": run_identity,
        "target_manifest": str(Path(args.targets).resolve()),
        "seed_predictions": str(seed_path),
        "target_count": len(targets),
        "prediction_count": len(normalized_facts),
        "failure_count": 0,
        "normalization_rules": NORMALIZATION_RULES,
        "normalization_change_count": len(changes),
        "normalization_changes": changes,
        "facts": normalized_facts,
        "protocol_messages": messages,
        "protocol_validation": protocol.validate_message_list(messages),
        "gold_answers_available": False,
        "external_api_used": False,
    }
    run_artifact_guard.atomic_write_json(
        Path(args.output).resolve(),
        result,
        run_identity=run_identity,
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output).resolve()
    input_paths = [args.targets, args.seed_predictions, args.pack_dir]
    if args.concept_contract:
        input_paths.append(args.concept_contract)
    input_identity = run_artifact_guard.build_input_identity(
        PIPELINE,
        {
            "external_api_used": False,
            "scientific_values_mutable": False,
            "evidence_lines_mutable": False,
            "normalization_rules": NORMALIZATION_RULES,
        },
        input_paths,
    )
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline=PIPELINE,
        output_path=output_path,
        input_identity=input_identity,
        artifact_paths=[output_path],
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
    run_artifact_guard.update_run_status(
        run_identity,
        "completed",
        output_status="success",
        failure_count=0,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--concept-contract")
    parser.add_argument("--seed-predictions", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "target_count",
                    "prediction_count",
                    "failure_count",
                    "normalization_change_count",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
