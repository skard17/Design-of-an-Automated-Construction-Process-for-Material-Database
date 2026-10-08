"""Executable five-stage replay fixtures for CARE-IE responsibility tests."""

from __future__ import annotations

import copy
from typing import Any

from care_counterfactual import (
    CASE_SCHEMA,
    COMPONENTS,
    INTERVENTION_SCHEMA,
    evaluate_replay,
    replay_intervention,
    responsibility_effects,
    strict_record_match,
)


CONTROLLED_CASE_SCHEMA = "care-ie.executable_controlled_case.v1"


CONCEPT_FIXTURES = (
    ("transition_temperature", 9.3, "K", {"criterion": "onset"}),
    ("upper_critical_field", 6.95, "T", {"temperature": "0 K"}),
    ("lower_critical_field", 18.4, "mT", {"temperature": "0 K"}),
    ("coherence_length", 6.89, "nm", {"temperature": "0 K"}),
    ("penetration_depth", 241.0, "nm", {"temperature": "0 K"}),
    ("superconducting_gap", 0.71, "k_B T_c", {"model": "single-gap fit"}),
    ("critical_current_density", 3.2e5, "A/cm2", {"temperature": "4.2 K"}),
    ("critical_current", 12.5, "mA", {"temperature": "4.2 K"}),
    ("pairing_symmetry", "mixed s+p-wave", None, {"status": "favored"}),
    ("crystal_structure", "trigonal P-3m1", None, {"pressure": "ambient"}),
)


def _candidate(
    *,
    paper_id: str,
    record_key: str,
    concept_id: str,
    value: Any,
    unit: Any,
    qualifiers: dict[str, Any],
    evidence_line: int,
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "record_key": record_key,
        "concept_id": concept_id,
        "value": value,
        "unit": unit,
        "qualifiers": copy.deepcopy(qualifiers),
        "evidence_line": evidence_line,
        "evidence": (
            f"{record_key}: {concept_id} = {value}"
            + (f" {unit}" if unit else "")
            + f" under {qualifiers}."
        ),
    }


def execute_trace_with_outputs(trace: dict[str, Any]) -> dict[str, Any]:
    """Run all five stages and return observable outputs at each boundary."""
    fixture = trace["fixture"]
    schema = trace["schema"]
    evidence = trace["evidence"]
    extraction = trace["extraction"]
    binding = trace["binding"]
    normalization = trace["normalization"]

    selected = fixture["candidates"][int(evidence["selected_index"])]
    concept_id = (
        schema["concept_id_override"]
        if schema.get("concept_id_override") is not None
        else selected["concept_id"]
    )
    extracted_value = (
        extraction["value_override"]
        if extraction.get("value_override") is not None
        else selected["value"]
    )
    extracted_unit = selected["unit"]
    qualifiers = copy.deepcopy(selected["qualifiers"])
    for key in extraction.get("drop_qualifiers", []):
        qualifiers.pop(key, None)

    record_key = (
        binding["record_key_override"]
        if binding.get("record_key_override") is not None
        else selected["record_key"]
    )
    normalized_value = extracted_value
    normalized_unit = extracted_unit
    if normalization.get("value_mode") == "stringify":
        normalized_value = str(normalized_value)
    if normalization.get("unit_override_enabled"):
        normalized_unit = normalization.get("unit_override")

    extracted = {
        "concept_id": concept_id,
        "value": extracted_value,
        "unit": extracted_unit,
        "qualifiers": qualifiers,
        "evidence_line": selected["evidence_line"],
    }
    bound = {**copy.deepcopy(extracted), "record_key": record_key}
    final_record = {
        "paper_id": fixture["paper_id"],
        "concept_id": concept_id,
        "record_key": record_key,
        "value": normalized_value,
        "unit": normalized_unit,
        "qualifiers": qualifiers,
        "evidence_line": selected["evidence_line"],
        "evidence": selected["evidence"],
        "binding": {"record_key": record_key},
    }
    return {
        "schema": {"concept_id": concept_id},
        "evidence": copy.deepcopy(selected),
        "extraction": extracted,
        "binding": bound,
        "normalization": copy.deepcopy(final_record),
        "final_record": final_record,
    }


def execute_trace_adapter(_component: str, trace: dict[str, Any]) -> dict[str, Any]:
    """Run all five fixed stage implementations from immutable fixture inputs."""

    return execute_trace_with_outputs(trace)["final_record"]


def oracle_trace(index: int) -> dict[str, Any]:
    concept_id, value, unit, qualifiers = CONCEPT_FIXTURES[index % len(CONCEPT_FIXTURES)]
    paper_id = f"controlled-paper-{index + 1:02d}"
    record_key = f"Sample-{index + 1:02d}"
    distractor_value = value + 1 if isinstance(value, (int, float)) else f"not-{value}"
    fixture = {
        "paper_id": paper_id,
        "candidates": [
            _candidate(
                paper_id=paper_id,
                record_key=record_key,
                concept_id=concept_id,
                value=value,
                unit=unit,
                qualifiers=qualifiers,
                evidence_line=10 + index * 2,
            ),
            _candidate(
                paper_id=paper_id,
                record_key=f"Distractor-{index + 1:02d}",
                concept_id=concept_id,
                value=distractor_value,
                unit=unit,
                qualifiers={**qualifiers, "context": "distractor"},
                evidence_line=11 + index * 2,
            ),
        ],
    }
    return oracle_trace_from_fixture(fixture)


def oracle_trace_from_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Create a clean executable trace from an externally supplied fixture."""

    fixture = copy.deepcopy(fixture)
    candidates = fixture.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("fixture must contain at least two source candidates")
    required = {
        "paper_id",
        "record_key",
        "concept_id",
        "value",
        "unit",
        "qualifiers",
        "evidence_line",
        "evidence",
    }
    for candidate in candidates:
        missing = sorted(required - set(candidate))
        if missing:
            raise ValueError(f"fixture candidate is missing required keys: {missing}")

    trace = {
        "fixture": fixture,
        "schema": {"concept_id_override": None},
        "evidence": {"selected_index": 0},
        "extraction": {"value_override": None, "drop_qualifiers": []},
        "binding": {"record_key_override": None},
        "normalization": {
            "value_mode": "identity",
            "unit_override_enabled": False,
            "unit_override": None,
        },
        "final_record": {},
    }
    trace["final_record"] = execute_trace_adapter("normalization", copy.deepcopy(trace))
    return trace


def inject_single_fault(trace: dict[str, Any], component: str, index: int) -> dict[str, Any]:
    faulty = copy.deepcopy(trace)
    distractor = faulty["fixture"]["candidates"][1]
    if component == "schema":
        faulty["schema"]["concept_id_override"] = f"wrong_concept_{index + 1:02d}"
    elif component == "evidence":
        faulty["evidence"]["selected_index"] = 1
    elif component == "extraction":
        faulty["extraction"]["value_override"] = distractor["value"]
    elif component == "binding":
        faulty["binding"]["record_key_override"] = distractor["record_key"]
    elif component == "normalization":
        value = faulty["fixture"]["candidates"][0]["value"]
        if isinstance(value, (int, float)):
            faulty["normalization"]["value_mode"] = "stringify"
        else:
            faulty["normalization"]["unit_override_enabled"] = True
            faulty["normalization"]["unit_override"] = "noncanonical"
    else:
        raise ValueError(f"unknown component: {component}")
    faulty["final_record"] = execute_trace_adapter(component, copy.deepcopy(faulty))
    return faulty


def build_controlled_case(case_number: int, component: str, fixture_index: int) -> dict[str, Any]:
    oracle = oracle_trace(fixture_index)
    case_id = f"controlled-{case_number:03d}"
    return build_controlled_case_from_oracle(
        case_id=case_id,
        component=component,
        oracle=oracle,
        fault_index=fixture_index,
        domain="superconductivity",
        split="dev_pilot",
    )


def build_controlled_case_from_oracle(
    *,
    case_id: str,
    component: str,
    oracle: dict[str, Any],
    fault_index: int,
    domain: str,
    split: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a controlled bundle without encoding its label in the case id."""

    if component not in COMPONENTS:
        raise ValueError(f"unknown component: {component}")
    baseline = inject_single_fault(oracle, component, fault_index)
    case = {
        "schema_version": CASE_SCHEMA,
        "controlled_schema_version": CONTROLLED_CASE_SCHEMA,
        "case_id": case_id,
        "domain": domain,
        "split": split,
        "paper_id": oracle["fixture"]["paper_id"],
        "baseline_trace": baseline,
    }
    if provenance:
        case["provenance"] = copy.deepcopy(provenance)
    interventions = {
        name: {
            "schema_version": INTERVENTION_SCHEMA,
            "intervention_id": f"{case_id}-{name}",
            "case_id": case_id,
            "component": name,
            "replacements": {f"/trace/{name}": copy.deepcopy(oracle[name])},
            "allowed_paths": [f"/trace/{name}"],
        }
        for name in COMPONENTS
    }
    return {
        "case": case,
        "gold_record": oracle["final_record"],
        "responsible_component": component,
        "interventions": interventions,
    }


def answer_free_observable(bundle: dict[str, Any]) -> dict[str, Any]:
    """Expose execution evidence without oracle configs or responsibility labels."""

    case = bundle["case"]
    trace = case["baseline_trace"]
    fixture = trace["fixture"]
    target = bundle["gold_record"]
    baseline = trace["final_record"]
    mismatch_fields = [key for key in target if baseline.get(key) != target.get(key)]
    return {
        "schema_version": "care-ie.controlled_observable.v1",
        "case_id": case["case_id"],
        "paper_id": case["paper_id"],
        "target_contract": {
            "record_key": target["record_key"],
            "concept_id": target["concept_id"],
            "required_fields": [
                "value",
                "unit",
                "qualifiers",
                "evidence_line",
                "evidence",
                "binding",
            ],
        },
        "source_candidates": copy.deepcopy(fixture["candidates"]),
        "stage_outputs": execute_trace_with_outputs(trace),
        "baseline_record": copy.deepcopy(baseline),
        "answer_free_failure_signal": {"mismatch_fields": mismatch_fields},
    }


def run_controlled_case(bundle: dict[str, Any], *, repeats: int = 2) -> dict[str, Any]:
    case = bundle["case"]
    gold = bundle["gold_record"]
    baseline_record = case["baseline_trace"]["final_record"]
    if strict_record_match(gold, baseline_record):
        raise AssertionError(f"controlled baseline unexpectedly matches gold: {case['case_id']}")

    evaluated = []
    for component in COMPONENTS:
        for repeat_id in range(1, repeats + 1):
            replay = replay_intervention(
                case,
                bundle["interventions"][component],
                execute_trace_adapter,
                repeat_id=repeat_id,
            )
            evaluated.append(evaluate_replay(gold, replay))
    attribution = responsibility_effects(evaluated, baseline_outcome=0)
    successful = sorted(
        {
            item["component"]
            for item in evaluated
            if item["strict_match"]
        }
    )
    deterministic = all(
        len(
            {
                item["final_record_sha256"]
                for item in evaluated
                if item["component"] == component
            }
        )
        == 1
        for component in COMPONENTS
    )
    return {
        "case_id": case["case_id"],
        "responsible_component": bundle["responsible_component"],
        "baseline_mismatch_fields": [
            key
            for key in gold
            if baseline_record.get(key) != gold.get(key)
        ],
        "successful_components": successful,
        "exact_unique_responsibility": successful == [bundle["responsible_component"]],
        "deterministic_repeats": deterministic,
        "attribution": attribution,
        "evaluated_replays": evaluated,
    }
