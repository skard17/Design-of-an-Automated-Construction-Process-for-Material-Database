"""Core contracts and replay utilities for CARE-IE counterfactual evaluation.

The module is intentionally independent from the live Step8/Step9 graph.  It
provides an auditable boundary around one-component oracle interventions before
API-backed adapters are connected.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from itertools import combinations
from typing import Any


CASE_SCHEMA = "care-ie.counterfactual_case.v1"
INTERVENTION_SCHEMA = "care-ie.oracle_intervention.v1"
REPLAY_SCHEMA = "care-ie.replay_result.v1"
ATTRIBUTION_SCHEMA = "care-ie.attribution_result.v1"
MINIMAL_SET_SCHEMA = "care-ie.minimal_sufficient_repairs.v1"

COMPONENTS = ("schema", "evidence", "extraction", "binding", "normalization")
EVIDENCE_SUBTYPES = (
    "ocr_error",
    "figure_alignment",
    "retrieval_miss",
    "context_truncation",
)
SPLITS = ("dev_pilot", "in_domain_frozen", "temporal_frozen")

COMPONENT_ROOTS = {
    component: f"/trace/{component}" for component in COMPONENTS
}

STRICT_RECORD_FIELDS = (
    "paper_id",
    "concept_id",
    "record_key",
    "value",
    "unit",
    "qualifiers",
    "evidence_line",
    "evidence",
    "binding",
)


class CounterfactualContractError(ValueError):
    """Raised when a case or intervention violates the replay contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(mapping: Mapping[str, Any], key: str, expected_type: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected_type) or (expected_type is str and not value.strip()):
        raise CounterfactualContractError(
            f"{key} must be a non-empty {expected_type.__name__}"
        )
    return value


def validate_case(case: Mapping[str, Any]) -> None:
    if case.get("schema_version") != CASE_SCHEMA:
        raise CounterfactualContractError(f"schema_version must be {CASE_SCHEMA}")
    _require(case, "case_id", str)
    _require(case, "domain", str)
    _require(case, "paper_id", str)
    if case.get("split") not in SPLITS:
        raise CounterfactualContractError(f"split must be one of {SPLITS}")
    trace = _require(case, "baseline_trace", dict)
    for component in COMPONENTS:
        if component not in trace:
            raise CounterfactualContractError(
                f"baseline_trace is missing component boundary: {component}"
            )
    _require(trace, "final_record", dict)


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise CounterfactualContractError(f"JSON pointer must start with '/': {pointer}")
    return [_decode_pointer_token(part) for part in pointer.split("/")[1:]]


def _set_pointer(document: Any, pointer: str, value: Any) -> None:
    parts = _pointer_parts(pointer)
    if not parts:
        raise CounterfactualContractError("root replacement is not allowed")
    current = document
    for part in parts[:-1]:
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise CounterfactualContractError(
                    f"invalid list pointer segment {part!r} in {pointer}"
                ) from exc
        elif isinstance(current, dict):
            current = current.setdefault(part, {})
        else:
            raise CounterfactualContractError(
                f"pointer traverses a scalar at {part!r} in {pointer}"
            )
    leaf = parts[-1]
    if isinstance(current, list):
        try:
            index = int(leaf)
        except ValueError as exc:
            raise CounterfactualContractError(
                f"invalid list pointer leaf {leaf!r} in {pointer}"
            ) from exc
        if index < 0 or index >= len(current):
            raise CounterfactualContractError(f"list index out of range in {pointer}")
        current[index] = copy.deepcopy(value)
    elif isinstance(current, dict):
        current[leaf] = copy.deepcopy(value)
    else:
        raise CounterfactualContractError(f"pointer parent is scalar in {pointer}")


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def changed_paths(before: Any, after: Any, pointer: str = "") -> list[str]:
    if type(before) is not type(after):
        return [pointer or "/"]
    if isinstance(before, dict):
        paths: list[str] = []
        for key in sorted(set(before) | set(after)):
            child = f"{pointer}/{_escape_pointer_token(str(key))}"
            if key not in before or key not in after:
                paths.append(child)
            else:
                paths.extend(changed_paths(before[key], after[key], child))
        return paths
    if isinstance(before, list):
        paths = []
        for index in range(max(len(before), len(after))):
            child = f"{pointer}/{index}"
            if index >= len(before) or index >= len(after):
                paths.append(child)
            else:
                paths.extend(changed_paths(before[index], after[index], child))
        return paths
    return [] if before == after else [pointer or "/"]


def _path_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def validate_intervention(intervention: Mapping[str, Any]) -> None:
    if intervention.get("schema_version") != INTERVENTION_SCHEMA:
        raise CounterfactualContractError(
            f"schema_version must be {INTERVENTION_SCHEMA}"
        )
    _require(intervention, "intervention_id", str)
    _require(intervention, "case_id", str)
    component = intervention.get("component")
    if component not in COMPONENTS:
        raise CounterfactualContractError(f"component must be one of {COMPONENTS}")
    if component == "evidence":
        subtype = intervention.get("evidence_subtype")
        if subtype is not None and subtype not in EVIDENCE_SUBTYPES:
            raise CounterfactualContractError(
                f"evidence_subtype must be one of {EVIDENCE_SUBTYPES}"
            )
    replacements = _require(intervention, "replacements", dict)
    allowed_paths = _require(intervention, "allowed_paths", list)
    if not replacements:
        raise CounterfactualContractError("replacements must not be empty")
    if not allowed_paths or not all(isinstance(path, str) for path in allowed_paths):
        raise CounterfactualContractError("allowed_paths must contain JSON pointers")
    component_root = COMPONENT_ROOTS[component]
    for path in [*replacements, *allowed_paths]:
        _pointer_parts(path)
        if not _path_within(path, component_root):
            raise CounterfactualContractError(
                f"{path} crosses the {component!r} boundary {component_root}"
            )
    for path in replacements:
        if not any(_path_within(path, allowed) for allowed in allowed_paths):
            raise CounterfactualContractError(
                f"replacement path {path} is not declared in allowed_paths"
            )


def apply_intervention(
    case: Mapping[str, Any], intervention: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    validate_case(case)
    validate_intervention(intervention)
    if intervention["case_id"] != case["case_id"]:
        raise CounterfactualContractError("case_id mismatch between case and intervention")

    document = {"trace": copy.deepcopy(case["baseline_trace"])}
    before = copy.deepcopy(document)
    for pointer, value in intervention["replacements"].items():
        _set_pointer(document, pointer, value)

    diff = changed_paths(before, document)
    allowed_paths = intervention["allowed_paths"]
    outside = [
        path
        for path in diff
        if not any(_path_within(path, allowed) for allowed in allowed_paths)
    ]
    if outside:
        raise CounterfactualContractError(
            f"intervention changed paths outside its whitelist: {outside}"
        )
    return document["trace"], diff


ReplayAdapter = Callable[[str, dict[str, Any]], dict[str, Any]]


def replay_intervention(
    case: Mapping[str, Any],
    intervention: Mapping[str, Any],
    adapter: ReplayAdapter,
    *,
    repeat_id: int = 1,
) -> dict[str, Any]:
    patched_trace, diff = apply_intervention(case, intervention)
    final_record = adapter(intervention["component"], copy.deepcopy(patched_trace))
    if not isinstance(final_record, dict):
        raise CounterfactualContractError("replay adapter must return a record object")
    return {
        "schema_version": REPLAY_SCHEMA,
        "case_id": case["case_id"],
        "intervention_id": intervention["intervention_id"],
        "component": intervention["component"],
        "evidence_subtype": intervention.get("evidence_subtype"),
        "repeat_id": repeat_id,
        "baseline_trace_sha256": content_sha256(case["baseline_trace"]),
        "patched_trace_sha256": content_sha256(patched_trace),
        "changed_paths": diff,
        "final_record": final_record,
        "final_record_sha256": content_sha256(final_record),
    }


def strict_record_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in STRICT_RECORD_FIELDS if key in record}


def strict_record_mismatches(
    gold_record: Mapping[str, Any], prediction: Mapping[str, Any]
) -> list[str]:
    expected = strict_record_projection(gold_record)
    actual = strict_record_projection(prediction)
    return [
        key for key, value in expected.items() if key not in actual or actual[key] != value
    ]


def strict_record_match(
    gold_record: Mapping[str, Any], prediction: Mapping[str, Any]
) -> bool:
    return not strict_record_mismatches(gold_record, prediction)


def evaluate_replay(
    gold_record: Mapping[str, Any], replay_result: Mapping[str, Any]
) -> dict[str, Any]:
    final_record = _require(replay_result, "final_record", dict)
    mismatches = strict_record_mismatches(gold_record, final_record)
    return {
        **dict(replay_result),
        "outcome": 0 if mismatches else 1,
        "strict_match": not mismatches,
        "mismatch_fields": mismatches,
    }


def responsibility_effects(
    evaluated_results: Iterable[Mapping[str, Any]], *, baseline_outcome: float
) -> dict[str, Any]:
    by_component: dict[str, list[float]] = defaultdict(list)
    for result in evaluated_results:
        component = str(result.get("component") or "")
        if component not in COMPONENTS:
            raise CounterfactualContractError(f"invalid result component: {component}")
        outcome = result.get("outcome")
        if outcome not in (0, 1, 0.0, 1.0):
            raise CounterfactualContractError("evaluated result outcome must be binary")
        by_component[component].append(float(outcome))

    effects = []
    for component in COMPONENTS:
        outcomes = by_component.get(component, [])
        if not outcomes:
            continue
        probability = sum(outcomes) / len(outcomes)
        effects.append(
            {
                "component": component,
                "repeat_count": len(outcomes),
                "success_probability": probability,
                "responsibility_effect": probability - float(baseline_outcome),
            }
        )
    effects.sort(key=lambda item: (-item["responsibility_effect"], item["component"]))
    return {
        "schema_version": ATTRIBUTION_SCHEMA,
        "baseline_outcome": float(baseline_outcome),
        "effects": effects,
        "ranked_components": [item["component"] for item in effects],
    }


def pairwise_interaction_effect(
    baseline: float, first: float, second: float, joint: float
) -> float:
    return float(joint) - float(first) - float(second) + float(baseline)


SubsetEvaluator = Callable[[tuple[str, ...]], bool]


def hierarchical_minimal_repair_search(
    evaluate_subset: SubsetEvaluator,
    *,
    components: Iterable[str] = COMPONENTS,
    max_order: int | None = None,
    repeats: int = 1,
) -> dict[str, Any]:
    """Find the smallest repair sets that make a failed case pass.

    Every subset is evaluated repeatedly. The search stops at the first repair
    cardinality with at least one successful subset, so every returned set is
    minimal by cardinality. Multiple successful sets are reported as ambiguous
    instead of being broken by an arbitrary component ordering.
    """

    ordered_components = tuple(dict.fromkeys(str(item) for item in components))
    invalid = [item for item in ordered_components if item not in COMPONENTS]
    if invalid:
        raise CounterfactualContractError(f"invalid search components: {invalid}")
    if not ordered_components:
        raise CounterfactualContractError("components must not be empty")
    if repeats < 1:
        raise CounterfactualContractError("repeats must be at least 1")

    depth = len(ordered_components) if max_order is None else int(max_order)
    if depth < 0:
        raise CounterfactualContractError("max_order must be non-negative")
    depth = min(depth, len(ordered_components))
    probes: list[dict[str, Any]] = []

    def probe(subset: tuple[str, ...]) -> tuple[bool, bool]:
        outcomes = [bool(evaluate_subset(subset)) for _ in range(repeats)]
        deterministic = len(set(outcomes)) == 1
        probes.append(
            {
                "repair_set": list(subset),
                "cardinality": len(subset),
                "outcomes": outcomes,
                "deterministic": deterministic,
                "success": deterministic and outcomes[0],
            }
        )
        return deterministic and outcomes[0], deterministic

    baseline_pass, baseline_deterministic = probe(())
    if not baseline_deterministic:
        status = "nondeterministic"
        minimal_sets: list[list[str]] = []
    elif baseline_pass:
        status = "no_fault"
        minimal_sets = [[]]
    else:
        status = "out_of_scope"
        minimal_sets = []
        for order in range(1, depth + 1):
            successful: list[list[str]] = []
            for subset in combinations(ordered_components, order):
                passed, deterministic = probe(subset)
                if not deterministic:
                    status = "nondeterministic"
                    break
                if passed:
                    successful.append(list(subset))
            if status == "nondeterministic":
                break
            if successful:
                minimal_sets = successful
                status = "identified" if len(successful) == 1 else "ambiguous"
                break

    return {
        "schema_version": MINIMAL_SET_SCHEMA,
        "status": status,
        "components": list(ordered_components),
        "max_order": depth,
        "repeats": repeats,
        "minimal_cardinality": len(minimal_sets[0]) if minimal_sets else None,
        "minimal_sufficient_sets": minimal_sets,
        "probe_count": len(probes),
        "probes": probes,
    }


def forbidden_keys(payload: Any, banned_keys: Iterable[str]) -> list[str]:
    banned = {key.casefold() for key in banned_keys}
    hits: list[str] = []

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_pointer = f"{pointer}/{_escape_pointer_token(str(key))}"
                if str(key).casefold() in banned:
                    hits.append(child_pointer)
                visit(child, child_pointer)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{pointer}/{index}")

    visit(payload, "")
    return hits
