#!/usr/bin/env python3
"""Evaluate three superconductivity pipeline stages with one frozen contract.

The evaluator deliberately separates four different questions:

1. field-schema coverage against the user's manual JSON;
2. scientific-value extraction against evidence-backed human facts;
3. CARE component-responsibility attribution;
4. structured protocol conformance.

It never treats concept-taxonomy coverage, CARE injected-fault accuracy, or a
gold-specific canonicalizer as scientific extraction accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import run_artifact_guard


EVALUATION_PIPELINE = "superconductivity_three_version_evaluation_fresh"


SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-")

PARENT_ALIASES = {
    "material_info.section1.transition_temperature": "material_info.section1.tc",
    "material_info.section1.critical_current": "material_info.section1.jc",
    "material_info.section1.critical_current_density": "material_info.section1.jc",
    "material_info.section1.lower_critical_field": "material_info.section1.hc1",
    "material_info.section1.upper_critical_field": "material_info.section1.hc2",
    "material_info.section1.thermodynamic_critical_field": "material_info.section1.hc",
    "material_info.section1.penetration_depth": "material_info.section1.lambda",
    "material_info.section1.coherence_length": "material_info.section1.xi",
    "material_info.section2.preparation_processes": "material_info.section2",
    "material_info.section4.resistance_temperature": "material_info.section4.r_t",
    "material_info.section4.resistance_field": "material_info.section4.r_h",
    "material_info.section4.current_voltage": "material_info.section4.i_v",
    "material_info.section4.heat_capacity": "material_info.section4.specific_heat",
    "material_info.section4.susceptibility": "material_info.section4.chi_t",
    "section5.pairing_symmetry": "section5.gap_symmetry",
}

USEFUL_SUPPLEMENT_TOKENS = {
    "composition",
    "crystal_structure",
    "sample",
    "criterion",
    "uncertainty",
    "confidence",
    "evidence",
    "source_text",
    "source_type",
    "result_status",
    "measurement_context",
    "canonical_record",
    "phase_boundaries",
    "topological_claims",
    "computational_methods",
}

INSTRUCTION_ARTIFACT_TOKENS = {
    "todo",
    "example",
    "placeholder",
    "instruction",
    "section_id",
    "field_registry",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _parse_utc_timestamp(value: str) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def clean_run_artifact_audit(
    path: Path | None,
    *,
    campaign_started_at: str = "",
    expected_run_id: str = "",
    expected_pipeline: str = "",
) -> dict[str, Any]:
    """Verify that an artifact belongs to a completed clean run in this campaign."""
    if path is None or not path.exists():
        return {"valid": False, "status": "missing", "path": str(path) if path else None}
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "status": "unreadable",
            "path": str(path),
            "reason": str(exc),
        }
    identity = payload.get("run_identity") if isinstance(payload, dict) else None
    if not isinstance(identity, dict):
        return {
            "valid": False,
            "status": "missing_run_identity",
            "path": str(path),
        }
    reasons = []
    try:
        manifest = run_artifact_guard.load_manifest(identity)
    except run_artifact_guard.RunIdentityError as exc:
        return {
            "valid": False,
            "status": "invalid_run_identity",
            "path": str(path),
            "run_id": identity.get("run_id"),
            "reason": str(exc),
        }
    expected_output = str(path.resolve(strict=False)).casefold()
    if str(identity.get("output_path") or "").casefold() != expected_output:
        reasons.append("artifact_path_does_not_match_run_output")
    if manifest.get("status") != "completed":
        reasons.append(f"run_status_is_{manifest.get('status')}")
    if expected_run_id and str(identity.get("run_id") or "") != str(expected_run_id):
        reasons.append("run_id_does_not_match_stage_contract")
    if expected_pipeline and str(identity.get("pipeline") or "") != str(expected_pipeline):
        reasons.append("pipeline_does_not_match_stage_contract")
    if campaign_started_at:
        try:
            if _parse_utc_timestamp(str(manifest.get("created_at") or "")) < _parse_utc_timestamp(
                campaign_started_at
            ):
                reasons.append("run_predates_campaign")
        except (TypeError, ValueError):
            reasons.append("invalid_campaign_or_manifest_timestamp")
    return {
        "valid": not reasons,
        "status": "complete_clean_run" if not reasons else "rejected",
        "path": str(path),
        "sha256": file_sha256(path),
        "run_id": identity.get("run_id"),
        "pipeline": identity.get("pipeline"),
        "manifest_path": identity.get("manifest_path"),
        "manifest_status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "reasons": reasons,
    }


def normalize_path(path: Any) -> str:
    text = str(path or "").replace("[]", "").strip().strip(".").casefold()
    text = re.sub(r"\.+", ".", text)
    for source in sorted(PARENT_ALIASES, key=len, reverse=True):
        if text == source or text.startswith(source + "."):
            text = PARENT_ALIASES[source] + text[len(source) :]
            break
    return text


def cardinality_signature(path: Any) -> tuple[int, ...]:
    """Return component indexes declared as repeated by ``[]``."""

    parts = str(path or "").strip().strip(".").split(".")
    return tuple(index for index, part in enumerate(parts) if "[]" in part)


def flatten_schema(value: Any, path: str = "") -> list[str]:
    if isinstance(value, dict):
        leaves: list[str] = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            leaves.extend(flatten_schema(child, child_path))
        return leaves or ([path] if path else [])
    if isinstance(value, list):
        array_path = f"{path}[]"
        if value:
            return flatten_schema(value[0], array_path)
        return [array_path]
    return [path] if path else []


def split_required_subfields(contract: dict[str, Any]) -> list[str]:
    raw = contract.get("required_subfields") or []
    if isinstance(raw, str):
        return [item for item in re.split(r"[\s,;|]+", raw.strip()) if item]
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def extract_system_fields(step8_payload: dict[str, Any]) -> dict[str, Any]:
    result = step8_payload.get("result")
    schema = (result.get("schema_definition") or {}) if isinstance(result, dict) else {}
    source = "result.schema_definition"
    if not schema:
        module_outputs = step8_payload.get("module_outputs") or {}
        schema = module_outputs.get("schema_design_module") or {}
        source = "module_outputs.schema_design_module"
    if not schema:
        schema = step8_payload.get("schema_definition") or {}
        source = "schema_definition"
    registry = schema.get("field_registry") or []
    leaves: list[str] = []
    declared: list[str] = []
    for item in registry:
        if not isinstance(item, dict):
            continue
        path = str(item.get("field_path") or "").strip()
        if not path:
            continue
        declared.append(path)
        data_type = str(item.get("data_type") or "").casefold()
        slots = split_required_subfields(item.get("object_contract") or {})
        if slots:
            array_suffix = "[]" if "array" in data_type else ""
            leaves.extend(f"{path}{array_suffix}.{slot}" for slot in slots)
        else:
            leaves.append(path + ("[]" if "array" in data_type and "object" not in data_type else ""))
    return {
        "inventory_source": source,
        "declared_field_count": len(dict.fromkeys(declared)),
        "declared_fields": list(dict.fromkeys(declared)),
        "comparison_basis": "field_registry_rows",
        "comparison_leaf_count": len(dict.fromkeys(declared)),
        "comparison_leaf_fields": list(dict.fromkeys(declared)),
        "explicit_leaf_count": len(dict.fromkeys(leaves)),
        "explicit_leaf_fields": list(dict.fromkeys(leaves)),
        "storage_slot_note": (
            "explicit_leaf_fields expands each field's object contract for storage-shape "
            "audit only; field coverage compares the field_registry rows so generic "
            "value/evidence slots are not multiplied into separate generated fields"
        ),
    }


def extract_fixed_contract_fields(contract_payload: dict[str, Any]) -> dict[str, Any]:
    """Expose a legacy fixed extraction schema without pretending it was Step8 output."""

    leaves = list(dict.fromkeys(flatten_schema(contract_payload)))
    return {
        "inventory_source": "fixed_extraction_field_contract",
        "declared_field_count": len(leaves),
        "declared_fields": leaves,
        "comparison_basis": "fixed_contract_leaves",
        "comparison_leaf_count": len(leaves),
        "comparison_leaf_fields": leaves,
        "explicit_leaf_count": len(leaves),
        "explicit_leaf_fields": leaves,
    }


def comparison_field_paths(system_inventory: dict[str, Any]) -> list[str]:
    """Return generated fields without multiplying shared object-contract slots."""

    paths = system_inventory.get("comparison_leaf_fields")
    if isinstance(paths, list) and paths:
        return [str(path) for path in paths]
    return [str(path) for path in system_inventory.get("explicit_leaf_fields") or []]


def match_field_inventory(
    gold_paths: list[str],
    system_paths: list[str],
    extra_adjudication: dict[str, str] | None = None,
    semantic_correspondence: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    unique_system_paths = list(dict.fromkeys(system_paths))
    normalized_system: dict[str, list[str]] = {}
    for path in unique_system_paths:
        normalized_system.setdefault(normalize_path(path), []).append(path)

    exact_or_alias: list[dict[str, str]] = []
    partial: list[dict[str, Any]] = []
    missing: list[str] = []
    used_system: set[str] = set()
    cardinality_conflicts: list[dict[str, Any]] = []
    semantic_correspondence = semantic_correspondence or {}

    for gold in gold_paths:
        normalized_gold = normalize_path(gold)
        candidates = normalized_system.get(normalized_gold) or []
        if candidates:
            system = candidates[0]
            used_system.add(system)
            if cardinality_signature(gold) != cardinality_signature(system):
                cardinality_conflicts.append(
                    {
                        "gold_path": gold,
                        "system_path": system,
                        "gold_repeated_components": list(cardinality_signature(gold)),
                        "system_repeated_components": list(cardinality_signature(system)),
                    }
                )
            exact_or_alias.append(
                {
                    "gold_path": gold,
                    "system_path": system,
                    "match_type": "exact" if gold.replace("[]", "").casefold() == system.replace("[]", "").casefold() else "approved_alias",
                }
            )
            continue

        semantic_item = semantic_correspondence.get(gold)
        if semantic_item:
            classification = str(semantic_item.get("classification") or "")
            system = str(semantic_item.get("system_path") or "")
            if classification == "semantic_equivalent" and system in unique_system_paths:
                if system in used_system:
                    partial.append(
                        {
                            "gold_path": gold,
                            "system_paths": [system],
                            "reason": "one generated leaf cannot count as exact coverage for multiple manual leaves",
                            "classification_source": "independent_semantic_adjudication",
                        }
                    )
                    continue
                used_system.add(system)
                if cardinality_signature(gold) != cardinality_signature(system):
                    cardinality_conflicts.append(
                        {
                            "gold_path": gold,
                            "system_path": system,
                            "gold_repeated_components": list(cardinality_signature(gold)),
                            "system_repeated_components": list(cardinality_signature(system)),
                        }
                    )
                exact_or_alias.append(
                    {
                        "gold_path": gold,
                        "system_path": system,
                        "match_type": "independent_semantic_equivalent",
                        "rationale": semantic_item.get("rationale"),
                    }
                )
                continue
            if classification in {"partial_broader_system", "partial_narrower_system"}:
                partial.append(
                    {
                        "gold_path": gold,
                        "system_paths": [system] if system in unique_system_paths else [],
                        "reason": semantic_item.get("rationale") or classification,
                        "match_type": classification,
                        "classification_source": "independent_semantic_adjudication",
                    }
                )
                continue

        parent_candidates = []
        gold_parts = normalized_gold.split(".")
        for system_normalized, originals in normalized_system.items():
            system_parts = system_normalized.split(".")
            same_parent = len(gold_parts) > 1 and len(system_parts) > 1 and gold_parts[:-1] == system_parts[:-1]
            generic_condition = (
                same_parent
                and system_parts[-1] in {"conditions", "measurement_conditions"}
                and gold_parts[-1] in {
                    "temperature",
                    "pressure",
                    "magnetic_field",
                    "direction",
                    "angles",
                    "current",
                    "frequency",
                }
            )
            if generic_condition:
                parent_candidates.extend(originals)
        if parent_candidates:
            partial.append(
                {
                    "gold_path": gold,
                    "system_paths": parent_candidates,
                    "reason": "generic condition container does not explicitly preserve this manual leaf",
                }
            )
        else:
            missing.append(gold)

    extras = []
    adjudication = {
        normalize_path(path): str(classification)
        for path, classification in (extra_adjudication or {}).items()
    }
    for path in unique_system_paths:
        if path in used_system:
            continue
        reviewed_classification = adjudication.get(normalize_path(path))
        lowered_tokens = set(re.split(r"[^a-z0-9_]+", normalize_path(path)))
        if reviewed_classification:
            classification = reviewed_classification
            classification_source = "independent_adjudication"
        elif lowered_tokens & INSTRUCTION_ARTIFACT_TOKENS:
            classification = "unresolved_requires_domain_review"
            classification_source = "heuristic_risk_flag_only"
        elif lowered_tokens & USEFUL_SUPPLEMENT_TOKENS:
            classification = "unresolved_requires_domain_review"
            classification_source = "heuristic_candidate_flag_only"
        else:
            classification = "unresolved_requires_domain_review"
            classification_source = "unreviewed"
        extras.append(
            {
                "system_path": path,
                "classification": classification,
                "classification_source": classification_source,
            }
        )

    covered_count = len(exact_or_alias)
    system_count = len(unique_system_paths)
    overlap_precision = covered_count / system_count if system_count else 1.0
    recall = covered_count / len(gold_paths) if gold_paths else 1.0
    overlap_f1 = (
        2 * overlap_precision * recall / (overlap_precision + recall)
        if overlap_precision + recall
        else 0.0
    )
    union_count = len(gold_paths) + system_count - covered_count
    adjudicated_extras = [
        item for item in extras if item["classification_source"] == "independent_adjudication"
    ]
    unresolved_extras = [
        item for item in extras if item["classification_source"] != "independent_adjudication"
    ]
    useful_classes = {"useful_supplement", "reasonable_specialization"}
    useful_extra_count = sum(
        1 for item in adjudicated_extras if item["classification"] in useful_classes
    )
    utility_precision = (
        (covered_count + useful_extra_count) / system_count
        if system_count and not unresolved_extras
        else None
    )
    return {
        "gold_leaf_count": len(gold_paths),
        "system_leaf_count": system_count,
        "covered_leaf_count": covered_count,
        "strict_or_alias_coverage": recall,
        "manual_reference_recall": recall,
        "manual_reference_overlap_precision": overlap_precision,
        "manual_reference_overlap_f1": overlap_f1,
        "manual_reference_jaccard": covered_count / union_count if union_count else 1.0,
        "partial_leaf_count": len(partial),
        "missing_leaf_count": len(missing),
        "matched": exact_or_alias,
        "partial": partial,
        "missing_gold_paths": missing,
        "system_extra_count": len(extras),
        "system_extras": extras,
        "extra_classification_counts": count_by(extras, "classification"),
        "extra_classification_source_counts": count_by(extras, "classification_source"),
        "independently_adjudicated_extra_count": len(adjudicated_extras),
        "unresolved_extra_count": len(unresolved_extras),
        "useful_extra_count": useful_extra_count,
        "design_utility_precision": utility_precision,
        "cardinality_conflict_count": len(cardinality_conflicts),
        "cardinality_conflicts": cardinality_conflicts,
        "set_equality": (
            not missing
            and not partial
            and not extras
            and not cardinality_conflicts
            and covered_count == len(gold_paths)
        ),
    }


def count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        label = str(item.get(key) or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def load_extra_adjudication(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        payload = payload["items"]
    if isinstance(payload, list):
        return {
            str(item.get("system_path") or ""): str(item.get("classification") or "")
            for item in payload
            if isinstance(item, dict)
            and item.get("system_path")
            and item.get("classification")
        }
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items() if str(value)}
    raise ValueError(f"Unsupported extra-field adjudication format: {path}")


def load_schema_correspondence(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError(f"Unsupported schema-correspondence format: {path}")
    result: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict) or not item.get("manual_path"):
            continue
        manual_path = str(item["manual_path"])
        result[manual_path] = {
            "classification": str(item.get("classification") or ""),
            "system_path": str(item.get("system_path") or ""),
            "rationale": str(item.get("rationale") or ""),
        }
    return result


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.translate(SUBSCRIPT_TRANSLATION)
    text = re.sub(r"_\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"_([A-Za-z0-9+-])", r"\1", text)
    text = text.replace("\\upmu", "u").replace("µ", "u").replace("μ", "u")
    text = re.sub(r"\\(?:mathrm|text|rm|operatorname)", "", text)
    return re.sub(r"[^a-z0-9.+-]", "", text.casefold())


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
    return normalize_text(left) == normalize_text(right)


SCIENTIFIC_STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "the",
    "to",
    "used",
    "using",
}


def _scientific_phrase(value: Any) -> str:
    """Normalize generic scientific phrasing without target- or gold-specific rewrites."""

    text = "" if value is None else str(value)
    text = text.translate(SUBSCRIPT_TRANSLATION).casefold()
    text = text.replace("−", "-").replace("–", "-").replace("→", "->")
    text = text.replace("µ", "u").replace("μ", "u")
    text = re.sub(r"t\s*(?:->|=)\s*0(?:\s*k)?", " 0 k ", text)
    text = re.sub(r"\bzero[-\s]*temperature\b", " 0 k ", text)
    text = re.sub(r"\bambient\s+pressure\b", " ambient ", text)
    text = re.sub(r"\bnormal[-\s]*state\s+resist(?:ance|ivity)\b", " rn ", text)
    text = re.sub(r"\btwo[-\s]*band\s+model\b", " two band ", text)
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    text = re.sub(r"\bmodel\b", " ", text)
    text = re.sub(r"\btc[_\s-]*onset\b", " tc onset ", text)
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.%+\-/]+", " ", text)).strip()
    return text


def _scientific_tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9.%+\-/]+", _scientific_phrase(value))
        if token not in SCIENTIFIC_STOP_WORDS
    }


def _criterion_fraction(value: Any) -> float | None:
    text = _scientific_phrase(value)
    if "rn" not in text:
        return None
    percent = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%", text)
    if percent:
        return float(percent.group(1)) / 100.0
    fraction = re.search(r"(?<![\d.])(0?\.\d+)(?![\d.])", text)
    return float(fraction.group(1)) if fraction else None


def qualifier_values_semantically_equal(key: str, left: Any, right: Any) -> bool:
    """Compare qualifier wording under documented, domain-neutral scientific rules."""

    if values_equal(left, right):
        return True
    if left is None or right is None:
        return False
    name = str(key).casefold()
    if name in {"approximate"}:
        return bool(left) is bool(right) if isinstance(left, bool) and isinstance(right, bool) else False
    if name == "uncertainty":
        left_number = re.search(r"[-+]?\d+(?:\.\d+)?", str(left).replace("±", ""))
        right_number = re.search(r"[-+]?\d+(?:\.\d+)?", str(right).replace("±", ""))
        if left_number and right_number:
            return math.isclose(
                abs(float(left_number.group())),
                abs(float(right_number.group())),
                rel_tol=1e-6,
                abs_tol=1e-9,
            )
    if name == "criterion":
        left_fraction = _criterion_fraction(left)
        right_fraction = _criterion_fraction(right)
        if left_fraction is not None and right_fraction is not None:
            return math.isclose(left_fraction, right_fraction, rel_tol=1e-6, abs_tol=1e-9)
    if name == "result_status":
        interpretive = {
            "author inference",
            "author interpretation",
            "indicated",
            "interpreted",
            "suggested",
        }
        if _scientific_phrase(left) in interpretive and _scientific_phrase(right) in interpretive:
            return True

    left_phrase = _scientific_phrase(left)
    right_phrase = _scientific_phrase(right)
    if left_phrase == right_phrase:
        return True
    left_tokens = _scientific_tokens(left)
    right_tokens = _scientific_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    dice = 2 * len(left_tokens & right_tokens) / (len(left_tokens) + len(right_tokens))
    return dice >= 0.8


def load_evidence_index(pack_dir: Path | None) -> dict[str, dict[int, str]]:
    """Load SOURCE_LINE blocks by paper id for alternative-evidence validation."""

    if pack_dir is None or not pack_dir.exists():
        return {}
    index: dict[str, dict[int, str]] = {}
    pattern = re.compile(
        r"<!--\s*SOURCE_LINE:\s*(\d+)\s*-->\s*(.*?)(?=<!--\s*SOURCE_LINE:|\Z)",
        re.DOTALL,
    )
    for path in sorted(pack_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        index[path.stem] = {
            int(match.group(1)): match.group(2).strip() for match in pattern.finditer(text)
        }
    return index


def gold_evidence_tokens_supported(
    gold: dict[str, Any],
    prediction: dict[str, Any],
    evidence_index: dict[str, dict[int, str]],
) -> bool:
    """Allow a different source line only when it contains all human evidence tokens."""

    paper_id = str(prediction.get("paper_id") or gold.get("paper_id") or "")
    try:
        line = int(prediction.get("evidence_line"))
    except (TypeError, ValueError):
        return False
    text = (evidence_index.get(paper_id) or {}).get(line)
    if text is None:
        return False
    tokens = [normalize_text(item) for item in gold.get("evidence_tokens") or []]
    tokens = [item for item in tokens if item]
    if not tokens:
        try:
            return abs(int(gold.get("evidence_line")) - line) <= 2
        except (TypeError, ValueError):
            return False
    normalized_line = normalize_text(text)
    return all(token in normalized_line for token in tokens)


def strict_fact_match(gold: dict[str, Any], prediction: dict[str, Any]) -> bool:
    for key in ("paper_id", "concept_id", "record_key", "unit"):
        if normalize_text(gold.get(key)) != normalize_text(prediction.get(key)):
            return False
    if str(gold.get("evidence_line")) != str(prediction.get("evidence_line")):
        return False
    if not values_equal(gold.get("value"), prediction.get("value")):
        return False
    predicted_qualifiers = prediction.get("qualifiers") or {}
    return all(
        key in predicted_qualifiers and values_equal(value, predicted_qualifiers[key])
        for key, value in (gold.get("qualifiers") or {}).items()
    )


def scientific_value_unit_match(gold: dict[str, Any], prediction: dict[str, Any]) -> bool:
    """Diagnostic match for identity plus value/unit, excluding qualifiers and evidence."""

    return (
        all(
            normalize_text(gold.get(key)) == normalize_text(prediction.get(key))
            for key in ("paper_id", "concept_id", "record_key", "unit")
        )
        and values_equal(gold.get("value"), prediction.get("value"))
    )


def generic_normalized_fact_match(gold: dict[str, Any], prediction: dict[str, Any]) -> bool:
    for key in ("paper_id", "concept_id", "record_key", "unit"):
        if normalize_text(gold.get(key)) != normalize_text(prediction.get(key)):
            return False
    if not values_equal(gold.get("value"), prediction.get("value")):
        return False
    try:
        if abs(int(gold.get("evidence_line")) - int(prediction.get("evidence_line"))) > 2:
            return False
    except (TypeError, ValueError):
        if str(gold.get("evidence_line")) != str(prediction.get("evidence_line")):
            return False
    predicted_qualifiers = prediction.get("qualifiers") or {}
    return all(
        key in predicted_qualifiers and values_equal(value, predicted_qualifiers[key])
        for key, value in (gold.get("qualifiers") or {}).items()
    )


def semantic_normalized_fact_match(gold: dict[str, Any], prediction: dict[str, Any]) -> bool:
    """Match facts with generic scientific qualifier normalization and line tolerance."""

    for key in ("paper_id", "concept_id", "record_key", "unit"):
        if normalize_text(gold.get(key)) != normalize_text(prediction.get(key)):
            return False
    if not values_equal(gold.get("value"), prediction.get("value")):
        return False
    try:
        if abs(int(gold.get("evidence_line")) - int(prediction.get("evidence_line"))) > 2:
            return False
    except (TypeError, ValueError):
        if str(gold.get("evidence_line")) != str(prediction.get("evidence_line")):
            return False
    predicted_qualifiers = prediction.get("qualifiers") or {}
    return all(
        key in predicted_qualifiers
        and qualifier_values_semantically_equal(key, value, predicted_qualifiers[key])
        for key, value in (gold.get("qualifiers") or {}).items()
    )


def semantic_evidence_supported_fact_match(
    gold: dict[str, Any],
    prediction: dict[str, Any],
    evidence_index: dict[str, dict[int, str]],
) -> bool:
    for key in ("paper_id", "concept_id", "record_key", "unit"):
        if normalize_text(gold.get(key)) != normalize_text(prediction.get(key)):
            return False
    if not values_equal(gold.get("value"), prediction.get("value")):
        return False
    if not required_qualifiers_semantic_match(gold, prediction):
        return False
    return gold_evidence_tokens_supported(gold, prediction, evidence_index)


def required_qualifiers_match(gold: dict[str, Any], prediction: dict[str, Any]) -> bool:
    """Check the human-required qualifier subset without penalizing extra metadata."""
    predicted_qualifiers = prediction.get("qualifiers") or {}
    return all(
        key in predicted_qualifiers and values_equal(value, predicted_qualifiers[key])
        for key, value in (gold.get("qualifiers") or {}).items()
    )


def required_qualifiers_semantic_match(
    gold: dict[str, Any], prediction: dict[str, Any]
) -> bool:
    predicted_qualifiers = prediction.get("qualifiers") or {}
    return all(
        key in predicted_qualifiers
        and qualifier_values_semantically_equal(key, value, predicted_qualifiers[key])
        for key, value in (gold.get("qualifiers") or {}).items()
    )


def component_audit(
    gold: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    evidence_index: dict[str, dict[int, str]] | None = None,
) -> dict[str, Any]:
    """Decompose targeted extraction errors without changing fact-level scoring."""
    by_target: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        target_id = str(prediction.get("target_id") or "")
        if target_id:
            by_target.setdefault(target_id, []).append(prediction)

    metric_names = (
        "paper_id",
        "record_key",
        "concept_id",
        "value",
        "unit",
        "required_qualifiers",
        "required_qualifiers_semantic",
        "evidence_line_exact",
        "evidence_line_within_1",
        "evidence_line_within_2",
        "evidence_gold_tokens_supported",
        "scientific_components_excluding_evidence",
        "scientific_components_excluding_evidence_semantic",
        "scientific_components_with_gold_evidence_semantic",
    )
    counts = {name: 0 for name in metric_names}
    failures: list[dict[str, Any]] = []
    aligned_count = 0

    for fact in gold:
        fact_id = str(fact.get("fact_id") or "")
        candidates = by_target.get(fact_id, [])
        if len(candidates) != 1:
            failures.append(
                {
                    "fact_id": fact_id,
                    "failure_reasons": [
                        "missing_prediction" if not candidates else "duplicate_target_predictions"
                    ],
                    "candidate_count": len(candidates),
                }
            )
            continue

        aligned_count += 1
        prediction = candidates[0]
        component_results = {
            "paper_id": normalize_text(fact.get("paper_id"))
            == normalize_text(prediction.get("paper_id")),
            "record_key": normalize_text(fact.get("record_key"))
            == normalize_text(prediction.get("record_key")),
            "concept_id": normalize_text(fact.get("concept_id"))
            == normalize_text(prediction.get("concept_id")),
            "value": values_equal(fact.get("value"), prediction.get("value")),
            "unit": normalize_text(fact.get("unit")) == normalize_text(prediction.get("unit")),
            "required_qualifiers": required_qualifiers_match(fact, prediction),
            "required_qualifiers_semantic": required_qualifiers_semantic_match(
                fact, prediction
            ),
        }
        try:
            line_delta = abs(int(fact.get("evidence_line")) - int(prediction.get("evidence_line")))
        except (TypeError, ValueError):
            line_delta = None
        component_results.update(
            {
                "evidence_line_exact": line_delta == 0,
                "evidence_line_within_1": line_delta is not None and line_delta <= 1,
                "evidence_line_within_2": line_delta is not None and line_delta <= 2,
                "evidence_gold_tokens_supported": gold_evidence_tokens_supported(
                    fact, prediction, evidence_index or {}
                ),
            }
        )
        component_results["scientific_components_excluding_evidence"] = all(
            component_results[name]
            for name in (
                "paper_id",
                "record_key",
                "concept_id",
                "value",
                "unit",
                "required_qualifiers",
            )
        )
        component_results["scientific_components_excluding_evidence_semantic"] = all(
            component_results[name]
            for name in (
                "paper_id",
                "record_key",
                "concept_id",
                "value",
                "unit",
                "required_qualifiers_semantic",
            )
        )
        component_results["scientific_components_with_gold_evidence_semantic"] = (
            component_results["scientific_components_excluding_evidence_semantic"]
            and component_results["evidence_gold_tokens_supported"]
        )
        for name, passed in component_results.items():
            counts[name] += int(passed)
        failed = [name for name, passed in component_results.items() if not passed]
        if failed:
            failures.append(
                {
                    "fact_id": fact_id,
                    "failure_reasons": failed,
                    "evidence_line_delta": line_delta,
                }
            )

    denominator = len(gold)
    return {
        "alignment_policy": "prediction.target_id must uniquely equal gold.fact_id",
        "gold_count": denominator,
        "aligned_prediction_count": aligned_count,
        "unaligned_gold_count": denominator - aligned_count,
        "metrics": {
            name: {
                "matched": count,
                "total": denominator,
                "rate": count / denominator if denominator else 1.0,
            }
            for name, count in counts.items()
        },
        "failure_count": len(failures),
        "failures": failures,
    }


def score_facts(
    gold: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    matcher: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> dict[str, Any]:
    unmatched = set(range(len(predictions)))
    matched_gold = []
    for fact in gold:
        index = next((i for i in sorted(unmatched) if matcher(fact, predictions[i])), None)
        if index is not None:
            unmatched.remove(index)
            matched_gold.append(str(fact.get("fact_id")))
    tp = len(matched_gold)
    precision = tp / len(predictions) if predictions else 0.0
    recall = tp / len(gold) if gold else 1.0
    return {
        "gold_count": len(gold),
        "prediction_count": len(predictions),
        "true_positive": tp,
        "false_positive": len(predictions) - tp,
        "false_negative": len(gold) - tp,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "matched_fact_ids": matched_gold,
        "unmatched_gold_fact_ids": [
            str(item.get("fact_id")) for item in gold if str(item.get("fact_id")) not in matched_gold
        ],
        "unmatched_prediction_indexes": sorted(unmatched),
    }


def extraction_report(
    gold_payload: dict[str, Any],
    prediction_path: Path | None,
    tiers: dict[str, str],
    evidence_index: dict[str, dict[int, str]] | None = None,
    freshness_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if prediction_path is None or not prediction_path.exists():
        return {"status": "not_scored_missing_fresh_predictions"}
    if freshness_audit is not None and not freshness_audit.get("valid"):
        return {
            "status": "not_scored_nonfresh_predictions",
            "prediction_path": str(prediction_path),
            "freshness_audit": freshness_audit,
        }
    payload = load_json(prediction_path)
    predictions = payload.get("facts") if isinstance(payload, dict) else payload
    if not isinstance(predictions, list):
        raise TypeError(f"Prediction artifact has no facts list: {prediction_path}")
    gold = gold_payload.get("facts") or []
    report: dict[str, Any] = {
        "status": "scored",
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_sha256(prediction_path),
        "scoring_policy": {
            "strict_exact": "paper + concept + record + value + unit + exact evidence line + gold qualifiers",
            "generic_normalized": "generic text/unit normalization plus evidence-line tolerance of two; no paper, target, or gold-value rules",
            "semantic_normalized": "generic scientific phrase normalization for qualifiers plus evidence-line tolerance of two; raw strict scores remain unchanged",
            "semantic_evidence_supported": "human evidence tokens must all occur in the cited SOURCE_LINE; permits valid alternative lines without answer-specific rewriting",
            "gold_specific_canonicalization_allowed": False,
        },
        "overall": {
            "strict_exact": score_facts(gold, predictions, strict_fact_match),
            "scientific_value_unit_diagnostic": score_facts(
                gold, predictions, scientific_value_unit_match
            ),
            "generic_normalized": score_facts(gold, predictions, generic_normalized_fact_match),
            "semantic_normalized": score_facts(
                gold, predictions, semantic_normalized_fact_match
            ),
            "semantic_evidence_supported": score_facts(
                gold,
                predictions,
                lambda expected, actual: semantic_evidence_supported_fact_match(
                    expected, actual, evidence_index or {}
                ),
            ),
        },
        "component_audit": component_audit(gold, predictions, evidence_index),
        "tiers": {},
    }
    for tier in ("core", "secondary"):
        tier_gold = [fact for fact in gold if tiers.get(str(fact.get("concept_id"))) == tier]
        tier_predictions = [
            fact for fact in predictions if tiers.get(str(fact.get("concept_id"))) == tier
        ]
        report["tiers"][tier] = {
            "strict_exact": score_facts(tier_gold, tier_predictions, strict_fact_match),
            "scientific_value_unit_diagnostic": score_facts(
                tier_gold, tier_predictions, scientific_value_unit_match
            ),
            "generic_normalized": score_facts(
                tier_gold, tier_predictions, generic_normalized_fact_match
            ),
            "semantic_normalized": score_facts(
                tier_gold, tier_predictions, semantic_normalized_fact_match
            ),
            "semantic_evidence_supported": score_facts(
                tier_gold,
                tier_predictions,
                lambda expected, actual: semantic_evidence_supported_fact_match(
                    expected, actual, evidence_index or {}
                ),
            ),
        }
    return report


def protocol_report(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        return {"status": "not_applicable", "artifact_count": 0}
    artifacts = []
    for path in paths:
        if not path.exists():
            artifacts.append({"path": str(path), "status": "missing"})
            continue
        payload = load_json(path)
        validation = payload.get("protocol_validation") or {}
        artifacts.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "status": "valid" if validation.get("valid") is True else "invalid_or_unvalidated",
                "validation": validation,
                "message_count": len(payload.get("protocol_messages") or []),
            }
        )
    valid = sum(item.get("status") == "valid" for item in artifacts)
    return {
        "status": "complete" if valid == len(artifacts) else "incomplete",
        "artifact_count": len(artifacts),
        "valid_artifact_count": valid,
        "artifacts": artifacts,
    }


def care_report(
    path: Path | None,
    freshness_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"status": "not_applicable_or_not_replayed"}
    if freshness_audit is not None and not freshness_audit.get("valid"):
        return {
            "status": "not_scored_nonfresh_care_artifact",
            "artifact_path": str(path),
            "freshness_audit": freshness_audit,
        }
    payload = load_json(path)
    accuracy = payload.get("accuracy")
    if accuracy is None:
        accuracy = payload.get("exact_unique_responsibility_accuracy")
    gold_case_count = payload.get("gold_case_count")
    if gold_case_count is None:
        gold_case_count = payload.get("case_count")
    prediction_case_count = payload.get("prediction_case_count")
    if prediction_case_count is None and payload.get("case_count") is not None:
        prediction_case_count = payload.get("case_count")
    coverage = payload.get("coverage")
    if coverage is None and gold_case_count:
        coverage = prediction_case_count / gold_case_count
    return {
        "status": payload.get("status") or "complete",
        "artifact_path": str(path),
        "artifact_sha256": file_sha256(path),
        "gold_case_count": gold_case_count,
        "prediction_case_count": prediction_case_count,
        "coverage": coverage,
        "accuracy": accuracy,
        "macro_f1": payload.get("macro_f1"),
        "scope": payload.get("scope"),
        "metric_semantics": "component responsibility on controlled injected faults; not scientific-value extraction accuracy",
    }


def resolve(base: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def schema_supervisor_report(path: Path | None, required: bool) -> dict[str, Any]:
    if not required:
        return {"required": False, "status": "not_required"}
    if path is None or not path.exists():
        return {
            "required": True,
            "status": "missing_supervisor_decision",
            "expert_gate_passed": False,
        }
    payload = load_json(path)
    return {
        "required": True,
        "status": "pass" if payload.get("expert_gate_passed") is True else "failed",
        "expert_gate_passed": payload.get("expert_gate_passed") is True,
        "path": str(path),
        "sha256": file_sha256(path),
        "decision": payload.get("decision") or payload.get("version_status"),
        "expert_rounds_used": payload.get("expert_rounds_used"),
    }


def terminal_supervisor_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"status": "not_audited_missing_fresh_state"}
    payload = load_json(path)
    decision = payload.get("supervisor_decision") or {}
    terminal_action = decision.get("action")
    return {
        "status": (
            "terminal_failure_preserved"
            if terminal_action == "finish_needs_review"
            else "audited_nonterminal_or_other_decision"
        ),
        "path": str(path),
        "sha256": file_sha256(path),
        "failure_routed_to_supervisor": bool(
            decision and payload.get("last_error_node") and payload.get("last_error_type")
        ),
        "terminal_action": terminal_action,
        "reason": decision.get("reason"),
        "error_node": decision.get("error_node") or payload.get("last_error_node"),
        "error_type": decision.get("error_type") or payload.get("last_error_type"),
        "retry_count": decision.get("retry_count"),
        "retry_counts": payload.get("retry_counts") or {},
        "current_node": payload.get("current_node"),
        "next_node": payload.get("next_node"),
    }


def step8_completion_audit(
    payload: dict[str, Any],
    freshness_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = payload.get("execution_provenance") or {}
    critic = (
        ((payload.get("module_outputs") or {}).get("specialization_critic_module") or {})
        if isinstance(payload.get("module_outputs"), dict)
        else {}
    )
    fallback_markers = []
    if provenance.get("fallback_used") is True:
        fallback_markers.append("execution_provenance.fallback_used")
    if provenance.get("mode") == "blind_deterministic_fallback":
        fallback_markers.append("execution_provenance.mode")
    if critic.get("specialization_status") == "local_blind_structural_review":
        fallback_markers.append("local_blind_structural_review")
    explicit_online_completion = (
        provenance.get("mode") == "online_full_pipeline"
        and provenance.get("all_required_modules_completed") is True
        and payload.get("status") == "success"
    )
    clean_run_valid = freshness_audit is None or bool(freshness_audit.get("valid"))
    return {
        "scorable": explicit_online_completion and not fallback_markers and clean_run_valid,
        "explicit_online_completion": explicit_online_completion,
        "fallback_detected": bool(fallback_markers),
        "fallback_markers": fallback_markers,
        "provenance": provenance,
        "freshness_audit": freshness_audit,
    }


def build_report(
    config_path: Path,
    *,
    require_clean_run_identity: bool = False,
) -> dict[str, Any]:
    config = load_json(config_path)
    campaign_started_at = str(config.get("campaign_started_at") or "").strip()
    if require_clean_run_identity and not campaign_started_at:
        raise ValueError(
            "official three-version evaluation requires config.campaign_started_at"
        )
    base = config_path.parent
    manual_path = resolve(base, config["manual_schema"])
    gold_path = resolve(base, config["gold_facts"])
    metric_path = resolve(base, config["metric_contract"])
    assert manual_path and gold_path and metric_path

    manual_paths = flatten_schema(load_json(manual_path))
    prompt_complete_paths = list(manual_paths)
    for path in config.get("prompt_complete_additions") or []:
        if path not in prompt_complete_paths:
            prompt_complete_paths.append(path)
    gold_payload = load_json(gold_path)
    metric = load_json(metric_path)
    evidence_pack_dir = resolve(base, config.get("evidence_pack_dir"))
    evidence_index = load_evidence_index(evidence_pack_dir)
    tiers = {str(item["concept_id"]): str(item["tier"]) for item in metric.get("concepts") or []}
    core_path_file = resolve(base, config.get("user_approved_core_field_paths"))
    core_paths = load_json(core_path_file) if core_path_file and core_path_file.exists() else None

    report: dict[str, Any] = {
        "evaluation_contract": {
            "manual_schema_path": str(manual_path),
            "manual_schema_sha256": file_sha256(manual_path),
            "manual_json_leaf_count": len(manual_paths),
            "prompt_complete_leaf_count": len(prompt_complete_paths),
            "prompt_complete_additions": config.get("prompt_complete_additions") or [],
            "gold_facts_path": str(gold_path),
            "gold_facts_sha256": file_sha256(gold_path),
            "gold_fact_count": len(gold_payload.get("facts") or []),
            "gold_paper_count": len(
                {str(item.get("paper_id")) for item in gold_payload.get("facts") or []}
            ),
            "core_field_policy": (
                "user_approved_mapping" if core_paths else "not_scored_no_user_approved_mapping"
            ),
            "matching_policy": "explicit hierarchical leaves; exact or approved aliases plus independent one-to-one semantic equivalence; generic EAV/condition containers and reused generated leaves are partial only",
            "evidence_pack_dir": str(evidence_pack_dir) if evidence_pack_dir else None,
            "clean_run_identity_required": require_clean_run_identity,
            "campaign_started_at": campaign_started_at or None,
        },
        "stages": [],
    }

    for stage in config.get("stages") or []:
        step8_path = resolve(base, stage.get("step8_output"))
        step8_state_path = resolve(base, stage.get("step8_state"))
        prediction_path = resolve(base, stage.get("predictions"))
        care_path = resolve(base, stage.get("care_evaluation"))
        protocol_paths = [
            path
            for path in (
                resolve(base, item) for item in stage.get("protocol_artifacts") or []
            )
            if path is not None
        ]
        step8_freshness = (
            clean_run_artifact_audit(
                step8_path,
                campaign_started_at=campaign_started_at,
                expected_run_id=str(stage.get("expected_step8_run_id") or ""),
                expected_pipeline=str(stage.get("expected_step8_pipeline") or ""),
            )
            if require_clean_run_identity
            else None
        )
        prediction_freshness = (
            clean_run_artifact_audit(
                prediction_path,
                campaign_started_at=campaign_started_at,
                expected_run_id=str(stage.get("expected_prediction_run_id") or ""),
                expected_pipeline=str(stage.get("expected_prediction_pipeline") or ""),
            )
            if require_clean_run_identity
            else None
        )
        care_freshness = (
            clean_run_artifact_audit(
                care_path,
                campaign_started_at=campaign_started_at,
                expected_run_id=str(stage.get("expected_care_run_id") or ""),
                expected_pipeline=str(stage.get("expected_care_pipeline") or ""),
            )
            if require_clean_run_identity and care_path is not None
            else None
        )
        protocol_freshness = (
            [
                clean_run_artifact_audit(
                    path,
                    campaign_started_at=campaign_started_at,
                )
                for path in protocol_paths
            ]
            if require_clean_run_identity
            else []
        )
        field_contract_path = resolve(base, stage.get("field_contract"))
        supervisor_decision_configured = bool(stage.get("schema_supervisor_decision"))
        supervisor_decision_path = resolve(base, stage.get("schema_supervisor_decision"))
        extra_adjudication_path = resolve(base, stage.get("extra_field_adjudication"))
        extra_adjudication = load_extra_adjudication(extra_adjudication_path)
        correspondence_path = resolve(base, stage.get("schema_correspondence_adjudication"))
        semantic_correspondence = load_schema_correspondence(correspondence_path)
        if (
            field_contract_path is not None
            and field_contract_path.exists()
            and field_contract_path.resolve() == manual_path.resolve()
        ):
            schema_report = {
                "status": "not_scored_reference_schema_leak",
                "schema_source_type": "invalid_gold_reference_as_system_contract",
                "field_contract": str(field_contract_path),
                "field_contract_sha256": file_sha256(field_contract_path),
                "step8_execution_status": "reference_schema_leak_rejected",
                "core_fields": {"status": "not_scored_reference_schema_leak"},
            }
        elif (
            require_clean_run_identity
            and field_contract_path is not None
            and field_contract_path.exists()
        ):
            schema_report = {
                "status": "not_scored_fixed_contract_is_not_fresh_step8",
                "schema_source_type": "fixed_extraction_field_contract",
                "field_contract": str(field_contract_path),
                "field_contract_sha256": file_sha256(field_contract_path),
                "step8_execution_status": "missing_fresh_task_adaptive_step8",
                "core_fields": {"status": "not_scored_incomplete_step8"},
            }
        elif field_contract_path is not None and field_contract_path.exists():
            system = extract_fixed_contract_fields(load_json(field_contract_path))
            schema_report = {
                "status": "scored",
                "schema_source_type": "fixed_extraction_field_contract",
                "field_contract": str(field_contract_path),
                "field_contract_sha256": file_sha256(field_contract_path),
                "step8_execution_status": "fixed_contract_valid",
                "extra_field_adjudication": str(extra_adjudication_path)
                if extra_adjudication_path
                else None,
                "schema_correspondence_adjudication": str(correspondence_path)
                if correspondence_path
                else None,
                "system_inventory": system,
                "manual_json_238": match_field_inventory(
                    manual_paths,
                    comparison_field_paths(system),
                    extra_adjudication,
                    semantic_correspondence,
                ),
                "prompt_complete_239": match_field_inventory(
                    prompt_complete_paths,
                    comparison_field_paths(system),
                    extra_adjudication,
                    semantic_correspondence,
                ),
            }
            schema_report["core_fields"] = (
                match_field_inventory(
                    core_paths,
                    comparison_field_paths(system),
                    semantic_correspondence=semantic_correspondence,
                )
                if core_paths
                else {"status": "not_scored_no_user_approved_mapping"}
            )
        elif step8_path is None or not step8_path.exists():
            schema_report = {"status": "not_scored_missing_fresh_step8_output"}
        else:
            step8_payload = load_json(step8_path)
            system = extract_system_fields(step8_payload)
            completion_audit = step8_completion_audit(step8_payload, step8_freshness)
            schema_report = {
                "status": (
                    "scored"
                    if completion_audit["scorable"]
                    else "audited_unscored_incomplete_step8"
                ),
                "schema_source_type": "task_adaptive_step8_output",
                "step8_output": str(step8_path),
                "step8_sha256": file_sha256(step8_path),
                "step8_execution_status": step8_payload.get("status"),
                "step8_completion_audit": completion_audit,
                "extra_field_adjudication": str(extra_adjudication_path)
                if extra_adjudication_path
                else None,
                "schema_correspondence_adjudication": str(correspondence_path)
                if correspondence_path
                else None,
                "system_inventory": system,
            }
            if completion_audit["scorable"]:
                schema_report["manual_json_238"] = match_field_inventory(
                    manual_paths,
                    comparison_field_paths(system),
                    extra_adjudication,
                    semantic_correspondence,
                )
                schema_report["prompt_complete_239"] = match_field_inventory(
                    prompt_complete_paths,
                    comparison_field_paths(system),
                    extra_adjudication,
                    semantic_correspondence,
                )
                schema_report["core_fields"] = (
                    match_field_inventory(
                        core_paths,
                        comparison_field_paths(system),
                        semantic_correspondence=semantic_correspondence,
                    )
                    if core_paths
                    else {"status": "not_scored_no_user_approved_mapping"}
                )
            else:
                not_scored = {
                    "status": "not_scored_incomplete_step8",
                    "reason": "field metrics require an accepted online full-pipeline Step8 schema",
                }
                schema_report["manual_json_238"] = dict(not_scored)
                schema_report["prompt_complete_239"] = dict(not_scored)
                schema_report["core_fields"] = (
                    dict(not_scored)
                    if core_paths
                    else {"status": "not_scored_no_user_approved_mapping"}
                )

        freshness_checks = [
            audit
            for audit in (step8_freshness, prediction_freshness, care_freshness)
            if audit is not None
        ] + protocol_freshness
        computed_fresh_run_status = (
            "complete"
            if freshness_checks and all(audit.get("valid") for audit in freshness_checks)
            else "incomplete_or_rejected"
        )
        stage_report = {
            "stage_id": stage.get("stage_id"),
            "label": stage.get("label"),
            "ablation": stage.get("ablation") or {},
            "schema": schema_report,
            "schema_supervisor": schema_supervisor_report(
                supervisor_decision_path, supervisor_decision_configured
            ),
            "terminal_supervisor": terminal_supervisor_report(step8_state_path),
            "extraction": extraction_report(
                gold_payload,
                prediction_path,
                tiers,
                evidence_index,
                prediction_freshness,
            ),
            "care": care_report(care_path, care_freshness),
            "protocol": protocol_report(protocol_paths),
            "freshness_audit": {
                "step8": step8_freshness,
                "predictions": prediction_freshness,
                "care": care_freshness,
                "protocol_artifacts": protocol_freshness,
            },
            "fresh_run_status": (
                computed_fresh_run_status
                if require_clean_run_identity
                else stage.get("fresh_run_status") or "unspecified"
            ),
        }
        report["stages"].append(stage_report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    contract = report["evaluation_contract"]
    lines = [
        "# Superconductivity three-stage comparison",
        "",
        "## Frozen contract",
        "",
        f"- Manual JSON: `{contract['manual_schema_path']}`",
        f"- Manual SHA-256: `{contract['manual_schema_sha256']}`",
        f"- Manual leaves: {contract['manual_json_leaf_count']}",
        f"- Prompt-complete leaves: {contract['prompt_complete_leaf_count']}",
        f"- Extraction gold: {contract['gold_paper_count']} papers / {contract['gold_fact_count']} facts",
        f"- Core-field status: `{contract['core_field_policy']}`",
        "",
        "## Stage summary",
        "",
        "| Version | Fresh status | Manual recall | Reference-overlap precision/F1 | Extras unresolved/total | Set equal | Strict extraction P/R/F1 | Value+unit diagnostic P/R/F1 | Scientific semantic P/R/F1 | CARE attribution | Protocol |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report["stages"]:
        schema = stage["schema"]
        coverage = (
            schema["manual_json_238"]["strict_or_alias_coverage"]
            if schema.get("status") == "scored"
            else None
        )
        extraction = stage["extraction"]
        strict = (
            extraction["overall"]["strict_exact"]
            if extraction.get("status") == "scored"
            else None
        )
        value_unit = (
            extraction["overall"].get(
                "scientific_value_unit_diagnostic",
                extraction["overall"]["strict_exact"],
            )
            if extraction.get("status") == "scored"
            else None
        )
        normalized = (
            extraction["overall"]["generic_normalized"]
            if extraction.get("status") == "scored"
            else None
        )
        semantic = (
            extraction["overall"]["semantic_normalized"]
            if extraction.get("status") == "scored"
            else None
        )
        semantic_evidence = (
            extraction["overall"]["semantic_evidence_supported"]
            if extraction.get("status") == "scored"
            else None
        )
        care = stage["care"]
        protocol = stage["protocol"]
        manual = schema.get("manual_json_238") or {}

        def metric_text(item: dict[str, Any] | None) -> str:
            if not item:
                return "n/a"
            return f"{item['precision']:.3f}/{item['recall']:.3f}/{item['f1']:.3f}"

        care_text = (
            f"{care['accuracy']:.3f}"
            if isinstance(care.get("accuracy"), (int, float))
            else "n/a"
        )
        lines.append(
            f"| {stage['label']} | {stage['fresh_run_status']} | "
            f"{coverage:.3f} | {manual.get('manual_reference_overlap_precision', 0):.3f}/"
            f"{manual.get('manual_reference_overlap_f1', 0):.3f} | "
            f"{manual.get('unresolved_extra_count', 0)}/{manual.get('system_extra_count', 0)} | "
            f"{manual.get('set_equality', False)} | {metric_text(strict)} | "
            f"{metric_text(value_unit)} | {metric_text(semantic)} | {care_text} | "
            f"{protocol.get('status', 'n/a')} |"
            if coverage is not None
            else f"| {stage['label']} | {stage['fresh_run_status']} | n/a | "
            f"n/a | n/a | n/a | {metric_text(strict)} | {metric_text(value_unit)} | "
            f"{metric_text(semantic)} | {care_text} | "
            f"{protocol.get('status', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Extraction component audit",
            "",
            "| Stage | Value | Unit | Required qualifiers exact | Required qualifiers semantic | Evidence exact | Gold tokens supported | Scientific components except evidence exact/semantic |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in report["stages"]:
        extraction = stage["extraction"]
        if extraction.get("status") != "scored":
            continue
        metrics = extraction["component_audit"]["metrics"]

        def component_text(name: str) -> str:
            item = metrics[name]
            return f"{item['matched']}/{item['total']} ({item['rate']:.3f})"

        lines.append(
            f"| {stage['label']} | {component_text('value')} | {component_text('unit')} | "
            f"{component_text('required_qualifiers')} | {component_text('required_qualifiers_semantic')} | "
            f"{component_text('evidence_line_exact')} | "
            f"{component_text('evidence_gold_tokens_supported')} | "
            f"{component_text('scientific_components_excluding_evidence')}/"
            f"{component_text('scientific_components_excluding_evidence_semantic')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- CARE attribution accuracy is not scientific-value extraction accuracy.",
            "- Concept-taxonomy coverage is not hierarchical field coverage.",
            "- Manual recall alone is not schema quality; reference-overlap precision, extras, conflicts, and set equality are reported separately.",
            "- Extra fields count as useful only after independent adjudication; token heuristics remain unresolved.",
            "- Gold-specific canonicalization is excluded from extraction scoring.",
            ("- Core-field coverage uses the supplied user-approved hierarchical mapping."
             if contract.get("core_field_policy") == "user_approved_mapping"
             else "- Core-field coverage remains unscored until a manual-JSON marker or user-approved mapping is supplied."),
        ]
    )
    return "\n".join(lines) + "\n"


def _metric_fraction(item: dict[str, Any]) -> str:
    return f"{item['matched']}/{item['total']} ({item['rate']:.3f})"


def build_specialized_reports(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contract = report["evaluation_contract"]
    field_stages = []
    extraction_stages = []
    counterfactual_stages = []
    failure_stages = []
    qc_stages = []
    for stage in report["stages"]:
        schema = stage["schema"]
        extraction = stage["extraction"]
        care = stage["care"]
        protocol = stage["protocol"]
        field_stages.append(
            {
                "stage_id": stage["stage_id"],
                "label": stage["label"],
                "step8_execution_status": schema.get("step8_execution_status"),
                "manual_json": schema.get("manual_json_238"),
                "prompt_complete": schema.get("prompt_complete_239"),
                "core_fields": schema.get("core_fields"),
                "system_inventory": schema.get("system_inventory"),
                "extra_field_adjudication": schema.get("extra_field_adjudication"),
            }
        )
        extraction_stages.append(
            {
                "stage_id": stage["stage_id"],
                "label": stage["label"],
                "extraction": extraction,
            }
        )
        counterfactual_stages.append(
            {
                "stage_id": stage["stage_id"],
                "label": stage["label"],
                "care": care,
            }
        )
        failure_stages.append(
            {
                "stage_id": stage["stage_id"],
                "label": stage["label"],
                "step8_execution_status": schema.get("step8_execution_status"),
                "field_evaluation_status": schema.get("status"),
                "extraction_evaluation_status": extraction.get("status"),
                "schema_supervisor": stage.get("schema_supervisor") or {},
                "terminal_supervisor": stage.get("terminal_supervisor") or {},
                "missing_manual_fields": (
                    (schema.get("manual_json_238") or {}).get("missing_gold_paths", [])
                    if schema.get("status") == "scored"
                    else None
                ),
                "partial_manual_fields": (
                    (schema.get("manual_json_238") or {}).get("partial", [])
                    if schema.get("status") == "scored"
                    else None
                ),
                "system_extras": (
                    (schema.get("manual_json_238") or {}).get("system_extras", [])
                    if schema.get("status") == "scored"
                    else None
                ),
                "extraction_failures": (
                    (extraction.get("component_audit") or {}).get("failures", [])
                    if extraction.get("status") == "scored"
                    else None
                ),
            }
        )

        care_required = bool((stage.get("ablation") or {}).get("care"))
        protocol_required = bool((stage.get("ablation") or {}).get("structured_protocol"))
        schema_supervisor = stage.get("schema_supervisor") or {"required": False}
        checks = {
            "fresh_stage_artifact_complete": stage.get("fresh_run_status") == "complete",
            "step8_supervisor_passed": schema.get("step8_execution_status")
            in {"success", "fixed_contract_valid"},
            "schema_expert_gate_passed_when_required": (
                not schema_supervisor.get("required")
                or schema_supervisor.get("expert_gate_passed") is True
            ),
            "schema_scored": schema.get("status") == "scored",
            "extraction_scored": extraction.get("status") == "scored",
            "care_passed_when_required": (not care_required)
            or care.get("status") in {"pass", "complete"},
            "protocol_passed_when_required": (not protocol_required)
            or protocol.get("status") == "complete",
        }
        qc_stages.append(
            {
                "stage_id": stage["stage_id"],
                "label": stage["label"],
                "checks": checks,
                "all_stage_checks_passed": all(checks.values()),
            }
        )

    core_scored = contract.get("core_field_policy") == "user_approved_mapping"
    all_stage_checks = all(item["all_stage_checks_passed"] for item in qc_stages)
    qc_payload = {
        "schema_version": "materials-db.three-stage-qc.v1",
        "status": "pass" if all_stage_checks and core_scored else "not_all_gates_passed",
        "core_field_gate": {
            "status": "scored" if core_scored else "not_scored_no_user_approved_mapping",
            "passed": core_scored,
        },
        "stages": qc_stages,
        "all_supervisor_and_evaluation_gates_passed": all_stage_checks and core_scored,
    }
    return {
        "FIELD_COVERAGE_REPORT": {
            "schema_version": "materials-db.field-coverage-report.v1",
            "evaluation_contract": contract,
            "metric_semantics": {
                "manual_missing": "manual hierarchical leaves not covered by an exact or approved alias system leaf",
                "manual_reference_recall": "matched manual leaves divided by all manual leaves",
                "manual_reference_overlap_precision": "matched system leaves divided by all generated system leaves; useful novel fields remain extras for this metric",
                "manual_reference_jaccard": "matched leaves divided by the union of manual and generated leaf inventories",
                "system_extras": "system leaves absent from the manual reference; only independent adjudication may classify utility or redundancy",
                "set_equality": "true only with full manual coverage, no partials, no extras, and no cardinality conflicts",
                "core_fields": "scored only from an explicit manual marker or user-approved hierarchical path mapping",
            },
            "stages": field_stages,
        },
        "EXTRACTION_QUALITY_REPORT": {
            "schema_version": "materials-db.extraction-quality-report.v1",
            "gold_contract": {
                key: contract[key]
                for key in (
                    "gold_facts_path",
                    "gold_facts_sha256",
                    "gold_fact_count",
                    "gold_paper_count",
                )
            },
            "stages": extraction_stages,
        },
        "QC_REPORT": qc_payload,
        "FAILURE_REPORT": {
            "schema_version": "materials-db.three-stage-failure-report.v1",
            "status": "measured_failures_preserved",
            "stages": failure_stages,
        },
        "COUNTERFACTUAL_REPORT": {
            "schema_version": "materials-db.three-stage-counterfactual-report.v1",
            "metric_semantics": "controlled component-responsibility attribution, not scientific extraction accuracy",
            "stages": counterfactual_stages,
        },
    }


def render_specialized_markdown(name: str, payload: dict[str, Any]) -> str:
    if name == "FIELD_COVERAGE_REPORT":
        lines = [
            "# Field coverage report",
            "",
            "| Version | Step8 gate | Manual covered/partial/missing | Recall | Overlap precision/F1 | Jaccard | Extras unresolved/total | Set equal | Cardinality conflicts | Core fields |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for stage in payload["stages"]:
            manual = stage.get("manual_json") or {}
            core = (stage.get("core_fields") or {}).get("status", "scored")
            if stage.get("field_evaluation_status") != "scored":
                lines.append(
                    f"| {stage['label']} | {stage.get('step8_execution_status')} | "
                    f"n/a | n/a | n/a | n/a | n/a | n/a | n/a | {core} |"
                )
                continue
            lines.append(
                f"| {stage['label']} | {stage.get('step8_execution_status')} | "
                f"{manual.get('covered_leaf_count', 0)}/{manual.get('partial_leaf_count', 0)}/"
                f"{manual.get('missing_leaf_count', 0)} | "
                f"{manual.get('strict_or_alias_coverage', 0):.3f} | "
                f"{manual.get('manual_reference_overlap_precision', 0):.3f}/"
                f"{manual.get('manual_reference_overlap_f1', 0):.3f} | "
                f"{manual.get('manual_reference_jaccard', 0):.3f} | "
                f"{manual.get('unresolved_extra_count', 0)}/{manual.get('system_extra_count', 0)} | "
                f"{manual.get('set_equality', False)} | "
                f"{manual.get('cardinality_conflict_count', 0)} | {core} |"
            )
        lines.extend(
            [
                "",
                "Detailed missing paths, partial matches, and every system-only leaf are in the JSON report.",
                ("Core-field coverage uses the supplied user-approved hierarchical mapping."
                 if (payload.get("evaluation_contract") or {}).get("core_field_policy") == "user_approved_mapping"
                 else "Core-field coverage is intentionally unscored without a user-approved hierarchical mapping."),
            ]
        )
        return "\n".join(lines) + "\n"

    if name == "EXTRACTION_QUALITY_REPORT":
        lines = [
            "# Extraction quality report",
            "",
            "| Stage | Strict P/R/F1 | Value+unit diagnostic P/R/F1 | Generic normalized P/R/F1 | Scientific semantic P/R/F1 | Semantic + gold-token evidence P/R/F1 | Value | Unit | Qualifiers exact/semantic | Evidence exact | Gold tokens supported |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for stage in payload["stages"]:
            extraction = stage["extraction"]
            if extraction.get("status") != "scored":
                lines.append(
                    f"| {stage['label']} ({extraction.get('status', 'not_scored')}) | "
                    "n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
                )
                continue
            strict = extraction["overall"]["strict_exact"]
            value_unit = extraction["overall"].get(
                "scientific_value_unit_diagnostic", strict
            )
            normalized = extraction["overall"]["generic_normalized"]
            semantic = extraction["overall"].get("semantic_normalized", normalized)
            semantic_evidence = extraction["overall"].get(
                "semantic_evidence_supported", semantic
            )
            metrics = extraction["component_audit"]["metrics"]
            semantic_qualifiers = metrics.get(
                "required_qualifiers_semantic", metrics["required_qualifiers"]
            )
            lines.append(
                f"| {stage['label']} | {strict['precision']:.3f}/{strict['recall']:.3f}/{strict['f1']:.3f} | "
                f"{value_unit['precision']:.3f}/{value_unit['recall']:.3f}/{value_unit['f1']:.3f} | "
                f"{normalized['precision']:.3f}/{normalized['recall']:.3f}/{normalized['f1']:.3f} | "
                f"{semantic['precision']:.3f}/{semantic['recall']:.3f}/{semantic['f1']:.3f} | "
                f"{semantic_evidence['precision']:.3f}/{semantic_evidence['recall']:.3f}/"
                f"{semantic_evidence['f1']:.3f} | "
                f"{_metric_fraction(metrics['value'])} | {_metric_fraction(metrics['unit'])} | "
                f"{_metric_fraction(metrics['required_qualifiers'])}/"
                f"{_metric_fraction(semantic_qualifiers)} | "
                f"{_metric_fraction(metrics['evidence_line_exact'])} | "
                f"{_metric_fraction(metrics.get('evidence_gold_tokens_supported', metrics['evidence_line_within_1']))} |"
            )
        return "\n".join(lines) + "\n"

    if name == "QC_REPORT":
        lines = ["# QC report", "", f"- Status: `{payload['status']}`"]
        lines.append(f"- Core-field gate: `{payload['core_field_gate']['status']}`")
        for stage in payload["stages"]:
            failed = [key for key, passed in stage["checks"].items() if not passed]
            lines.append(
                f"- {stage['label']}: "
                + ("pass" if not failed else "failed checks: " + ", ".join(failed))
            )
        return "\n".join(lines) + "\n"

    if name == "FAILURE_REPORT":
        lines = ["# Failure report", ""]
        for stage in payload["stages"]:
            supervisor = stage.get("terminal_supervisor") or {}

            def count_or_na(value: Any) -> str:
                return str(len(value)) if isinstance(value, list) else "n/a"

            lines.extend(
                [
                    f"## {stage['label']}",
                    "",
                    f"- Step8 execution status: `{stage.get('step8_execution_status')}`",
                    f"- Terminal supervisor action: `{supervisor.get('terminal_action', 'not_audited')}`",
                    f"- Terminal error: `{supervisor.get('error_node', 'n/a')}` / `{supervisor.get('error_type', 'n/a')}`",
                    f"- Terminal supervisor retry count: {supervisor.get('retry_count', 'n/a')}",
                    f"- Failure routed to supervisor: {supervisor.get('failure_routed_to_supervisor', False)}",
                    f"- Field evaluation status: `{stage.get('field_evaluation_status')}`",
                    f"- Extraction evaluation status: `{stage.get('extraction_evaluation_status')}`",
                    f"- Missing manual fields: {count_or_na(stage['missing_manual_fields'])}",
                    f"- Partial manual fields: {count_or_na(stage['partial_manual_fields'])}",
                    f"- System-only fields: {count_or_na(stage['system_extras'])}",
                    f"- Extraction facts with at least one failed component: {count_or_na(stage['extraction_failures'])}",
                    "",
                ]
            )
        lines.append("Per-field and per-fact failure details are in the JSON report.")
        return "\n".join(lines) + "\n"

    lines = [
        "# Counterfactual report",
        "",
        "CARE values below measure controlled component responsibility, not extraction accuracy.",
        "",
        "| Stage | Status | Cases | Coverage | Accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for stage in payload["stages"]:
        care = stage["care"]
        coverage = care.get("coverage")
        accuracy = care.get("accuracy")
        lines.append(
            f"| {stage['label']} | {care.get('status')} | {care.get('gold_case_count', 'n/a')} | "
            f"{coverage:.3f} | {accuracy:.3f} |"
            if isinstance(coverage, (int, float)) and isinstance(accuracy, (int, float))
            else f"| {stage['label']} | {care.get('status')} | n/a | n/a | n/a |"
        )
    return "\n".join(lines) + "\n"


def write_specialized_reports(
    report: dict[str, Any],
    output_dir: Path,
    run_identity: dict[str, Any] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    payloads = build_specialized_reports(report)
    paths: list[Path] = []
    for name, payload in payloads.items():
        json_path = output_dir / f"{name}.json"
        markdown_path = output_dir / f"{name}.md"
        run_artifact_guard.atomic_write_json(
            json_path,
            payload,
            run_identity=run_identity,
        )
        run_artifact_guard.atomic_write_text(
            markdown_path,
            render_specialized_markdown(name, payload),
            run_identity=run_identity,
        )
        paths.extend((json_path, markdown_path))
    return paths, payloads["QC_REPORT"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    json_path = output_dir / "THREE_STAGE_COMPARISON_REPORT.json"
    md_path = output_dir / "THREE_STAGE_COMPARISON_REPORT.md"
    input_identity = run_artifact_guard.build_input_identity(
        EVALUATION_PIPELINE,
        {"strict_clean_run_inputs": True},
        [config_path],
    )
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline=EVALUATION_PIPELINE,
        output_path=json_path,
        input_identity=input_identity,
        artifact_paths=[output_dir],
    )
    try:
        report = build_report(config_path, require_clean_run_identity=True)
        report["evaluation_run_identity"] = run_identity
        run_artifact_guard.atomic_write_json(
            json_path,
            report,
            run_identity=run_identity,
        )
        run_artifact_guard.atomic_write_text(
            md_path,
            render_markdown(report),
            run_identity=run_identity,
        )
        specialized_paths, qc_payload = write_specialized_reports(
            report,
            output_dir,
            run_identity,
        )
        manifest = {
            "run_identity": run_identity,
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "report_json": str(json_path),
            "report_json_sha256": file_sha256(json_path),
            "report_markdown": str(md_path),
            "report_markdown_sha256": file_sha256(md_path),
            "status": "evaluation_complete_with_terminal_failed_or_unscored_stages"
            if any(stage["fresh_run_status"] != "complete" for stage in report["stages"])
            else "complete",
        }
        comparison_manifest_path = output_dir / "THREE_STAGE_COMPARISON_MANIFEST.json"
        run_artifact_guard.atomic_write_json(
            comparison_manifest_path,
            manifest,
            run_identity=run_identity,
        )
        output_artifacts = [json_path, md_path, *specialized_paths]
        final_manifest = {
            "schema_version": "materials-db.three-stage-final-manifest.v1",
            "run_identity": run_identity,
            "status": (
                "all_supervisor_and_evaluation_gates_passed"
                if qc_payload["all_supervisor_and_evaluation_gates_passed"]
                else "evaluation_complete_with_failed_or_unscored_gates"
            ),
            "config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "artifacts": [
                {"path": str(path), "size": path.stat().st_size, "sha256": file_sha256(path)}
                for path in output_artifacts
            ],
            "qc_summary": qc_payload,
        }
        final_manifest_path = output_dir / "FINAL_MANIFEST.json"
        run_artifact_guard.atomic_write_json(
            final_manifest_path,
            final_manifest,
            run_identity=run_identity,
        )
        manifest["final_manifest"] = str(final_manifest_path)
        manifest["final_manifest_sha256"] = file_sha256(final_manifest_path)
        run_artifact_guard.atomic_write_json(
            comparison_manifest_path,
            manifest,
            run_identity=run_identity,
        )
        gates_passed = bool(qc_payload["all_supervisor_and_evaluation_gates_passed"])
        run_artifact_guard.update_run_status(
            run_identity,
            "completed",
            output_status="success" if gates_passed else "needs_review",
            all_supervisor_and_evaluation_gates_passed=gates_passed,
        )
    except BaseException as exc:
        run_artifact_guard.update_run_status(
            run_identity,
            "interrupted",
            error_type=type(exc).__name__,
            error_message=str(exc)[:1000],
        )
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if gates_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
