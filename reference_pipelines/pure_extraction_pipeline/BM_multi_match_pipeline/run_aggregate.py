from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from utils.io_utils import load_config, load_json, resolve_target_output_path, slugify_filename, write_paper_scoped_output
from utils.log_utils import setup_logger


SECTION0_TYPE_MAP = {
    "material_identity": "material_identity",
    "composition": "composition",
    "structure": "structure",
    "phase": "phase",
    "synthesis": "synthesis",
    "synthesis_method": "synthesis",
    "processing": "processing",
    "processing_condition": "processing",
    "defect_or_doping": "defect_or_doping",
    "morphology": "morphology",
    "measurement_condition": "measurement_conditions",
    "characterization": "characterization",
    "characterization_result": "characterization",
    "computational_result": "computational_result",
    "mechanism": "mechanism_or_interpretation",
    "mechanism_or_interpretation": "mechanism_or_interpretation",
    "relation_or_trend": "relation_or_trend",
    "tuning": "electronic_state_tuning_mechanism",
    "electronic_state_tuning_mechanism": "electronic_state_tuning_mechanism",
    "carrier_concentration": "carrier_concentration",
    "secondary_phase": "secondary_phases",
    "secondary_phases": "secondary_phases",
    "stack_descriptor": "stack_descriptor",
}
LEGACY_PROPERTY_TYPES = {"Tc", "Jc", "Hc1", "Hc2", "Hc", "P_sc", "P_nsc", "lambda", "xi"}
GENERIC_PROPERTY_TYPES = {
    "property",
    "property_value",
    "performance_metric",
    "transition_temperature",
    "critical_field",
    "critical_current_density",
    "band_gap",
    "Curie_temperature",
    "Neel_temperature",
    "magnetization",
    "coercivity",
    "resistivity",
    "conductivity",
    "ionic_conductivity",
    "specific_capacity",
    "overpotential",
    "Seebeck_coefficient",
    "thermal_conductivity",
    "hardness",
}
PROPERTY_FACT_TYPES = LEGACY_PROPERTY_TYPES | GENERIC_PROPERTY_TYPES
MOJIBAKE_MARKERS = ("è", "é", "å", "ç", "æ", "î", "â", "œ", "‰", "ˆ")
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def choose_display_name(target: dict) -> str:
    aliases = [str(x) for x in target.get("aliases", []) if str(x).strip()]
    for alias in aliases:
        if all(ord(ch) < 128 for ch in alias):
            return alias
    canonical = str(target.get("canonical_name") or "").strip()
    if canonical:
        return canonical
    return str(target.get("target_id") or "")


def maybe_repair_mojibake(text: str) -> str:
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
        return repaired or text
    except UnicodeError:
        return text


def strip_latex(text: str) -> str:
    text = text.replace("\\times", "x")
    text = text.replace("\\perp", "perp")
    text = text.replace("\\parallel", "parallel")
    text = text.replace("\\mathrm", "")
    text = text.replace("\\text", "")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("\\|", "|")
    text = text.replace("\\;", " ")
    text = text.replace("\\,", " ")
    text = text.replace("\\_", "_")
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"([A-Za-z])_([0-9A-Za-z]+)", r"\1\2", text)
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = text.replace("$", "")
    return text


def normalize_string(value: str | None) -> str | None:
    if value is None:
        return None

    text = maybe_repair_mojibake(str(value))
    text = text.translate(SUBSCRIPT_MAP)
    text = strip_latex(text)

    replacements = [
        ("\u2013", "-"),
        ("\u2014", "-"),
        ("\u00d7", "x"),
        ("\u00c5^3", "A^3"),
        ("\u00c5", "A"),
        ("\u03c1-onset", "rho-onset"),
        ("\u03c1-mid", "rho-mid"),
        ("\u03c1-zero", "rho-zero"),
        ("\u03c7-onset", "chi-onset"),
        ("degC", "deg C"),
        ("H ||| [001]", "H || [001]"),
        ("H || [001 ]", "H || [001]"),
        ("H | [001]", "H || [001]"),
        ("H perp", "H_perp"),
        ("H parallel", "H_parallel"),
        ("Hperp", "H_perp"),
        ("Hparallel", "H_parallel"),
        ("Fig. ", "Fig. "),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    text = re.sub(r"Fig\.\s*([0-9]+)\(([a-z])\)-\(([a-z])\)", r"Fig. \1(\2-\3)", text)
    text = re.sub(r"Fig\.\s*([0-9]+)\(([a-z])\)-([a-z])", r"Fig. \1(\2-\3)", text)
    text = re.sub(r"\(\s*H\s*\|\s*\[\s*001\s*\]\s*\)", "(H || [001])", text)
    text = re.sub(r"\bNb\s*6\s*X\b", "Nb6X", text)
    text = re.sub(r"\bNb\s*6\s*Ti\b", "Nb6Ti", text)
    text = re.sub(r"\bNb\s*6\s*Zr\b", "Nb6Zr", text)
    text = re.sub(r"\bNb\s*6\s*Hf\b", "Nb6Hf", text)
    text = re.sub(r"\bT\s*c\b", "Tc", text)
    text = re.sub(r"\bH\s*c1\b", "Hc1", text)
    text = re.sub(r"\bH\s*c2\b", "Hc2", text)
    text = re.sub(r"\bJ\s*c\b", "Jc", text)
    text = re.sub(r"\bxi\s*GL\(0\)", "xi_GL(0)", text)
    text = re.sub(r"\blambda\s*GL\(0\)", "lambda_GL(0)", text)
    text = re.sub(r"\bH\s*c\(0\)", "Hc(0)", text)
    text = re.sub(r"\bH\s*2\^orb\(0\)", "Hc2^orb(0)", text)
    text = re.sub(r"\bH\s*2\^P\b", "Hc2^P", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("( ", "(").replace(" )", ")")
    return text


def normalize_field_key(value: str | None) -> str | None:
    if value is None:
        return None
    text = maybe_repair_mojibake(str(value))
    text = text.translate(SUBSCRIPT_MAP)
    text = text.replace("\\_", "_")
    text = text.replace("$", "")
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^0-9A-Za-z_./()+\-]+", "_", text)
    text = text.strip("_")
    return text or None


def normalize_conditions(conditions: dict) -> dict:
    return {key: normalize_string(value) for key, value in (conditions or {}).items()}


def choose_aliases(target: dict) -> list[str]:
    aliases: list[str] = []
    for alias in target.get("aliases", []):
        cleaned = normalize_string(str(alias))
        if not cleaned:
            continue
        if not all(ord(ch) < 128 for ch in cleaned):
            continue
        if cleaned not in aliases:
            aliases.append(cleaned)
    return aliases


def dedupe_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], dict] = {}
    for record in records:
        conditions_key = json.dumps(record.get("conditions") or {}, ensure_ascii=False, sort_keys=True)
        key = (
            str(record.get("value") or ""),
            str(record.get("unit") or ""),
            conditions_key,
        )
        if key not in grouped:
            grouped[key] = dict(record)
            continue

        existing = grouped[key]
        supports = list(existing.get("supporting_evidence", []))
        extra = {
            "verbatim_evidence": record.get("verbatim_evidence"),
            "source_anchor": record.get("source_anchor"),
        }
        if extra not in supports and (
            extra.get("verbatim_evidence") != existing.get("verbatim_evidence")
            or extra.get("source_anchor") != existing.get("source_anchor")
        ):
            supports.append(extra)
        if supports:
            existing["supporting_evidence"] = supports

    return list(grouped.values())


def build_fact_record(candidate: dict) -> dict:
    record = {
        "fact_type": normalize_field_key(candidate.get("fact_type")),
        "value": normalize_string(candidate.get("value")),
        "unit": normalize_string(candidate.get("unit")),
        "conditions": normalize_conditions(candidate.get("conditions", {}) or {}),
    }
    if candidate.get("property_name"):
        record["property_name"] = normalize_field_key(candidate.get("property_name"))
    for key in (
        "property_category",
        "method",
        "measurement_method",
        "sample_form",
        "phase",
        "relation",
    ):
        if candidate.get(key):
            record[key] = normalize_string(candidate.get(key))
    if candidate.get("verbatim_evidence"):
        record["verbatim_evidence"] = normalize_string(candidate["verbatim_evidence"])
    if candidate.get("source_anchor"):
        record["source_anchor"] = {
            key: normalize_string(value)
            for key, value in candidate["source_anchor"].items()
        }
    if candidate.get("local_material_mentions"):
        mentions = [
            normalize_string(value)
            for value in candidate.get("local_material_mentions", [])
            if normalize_string(value)
        ]
        if mentions:
            record["local_material_mentions"] = mentions
    return record


def normalize_section0_key(fact_type: str | None) -> str | None:
    return SECTION0_TYPE_MAP.get(str(fact_type or ""))


def property_output_key(candidate: dict) -> str | None:
    property_name = str(candidate.get("property_name") or "").strip()
    if property_name:
        return normalize_field_key(property_name)

    fact_type = str(candidate.get("fact_type") or "").strip()
    if fact_type in PROPERTY_FACT_TYPES:
        return normalize_field_key(fact_type)

    if fact_type and fact_type not in SECTION0_TYPE_MAP and candidate.get("value") is not None:
        return normalize_field_key(fact_type)

    return None


def run_aggregate(paper_id: str, target_id: str) -> None:
    logger = setup_logger("aggregate")
    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    outputs_dir = str((root / config["paths"]["outputs_dir"]).resolve())

    manifest = load_json(str(Path(outputs_dir) / "manifests" / f"{paper_id}.json"))
    matched = load_json(resolve_target_output_path(outputs_dir, "matched", paper_id, slugify_filename(target_id)))
    fact_candidates = load_json(str(Path(outputs_dir) / "fact_candidates" / f"{paper_id}.json"))
    target = next(item for item in manifest["material_targets"] if item["target_id"] == target_id)
    display_name = choose_display_name(target)
    sibling_display_names = [
        choose_display_name(item)
        for item in manifest.get("material_targets", [])
        if item["target_id"] != target_id
    ]

    candidates_by_id = {
        item["candidate_id"]: item
        for item in fact_candidates.get("paper_level_candidates", [])
    }

    section0: dict = {}
    section1: dict = {}
    accepted_ids: list[str] = []

    for item in matched.get("accepted_candidates", []):
        candidate_id = item["candidate_id"]
        candidate = candidates_by_id.get(candidate_id)
        if not candidate:
            continue
        accepted_ids.append(candidate_id)
        fact_type = candidate.get("fact_type")
        record = build_fact_record(candidate)
        normalized_section0_key = normalize_section0_key(fact_type)
        property_key = property_output_key(candidate)
        if normalized_section0_key:
            section0.setdefault(normalized_section0_key, []).append(record)
        elif property_key:
            section1.setdefault(property_key, []).append(record)

    for key, records in list(section0.items()):
        section0[key] = dedupe_records(records)
    for key, records in list(section1.items()):
        section1[key] = dedupe_records(records)

    ambiguous_ids = [item["candidate_id"] for item in matched.get("ambiguous_candidates", [])]
    active_section0_types = set(section0.keys())
    ambiguity_flags: list[str] = []
    for item in matched.get("ambiguous_candidates", []):
        fact_type = str(item.get("fact_type") or "")
        candidate = candidates_by_id.get(str(item.get("candidate_id") or ""), item)
        normalized_section0_key = normalize_section0_key(fact_type)
        property_key = property_output_key(candidate)
        # Keep ambiguity notes when they are directly tied to emitted section1 facts,
        # or to section0 categories that we actually kept for this target. This trims
        # noisy family-level structural ambiguities that do not surface in the output.
        if not property_key and normalized_section0_key not in active_section0_types:
            continue
        reason = normalize_string(item.get("reason"))
        if reason and reason not in ambiguity_flags:
            ambiguity_flags.append(reason)

    omission_reasons: list[str] = []
    if not section0:
        omission_reasons.append("No accepted target context facts for this target.")
    if not section1:
        omission_reasons.append("No accepted target-specific property facts for this target.")

    final = {
        "primary_signature": normalize_string(display_name),
        "material_info": {
            "section0": section0,
            "section1": section1,
            "section2": [],
            "section3": {},
            "section4": {},
        },
        "section5": {},
        "paper_info": {
            "metadata": {
                "paper_id": paper_id,
                "target_id": target_id,
                "canonical_name": normalize_string(display_name),
                "aliases_used": choose_aliases(target),
                "excluded_siblings": [normalize_string(name) for name in sibling_display_names],
                "multi_material_provenance": {
                    "accepted_candidate_ids": accepted_ids,
                    "ambiguous_candidate_ids": ambiguous_ids,
                },
                "quality_control": {
                    "has_target_specific_property_evidence": bool(section1),
                    "ambiguity_flags": ambiguity_flags,
                    "omission_reasons": omission_reasons,
                },
            },
            "resources": {},
        },
    }

    out_path = write_paper_scoped_output(
        outputs_dir,
        "final_targets",
        paper_id,
        slugify_filename(target_id),
        json.dumps(final, ensure_ascii=False, indent=2),
    )
    logger.info("done output=%s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_id", required=True)
    parser.add_argument("--target_id", required=True)
    args = parser.parse_args()
    run_aggregate(args.paper_id, args.target_id)


if __name__ == "__main__":
    main()
