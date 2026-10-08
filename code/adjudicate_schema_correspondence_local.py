#!/usr/bin/env python3
"""Locally adjudicate schema correspondence without exporting the manual gold.

This benchmark helper is deliberately conservative.  It counts only explicit
atomic synonyms as coverage, records narrower or broader representations as
partial, and leaves unsupported concepts missing.  The generated-schema side
is classified independently so a large schema cannot receive a perfect score
merely by containing the manual fields somewhere among many extras.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import evaluate_superconductivity_three_stage as evaluator
import run_artifact_guard


PIPELINE = "local_schema_correspondence_adjudication_fresh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def clean(path: str) -> str:
    return re.sub(r"\[\]", "", str(path)).casefold()


def _candidate(
    system_paths: set[str],
    path: str,
    classification: str,
    rationale: str,
) -> dict[str, str] | None:
    if path not in system_paths:
        return None
    return {
        "classification": classification,
        "system_path": path,
        "rationale": rationale,
    }


def explicit_rule(manual_path: str, system_paths: set[str]) -> dict[str, str] | None:
    """Return a conservative benchmark-domain rule for one unmatched leaf."""

    path = clean(manual_path)

    paper_leaf_aliases = {
        "title": "title",
        "year": "year",
        "paper_type": "article_type",
        "doi": "doi",
        "arxiv_id": "arxiv_id",
        "venue": "journal",
    }
    match = re.fullmatch(r"paper_info\.metadata\.([a-z0-9_]+)", path)
    if match and match.group(1) in paper_leaf_aliases:
        leaf = paper_leaf_aliases[match.group(1)]
        return _candidate(
            system_paths,
            f"paper_info.{leaf}",
            "semantic_equivalent",
            "The generated bibliographic leaf stores the same atomic metadata value.",
        )

    if path == "primary_signature":
        return _candidate(
            system_paths,
            "material_info.section0.material_system.formula",
            "partial_narrower_system",
            "A formula covers formula-based targets but not families, stacks, interfaces, or other canonical target signatures.",
        )

    section0_rules = {
        "material_info.section0.electronic_state_tuning_mechanism.tuning_type": (
            "material_info.section0.doping.doping_type",
            "partial_narrower_system",
        ),
        "material_info.section0.electronic_state_tuning_mechanism.dopant_defect": (
            "material_info.section0.doping.doping_element",
            "partial_narrower_system",
        ),
        "material_info.section0.electronic_state_tuning_mechanism.stoichiometry": (
            "material_info.section0.material_system.stoichiometry",
            "partial_broader_system",
        ),
        "material_info.section0.electronic_state_tuning_mechanism.carrier_effect": (
            "material_info.section0.doping.carrier_type",
            "partial_narrower_system",
        ),
        "material_info.section0.carrier_concentration.level": (
            "material_info.section0.doping.carrier_concentration",
            "semantic_equivalent",
        ),
        "material_info.section0.secondary_phases.phase": (
            "material_info.section3.xrd.impurity_phases",
            "partial_narrower_system",
        ),
        "material_info.section0.stack_descriptor.layers.material": (
            "material_info.section2.substrate.material",
            "partial_narrower_system",
        ),
        "material_info.section0.stack_descriptor.layers.thickness": (
            "material_info.section2.film_thickness",
            "partial_narrower_system",
        ),
    }
    if path in section0_rules:
        target, classification = section0_rules[path]
        return _candidate(
            system_paths,
            target,
            classification,
            "The generated leaf captures this information only for a narrower owner or with a broader binding.",
        )

    match = re.fullmatch(r"material_info\.section1\.([a-z0-9_]+)\.([a-z0-9_]+)", path)
    if match:
        family, leaf = match.groups()
        specialized = {
            "tc": "tc_onset",
            "hc2": "hc2_parallel",
        }
        if family in specialized:
            target_family = specialized[family]
            leaf_alias = {
                "direction": None,
                "figure": "evidence",
                "characteristics": "criterion",
            }.get(leaf, leaf)
            if leaf_alias:
                classification = (
                    "partial_broader_system"
                    if leaf == "figure"
                    else "partial_narrower_system"
                )
                return _candidate(
                    system_paths,
                    f"material_info.section1.{target_family}.{leaf_alias}",
                    classification,
                    "The generated schema splits the manual quantity by criterion or orientation, so one generated leaf is not a full one-to-one replacement.",
                )
        family_alias = {"lambda": "penetration_depth", "xi": "coherence_length"}
        if family in family_alias and leaf in {"figure", "characteristics"}:
            target_leaf = "evidence" if leaf == "figure" else (
                "determination_method" if family == "xi" else "measurement_method"
            )
            return _candidate(
                system_paths,
                f"material_info.section1.{family_alias[family]}.{target_leaf}",
                "partial_broader_system" if leaf == "figure" else "partial_narrower_system",
                "The generated field contains only part of the broader manual qualifier or evidence locator.",
            )
        if family in {"jc", "hc1"} and leaf in {"figure", "characteristics"}:
            target_family = "critical_current_density" if family == "jc" else family
            target_leaf = "evidence" if leaf == "figure" else "criterion"
            return _candidate(
                system_paths,
                f"material_info.section1.{target_family}.{target_leaf}",
                "partial_broader_system" if leaf == "figure" else "partial_narrower_system",
                "The generated field preserves only a specific evidence or criterion component of the manual qualifier.",
            )

    section2_rules = {
        "material_info.section2.method": "material_info.section2.synthesis.method",
        "material_info.section2.description": "material_info.section2.synthesis_process",
        "material_info.section2.conditions": "material_info.section2.synthesis.temperature",
    }
    if path in section2_rules:
        return _candidate(
            system_paths,
            section2_rules[path],
            "partial_narrower_system",
            "The generated leaf represents only one preparation stage or one part of the generic manual process record.",
        )

    match = re.fullmatch(r"material_info\.section3\.([a-z0-9_]+)\.(figure|raw_data)", path)
    if match and match.group(2) == "figure":
        family = match.group(1)
        direct = {"xrd", "nmr", "afm", "sem", "tem", "stm"}
        if family in direct:
            return _candidate(
                system_paths,
                f"material_info.section3.{family}.figure_number",
                "semantic_equivalent",
                "The technique-specific generated figure locator stores the same atomic identifier.",
            )
        if family in {"sts", "band_structure", "band_dispersion", "edc", "mdc", "fermi_surface"}:
            parent = "stm" if family == "sts" else "arpes"
            return _candidate(
                system_paths,
                f"material_info.section3.{parent}.figure_number",
                "partial_broader_system",
                "The generated locator groups several related result types and cannot preserve this manual subtype independently.",
            )

    curve_parents = {
        "r_t": "resistivity_curve",
        "r_h": "magnetoresistance_curve",
        "i_v": "iv_curve",
        "specific_heat": "specific_heat_curve",
        "m_t": "magnetization_curve",
        "chi_t": "susceptibility_curve",
    }
    match = re.fullmatch(r"material_info\.section4\.([a-z0-9_]+)\.([a-z0-9_]+)", path)
    if match and match.group(1) in curve_parents:
        family, leaf = match.groups()
        parent = curve_parents[family]
        leaf_aliases: dict[str, tuple[str, str]] = {
            "figure": ("figure_number", "semantic_equivalent"),
            "temperature": ("temperature", "semantic_equivalent"),
            "magnetic_field": ("magnetic_field", "semantic_equivalent"),
            "current": ("current", "semantic_equivalent"),
            "frequency": ("frequency", "semantic_equivalent"),
            "magnetic_direction": ("orientation", "semantic_equivalent"),
            "field_cooling": ("zfc_fc", "semantic_equivalent"),
        }
        target_leaf, classification = leaf_aliases.get(leaf, ("", ""))
        candidate = _candidate(
            system_paths,
            f"material_info.section4.{parent}.{target_leaf}",
            classification,
            "The generated curve-specific leaf stores the same explicit condition or figure identifier.",
        ) if target_leaf else None
        if candidate:
            return candidate
        range_aliases = {
            "temperature": "temperature_range",
            "magnetic_field": "field_range",
        }
        if leaf in range_aliases:
            return _candidate(
                system_paths,
                f"material_info.section4.{parent}.{range_aliases[leaf]}",
                "partial_broader_system",
                "A range field can contain this condition but does not preserve the same atomic condition contract.",
            )

    section5_rules = {
        "section5.gap_symmetry": ("section5.pairing_symmetry.type", "semantic_equivalent"),
        "section5.pairing_mechanism": (
            "section5.mechanism_classification.label",
            "semantic_equivalent",
        ),
        "section5.competing_orders": ("section5.coexisting_order.type", "semantic_equivalent"),
        "section5.calculation_method": (
            "section5.theoretical_model.model_type",
            "partial_narrower_system",
        ),
        "section5.discussion_and_analysis": (
            "section5.mechanism_classification.assignment_basis",
            "partial_narrower_system",
        ),
    }
    if path in section5_rules:
        target, classification = section5_rules[path]
        return _candidate(
            system_paths,
            target,
            classification,
            "The generated theory field represents the same concept or a clearly narrower structured component.",
        )
    return None


def classify_extra(path: str) -> tuple[str, str]:
    lowered = clean(path)
    leaf = lowered.rsplit(".", 1)[-1]
    if leaf in {"experimental_or_computational_conditions", "explicit_missingness", "provenance"}:
        return (
            "workflow_artifact",
            "This generic policy-style field duplicates record contracts rather than defining an independently queryable scientific quantity.",
        )
    if lowered == "material_info.section2.synthesis_process":
        return (
            "redundant",
            "A generic process string duplicates the explicit synthesis, annealing, reduction, substrate, and deposition fields.",
        )
    if any(
        marker in lowered
        for marker in (
            ".tc_onset.",
            ".tc_midpoint.",
            ".tc_zero.",
            ".hc2_parallel.",
            ".hc2_perpendicular.",
        )
    ):
        return (
            "reasonable_specialization",
            "This field is a defensible criterion- or orientation-specific refinement of a broader manual superconductivity field.",
        )
    if leaf in {
        "uncertainty",
        "criterion",
        "measurement_method",
        "determination_method",
        "missingness",
        "evidence",
        "unit",
    }:
        return (
            "reasonable_specialization",
            "This explicit quality, method, unit, missingness, or evidence field improves normalization and provenance beyond the manual leaf.",
        )
    useful_markers = (
        "material_system.",
        "crystal_structure.",
        ".sample.",
        ".pressure.",
        ".strain.",
        "section2.",
        "superfluid_density.",
        "electron_phonon_coupling.",
        "residual_resistivity_ratio.",
        "hall_carrier_density.",
        "topological_classification.",
        "correlation_regime.",
        "spin_fluctuations.",
    )
    if any(marker in lowered for marker in useful_markers):
        return (
            "useful_supplement",
            "This is a distinct, queryable materials or superconductivity datum absent from the manual leaf inventory.",
        )
    if lowered.startswith("paper_info."):
        return (
            "useful_supplement",
            "This bibliographic field improves source discovery or provenance and is not workflow metadata.",
        )
    if lowered.startswith("material_info.section3.") or lowered.startswith("material_info.section4."):
        return (
            "reasonable_specialization",
            "This technique- or curve-specific field preserves a nonredundant result subtype or condition.",
        )
    if lowered.startswith("section5."):
        return (
            "reasonable_specialization",
            "This structured theory leaf refines a broader mechanism, symmetry, model, or competing-order record.",
        )
    return (
        "unclear_requires_review",
        "The local rules cannot establish independent utility without a domain reviewer.",
    )


def adjudicate(manual_paths: list[str], system_paths: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    normalized_system: dict[str, list[str]] = {}
    for path in system_paths:
        normalized_system.setdefault(evaluator.normalize_path(path), []).append(path)
    system_set = set(system_paths)
    used: set[str] = set()
    items: list[dict[str, str]] = []
    for manual_path in manual_paths:
        exact = normalized_system.get(evaluator.normalize_path(manual_path)) or []
        exact = [path for path in exact if path not in used]
        if exact:
            system_path = exact[0]
            used.add(system_path)
            item = {
                "classification": "semantic_equivalent",
                "system_path": system_path,
                "rationale": "Exact or evaluator-approved path alias with the same atomic leaf meaning.",
            }
        else:
            item = explicit_rule(manual_path, system_set) or {
                "classification": "missing",
                "system_path": "",
                "rationale": "No explicit generated leaf preserves this manual information unit.",
            }
            if item["classification"] == "semantic_equivalent":
                if item["system_path"] in used:
                    item = {
                        "classification": "partial_broader_system",
                        "system_path": item["system_path"],
                        "rationale": "The same generated leaf was already used by another manual information unit, so it cannot count twice.",
                    }
                else:
                    used.add(item["system_path"])
        items.append({"manual_path": manual_path, **item})

    extras = []
    for path in system_paths:
        if path in used:
            continue
        classification, rationale = classify_extra(path)
        extras.append(
            {
                "system_path": path,
                "classification": classification,
                "rationale": rationale,
            }
        )
    return items, extras


def run(args: argparse.Namespace) -> dict[str, Any]:
    report_path = Path(args.comparison_report).resolve()
    manual_schema_path = Path(args.manual_schema).resolve()
    correspondence_output = Path(args.correspondence_output).resolve()
    extras_output = Path(args.extras_output).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manual_paths = evaluator.flatten_schema(
        json.loads(manual_schema_path.read_text(encoding="utf-8"))
    )
    stage = next(
        (item for item in report.get("stages") or [] if item.get("stage_id") == args.stage_id),
        None,
    )
    if stage is None:
        raise ValueError(f"stage not found: {args.stage_id}")
    inventory = (stage.get("schema") or {}).get("system_inventory") or {}
    system_paths = evaluator.comparison_field_paths(inventory)
    if not system_paths:
        raise ValueError("comparison report has no generated field-registry paths")

    input_identity = run_artifact_guard.build_input_identity(
        PIPELINE,
        {"stage_id": args.stage_id, "policy": "conservative_local_rules_v1"},
        [report_path, manual_schema_path],
    )
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline=PIPELINE,
        output_path=correspondence_output,
        input_identity=input_identity,
        artifact_paths=[correspondence_output, extras_output],
    )
    try:
        items, extras = adjudicate(manual_paths, system_paths)
        correspondence = {
            "schema_version": "materials-db.local-schema-correspondence.v1",
            "run_identity": run_identity,
            "stage_id": args.stage_id,
            "policy": "conservative_local_rules_v1",
            "gold_exported": False,
            "comparison_report": str(report_path),
            "comparison_report_sha256": sha256(report_path),
            "manual_schema": str(manual_schema_path),
            "manual_schema_sha256": sha256(manual_schema_path),
            "manual_leaf_count": len(manual_paths),
            "system_field_count": len(system_paths),
            "semantic_equivalent_count": sum(item["classification"] == "semantic_equivalent" for item in items),
            "partial_count": sum(item["classification"].startswith("partial_") for item in items),
            "missing_count": sum(item["classification"] == "missing" for item in items),
            "items": items,
        }
        extra_payload = {
            "schema_version": "materials-db.local-generated-extra-adjudication.v1",
            "run_identity": run_identity,
            "stage_id": args.stage_id,
            "policy": "conservative_local_rules_v1",
            "gold_exported": False,
            "system_extra_count": len(extras),
            "classification_counts": evaluator.count_by(extras, "classification"),
            "items": extras,
        }
        run_artifact_guard.atomic_write_json(
            correspondence_output, correspondence, run_identity=run_identity
        )
        run_artifact_guard.atomic_write_json(
            extras_output, extra_payload, run_identity=run_identity
        )
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
        semantic_equivalent_count=correspondence["semantic_equivalent_count"],
        partial_count=correspondence["partial_count"],
        missing_count=correspondence["missing_count"],
        system_extra_count=extra_payload["system_extra_count"],
    )
    return {"correspondence": correspondence, "extras": extra_payload}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-report", required=True)
    parser.add_argument("--manual-schema", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--correspondence-output", required=True)
    parser.add_argument("--extras-output", required=True)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "semantic_equivalent_count": result["correspondence"]["semantic_equivalent_count"],
                "partial_count": result["correspondence"]["partial_count"],
                "missing_count": result["correspondence"]["missing_count"],
                "system_extra_count": result["extras"]["system_extra_count"],
                "extra_classification_counts": result["extras"]["classification_counts"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
