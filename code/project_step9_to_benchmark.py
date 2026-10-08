#!/usr/bin/env python3
"""Project a fresh schema-driven Step9 run onto an answer-free benchmark manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import run_artifact_guard


PIPELINE = "step9_answer_free_benchmark_projection_fresh"
SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-")


def canonical_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.translate(SUBSCRIPT_TRANSLATION)
    text = re.sub(r"_\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"_([A-Za-z0-9+-])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize(value: Any) -> str:
    text = canonical_text(value).replace("µ", "u").replace("μ", "u")
    return re.sub(r"[^a-z0-9.+-]", "", text.casefold())


def tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", canonical_text(value).casefold())
        if len(token) > 1
        and token
        not in {"the", "and", "for", "with", "from", "this", "that", "were", "was"}
    }


def canonicalize_measurement(value: Any, unit: Any) -> tuple[Any, Any, dict[str, Any]]:
    if not isinstance(value, str):
        return value, unit, {}
    text = canonical_text(value)
    qualifiers: dict[str, Any] = {}
    relation = re.match(r"^(>=|<=|>|<|≥|≤)\s*", text)
    if relation:
        marker = relation.group(1)
        qualifiers["relation"] = {">": ">=", "≥": ">=", "≤": "<="}.get(marker, marker)
        text = text[relation.end() :].strip()
    if re.match(r"^[~≈]", text):
        qualifiers["approximate"] = True
        text = re.sub(r"^[~≈]\s*", "", text)
    unit_text = canonical_text(unit)
    number = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text)
    number_with_unit = (
        re.fullmatch(r"([-+]?\d+(?:\.\d+)?)(?:\(\d+\))?\s*" + re.escape(unit_text), text, re.I)
        if unit_text
        else None
    )
    if number_with_unit:
        number = number_with_unit
        text = number.group(1)
    if number:
        numeric = float(text)
        return int(numeric) if numeric.is_integer() else numeric, unit, qualifiers
    return value, unit, qualifiers


def project_concept(field_path: str) -> tuple[str, int] | None:
    """Map executable field paths to benchmark concepts without seeing gold values."""

    path = field_path.casefold()
    rules: tuple[tuple[str, str, int], ...] = (
        (r"^paper_info\.(doi|arxiv_id|title|authors|year)$", "canonical_metadata", 1),
        (r"^material_info\.section0\.material_system\.formula$", "material_identity", 1),
        (r"^material_info\.section0\.material_system\.(chemical_composition|stoichiometry)$", "composition", 1),
        (r"^material_info\.section0\.crystal_structure\.space_group$", "crystal_structure", 1),
        (r"^material_info\.section0\.sample\.sample_id$", "sample_identity", 1),
        (r"^material_info\.section0\.sample\.(form|dimensions)$", "sample_form", 1),
        (r"^material_info\.section0\.(pressure\.value|strain\.value|doping\.doping_level)$", "tuning_conditions", 1),
        (r"^material_info\.section2\.synthesis_process$", "preparation_process", 1),
        (r"^material_info\.section1\.tc_(onset|midpoint|zero)\.value$", "transition_temperature", 1),
        (r"^material_info\.section1\.tc_(onset|midpoint|zero)\.(criterion|uncertainty)$", "transition_criterion", 1),
        (r"^material_info\.section1\.hc1\.value$", "lower_critical_field", 1),
        (r"^material_info\.section1\.hc2_(parallel|perpendicular)\.value$", "upper_critical_field", 1),
        (r"^material_info\.section1\.critical_current_density\.value$", "critical_current_density", 1),
        (r"^material_info\.section1\.gap\.magnitude$", "superconducting_gap", 1),
        (r"^material_info\.section1\.penetration_depth\.value$", "penetration_depth", 1),
        (r"^material_info\.section1\.coherence_length\.value$", "coherence_length", 1),
        (r"^section5\.pairing_symmetry\.type$", "pairing_symmetry", 1),
        (r"^section5\.mechanism_classification\.label$", "pairing_mechanism", 1),
        (r"^section5\.coexisting_order\.type$", "phase_diagram", 1),
        (r"^material_info\.section1\.experimental_or_computational_conditions$", "measurement_context", 1),
        (r"^material_info\.section1\.provenance$", "evidence_provenance", 1),
        (r"^material_info\.section4\.resistivity_curve\.figure_number$", "resistance_temperature_curve", 1),
        (r"^material_info\.section4\.magnetoresistance_curve\.figure_number$", "resistance_field_curve", 1),
        (r"^material_info\.section4\.iv_curve\.figure_number$", "current_voltage_curve", 1),
        (r"^material_info\.section4\.magnetization_curve\.figure_number$", "magnetization_curve", 1),
        (r"^material_info\.section4\.susceptibility_curve\.figure_number$", "susceptibility", 1),
        (r"^material_info\.section4\.specific_heat_curve\.figure_number$", "heat_capacity", 1),
        (r"^material_info\.section4\.hall_curve\.figure_number$", "hall_transport", 2),
        (r"^material_info\.section1\.hall_carrier_density\.value$", "hall_transport", 1),
        (r"^material_info\.section3\.(xrd|nmr|afm|sem|tem|stm|arpes|xps|musr|neutron)\.figure_number$", "spectroscopy", 2),
        (r"^section5\.topological_classification\.type$", "topological_claims", 1),
        (r"^material_info\.section1\.residual_resistivity_ratio\.value$", "normal_state", 1),
        (r"^section5\.correlation_regime\.type$", "normal_state", 2),
        (r"^material_info\.section1\.electron_phonon_coupling\.lambda$", "electron_phonon_coupling", 1),
        (r"^section5\.theoretical_model\.software$", "computational_method", 1),
        (r"^section5\.theoretical_model\.model_type$", "computational_method", 2),
        # Task-adaptive schemas may name the scientific quantity directly instead
        # of expanding the legacy nested path. These aliases are answer-free and
        # are based only on field semantics emitted by Step8.
        (r"^material_info\.section0\.material_name$", "material_identity", 1),
        (r"^material_info\.section0\.material_formula$", "composition", 1),
        (r"^material_info\.section0\.(crystal_structure|space_group)$", "crystal_structure", 1),
        (r"^material_info\.section0\.sample_id$", "sample_identity", 1),
        (r"^(material_info\.section0\.sample_form|sample_info\.sample_dimensions)$", "sample_form", 1),
        (
            r"^material_info\.section0\.(applied_pressure|applied_strain|doping_level|substitution|oxygen_stoichiometry)$",
            "tuning_conditions",
            1,
        ),
        (
            r"^(material_info\.section2\.(synthesis_method|growth_method|process_sequence|topotactic_reduction_method)|process_info\.(annealing_conditions|processing_parameters))$",
            "preparation_process",
            1,
        ),
        (
            r"^material_info\.section1\.superconducting_transition_temperature$",
            "transition_temperature",
            1,
        ),
        (r"^material_info\.section1\.measurement_criterion$", "transition_criterion", 2),
        (r"^material_info\.section1\.lower_critical_field_hc1$", "lower_critical_field", 1),
        (
            r"^material_info\.section1\.upper_critical_field_hc2_parallel_(c|ab)$",
            "upper_critical_field",
            1,
        ),
        (r"^material_info\.section1\.critical_current_density$", "critical_current_density", 1),
        (r"^material_info\.section1\.superconducting_gap_magnitude$", "superconducting_gap", 1),
        (r"^material_info\.section1\.superconducting_gap_symmetry$", "pairing_symmetry", 1),
        (r"^material_info\.section1\.penetration_depth_(c|ab)$", "penetration_depth", 1),
        (r"^material_info\.section1\.coherence_length_(c|ab)$", "coherence_length", 1),
        (r"^section5\.(pairing_mechanism|mechanism_claim)$", "pairing_mechanism", 1),
        (r"^material_info\.section1\.(magnetic_ordering_type|charge_density_wave_presence)$", "phase_diagram", 2),
        (
            r"^material_info\.section1\.measurement_(type|temperature|magnetic_field|pressure|direction|uncertainty)$",
            "measurement_context",
            2,
        ),
        (
            r"^material_info\.section1\.(evidence_locator|provenance_type)$",
            "evidence_provenance",
            1,
        ),
        (r"^material_info\.section1\.residual_resistivity_ratio$", "normal_state", 1),
        (
            r"^material_info\.section1\.electron_phonon_coupling_constant$",
            "electron_phonon_coupling",
            1,
        ),
        (
            r"^material_info\.section3\.(rietveld_refinement|eds_maps|band_structure|fermi_surface_maps|phonon_dispersion|electron_phonon_spectral_function)$",
            "spectroscopy",
            2,
        ),
        (
            r"^(section5\.(topological_invariant|chern_number|z2_invariant|weyl_points|dirac_points|edge_state|fermi_arc)|property_observation_info\.topological_state_indicator)$",
            "topological_claims",
            1,
        ),
        (
            r"^section5\.(theoretical_model|simulation_method|software_used)$",
            "computational_method",
            1,
        ),
    )
    for pattern, concept, priority in rules:
        if re.fullmatch(pattern, path):
            return concept, priority
    return None


def iter_values(value: Any) -> Iterable[tuple[Any, dict[str, Any]]]:
    if isinstance(value, list):
        for item in value:
            yield from iter_values(item)
        return
    if isinstance(value, dict):
        primary = next((key for key in ("value", "label", "identity", "method", "type") if key in value), None)
        if primary:
            yield value[primary], value
            return
        for item in value.values():
            yield from iter_values(item)
        return
    yield value, {}


def locate_evidence_line(
    lines: list[str],
    evidence_text: str,
    source_hint: str,
    anchors: Iterable[Any] = (),
) -> tuple[int | None, float]:
    marker = re.search(r"SOURCE_LINE\s*:\s*(\d+)", f"{source_hint}\n{evidence_text}", re.I)
    if marker:
        return int(marker.group(1)), 1.0
    needle = normalize(evidence_text)
    if len(needle) >= 12:
        exact = [
            index
            for index, line in enumerate(lines, start=1)
            if needle in normalize(line) or (len(normalize(line)) >= 24 and normalize(line) in needle)
        ]
        if len(exact) == 1:
            return exact[0], 1.0
    query = tokens(evidence_text)
    anchor_tokens = tokens(" ".join(canonical_text(item) for item in anchors if item is not None))
    if not query:
        return None, 0.0
    scored: list[tuple[float, int]] = []
    for index, line in enumerate(lines, start=1):
        candidate = tokens(line)
        if not candidate:
            continue
        evidence_score = 2 * len(query & candidate) / (len(query) + len(candidate))
        anchor_score = (
            len(anchor_tokens & candidate) / len(anchor_tokens) if anchor_tokens else 0.0
        )
        score = 0.6 * evidence_score + 0.4 * anchor_score
        scored.append((score, index))
    # Prefer the earliest equally supported source line instead of a later
    # repeated discussion of the same fact.
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored or scored[0][0] < 0.22:
        return None, scored[0][0] if scored else 0.0
    return scored[0][1], scored[0][0]


def parent_path(field_path: str) -> str:
    return field_path.rsplit(".", 1)[0] if "." in field_path else field_path


def inferred_result_status(text: str) -> str:
    lowered = canonical_text(text).casefold()
    if any(word in lowered for word in ("predict", "calculated", "calculation", "theoretical")):
        return "calculated"
    if any(word in lowered for word in ("fit", "fitted", "extrapolat")):
        return "fitted"
    if any(word in lowered for word in ("suggest", "indicat", "interpret")):
        return "author interpretation"
    return "reported"


def build_qualifiers(
    item: dict[str, Any],
    siblings: list[dict[str, Any]],
    required_keys: set[str],
) -> dict[str, Any]:
    qualifiers: dict[str, Any] = {}
    path = str(item.get("field_path") or "").casefold()
    evidence = str(item.get("evidence_text") or "")
    sibling_values = {
        str(sibling.get("field_path") or "").rsplit(".", 1)[-1].casefold(): sibling.get("value")
        for sibling in siblings
        if sibling.get("value") not in (None, "", [])
    }
    embedded_conditions = item.get("conditions")
    if isinstance(embedded_conditions, dict):
        sibling_values.update(
            {
                str(key).casefold(): value
                for key, value in embedded_conditions.items()
                if value not in (None, "", [])
            }
        )
    aliases = {
        "method": ("measurement_method", "determination_method", "method"),
        "criterion": ("criterion",),
        "uncertainty": ("uncertainty",),
        "temperature": ("temperature",),
        "pressure": ("pressure",),
        "magnetic_field": ("magnetic_field", "field_range"),
        "model": ("model", "fitting_model"),
        "direction": ("direction", "orientation"),
    }
    for key in required_keys:
        for alias in aliases.get(key, (key,)):
            if alias in sibling_values:
                qualifiers[key] = sibling_values[alias]
                break
    if "criterion" in required_keys and "criterion" not in qualifiers:
        if ".tc_onset." in path:
            qualifiers["criterion"] = "onset"
        elif ".tc_midpoint." in path:
            qualifiers["criterion"] = "midpoint"
        elif ".tc_zero." in path:
            qualifiers["criterion"] = "zero resistance"
    if "direction" in required_keys and "direction" not in qualifiers:
        if "_parallel." in path:
            qualifiers["direction"] = "parallel"
        elif "_perpendicular." in path:
            qualifiers["direction"] = "perpendicular"
        elif "_parallel_c" in path:
            qualifiers["direction"] = "parallel c"
        elif "_parallel_ab" in path:
            qualifiers["direction"] = "parallel ab"
    if "result_status" in required_keys:
        qualifiers["result_status"] = inferred_result_status(evidence)
    if "approximate" in required_keys and re.search(r"[~≈]|\bapproximately\b", evidence, re.I):
        qualifiers["approximate"] = True
    return qualifiers


def run(args: argparse.Namespace) -> dict[str, Any]:
    extraction_dir = Path(args.extraction_dir).resolve()
    step8_path = Path(args.step8_output).resolve()
    target_path = Path(args.target_manifest).resolve()
    corpus_dir = Path(args.corpus_dir).resolve()
    output_path = Path(args.output).resolve()
    target_payload = json.loads(target_path.read_text(encoding="utf-8"))
    targets = target_payload.get("targets") or []
    target_papers = {str(item.get("paper_id") or "") for item in targets}
    targets_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    required_by_paper_concept: dict[tuple[str, str], set[str]] = defaultdict(set)
    for target in targets:
        key = (
            str(target.get("paper_id") or ""),
            str(target.get("concept_id") or ""),
            normalize(target.get("record_key")),
        )
        targets_by_group[key].append(target)
        required_by_paper_concept[(key[0], key[1])].update(
            str(item) for item in target.get("qualifier_keys") or []
        )

    input_identity = run_artifact_guard.build_input_identity(
        PIPELINE,
        {
            "target_protocol": target_payload.get("protocol"),
            "target_count": len(targets),
            "gold_answer_fields_read": False,
            "projection_policy": "schema_path_primary_values_v1",
        },
        [extraction_dir, step8_path, target_path, corpus_dir],
    )
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline=PIPELINE,
        output_path=output_path,
        input_identity=input_identity,
        artifact_paths=[output_path],
    )
    try:
        source_lines = {
            paper: (corpus_dir / f"{paper}.md").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            for paper in target_papers
            if (corpus_dir / f"{paper}.md").exists()
        }
        rows: list[tuple[str, dict[str, Any], str, int]] = []
        all_items: list[tuple[str, dict[str, Any]]] = []
        input_files = []
        error_files = []
        for path in sorted(extraction_dir.rglob("*.json")):
            if "_batches" in path.parts:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                error_files.append({"path": str(path), "error": str(exc)})
                continue
            if not isinstance(payload.get("extracted_fields"), list):
                continue
            paper_id = Path(str(payload.get("document") or path.stem)).stem
            if paper_id not in target_papers:
                continue
            input_files.append(path)
            for item in payload.get("extracted_fields") or []:
                all_items.append((paper_id, item))
                mapping = project_concept(str(item.get("field_path") or ""))
                if mapping and (paper_id, mapping[0]) in required_by_paper_concept:
                    rows.append((paper_id, item, mapping[0], mapping[1]))

        siblings: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for paper_id, item in all_items:
            siblings[(paper_id, parent_path(str(item.get("field_path") or "")), normalize(item.get("material_system")))].append(item)

        candidates = []
        unresolved_evidence = []
        for paper_id, item, concept_id, priority in rows:
            record_key = canonical_text(
                item.get("record_key")
                or item.get("entity_ref")
                or item.get("material_system")
                or ""
            )
            group = targets_by_group.get((paper_id, concept_id, normalize(record_key)), [])
            required = required_by_paper_concept[(paper_id, concept_id)]
            sibling_group = siblings[
                (paper_id, parent_path(str(item.get("field_path") or "")), normalize(item.get("material_system")))
            ]
            qualifiers = build_qualifiers(item, sibling_group, required)
            evidence_text = str(item.get("evidence_text") or "")
            for value, nested in iter_values(item.get("value")):
                nested_unit = nested.get("unit") if isinstance(nested, dict) else None
                value, unit, measurement_qualifiers = canonicalize_measurement(
                    value, nested_unit if nested_unit is not None else item.get("unit")
                )
                evidence_line, line_score = locate_evidence_line(
                    source_lines.get(paper_id, []),
                    evidence_text,
                    str(item.get("source_hint") or ""),
                    anchors=(record_key, value, unit),
                )
                if evidence_line is None:
                    unresolved_evidence.append(
                        {
                            "paper_id": paper_id,
                            "field_path": item.get("field_path"),
                            "evidence_text": evidence_text[:240],
                            "best_score": line_score,
                        }
                    )
                fact = {
                    "target_id": str(group[0].get("target_id") or "") if len(group) == 1 else "",
                    "paper_id": paper_id,
                    "record_key": record_key,
                    "concept_id": concept_id,
                    "value": value,
                    "unit": unit,
                    "qualifiers": {**qualifiers, **measurement_qualifiers},
                    "evidence_line": evidence_line,
                    "evidence_text": evidence_text,
                    "source_hint": str(item.get("source_hint") or ""),
                    "source_field_path": str(item.get("field_path") or ""),
                    "projection_priority": priority,
                    "evidence_locator_score": line_score,
                }
                candidates.append(fact)

        unique = []
        seen = set()
        for item in sorted(
            candidates,
            key=lambda row: (
                row["paper_id"],
                row["concept_id"],
                normalize(row["record_key"]),
                row["projection_priority"],
                str(row["source_field_path"]),
                normalize(row["value"]),
            ),
        ):
            key = (
                item["paper_id"],
                item["concept_id"],
                normalize(item["record_key"]),
                normalize(item["value"]),
                normalize(item["unit"]),
                item["evidence_line"],
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)

        result = {
            "schema_version": "materials-db.step9-answer-free-projection.v1",
            "run_identity": run_identity,
            "source": "fresh_schema_driven_step9_extraction",
            "gold_answer_fields_read": False,
            "target_manifest": str(target_path),
            "target_manifest_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest().upper(),
            "step8_output": str(step8_path),
            "step8_sha256": hashlib.sha256(step8_path.read_bytes()).hexdigest().upper(),
            "input_file_count": len(input_files),
            "target_paper_count": len(target_papers),
            "facts": unique,
            "conversion_qc": {
                "prediction_count": len(unique),
                "unique_target_id_count": len({item["target_id"] for item in unique if item["target_id"]}),
                "unresolved_evidence_count": len(unresolved_evidence),
                "unresolved_evidence": unresolved_evidence,
                "error_file_count": len(error_files),
                "error_files": error_files,
            },
        }
        run_artifact_guard.atomic_write_json(output_path, result, run_identity=run_identity)
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
        prediction_count=len(result["facts"]),
        target_paper_count=result["target_paper_count"],
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", required=True)
    parser.add_argument("--step8-output", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "input_file_count": result["input_file_count"],
                "target_paper_count": result["target_paper_count"],
                **{
                    key: value
                    for key, value in result["conversion_qc"].items()
                    if key.endswith("_count")
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
