"""Replay CARE interventions through persisted Step9 production boundaries."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import downstream_extraction_runner as extraction_runner
import run_artifact_guard
from care_counterfactual import (
    CASE_SCHEMA,
    COMPONENTS,
    content_sha256,
    file_sha256,
    hierarchical_minimal_repair_search,
    strict_record_match,
    strict_record_mismatches,
)


SNAPSHOT_SCHEMA = "care-ie.step9_snapshot_manifest.v1"
REPLAY_SCHEMA = "care-ie.step9_production_replay.v1"
STAGES = (
    "paper_info",
    "material_info.section0",
    "material_info.section2.method_pass",
    "material_info.section2.conditions_pass",
    "material_info.section3",
    "figure_classification",
    "material_info.section4",
    "material_info.section1",
    "section5",
)
DEFAULT_PAPERS = (
    "arxiv__2607.08553v1",
    "arxiv__2607.12442v1",
    "arxiv__2607.11003v1",
    "arxiv__2607.04928v1",
    "arxiv__2607.10668v1",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(
    path: Path,
    payload: Any,
    *,
    run_identity: dict[str, Any] | None = None,
) -> None:
    run_artifact_guard.atomic_write_json(
        path,
        payload,
        run_identity=run_identity,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_immutable(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise run_artifact_guard.CleanRunRequiredError(
            f"Fresh CARE replay refused to reuse an existing snapshot: {destination}"
        )
    shutil.copy2(source, destination)


def _artifact(path: Path, snapshot_root: Path) -> dict[str, Any]:
    return {
        "path": relative(path, snapshot_root),
        "size": path.stat().st_size,
        "sha256": file_sha256(str(path)),
    }


def verify_snapshot_manifest(snapshot_root: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
        raise RuntimeError("unexpected Step9 snapshot manifest schema")
    failures = []
    for paper in manifest.get("papers", []):
        for artifact in paper.get("artifacts", []):
            path = snapshot_root / artifact["path"]
            actual = file_sha256(str(path)) if path.exists() else None
            if actual != artifact.get("sha256"):
                failures.append({"path": artifact["path"], "actual_sha256": actual})
    if failures:
        raise RuntimeError(f"snapshot verification failed: {failures}")


def verify_source_step9_run(
    run_root: Path,
    paper_ids: tuple[str, ...],
    source_stage_root: Path | None = None,
) -> dict[str, Any]:
    """Require every replayed stage artifact to belong to one completed Step9 run."""
    stage_root = (
        source_stage_root
        if source_stage_root is not None
        else run_root / "optimization" / "step9_full_corpus_extraction_runs"
    ).resolve()
    identities: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        for stage in STAGES:
            path = stage_root / stage / f"{paper_id}.json"
            payload = read_json(path)
            identity = payload.get("run_identity") if isinstance(payload, dict) else None
            if not isinstance(identity, dict) or not identity.get("run_id"):
                raise run_artifact_guard.RunIdentityError(
                    f"CARE replay requires a fresh identity-bearing Step9 artifact: {path}"
                )
            identities.append(identity)

    run_ids = {str(identity.get("run_id")) for identity in identities}
    fingerprints = {
        str(identity.get("input_fingerprint")) for identity in identities
    }
    if len(run_ids) != 1 or len(fingerprints) != 1:
        raise run_artifact_guard.RunIdentityError(
            "CARE replay refused mixed Step9 artifacts from different runs or inputs."
        )

    identity = identities[0]
    if identity.get("pipeline") != "step9_extraction_build":
        raise run_artifact_guard.RunIdentityError(
            "CARE replay source identity is not a Step9 extraction-build run."
        )
    manifest = run_artifact_guard.load_manifest(identity)
    if manifest.get("status") != "completed":
        raise run_artifact_guard.RunIdentityError(
            f"CARE replay requires completed Step9 artifacts; source is {manifest.get('status')}."
        )
    declared_paths = {
        str(Path(path).resolve(strict=False))
        for path in manifest.get("artifact_paths", [])
    }
    stage_root_owned = any(
        stage_root == declared or stage_root.is_relative_to(declared)
        for declared in (Path(path) for path in declared_paths)
    )
    if not stage_root_owned:
        raise run_artifact_guard.RunIdentityError(
            "The Step9 run manifest does not own the extraction directory being replayed."
        )
    return identity


def build_replay_input_identity(
    *,
    run_root: Path,
    repo_root: Path,
    paper_ids: tuple[str, ...],
    repeats: int,
    source_step9_run_identity: dict[str, Any],
    source_stage_root: Path | None = None,
    schema_source: Path | None = None,
) -> dict[str, Any]:
    opt = run_root / "optimization"
    stage_root = source_stage_root or opt / "step9_full_corpus_extraction_runs"
    schema_path = schema_source or opt / "step8_superconductor_28paper_verified.json"
    return run_artifact_guard.build_input_identity(
        "care_step9_replay",
        {
            "paper_ids": list(paper_ids),
            "repeats": repeats,
            "scope_label": "superconductivity_only",
            "source_step9_run_id": source_step9_run_identity["run_id"],
            "source_step9_input_fingerprint": source_step9_run_identity[
                "input_fingerprint"
            ],
        },
        [
            opt / "STEP9_FULL_PACK_COVERAGE.json",
            opt / "counterfactual_eval_v1" / "FROZEN_INPUT_MANIFEST.json",
            schema_path,
            opt / "convert_step9_output_to_predictions.py",
            stage_root,
            repo_root / "code" / "care_step9_replay.py",
            repo_root / "code" / "care_counterfactual.py",
            repo_root / "code" / "step9_extraction_build_graph.py",
            repo_root / "code" / "downstream_extraction_runner.py",
        ],
    )


def build_step9_snapshots(
    *,
    run_root: Path,
    repo_root: Path,
    snapshot_root: Path,
    source_step9_run_identity: dict[str, Any],
    run_identity: dict[str, Any],
    paper_ids: tuple[str, ...] = DEFAULT_PAPERS,
    source_stage_root: Path | None = None,
    schema_source: Path | None = None,
) -> dict[str, Any]:
    manifest_path = snapshot_root / "SNAPSHOT_MANIFEST.json"
    if manifest_path.exists():
        raise run_artifact_guard.CleanRunRequiredError(
            f"Fresh CARE replay refused to reuse {manifest_path}"
        )

    opt = run_root / "optimization"
    coverage = read_json(opt / "STEP9_FULL_PACK_COVERAGE.json")
    pack_by_id = {item["paper_id"]: item for item in coverage["packs"]}
    frozen = set(
        read_json(opt / "counterfactual_eval_v1" / "FROZEN_INPUT_MANIFEST.json")[
            "in_domain_frozen_papers"
        ]
    )
    missing = sorted(set(paper_ids) - set(pack_by_id))
    not_frozen = sorted(set(paper_ids) - frozen)
    if missing or not_frozen:
        raise RuntimeError(
            f"snapshot selection gate failed: missing={missing}, not_frozen={not_frozen}"
        )

    production_paths = {
        "step9_graph": repo_root / "code" / "step9_extraction_build_graph.py",
        "extraction_runner": repo_root / "code" / "downstream_extraction_runner.py",
        "prediction_converter": opt / "convert_step9_output_to_predictions.py",
    }
    stage_root = source_stage_root or opt / "step9_full_corpus_extraction_runs"
    schema_source = schema_source or opt / "step8_superconductor_28paper_verified.json"
    papers = []
    for paper_id in paper_ids:
        item = pack_by_id[paper_id]
        paper_root = snapshot_root / "papers" / paper_id
        sources = {
            "source.md": Path(item["source_path"]),
            "evidence_pack.md": Path(item["pack_path"]),
            "metadata.json": Path(item["metadata_path"]),
            "schema.json": schema_source,
        }
        for name, source in sources.items():
            _copy_immutable(source, paper_root / name)
        for stage in STAGES:
            _copy_immutable(
                stage_root / stage / f"{paper_id}.json",
                paper_root / "stages" / f"{stage}.json",
            )
        artifacts = [
            _artifact(path, snapshot_root)
            for path in sorted(paper_root.rglob("*"))
            if path.is_file()
        ]
        papers.append(
            {
                "paper_id": paper_id,
                "split": "in_domain_frozen",
                "stage_count": len(STAGES),
                "artifacts": artifacts,
                "snapshot_sha256": content_sha256(artifacts),
            }
        )

    production_code = {
        name: {
            "path": str(path.resolve()),
            "sha256": file_sha256(str(path)),
        }
        for name, path in production_paths.items()
    }
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_label": "superconductivity_only",
        "run_identity": run_identity,
        "source_step9_run_identity": source_step9_run_identity,
        "replay_scope": "persisted_step9_outputs_plus_production_postprocess_nodes",
        "online_llm_regeneration": False,
        "paper_count": len(papers),
        "expected_stage_count_per_paper": len(STAGES),
        "production_code": production_code,
        "production_version_sha256": content_sha256(production_code),
        "papers": papers,
    }
    write_json(manifest_path, manifest, run_identity=run_identity)
    return manifest


def load_snapshot_trace(snapshot_root: Path, paper_id: str) -> dict[str, Any]:
    paper_root = snapshot_root / "papers" / paper_id
    schema = read_json(paper_root / "schema.json")
    registry = schema["result"]["schema_definition"]["field_registry"]
    field_map = {}
    for field in registry:
        for concept_id in field.get("concept_ids", []):
            field_map[field["field_path"]] = concept_id
    stages = {
        stage: read_json(paper_root / "stages" / f"{stage}.json")
        for stage in STAGES
    }
    return {
        "schema": {"field_map": field_map},
        "evidence": {
            "source_lines": (paper_root / "source.md")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines(),
            "item_overrides": {},
        },
        "extraction": {"stage_payloads": stages, "item_overrides": {}},
        "binding": {"record_key_overrides": {}},
        "normalization": {"fact_overrides": {}},
        "target_locator": "",
        "final_record": {},
    }


def _locator(stage: str, extracted_index: int, value_index: int) -> str:
    return f"{stage}:{extracted_index}:{value_index}"


def _record_projection(fact: dict[str, Any]) -> dict[str, Any]:
    record_key = fact["record_key"]
    return {
        "paper_id": fact["paper_id"],
        "concept_id": fact["concept_id"],
        "record_key": record_key,
        "value": fact["value"],
        "unit": fact["unit"],
        "qualifiers": fact["qualifiers"],
        "evidence_line": fact["evidence_line"],
        "evidence": fact["evidence_text"],
        "binding": {"record_key": record_key},
    }


def production_replay(trace: dict[str, Any], converter: ModuleType) -> dict[str, Any]:
    """Run persisted stage outputs through the production postprocessing nodes."""

    field_map = trace["schema"]["field_map"]
    source_lines = trace["evidence"]["source_lines"]
    evidence_overrides = trace["evidence"].get("item_overrides") or {}
    extraction_overrides = trace["extraction"].get("item_overrides") or {}
    binding_overrides = trace["binding"].get("record_key_overrides") or {}
    fact_overrides = trace["normalization"].get("fact_overrides") or {}
    facts = []
    stage_checks = {}

    for stage, original_payload in trace["extraction"]["stage_payloads"].items():
        payload = extraction_runner.sanitize_payload(copy.deepcopy(original_payload))
        stage_checks[stage] = extraction_runner.validate_stage_payload(
            copy.deepcopy(payload)
        )
        paper_id = converter.resolve_paper_id(
            str(payload.get("document") or f"{stage}.json")
        )
        for extracted_index, original_item in enumerate(payload.get("extracted_fields") or []):
            if not isinstance(original_item, dict):
                continue
            base_locator = _locator(stage, extracted_index, 0)
            item = copy.deepcopy(original_item)
            item.update(extraction_overrides.get(base_locator) or {})
            item.update(evidence_overrides.get(base_locator) or {})
            field_path = str(item.get("field_path") or "")
            matching_paths = [
                path
                for path in field_map
                if field_path == path or field_path.startswith(path + ".")
            ]
            if not matching_paths:
                continue
            registered_path = max(matching_paths, key=len)
            concept_id = field_map[registered_path]
            evidence_text = str(item.get("evidence_text") or "")
            source_hint = str(item.get("source_hint") or "")
            evidence_line = converter.locate_evidence_line(
                source_lines, evidence_text, source_hint
            )
            for value_index, (value, nested) in enumerate(
                converter.iter_values(item.get("value"))
            ):
                locator = _locator(stage, extracted_index, value_index)
                qualifiers = {
                    **(nested.get("_qualifiers") or {}),
                    **{
                        key: item[key]
                        for key in ("method", "criterion", "result_status", "confidence")
                        if key in item
                    },
                }
                record_key = (
                    binding_overrides.get(locator)
                    if locator in binding_overrides
                    else nested.get("record_key")
                    or nested.get("entity_ref")
                    or nested.get("material_system")
                    or item.get("material_system")
                    or ""
                )
                record_key, record_qualifiers = converter.canonicalize_record_key(
                    record_key
                )
                unit = nested.get("unit", item.get("unit"))
                value, unit, measurement_qualifiers = converter.canonicalize_measurement(
                    value, unit
                )
                qualifiers = converter.infer_qualifiers(
                    concept_id,
                    value,
                    evidence_text,
                    {**qualifiers, **record_qualifiers, **measurement_qualifiers},
                )
                fact = {
                    "paper_id": paper_id,
                    "record_key": record_key,
                    "concept_id": concept_id,
                    "value": value,
                    "unit": unit,
                    "qualifiers": qualifiers,
                    "evidence_line": evidence_line,
                    "evidence_text": evidence_text,
                    "source_hint": source_hint,
                    "source_field_path": field_path,
                    "registered_field_path": registered_path,
                    "locator": locator,
                }
                fact.update(fact_overrides.get(locator) or {})
                facts.append(fact)

    unique = []
    seen = set()
    for fact in facts:
        key = (
            fact["paper_id"],
            fact["concept_id"],
            converter.normalize(fact["record_key"]),
            converter.normalize(fact["value"]),
            converter.normalize(fact["unit"]),
            fact["evidence_line"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(fact)

    target_locator = trace.get("target_locator")
    candidates = [fact for fact in unique if fact["locator"] == target_locator]
    if not candidates:
        raise RuntimeError(f"target locator did not survive production replay: {target_locator}")
    return {
        "schema_version": REPLAY_SCHEMA,
        "final_record": _record_projection(candidates[0]),
        "fact_count": len(unique),
        "stage_checks": stage_checks,
    }


def select_target(trace: dict[str, Any], converter: ModuleType) -> dict[str, Any]:
    trace = copy.deepcopy(trace)
    trace["target_locator"] = ""
    # The temporary target is replaced after directly enumerating all candidate facts.
    field_map = trace["schema"]["field_map"]
    candidates = []
    for stage, payload in trace["extraction"]["stage_payloads"].items():
        clean = extraction_runner.sanitize_payload(copy.deepcopy(payload))
        for extracted_index, item in enumerate(clean.get("extracted_fields") or []):
            if not isinstance(item, dict):
                continue
            field_path = str(item.get("field_path") or "")
            matching = [
                path for path in field_map if field_path == path or field_path.startswith(path + ".")
            ]
            if not matching:
                continue
            for value_index, (value, nested) in enumerate(converter.iter_values(item.get("value"))):
                if value_index != 0:
                    continue
                if extraction_runner.is_null_like(value):
                    continue
                locator = _locator(stage, extracted_index, value_index)
                record_key = (
                    nested.get("record_key")
                    or nested.get("entity_ref")
                    or nested.get("material_system")
                    or item.get("material_system")
                    or ""
                )
                score = (
                    int(bool(record_key)),
                    int(bool(item.get("evidence_text"))),
                    int(item.get("unit") not in {None, ""}),
                    int(stage != "paper_info"),
                )
                candidates.append((score, locator, max(matching, key=len)))
    if not candidates:
        raise RuntimeError("snapshot has no replayable mapped facts")
    _, locator, registered_path = max(candidates, key=lambda item: (item[0], item[1]))
    trace["target_locator"] = locator
    replay = production_replay(trace, converter)
    return {
        "trace": trace,
        "registered_path": registered_path,
        "locator": locator,
        "gold_record": replay["final_record"],
        "stage_checks": replay["stage_checks"],
    }


def inject_faults(
    oracle_trace: dict[str, Any],
    target: dict[str, Any],
    components: tuple[str, ...],
    case_index: int,
) -> dict[str, Any]:
    trace = copy.deepcopy(oracle_trace)
    locator = target["locator"]
    for component in components:
        if component == "schema":
            trace["schema"]["field_map"][target["registered_path"]] = (
                f"fault_concept_{case_index:03d}"
            )
        elif component == "evidence":
            trace["evidence"]["item_overrides"][locator] = {
                "evidence_text": "",
                "source_hint": "",
            }
        elif component == "extraction":
            value = target["gold_record"]["value"]
            trace["extraction"]["item_overrides"][locator] = {
                "value": value + 137 if isinstance(value, (int, float)) else f"fault::{value}"
            }
        elif component == "binding":
            trace["binding"]["record_key_overrides"][locator] = (
                f"fault-record-{case_index:03d}"
            )
        elif component == "normalization":
            trace["normalization"]["fact_overrides"][locator] = {
                "unit": "noncanonical-fault-unit"
            }
        else:
            raise ValueError(f"unknown component: {component}")
    return trace


def evaluate_fault_case(
    *,
    case_id: str,
    paper_id: str,
    oracle_trace: dict[str, Any],
    target: dict[str, Any],
    faulty_components: tuple[str, ...],
    converter: ModuleType,
    repeats: int,
) -> dict[str, Any]:
    faulty = inject_faults(oracle_trace, target, faulty_components, int(case_id.rsplit("-", 1)[-1]))
    baseline = production_replay(faulty, converter)["final_record"]
    gold = target["gold_record"]

    def evaluate_subset(subset: tuple[str, ...]) -> bool:
        patched = copy.deepcopy(faulty)
        for component in subset:
            patched[component] = copy.deepcopy(oracle_trace[component])
        return strict_record_match(
            gold, production_replay(patched, converter)["final_record"]
        )

    search = hierarchical_minimal_repair_search(
        evaluate_subset, max_order=len(COMPONENTS), repeats=repeats
    )
    expected = sorted(faulty_components)
    predicted = sorted(search["minimal_sufficient_sets"][0]) if search["status"] == "identified" else []
    exact = search["status"] == "identified" and predicted == expected
    return {
        "case_id": case_id,
        "paper_id": paper_id,
        "case_schema_version": CASE_SCHEMA,
        "target_locator": target["locator"],
        "target_concept_id": gold["concept_id"],
        "fault_cardinality": len(faulty_components),
        "expected_minimal_repair_set": expected,
        "baseline_mismatch_fields": strict_record_mismatches(gold, baseline),
        "search": search,
        "predicted_minimal_repair_set": predicted,
        "exact_minimal_repair": exact,
        "oracle_trace_sha256": content_sha256(oracle_trace),
        "faulty_trace_sha256": content_sha256(faulty),
    }


def _manifest_artifacts(output_root: Path) -> list[dict[str, Any]]:
    return [
        _artifact(path, output_root)
        for path in sorted(output_root.glob("CARE_STEP9_REPLAY_*"))
        if path.is_file()
        and path.name != "CARE_STEP9_REPLAY_MANIFEST.json"
        and not path.name.endswith(".run.json")
    ]


def run_real_step9_replay(
    *,
    run_root: Path,
    repo_root: Path,
    output_root: Path,
    source_step9_run_identity: dict[str, Any],
    run_identity: dict[str, Any],
    paper_ids: tuple[str, ...] = DEFAULT_PAPERS,
    repeats: int = 2,
    source_stage_root: Path | None = None,
    schema_source: Path | None = None,
) -> dict[str, Any]:
    run_artifact_guard.assert_active_run(run_identity)
    allowed_existing = {Path(run_identity["manifest_path"]).resolve(strict=False)}
    unexpected_existing = [
        path
        for path in output_root.iterdir()
        if path.resolve(strict=False) not in allowed_existing
    ]
    if unexpected_existing:
        raise run_artifact_guard.CleanRunRequiredError(
            "Fresh CARE replay output directory contains undeclared artifacts: "
            + ", ".join(str(path) for path in unexpected_existing)
        )
    snapshot_manifest = build_step9_snapshots(
        run_root=run_root,
        repo_root=repo_root,
        snapshot_root=output_root,
        source_step9_run_identity=source_step9_run_identity,
        run_identity=run_identity,
        paper_ids=paper_ids,
        source_stage_root=source_stage_root,
        schema_source=schema_source,
    )
    verify_snapshot_manifest(output_root, snapshot_manifest)
    converter_path = run_root / "optimization" / "convert_step9_output_to_predictions.py"
    converter = load_module(converter_path, "care_step9_prediction_converter")
    current_code_hashes = {
        name: file_sha256(item["path"])
        for name, item in snapshot_manifest["production_code"].items()
    }
    expected_code_hashes = {
        name: item["sha256"]
        for name, item in snapshot_manifest["production_code"].items()
    }
    if current_code_hashes != expected_code_hashes:
        raise RuntimeError("production code changed after Step9 snapshot creation")

    results = []
    paper_qc = []
    case_number = 0
    for paper_id in paper_ids:
        oracle = load_snapshot_trace(output_root, paper_id)
        target = select_target(oracle, converter)
        oracle = target["trace"]
        oracle["final_record"] = target["gold_record"]
        checks = target["stage_checks"]
        paper_qc.append(
            {
                "paper_id": paper_id,
                "target_locator": target["locator"],
                "target_concept_id": target["gold_record"]["concept_id"],
                "stage_count": len(checks),
                "all_stage_json_valid": all(item["has_json"] for item in checks.values()),
            }
        )
        fault_sets = [
            *(tuple([component]) for component in COMPONENTS),
            ("schema", "evidence"),
            ("extraction", "binding", "normalization"),
        ]
        for fault_set in fault_sets:
            case_number += 1
            results.append(
                evaluate_fault_case(
                    case_id=f"step9-real-{case_number:03d}",
                    paper_id=paper_id,
                    oracle_trace=oracle,
                    target=target,
                    faulty_components=fault_set,
                    converter=converter,
                    repeats=repeats,
                )
            )

    failures = [item for item in results if not item["exact_minimal_repair"]]
    status = "pass" if not failures and all(item["all_stage_json_valid"] for item in paper_qc) else "fail"
    coverage = {
        "status": status,
        "scope_label": "superconductivity_only",
        "run_identity": run_identity,
        "source_step9_run_identity": source_step9_run_identity,
        "paper_count": len(paper_ids),
        "paper_ids": list(paper_ids),
        "frozen_papers_only": True,
        "stage_count_per_paper": len(STAGES),
        "snapshotted_stage_artifact_count": len(paper_ids) * len(STAGES),
        "single_fault_case_count": len(paper_ids) * len(COMPONENTS),
        "pair_fault_case_count": len(paper_ids),
        "triple_fault_case_count": len(paper_ids),
        "total_case_count": len(results),
        "components": list(COMPONENTS),
    }
    result_report = {
        "status": status,
        "scope_label": "superconductivity_only",
        "run_identity": run_identity,
        "source_step9_run_identity": source_step9_run_identity,
        "replay_scope": snapshot_manifest["replay_scope"],
        "online_llm_regeneration": False,
        "production_functions": [
            "downstream_extraction_runner.sanitize_payload",
            "downstream_extraction_runner.validate_stage_payload",
            "convert_step9_output_to_predictions.locate_evidence_line",
            "convert_step9_output_to_predictions.iter_values",
            "convert_step9_output_to_predictions.canonicalize_record_key",
            "convert_step9_output_to_predictions.canonicalize_measurement",
            "convert_step9_output_to_predictions.infer_qualifiers",
        ],
        "exact_minimal_repair_count": sum(item["exact_minimal_repair"] for item in results),
        "case_count": len(results),
        "exact_minimal_repair_accuracy": (
            sum(item["exact_minimal_repair"] for item in results) / len(results)
        ),
        "results": results,
    }
    qc = {
        "status": status,
        "scope_label": "superconductivity_only",
        "run_identity": run_identity,
        "source_step9_run_identity": source_step9_run_identity,
        "snapshot_hash_verification": "pass",
        "production_code_hash_verification": "pass",
        "repeat_count_per_probe": repeats,
        "all_repair_probes_deterministic": all(
            all(probe["deterministic"] for probe in item["search"]["probes"])
            for item in results
        ),
        "paper_qc": paper_qc,
        "limitations": [
            "Replays use persisted LLM stage outputs and rerun deterministic production postprocessing nodes.",
            "This run does not regenerate the nine LLM extraction stages through the remote API.",
            "All selected papers are superconductivity papers from the frozen in-domain split.",
        ],
    }
    failure_report = {
        "status": "pass" if not failures else "fail",
        "scope_label": "superconductivity_only",
        "run_identity": run_identity,
        "source_step9_run_identity": source_step9_run_identity,
        "failure_count": len(failures),
        "failures": failures,
    }
    write_json(
        output_root / "CARE_STEP9_REPLAY_COVERAGE_REPORT.json",
        coverage,
        run_identity=run_identity,
    )
    write_json(
        output_root / "CARE_STEP9_REPLAY_RESULT_REPORT.json",
        result_report,
        run_identity=run_identity,
    )
    write_json(
        output_root / "CARE_STEP9_REPLAY_QC_REPORT.json",
        qc,
        run_identity=run_identity,
    )
    write_json(
        output_root / "CARE_STEP9_REPLAY_FAILURE_REPORT.json",
        failure_report,
        run_identity=run_identity,
    )
    markdown = (
        "# CARE-IE Real Step9 Replay\n\n"
        f"- Status: {status}\n"
        f"- Frozen papers: {len(paper_ids)}\n"
        f"- Persisted production stage snapshots: {len(paper_ids) * len(STAGES)}\n"
        f"- Counterfactual cases: {len(results)}\n"
        f"- Exact minimal repair sets: {result_report['exact_minimal_repair_count']}/{len(results)}\n"
        f"- Deterministic repeats per probe: {repeats}\n"
        "- Scope: persisted Step9 outputs plus production postprocessing node replay.\n"
        "- Online LLM stage regeneration: not performed in this run.\n"
    )
    run_artifact_guard.atomic_write_text(
        output_root / "CARE_STEP9_REPLAY_REPORT.md",
        markdown,
        run_identity=run_identity,
    )
    manifest = {
        "status": status,
        "scope_label": "superconductivity_only",
        "run_identity": run_identity,
        "source_step9_run_identity": source_step9_run_identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_manifest_sha256": file_sha256(str(output_root / "SNAPSHOT_MANIFEST.json")),
        "artifacts": _manifest_artifacts(output_root),
    }
    write_json(
        output_root / "CARE_STEP9_REPLAY_MANIFEST.json",
        manifest,
        run_identity=run_identity,
    )
    return {
        "status": status,
        "output_root": str(output_root),
        "paper_count": len(paper_ids),
        "case_count": len(results),
        "failure_count": len(failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--paper-ids", nargs="*", default=list(DEFAULT_PAPERS))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--source-stage-root", default="")
    parser.add_argument("--schema-source", default="")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else run_root
        / "optimization"
        / "counterfactual_eval_v1"
        / "real_step9_replay_v1"
    )
    repo_root = Path(args.repo_root).resolve()
    paper_ids = tuple(args.paper_ids)
    source_stage_root = (
        Path(args.source_stage_root).resolve()
        if args.source_stage_root
        else run_root / "optimization" / "step9_full_corpus_extraction_runs"
    )
    schema_source = (
        Path(args.schema_source).resolve()
        if args.schema_source
        else run_root / "optimization" / "step8_superconductor_28paper_verified.json"
    )
    source_step9_run_identity = verify_source_step9_run(
        run_root, paper_ids, source_stage_root
    )
    input_identity = build_replay_input_identity(
        run_root=run_root,
        repo_root=repo_root,
        paper_ids=paper_ids,
        repeats=args.repeats,
        source_step9_run_identity=source_step9_run_identity,
        source_stage_root=source_stage_root,
        schema_source=schema_source,
    )
    primary_output = output_root / "CARE_STEP9_REPLAY_RESULT_REPORT.json"
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline="care_step9_replay",
        output_path=primary_output,
        input_identity=input_identity,
        artifact_paths=[output_root],
    )
    try:
        result = run_real_step9_replay(
            run_root=run_root,
            repo_root=repo_root,
            output_root=output_root,
            source_step9_run_identity=source_step9_run_identity,
            run_identity=run_identity,
            paper_ids=paper_ids,
            repeats=args.repeats,
            source_stage_root=source_stage_root,
            schema_source=schema_source,
        )
    except BaseException as exc:
        run_artifact_guard.update_run_status(
            run_identity,
            "interrupted",
            error_type=type(exc).__name__,
        )
        raise
    run_artifact_guard.update_run_status(
        run_identity,
        "completed" if result["status"] == "pass" else "needs_review",
        result_status=result["status"],
        scope_label="superconductivity_only",
        source_step9_run_id=source_step9_run_identity["run_id"],
    )
    result["run_id"] = run_identity["run_id"]
    result["scope_label"] = "superconductivity_only"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
