import copy
import json
import sys
import unittest
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import care_step9_replay as replay  # noqa: E402
import run_artifact_guard  # noqa: E402


def _build_identity_bound_step9_run(tmp_path, paper_id="paper-1"):
    run_root = tmp_path / "benchmark"
    stage_root = run_root / "optimization" / "step9_full_corpus_extraction_runs"
    output_path = run_root / "optimization" / "step9-result.json"
    identity = run_artifact_guard.build_input_identity(
        "step9_extraction_build",
        {"test": "fresh CARE source"},
        [],
    )
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline="step9_extraction_build",
        output_path=output_path,
        input_identity=identity,
        artifact_paths=[output_path, stage_root],
    )
    for stage in replay.STAGES:
        path = stage_root / stage / f"{paper_id}.json"
        run_artifact_guard.atomic_write_json(
            path,
            {"stage_id": stage, "run_identity": run_identity},
            run_identity=run_identity,
        )
    run_artifact_guard.update_run_status(run_identity, "completed")
    return run_root, run_identity


def test_care_accepts_only_one_completed_identity_bound_step9_run(tmp_path):
    run_root, run_identity = _build_identity_bound_step9_run(tmp_path)

    verified = replay.verify_source_step9_run(run_root, ("paper-1",))

    assert verified["run_id"] == run_identity["run_id"]


def test_care_accepts_explicit_attempt_directory_owned_by_declared_parent(tmp_path):
    run_root = tmp_path / "benchmark"
    declared_root = run_root / "step9_runs" / "r2"
    stage_root = declared_root / "attempt_008"
    output_path = run_root / "step9-result.json"
    identity = run_artifact_guard.build_input_identity(
        "step9_extraction_build",
        {"test": "fresh nested CARE source"},
        [],
    )
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline="step9_extraction_build",
        output_path=output_path,
        input_identity=identity,
        artifact_paths=[output_path, declared_root],
    )
    for stage in replay.STAGES:
        run_artifact_guard.atomic_write_json(
            stage_root / stage / "paper-1.json",
            {"stage_id": stage, "run_identity": run_identity},
            run_identity=run_identity,
        )
    run_artifact_guard.update_run_status(run_identity, "completed")

    verified = replay.verify_source_step9_run(
        run_root,
        ("paper-1",),
        stage_root,
    )

    assert verified["run_id"] == run_identity["run_id"]


def test_care_rejects_stage_artifacts_without_a_run_identity(tmp_path):
    run_root = tmp_path / "benchmark"
    stage_root = run_root / "optimization" / "step9_full_corpus_extraction_runs"
    for stage in replay.STAGES:
        path = stage_root / stage / "paper-1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stage_id": stage}), encoding="utf-8")

    with pytest.raises(run_artifact_guard.RunIdentityError, match="identity-bearing"):
        replay.verify_source_step9_run(run_root, ("paper-1",))


def test_care_refuses_an_existing_snapshot_manifest(tmp_path):
    snapshot_root = tmp_path / "care-output"
    snapshot_root.mkdir()
    (snapshot_root / "SNAPSHOT_MANIFEST.json").write_text("{}", encoding="utf-8")

    with pytest.raises(run_artifact_guard.CleanRunRequiredError, match="refused to reuse"):
        replay.build_step9_snapshots(
            run_root=tmp_path / "benchmark",
            repo_root=REPO_ROOT,
            snapshot_root=snapshot_root,
            source_step9_run_identity={"run_id": "source"},
            run_identity={"run_id": "care"},
            paper_ids=("paper-1",),
        )


class FakeConverter:
    @staticmethod
    def resolve_paper_id(document):
        return Path(document).stem

    @staticmethod
    def iter_values(value):
        yield value, {}

    @staticmethod
    def locate_evidence_line(lines, evidence, _source_hint):
        return lines.index(evidence) + 1 if evidence in lines else None

    @staticmethod
    def canonicalize_record_key(value):
        return str(value or "").strip(), {}

    @staticmethod
    def canonicalize_measurement(value, unit):
        return value, unit, {}

    @staticmethod
    def infer_qualifiers(_concept, _value, _evidence, qualifiers):
        return qualifiers

    @staticmethod
    def normalize(value):
        return str(value or "").strip().casefold()


def example_trace():
    stage_payload = {
        "stage_id": "material_info.section1",
        "section_id": "material_info.section1",
        "document": "paper-1.md",
        "extracted_fields": [
            {
                "field_path": "material_info.section1.transition_temperature",
                "value": 9.3,
                "unit": "K",
                "material_system": "Sample-A",
                "evidence_text": "Sample-A has Tc 9.3 K.",
                "source_hint": "SOURCE_LINE: 1",
                "confidence": 1.0,
            }
        ],
        "missing_fields": [],
    }
    return {
        "schema": {
            "field_map": {
                "material_info.section1.transition_temperature": "transition_temperature"
            }
        },
        "evidence": {
            "source_lines": ["Sample-A has Tc 9.3 K."],
            "item_overrides": {},
        },
        "extraction": {
            "stage_payloads": {"material_info.section1": stage_payload},
            "item_overrides": {},
        },
        "binding": {"record_key_overrides": {}},
        "normalization": {"fact_overrides": {}},
        "target_locator": "material_info.section1:0:0",
        "final_record": {},
    }


class CareStep9ReplayTests(unittest.TestCase):
    def test_production_replay_builds_strict_record(self):
        result = replay.production_replay(example_trace(), FakeConverter)

        self.assertEqual("transition_temperature", result["final_record"]["concept_id"])
        self.assertEqual("Sample-A", result["final_record"]["record_key"])
        self.assertEqual(9.3, result["final_record"]["value"])
        self.assertTrue(result["stage_checks"]["material_info.section1"]["has_json"])

    def test_each_component_fault_changes_the_target_record(self):
        oracle = example_trace()
        target = {
            "locator": oracle["target_locator"],
            "registered_path": "material_info.section1.transition_temperature",
            "gold_record": replay.production_replay(oracle, FakeConverter)["final_record"],
        }
        for index, component in enumerate(replay.COMPONENTS, start=1):
            with self.subTest(component=component):
                faulty = replay.inject_faults(oracle, target, (component,), index)
                record = replay.production_replay(faulty, FakeConverter)["final_record"]
                self.assertNotEqual(target["gold_record"], record)

    def test_two_fault_case_requires_both_repairs(self):
        oracle = example_trace()
        target = {
            "locator": oracle["target_locator"],
            "registered_path": "material_info.section1.transition_temperature",
            "gold_record": replay.production_replay(oracle, FakeConverter)["final_record"],
        }
        oracle["final_record"] = copy.deepcopy(target["gold_record"])

        result = replay.evaluate_fault_case(
            case_id="step9-real-001",
            paper_id="paper-1",
            oracle_trace=oracle,
            target=target,
            faulty_components=("schema", "binding"),
            converter=FakeConverter,
            repeats=2,
        )

        self.assertTrue(result["exact_minimal_repair"])
        self.assertEqual(["binding", "schema"], result["predicted_minimal_repair_set"])
        self.assertEqual(2, result["search"]["minimal_cardinality"])


if __name__ == "__main__":
    unittest.main()
