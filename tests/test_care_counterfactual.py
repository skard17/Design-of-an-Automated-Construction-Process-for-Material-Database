import copy
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import care_counterfactual as care  # noqa: E402


def example_case():
    return {
        "schema_version": care.CASE_SCHEMA,
        "case_id": "case-1",
        "domain": "superconductivity",
        "split": "dev_pilot",
        "paper_id": "paper-1",
        "baseline_trace": {
            "schema": {"field": "Tc"},
            "evidence": {"lines": ["Tc is 9 K"]},
            "extraction": {"value": 8, "unit": "K"},
            "binding": {"sample_id": "sample-a"},
            "normalization": {"value": 8, "unit": "K"},
            "final_record": {
                "paper_id": "paper-1",
                "concept_id": "transition_temperature",
                "record_key": "sample-a",
                "value": 8,
                "unit": "K",
                "qualifiers": {},
                "evidence_line": 1,
                "binding": {"sample_id": "sample-a"},
            },
        },
    }


def normalization_intervention():
    return {
        "schema_version": care.INTERVENTION_SCHEMA,
        "intervention_id": "case-1-normalization",
        "case_id": "case-1",
        "component": "normalization",
        "replacements": {"/trace/normalization/value": 9},
        "allowed_paths": ["/trace/normalization/value"],
    }


class CareCounterfactualTests(unittest.TestCase):
    def test_applies_only_whitelisted_component_change(self):
        patched, diff = care.apply_intervention(
            example_case(), normalization_intervention()
        )

        self.assertEqual(9, patched["normalization"]["value"])
        self.assertEqual(["/trace/normalization/value"], diff)

    def test_rejects_cross_component_replacement(self):
        intervention = normalization_intervention()
        intervention["replacements"] = {"/trace/evidence/lines": ["oracle"]}
        intervention["allowed_paths"] = ["/trace/evidence/lines"]

        with self.assertRaises(care.CounterfactualContractError):
            care.apply_intervention(example_case(), intervention)

    def test_rejects_undeclared_nested_change(self):
        intervention = normalization_intervention()
        intervention["allowed_paths"] = ["/trace/normalization/unit"]

        with self.assertRaises(care.CounterfactualContractError):
            care.apply_intervention(example_case(), intervention)

    def test_replay_records_hashes_and_changed_paths(self):
        def adapter(component, trace):
            self.assertEqual("normalization", component)
            record = copy.deepcopy(trace["final_record"])
            record["value"] = trace["normalization"]["value"]
            return record

        result = care.replay_intervention(
            example_case(), normalization_intervention(), adapter
        )

        self.assertEqual(9, result["final_record"]["value"])
        self.assertNotEqual(
            result["baseline_trace_sha256"], result["patched_trace_sha256"]
        )

    def test_strict_record_requires_binding_and_evidence(self):
        gold = copy.deepcopy(example_case()["baseline_trace"]["final_record"])
        prediction = copy.deepcopy(gold)
        prediction["binding"] = {"sample_id": "sample-b"}

        self.assertFalse(care.strict_record_match(gold, prediction))
        self.assertEqual(["binding"], care.strict_record_mismatches(gold, prediction))

    def test_evaluate_replay_sets_binary_outcome(self):
        gold = copy.deepcopy(example_case()["baseline_trace"]["final_record"])
        result = {"final_record": copy.deepcopy(gold), "component": "normalization"}

        evaluated = care.evaluate_replay(gold, result)

        self.assertEqual(1, evaluated["outcome"])
        self.assertEqual([], evaluated["mismatch_fields"])

    def test_responsibility_effects_rank_successful_intervention(self):
        report = care.responsibility_effects(
            [
                {"component": "schema", "outcome": 0},
                {"component": "normalization", "outcome": 1},
                {"component": "normalization", "outcome": 1},
            ],
            baseline_outcome=0,
        )

        self.assertEqual("normalization", report["ranked_components"][0])
        self.assertEqual(1.0, report["effects"][0]["responsibility_effect"])

    def test_pairwise_interaction_uses_difference_in_differences(self):
        self.assertEqual(1.0, care.pairwise_interaction_effect(0, 0, 0, 1))

    def test_forbidden_key_scan_finds_evaluator_leak(self):
        hits = care.forbidden_keys(
            {"cases": [{"gold_record": {"value": 9}}]}, {"gold_record"}
        )

        self.assertEqual(["/cases/0/gold_record"], hits)


if __name__ == "__main__":
    unittest.main()
