import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import care_executable_replay as executable  # noqa: E402
from care_counterfactual import COMPONENTS, strict_record_match  # noqa: E402


class CareExecutableReplayTests(unittest.TestCase):
    def test_oracle_trace_produces_its_final_record(self):
        trace = executable.oracle_trace(0)

        record = executable.execute_trace_adapter("schema", trace)

        self.assertTrue(strict_record_match(trace["final_record"], record))

    def test_external_fixture_can_drive_an_oracle_trace(self):
        fixture = {
            "paper_id": "frozen-paper",
            "candidates": [
                {
                    "paper_id": "frozen-paper",
                    "record_key": "Material-A",
                    "concept_id": "material_info.section1.transition_temperature",
                    "value": 7.2,
                    "unit": "K",
                    "qualifiers": {"source_hint": "Page 2, line 14"},
                    "evidence_line": 14,
                    "evidence": "Material-A has a transition temperature of 7.2 K.",
                },
                {
                    "paper_id": "other-paper",
                    "record_key": "Material-B",
                    "concept_id": "material_info.section1.transition_temperature",
                    "value": 3.1,
                    "unit": "K",
                    "qualifiers": {"source_hint": "Page 3, line 20"},
                    "evidence_line": 20,
                    "evidence": "Material-B has a transition temperature of 3.1 K.",
                },
            ],
        }

        oracle = executable.oracle_trace_from_fixture(fixture)
        bundle = executable.build_controlled_case_from_oracle(
            case_id="frozen-real-001",
            component="evidence",
            oracle=oracle,
            fault_index=0,
            domain="superconductivity",
            split="in_domain_frozen",
        )

        result = executable.run_controlled_case(bundle, repeats=2)
        self.assertEqual(["evidence"], result["successful_components"])
        self.assertTrue(result["deterministic_repeats"])

    def test_every_fault_makes_baseline_incorrect(self):
        oracle = executable.oracle_trace(0)
        for component in COMPONENTS:
            with self.subTest(component=component):
                faulty = executable.inject_single_fault(oracle, component, 0)
                self.assertFalse(
                    strict_record_match(oracle["final_record"], faulty["final_record"])
                )

    def test_only_responsible_component_flips_each_case(self):
        for case_number, component in enumerate(COMPONENTS, start=1):
            with self.subTest(component=component):
                bundle = executable.build_controlled_case(case_number, component, 0)
                result = executable.run_controlled_case(bundle, repeats=2)
                self.assertEqual([component], result["successful_components"])
                self.assertTrue(result["exact_unique_responsibility"])
                self.assertTrue(result["deterministic_repeats"])

    def test_case_id_does_not_reveal_component(self):
        bundle = executable.build_controlled_case(17, "evidence", 3)

        self.assertEqual("controlled-017", bundle["case"]["case_id"])
        self.assertNotIn("evidence", bundle["case"]["case_id"])

    def test_answer_free_observable_hides_oracle_configs_and_label(self):
        bundle = executable.build_controlled_case(17, "evidence", 3)

        observable = executable.answer_free_observable(bundle)
        serialized = str(observable)

        self.assertNotIn("responsible_component", serialized)
        self.assertNotIn("concept_id_override", serialized)
        self.assertNotIn("selected_index", serialized)
        self.assertIn("stage_outputs", observable)

    def test_normalization_fault_appears_only_after_binding_output(self):
        bundle = executable.build_controlled_case(5, "normalization", 0)

        outputs = executable.execute_trace_with_outputs(
            bundle["case"]["baseline_trace"]
        )

        self.assertEqual(9.3, outputs["extraction"]["value"])
        self.assertEqual(9.3, outputs["binding"]["value"])
        self.assertEqual("9.3", outputs["normalization"]["value"])


if __name__ == "__main__":
    unittest.main()
