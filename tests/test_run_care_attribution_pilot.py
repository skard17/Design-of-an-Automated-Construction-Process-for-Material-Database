import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import run_care_attribution_pilot as runner  # noqa: E402
import tempfile


class RunCareAttributionPilotTests(unittest.TestCase):
    def test_parser_supports_comment_source_line_blocks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pack.md"
            path.write_text(
                "<!-- SOURCE_LINE: 7 -->\nTc is 9 K.\n<!-- SOURCE_LINE: 8 -->\n\n",
                encoding="utf-8",
            )

            lines = runner.parse_source_lines(path)

        self.assertEqual("Tc is 9 K.", lines[7])
        self.assertEqual("", lines[8])

    def test_excerpt_includes_baseline_neighborhood_and_ranked_terms(self):
        case = {
            "case_id": "c1",
            "target_record": {
                "record_key": "Sample A",
                "concept_id": "transition_temperature",
            },
            "baseline_record": {"evidence_line": 10},
        }
        lines = {number: f"line {number}" for number in range(1, 20)}
        lines[2] = "Sample A has a superconducting transition at 9 K."

        excerpt = runner.evidence_excerpt(case, lines, max_lines=10)
        numbers = {item["source_line"] for item in excerpt}

        self.assertIn(2, numbers)
        self.assertIn(10, numbers)

    def test_excerpt_preserves_long_recorded_evidence_line(self):
        case = {
            "case_id": "c1",
            "target_record": {
                "record_key": "Sample A",
                "concept_id": "upper_critical_field",
            },
            "baseline_record": {"evidence_line": 10},
        }
        lines = {10: "x" * 1100 + " Hc2(0) is 6.95 T."}

        excerpt = runner.evidence_excerpt(case, lines)

        self.assertIn("6.95", excerpt[0]["text"])

    def test_counterfactual_result_requires_all_components(self):
        payload = {
            "primary_component": "extraction",
            "confidence": 0.9,
            "assessments": {"extraction": {"predicted_flip": True}},
        }

        with self.assertRaises(ValueError):
            runner.validate_model_result(payload, "c1", "counterfactual")

    def test_supervisor_result_clamps_confidence(self):
        result = runner.validate_model_result(
            {"primary_component": "normalization", "confidence": 2},
            "c1",
            "supervisor",
        )

        self.assertEqual(1.0, result["confidence"])

    def test_evidence_mismatch_prevents_normalization_only_attribution(self):
        result = {
            "case_id": "c1",
            "primary_component": "normalization",
            "confidence": 0.8,
        }

        guarded = runner.apply_answer_free_guards(
            result, ["value", "evidence_line"]
        )

        self.assertEqual("extraction", guarded["primary_component"])
        self.assertEqual("normalization", guarded["model_primary_component"])
        self.assertEqual(
            "normalization_cannot_repair_evidence_provenance",
            guarded["protocol_guard"]["rule"],
        )

    def test_normalization_survives_without_evidence_mismatch(self):
        result = {
            "case_id": "c1",
            "primary_component": "normalization",
            "confidence": 0.8,
        }

        guarded = runner.apply_answer_free_guards(result, ["value", "unit"])

        self.assertEqual("normalization", guarded["primary_component"])
        self.assertNotIn("protocol_guard", guarded)

    def test_prompt_contains_answer_free_protocol_constraints(self):
        case = {
            "case_id": "c1",
            "target_record": {"record_key": "sample", "concept_id": "upper_critical_field"},
            "baseline_record": {"value": "6.95", "evidence_line": 8},
        }

        messages = runner.build_prompt(case, ["value"], [], "supervisor")

        self.assertIn("target_contract_complete", messages[1]["content"])
        self.assertIn("do not re-pair values", messages[1]["content"])
        self.assertIn("distinct quantities", messages[1]["content"])

    def test_direct_recorded_evidence_rules_out_evidence_root_cause(self):
        case = {
            "baseline_record": {"value": "6.95", "evidence_line": 81},
        }
        excerpt = [
            {"source_line": 81, "text": "Hc2(0) is 6.95 T at 19.7 GPa."}
        ]
        result = {
            "case_id": "c1",
            "primary_component": "evidence",
            "confidence": 0.6,
        }

        guarded = runner.apply_answer_free_guards(
            result, ["value", "qualifiers"], case, excerpt
        )

        self.assertEqual("normalization", guarded["primary_component"])
        self.assertEqual("evidence", guarded["model_primary_component"])
        self.assertEqual(
            "recorded_evidence_directly_supports_baseline_surface_value",
            guarded["protocol_guard"]["rule"],
        )

    def test_direct_recorded_evidence_guard_does_not_hide_provenance_mismatch(self):
        case = {
            "baseline_record": {"value": "6.95", "evidence_line": 81},
        }
        excerpt = [
            {"source_line": 81, "text": "Hc2(0) is 6.95 T at 19.7 GPa."}
        ]
        result = {
            "case_id": "c1",
            "primary_component": "evidence",
            "confidence": 0.6,
        }

        guarded = runner.apply_answer_free_guards(
            result, ["value", "evidence_line"], case, excerpt
        )

        self.assertEqual("evidence", guarded["primary_component"])
        self.assertNotIn("protocol_guard", guarded)


if __name__ == "__main__":
    unittest.main()
