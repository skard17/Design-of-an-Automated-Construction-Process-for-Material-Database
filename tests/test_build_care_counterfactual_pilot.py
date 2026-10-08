import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import build_care_counterfactual_pilot as builder  # noqa: E402


class BuildCareCounterfactualPilotTests(unittest.TestCase):
    def test_build_candidates_uses_precanonicalization_mismatches(self):
        gold = [
            {
                "fact_id": "f1",
                "paper_id": "p1",
                "record_key": "sample-a",
                "concept_id": "transition_temperature",
                "value": 9,
                "unit": "K",
                "qualifiers": {},
                "evidence_line": 2,
            }
        ]
        predictions = [
            {
                "target_id": "f1",
                "paper_id": "p1",
                "record_key": "sample-a",
                "concept_id": "transition_temperature",
                "value": 8,
                "unit": "K",
                "qualifiers": {},
                "evidence_line": 2,
            }
        ]

        candidates = builder.build_candidates(gold, predictions)

        self.assertEqual(1, len(candidates))
        self.assertEqual(["value"], candidates[0]["mismatch_fields"])

    def test_diverse_selection_balances_concepts_before_repeats(self):
        candidates = [
            {
                "case_id": case_id,
                "paper_id": paper,
                "concept_id": concept,
            }
            for case_id, paper, concept in (
                ("a1", "p1", "a"),
                ("a2", "p1", "a"),
                ("b1", "p1", "b"),
                ("c1", "p2", "c"),
            )
        ]

        selected = builder.select_diverse(candidates, 3)

        self.assertEqual({"a", "b", "c"}, {item["concept_id"] for item in selected})

    def test_annotation_template_has_two_reviewers_and_adjudication(self):
        row = builder.annotation_row(
            {
                "case_id": "f1",
                "paper_id": "p1",
                "record_key": "sample-a",
                "concept_id": "transition_temperature",
            }
        )

        self.assertIn("reviewer_a", row)
        self.assertIn("reviewer_b", row)
        self.assertIn("adjudication", row)

    def test_public_candidates_do_not_claim_executable_case_contract(self):
        source = (REPO_ROOT / "code" / "build_care_counterfactual_pilot.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('"schema_version": "care-ie.pilot_candidate.v1"', source)


if __name__ == "__main__":
    unittest.main()
