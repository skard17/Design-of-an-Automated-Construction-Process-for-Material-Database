import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import run_normalization_completeness_audit as audit  # noqa: E402


class NormalizationCompletenessAuditTests(unittest.TestCase):
    def test_accepts_complete_extraction_decision(self):
        result = audit.validate_audit(
            {
                "decision": "extraction",
                "confidence": 0.8,
                "checks": {
                    "complete_value": False,
                    "complete_qualifiers": True,
                    "surface_equivalent": False,
                },
            },
            "c1",
        )

        self.assertEqual("extraction", result["decision"])

    def test_rejects_missing_completeness_checks(self):
        with self.assertRaises(ValueError):
            audit.validate_audit(
                {"decision": "normalization", "checks": {}}, "c1"
            )

    def test_prompt_contains_approximation_contract_policy(self):
        case = {
            "case_id": "c1",
            "target_record": {"record_key": "s", "concept_id": "coherence_length"},
            "baseline_record": {"value": "40", "unit": "nm", "qualifiers": {}},
        }
        messages = audit.build_messages(
            case,
            ["value"],
            [{"source_line": 1, "text": "xi is approximately 40 nm"}],
            {"primary_component": "normalization", "confidence": 0.8, "rationale": "x"},
        )

        self.assertIn("normalization_only_unless_explicitly_requested", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
