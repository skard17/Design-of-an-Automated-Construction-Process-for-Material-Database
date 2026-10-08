import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import run_controlled_k3_attribution as runner  # noqa: E402
from care_counterfactual import COMPONENTS  # noqa: E402


class ControlledK3AttributionTests(unittest.TestCase):
    def test_validator_requires_all_component_scores(self):
        payload = {
            "primary_component": "schema",
            "confidence": 0.8,
            "component_scores": {"schema": {"responsibility_probability": 1}},
        }

        with self.assertRaises(ValueError):
            runner.validate_result(payload, "c1")

    def test_validator_accepts_complete_result(self):
        payload = {
            "primary_component": "evidence",
            "confidence": 1.4,
            "component_scores": {
                component: {"responsibility_probability": 1 if component == "evidence" else 0, "rationale": "x"}
                for component in COMPONENTS
            },
        }

        result = runner.validate_result(payload, "c1")

        self.assertEqual("evidence", result["primary_component"])
        self.assertEqual(1.0, result["confidence"])


if __name__ == "__main__":
    unittest.main()
