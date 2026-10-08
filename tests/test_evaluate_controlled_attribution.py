import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import evaluate_controlled_attribution as evaluator  # noqa: E402


class EvaluateControlledAttributionTests(unittest.TestCase):
    def test_frozen_report_requires_complete_prediction_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluator").mkdir()
            (root / "evaluator" / "controlled_gold.json").write_text(
                json.dumps(
                    {
                        "schema_version": "care-ie.frozen_real_controlled_evaluator_set.v1",
                        "records": [
                            {"case_id": "a", "responsible_component": "schema"},
                            {"case_id": "b", "responsible_component": "evidence"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            predictions = root / "predictions.json"
            predictions.write_text(
                json.dumps(
                    {
                        "model": "test-model",
                        "results": [
                            {"case_id": "a", "primary_component": "schema", "confidence": 0.8}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "report.json"

            report = evaluator.evaluate(root, predictions, output)

            self.assertEqual("incomplete", report["status"])
            self.assertEqual(0.5, report["coverage"])
            self.assertEqual(["b"], report["missing_predictions"])
            self.assertIn("Frozen in-domain", report["scope"])
            self.assertAlmostEqual(0.04, report["calibration"]["brier_score"])


if __name__ == "__main__":
    unittest.main()
