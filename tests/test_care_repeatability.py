import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import evaluate_care_repeatability as evaluator  # noqa: E402


class CareRepeatabilityTests(unittest.TestCase):
    def test_reports_label_agreement_and_confidence_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluator").mkdir()
            (root / "evaluator" / "stress_gold.json").write_text(
                json.dumps({"records": [{"case_id": "c1", "responsible_components": ["schema"]}]}),
                encoding="utf-8",
            )
            paths = []
            for index, confidence in enumerate((0.8, 0.9, 0.85)):
                path = root / f"run-{index}.json"
                result = {
                    "case_id": "c1",
                    "responsible_components": ["schema"],
                    "minimal_sufficient_set": ["schema"],
                    "confidence": confidence,
                }
                path.write_text(
                    json.dumps({"case_count": 1, "results": [result], "model_results": [result]}),
                    encoding="utf-8",
                )
                paths.append(path)

            report = evaluator.evaluate(root, paths, root / "report.json")

            self.assertEqual(1.0, report["model_label_agreement_rate"])
            self.assertEqual(1.0, report["all_model_runs_correct_rate"])
            self.assertAlmostEqual(0.1, report["max_confidence_range"])


if __name__ == "__main__":
    unittest.main()
