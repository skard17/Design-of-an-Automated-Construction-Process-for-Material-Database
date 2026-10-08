import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import evaluate_care_attribution as evaluator  # noqa: E402


class EvaluateCareAttributionTests(unittest.TestCase):
    def test_binary_class_metrics(self):
        metrics = evaluator.precision_recall_f1(
            ["extraction", "extraction", "normalization"],
            ["extraction", "normalization", "normalization"],
            "extraction",
        )

        self.assertEqual(2, metrics["support"])
        self.assertEqual(1.0, metrics["precision"])
        self.assertEqual(0.5, metrics["recall"])


if __name__ == "__main__":
    unittest.main()
