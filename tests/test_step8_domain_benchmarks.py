import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import step8_benchmark_evaluator as evaluator  # noqa: E402


class Step8DomainBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarks = json.loads(
            (REPO_ROOT / "tests" / "fixtures" / "step8_domain_benchmarks.json").read_text(encoding="utf-8")
        )

    def test_reports_missing_and_forbidden_fields(self):
        result = {
            "schema_definition": {
                "top_level_keys": [{"key": "material_info"}],
                "field_registry": [
                    {"field_path": "material_info.section1.Tc"},
                    {"field_path": "material_info.section1.coercive_field"},
                ],
            }
        }

        report = evaluator.evaluate_result(result, self.benchmarks["superconductor"])

        self.assertIn("Jc", report["missing_concepts"])
        self.assertIn("material_info.section1.coercive_field", report["forbidden_fields"])
        self.assertLess(report["required_concept_recall"], 1.0)

    def test_alias_matches_nested_field_paths(self):
        result = {
            "schema_definition": {
                "top_level_keys": [{"key": "material_info"}],
                "field_registry": [
                    {"field_path": "material_info.section1.lambda.value"},
                ],
            }
        }
        benchmark = {
            "required_concepts": {"penetration_depth": ["material_info.section1.lambda"]},
            "forbidden_patterns": [],
        }

        report = evaluator.evaluate_result(result, benchmark)

        self.assertEqual(1.0, report["required_concept_recall"])
        self.assertEqual([], report["missing_concepts"])

    def test_unwraps_langgraph_state_module_outputs(self):
        payload = {
            "result": None,
            "module_outputs": {
                "schema_design_module": {
                    "top_level_keys": [{"key": "material_info"}],
                    "field_registry": [{"field_path": "material_info.section4.M_H"}],
                }
            },
        }
        benchmark = {
            "required_concepts": {"M_H_curve": ["material_info.section4.M_H"]},
            "forbidden_patterns": [],
        }

        report = evaluator.evaluate_result(payload, benchmark)

        self.assertEqual(1.0, report["required_concept_recall"])


if __name__ == "__main__":
    unittest.main()
