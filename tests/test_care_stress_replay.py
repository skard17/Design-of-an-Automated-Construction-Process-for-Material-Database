import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import build_care_stress_replay as stress  # noqa: E402
import run_care_stress_attribution as runner  # noqa: E402
from care_executable_replay import oracle_trace  # noqa: E402


class CareStressReplayTests(unittest.TestCase):
    def test_two_faults_require_exact_joint_repair(self):
        oracle = oracle_trace(0)
        baseline = stress.inject_faults(oracle, ("schema", "binding"), 0)
        bundle = {
            "case": stress.make_case(
                case_id="stress-001",
                oracle=oracle,
                baseline=baseline,
                provenance={"stress_kind": "multi_fault", "observable_variant": "identity"},
            ),
            "oracle": oracle,
            "gold_record": oracle["final_record"],
            "responsible_components": ["schema", "binding"],
        }

        result = stress.evaluate_multi_bundle(bundle, repeats=2)

        self.assertFalse(result["baseline_matches"])
        self.assertEqual([], result["single_component_successes"])
        self.assertTrue(result["expected_pair_succeeds"])
        self.assertEqual([], result["unexpected_pair_successes"])

    def test_stress_validator_allows_no_fault_and_multi_fault(self):
        scores = {
            name: {"responsibility_probability": 0.0, "rationale": "x"}
            for name in ("schema", "evidence", "extraction", "binding", "normalization")
        }
        no_fault = runner.validate_result(
            {
                "fault_status": "no_fault",
                "primary_component": None,
                "responsible_components": [],
                "minimal_sufficient_set": [],
                "confidence": 0.8,
                "component_scores": scores,
            },
            "stress-001",
        )
        multi = runner.validate_result(
            {
                "fault_status": "multi_fault",
                "primary_component": "schema",
                "responsible_components": ["binding", "schema"],
                "minimal_sufficient_set": ["schema", "binding"],
                "confidence": 0.8,
                "component_scores": scores,
            },
            "stress-002",
        )

        self.assertEqual([], no_fault["responsible_components"])
        self.assertEqual(["schema", "binding"], multi["responsible_components"])

    def test_probe_verdict_uses_unique_minimal_sufficient_set(self):
        verdict = runner.derive_probe_verdict(
            {
                "counterfactual_replay": {
                    "baseline_contract_match": False,
                    "probes": [
                        {"repair_components": ["schema"], "contract_match": False, "deterministic": True},
                        {"repair_components": ["binding"], "contract_match": False, "deterministic": True},
                        {
                            "repair_components": ["schema", "binding"],
                            "contract_match": True,
                            "deterministic": True,
                        },
                    ],
                }
            }
        )

        self.assertEqual("identified", verdict["status"])
        self.assertEqual(["schema", "binding"], verdict["responsible_components"])

    def test_probe_verdict_reports_no_fault_when_baseline_matches(self):
        verdict = runner.derive_probe_verdict(
            {"counterfactual_replay": {"baseline_contract_match": True, "probes": []}}
        )

        self.assertEqual("no_fault", verdict["fault_status"])
        self.assertEqual([], verdict["responsible_components"])


if __name__ == "__main__":
    unittest.main()
