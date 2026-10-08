import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import care_counterfactual as care  # noqa: E402


class CareMinimalRepairSearchTests(unittest.TestCase):
    def test_identifies_a_unique_single_repair(self):
        report = care.hierarchical_minimal_repair_search(
            lambda subset: "binding" in subset,
            repeats=2,
        )

        self.assertEqual("identified", report["status"])
        self.assertEqual([["binding"]], report["minimal_sufficient_sets"])
        self.assertTrue(all(probe["deterministic"] for probe in report["probes"]))

    def test_reports_ambiguous_minimal_repairs(self):
        report = care.hierarchical_minimal_repair_search(
            lambda subset: "schema" in subset or "evidence" in subset,
        )

        self.assertEqual("ambiguous", report["status"])
        self.assertEqual(
            [["schema"], ["evidence"]], report["minimal_sufficient_sets"]
        )

    def test_searches_through_triples(self):
        required = {"extraction", "binding", "normalization"}
        report = care.hierarchical_minimal_repair_search(
            lambda subset: required.issubset(subset),
            max_order=3,
        )

        self.assertEqual("identified", report["status"])
        self.assertEqual(3, report["minimal_cardinality"])
        self.assertEqual(
            [["extraction", "binding", "normalization"]],
            report["minimal_sufficient_sets"],
        )

    def test_reports_out_of_scope_at_bounded_depth(self):
        report = care.hierarchical_minimal_repair_search(
            lambda _subset: False,
            max_order=2,
        )

        self.assertEqual("out_of_scope", report["status"])
        self.assertEqual([], report["minimal_sufficient_sets"])

    def test_reports_no_fault_when_baseline_passes(self):
        report = care.hierarchical_minimal_repair_search(lambda _subset: True)

        self.assertEqual("no_fault", report["status"])
        self.assertEqual([[]], report["minimal_sufficient_sets"])
        self.assertEqual(1, report["probe_count"])

    def test_rejects_unknown_components(self):
        with self.assertRaises(care.CounterfactualContractError):
            care.hierarchical_minimal_repair_search(
                lambda _subset: False,
                components=["schema", "unknown"],
            )


if __name__ == "__main__":
    unittest.main()
