import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import downstream_extraction_runner as runner  # noqa: E402
import step9_extraction_build_graph as extraction_graph  # noqa: E402


class DownstreamExtractionQualityTests(unittest.TestCase):
    def test_negative_placeholder_is_moved_to_missing(self):
        payload = {
            "stage_id": "core_parameters",
            "section_id": "material_info.section1",
            "extracted_fields": [
                {
                    "field_path": "material_info.section1.coercive_field",
                    "value": "Not explicitly extracted; no coercive field was reported.",
                    "evidence_text": "The paper does not report a coercive field.",
                }
            ],
            "missing_fields": [],
        }

        sanitized = runner.sanitize_payload(payload)

        self.assertEqual([], sanitized["extracted_fields"])
        self.assertEqual(
            "material_info.section1.coercive_field",
            sanitized["missing_fields"][0]["field_path"],
        )

    def test_scientific_negative_observation_is_preserved(self):
        self.assertFalse(runner.is_null_like("No magnetic ordering was observed above 2 K."))

    def test_duplicate_values_merge_evidence(self):
        payload = {
            "stage_id": "core_parameters",
            "section_id": "material_info.section1",
            "extracted_fields": [
                {
                    "field_path": "material_info.section1.transition_temperature",
                    "value": 14.6,
                    "unit": "K",
                    "material_system": "EuNi2As2",
                    "evidence_text": "A transition occurs at 14.6 K.",
                    "source_hint": "page 2",
                    "confidence": 0.91,
                },
                {
                    "field_path": "material_info.section1.transition_temperature",
                    "value": 14.6,
                    "unit": "K",
                    "material_system": "EuNi2As2",
                    "evidence_text": "TN = 14.6 K.",
                    "source_hint": "Figure 3",
                    "confidence": 0.96,
                },
            ],
            "missing_fields": [],
        }

        sanitized = runner.sanitize_payload(payload)

        self.assertEqual(1, len(sanitized["extracted_fields"]))
        merged = sanitized["extracted_fields"][0]
        self.assertEqual(2, len(merged["evidence_items"]))
        self.assertEqual(0.96, merged["confidence"])

    def test_extracted_value_removes_conflicting_missing_entry(self):
        payload = {
            "stage_id": "paper_info",
            "section_id": "paper_info",
            "extracted_fields": [
                {
                    "field_path": "paper_info.metadata.doi",
                    "value": "10.1000/example",
                    "evidence_text": "doi:10.1000/example",
                }
            ],
            "missing_fields": [
                {
                    "field_path": "paper_info.metadata.doi",
                    "missing_reason": "not provided in document",
                }
            ],
        }

        sanitized = runner.sanitize_payload(payload)

        self.assertEqual(1, len(sanitized["extracted_fields"]))
        self.assertEqual([], sanitized["missing_fields"])

    def test_unstructured_inferred_label_is_not_treated_as_fact(self):
        payload = {
            "stage_id": "mechanism",
            "section_id": "section5",
            "extracted_fields": [
                {
                    "field_path": "section5.gap_symmetry",
                    "value": "s-wave (implied conventional)",
                    "evidence_text": "The discussion describes conventional behavior.",
                }
            ],
            "missing_fields": [],
        }

        sanitized = runner.sanitize_payload(payload)

        self.assertEqual([], sanitized["extracted_fields"])
        self.assertIn("structured inference basis", sanitized["missing_fields"][0]["missing_reason"])

    def test_structured_inference_is_preserved(self):
        payload = {
            "stage_id": "mechanism",
            "section_id": "section5",
            "extracted_fields": [
                {
                    "field_path": "section5.state_assignment",
                    "value": "candidate state (inferred)",
                    "assignment_basis": "fit to the explicitly reported model",
                    "source_type": "author_interpretation",
                    "confidence": 0.7,
                    "evidence_text": "The authors assign this state from the fit.",
                }
            ],
            "missing_fields": [],
        }

        sanitized = runner.sanitize_payload(payload)

        self.assertEqual(1, len(sanitized["extracted_fields"]))

    def test_complete_missing_coverage_is_not_low_quality(self):
        checks = {
            "extracted_count": 0,
            "missing_count": 2,
            "addressed_field_paths": ["device_info.forward_current", "device_info.reverse_current"],
        }

        assessment = runner.assess_stage_yield(
            [checks],
            ["device_info.forward_current", "device_info.reverse_current"],
            stage_id="device_response",
        )

        self.assertEqual("passed", assessment["status"])
        self.assertEqual("no_values_found", assessment["extraction_outcome"])
        self.assertEqual(1.0, assessment["field_coverage_ratio"])

    def test_supervisor_accepts_complete_negative_result(self):
        section_result = {
            "stage_id": "device_response",
            "section_id": "device_info",
            "status": "passed",
            "summary": {
                "extracted_total": 0,
                "missing_total": 2,
                "field_coverage_ratio": 1.0,
                "extraction_outcome": "no_values_found",
            },
            "document_results": [],
        }

        self.assertEqual([], extraction_graph.diagnose_section_result(section_result))


if __name__ == "__main__":
    unittest.main()
