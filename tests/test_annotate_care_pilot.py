import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import annotate_care_pilot as annotation  # noqa: E402
import tempfile


class AnnotateCarePilotTests(unittest.TestCase):
    def test_source_lines_support_comment_markers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pack.md"
            path.write_text("<!-- SOURCE_LINE: 3 -->\nHc2 = 7 T.\n", encoding="utf-8")

            lines = annotation.source_lines(path)

        self.assertEqual("Hc2 = 7 T.", lines[3])

    def test_unit_scale_equivalence(self):
        self.assertTrue(
            annotation.semantically_equivalent(
                "-0.2817", "10^{-3} cm^3/C", "-0.0002817", "cm3/C"
            )
        )

    def test_unicode_unit_equivalence(self):
        self.assertTrue(annotation.semantically_equivalent("1", "μm", "1", "um"))

    def test_distinct_scientific_claim_is_not_normalization(self):
        self.assertFalse(
            annotation.semantically_equivalent(
                "soft phonon modes", None, "dirty-limit scattering", None
            )
        )

    def test_kappa_is_one_for_identical_reviews(self):
        self.assertEqual(
            1.0,
            annotation.cohen_kappa(
                ["extraction", "normalization"], ["extraction", "normalization"]
            ),
        )

    def test_expert_override_requires_a_valid_component_and_reason(self):
        component, notes, override = annotation.apply_expert_override(
            "c1",
            "extraction",
            "automatic",
            {
                "c1": {
                    "primary_component": "binding",
                    "decision_notes": "The fact is explicitly associated with another material.",
                }
            },
        )

        self.assertEqual("binding", component)
        self.assertIn("another material", notes)
        self.assertIsNotNone(override)


if __name__ == "__main__":
    unittest.main()
