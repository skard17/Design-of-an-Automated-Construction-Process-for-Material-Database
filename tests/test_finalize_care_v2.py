import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import finalize_care_v2 as finalizer  # noqa: E402


class FinalizeCareV2Tests(unittest.TestCase):
    def test_manifest_verification_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {"path": "artifact.json", "sha256": finalizer.sha256(artifact)}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual([], finalizer.verify_manifest(root, manifest))
            artifact.write_text('{"changed":true}', encoding="utf-8")
            self.assertEqual(1, len(finalizer.verify_manifest(root, manifest)))

    def test_junit_parser_sums_suites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tests.xml"
            path.write_text(
                '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1"/>'
                '<testsuite tests="3" failures="1" errors="0" skipped="0"/></testsuites>',
                encoding="utf-8",
            )
            self.assertEqual(
                {"tests": 5, "failures": 1, "errors": 0, "skipped": 1},
                finalizer.parse_junit(path),
            )


if __name__ == "__main__":
    unittest.main()
