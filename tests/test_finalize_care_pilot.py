import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import finalize_care_pilot as finalizer  # noqa: E402


class FinalizeCarePilotTests(unittest.TestCase):
    def test_secret_scan_returns_file_names_without_secret_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "clean.json").write_text('{"ok": true}', encoding="utf-8")
            (root / "bad.txt").write_text(
                "sk-abcdefghijklmnopqrstuvwxyz123456", encoding="utf-8"
            )

            matches = finalizer.scan_credentials(root)

        self.assertEqual(["bad.txt"], matches)

    def test_sha256_is_stable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "x.txt"
            path.write_text("abc", encoding="utf-8")

            digest = finalizer.sha256(path)

        self.assertEqual(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            digest,
        )


if __name__ == "__main__":
    unittest.main()
