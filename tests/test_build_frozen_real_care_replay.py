import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import build_frozen_real_care_replay as builder  # noqa: E402
from care_counterfactual import COMPONENTS  # noqa: E402


class FrozenRealCareReplayTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path]:
        frozen_ids = ["paper-a", "paper-b", "paper-c"]
        manifest = root / "frozen.json"
        manifest.write_text(
            json.dumps({"in_domain_frozen_papers": frozen_ids}), encoding="utf-8"
        )
        documents = []
        for paper_index, paper_id in enumerate(frozen_ids):
            fields = []
            for field_index in range(10):
                fields.append(
                    {
                        "stage": "material_info.section1",
                        "field_path": f"material_info.section1.field_{field_index % 2}",
                        "value": paper_index * 100 + field_index,
                        "unit": "K",
                        "material_system": f"Material-{paper_id}",
                        "evidence_text": (
                            f"Measured evidence for {paper_id} field {field_index} has a sufficiently long sentence."
                        ),
                        "source_hint": f"Page 2, line {10 + field_index}",
                        "source_type": "ocr_markdown",
                        "confidence": 0.9,
                    }
                )
            documents.append({"paper_id": paper_id, "extracted_fields": fields})
        corpus = root / "corpus.json"
        corpus.write_text(json.dumps({"documents": documents}), encoding="utf-8")
        return manifest, corpus

    def test_builds_balanced_leak_free_replay_covering_all_frozen_papers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, corpus = self._inputs(root)
            output = root / "output"

            report = builder.build_and_run(
                frozen_manifest_path=manifest,
                full_corpus_path=corpus,
                output_root=output,
                per_component=2,
                repeats=2,
            )

            self.assertEqual("pass", report["status"])
            self.assertEqual(10, report["case_count"])
            self.assertEqual(3, report["frozen_paper_count"])
            self.assertEqual(1.0, report["exact_unique_responsibility_accuracy"])
            self.assertEqual({name: 2 for name in COMPONENTS}, report["gold_distribution"])
            public = json.loads(
                (output / "public" / "controlled_cases.json").read_text(encoding="utf-8")
            )
            serialized = json.dumps(public)
            self.assertNotIn("responsible_component", serialized)
            self.assertNotIn("gold_record", serialized)


if __name__ == "__main__":
    unittest.main()
