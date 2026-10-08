import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import build_targeted_care_replay as builder  # noqa: E402


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_targeted_replay_uses_captured_lines_and_covers_all_papers(tmp_path):
    papers = []
    facts = []
    for index in range(5):
        paper_id = f"paper-{index}"
        markdown = tmp_path / f"{paper_id}.md"
        _write(markdown, "header\nmeasured value is here\n")
        papers.append(
            {
                "paper_id": paper_id,
                "markdown_path": str(markdown),
                "in_extraction_gold": True,
            }
        )
        facts.append(
            {
                "target_id": f"target-{index}",
                "paper_id": paper_id,
                "record_key": f"sample-{index}",
                "concept_id": "transition_temperature",
                "value": index + 1.0,
                "unit": "K",
                "qualifiers": {"criterion": "onset"},
                "evidence_line": 2,
            }
        )
    predictions = tmp_path / "predictions.json"
    manifest = tmp_path / "manifest.json"
    _write(predictions, {"facts": facts})
    _write(manifest, {"papers": papers})

    report = builder.build_and_run(
        predictions_path=predictions,
        frozen_manifest_path=manifest,
        output_root=tmp_path / "replay",
        per_component=1,
        repeats=2,
        domain_label="materials_literature",
    )

    assert report["status"] == "pass"
    assert report["source_paper_count"] == 5
    assert report["exact_unique_responsibility_accuracy"] == 1.0
    assert report["deterministic_rate"] == 1.0
    public = json.loads((tmp_path / "replay" / "public" / "controlled_cases.json").read_text())
    serialized = json.dumps(public)
    assert "responsible_component" not in serialized
    assert "gold_record" not in serialized


def test_singleton_concept_uses_same_value_type_fallback(tmp_path):
    records = [
        {
            "paper_id": "paper-a",
            "record_key": "sample-a",
            "concept_id": "singleton-a",
            "value": 1.0,
            "unit": "K",
            "evidence": "a",
            "source_index": 0,
            "target_id": "a",
        },
        {
            "paper_id": "paper-b",
            "record_key": "sample-b",
            "concept_id": "singleton-b",
            "value": 2.0,
            "unit": "T",
            "evidence": "b",
            "source_index": 1,
            "target_id": "b",
        },
    ]

    distractor, mode = builder.find_distractor(records[0], records)

    assert distractor["target_id"] == "b"
    assert mode == "same_value_type_cross_paper"


def test_source_lines_accepts_unmarked_manifest_path_contract(tmp_path):
    markdown = tmp_path / "paper.md"
    _write(markdown, "line one\nline two\n")

    result = builder._source_lines(
        {"papers": [{"paper_id": "paper-a", "path": str(markdown)}]}
    )

    assert result == {"paper-a": ["line one", "line two"]}
