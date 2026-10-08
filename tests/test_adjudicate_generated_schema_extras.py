import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import adjudicate_generated_schema_extras as adjudicator


def test_validate_items_requires_exact_paths_and_allowed_classes():
    items = adjudicator.validate_items(
        {
            "items": [
                {
                    "system_path": "sample.lineage_id",
                    "classification": "useful_supplement",
                    "rationale": "Preserves sample identity across observations.",
                }
            ]
        },
        ["sample.lineage_id"],
    )

    assert items[0]["classification"] == "useful_supplement"


def test_split_retry_batch_preserves_order_and_remainder():
    assert adjudicator.split_retry_batch(["a", "b", "c", "d", "e"], 2) == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_run_writes_evaluator_compatible_items(tmp_path, monkeypatch):
    manual = tmp_path / "manual.json"
    manual.write_text(json.dumps({"material": {"name": None}}), encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "stages": [
                    {
                        "stage_id": "v1",
                        "schema": {
                            "manual_json_238": {
                                "missing_gold_paths": [],
                                "system_extras": [
                                    {"system_path": "sample.lineage_id"}
                                ],
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_AGENT_API_KEY", "test-key")
    monkeypatch.setattr(
        adjudicator.prompt_agent,
        "litellm_chat",
        lambda **kwargs: json.dumps(
            {
                "items": [
                    {
                        "system_path": "sample.lineage_id",
                        "classification": "useful_supplement",
                        "rationale": "Preserves sample identity across observations.",
                    }
                ]
            }
        ),
    )
    output = tmp_path / "adjudication.json"
    args = adjudicator.build_parser().parse_args(
        [
            "--comparison-report",
            str(report),
            "--manual-schema",
            str(manual),
            "--stage-id",
            "v1",
            "--run-dir",
            str(tmp_path / "runs"),
            "--output",
            str(output),
            "--base-url",
            "https://example.invalid",
            "--model",
            "test-model",
        ]
    )

    result = adjudicator.run(args)

    assert result["extra_count"] == 1
    assert result["adjudicated_count"] == 1
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["items"][0]["system_path"] == "sample.lineage_id"
