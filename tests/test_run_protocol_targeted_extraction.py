import sys
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import materials_agent_protocol as protocol
import run_protocol_targeted_extraction as runner
from target_extraction_semantics import enrich_targets


def target():
    return {
        "target_id": "t1",
        "paper_id": "p1",
        "record_key": "sample-a",
        "concept_id": "transition_temperature",
        "qualifier_keys": ["criterion"],
    }


def test_protocol_message_stays_inside_materials_literature_task(tmp_path):
    pack = tmp_path / "p1.md"
    pack.write_text("<!-- SOURCE_LINE: 9 -->\nTc is 4 K.\n", encoding="utf-8")
    message = runner.extraction_protocol_message("p1", [target()], pack)
    assert protocol.validate_message(message) == []
    assert message["task_contract"]["task_type"] == "automated_materials_database_construction"
    assert message["task_contract"]["source_scope"] == "scientific_literature_only"
    assert message["decision"]["answer_values_present_in_manifest"] is False


def test_supervisor_accepts_only_source_line_and_exact_qualifier_contract():
    payload = {
        "facts": [
            {
                "target_id": "t1",
                "value": 4,
                "unit": "K",
                "qualifiers": {"criterion": "resistive onset"},
                "evidence_line": 9,
            }
        ]
    }
    facts = runner.validate_response(payload, [target()], {9})
    assert facts[0]["paper_id"] == "p1"
    payload["facts"][0]["evidence_line"] = 10
    with pytest.raises(ValueError, match="SOURCE_LINE"):
        runner.validate_response(payload, [target()], {9})


def test_prompt_contains_protocol_but_no_expected_answer(tmp_path):
    pack = tmp_path / "p1.md"
    pack.write_text("<!-- SOURCE_LINE: 9 -->\nTc is 4 K.\n", encoding="utf-8")
    message = runner.extraction_protocol_message("p1", [target()], pack)
    prompt = runner.build_prompt("p1", [target()], pack.read_text(), message)
    assert "automated_materials_database_construction" in prompt
    assert "answer_values_present_in_manifest" in prompt
    assert '"target_id": "t1"' in prompt


def test_target_enrichment_adds_semantics_without_answers():
    enriched = enrich_targets(
        [target()],
        {
            "concepts": [
                {
                    "concept_id": "transition_temperature",
                    "label": "superconducting transition temperature",
                    "tier": "core",
                }
            ]
        },
    )[0]

    assert enriched["concept_contract"]["label"] == "superconducting transition temperature"
    assert "criterion" in enriched["qualifier_contract"]
    assert "value" not in enriched


def test_supervisor_prompt_compares_candidates_without_expected_answer(tmp_path):
    pack = tmp_path / "p1.md"
    evidence = "<!-- SOURCE_LINE: 9 -->\nTc is 4 K.\n"
    pack.write_text(evidence, encoding="utf-8")
    enriched = enrich_targets([target()])
    message = runner.extraction_protocol_message("p1", enriched, pack)
    candidate = {
        "target_id": "t1",
        "paper_id": "p1",
        "record_key": "sample-a",
        "concept_id": "transition_temperature",
        "value": 4,
        "unit": "K",
        "qualifiers": {"criterion": "reported criterion"},
        "evidence_line": 9,
    }

    prompt = runner.build_supervisor_prompt(
        "p1", enriched, evidence, message, [candidate], [candidate]
    )

    assert "expected_value" not in prompt
    assert "gold_value" not in prompt
    assert "Target semantic contracts" in prompt
    assert "prior CARE-reviewed candidate" in prompt


def test_run_reports_provider_failure_instead_of_preserving_seed(
    tmp_path, monkeypatch
):
    target_path = tmp_path / "targets.json"
    target_path.write_text(json.dumps({"targets": [target()]}), encoding="utf-8")
    pack_dir = tmp_path / "packs"
    pack_dir.mkdir()
    (pack_dir / "p1.md").write_text(
        "<!-- SOURCE_LINE: 9 -->\nTc is 4 K.\n", encoding="utf-8"
    )
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "facts": [
                    {
                        "target_id": "t1",
                        "value": 4,
                        "unit": "K",
                        "qualifiers": {"criterion": "resistive onset"},
                        "evidence_line": 9,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "predictions.json"
    monkeypatch.setenv("CODE_AGENT_API_KEY", "test-key")
    monkeypatch.setattr(
        runner.prompt_agent,
        "litellm_chat",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("provider 504")),
    )
    args = runner.build_parser().parse_args(
        [
            "--targets",
            str(target_path),
            "--pack-dir",
            str(pack_dir),
            "--run-dir",
            str(tmp_path / "runs"),
            "--output",
            str(output_path),
            "--seed-predictions",
            str(seed_path),
            "--supervisor-review",
            "--base-url",
            "https://example.invalid",
            "--model",
            "test-model",
            "--max-attempts",
            "2",
            "--max-targets-per-call",
            "1",
        ]
    )

    result = runner.run(args)

    assert result["prediction_count"] == 0
    assert result["failure_count"] == 1
    assert result["supervisor_reviewed_target_count"] == 0
    assert result["seed_fallback_paper_count"] == 0
    assert result["seed_fallback_target_count"] == 0
    assert result["seed_fallback_disabled"] is True
    assert result["protocol_validation"]["valid"] is True

    with pytest.raises(runner.run_artifact_guard.CleanRunRequiredError):
        runner.run(args)


def test_seed_fallback_cli_option_was_removed():
    option_strings = {
        option
        for action in runner.build_parser()._actions
        for option in action.option_strings
    }

    assert "--fallback-to-seed-on-review-failure" not in option_strings
