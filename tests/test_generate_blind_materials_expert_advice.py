import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from generate_blind_materials_expert_advice import (
    build_prompt,
    build_structured_protocol_advice,
    load_blind_task_contract,
)


def test_blind_task_contract_excludes_evaluator_only_inputs(tmp_path):
    spec = {
        "task": {
            "objective": "Build a materials database from literature.",
            "discipline": "Materials science",
        },
        "evaluation_only_inputs": {
            "manual_schema": "secret-gold.json",
            "manual_leaf_count": 239,
        },
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    task = load_blind_task_contract(path)
    prompt = build_prompt(task, 1)

    assert task == {
        "objective": "Build a materials database from literature.",
        "discipline": "Materials science",
    }
    assert "secret-gold.json" not in prompt
    assert "239" not in prompt


def test_round_one_prompt_has_no_cross_version_artifact():
    prompt = build_prompt({"objective": "Build from papers."}, 1)

    assert "No current-version artifact is available" in prompt
    assert "output from another version" in prompt
    assert "forbidden_information_used" in prompt


def test_structured_protocol_advice_is_valid_and_task_blind():
    task = {"objective": "Build a materials database from papers."}
    advice = {
        "round": 1,
        "verdict": "revision_required",
        "blocking_issues": ["Measurement records need explicit sample binding."],
        "recommendations": [],
        "schema_concepts": [
            {
                "concept_id": "sample_identity",
                "label": "sample identity",
                "entity_id": "sample",
                "owner_key": "material_info",
                "object_kind": "scalar",
                "required": True,
                "condition_requirements": [],
                "evidence_types": ["text"],
            }
        ],
        "forbidden_information_used": False,
    }

    payload = build_structured_protocol_advice(advice, task)

    assert payload["structured_protocol_enabled"] is True
    assert payload["protocol_validation"] == {
        "valid": True,
        "message_count": 1,
        "errors": [],
    }
    assert payload["protocol_messages"][0]["receiver"] == "step8_supervisor"
    assert payload["blind_task_contract"] == task
    assert "manual_schema" not in json.dumps(payload)


def test_structured_prompt_requests_concepts_without_a_domain_catalog():
    prompt = build_prompt({"objective": "Build from papers."}, 1, structured_protocol=True)

    assert "schema_concepts" in prompt
    assert "cross-domain" in prompt
    assert "do not invent a superconductivity field catalog" in prompt
