import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import section_design_langgraph_human_gate as graph


def test_prepare_state_enables_structured_protocol_before_design(tmp_path, monkeypatch):
    key_description = tmp_path / "context.yaml"
    key_description.write_text("schema_policy: derive from task and papers\n", encoding="utf-8")
    advice = tmp_path / "advice.json"
    advice.write_text('{"schema_concepts": []}', encoding="utf-8")
    monkeypatch.setattr(
        graph,
        "write_state_snapshot",
        lambda state, current_node, next_node=None: {
            **state,
            "current_node": current_node,
            "next_node": next_node,
        },
    )
    state = {
        "args": {
            "database_goal": "Build a materials database from literature.",
            "discipline": "Materials science",
            "query_requirements": "Preserve sample identity and evidence.",
            "key_description_path": str(key_description),
            "reference_papers": [],
            "human_advice": "",
            "human_advice_path": str(advice),
            "structured_protocol": True,
        },
        "next_node": "prepare",
    }

    prepared = graph.prepare_state(state)

    assert prepared["next_node"] == "locating"
    assert prepared["shared_context"]["structured_protocol_enabled"] is True
    assert prepared["shared_context"]["structured_concept_seeding_only"] is True
    assert prepared["shared_context"]["human_advice_available_before_design"] is True


def test_parser_defaults_to_unstructured_protocol_mode():
    parser = graph.build_parser()
    action = next(item for item in parser._actions if item.dest == "structured_protocol")

    assert action.default is False


def test_care_queries_become_distinct_shared_requirements(tmp_path):
    advice = {
        "care_counterfactual_enabled": True,
        "counterfactual_queries": [
            {
                "query_id": "care_binding_swap",
                "category": "binding",
                "query": "Can sample ownership be swapped without changing the record?",
                "required_distinctions": ["sample owner", "conditions"],
                "failure_exposed": "detached values",
            }
        ],
    }
    advice_path = tmp_path / "advice.json"
    advice_path.write_text(json.dumps(advice), encoding="utf-8")
    key_description = tmp_path / "context.yaml"
    key_description.write_text("schema_policy: task adaptive\n", encoding="utf-8")

    shared, _ = graph.build_shared_context_from_args(
        {
            "database_goal": "Build a materials database from literature.",
            "discipline": "Materials science",
            "query_requirements": "Retrieve material identity.",
            "key_description_path": str(key_description),
            "reference_papers": [],
            "human_advice": "",
            "human_advice_path": str(advice_path),
            "structured_protocol": True,
        }
    )

    assert shared["base_query_requirement_count"] == 1
    assert shared["care_counterfactual_enabled"] is True
    assert shared["care_counterfactual_requirement_ids"] == ["query_2"]
    assert shared["care_counterfactual_queries"][0]["query_id"] == "care_binding_swap"
    assert "sample owner" in shared["query_requirements"][1]
