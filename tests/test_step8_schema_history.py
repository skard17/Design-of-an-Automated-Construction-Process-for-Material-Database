import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from section_design_langgraph_human_gate import (
    append_schema_iteration_record,
    build_schema_iteration_context,
    schema_iteration_log_path_from_args,
    write_schema_revision_snapshot,
    write_transition_snapshot,
)


def make_state(tmp_path, field_path="material_info.identity.formula"):
    return {
        "args": {
            "output": str(tmp_path / "step8.json"),
            "checkpoint_output": str(tmp_path / "step8.state.json"),
        },
        "schema_revision": 10,
        "critic_reviewed_schema_revision": 9,
        "module_outputs": {
            "schema_design_module": {
                "top_level_keys": ["material_info"],
                "field_registry": [{"field_path": field_path}],
            }
        },
        "supervisor_decision": {"action": "repair_schema_design"},
    }


def test_schema_revision_snapshot_is_versioned_and_idempotent(tmp_path):
    state = make_state(tmp_path)

    first = write_schema_revision_snapshot(state, "schema_design", "specialization_critic")
    second = write_schema_revision_snapshot(state, "aggregation", "supervisor_router")

    assert first == second
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["schema_revision"] == 10
    assert payload["parent_schema_revision"] == 9
    assert payload["field_count"] == 1
    assert payload["schema"]["field_registry"][0]["field_path"] == (
        "material_info.identity.formula"
    )
    assert len(list(first.parent.glob("revision_000010*.json"))) == 1


def test_schema_revision_collision_preserves_both_variants(tmp_path):
    first = write_schema_revision_snapshot(
        make_state(tmp_path), "schema_design", "specialization_critic"
    )
    second = write_schema_revision_snapshot(
        make_state(tmp_path, "material_info.identity.name"),
        "schema_design",
        "specialization_critic",
    )

    assert first != second
    assert first.exists()
    assert second.exists()
    assert len(list(first.parent.glob("revision_000010*.json"))) == 2


def test_transition_snapshot_preserves_module_result_and_supervisor_context(tmp_path):
    state = make_state(tmp_path)
    state["module_outputs"]["specialization_critic_module"] = {
        "redo_needed": True,
        "findings": [{"code": "redundant_field"}],
    }
    state["module_attempts"] = {
        "specialization_critic_module": [
            {"round": 3, "raw_response": '{"redo_needed": true}'}
        ]
    }
    state["module_errors"] = {"specialization_critic_module": []}
    state["validation_errors"] = ["critic patch rejected"]
    state["repair_instructions"] = ["repair the rejected patch"]
    state["blocker_history"] = [{"decision": "retry_node"}]

    first = write_transition_snapshot(
        state, "specialization_critic", "supervisor_router"
    )
    second = write_transition_snapshot(
        state, "specialization_critic", "supervisor_router"
    )

    assert first == second
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["module_name"] == "specialization_critic_module"
    assert payload["module_output"]["redo_needed"] is True
    assert payload["latest_module_attempt"]["round"] == 3
    assert payload["supervisor_decision"]["action"] == "repair_schema_design"
    assert payload["validation_errors"] == ["critic patch rejected"]
    assert payload["latest_blocker"] == {"decision": "retry_node"}
    assert len(list(first.parent.glob("e_*.json"))) == 1


def test_transition_snapshot_preserves_non_agent_node_output(tmp_path):
    state = make_state(tmp_path)
    state["result"] = {"status": "needs_review", "reason": "gate rejected"}

    archived = write_transition_snapshot(state, "write_output", None)

    payload = json.loads(archived.read_text(encoding="utf-8"))
    assert payload["module_name"] == "__result__"
    assert payload["module_output"] == state["result"]


def test_iteration_log_persists_changes_feedback_and_prompt_context(tmp_path):
    state = make_state(tmp_path)
    state["schema_iteration_history"] = []
    state["validation_errors"] = ["coverage:missing_concepts:critical_temperature"]
    state["repair_instructions"] = ["add a task-supported mapping"]
    state["module_outputs"]["specialization_critic_module"] = {
        "reviewed_schema_revision": 10,
        "redo_needed": True,
        "redo_directives": ["add a task-supported mapping"],
        "patch_operations": [],
    }

    updated = append_schema_iteration_record(
        state, "specialization_critic", "supervisor_router"
    )

    log_path = schema_iteration_log_path_from_args(type("Args", (), state["args"])())
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 1
    entry = payload["entries"][0]
    assert entry["schema_revision"] == 10
    assert entry["critic_review"]["redo_needed"] is True
    assert entry["repair_instructions"] == ["add a task-supported mapping"]
    context = build_schema_iteration_context(updated)
    assert "SCHEMA_ITERATION_HISTORY" in context
    assert "critical_temperature" in context
