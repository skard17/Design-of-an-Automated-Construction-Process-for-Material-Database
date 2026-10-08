import io
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import section_design_langgraph_human_gate as graph  # noqa: E402
import step9_extraction_build_graph as step9_graph  # noqa: E402


def test_console_output_escapes_characters_missing_from_windows_code_page(monkeypatch):
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="gbk")
    monkeypatch.setattr(graph.sys, "stdout", stream)

    graph.print_console_safe("review \u2022 paused")
    stream.flush()

    assert b"review \\u2022 paused" in buffer.getvalue()


def test_human_gate_context_includes_task_schema_and_internal_review(monkeypatch, tmp_path):
    captured = {}
    context_path = tmp_path / "human_gate_context.json"
    monkeypatch.setattr(graph, "human_gate_context_path_from_args", lambda _args: context_path)
    monkeypatch.setattr(
        graph.run_artifact_guard,
        "atomic_write_json",
        lambda path, payload, run_identity=None: captured.update(
            path=path, payload=payload, run_identity=run_identity
        ),
    )
    state = {
        "args": {"reference_papers": ["paper.md"], "output": str(tmp_path / "step8.json")},
        "shared_context": {
            "database_goal": "Build a materials database from literature.",
            "discipline": "materials science",
            "query_requirements": ["Preserve evidence and sample identity."],
            "reference_paper_count": 1,
            "reference_paper_context": "paper preview",
        },
        "module_outputs": {
            "section_partition_module": {"core_sections": []},
            "field_planning_module": {"field_groups": []},
            "schema_design_module": {"field_registry": [{"field_path": "material.formula"}]},
            "specialization_critic_module": {"specialization_status": "pass"},
            "aggregation": {"status": "success"},
        },
        "supervisor_decision": {"action": "accept"},
        "schema_revision": 2,
        "schema_iteration_history": [{"schema_revision": 1}],
    }

    path, context = graph.write_human_gate_context(state)

    available = context["available_context"]
    assert path == str(context_path)
    assert available["task_contract"]["database_goal"].startswith("Build")
    assert available["schema_design_module"]["field_registry"]
    assert available["specialization_critic_module"]["specialization_status"] == "pass"
    assert available["aggregation"]["status"] == "success"
    assert available["supervisor_decision"]["action"] == "accept"
    assert available["schema_revision"] == 2
    assert captured["payload"] == graph.json_safe_state(context)


def base_state():
    return {
        "args": {"skip_human_expert_review": False},
        "shared_context": {
            "human_advice": '{"round": 1, "verdict": "revision_required"}',
            "human_advice_available_before_design": True,
        },
        "human_advice": '{"round": 1, "verdict": "revision_required"}',
        "module_outputs": {},
        "validation_errors": [],
        "retry_counts": {},
    }


def passthrough_snapshot(state, current_node, next_node):
    state["current_node"] = current_node
    state["next_node"] = next_node
    return state


def test_round_one_seed_requires_fresh_post_design_review(monkeypatch):
    requested = []
    monkeypatch.setattr(graph, "write_state_snapshot", passthrough_snapshot)
    monkeypatch.setattr(
        graph,
        "write_human_gate_context",
        lambda _state: ("context.json", {"required_advice": {"maximum_rounds": 3}}),
    )
    monkeypatch.setattr(
        graph,
        "interrupt",
        lambda request: requested.append(request) or '{"round": 2, "verdict": "approved"}',
    )

    result = graph.human_advice_gate_node(base_state())

    assert len(requested) == 1
    assert result["status"] == "success"
    assert result["human_review_round"] == 2
    assert result["module_outputs"]["human_advice_gate"]["status"] == "accepted"


def test_revision_request_is_consumed_before_another_review(monkeypatch):
    monkeypatch.setattr(graph, "write_state_snapshot", passthrough_snapshot)
    state = base_state()
    state["post_design_human_advice"] = (
        '{"round": 2, "verdict": "revision_required", "recommendations": ["repair binding"]}'
    )

    result = graph.human_advice_gate_node(state)

    assert result["status"] == "awaiting_supervisor_decision"
    assert result["next_node"] == "supervisor_router"
    assert result["human_review_round"] == 2
    assert result["post_design_human_advice"] == ""


def test_structured_accept_verdict_is_accepted(monkeypatch):
    monkeypatch.setattr(graph, "write_state_snapshot", passthrough_snapshot)
    state = base_state()
    state["post_design_human_advice"] = '{"round": 2, "verdict": "accept"}'

    result = graph.human_advice_gate_node(state)

    assert result["status"] == "success"
    assert result["human_review_round"] == 2
    assert result["module_outputs"]["human_advice_gate"]["status"] == "accepted"


def test_graph_state_preserves_post_design_review_resume_fields():
    annotations = graph.SectionDesignGraphState.__annotations__

    assert "post_design_human_advice" in annotations
    assert "applied_human_advice" in annotations
    assert "human_review_round" in annotations
    assert "human_review_applied" in annotations
    assert "allow_local_critic_bypass_after_human_advice" in annotations


def test_step9_graph_state_preserves_applied_human_review_fields():
    annotations = step9_graph.Step9State.__annotations__

    assert "applied_human_advice" in annotations
    assert "human_review_applied" in annotations


def test_step9_resume_injects_advice_and_returns_to_human_review():
    state = {
        "status": "waiting_for_human_advice",
        "next_node": "feedback_classification",
        "human_advice": [],
        "human_expert_review": {"status": "waiting_for_human_advice"},
    }
    advice = [{"round": 1, "verdict": "accept"}]

    result = step9_graph.inject_resume_human_advice(state, advice)

    assert result["human_advice"] == advice
    assert result["status"] == "running"
    assert result["next_node"] == "human_expert_review"


def test_step9_resume_without_advice_preserves_waiting_state():
    state = {
        "status": "waiting_for_human_advice",
        "next_node": "feedback_classification",
    }

    result = step9_graph.inject_resume_human_advice(state, [])

    assert result is state
    assert result["status"] == "waiting_for_human_advice"
    assert result["next_node"] == "feedback_classification"


def test_step9_resume_can_recover_accepted_review_from_stale_impact_hold():
    state = {
        "status": "needs_human_review",
        "next_node": "__end__",
        "human_expert_review": {"status": "accepted"},
    }
    advice = [{"round": 1, "verdict": "accepted"}]

    result = step9_graph.inject_resume_human_advice(state, advice)

    assert result["status"] == "running"
    assert result["next_node"] == "human_expert_review"


def test_step9_accepted_review_text_does_not_trigger_reprocessing():
    state = {
        "human_advice": [
            {
                "verdict": "accepted",
                "advice": "Field coverage and schema evidence notes are non-blocking.",
            }
        ],
        "human_expert_review": {"status": "accepted"},
        "extraction_eval": {"failure_diagnoses": []},
        "schema_feedback": None,
    }

    classification, impact = step9_graph.build_feedback_classification_and_impact(state)

    assert classification["feedback_type"] == "none"
    assert impact["status"] == "no_reprocessing_required"
    assert impact["affected_documents"] == []
    assert impact["affected_stage_ids"] == []
    assert impact["affected_field_paths"] == []


def test_three_round_limit_returns_control_to_supervisor(monkeypatch):
    monkeypatch.setattr(graph, "write_state_snapshot", passthrough_snapshot)
    state = base_state()
    state["human_review_round"] = 3

    result = graph.human_advice_gate_node(state)

    assert result["status"] == "awaiting_supervisor_decision"
    assert result["next_node"] == "supervisor_router"
    assert result["last_error_node"] == "human_advice_gate"
    assert result["module_outputs"]["human_advice_gate"]["status"] == "maximum_rounds_exhausted"
