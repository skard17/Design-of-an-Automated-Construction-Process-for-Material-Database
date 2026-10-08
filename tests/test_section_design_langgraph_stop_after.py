import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from section_design_langgraph_human_gate import (
    build_parser,
    explicit_cli_destinations,
    inject_resume_human_advice,
    requested_stop_after_module,
    stop_if_errors,
)


def test_resume_overrides_only_explicit_cli_options():
    parser = build_parser()
    argv = [
        "--resume-from-state",
        "checkpoint.json",
        "--human-advice-path",
        "expert.json",
        "--max-retries=10",
    ]

    destinations = explicit_cli_destinations(parser, argv)

    assert "resume_from_state" in destinations
    assert "human_advice_path" in destinations
    assert "max_retries" in destinations
    assert "base_url" not in destinations
    assert "model" not in destinations
    assert "request_timeout" not in destinations


def test_stop_after_routes_completed_module_to_output():
    state = {
        "args": {"stop_after_module": "figure_classification_module"},
        "current_node": "figure_classification",
        "validation_errors": [],
    }

    assert requested_stop_after_module(state) is True
    assert stop_if_errors(state, "schema_design") == "write_output"


def test_stop_after_does_not_match_another_module():
    state = {
        "args": {"stop_after_module": "figure_classification_module"},
        "current_node": "supervisor",
        "validation_errors": [],
    }

    assert requested_stop_after_module(state) is False
    assert stop_if_errors(state, "figure_classification") == "figure_classification"


def test_validation_errors_take_precedence_over_partial_stop():
    state = {
        "args": {"stop_after_module": "figure_classification_module"},
        "current_node": "figure_classification",
        "validation_errors": ["invalid module output"],
    }

    assert stop_if_errors(state, "schema_design") == "supervisor_router"


def test_mechanism_node_uses_requirement_module_name():
    state = {
        "args": {"stop_after_module": "mechanism_requirement_module"},
        "current_node": "mechanism",
        "validation_errors": [],
    }

    assert requested_stop_after_module(state) is True


def test_resume_human_advice_is_marked_available_before_design():
    state = {"shared_context": {"human_advice_available_before_design": False}}

    updated = inject_resume_human_advice(state, "blind expert advice")

    assert updated["human_advice"] == "blind expert advice"
    assert updated["shared_context"]["human_advice"] == "blind expert advice"
    assert updated["shared_context"]["human_advice_available_before_design"] is True


def test_resume_human_advice_at_gate_is_marked_as_post_design_review():
    state = {
        "status": "waiting_for_human_advice",
        "current_node": "human_advice_gate",
        "next_node": "human_advice_gate",
        "shared_context": {"human_advice_available_before_design": True},
    }

    updated = inject_resume_human_advice(state, '{"round": 2, "verdict": "accept"}')

    assert updated["post_design_human_advice"] == '{"round": 2, "verdict": "accept"}'
    assert updated["shared_context"]["human_advice_available_before_design"] is True
