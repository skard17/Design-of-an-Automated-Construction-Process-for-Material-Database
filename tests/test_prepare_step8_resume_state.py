import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from prepare_step8_resume_state import (
    prepare_forked_resume_state,
    prepare_resume_state,
    remap_fork_args,
)


def test_restart_discards_target_and_downstream_modules():
    state = {
        "args": {
            "output": "old.json",
            "checkpoint_output": "old.state.json",
            "stop_after_module": "figure_classification_module",
        },
        "module_outputs": {
            "locating_module": {"ok": True},
            "evidence_model_module": {"ok": True},
            "subjective_supervisor_module": {"bad": True},
            "topic_adaptation_module": {"bad": True},
        },
        "module_attempts": {
            "evidence_model_module": [1],
            "subjective_supervisor_module": [2],
        },
        "module_errors": {"subjective_supervisor_module": ["bad"]},
        "validation_errors": ["bad"],
        "status": "needs_review",
        "current_node": "topic_adaptation",
        "next_node": "section_partition",
        "result": {"stale": True},
        "run_identity": {"run_id": "run-1"},
        "schema_revision": 4,
        "critic_reviewed_schema_revision": 4,
    }

    prepared = prepare_resume_state(
        state,
        "subjective_supervisor",
        run_output="old.json",
        checkpoint_output="old.state.json",
    )

    assert set(prepared["module_outputs"]) == {
        "locating_module",
        "evidence_model_module",
    }
    assert "subjective_supervisor_module" not in prepared["module_attempts"]
    assert prepared["module_errors"] == {}
    assert prepared["validation_errors"] == []
    assert prepared["current_node"] == "evidence_model"
    assert prepared["next_node"] == "subjective_supervisor"
    assert prepared["result"] is None
    assert prepared["args"]["output"] == "old.json"
    assert prepared["args"]["checkpoint_output"] == "old.state.json"
    assert prepared["args"]["stop_after_module"] == ""
    assert prepared["schema_revision"] == 0
    assert prepared["critic_reviewed_schema_revision"] == -1
    assert prepared["blocker_history"] == []


def test_restart_at_critic_keeps_schema_revision_but_invalidates_old_review():
    state = {
        "args": {"output": "old.json", "checkpoint_output": "old.state.json"},
        "module_outputs": {"schema_design_module": {"field_registry": [{"field_path": "x.y"}]}},
        "run_identity": {"run_id": "run-1"},
        "schema_revision": 7,
        "critic_reviewed_schema_revision": 7,
    }

    prepared = prepare_resume_state(state, "specialization_critic")

    assert prepared["schema_revision"] == 7
    assert prepared["critic_reviewed_schema_revision"] == -1


def test_restart_at_supervisor_preserves_current_critic_patch_and_restores_errors():
    feedback = [
        "field x is figure-based but missing figure_constraint",
        "critic supplied pending structured patch_operations",
    ]
    critic = {
        "reviewed_schema_revision": 7,
        "patch_operations": [
            {"op": "remove_field", "field_path": "x.y", "reason": "duplicate"}
        ],
    }
    state = {
        "args": {"output": "old.json", "checkpoint_output": "old.state.json"},
        "module_outputs": {
            "schema_design_module": {"field_registry": [{"field_path": "x.y"}]},
            "specialization_critic_module": critic,
            "aggregation": {"stale": True},
        },
        "module_attempts": {"specialization_critic_module": [{"round": 1}]},
        "module_errors": {},
        "run_identity": {"run_id": "run-1"},
        "schema_revision": 7,
        "critic_reviewed_schema_revision": 7,
        "repair_instructions": feedback,
        "validation_errors": [],
        "retry_counts": {"old": 2},
        "blocker_history": [{"old": True}],
    }

    prepared = prepare_resume_state(state, "supervisor_router")

    assert prepared["next_node"] == "supervisor_router"
    assert prepared["current_node"] == "specialization_critic"
    assert prepared["validation_errors"] == feedback
    assert prepared["repair_instructions"] == feedback
    assert prepared["last_error_node"] == "schema_design"
    assert prepared["last_error_type"] == "schema_validation_failure"
    assert prepared["module_outputs"]["specialization_critic_module"] == critic
    assert "aggregation" not in prepared["module_outputs"]
    assert prepared["retry_counts"] == {}
    assert prepared["blocker_history"] == []
    assert prepared["schema_revision"] == 7
    assert prepared["critic_reviewed_schema_revision"] == 7


def test_restart_cannot_relabel_old_state_as_a_fresh_run():
    state = {
        "args": {"output": "old.json", "checkpoint_output": "old.state.json"},
        "module_outputs": {},
        "run_identity": {"run_id": "run-1"},
    }

    try:
        prepare_resume_state(state, "locating", run_output="new.json")
    except ValueError as exc:
        assert "cannot change" in str(exc)
    else:
        raise AssertionError("changing output paths during resume must be rejected")


def test_checkpoint_fork_uses_new_identity_and_rehydrates_context(tmp_path):
    source_path = tmp_path / "source.state.json"
    source_path.write_text("{}", encoding="utf-8")
    state = {
        "args": {"output": "old.json", "checkpoint_output": "old.state.json"},
        "module_outputs": {
            "locating_module": {"ok": True},
            "schema_design_module": {
                "field_registry": [{"field_path": "x.y"}]
            },
            "specialization_critic_module": {"stale": True},
        },
        "run_identity": {"run_id": "source-run"},
        "blocker_history": [{"old": True}],
        "schema_revision": 2,
    }
    new_args = {
        "output": "new.json",
        "checkpoint_output": "new.state.json",
        "api_key": "must-not-persist",
    }
    shared_context = {
        "reference_paper_context": "complete current corpus",
        "human_advice": "task-only advice",
    }

    prepared = prepare_forked_resume_state(
        state,
        "specialization_critic",
        args=new_args,
        run_identity={"run_id": "destination-run"},
        source_state_path=source_path,
        shared_context=shared_context,
    )

    assert prepared["run_identity"]["run_id"] == "destination-run"
    assert prepared["args"]["output"] == "new.json"
    assert prepared["args"]["api_key"] == "[REDACTED]"
    assert prepared["shared_context"] == shared_context
    assert "specialization_critic_module" not in prepared["module_outputs"]
    assert prepared["checkpoint_fork"]["source_run_id"] == "source-run"
    assert prepared["checkpoint_fork"]["source_artifacts_mutated"] is False
    assert prepared["checkpoint_fork"]["preserved_active_repair_feedback_count"] == 0


def test_checkpoint_fork_preserves_only_active_same_critic_feedback(tmp_path):
    source_path = tmp_path / "source.state.json"
    source_path.write_text("{}", encoding="utf-8")
    feedback = [
        "specialization_critic field_utility_audit references unknown fields: bad.path",
        "specialization_critic utility findings lack executable patch operations: x.y",
    ]
    state = {
        "args": {"output": "old.json", "checkpoint_output": "old.state.json"},
        "module_outputs": {
            "schema_design_module": {"field_registry": [{"field_path": "x.y"}]},
            "specialization_critic_module": {"stale": True},
        },
        "run_identity": {"run_id": "source-run"},
        "current_node": "supervisor_router",
        "next_node": "specialization_critic",
        "last_error_node": "specialization_critic",
        "repair_instructions": feedback,
        "retry_counts": {"old": 3},
        "blocker_history": [{"old": True}],
        "schema_revision": 2,
    }

    prepared = prepare_forked_resume_state(
        state,
        "specialization_critic",
        args={
            "output": "new.json",
            "checkpoint_output": "new.state.json",
            "api_key": "must-not-persist",
        },
        run_identity={"run_id": "destination-run"},
        source_state_path=source_path,
        shared_context={"reference_paper_context": "fresh corpus"},
    )

    assert prepared["repair_instructions"] == feedback
    assert prepared["retry_counts"] == {}
    assert prepared["blocker_history"] == []
    assert "specialization_critic_module" not in prepared["module_outputs"]
    assert prepared["checkpoint_fork"]["preserved_active_repair_feedback_count"] == 2


def test_checkpoint_fork_preserves_supervisor_router_feedback(tmp_path):
    source_path = tmp_path / "source.state.json"
    source_path.write_text("{}", encoding="utf-8")
    feedback = [
        "field x is figure-based but missing figure_constraint",
        "critic supplied pending structured patch_operations",
    ]
    state = {
        "args": {"output": "old.json", "checkpoint_output": "old.state.json"},
        "module_outputs": {
            "schema_design_module": {"field_registry": [{"field_path": "x.y"}]},
            "specialization_critic_module": {
                "reviewed_schema_revision": 3,
                "patch_operations": [
                    {
                        "op": "remove_field",
                        "field_path": "x.y",
                        "reason": "duplicate",
                    }
                ],
            },
        },
        "run_identity": {"run_id": "source-run"},
        "schema_revision": 3,
        "critic_reviewed_schema_revision": 3,
        "repair_instructions": feedback,
    }

    prepared = prepare_forked_resume_state(
        state,
        "supervisor_router",
        args={
            "output": "new.json",
            "checkpoint_output": "new.state.json",
            "api_key": "must-not-persist",
        },
        run_identity={"run_id": "destination-run"},
        source_state_path=source_path,
        shared_context={"reference_paper_context": "fresh corpus"},
    )

    assert prepared["validation_errors"] == feedback
    assert prepared["repair_instructions"] == feedback
    assert prepared["checkpoint_fork"]["preserved_active_repair_feedback_count"] == 2


def test_checkpoint_fork_remaps_only_within_isolated_namespace(tmp_path):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    for root in (source_root, destination_root):
        (root / "workspace" / "tests" / "fixtures").mkdir(parents=True)
        (root / "inputs" / "corpus").mkdir(parents=True)
        (root / "inputs" / "expert.json").write_text("{}", encoding="utf-8")
        (root / "workspace" / "tests" / "fixtures" / "catalog.txt").write_text(
            "catalog", encoding="utf-8"
        )
        (root / "inputs" / "corpus" / "paper.md").write_text(
            "paper", encoding="utf-8"
        )
    state_args = {
        "key_description_path": str(
            source_root / "workspace" / "tests" / "fixtures" / "catalog.txt"
        ),
        "reference_papers": [
            str(source_root / "inputs" / "corpus" / "paper.md")
        ],
        "human_advice_path": str(source_root / "inputs" / "expert.json"),
        "human_advice": "stale embedded source advice",
        "output": str(source_root / "artifacts" / "step8.json"),
        "checkpoint_output": str(source_root / "artifacts" / "step8.state.json"),
    }

    args = remap_fork_args(
        state_args,
        source_root=source_root,
        destination_root=destination_root,
        model="deepseek/deepseek-v4-pro",
        base_url="https://api.qnaigc.com/v1",
        max_supervisor_retries=10,
        max_total_supervisor_repairs=60,
    )

    assert args["output"].startswith(str(destination_root))
    assert args["reference_papers"] == [
        str(destination_root / "inputs" / "corpus" / "paper.md")
    ]
    assert args["max_supervisor_retries"] == 10
    assert args["max_total_supervisor_repairs"] == 60
    assert args["max_retries"] == 10
    assert args["api_key"] == "[REDACTED]"
    assert args["human_advice"] == ""


def test_checkpoint_fork_preserves_prior_schema_when_rewinding_to_query_semantics(
    tmp_path,
):
    source_path = tmp_path / "source.state.json"
    source_path.write_text("{}", encoding="utf-8")
    state = {
        "args": {"output": "old.json", "checkpoint_output": "old.state.json"},
        "module_outputs": {
            "locating_module": {"database_goal": "materials database"},
            "mechanism_requirement_module": {"must_have_concepts": ["property"]},
            "schema_design_module": {
                "field_registry": [
                    {"field_path": "material_info.section1.property"},
                    {"field_path": "material_info.section1.property_criterion"},
                ]
            },
        },
        "run_identity": {"run_id": "source-run"},
        "schema_revision": 7,
    }

    prepared = prepare_forked_resume_state(
        state,
        "query_semantics",
        args={"output": "new.json", "checkpoint_output": "new.state.json"},
        run_identity={"run_id": "destination-run"},
        source_state_path=source_path,
        shared_context={"query_requirements": ["property"]},
    )

    assert set(prepared["module_outputs"]) == {
        "locating_module",
        "mechanism_requirement_module",
    }
    assert prepared["resume_prior_schema_candidate"]["field_count"] == 2
    assert prepared["resume_prior_schema_candidate"]["source_schema_revision"] == 7
    assert prepared["shared_context"]["checkpoint_prior_schema"]["field_paths"] == [
        "material_info.section1.property",
        "material_info.section1.property_criterion",
    ]
    assert prepared["checkpoint_fork"]["preserved_prior_schema_candidate"] is True
