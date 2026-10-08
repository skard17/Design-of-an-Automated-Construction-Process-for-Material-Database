#!/usr/bin/env python3
"""Create a clean Step8 resume checkpoint at a selected graph node."""

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import run_artifact_guard


MODULE_ORDER = [
    ("locating", "locating_module"),
    ("mechanism", "mechanism_requirement_module"),
    ("query_semantics", "query_semantics_module"),
    ("evidence_model", "evidence_model_module"),
    ("subjective_supervisor", "subjective_supervisor_module"),
    ("topic_adaptation", "topic_adaptation_module"),
    ("section_partition", "section_partition_module"),
    ("field_planning", "field_planning_module"),
    ("supervisor", "supervisor_module"),
    ("figure_classification", "figure_classification_module"),
    ("schema_design", "schema_design_module"),
    ("specialization_critic", "specialization_critic_module"),
    ("aggregation", "aggregation_module"),
]
NODE_INDEX = {node: index for index, (node, _) in enumerate(MODULE_ORDER)}
SPECIAL_RESTART_NODES = {"supervisor_router"}


def _prepare_supervisor_router_resume(state):
    critic = (state.get("module_outputs") or {}).get("specialization_critic_module") or {}
    operations = critic.get("patch_operations") or []
    try:
        reviewed_revision = int(critic.get("reviewed_schema_revision"))
        schema_revision = int(state.get("schema_revision", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "supervisor-router resume requires a critic reviewed against the current schema"
        ) from exc
    if not isinstance(operations, list) or not operations or reviewed_revision != schema_revision:
        raise ValueError(
            "supervisor-router resume requires a non-empty current-revision critic patch"
        )
    feedback = [
        str(item).strip()
        for item in (
            state.get("repair_instructions")
            or state.get("validation_errors")
            or []
        )
        if str(item).strip()
    ]
    if not feedback:
        raise ValueError("supervisor-router resume requires persisted validation feedback")

    keep_modules = {
        module_name
        for _, module_name in MODULE_ORDER[: NODE_INDEX["aggregation"]]
    }
    prepared = copy.deepcopy(state)
    for key in ("module_outputs", "module_attempts", "module_errors"):
        prepared[key] = {
            name: value
            for name, value in (state.get(key) or {}).items()
            if name in keep_modules
        }
    prepared.update(
        {
            "validation_errors": feedback,
            "repair_instructions": feedback,
            "status": "running",
            "current_node": "specialization_critic",
            "next_node": "supervisor_router",
            "supervisor_decision": {},
            "last_error_node": "schema_design",
            "last_error_type": "schema_validation_failure",
            "retry_counts": {},
            "blocker_history": [],
            "rejected_critic_patch_hashes": [],
            "result": None,
        }
    )
    prepared.pop("output", None)
    return prepared, keep_modules


def prepare_resume_state(state, restart_node, run_output=None, checkpoint_output=None):
    if restart_node not in NODE_INDEX and restart_node not in SPECIAL_RESTART_NODES:
        raise ValueError(f"unsupported restart node: {restart_node}")
    run_identity = state.get("run_identity")
    if not isinstance(run_identity, dict) or not run_identity.get("run_id"):
        raise ValueError(
            "legacy state has no clean-run identity; start a fresh run instead of relabeling old artifacts"
        )
    prior_blocker_history = list(state.get("blocker_history") or [])
    prior_schema = copy.deepcopy(
        (state.get("module_outputs") or {}).get("schema_design_module")
    )
    if restart_node == "supervisor_router":
        prepared, keep_modules = _prepare_supervisor_router_resume(state)
    else:
        keep_modules = {
            module_name for _, module_name in MODULE_ORDER[: NODE_INDEX[restart_node]]
        }
        prepared = copy.deepcopy(state)
        prepared["module_outputs"] = {
            key: value
            for key, value in (state.get("module_outputs") or {}).items()
            if key in keep_modules
        }
        prepared["module_attempts"] = {
            key: value
            for key, value in (state.get("module_attempts") or {}).items()
            if key in keep_modules
        }
        prepared["module_errors"] = {
            key: value
            for key, value in (state.get("module_errors") or {}).items()
            if key in keep_modules
        }
        prepared["validation_errors"] = []
        prepared["status"] = "running"
        prepared["current_node"] = MODULE_ORDER[NODE_INDEX[restart_node] - 1][0] if keep_modules else "prepare"
        prepared["next_node"] = restart_node
        prepared["supervisor_decision"] = {}
        prepared["last_error_node"] = ""
        prepared["last_error_type"] = ""
        prepared["retry_counts"] = {}
        prepared["repair_instructions"] = []
        prepared["blocker_history"] = []
        prepared["rejected_critic_patch_hashes"] = []
        if NODE_INDEX[restart_node] <= NODE_INDEX["schema_design"]:
            prepared["schema_revision"] = 0
            prepared["critic_reviewed_schema_revision"] = -1
        elif NODE_INDEX[restart_node] <= NODE_INDEX["specialization_critic"]:
            prepared["critic_reviewed_schema_revision"] = -1
        prepared["result"] = None
        prepared.pop("output", None)
    if (
        isinstance(prior_schema, dict)
        and prior_schema.get("field_registry")
        and restart_node != "supervisor_router"
        and NODE_INDEX[restart_node] <= NODE_INDEX["schema_design"]
    ):
        prepared["resume_prior_schema_candidate"] = {
            "source_schema_revision": int(state.get("schema_revision", 0) or 0),
            "field_count": len(prior_schema.get("field_registry") or []),
            "schema_design_module": prior_schema,
            "policy": (
                "checkpoint candidate only; rerun affected modules, then preserve each valid "
                "leaf or record an explicit merge, replacement, or removal rationale"
            ),
        }
    args = copy.deepcopy(prepared.get("args") or {})
    current_output = str(args.get("output") or "")
    current_checkpoint = str(args.get("checkpoint_output") or "")
    if run_output and str(run_output) != current_output:
        raise ValueError("run_output cannot change during an explicit resume")
    if checkpoint_output and str(checkpoint_output) != current_checkpoint:
        raise ValueError("checkpoint_output cannot change during an explicit resume")
    args["stop_after_module"] = ""
    prepared["args"] = args
    prepared["resume_preparation"] = {
        "restart_node": restart_node,
        "kept_modules": sorted(keep_modules),
        "prior_blocker_history_count": len(prior_blocker_history),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    return prepared


def prepare_forked_resume_state(
    state,
    restart_node,
    *,
    args,
    run_identity,
    source_state_path,
    shared_context,
):
    prepared = prepare_resume_state(state, restart_node)
    active_repair_feedback = (
        list(prepared.get("repair_instructions") or [])
        if restart_node == "supervisor_router"
        else []
    )
    if (
        restart_node == "specialization_critic"
        and state.get("current_node") == "supervisor_router"
        and state.get("next_node") == "specialization_critic"
        and state.get("last_error_node") == "specialization_critic"
    ):
        active_repair_feedback = [
            str(item).strip()
            for item in state.get("repair_instructions") or []
            if str(item).strip()
        ]
    prepared["repair_instructions"] = active_repair_feedback
    source_path = Path(source_state_path).resolve(strict=True)
    source_run_id = str((state.get("run_identity") or {}).get("run_id") or "")
    destination_run_id = str(run_identity.get("run_id") or "")
    if not source_run_id or not destination_run_id:
        raise ValueError("checkpoint fork requires source and destination run identities")
    if source_run_id == destination_run_id:
        raise ValueError("checkpoint fork must reserve a distinct destination run identity")
    prepared["args"] = copy.deepcopy(args)
    prepared["args"]["api_key"] = "[REDACTED]"
    prepared["shared_context"] = copy.deepcopy(shared_context)
    prior_candidate = prepared.get("resume_prior_schema_candidate") or {}
    prior_schema_module = prior_candidate.get("schema_design_module") or {}
    if prior_schema_module:
        prepared["shared_context"]["checkpoint_prior_schema"] = {
            "source_schema_revision": prior_candidate.get("source_schema_revision"),
            "field_count": prior_candidate.get("field_count"),
            "field_paths": [
                str(field.get("field_path"))
                for field in prior_schema_module.get("field_registry") or []
                if isinstance(field, dict) and field.get("field_path")
            ],
            "policy": prior_candidate.get("policy"),
        }
    prepared["human_advice"] = str(shared_context.get("human_advice") or "")
    prepared["run_identity"] = copy.deepcopy(run_identity)
    prepared["checkpoint_fork"] = {
        "contract_version": "step8-checkpoint-fork/v1",
        "source_state": str(source_path),
        "source_state_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_run_id": source_run_id,
        "destination_run_id": destination_run_id,
        "restart_node": restart_node,
        "kept_modules": prepared["resume_preparation"]["kept_modules"],
        "source_artifacts_mutated": False,
        "cross_version_artifacts_used": False,
        "preserved_active_repair_feedback_count": len(active_repair_feedback),
        "preserved_prior_schema_candidate": bool(prior_schema_module),
        "preserved_prior_schema_field_count": int(
            prior_candidate.get("field_count", 0) or 0
        ),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    return prepared


def remap_fork_args(
    state_args,
    *,
    source_root,
    destination_root,
    model,
    base_url,
    max_supervisor_retries,
    max_total_supervisor_repairs,
):
    source_root = Path(source_root).resolve(strict=True)
    destination_root = Path(destination_root).resolve(strict=True)

    def remap_path(raw_path):
        path = Path(raw_path).resolve(strict=False)
        try:
            relative = path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(
                f"checkpoint input path escapes source namespace: {path}"
            ) from exc
        return str(destination_root / relative)

    args = copy.deepcopy(state_args)
    args["key_description_path"] = remap_path(args["key_description_path"])
    args["reference_papers"] = [
        remap_path(path) for path in list(args.get("reference_papers") or [])
    ]
    if args.get("human_advice_path"):
        args["human_advice_path"] = remap_path(args["human_advice_path"])
        # The destination advice file is an immutable fork input and must win
        # over any stale advice text cached inside the source checkpoint args.
        args["human_advice"] = ""
    args["output"] = remap_path(args["output"])
    args["checkpoint_output"] = remap_path(args["checkpoint_output"])
    args.update(
        {
            "model": str(model),
            "base_url": str(base_url),
            "api_key": "[REDACTED]",
            "llm_backend": "qiniu",
            "request_timeout": 1200.0,
            "max_retries": 10,
            "max_supervisor_retries": int(max_supervisor_retries),
            "max_total_supervisor_repairs": int(max_total_supervisor_repairs),
            "stop_after_module": "",
        }
    )
    required_paths = [
        args["key_description_path"],
        *args["reference_papers"],
    ]
    if args.get("human_advice_path"):
        required_paths.append(args["human_advice_path"])
    missing = [path for path in required_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"fork destination inputs are missing: {missing}")
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--restart-node",
        required=True,
        choices=sorted(set(NODE_INDEX) | SPECIAL_RESTART_NODES),
    )
    parser.add_argument("--run-output")
    parser.add_argument("--checkpoint-output")
    parser.add_argument("--fork-source-root")
    parser.add_argument("--fork-destination-root")
    parser.add_argument("--fork-model")
    parser.add_argument("--fork-base-url")
    parser.add_argument("--max-supervisor-retries", type=int, default=10)
    parser.add_argument("--max-total-supervisor-repairs", type=int, default=60)
    args = parser.parse_args()

    source = Path(args.state)
    destination = Path(args.output)
    state = json.loads(source.read_text(encoding="utf-8"))
    run_artifact_guard.assert_active_run(state.get("run_identity"))
    fork_values = (
        args.fork_source_root,
        args.fork_destination_root,
        args.fork_model,
        args.fork_base_url,
    )
    if any(fork_values) and not all(fork_values):
        raise ValueError(
            "checkpoint fork requires source root, destination root, model, and base URL"
        )
    if all(fork_values):
        import section_design_langgraph_human_gate as graph

        fork_args = remap_fork_args(
            state.get("args") or {},
            source_root=args.fork_source_root,
            destination_root=args.fork_destination_root,
            model=args.fork_model,
            base_url=args.fork_base_url,
            max_supervisor_retries=args.max_supervisor_retries,
            max_total_supervisor_repairs=args.max_total_supervisor_repairs,
        )
        if destination.resolve(strict=False) != Path(
            fork_args["checkpoint_output"]
        ).resolve(strict=False):
            raise ValueError("fork output must equal the remapped checkpoint path")
        shared_context, human_advice = graph.build_shared_context_from_args(fork_args)
        fork_args["human_advice"] = human_advice
        input_identity = graph.build_graph_step8_input_identity(fork_args)
        output_path = Path(fork_args["output"])
        human_context_path = graph.human_gate_context_path_from_args(
            graph.dict_to_namespace(fork_args)
        )
        run_identity = run_artifact_guard.reserve_fresh_run(
            pipeline="step8_section_design_langgraph",
            output_path=output_path,
            input_identity=input_identity,
            artifact_paths=[output_path, destination, human_context_path],
        )
        prepared = prepare_forked_resume_state(
            state,
            args.restart_node,
            args=fork_args,
            run_identity=run_identity,
            source_state_path=source,
            shared_context=shared_context,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        run_artifact_guard.atomic_write_json(
            destination,
            prepared,
            run_identity=run_identity,
        )
        run_artifact_guard.update_run_status(
            run_identity,
            "interrupted",
            pause_reason="checkpoint_fork_prepared",
            source_run_id=prepared["checkpoint_fork"]["source_run_id"],
            source_state_sha256=prepared["checkpoint_fork"]["source_state_sha256"],
        )
    else:
        prepared = prepare_resume_state(
            state,
            args.restart_node,
            run_output=args.run_output,
            checkpoint_output=args.checkpoint_output,
        )
        prepared["resume_preparation"]["source_state"] = str(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(prepared, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "output": str(destination),
                "restart_node": args.restart_node,
                "kept_modules": prepared["resume_preparation"]["kept_modules"],
            }
        )
    )


if __name__ == "__main__":
    main()
