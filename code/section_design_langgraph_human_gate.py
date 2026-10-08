import argparse
import hashlib
import json
from task_domain_knowledge import attach_to_context
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import section_design_agent as section_agent
import run_artifact_guard
from field_definition_contract import apply_definitions
from section_design_agent_prompt import (
    STEP8_FRAMEWORK,
    build_aggregation_prompt,
    build_evidence_model_prompt,
    build_field_planning_prompt,
    build_figure_classification_prompt,
    build_locating_prompt,
    build_mechanism_requirement_prompt,
    build_query_semantics_prompt,
    build_schema_design_prompt,
    build_section_partition_prompt,
    build_shared_context_block,
    build_specialization_critic_prompt,
    build_supervisor_prompt,
    build_subjective_supervisor_prompt,
    build_topic_adaptation_prompt,
)


def print_console_safe(value):
    """Print dynamic CLI output without failing on narrow Windows code pages."""
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="backslashreplace").decode(encoding))


class SectionDesignGraphState(TypedDict, total=False):
    args: dict[str, Any]
    shared_context: dict[str, Any]
    module_outputs: dict[str, Any]
    module_attempts: dict[str, Any]
    module_errors: dict[str, Any]
    validation_errors: list[str]
    human_advice: str
    post_design_human_advice: str
    applied_human_advice: str
    human_review_round: int
    human_review_applied: bool
    allow_local_critic_bypass_after_human_advice: bool
    result: dict[str, Any]
    status: str
    output: str
    current_node: str
    next_node: str
    supervisor_decision: dict[str, Any]
    last_error_node: str
    last_error_type: str
    retry_counts: dict[str, int]
    repair_instructions: list[str]
    force_deterministic_schema_compilation: bool
    schema_revision: int
    critic_reviewed_schema_revision: int
    blocker_history: list[dict[str, Any]]
    schema_iteration_history: list[dict[str, Any]]
    schema_inspection_approved_through: int
    schema_inspection_resume_node: str
    rejected_critic_patch_hashes: list[str]
    run_identity: dict[str, Any]


def namespace_to_dict(args):
    return vars(args).copy()


def dict_to_namespace(values):
    return argparse.Namespace(**values)


GRAPH_SEQUENCE = [
    "prepare",
    "supervisor_router",
    "locating",
    "mechanism",
    "query_semantics",
    "evidence_model",
    "subjective_supervisor",
    "topic_adaptation",
    "section_partition",
    "field_planning",
    "human_advice_gate",
    "supervisor",
    "figure_classification",
    "schema_design",
    "schema_design_repair",
    "specialization_critic",
    "aggregation",
    "write_output",
]

NODE_MODULE_OUTPUTS = {
    "prepare": "__shared_context__",
    "supervisor_router": "__supervisor_decision__",
    "locating": "locating_module",
    "mechanism": "mechanism_requirement_module",
    "query_semantics": "query_semantics_module",
    "evidence_model": "evidence_model_module",
    "subjective_supervisor": "subjective_supervisor_module",
    "topic_adaptation": "topic_adaptation_module",
    "section_partition": "section_partition_module",
    "field_planning": "field_planning_module",
    "human_advice_gate": "__human_gate__",
    "supervisor": "supervisor_module",
    "figure_classification": "figure_classification_module",
    "schema_design": "schema_design_module",
    "schema_design_repair": "schema_design_module",
    "specialization_critic": "specialization_critic_module",
    "aggregation": "aggregation",
    "write_output": "__result__",
}


def state_path_from_args(args):
    if getattr(args, "checkpoint_output", ""):
        return Path(args.checkpoint_output)
    output_path = Path(args.output)
    return output_path.with_suffix(output_path.suffix + ".state.json")


def human_gate_context_path_from_args(args):
    output_path = Path(args.output)
    return output_path.with_suffix(output_path.suffix + ".human_gate_context.json")


def schema_history_dir_from_args(args):
    snapshot_path = state_path_from_args(args)
    return snapshot_path.parent / f"{snapshot_path.stem}.schema_history"


def event_history_dir_from_args(args):
    snapshot_path = state_path_from_args(args)
    return snapshot_path.parent / f"{snapshot_path.stem}.events"


def schema_iteration_log_path_from_args(args):
    snapshot_path = state_path_from_args(args)
    return snapshot_path.parent / f"{snapshot_path.stem}.schema_iterations.json"


def json_safe_state(state):
    def clean(value, key_name=""):
        if key_name == "api_key" and value:
            return "[REDACTED]"
        if key_name == "module_attempts" and isinstance(value, dict):
            compact = {}
            for module_name, attempts in value.items():
                if isinstance(attempts, list) and attempts:
                    compact[module_name] = [clean(attempts[-1], "attempt")]
                else:
                    compact[module_name] = clean(attempts, "attempt")
            return compact
        if isinstance(value, dict):
            return {str(key): clean(item, str(key)) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item, key_name) for item in value]
        if isinstance(value, str):
            limit = (
                2000000
                if key_name
                in {
                    "prompt",
                    "raw_response",
                    "reference_paper_context",
                    "shared_context_block",
                }
                else 500000
            )
            if len(value) > limit:
                return value[:limit] + "\n...[truncated in state snapshot]..."
            return value
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)

    return clean(state)


SCHEMA_ITERATION_NODES = {
    "field_planning",
    "schema_design",
    "schema_design_repair",
    "specialization_critic",
    "aggregation",
    "supervisor_router",
    "human_advice_gate",
}


def _schema_field_map(schema):
    return {
        str(item.get("field_path")): item
        for item in (schema or {}).get("field_registry") or []
        if isinstance(item, dict) and item.get("field_path")
    }


def _schema_digest(schema):
    return hashlib.sha256(
        json.dumps(
            json_safe_state(schema or {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _critic_iteration_summary(critic):
    if not isinstance(critic, dict):
        return None
    operations = []
    for item in critic.get("patch_operations") or []:
        if not isinstance(item, dict):
            continue
        operations.append(
            {
                "op": item.get("op"),
                "field_path": item.get("field_path"),
                "target_path": item.get("target_path"),
                "reason": item.get("reason"),
            }
        )
    return json_safe_state(
        {
            "reviewed_schema_revision": critic.get("reviewed_schema_revision"),
            "redo_needed": critic.get("redo_needed"),
            "is_generic": critic.get("is_generic"),
            "missing_concepts": critic.get("missing_concepts") or [],
            "redo_directives": critic.get("redo_directives") or [],
            "structural_weaknesses": critic.get("structural_weaknesses") or [],
            "field_utility_audit": critic.get("field_utility_audit") or {},
            "patch_operations": operations,
        }
    )


def _previous_schema_for_revision(args, revision):
    history_dir = schema_history_dir_from_args(args)
    candidates = []
    paths = history_dir.glob("revision_*.json") if history_dir.exists() else []
    for path in paths:
        match = re.match(r"revision_(\d{6})(?:_[0-9a-f]+)?\.json$", path.name)
        if match and int(match.group(1)) < revision:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None, None
    previous_revision, previous_path = max(candidates, key=lambda item: item[0])
    try:
        payload = json.loads(previous_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    return previous_revision, payload.get("schema")


def append_schema_iteration_record(state, current_node, next_node=None):
    if current_node not in SCHEMA_ITERATION_NODES:
        return state
    snapshot = deepcopy(state)
    args = dict_to_namespace(snapshot["args"])
    schema = (snapshot.get("module_outputs") or {}).get("schema_design_module") or {}
    revision = int(snapshot.get("schema_revision", 0) or 0)
    current_fields = _schema_field_map(schema)
    previous_revision, previous_schema = _previous_schema_for_revision(args, revision)
    previous_fields = _schema_field_map(previous_schema)
    added_paths = sorted(set(current_fields) - set(previous_fields)) if previous_schema else []
    removed_paths = sorted(set(previous_fields) - set(current_fields)) if previous_schema else []
    updated_paths = []
    if previous_schema:
        for path in sorted(set(current_fields) & set(previous_fields)):
            if _schema_digest({"field_registry": [current_fields[path]]}) != _schema_digest(
                {"field_registry": [previous_fields[path]]}
            ):
                updated_paths.append(path)

    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "schema_revision": revision,
        "schema_sha256": _schema_digest(schema) if schema else None,
        "field_count": len(current_fields),
        "previous_schema_revision": previous_revision,
        "current_node": current_node,
        "next_node": next_node or snapshot.get("next_node"),
        "status": snapshot.get("status"),
        "changes": {
            "added_count": len(added_paths),
            "removed_count": len(removed_paths),
            "updated_count": len(updated_paths),
            "added_paths": added_paths,
            "removed_paths": removed_paths,
            "updated_paths": updated_paths,
        },
        "critic_review": _critic_iteration_summary(
            (snapshot.get("module_outputs") or {}).get("specialization_critic_module")
        )
        if current_node == "specialization_critic"
        else None,
        "supervisor_decision": json_safe_state(snapshot.get("supervisor_decision"))
        if current_node == "supervisor_router"
        else None,
        "validation_errors": json_safe_state(snapshot.get("validation_errors") or []),
        "repair_instructions": json_safe_state(snapshot.get("repair_instructions") or []),
    }
    digest_payload = deepcopy(record)
    digest_payload.pop("recorded_at", None)
    record_id = hashlib.sha256(
        json.dumps(
            digest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    record["record_id"] = record_id
    history = [
        item
        for item in deepcopy(snapshot.get("schema_iteration_history") or [])
        if isinstance(item, dict)
    ]
    if any(item.get("record_id") == record_id for item in history):
        return snapshot
    history.append(record)
    snapshot["schema_iteration_history"] = history
    run_artifact_guard.atomic_write_json(
        schema_iteration_log_path_from_args(args),
        {
            "artifact_type": "step8_schema_iteration_history",
            "run_identity": json_safe_state(snapshot.get("run_identity")),
            "entries": history,
        },
        run_identity=snapshot.get("run_identity"),
    )
    return snapshot


def build_schema_iteration_context(state):
    history = [
        item
        for item in state.get("schema_iteration_history") or []
        if isinstance(item, dict)
    ]
    if not history:
        return ""
    milestone_ids = {
        item.get("record_id")
        for item in history
        if int(item.get("schema_revision", 0) or 0) > 0
        and int(item.get("schema_revision", 0) or 0) % 10 == 0
    }
    selected = [item for item in history if item.get("record_id") in milestone_ids]
    selected.extend(history[-12:])
    unique = []
    seen = set()
    for item in selected:
        record_id = item.get("record_id")
        if record_id in seen:
            continue
        seen.add(record_id)
        unique.append(item)
    return (
        "SCHEMA_ITERATION_HISTORY (authoritative prior decisions; avoid reversing an earlier "
        "change unless you explicitly resolve the recorded conflict):\n"
        + json.dumps(unique, ensure_ascii=False, separators=(",", ":"))
    )


def write_schema_revision_snapshot(state, current_node, next_node=None):
    revision = int(state.get("schema_revision", 0) or 0)
    schema = (state.get("module_outputs") or {}).get("schema_design_module")
    if revision <= 0 or not isinstance(schema, dict) or not schema.get("field_registry"):
        return None

    args = dict_to_namespace(state["args"])
    safe_schema = json_safe_state(schema)
    schema_digest = hashlib.sha256(
        json.dumps(
            safe_schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    history_dir = schema_history_dir_from_args(args)
    target = history_dir / f"revision_{revision:06d}.json"
    payload = {
        "artifact_type": "step8_schema_revision",
        "schema_revision": revision,
        "parent_schema_revision": revision - 1 if revision > 1 else None,
        "schema_sha256": schema_digest,
        "field_count": len(schema.get("field_registry") or []),
        "current_node": current_node,
        "next_node": next_node or state.get("next_node"),
        "critic_reviewed_schema_revision": int(
            state.get("critic_reviewed_schema_revision", -1) or -1
        ),
        "supervisor_decision": json_safe_state(state.get("supervisor_decision")),
        "run_identity": json_safe_state(state.get("run_identity")),
        "schema": safe_schema,
    }

    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("schema_sha256") == schema_digest:
            return target
        target = history_dir / f"revision_{revision:06d}_{schema_digest[:12]}.json"
        if target.exists():
            return target

    run_artifact_guard.atomic_write_json(
        target,
        payload,
        run_identity=state.get("run_identity"),
    )
    return target


def write_transition_snapshot(state, current_node, next_node=None):
    args = dict_to_namespace(state["args"])
    module_name = NODE_MODULE_OUTPUTS.get(current_node)
    outputs = state.get("module_outputs") or {}
    attempts = state.get("module_attempts") or {}
    module_errors = state.get("module_errors") or {}
    latest_attempt = None
    if module_name and isinstance(attempts.get(module_name), list) and attempts[module_name]:
        latest_attempt = attempts[module_name][-1]

    if module_name == "__shared_context__":
        module_output = json_safe_state(state.get("shared_context"))
    elif module_name == "__supervisor_decision__":
        module_output = json_safe_state(state.get("supervisor_decision"))
    elif module_name == "__human_gate__":
        module_output = {
            "human_advice": json_safe_state(state.get("human_advice")),
            "post_design_human_advice": json_safe_state(
                state.get("post_design_human_advice")
            ),
        }
    elif module_name == "__result__":
        module_output = json_safe_state(state.get("result"))
    elif module_name == "schema_design_module":
        module_output = None
    else:
        module_output = json_safe_state(outputs.get(module_name)) if module_name else None

    payload = {
        "artifact_type": "step8_transition",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "current_node": current_node,
        "next_node": next_node or state.get("next_node"),
        "status": state.get("status"),
        "schema_revision": int(state.get("schema_revision", 0) or 0),
        "critic_reviewed_schema_revision": int(
            state.get("critic_reviewed_schema_revision", -1) or -1
        ),
        "module_name": module_name,
        "module_output": module_output,
        "schema_history_reference": (
            f"revision_{int(state.get('schema_revision', 0) or 0):06d}.json"
            if module_name == "schema_design_module"
            else None
        ),
        "latest_module_attempt": json_safe_state(latest_attempt),
        "module_errors": json_safe_state(module_errors.get(module_name))
        if module_name and not module_name.startswith("__")
        else None,
        "validation_errors": json_safe_state(state.get("validation_errors") or []),
        "repair_instructions": json_safe_state(state.get("repair_instructions") or []),
        "supervisor_decision": json_safe_state(state.get("supervisor_decision")),
        "blocker_history_length": len(state.get("blocker_history") or []),
        "latest_blocker": json_safe_state(
            (state.get("blocker_history") or [None])[-1]
        ),
        "human_advice": json_safe_state(state.get("human_advice")),
        "post_design_human_advice": json_safe_state(
            state.get("post_design_human_advice")
        ),
        "run_identity": json_safe_state(state.get("run_identity")),
    }
    digest_payload = deepcopy(payload)
    digest_payload.pop("recorded_at", None)
    event_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload["event_sha256"] = event_digest
    history_dir = event_history_dir_from_args(args)
    target = history_dir / f"e_{event_digest[:16]}.json"
    collision_index = 1
    while target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("event_sha256") == event_digest:
            return target
        collision_index += 1
        target = history_dir / f"e_{event_digest[:16]}_{collision_index}.json"
    run_artifact_guard.atomic_write_json(
        target,
        payload,
        run_identity=state.get("run_identity"),
    )
    return target


def write_state_snapshot(state, current_node, next_node=None):
    args = dict_to_namespace(state["args"])
    snapshot = deepcopy(state)
    snapshot["current_node"] = current_node
    if next_node:
        snapshot["next_node"] = next_node
    snapshot = append_schema_iteration_record(snapshot, current_node, next_node)
    snapshot_path = state_path_from_args(args)
    write_schema_revision_snapshot(snapshot, current_node, next_node)
    write_transition_snapshot(snapshot, current_node, next_node)
    run_artifact_guard.atomic_write_json(
        snapshot_path,
        json_safe_state(snapshot),
        run_identity=state.get("run_identity"),
    )
    return snapshot


def load_state_snapshot(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_shared_context_from_args(args_dict):
    args = dict_to_namespace(args_dict)
    base_query_requirements = section_agent.load_query_requirements(args.query_requirements)
    key_description_text = section_agent.load_key_description_text(
        args.key_description_path
    )
    reference_field_contract = section_agent.build_reference_field_contract(
        key_description_text,
        args.key_description_path,
    )
    reference_papers = list(args.reference_papers or [])
    reference_paper_context = section_agent.load_reference_paper_context(
        reference_papers
    )
    human_advice = args.human_advice or section_agent.load_optional_text(
        args.human_advice_path
    )
    care_queries = section_agent.explicit_human_advice_counterfactual_queries(
        human_advice
    )
    query_requirements = list(base_query_requirements)
    care_requirement_records = []
    for care_query in care_queries:
        distinctions = ", ".join(care_query.get("required_distinctions") or [])
        requirement_text = str(care_query["query"])
        if distinctions:
            requirement_text += f" Required distinctions: {distinctions}."
        query_requirements.append(requirement_text)
        care_requirement_records.append(
            {
                **care_query,
                "requirement_id": f"query_{len(query_requirements)}",
                "requirement_text": requirement_text,
            }
        )
    shared_context = {
        "task_contract": dict(section_agent.MATERIAL_LITERATURE_TASK_CONTRACT),
        "database_goal": args.database_goal,
        "discipline": args.discipline,
        "query_requirements": query_requirements,
        "base_query_requirement_count": len(base_query_requirements),
        "care_counterfactual_enabled": bool(care_requirement_records),
        "care_counterfactual_queries": care_requirement_records,
        "care_counterfactual_requirement_ids": [
            item["requirement_id"] for item in care_requirement_records
        ],
        "key_description_text": key_description_text,
        "reference_field_contract": reference_field_contract,
        "reference_paper_context": reference_paper_context,
        "reference_paper_count": len(reference_papers),
        "human_advice": human_advice,
        "human_advice_available_before_design": bool(
            str(human_advice or "").strip()
        ),
        "structured_protocol_enabled": bool(
            getattr(args, "structured_protocol", False)
        ),
        "structured_concept_seeding_only": bool(
            getattr(args, "structured_protocol", False)
        ),
        "shared_context_block": build_shared_context_block(
            args.database_goal,
            args.discipline,
            query_requirements,
            key_description_text,
            reference_paper_context,
        ),
    }
    attach_to_context(shared_context, getattr(args, "domain_knowledge_pack", ""))
    return shared_context, human_advice


def migrate_legacy_modular_checkpoint(snapshot, cli_args):
    inputs = deepcopy(snapshot.get("inputs") or {})
    args = deepcopy(cli_args)
    for key in ("database_goal", "discipline", "query_requirements"):
        if inputs.get(key):
            args[key] = inputs[key]
    if not args.get("key_description_path"):
        args["key_description_path"] = inputs.get("key_description_path", "")
    if not args.get("reference_papers"):
        args["reference_papers"] = list(inputs.get("reference_papers") or [])

    shared_context, human_advice = build_shared_context_from_args(args)
    args["human_advice"] = human_advice
    args["human_advice_path"] = ""
    return {
        "args": args,
        "shared_context": shared_context,
        "module_outputs": deepcopy(snapshot.get("module_outputs") or {}),
        "module_attempts": {},
        "module_errors": deepcopy(snapshot.get("module_errors") or {}),
        "validation_errors": [],
        "human_advice": human_advice,
        "status": "running",
        "next_node": "schema_design",
        "force_deterministic_schema_compilation": True,
        "retry_counts": {},
        "repair_instructions": [],
        "schema_revision": 0,
        "critic_reviewed_schema_revision": -1,
        "blocker_history": [],
        "rejected_critic_patch_hashes": [],
    }


def write_human_gate_context(state):
    args = dict_to_namespace(state["args"])
    outputs = state.get("module_outputs", {})
    shared = state.get("shared_context", {})
    context = {
        "status": "waiting_for_human_advice",
        "gate": "after_internal_field_design_pass",
        "next_node_after_advice": "supervisor_router",
        "state_path": str(state_path_from_args(args)),
        "reference_papers": args.reference_papers,
        "reference_paper_context_preview": shared.get("reference_paper_context", "")[:4000],
        "available_context": {
            "task_contract": {
                "database_goal": shared.get("database_goal"),
                "discipline": shared.get("discipline"),
                "query_requirements": shared.get("query_requirements") or [],
                "reference_paper_count": shared.get("reference_paper_count"),
            },
            "section_partition_module": outputs.get("section_partition_module"),
            "field_planning_module": outputs.get("field_planning_module"),
            "schema_design_module": outputs.get("schema_design_module"),
            "specialization_critic_module": outputs.get("specialization_critic_module"),
            "aggregation": outputs.get("aggregation"),
            "supervisor_decision": state.get("supervisor_decision"),
            "schema_revision": state.get("schema_revision", 0),
            "schema_iteration_history": state.get("schema_iteration_history") or [],
        },
        "required_advice": (
            "Review the internally accepted field design as a materials-domain expert. "
            "If it is not acceptable, describe only domain/schema problems in the current output; "
            "do not provide code or prompt implementation instructions. If acceptable, say it can pass."
        ),
    }
    context_path = human_gate_context_path_from_args(args)
    run_artifact_guard.atomic_write_json(
        context_path,
        json_safe_state(context),
        run_identity=state.get("run_identity"),
    )
    return str(context_path), context


def merge_update(state, update):
    merged = deepcopy(state)
    merged.update(update or {})
    return merged


def clear_downstream_outputs(state, module_names):
    """Drop downstream artifacts that are invalid after an upstream repair."""
    update = {}
    for key in ("module_outputs", "module_attempts", "module_errors"):
        values = deepcopy(state.get(key, {}))
        for module_name in module_names:
            values.pop(module_name, None)
        update[key] = values
    update["result"] = None
    return merge_update(state, update)


def interactive_resume_node(state):
    next_node = state.get("next_node") or "prepare"
    if next_node in {"write_output", "__end__"}:
        return "write_output"
    return next_node


def classify_errors(errors):
    text = "\n".join(str(item) for item in (errors or [])).lower()
    if not text:
        return ""
    if (
        "invalid json response" in text
        or "unterminated string" in text
        or "expecting value" in text
        or "extra data" in text
        or "jsondecodeerror" in text
    ):
        return "json_parse_failure"
    if "timeout" in text or "timed out" in text or "readtimeout" in text:
        return "request_timeout"
    if (
        "field_registry" in text
        or "missing" in text
        or "validation" in text
        or "schema_design_module" in text
        or "not approved by the step 8 section plan" in text
        or "unapproved schema" in text
        or "materials-database fields" in text
    ):
        return "schema_validation_failure"
    return "module_failure"


def supervisor_next_node(state):
    decision = state.get("supervisor_decision") or {}
    return decision.get("next_node", "write_output")


def schema_inspection_summary(state, start_revision, end_revision):
    entries = [
        item
        for item in state.get("schema_iteration_history") or []
        if isinstance(item, dict)
        and start_revision < int(item.get("schema_revision", 0) or 0) <= end_revision
    ]
    critic_entries = [item for item in entries if item.get("critic_review")]
    redesign_count = sum(
        bool((item.get("critic_review") or {}).get("redo_needed"))
        or bool((item.get("critic_review") or {}).get("is_generic"))
        for item in critic_entries
    )
    patch_rejections = sum(
        any(
            str(error).startswith("schema_patch_")
            for error in item.get("validation_errors") or []
        )
        for item in entries
    )
    blocker_counts = {}
    for item in entries:
        decision = item.get("supervisor_decision") or {}
        fingerprint = str(decision.get("blocker_fingerprint") or "")
        if fingerprint:
            blocker_counts[fingerprint] = blocker_counts.get(fingerprint, 0) + 1
    max_churn_ratio = 0.0
    for item in entries:
        changes = item.get("changes") or {}
        field_count = max(int(item.get("field_count", 0) or 0), 1)
        churn = int(changes.get("added_count", 0) or 0) + int(
            changes.get("removed_count", 0) or 0
        )
        max_churn_ratio = max(max_churn_ratio, churn / field_count)
    critic_count = len(critic_entries)
    return {
        "revision_window": [start_revision + 1, end_revision],
        "critic_reviews": critic_count,
        "critic_redesign_count": redesign_count,
        "critic_redesign_rate": redesign_count / critic_count if critic_count else 0.0,
        "schema_patch_rejection_events": patch_rejections,
        "maximum_repeated_blocker_count": max(blocker_counts.values(), default=0),
        "maximum_schema_churn_ratio": round(max_churn_ratio, 6),
        "strictness_review_required": bool(
            critic_count
            and (
                redesign_count == critic_count
                or patch_rejections >= max(2, critic_count // 2)
            )
        ),
        "inspection_requirements": [
            "check field-count and path churn for oscillation",
            "check whether prior critic advice was reversed without resolution",
            "check internal task-contract capability regressions",
            "check critic rejection rate, repeated patches, and rule conflicts",
            "check whether critic or supervisor acceptance rules are too strict",
        ],
    }


def resume_router_node(state):
    decision = state.get("supervisor_decision") or {}
    inspection_revision = int(decision.get("inspection_revision", 0) or 0)
    approved_through = int(state.get("schema_inspection_approved_through", 0) or 0)
    if (
        decision.get("action") == "schema_inspection_required"
        and approved_through >= inspection_revision
    ):
        errors = list(
            state.get("validation_errors") or state.get("repair_instructions") or []
        )
        boundary_errors = [
            error
            for error in errors
            if str(error).startswith("schema_inspection_boundary_reached:")
        ]
        repair_errors = [error for error in errors if error not in boundary_errors]
        repair_target = (
            state.get("schema_inspection_resume_node") if boundary_errors else None
        ) or coverage_repair_target(repair_errors)
        if not repair_target:
            repair_target = (
                "schema_design_repair"
                if state.get("module_outputs", {}).get("schema_design_module")
                else state.get("last_error_node") or "supervisor_router"
            )
        resumed_decision = {
            "action": "schema_inspection_approved",
            "reason": (
                f"schema revisions through {inspection_revision} were inspected; "
                "continue under the next periodic inspection boundary"
            ),
            "inspection_revision": inspection_revision,
            "next_node": repair_target,
        }
        return merge_update(
            state,
            {
                "next_node": repair_target,
                "status": "running",
                "validation_errors": [],
                "repair_instructions": repair_errors,
                "supervisor_decision": resumed_decision,
                "schema_inspection_resume_node": "",
            },
        )
    repair_target = coverage_repair_target(state.get("repair_instructions") or [])
    if (
        repair_target == "subjective_supervisor"
        and state.get("last_error_node") == "aggregation"
        and state.get("module_outputs", {}).get("field_planning_module")
    ):
        repaired = compile_schema_from_field_planning(
            state,
            "resume-time deterministic contract repair after aggregation coverage failure",
        )
        repaired = clear_downstream_outputs(
            repaired,
            ["specialization_critic_module", "aggregation"],
        )
        return merge_update(
            repaired,
            {
                "validation_errors": [],
                "repair_instructions": [],
                "status": "running",
                "next_node": "specialization_critic",
            },
        )
    if (
        repair_target
        and state.get("last_error_node") == "aggregation"
        and state.get("next_node") != repair_target
    ):
        decision = deepcopy(state.get("supervisor_decision") or {})
        decision.update(
            {
                "action": "resume_coverage_repair",
                "reason": "Re-evaluated persisted coverage errors with the current deterministic router.",
                "next_node": repair_target,
            }
        )
        return merge_update(state, {"next_node": repair_target, "supervisor_decision": decision})
    return state


def prepare_state(state):
    if state.get("shared_context"):
        return write_state_snapshot(state, "prepare", "locating")

    shared_context, human_advice = build_shared_context_from_args(state["args"])
    update = {
        "shared_context": shared_context,
        "module_outputs": {},
        "module_attempts": {},
        "module_errors": {},
        "validation_errors": [],
        "human_advice": human_advice,
        "status": "running",
        "retry_counts": {},
        "schema_revision": 0,
        "critic_reviewed_schema_revision": -1,
        "blocker_history": [],
        "rejected_critic_patch_hashes": [],
        "schema_iteration_history": [],
        "schema_inspection_approved_through": 0,
        "schema_inspection_resume_node": "",
    }
    next_state = merge_update(state, update)
    return write_state_snapshot(next_state, "prepare", "locating")


def call_module_node(state, module_name, prompt, current_node, next_node):
    args = dict_to_namespace(state["args"])
    if module_name in {
        "field_planning_module",
        "schema_design_module",
        "specialization_critic_module",
        "aggregation",
    }:
        iteration_context = build_schema_iteration_context(state)
        if iteration_context:
            prompt = f"{prompt}\n\n{iteration_context}"
    client = section_agent.get_client(
        base_url=args.base_url,
        api_key=args.api_key,
        backend=args.llm_backend,
        timeout=getattr(args, "request_timeout", 180),
    )
    result, errors, attempts = section_agent.call_module(
        client,
        args.model,
        module_name,
        prompt,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    if module_name == "schema_design_module" and not errors and isinstance(result, dict):
        result = canonicalize_schema_definition(result)
    outputs = deepcopy(state.get("module_outputs", {}))
    module_attempts = deepcopy(state.get("module_attempts", {}))
    module_errors = deepcopy(state.get("module_errors", {}))
    outputs[module_name] = result
    prior_attempts = list(module_attempts.get(module_name) or [])
    round_offset = max(
        [int(item.get("round", 0) or 0) for item in prior_attempts if isinstance(item, dict)],
        default=0,
    )
    appended_attempts = []
    for index, attempt in enumerate(attempts or [], start=1):
        item = deepcopy(attempt) if isinstance(attempt, dict) else {"raw_response": attempt}
        item["round"] = round_offset + index
        appended_attempts.append(item)
    module_attempts[module_name] = [*prior_attempts, *appended_attempts]
    module_errors[module_name] = errors
    update = {
        "module_outputs": outputs,
        "module_attempts": module_attempts,
        "module_errors": module_errors,
    }
    if module_name == "schema_design_module" and not errors and isinstance(result, dict):
        update["schema_revision"] = int(state.get("schema_revision", 0) or 0) + 1
        update["critic_reviewed_schema_revision"] = -1
    if errors:
        error_type = classify_errors(errors)
        update["validation_errors"] = errors
        update["status"] = "awaiting_supervisor_decision"
        update["last_error_node"] = current_node
        update["last_error_type"] = error_type
        next_node = "supervisor_router"
    return write_state_snapshot(merge_update(state, update), current_node, next_node)


STOP_AFTER_MODULE_BY_NODE = {
    "locating": "locating_module",
    "mechanism": "mechanism_requirement_module",
    "query_semantics": "query_semantics_module",
    "evidence_model": "evidence_model_module",
    "subjective_supervisor": "subjective_supervisor_module",
    "topic_adaptation": "topic_adaptation_module",
    "section_partition": "section_partition_module",
    "field_planning": "field_planning_module",
    "supervisor": "supervisor_module",
    "figure_classification": "figure_classification_module",
    "schema_design": "schema_design_module",
    "schema_design_repair": "schema_design_module",
    "specialization_critic": "specialization_critic_module",
    "aggregation": "aggregation",
}


def requested_stop_after_module(state):
    args = state.get("args", {})
    requested = args.get("stop_after_module", "") if isinstance(args, dict) else ""
    completed = STOP_AFTER_MODULE_BY_NODE.get(state.get("current_node", ""), "")
    return bool(requested and completed and requested == completed)


def stop_if_errors(state, next_node):
    if state.get("validation_errors"):
        return "supervisor_router"
    if requested_stop_after_module(state):
        return "write_output"
    return next_node


def pause_for_schema_inspection_if_due(state, resume_node):
    """Gate immediately after a successful schema revision reaches a boundary."""
    args = state.get("args") or {}
    interval = int(args.get("schema_inspection_interval", 10) or 10)
    if interval <= 0:
        return state
    revision = int(state.get("schema_revision", 0) or 0)
    approved_through = int(state.get("schema_inspection_approved_through", 0) or 0)
    boundary = ((approved_through // interval) + 1) * interval
    if revision < boundary:
        return state
    return merge_update(
        state,
        {
            "validation_errors": [f"schema_inspection_boundary_reached:{boundary}"],
            "repair_instructions": [],
            "status": "awaiting_supervisor_decision",
            "last_error_node": "schema_inspection",
            "last_error_type": "schema_inspection_boundary",
            "schema_inspection_resume_node": resume_node,
            "next_node": "supervisor_router",
        },
    )


def coverage_repair_target(errors):
    messages = [str(error) for error in errors or []]
    if any(
        message.startswith("schema_patch_contract_regression:")
        for message in messages
    ):
        return "specialization_critic"
    if any(message.startswith("coverage:missing_entity_owners:") for message in messages):
        return "subjective_supervisor"
    if any(message.startswith("coverage:missing_concepts:") for message in messages):
        return "schema_design_repair"
    if any(
        message.startswith("coverage:untraced_care_counterfactuals:")
        for message in messages
    ):
        return "query_semantics"
    if any(
        message.startswith((
            "coverage:unresolved_reference_fields:",
            "coverage:invalid_reference_decisions:",
        ))
        for message in messages
    ):
        return "field_planning"
    if any(
        message.startswith((
            "coverage:unmapped_domain_fields:",
            "coverage:invalid_concept_references:",
        ))
        for message in messages
    ):
        return "specialization_critic"
    if any(
        message.startswith((
            "coverage:incomplete_object_contracts:",
            "coverage:missing_evidence_contracts:",
            "coverage:untraced_query_requirements:",
            "coverage:missing_reference_fields:",
            "coverage:redundant_fields:",
            "coverage:missing_field_utility_reasons:",
        ))
        for message in messages
    ):
        return "schema_design_repair"
    if any("figure_constraint" in message or message.startswith("coverage:figure_") for message in messages):
        return "figure_classification"
    return None


def critic_patch_hash(operations):
    payload = json.dumps(
        operations or [],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def has_current_revision_critic_patch(state, error_node, error_type):
    if error_node != "schema_design" or error_type != "schema_validation_failure":
        return False
    critic = (state.get("module_outputs") or {}).get("specialization_critic_module") or {}
    operations = critic.get("patch_operations") or []
    if not isinstance(operations, list) or not operations:
        return False
    try:
        reviewed_revision = int(critic.get("reviewed_schema_revision"))
        schema_revision = int(state.get("schema_revision", 0) or 0)
    except (TypeError, ValueError):
        return False
    if reviewed_revision != schema_revision:
        return False
    return critic_patch_hash(operations) not in set(
        state.get("rejected_critic_patch_hashes") or []
    )


def validate_current_revision_critic_patch(state):
    critic = (state.get("module_outputs") or {}).get("specialization_critic_module") or {}
    operations = critic.get("patch_operations") or []
    if not isinstance(operations, list) or not operations:
        return "", []
    try:
        reviewed_revision = int(critic.get("reviewed_schema_revision"))
        schema_revision = int(state.get("schema_revision", 0) or 0)
    except (TypeError, ValueError):
        return "", []
    if reviewed_revision != schema_revision:
        return "", []
    patch_hash = critic_patch_hash(operations)
    schema = deepcopy(
        (state.get("module_outputs") or {}).get("schema_design_module") or {}
    )
    _normalized, errors = validate_schema_patch_operations(schema, operations, state)
    return patch_hash, errors


def blocker_fingerprint(error_node, error_type, errors):
    normalized = [
        re.sub(r"\s+", " ", str(error or "").strip().lower())
        for error in errors or []
        if str(error or "").strip()
    ]
    payload = json.dumps(
        {
            "error_node": str(error_node or ""),
            "error_type": str(error_type or ""),
            "errors": sorted(set(normalized)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def supervisor_router_node(state):
    """Supervisor node: classify failures and choose the next graph step."""
    errors = state.get("validation_errors") or []
    if not errors:
        decision = {
            "action": "continue",
            "reason": "No active validation errors.",
            "next_node": state.get("next_node") or "write_output",
        }
        return write_state_snapshot(
            merge_update(state, {"supervisor_decision": decision}),
            "supervisor_router",
            decision["next_node"],
        )

    error_node = state.get("last_error_node") or state.get("current_node") or ""
    error_type = state.get("last_error_type") or classify_errors(errors)
    retry_counts = deepcopy(state.get("retry_counts", {}))
    args = dict_to_namespace(state["args"])
    max_supervisor_retries = int(getattr(args, "max_supervisor_retries", 1) or 1)
    max_total_repairs = int(getattr(args, "max_total_supervisor_repairs", 12) or 12)
    schema_inspection_interval = int(
        getattr(args, "schema_inspection_interval", 10) or 10
    )
    repeated_blocker_inspection_threshold = int(
        getattr(args, "repeated_blocker_inspection_threshold", 3) or 3
    )
    approved_through = int(state.get("schema_inspection_approved_through", 0) or 0)
    inspection_boundary = (
        (approved_through // schema_inspection_interval) + 1
    ) * schema_inspection_interval
    fingerprint = blocker_fingerprint(error_node, error_type, errors)
    repeated_blocker_count = 1 + sum(
        1
        for item in state.get("blocker_history") or []
        if isinstance(item, dict) and item.get("fingerprint") == fingerprint
    )
    retry_key = f"{error_node}:{error_type}:{fingerprint}"
    retry_count = int(retry_counts.get(retry_key, 0))
    total_repairs = sum(int(value or 0) for value in retry_counts.values())
    repair_target = coverage_repair_target(errors)
    current_critic_patch_pending = has_current_revision_critic_patch(
        state,
        error_node,
        error_type,
    )
    current_critic_patch_hash = ""
    current_critic_patch_errors = []
    if current_critic_patch_pending:
        current_critic_patch_hash, current_critic_patch_errors = (
            validate_current_revision_critic_patch(state)
        )

    if int(state.get("schema_revision", 0) or 0) >= inspection_boundary:
        decision = {
            "action": "schema_inspection_required",
            "reason": (
                "periodic schema inspection boundary reached; pause automatic iteration "
                "and inspect convergence and critic strictness before continuing"
            ),
            "error_node": error_node,
            "error_type": error_type,
            "blocker_fingerprint": fingerprint,
            "retry_count": retry_count,
            "schema_revision": int(state.get("schema_revision", 0) or 0),
            "inspection_revision": inspection_boundary,
            "schema_inspection_interval": schema_inspection_interval,
            "inspection_summary": schema_inspection_summary(
                state,
                max(0, inspection_boundary - schema_inspection_interval),
                inspection_boundary,
            ),
            "next_node": "write_output",
        }
    elif repeated_blocker_count >= repeated_blocker_inspection_threshold:
        decision = {
            "action": "repeated_blocker_inspection_required",
            "reason": (
                "the same blocker repeated without a schema-version transition; pause "
                "to inspect protocol correctness, rule conflicts, and critic strictness"
            ),
            "error_node": error_node,
            "error_type": error_type,
            "blocker_fingerprint": fingerprint,
            "repeated_blocker_count": repeated_blocker_count,
            "repeated_blocker_inspection_threshold": repeated_blocker_inspection_threshold,
            "schema_revision": int(state.get("schema_revision", 0) or 0),
            "inspection_summary": {
                "strictness_review_required": True,
                "errors": list(errors),
                "inspection_requirements": [
                    "check whether the blocker is caused by protocol or path normalization",
                    "check whether critic and supervisor rules conflict",
                    "check whether critic or supervisor acceptance rules are too strict",
                    "check whether the same rejected patch or advice is being replayed",
                ],
            },
            "next_node": "write_output",
        }
    elif total_repairs >= max_total_repairs:
        decision = {
            "action": "finish_needs_review",
            "reason": "total supervisor repair budget exhausted",
            "error_node": error_node,
            "error_type": error_type,
            "blocker_fingerprint": fingerprint,
            "retry_count": retry_count,
            "next_node": "write_output",
        }
    elif (
        current_critic_patch_pending
        and current_critic_patch_errors
        and retry_count < max_supervisor_retries
    ):
        retry_counts[retry_key] = retry_count + 1
        decision = {
            "action": "retry_critic_patch",
            "reason": (
                "The current-revision critic patch failed deterministic protocol "
                "validation; return the exact errors to the critic instead of replaying "
                "the rejected patch."
            ),
            "error_node": "specialization_critic",
            "error_type": "schema_validation_failure",
            "blocker_fingerprint": fingerprint,
            "critic_patch_hash": current_critic_patch_hash,
            "critic_patch_errors": list(current_critic_patch_errors),
            "retry_count": retry_counts[retry_key],
            "next_node": "specialization_critic",
        }
    elif current_critic_patch_pending and retry_count < max_supervisor_retries:
        retry_counts[retry_key] = retry_count + 1
        decision = {
            "action": "repair_schema_design",
            "reason": (
                "The specialization critic supplied a structured patch bound to the "
                "current schema revision; apply it before rerunning any upstream module."
            ),
            "error_node": error_node,
            "error_type": error_type,
            "blocker_fingerprint": fingerprint,
            "retry_count": retry_counts[retry_key],
            "next_node": "schema_design_repair",
        }
    elif repair_target and retry_count < max_supervisor_retries:
        retry_counts[retry_key] = retry_count + 1
        decision = {
            "action": "repair_coverage",
            "reason": "Deterministic coverage validation selected the owning upstream module.",
            "error_node": error_node,
            "error_type": error_type,
            "blocker_fingerprint": fingerprint,
            "retry_count": retry_counts[retry_key],
            "next_node": repair_target,
        }
    elif (
        error_node == "schema_design"
        and error_type in {"json_parse_failure", "schema_validation_failure", "module_failure"}
        and retry_count < max_supervisor_retries
    ):
        retry_counts[retry_key] = retry_count + 1
        decision = {
            "action": "repair_schema_design",
            "reason": "schema_design failed or critic requested redesign; retry with compact repair prompt.",
            "error_node": error_node,
            "error_type": error_type,
            "blocker_fingerprint": fingerprint,
            "retry_count": retry_counts[retry_key],
            "next_node": "schema_design_repair",
        }
    elif retry_count < max_supervisor_retries and error_node in {
        "locating",
        "mechanism",
        "query_semantics",
        "evidence_model",
        "subjective_supervisor",
        "topic_adaptation",
        "section_partition",
        "field_planning",
        "supervisor",
        "figure_classification",
        "schema_design",
        "schema_design_repair",
        "specialization_critic",
        "aggregation",
    }:
        retry_counts[retry_key] = retry_count + 1
        decision = {
            "action": "retry_node",
            "reason": "module failed once; supervisor allows one generic retry.",
            "error_node": error_node,
            "error_type": error_type,
            "blocker_fingerprint": fingerprint,
            "retry_count": retry_counts[retry_key],
            "next_node": error_node,
        }
    else:
        decision = {
            "action": "finish_needs_review",
            "reason": "supervisor retry budget exhausted or no safe route available.",
            "error_node": error_node,
            "error_type": error_type,
            "blocker_fingerprint": fingerprint,
            "retry_count": retry_count,
            "next_node": "write_output",
        }

    blocker_history = deepcopy(state.get("blocker_history") or [])
    blocker_history.append(
        {
            "fingerprint": fingerprint,
            "error_node": error_node,
            "error_type": error_type,
            "errors": list(errors),
            "decision": decision.get("action"),
            "retry_count": decision.get("retry_count", retry_count),
        }
    )
    update = {
        "supervisor_decision": decision,
        "retry_counts": retry_counts,
        "blocker_history": blocker_history,
        "status": "running" if decision["next_node"] != "write_output" else "needs_review",
    }
    if decision.get("action") == "retry_critic_patch" and current_critic_patch_hash:
        rejected_hashes = list(state.get("rejected_critic_patch_hashes") or [])
        if current_critic_patch_hash not in rejected_hashes:
            rejected_hashes.append(current_critic_patch_hash)
        update["rejected_critic_patch_hashes"] = rejected_hashes
    if decision["next_node"] != "write_output":
        update["repair_instructions"] = (
            [f"schema_patch_rejected:{item}" for item in current_critic_patch_errors]
            if decision.get("action") == "retry_critic_patch"
            else errors
        )
        update["validation_errors"] = []
    return write_state_snapshot(merge_update(state, update), "supervisor_router", decision["next_node"])


def locating_node(state):
    args = dict_to_namespace(state["args"])
    shared = state["shared_context"]
    return call_module_node(
        state,
        "locating_module",
        build_locating_prompt(
            args.database_goal,
            args.discipline,
            shared["query_requirements"],
            shared["key_description_text"],
            shared.get("reference_paper_context", ""),
        ),
        "locating",
        "mechanism",
    )


def mechanism_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "mechanism_requirement_module",
        build_mechanism_requirement_prompt(state["shared_context"], outputs["locating_module"]),
        "mechanism",
        "query_semantics",
    )


def query_semantics_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "query_semantics_module",
        build_query_semantics_prompt(state["shared_context"], outputs["locating_module"]),
        "query_semantics",
        "evidence_model",
    )


def evidence_model_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "evidence_model_module",
        build_evidence_model_prompt(state["shared_context"], outputs["locating_module"]),
        "evidence_model",
        "subjective_supervisor",
    )


def subjective_supervisor_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "subjective_supervisor_module",
        build_subjective_supervisor_prompt(
            state["shared_context"],
            outputs["locating_module"],
            outputs["mechanism_requirement_module"],
            outputs["query_semantics_module"],
            outputs["evidence_model_module"],
        ),
        "subjective_supervisor",
        "topic_adaptation",
    )


def topic_adaptation_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "topic_adaptation_module",
        build_topic_adaptation_prompt(state["shared_context"], outputs["locating_module"]),
        "topic_adaptation",
        "section_partition",
    )


def section_partition_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "section_partition_module",
        build_section_partition_prompt(
            state["shared_context"],
            outputs["locating_module"],
            outputs["mechanism_requirement_module"],
            outputs["query_semantics_module"],
            outputs["evidence_model_module"],
            outputs["subjective_supervisor_module"],
            outputs["topic_adaptation_module"],
        ),
        "section_partition",
        "field_planning",
    )


def field_planning_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "field_planning_module",
        build_field_planning_prompt(
            state["shared_context"],
            outputs["locating_module"],
            outputs["mechanism_requirement_module"],
            outputs["query_semantics_module"],
            outputs["evidence_model_module"],
            outputs["subjective_supervisor_module"],
            outputs["topic_adaptation_module"],
            outputs["section_partition_module"],
        ),
        "field_planning",
        "human_advice_gate",
    )


def advice_is_acceptance(advice):
    text = str(advice or "").strip().lower()
    if not text:
        return False
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        verdict = str(payload.get("verdict") or payload.get("decision") or "").strip().lower()
        if verdict in {"accept", "accepted", "approve", "approved", "pass", "passed"}:
            return True
        if verdict in {
            "reject",
            "rejected",
            "revise",
            "revision_required",
            "needs_revision",
            "need_review",
        }:
            return False
    reject_tokens = [
        "not acceptable",
        "needs revision",
        "revise",
        "reject",
        "不合格",
        "不通过",
        "不能通过",
        "不可通过",
        "需要修改",
        "需要补充",
        "请修改",
        "请补充",
        "仍缺少",
        "缺少",
        "不合理",
        "不能接受",
    ]
    if any(token in text for token in reject_tokens):
        return False
    accept_tokens = [
        "pass",
        "approve",
        "approved",
        "accepted",
        "acceptable",
        "合格",
        "通过",
        "可以进入",
        "没问题",
        "无问题",
        "可用",
        "可以接受",
    ]
    return any(token in text for token in accept_tokens)


def human_advice_gate_node(state):
    args = dict_to_namespace(state["args"])
    shared_at_entry = state.get("shared_context") or {}
    initial_advice = (state.get("human_advice") or "").strip()
    post_design_advice = (state.get("post_design_human_advice") or "").strip()
    advice_was_seeded_before_design = bool(
        shared_at_entry.get("human_advice_available_before_design")
    )
    human_advice = (
        post_design_advice
        if advice_was_seeded_before_design
        else initial_advice
    )
    completed_review_round = int(
        state.get("human_review_round")
        or (1 if advice_was_seeded_before_design and initial_advice else 0)
    )
    if getattr(args, "skip_human_expert_review", False) and not human_advice:
        outputs = deepcopy(state.get("module_outputs", {}))
        outputs["human_advice_gate"] = {
            "status": "skipped",
            "position": "after_internal_field_design_pass",
            "reason": "Human expert review was explicitly skipped by --skip-human-expert-review.",
        }
        return write_state_snapshot(
            merge_update(state, {"module_outputs": outputs}),
            "human_advice_gate",
            "write_output",
        )
    if not human_advice and completed_review_round >= 3:
        outputs = deepcopy(state.get("module_outputs", {}))
        outputs["human_advice_gate"] = {
            "status": "maximum_rounds_exhausted",
            "maximum_rounds": 3,
            "reason": "Three blind expert-review rounds completed without acceptance.",
        }
        return write_state_snapshot(
            merge_update(
                state,
                {
                    "module_outputs": outputs,
                    "status": "awaiting_supervisor_decision",
                    "validation_errors": [
                        "blind human expert review exhausted the maximum of three rounds"
                    ],
                    "last_error_node": "human_advice_gate",
                    "last_error_type": "expert_review_failure",
                    "next_node": "supervisor_router",
                },
            ),
            "human_advice_gate",
            "supervisor_router",
        )
    if not human_advice:
        context_path, context = write_human_gate_context(state)
        waiting_state = merge_update(
            state,
            {
                "status": "waiting_for_human_advice",
                "next_node": "human_advice_gate",
                "module_outputs": {
                    **state.get("module_outputs", {}),
                    "human_advice_gate": {
                        "status": "waiting_for_human_advice",
                        "context_path": context_path,
                    },
                },
            },
        )
        write_state_snapshot(waiting_state, "human_advice_gate", "human_advice_gate")
        outputs = state["module_outputs"]
        human_advice = interrupt(
            {
                "status": "waiting_for_human_advice",
                "gate": "after_internal_field_design_pass",
                "context_path": context_path,
                "available_context": {
                    "result": state.get("result"),
                    "specialization_critic_module": outputs.get("specialization_critic_module"),
                    "schema_design_module": outputs.get("schema_design_module"),
                },
                "required_advice": context["required_advice"],
            }
        )

    shared = deepcopy(state["shared_context"])
    shared["human_advice"] = str(human_advice).strip()
    outputs = deepcopy(state.get("module_outputs", {}))
    current_advice = shared["human_advice"]
    review_round = completed_review_round + 1
    try:
        parsed_advice = json.loads(current_advice)
        declared_round = int(parsed_advice.get("round") or 0)
        if declared_round:
            review_round = declared_round
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    review_round = min(max(review_round, 1), 3)
    if advice_is_acceptance(current_advice):
        outputs["human_advice_gate"] = {
            "status": "accepted",
            "position": "before_requirement_contract" if shared.get("human_advice_available_before_design") else "after_internal_field_design_pass",
            "human_advice": current_advice,
            "reason": "Human expert review explicitly accepted the internally passing field design.",
        }
        return write_state_snapshot(
            merge_update(
                state,
                {
                    "shared_context": shared,
                    "module_outputs": outputs,
                    "status": "success",
                    "validation_errors": [],
                    "human_review_round": review_round,
                    "post_design_human_advice": "",
                },
            ),
            "human_advice_gate",
            "write_output",
        )

    outputs["human_advice_gate"] = {
        "status": "expert_revision_requested",
        "position": "after_internal_field_design_pass",
        "human_advice": shared["human_advice"],
        "policy": "Expert advice is applied only after internal critic/supervisor acceptance.",
    }
    retry_counts = deepcopy(state.get("retry_counts", {}))
    retry_counts.pop("schema_design", None)
    retry_counts.pop("schema_design_repair", None)
    update = {
        "human_advice": shared["human_advice"],
        "shared_context": shared,
        "module_outputs": outputs,
        "human_review_applied": True,
        "validation_errors": [
            "human expert review requested field-design revision after internal acceptance",
            current_advice,
        ],
        "repair_instructions": [current_advice],
        "applied_human_advice": current_advice,
        "human_review_round": review_round,
        "post_design_human_advice": "",
        "retry_counts": retry_counts,
        "status": "awaiting_supervisor_decision",
        "last_error_node": "schema_design",
        "last_error_type": "schema_validation_failure",
        "next_node": "supervisor_router",
    }
    return write_state_snapshot(merge_update(state, update), "human_advice_gate", "supervisor_router")


def supervisor_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "supervisor_module",
        build_supervisor_prompt(
            state["shared_context"],
            outputs["locating_module"],
            outputs["topic_adaptation_module"],
            outputs["section_partition_module"],
            outputs["field_planning_module"],
            human_advice="",
        ),
        "supervisor",
        "figure_classification",
    )


def figure_classification_node(state):
    outputs = state["module_outputs"]
    update = call_module_node(
        state,
        "figure_classification_module",
        build_figure_classification_prompt(
            state["shared_context"],
            outputs["section_partition_module"],
            outputs["field_planning_module"],
            outputs["supervisor_module"],
        ),
        "figure_classification",
        "schema_design",
    )
    figure_result = update["module_outputs"]["figure_classification_module"]
    supervisor_result = state["module_outputs"]["supervisor_module"]
    if not update.get("validation_errors") and (
        supervisor_result.get("enable_figure_classification")
        != figure_result.get("enable_figure_classification")
    ):
        update["validation_errors"] = [
            "Supervisor decision and figure classification branch are inconsistent"
        ]
        update["status"] = "needs_review"
        update["next_node"] = "write_output"
    return write_state_snapshot(update, "figure_classification", update.get("next_node", "schema_design"))


def schema_design_node(state):
    if should_compile_schema_deterministically(state):
        update = compile_schema_from_field_planning(
            state,
            "the complete field plan is too large for one reliable model response",
        )
        update = clear_downstream_outputs(
            update,
            ["specialization_critic_module", "aggregation"],
        )
        update["validation_errors"] = []
        update["status"] = "running"
        update["next_node"] = "specialization_critic"
        update = pause_for_schema_inspection_if_due(update, "specialization_critic")
        return write_state_snapshot(update, "schema_design", update["next_node"])

    outputs = state["module_outputs"]
    update = call_module_node(
        state,
        "schema_design_module",
        build_schema_design_prompt(
            state["shared_context"],
            outputs["locating_module"],
            outputs["mechanism_requirement_module"],
            outputs["query_semantics_module"],
            outputs["evidence_model_module"],
            outputs["subjective_supervisor_module"],
            outputs["topic_adaptation_module"],
            outputs["section_partition_module"],
            outputs["field_planning_module"],
            outputs["supervisor_module"],
            outputs["figure_classification_module"],
        ),
        "schema_design",
        "specialization_critic",
    )
    if not update.get("validation_errors"):
        update = clear_downstream_outputs(
            update,
            ["specialization_critic_module", "aggregation"],
        )
        update["validation_errors"] = []
        update["status"] = "running"
        update["next_node"] = "specialization_critic"
        update = pause_for_schema_inspection_if_due(update, "specialization_critic")
    return write_state_snapshot(update, "schema_design", update.get("next_node", "specialization_critic"))


def build_schema_repair_prompt(state):
    outputs = state["module_outputs"]
    attempts = state.get("module_attempts", {}).get("schema_design_module", [])
    previous_raw = ""
    if attempts:
        previous_raw = str(attempts[-1].get("raw_response") or "")[-500000:]
    errors = (
        state.get("repair_instructions")
        or state.get("validation_errors")
        or state.get("module_errors", {}).get("schema_design_module", [])
    )
    return f"""
You are the schema_design repair node supervised by the Step 8 supervisor.

The previous schema_design attempt failed and must be repaired.

Failure type: {state.get("last_error_type", "")}
Validation errors:
{json.dumps(errors, ensure_ascii=False, indent=2)}

Previous raw response tail:
{previous_raw}

Supervisor repair instruction:
- Return one complete valid JSON object only.
- Do not use markdown fences.
- Keep the output compact enough to avoid truncation.
- Do not use a numeric field-count target. Preserve every justified task/reference leaf, but reject speculative fields, aliases represented as separate fields, deterministic duplicates, and fields without requirement/evidence traceability.
- If the failed schema contains roots that are absent from the approved section architecture,
  field_planning_module, and supervisor-approved extension owners, discard the entire schema.
  Do not incrementally patch it. Rebuild from field_planning_module and the six-section materials backbone.
- For materials-literature tasks, default roots are paper_info, primary_signature,
  primary_signature_normalized, material_info, section5, normalization_aliases,
  material_name_aliases, formula_aliases, and sample_id_aliases. Additional roots require
  explicit supervisor justification.
- Do not satisfy a missing concept by adding only its parent object. Add explicit child paths for values, units, methods,
  conditions, source evidence, sample linkage, and confidence whenever the validation error asks for them.
- Use array/object parent fields plus explicit child paths for children that are essential for extraction, storage, or validation.
- Every important numeric record should preserve value/raw_value/unit/conditions/evidence where relevant, but do not enumerate every possible synonym.
- Ensure top_level_keys and field_registry are both non-empty.
- Every field_registry item must include: field_path, section_id, field_name, data_type, required, source_basis, figure_constraint, reason.

Context for rebuilding:
Section architecture:
{json.dumps(outputs.get("section_partition_module"), ensure_ascii=False, indent=2)}

Field planning:
{json.dumps(outputs.get("field_planning_module"), ensure_ascii=False, indent=2)}

Supervisor:
{json.dumps(outputs.get("supervisor_module"), ensure_ascii=False, indent=2)}

Figure classification:
{json.dumps(outputs.get("figure_classification_module"), ensure_ascii=False, indent=2)}

Return valid JSON only using this shape:
{{
  "top_level_keys": [
    {{"key": "string", "description": "string"}}
  ],
  "field_registry": [
    {{
      "field_path": "string",
      "section_id": "string",
      "field_name": "string",
      "data_type": "string",
      "required": false,
      "source_basis": ["text"],
      "figure_constraint": null,
      "reason": "string"
    }}
  ]
}}
""".strip()


AXIS_PAIR_SEGMENT_PATTERN = re.compile(
    r"(?:^|\.)([a-z][a-z0-9]{0,2})_([a-z][a-z0-9]{0,2})(?:\.|$)",
    re.IGNORECASE,
)


def has_axis_pair_segment(value):
    return bool(AXIS_PAIR_SEGMENT_PATTERN.search(str(value or "")))


def section_for_repair_parent(parent, source_text):
    text = f"{parent} {source_text}".lower()
    if "section1" in text:
        return "material_info.section1"
    if "section2" in text or "synthesis" in text or "anneal" in text or "process" in text:
        return "material_info.section2"
    if "section3" in text or any(
        token in text for token in ("characterization", "microscopy", "spectroscopy", "diffraction")
    ):
        return "material_info.section3"
    if "section4" in text or any(
        token in text for token in ("curve", "plot", "map", "series", "response")
    ) or has_axis_pair_segment(parent):
        return "material_info.section4"
    if "theory" in text or "simulation" in text:
        return "section5"
    if "section0" in text or "sample" in text or "microstruct" in text or "material" in text:
        return "material_info.section0"
    return "material_info.section1"


def material_section_path(section_id, suffix):
    if suffix.startswith(("material_info.", "paper_info.", "section5.")):
        return suffix
    if section_id.startswith("material_info.") or section_id.startswith("paper_info"):
        return f"{section_id}.{suffix}"
    return f"material_info.{section_id}.{suffix}"


def data_type_for_repair_child(child):
    child = child.lower()
    if child in {"value", "raw_value", "uncertainty", "confidence", "frequency", "field_amplitude"}:
        return "number"
    if child in {"conditions", "measurement_conditions", "magnetic_field", "temperature", "pressure"}:
        return "object"
    return "string"


def add_repair_field(registry, existing_paths, field_path, section_id, field_name, reason):
    bad_segments = {"e", "g", "eg", "e.g", "etc", "example", "examples"}
    segments = [segment.strip().lower() for segment in str(field_path).split(".") if segment.strip()]
    if any(segment in bad_segments for segment in segments):
        return
    if field_path in existing_paths:
        return
    existing_paths.add(field_path)
    inferred_type = data_type_for_repair_child(field_name)
    if "parent field" in reason or "section object" in reason:
        inferred_type = "object"
    registry.append(
        {
            "field_path": field_path,
            "section_id": section_id,
            "field_name": field_name,
            "data_type": inferred_type,
            "required": False,
            "source_basis": ["text", "table"],
            "figure_constraint": None,
            "reason": reason,
        }
    )


MATERIAL_TOP_LEVEL_KEYS = [
    {"key": "paper_info", "description": "Bibliographic metadata, abstract, source links, and paper-level resources."},
    {"key": "primary_signature", "description": "Canonical identifier for the primary material system or interface."},
    {"key": "primary_signature_normalized", "description": "Normalized primary material-system identifier used for matching."},
    {"key": "material_info", "description": "Six-section material record covering identity, properties, processing, evidence, and curves."},
    {"key": "section5", "description": "Paper-level theory and mechanism interpretation kept separate from experimental values."},
]

DEFAULT_SCHEMA_ROOTS = {
    "paper_info",
    "primary_signature",
    "primary_signature_normalized",
    "material_info",
    "section5",
    "normalization_aliases",
    "material_name_aliases",
    "formula_aliases",
    "sample_id_aliases",
}

PREFERRED_FIELD_REGISTRY_SIZE = None
HARD_FIELD_REGISTRY_SIZE = None


def field_path_depth(field_path):
    return len([part for part in str(field_path or "").split(".") if part])


def compact_field_registry(registry, preferred_size=PREFERRED_FIELD_REGISTRY_SIZE, hard_size=HARD_FIELD_REGISTRY_SIZE):
    """Deduplicate malformed paths without silently truncating valid schema fields."""
    bad_segments = {"e", "g", "eg", "e.g", "etc", "example", "examples"}
    deduped = []
    seen = set()
    for item in registry or []:
        if not isinstance(item, dict):
            continue
        field_path = str(item.get("field_path") or "").strip()
        if not field_path or field_path in seen:
            continue
        segments = [segment.strip().lower() for segment in field_path.split(".") if segment.strip()]
        if any(segment in bad_segments for segment in segments):
            continue
        seen.add(field_path)
        deduped.append(item)

    if hard_size is None or len(deduped) <= hard_size:
        return deduped

    preferred_size = min(preferred_size or hard_size, hard_size)

    essential_leaf_names = {
        "value",
        "unit",
        "temperature",
        "magnetic_field",
        "pressure",
        "measurement_conditions",
        "conditions",
        "figure",
        "source_figure",
        "source_text",
        "characteristics",
        "method",
        "description",
        "geometry",
        "atmosphere",
        "annealing_temperature",
        "annealing_time",
        "raw_data",
        "extracted_information",
        "authors",
        "abstract",
        "doi",
        "url",
        "year",
        "venue",
        "title",
    }
    section_quota = {
        "paper_info": 24,
        "material_info.section0": 24,
        "material_info.section1": 55,
        "material_info.section2": 28,
        "material_info.section3": 28,
        "material_info.section4": 28,
        "section5": 24,
    }
    section_counts = {section: 0 for section in section_quota}
    parent_seen = set()

    def priority(item):
        path = str(item.get("field_path") or "")
        section = str(item.get("section_id") or "")
        parts = [part for part in path.split(".") if part]
        leaf = parts[-1].lower() if parts else ""
        depth = len(parts)
        score = 0
        if section in section_quota:
            score += 100
        if path.startswith(("paper_info.metadata", "paper_info.resources")):
            score += 80
        if depth <= 3:
            score += 70
        elif depth == 4:
            score += 35
        if leaf in essential_leaf_names:
            score += 30
        if leaf in {"confidence", "raw_value", "uncertainty"}:
            score -= 20
        if depth >= 6:
            score -= 50
        return -score, depth, path.lower()

    compacted = []
    for item in sorted(deduped, key=priority):
        path = str(item.get("field_path") or "")
        section = str(item.get("section_id") or "")
        parts = [part for part in path.split(".") if part]
        leaf = parts[-1].lower() if parts else ""
        quota_key = section if section in section_quota else path.split(".", 2)[0]
        if section_counts.get(quota_key, 0) >= section_quota.get(quota_key, 12):
            continue
        if field_path_depth(path) >= 5 and leaf not in essential_leaf_names:
            continue
        parent_key = ".".join(parts[:-1]) if len(parts) > 3 else path
        if len(parts) > 4 and parent_key in parent_seen and leaf not in essential_leaf_names:
            continue
        compacted.append(item)
        section_counts[quota_key] = section_counts.get(quota_key, 0) + 1
        parent_seen.add(parent_key)
        if len(compacted) >= preferred_size:
            break

    if len(compacted) < min(preferred_size, len(deduped)):
        kept = {item.get("field_path") for item in compacted}
        for item in deduped:
            if item.get("field_path") in kept:
                continue
            compacted.append(item)
            kept.add(item.get("field_path"))
            if len(compacted) >= hard_size:
                break
    return compacted

SUPERVISOR_EXTENSION_ROOTS = {
    "device_info",
    "reaction_info",
    "dataset_info",
    "interface_info",
}

SECTION_HINTS = {
    "paper_info": "paper_info",
    "primary_signature": "material_info.section0",
    "material_info.section0": "material_info.section0",
    "material_info.section1": "material_info.section1",
    "material_info.section2": "material_info.section2",
    "material_info.section3": "material_info.section3",
    "material_info.section4": "material_info.section4",
    "section5": "section5",
}


def collect_schema_roots(schema):
    registry = schema.get("field_registry") or []
    top_level_keys = schema.get("top_level_keys") or []
    roots = set()
    for item in top_level_keys:
        if isinstance(item, dict) and item.get("key"):
            roots.add(str(item["key"]).split(".", 1)[0])
    for item in registry:
        if isinstance(item, dict) and item.get("field_path"):
            roots.add(str(item["field_path"]).split(".", 1)[0])
    return roots


def approved_schema_roots_from_state(state):
    roots = set(DEFAULT_SCHEMA_ROOTS)
    roots.update(SUPERVISOR_EXTENSION_ROOTS)
    # The supervisor can explicitly approve additional owners, but schema output
    # cannot self-approve arbitrary roots through its own top_level_keys.
    supervisor = (state.get("module_outputs") or {}).get("supervisor_module") or {}
    for key in ("approved_schema_roots", "additional_top_level_owners", "allowed_extension_roots"):
        for item in supervisor.get(key) or []:
            if isinstance(item, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", item):
                roots.add(item)
            elif isinstance(item, dict) and item.get("key"):
                value = str(item["key"])
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
                    roots.add(value)
    subjective = (state.get("module_outputs") or {}).get("subjective_supervisor_module") or {}
    for entity in subjective.get("entity_registry") or []:
        if isinstance(entity, dict) and section_agent.is_safe_entity_owner_key(entity.get("owner_key")):
            roots.add(str(entity["owner_key"]))
    human_advice = (
        state.get("human_advice")
        or (state.get("shared_context") or {}).get("human_advice")
        or ""
    )
    for concept in section_agent.explicit_human_advice_schema_concepts(human_advice):
        if section_agent.is_safe_entity_owner_key(concept.get("owner_key")):
            roots.add(str(concept["owner_key"]))
    return roots


def synchronize_top_level_keys(schema):
    """Declare exactly the owners represented by field_registry."""
    schema = deepcopy(schema or {})
    registry = [item for item in schema.get("field_registry") or [] if isinstance(item, dict)]
    roots = {
        str(item.get("field_path") or "").split(".", 1)[0]
        for item in registry
        if str(item.get("field_path") or "").strip()
    }
    existing_descriptions = {
        str(item.get("key")): str(item.get("description") or "")
        for item in schema.get("top_level_keys") or []
        if isinstance(item, dict) and item.get("key")
    }
    canonical_descriptions = {
        item["key"]: item["description"] for item in MATERIAL_TOP_LEVEL_KEYS
    }
    ordered_roots = [
        item["key"] for item in MATERIAL_TOP_LEVEL_KEYS if item["key"] in roots
    ]
    ordered_roots.extend(sorted(roots - set(ordered_roots)))
    schema["top_level_keys"] = [
        {
            "key": root,
            "description": existing_descriptions.get(root)
            or canonical_descriptions.get(root)
            or f"Task-approved owner for {root} records.",
        }
        for root in ordered_roots
    ]
    return schema


def schema_has_unapproved_roots(schema, state=None):
    approved_roots = approved_schema_roots_from_state(state or {})
    return sorted(root for root in collect_schema_roots(schema) if root and root not in approved_roots)


def infer_section_from_field_path(field_path, fallback_section_id="material_info.section1"):
    field_path = str(field_path)
    root = field_path.split(".", 1)[0]
    for prefix, section_id in SECTION_HINTS.items():
        if field_path == prefix or field_path.startswith(f"{prefix}."):
            return section_id
    if field_path.startswith(("formula_aliases", "material_name_aliases", "sample_id_aliases")):
        return "material_info.section0"
    if any(token in field_path.lower() for token in ["fabrication", "synthesis", "anneal", "growth", "processing"]):
        return "material_info.section2"
    if any(token in field_path.lower() for token in ["xrd", "afm", "sem", "tem", "stm", "sts", "arpes", "xas", "xmcd", "microscopy", "characterization"]):
        return "material_info.section3"
    if any(
        token in field_path.lower()
        for token in ("curve", "plot", "map", "series", "response")
    ) or has_axis_pair_segment(field_path):
        return "material_info.section4"
    if any(token in field_path.lower() for token in ["theory", "mechanism", "calculated", "simulation", "model"]):
        return "section5"
    if section_agent.is_safe_entity_owner_key(root):
        return root
    return fallback_section_id


def canonicalize_rebuild_field_path(field_path, fallback_section_id):
    field_path = re.sub(r"\s*\([^)]*\)", "", str(field_path).strip().lstrip("."))
    field_path = field_path.replace("[]", "")
    field_path = field_path.replace(
        "material_info.section1.core_parameter.",
        "material_info.section1.core_parameters.",
    )
    if field_path.startswith("material_info.section5."):
        field_path = "section5." + field_path[len("material_info.section5.") :]
    fallback_section_id = str(fallback_section_id or "material_info.section1")
    if re.fullmatch(r"section[0-4]", fallback_section_id):
        fallback_section_id = f"material_info.{fallback_section_id}"
    elif fallback_section_id == "material_info.section5":
        fallback_section_id = "section5"
    root = field_path.split(".", 1)[0]
    allowed_roots = {
        "paper_info",
        "primary_signature",
        "primary_signature_normalized",
        "material_info",
        "section5",
        "normalization_aliases",
        "material_name_aliases",
        "formula_aliases",
        "sample_id_aliases",
    }
    if root in allowed_roots or section_agent.is_safe_entity_owner_key(root):
        return field_path
    if str(fallback_section_id).startswith("material_info."):
        return f"{fallback_section_id}.{field_path}"
    if str(fallback_section_id) == "section5":
        return f"section5.{field_path}"
    if str(fallback_section_id) == "paper_info":
        return f"paper_info.{field_path}"
    return field_path


def canonicalize_schema_definition(schema):
    """Normalize schema field paths before critics or patch operations consume them."""
    schema = deepcopy(schema or {})
    normalized_registry = []
    for raw_field in schema.get("field_registry") or []:
        if not isinstance(raw_field, dict):
            continue
        field = deepcopy(raw_field)
        section_id = section_agent.canonical_section_id(field.get("section_id"))
        field_path = canonicalize_rebuild_field_path(
            field.get("field_path") or "",
            section_id or "material_info.section1",
        )
        field["field_path"] = field_path
        if field_path.startswith("section5."):
            section_id = "section5"
        field["section_id"] = section_id or infer_section_from_field_path(field_path)
        normalized_registry.append(field)
    schema["field_registry"] = compact_field_registry(normalized_registry)
    return synchronize_top_level_keys(schema)


def infer_data_type_from_field_path(field_path, planned_path=""):
    leaf = str(field_path).rsplit(".", 1)[-1].lower()
    if leaf in {"value", "raw_value", "uncertainty", "confidence_score", "year"}:
        return "number"
    if leaf in {"required", "available", "raw_data"} or leaf.endswith("_flag"):
        return "boolean"
    if leaf in {"conditions", "metadata", "resources", "figure_constraint", "normalization_aliases"}:
        return "object"
    if (
        str(planned_path).strip().endswith("[]")
        or leaf.endswith(("_refs", "_ids"))
        or leaf in {"authors", "keywords", "aliases", "evidence_links", "supporting_methods"}
    ):
        return "array"
    return "string"


def infer_source_basis_from_field_path(field_path, evidence_strategy=""):
    text = str(field_path).lower()
    basis = ["text"]
    if "source_table" in text or text.endswith(".table"):
        basis.append("table")
    if has_axis_pair_segment(text) or any(
        token in text
        for token in [
            "figure",
            "source_figure",
            "theory_figures",
            "curve",
            "plot",
            "xrd",
            "rsm",
            "afm",
            "sem",
            "tem",
            "stm",
            "sts",
            "arpes",
            "xas",
            "xmcd",
            "spectrum",
            "microscopy",
        ]
    ):
        basis.append("figure")
    return list(dict.fromkeys(basis))


def infer_rebuild_figure_constraint(field_path, section_id, source_basis):
    if "figure" not in source_basis:
        return None
    lowered = str(field_path).lower()
    if section_id == "material_info.section3":
        category = "characterization_figure"
    elif section_id == "material_info.section4":
        category = "property_curve"
    elif section_id == "material_info.section1":
        category = "domain_relevant_figure"
    elif section_id == "section5":
        category = "domain_relevant_figure"
    elif any(
        token in lowered for token in ("curve", "plot", "map", "series", "response")
    ) or has_axis_pair_segment(lowered):
        category = "property_curve"
    elif any(token in lowered for token in ["xrd", "afm", "sem", "tem", "stm", "arpes", "spectrum", "microscopy"]):
        category = "characterization_figure"
    else:
        category = "domain_relevant_figure"
    return {
        "uses_figure_classification": True,
        "allowed_sections": [section_id],
        "allowed_figure_categories": [category],
        "why_needed": "Figure-linked evidence must stay owned by the section that contains the measured or observed information.",
    }


# Routing threshold only: large plans are compiled in batches to avoid response
# truncation. It is not a minimum, maximum, or target schema size.
DETERMINISTIC_SCHEMA_COMPILATION_ROUTING_THRESHOLD = 100


def planned_field_count(state):
    field_plan = ((state.get("module_outputs") or {}).get("field_planning_module") or {})
    groups = field_plan.get("field_groups") or []
    grouped_count = sum(
        len(group.get("recommended_fields") or [])
        for group in groups
        if isinstance(group, dict)
    )
    audit = field_plan.get("reference_field_audit") or {}
    audited_count = len(audit.get("included_leaf_paths") or []) + len(
        audit.get("adapted_leaf_mappings") or []
    )
    return max(grouped_count, audited_count)


def should_compile_schema_deterministically(state):
    return bool(state.get("force_deterministic_schema_compilation")) or (
        planned_field_count(state) >= DETERMINISTIC_SCHEMA_COMPILATION_ROUTING_THRESHOLD
    )


def concept_match_score(concept, field_path):
    def normalized_tokens(value):
        tokens = set(section_agent.stable_concept_id(value).split("_"))
        return {token[:-1] if len(token) > 4 and token.endswith("s") else token for token in tokens}

    path_tokens = normalized_tokens(field_path)
    concept_tokens = normalized_tokens(
        section_agent.stable_concept_id(
            f"{concept.get('concept_id', '')} {concept.get('label', '')}"
        )
    )
    ignored = {"a", "an", "and", "of", "the", "value", "data", "information"}
    concept_tokens -= ignored
    overlap = len(path_tokens & concept_tokens)
    required_overlap = 1 if len(concept_tokens) <= 1 else 2
    return overlap if overlap >= required_overlap else 0


def object_contract_for_kind(object_kind):
    object_kind = str(object_kind or "measurement").strip().lower()
    required = {
        "measurement": ["value", "conditions", "entity_ref", "source_type", "evidence", "extraction_confidence"],
        "classification": ["label", "assignment_basis", "source_type", "evidence", "extraction_confidence"],
        "entity_descriptor": ["identity", "evidence"],
        "process": ["method", "conditions", "entity_ref", "evidence"],
        "evidence_collection": ["evidence"],
        "freeform_object": [],
    }
    if object_kind not in required:
        object_kind = "measurement"
    return {"object_kind": object_kind, "required_subfields": required[object_kind]}


def infer_object_kind(field_path, concepts):
    explicit_kinds = [
        str(concept.get("object_kind") or "").strip().lower()
        for concept in concepts
        if concept.get("object_kind") and concept.get("object_kind") != "scalar"
    ]
    if explicit_kinds:
        return explicit_kinds[0]
    lowered = str(field_path).lower()
    if any(token in lowered for token in ("fabrication", "synthesis", "processing", "growth")):
        return "process"
    if any(token in lowered for token in ("classification", "assignment", "phase_label")):
        return "classification"
    if any(token in lowered for token in ("metadata", "identity", "descriptor")):
        return "entity_descriptor"
    if any(token in lowered for token in ("evidence", "figure", "table", "source")):
        return "evidence_collection"
    return "measurement"


def merge_structured_object_contract(custom_contract, field_path, concepts):
    custom_contract = deepcopy(custom_contract) if isinstance(custom_contract, dict) else {}
    object_kind = str(custom_contract.get("object_kind") or "").strip().lower()
    if not object_kind:
        object_kind = infer_object_kind(field_path, concepts)
    base_contract = object_contract_for_kind(object_kind)
    required_subfields = list(
        dict.fromkeys(
            [
                *base_contract.get("required_subfields", []),
                *custom_contract.get("required_subfields", []),
            ]
        )
    )
    fields = deepcopy(custom_contract.get("fields")) if isinstance(custom_contract.get("fields"), dict) else {}
    for slot in base_contract.get("required_subfields", []):
        fields.setdefault(slot, "required_semantic_slot")
    return {
        **custom_contract,
        "object_kind": base_contract["object_kind"],
        "required_subfields": required_subfields,
        "fields": fields,
    }


def concept_owner_prefix(concept):
    owner = str(concept.get("owner_key") or "material_info")
    if (
        owner not in DEFAULT_SCHEMA_ROOTS
        and owner not in SUPERVISOR_EXTENSION_ROOTS
        and not section_agent.is_safe_entity_owner_key(owner)
    ):
        owner = "material_info"
    if owner != "material_info":
        return owner
    object_kind = str(concept.get("object_kind") or "scalar").lower()
    label = str(concept.get("label") or "").lower()
    if object_kind == "process":
        return "material_info.section2"
    if object_kind == "evidence_collection":
        return "material_info.section4"
    if any(token in label for token in ("theory", "mechanism", "calculated", "simulation", "model")):
        return "section5"
    if any(token in label for token in ("identity", "composition", "formula", "material system", "sample name")):
        return "material_info.section0"
    return "material_info.section1"


def compile_schema_from_field_planning(state, reason):
    """Compile every planned field locally when one-shot schema generation is too large."""
    outputs = deepcopy(state.get("module_outputs", {}))
    shared = state.get("shared_context") or {}
    subjective = outputs.get("subjective_supervisor_module") or {}
    requirement_contract = section_agent.normalize_requirement_contract(shared, subjective)
    concepts = requirement_contract.get("concepts") or []
    entities = section_agent.normalize_entity_registry(subjective, requirement_contract)
    field_plan = outputs.get("field_planning_module") or {}
    field_groups = field_plan.get("field_groups") or []
    declared_entity_roots = {
        str(entity.get("owner_key") or "")
        for entity in entities
        if section_agent.is_safe_entity_owner_key(entity.get("owner_key"))
    }
    reference_contract = shared.get("reference_field_contract") or {}
    reference_index = {
        section_agent.normalize_reference_field_path(item.get("path")): item
        for item in reference_contract.get("leaf_fields", []) or []
        if isinstance(item, dict) and item.get("path")
    }
    reference_audit = section_agent.normalize_reference_field_audit(
        reference_contract,
        field_plan.get("reference_field_audit"),
        field_groups,
    )
    reference_target_specs = {}
    for source_path in reference_audit.get("included_leaf_paths") or []:
        normalized_source = section_agent.normalize_reference_field_path(source_path)
        reference_target_specs[normalized_source] = reference_index.get(normalized_source)
    for mapping in reference_audit.get("adapted_leaf_mappings") or []:
        if not isinstance(mapping, dict):
            continue
        normalized_source = section_agent.normalize_reference_field_path(mapping.get("source_path"))
        normalized_target = section_agent.normalize_reference_field_path(mapping.get("target_path"))
        if normalized_target:
            reference_target_specs[normalized_target] = reference_index.get(normalized_source)
    registry = []
    existing_paths = set()

    for group in field_groups:
        if not isinstance(group, dict):
            continue
        fallback_section_id = str(group.get("section_id") or "material_info.section1")
        evidence_strategy = str(group.get("evidence_strategy") or "")
        group_context = " ".join(
            str(group.get(key) or "") for key in ("group_name", "purpose", "evidence_strategy")
        )
        for raw_path in group.get("recommended_fields") or []:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            field_path = canonicalize_rebuild_field_path(raw_path, fallback_section_id)
            root = field_path.split(".", 1)[0]
            if (
                root not in DEFAULT_SCHEMA_ROOTS
                and root not in SUPERVISOR_EXTENSION_ROOTS
                and root not in declared_entity_roots
            ):
                continue
            if field_path in existing_paths:
                continue
            match_context = f"{field_path} {group_context}"
            matches = [concept for concept in concepts if concept_match_score(concept, match_context) > 0]
            field_name = field_path.rsplit(".", 1)[-1]
            exact_matches = [
                concept
                for concept in matches
                if str(concept.get("concept_id") or "") == field_name
            ]
            if exact_matches:
                matches = exact_matches
            reference_spec = reference_target_specs.get(
                section_agent.normalize_reference_field_path(field_path)
            ) or reference_index.get(section_agent.normalize_reference_field_path(field_path))
            data_type = str((reference_spec or {}).get("data_type") or "")
            if not data_type or data_type == "unspecified":
                data_type = infer_data_type_from_field_path(field_path, raw_path)
                if exact_matches and any(
                    concept.get("object_kind") not in {None, "", "scalar"}
                    for concept in exact_matches
                ):
                    data_type = "array of objects" if raw_path.strip().endswith("[]") else "object"
            custom_object_contract = next(
                (
                    deepcopy(concept.get("object_contract"))
                    for concept in matches
                    if isinstance(concept.get("object_contract"), dict)
                ),
                None,
            )
            field = {
                "field_path": field_path,
                "section_id": infer_section_from_field_path(field_path, fallback_section_id),
                "field_name": field_path.rsplit(".", 1)[-1],
                "description": str((reference_spec or {}).get("description_hint") or group.get("purpose") or ""),
                "extraction_notes": str(
                    (reference_spec or {}).get("extraction_notes")
                    or (
                        f"Extract only explicit {field_path.rsplit('.', 1)[-1]} evidence for the target "
                        "material, sample, or entity named in the same evidence span. Keep distinct "
                        "methods, conditions, units, and source locations separate; do not infer or "
                        "merge values across samples, figures, tables, or calculation provenance."
                    )
                ),
                "data_type": data_type,
                "required": False,
                "source_basis": infer_source_basis_from_field_path(field_path, evidence_strategy),
                "concept_ids": [concept["concept_id"] for concept in matches],
                "figure_constraint": None,
                "reason": f"Compiled from field_planning_module because {reason}",
                "field_plan_trace": {
                    "recommended_field": raw_path,
                    "canonical_field_path": field_path,
                    "section_id": section_agent.canonical_section_id(
                        fallback_section_id
                    ),
                },
            }
            field["figure_constraint"] = infer_rebuild_figure_constraint(
                field_path, field["section_id"], field["source_basis"]
            )
            if custom_object_contract and "object" in data_type.lower():
                field["object_contract"] = merge_structured_object_contract(
                    custom_object_contract,
                    field_path,
                    matches,
                )
            elif "object" in data_type:
                object_kind = (
                    "freeform_object"
                    if reference_spec and data_type.strip().lower() == "object"
                    else infer_object_kind(field_path, matches)
                )
                field["object_contract"] = object_contract_for_kind(object_kind)
            registry.append(field)
            existing_paths.add(field_path)

    for raw_path, reference_spec in reference_target_specs.items():
        field_path = canonicalize_rebuild_field_path(raw_path, infer_section_from_field_path(raw_path))
        if field_path in existing_paths:
            continue
        matches = [concept for concept in concepts if concept_match_score(concept, field_path) > 0]
        data_type = str((reference_spec or {}).get("data_type") or "")
        if not data_type or data_type == "unspecified":
            data_type = infer_data_type_from_field_path(field_path)
        field = {
            "field_path": field_path,
            "section_id": infer_section_from_field_path(field_path),
            "field_name": field_path.rsplit(".", 1)[-1],
            "description": str((reference_spec or {}).get("description_hint") or ""),
            "extraction_notes": str((reference_spec or {}).get("extraction_notes") or ""),
            "data_type": data_type,
            "required": False,
            "source_basis": infer_source_basis_from_field_path(field_path),
            "concept_ids": [concept["concept_id"] for concept in matches],
            "figure_constraint": infer_rebuild_figure_constraint(
                field_path,
                infer_section_from_field_path(field_path),
                infer_source_basis_from_field_path(field_path),
            ),
            "reason": "Selected by the task-adaptive reference field audit.",
        }
        if "object" in data_type.lower():
            field["object_contract"] = object_contract_for_kind(
                "freeform_object"
                if reference_spec and data_type.strip().lower() == "object"
                else infer_object_kind(field_path, matches)
            )
        registry.append(field)
        existing_paths.add(field_path)

    mapped = {
        concept_id
        for field in registry
        for concept_id in field.get("concept_ids") or []
    }
    for concept in concepts:
        concept_id = concept.get("concept_id")
        if not concept_id or concept_id in mapped:
            continue
        owner = concept_owner_prefix(concept)
        field_path = f"{owner}.{concept_id}"
        if field_path in existing_paths:
            continue
        object_kind = str(concept.get("object_kind") or "scalar").lower()
        data_type = str(concept.get("data_type") or "").strip()
        if not data_type:
            data_type = "array of objects" if object_kind != "scalar" else "string"
        field = {
            "field_path": field_path,
            "section_id": infer_section_from_field_path(field_path),
            "field_name": concept_id,
            "description": str(concept.get("definition") or concept.get("label") or concept_id),
            "extraction_notes": "Extract only when explicitly supported and bind the value to the correct target entity, sample, conditions, and evidence source.",
            "data_type": data_type,
            "required": bool(concept.get("required")),
            "source_basis": list(concept.get("evidence_types") or ["text"]),
            "concept_ids": [concept_id],
            "figure_constraint": None,
            "reason": "Fallback owner field ensures every required concept has an executable schema target.",
        }
        if "object" in data_type:
            field["object_contract"] = object_contract_for_kind(object_kind)
        registry.append(field)
        existing_paths.add(field_path)

    top_level_keys = []
    top_level_names = set()
    for entity in entities:
        owner = str(entity.get("owner_key") or "").strip()
        if not owner or owner in top_level_names:
            continue
        top_level_keys.append(
            {"key": owner, "description": f"Independent owner for {entity.get('label') or entity.get('entity_id')} records."}
        )
        top_level_names.add(owner)
        if entity.get("required") and entity.get("independent_owner") and not any(
            item["field_path"].startswith(f"{owner}.") for item in registry
        ):
            entity_concept_ids = [
                concept.get("concept_id")
                for concept in concepts
                if concept.get("entity_id") == entity.get("entity_id") and concept.get("concept_id")
            ]
            registry.append(
                {
                    "field_path": f"{owner}.identity",
                    "section_id": owner,
                    "field_name": "identity",
                    "data_type": "object",
                    "required": True,
                    "source_basis": ["text"],
                    "concept_ids": entity_concept_ids,
                    "object_contract": object_contract_for_kind("entity_descriptor"),
                    "figure_constraint": None,
                    "reason": "Required independent entity owner identity contract.",
                }
            )

    schema = synchronize_top_level_keys(
        {"top_level_keys": top_level_keys, "field_registry": registry}
    )
    schema["field_registry"] = apply_definitions(schema["field_registry"], outputs.get("field_planning_module") or {})
    outputs["schema_design_module"] = schema
    module_attempts = deepcopy(state.get("module_attempts", {}))
    module_attempts["schema_design_module"] = [
        *module_attempts.get("schema_design_module", []),
        {
            "round": len(module_attempts.get("schema_design_module", [])) + 1,
            "prompt": "deterministic_compile_schema_from_field_planning",
            "raw_response": json.dumps(schema, ensure_ascii=False),
        },
    ]
    module_errors = deepcopy(state.get("module_errors", {}))
    module_errors["schema_design_module"] = []
    return merge_update(
        state,
        {
            "module_outputs": outputs,
            "module_attempts": module_attempts,
            "module_errors": module_errors,
            "schema_revision": int(state.get("schema_revision", 0) or 0) + 1,
            "critic_reviewed_schema_revision": -1,
        },
    )


def rebuild_material_schema_from_field_planning(state, reason):
    outputs = deepcopy(state.get("module_outputs", {}))
    field_plan = outputs.get("field_planning_module") or {}
    field_groups = field_plan.get("field_groups") or []
    reference_contract = (state.get("shared_context") or {}).get("reference_field_contract") or {}
    reference_index = {
        section_agent.normalize_reference_field_path(item.get("path")): item
        for item in reference_contract.get("leaf_fields", []) or []
        if isinstance(item, dict) and item.get("path")
    }
    reference_audit = section_agent.normalize_reference_field_audit(
        reference_contract,
        field_plan.get("reference_field_audit"),
        field_groups,
    )
    reference_target_specs = {}
    for source_path in reference_audit.get("included_leaf_paths") or []:
        normalized_source = section_agent.normalize_reference_field_path(source_path)
        reference_target_specs[normalized_source] = reference_index.get(normalized_source)
    for mapping in reference_audit.get("adapted_leaf_mappings") or []:
        if not isinstance(mapping, dict):
            continue
        normalized_source = section_agent.normalize_reference_field_path(mapping.get("source_path"))
        normalized_target = section_agent.normalize_reference_field_path(mapping.get("target_path"))
        if normalized_target:
            reference_target_specs[normalized_target] = reference_index.get(normalized_source)
    registry = []
    existing_paths = set()

    for group in field_groups:
        if not isinstance(group, dict):
            continue
        fallback_section_id = str(group.get("section_id") or "material_info.section1")
        evidence_strategy = str(group.get("evidence_strategy") or "")
        for raw_path in group.get("recommended_fields") or []:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            field_path = canonicalize_rebuild_field_path(raw_path, fallback_section_id)
            root = field_path.split(".", 1)[0]
            if root not in DEFAULT_SCHEMA_ROOTS and root not in SUPERVISOR_EXTENSION_ROOTS:
                continue
            if field_path in existing_paths:
                continue
            section_id = infer_section_from_field_path(field_path, fallback_section_id)
            source_basis = infer_source_basis_from_field_path(field_path, evidence_strategy)
            normalized_path = section_agent.normalize_reference_field_path(field_path)
            reference_spec = reference_target_specs.get(normalized_path) or reference_index.get(normalized_path)
            data_type = str((reference_spec or {}).get("data_type") or "")
            if not data_type or data_type == "unspecified":
                data_type = infer_data_type_from_field_path(field_path, raw_path)
            field = {
                "field_path": field_path,
                "section_id": section_id,
                "field_name": field_path.rsplit(".", 1)[-1],
                "description": str((reference_spec or {}).get("description_hint") or group.get("purpose") or ""),
                "extraction_notes": str((reference_spec or {}).get("extraction_notes") or ""),
                "data_type": data_type,
                "required": field_path in {
                    "paper_info.metadata.title",
                    "paper_info.metadata.doi",
                    "paper_info.metadata.abstract",
                    "primary_signature",
                    "primary_signature_normalized",
                    "material_info.section0.primary_material_identity",
                },
                "source_basis": source_basis,
                "figure_constraint": infer_rebuild_figure_constraint(field_path, section_id, source_basis),
                "reason": f"Rebuilt from field_planning_module because {reason}",
            }
            if "object" in data_type.lower():
                field["object_contract"] = object_contract_for_kind(
                    "freeform_object"
                    if reference_spec and data_type.strip().lower() == "object"
                    else infer_object_kind(field_path, [])
                )
            registry.append(field)
            existing_paths.add(field_path)

    for raw_path, reference_spec in reference_target_specs.items():
        field_path = canonicalize_rebuild_field_path(raw_path, infer_section_from_field_path(raw_path))
        if field_path in existing_paths:
            continue
        section_id = infer_section_from_field_path(field_path)
        source_basis = infer_source_basis_from_field_path(field_path)
        data_type = str((reference_spec or {}).get("data_type") or "")
        if not data_type or data_type == "unspecified":
            data_type = infer_data_type_from_field_path(field_path)
        field = {
            "field_path": field_path,
            "section_id": section_id,
            "field_name": field_path.rsplit(".", 1)[-1],
            "description": str((reference_spec or {}).get("description_hint") or ""),
            "extraction_notes": str((reference_spec or {}).get("extraction_notes") or ""),
            "data_type": data_type,
            "required": False,
            "source_basis": source_basis,
            "figure_constraint": infer_rebuild_figure_constraint(field_path, section_id, source_basis),
            "reason": "Selected by the task-adaptive reference field audit during schema rebuild.",
        }
        if "object" in data_type.lower():
            field["object_contract"] = object_contract_for_kind(
                "freeform_object"
                if reference_spec and data_type.strip().lower() == "object"
                else infer_object_kind(field_path, [])
            )
        registry.append(field)
        existing_paths.add(field_path)

    schema = {
        "top_level_keys": deepcopy(MATERIAL_TOP_LEVEL_KEYS),
        "field_registry": registry,
    }
    schema["field_registry"] = apply_definitions(schema["field_registry"], outputs.get("field_planning_module") or {})
    outputs["schema_design_module"] = schema
    module_attempts = deepcopy(state.get("module_attempts", {}))
    module_attempts["schema_design_module"] = [
        *module_attempts.get("schema_design_module", []),
        {
            "round": len(module_attempts.get("schema_design_module", [])) + 1,
            "prompt": "deterministic_rebuild_material_schema_from_field_planning",
            "raw_response": json.dumps(schema, ensure_ascii=False),
        },
    ]
    module_errors = deepcopy(state.get("module_errors", {}))
    module_errors["schema_design_module"] = []
    return merge_update(
        state,
        {
            "module_outputs": outputs,
            "module_attempts": module_attempts,
            "module_errors": module_errors,
        },
    )


def repair_children_from_text(parent, text):
    lower = f"{parent} {text}".lower()
    parent_lower = parent.lower()
    children = [
        "value",
        "raw_value",
        "unit",
        "uncertainty",
        "conditions",
        "conditions.temperature",
        "conditions.external_field.value",
        "conditions.external_field.direction",
        "conditions.pressure",
        "conditions.protocol",
        "conditions.sweep_rate",
        "measurement_method",
        "measurement_conditions",
        "sample_identifier",
        "source_figure",
        "source_table",
        "source_text",
        "source_method",
        "origin",
        "confidence",
    ]
    keyword_children = {
        "transition": ["type", "criterion"],
        "classification": ["label", "assignment_basis", "supporting_methods"],
        "process": ["method", "sequence", "inputs", "outputs"],
        "synthesis": ["method", "description", "temperature", "time", "pressure", "atmosphere"],
        "curve": ["figure", "variable", "x_axis", "y_axis", "measurement_conditions", "protocol"],
        "spectrum": ["figure", "x_axis", "y_axis", "resolution", "measurement_conditions"],
        "map": ["figure", "x_axis", "y_axis", "color_scale", "measurement_conditions"],
        "fit": ["function", "parameters", "goodness_of_fit", "confidence_interval"],
    }
    for keyword, extra_children in keyword_children.items():
        if keyword in parent_lower or keyword.replace("_", " ") in parent_lower:
            children.extend(extra_children)

    for concept_match in re.findall(r"\b([a-z][a-z0-9_]{2,})\b", lower):
        if concept_match in {
            "source_figure",
            "source_table",
            "source_text",
            "measurement_conditions",
            "sample_identifier",
            "confidence",
            "assignment_basis",
            "supporting_methods",
        }:
            children.append(concept_match)
    return list(dict.fromkeys(children))


def is_repair_parent_candidate(token):
    token = str(token or "").strip().strip(".,;:()[]{}\uff0c\u3002\uff1b\uff1a\u3001")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token):
        return False
    lower = token.lower()
    skip = {
        "and",
        "or",
        "with",
        "without",
        "section",
        "section0",
        "section1",
        "section2",
        "section3",
        "section4",
        "section5",
        "value",
        "unit",
        "figure",
        "raw_data",
        "source_text",
        "source_figure",
        "source_table",
        "measurement_conditions",
        "sample_variant",
        "family",
        "families",
        "others",
        "other",
        "e",
        "g",
        "eg",
    }
    if lower in skip:
        return False
    return "_" in token or any(char.isupper() for char in token) or len(token) <= 24


def _legacy_text_schema_repair(state):
    outputs = deepcopy(state.get("module_outputs", {}))
    schema = deepcopy(outputs.get("schema_design_module") or {})
    unapproved_roots = schema_has_unapproved_roots(schema, state)
    errors = [str(item) for item in (state.get("repair_instructions") or state.get("validation_errors", []))]
    error_text = " ".join(errors).lower()
    if unapproved_roots or "unrelated template" in error_text or "not approved by the step 8 section plan" in error_text:
        reason = (
            f"unapproved schema roots detected by supervisor/checker: {unapproved_roots}"
            if unapproved_roots
            else "supervisor/critic identified an unrelated schema template"
        )
        return compile_schema_from_field_planning(state, reason)

    registry = deepcopy(schema.get("field_registry") or [])
    bad_segments = {"e", "g", "eg", "e.g", "etc", "example", "examples"}
    registry = [
        item
        for item in registry
        if not any(
            segment.strip().lower() in bad_segments
            for segment in str(item.get("field_path") or "").split(".")
            if segment.strip()
        )
    ]
    existing_paths = {item.get("field_path") for item in registry if item.get("field_path")}

    parent_candidates = []
    for error in errors:
        for parent, _child in re.findall(
            r"\b([A-Za-z][A-Za-z0-9_]*)\.(value|raw_value|unit|temperature|magnetic_field|source_figure|source_text|figure|raw_data|conditions|measurement_conditions)\b",
            error,
            flags=re.IGNORECASE,
        ):
            parent_candidates.append((parent, error))
        matches = re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(([^)]*)\)", error)
        if matches:
            parent_candidates.extend((parent, error) for parent, _ in matches)
        for parent in re.findall(
            r"\b([A-Za-z][A-Za-z0-9_]*)\s+(?:with|must have|should have)\s+(?:explicit\s+)?child fields",
            error,
        ):
            parent_candidates.append((parent, error))
        for parent in re.findall(r"\beach array:\s*([A-Za-z0-9_,\s]+)", error, flags=re.IGNORECASE):
            for item in parent.split(","):
                item = item.strip()
                if item:
                    parent_candidates.append((item, error))
        for family_items in re.findall(
            r"(?:family|families|\u5bb6\u65cf)[\uff1a:]\s*([^\u3002\n\uff1b;]+)",
            error,
            flags=re.IGNORECASE,
        ):
            for item in re.split(r"[,\uff0c\u3001\s]+", family_items):
                item = item.strip()
                if item:
                    parent_candidates.append((item, error))
        if not matches and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]+", error.strip()):
            parent_candidates.append((error.strip(), error))

    skip = {
        "section0",
        "section1",
        "section2",
        "section3",
        "section4",
        "value",
        "unit",
        "source_figure",
        "source_table",
        "source_text",
        "confidence",
        "measurement_conditions",
    }
    for parent, source_text in parent_candidates:
        parent = parent.strip()
        if parent.lower() in skip or not is_repair_parent_candidate(parent):
            continue
        section_id = section_for_repair_parent(parent, source_text)
        parent_path = material_section_path(section_id, parent)
        add_repair_field(
            registry,
            existing_paths,
            parent_path,
            section_id,
            parent,
            f"Deterministic supervisor repair added parent field from critic directive: {parent}.",
        )
        for child in repair_children_from_text(parent, source_text):
            add_repair_field(
                registry,
                existing_paths,
                f"{parent_path}.{child}",
                section_id,
                child,
                f"Deterministic supervisor repair expanded {parent} with child field {child}.",
            )

    dotted_field_candidates = []
    for error in errors:
        dotted_field_candidates.extend(
            re.findall(r"\b([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*){1,})\b", error)
        )
    for dotted_path in dotted_field_candidates:
        dotted_path = dotted_path.strip(".")
        dotted_lower = dotted_path.lower()
        if dotted_lower.startswith(("magnetic_field.", "temperature.", "pressure.")):
            continue
        if any(segment in {"e", "g", "eg", "e.g", "example", "examples"} for segment in dotted_lower.split(".")):
            continue
        if dotted_lower.startswith(("material_info.", "paper_info.", "section5.")):
            section_id = dotted_path.rsplit(".", 1)[0]
        elif dotted_lower.startswith("dimensional_features."):
            section_id = "material_info.section0"
        elif dotted_lower.startswith(("synthesis_process_steps.", "conditions.", "annealing.")):
            section_id = "material_info.section2"
        else:
            section_id = "material_info.section1"
        add_repair_field(
            registry,
            existing_paths,
            material_section_path(section_id, dotted_path),
            section_id,
            dotted_path.split(".")[-1],
            f"Deterministic supervisor repair added explicit dotted field path from critic directive: {dotted_path}.",
        )

    list_section_specs = [
        ("section2", "material_info.section2", "direct"),
        ("section3", "material_info.section3", "technique"),
        ("section4", "material_info.section4", "curve"),
        ("section5", "section5", "direct"),
        ("paper_info", "paper_info.metadata", "direct"),
    ]
    child_templates = {
        "technique": ["figure", "raw_data", "observed_object", "extracted_information", "sample_variant", "source_text"],
        "curve": [
            "figure",
            "variable",
            "temperature",
            "magnetic_field",
            "field_direction",
            "protocol",
            "frequency",
            "sweep_rate",
            "raw_data",
            "sample_variant",
            "source_text",
        ],
    }
    for error in errors:
        chunks = [*re.split(r"\n\s*\d+\.\s*", error), *re.split(r"[\u3002\n]", error)]
        for chunk in chunks:
            sentence_lower = chunk.lower()
            if not any(
                token in sentence_lower
                for token in [
                    "\u4fdd\u5b58",
                    "\u5305\u62ec",
                    "\u8986\u76d6",
                    "\u4fdd\u7559",
                    "\u5141\u8bb8",
                    "required",
                    "include",
                    "cover",
                ]
            ):
                continue
            for marker, section_id, mode in list_section_specs:
                if marker not in sentence_lower:
                    continue
                body = chunk
                tokens = [
                    token.strip().strip(".,;:()[]{}\uff0c\u3002\uff1b\uff1a\u3001")
                    for token in re.split(r"[,\uff0c\u3001\s]+", body)
                    if token.strip()
                ]
                for token in tokens:
                    if not is_repair_parent_candidate(token):
                        continue
                    if mode in child_templates:
                        parent_path = material_section_path(section_id, token)
                        add_repair_field(
                            registry,
                            existing_paths,
                            parent_path,
                            section_id,
                            token,
                            f"Deterministic supervisor repair added listed section object from expert directive: {token}.",
                        )
                        for child in child_templates[mode]:
                            add_repair_field(
                                registry,
                                existing_paths,
                                f"{parent_path}.{child}",
                                section_id,
                                child,
                                f"Deterministic supervisor repair expanded listed section object {token} with {child}.",
                            )
                    else:
                        target_section = section_id
                        target_token = token
                        if section_id == "paper_info.metadata" and token.lower() in {"raw_data", "code", "cif"}:
                            target_section = "paper_info.resources"
                        add_repair_field(
                            registry,
                            existing_paths,
                            material_section_path(target_section, target_token),
                            "paper_info" if target_section.startswith("paper_info.") else target_section,
                            target_token,
                            f"Deterministic supervisor repair added listed field from expert directive: {token}.",
                        )

    original_count = len(registry)
    registry = compact_field_registry(registry)
    schema["field_registry"] = registry
    schema.setdefault("schema_quality_notes", []).append(
        {
            "type": "field_inventory_preservation",
            "original_field_count": original_count,
            "final_field_count": len(registry),
            "truncated": len(registry) < original_count,
            "policy": (
                "Preserve every valid task-derived independently queryable field after path validation and "
                "deduplication. Field count is evaluated through coverage, utility, redundancy, and binding "
                "quality rather than an arbitrary fixed-size truncation."
            ),
        }
    )
    if not schema.get("top_level_keys"):
        schema["top_level_keys"] = [
            {"key": "material_info", "description": "Sectioned material database record."}
        ]
    outputs["schema_design_module"] = schema

    module_attempts = deepcopy(state.get("module_attempts", {}))
    module_attempts["schema_design_module"] = [
        *module_attempts.get("schema_design_module", []),
        {
            "round": len(module_attempts.get("schema_design_module", [])) + 1,
            "prompt": "deterministic_schema_repair_from_supervisor_errors",
            "raw_response": json.dumps(schema, ensure_ascii=False),
        },
    ]
    module_errors = deepcopy(state.get("module_errors", {}))
    module_errors["schema_design_module"] = []
    return merge_update(
        state,
        {
            "module_outputs": outputs,
            "module_attempts": module_attempts,
            "module_errors": module_errors,
        },
    )


PATCH_CHANGE_KEYS = {
    "section_id",
    "field_name",
    "description",
    "extraction_notes",
    "data_type",
    "required",
    "source_basis",
    "concept_ids",
    "figure_constraint",
    "object_contract",
    "reason",
    "inclusion_rule",
    "absence_rule",
    "evidence_requirements",
    "relation_constraints",
    "core_field",
}

PATCH_NESTED_CHANGE_KEYS = {
    "object_contract.object_kind",
    "object_contract.required_subfields",
    "object_contract.fields",
}


def _canonical_patch_path(raw_path, fallback_section_id="material_info.section1"):
    path = canonicalize_rebuild_field_path(raw_path, fallback_section_id)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*", path):
        return ""
    return path


def _normalize_patch_field(raw_field, field_path, state, reason):
    raw_field = deepcopy(raw_field) if isinstance(raw_field, dict) else {}
    section_id = section_agent.canonical_section_id(
        raw_field.get("section_id") or infer_section_from_field_path(field_path)
    )
    if field_path.startswith("section5."):
        section_id = "section5"
    source_basis = raw_field.get("source_basis") or infer_source_basis_from_field_path(field_path)
    if isinstance(source_basis, str):
        source_basis = [source_basis]
    source_basis = list(dict.fromkeys(str(item) for item in source_basis if str(item).strip())) or ["text"]
    data_type = str(raw_field.get("data_type") or infer_data_type_from_field_path(field_path))
    field = {
        "field_path": field_path,
        "section_id": section_id,
        "field_name": str(raw_field.get("field_name") or field_path.rsplit(".", 1)[-1]),
        "description": str(raw_field.get("description") or ""),
        "extraction_notes": str(
            raw_field.get("extraction_notes")
            or "Extract only explicit literature evidence; preserve entity, sample, conditions, units, and source location without cross-record merging."
        ),
        "data_type": data_type,
        "required": bool(raw_field.get("required", False)),
        "source_basis": source_basis,
        "concept_ids": list(dict.fromkeys(raw_field.get("concept_ids") or [])),
        "figure_constraint": raw_field.get("figure_constraint"),
        "reason": str(raw_field.get("reason") or reason),
    }
    for key in (
        "inclusion_rule",
        "absence_rule",
        "evidence_requirements",
        "relation_constraints",
        "core_field",
    ):
        if key in raw_field:
            field[key] = deepcopy(raw_field[key])
    if field["figure_constraint"] is None:
        field["figure_constraint"] = infer_rebuild_figure_constraint(
            field_path,
            section_id,
            source_basis,
        )
    if "object" in data_type.lower():
        field["object_contract"] = merge_structured_object_contract(
            raw_field.get("object_contract"),
            field_path,
            [],
        )
    return field


def validate_schema_patch_operations(schema, operations, state):
    schema = canonicalize_schema_definition(schema)
    registry = {
        str(item.get("field_path")): item
        for item in schema.get("field_registry") or []
        if isinstance(item, dict) and item.get("field_path")
    }
    approved_roots = approved_schema_roots_from_state(state)
    shared = state.get("shared_context") or {}
    subjective = (state.get("module_outputs") or {}).get(
        "subjective_supervisor_module"
    ) or {}
    requirement_contract = section_agent.normalize_requirement_contract(
        shared,
        subjective,
    )
    valid_concept_ids = {
        str(item.get("concept_id"))
        for item in requirement_contract.get("concepts", []) or []
        if isinstance(item, dict) and item.get("concept_id")
    }
    errors = []
    normalized = []
    simulated_paths = set(registry)
    for index, raw_operation in enumerate(operations or []):
        if not isinstance(raw_operation, dict):
            errors.append(f"patch[{index}] must be an object")
            continue
        operation = deepcopy(raw_operation)
        op = str(operation.get("op") or "").strip().lower()
        field_path = _canonical_patch_path(
            operation.get("field_path")
            or (operation.get("field") or {}).get("field_path")
        )
        if op not in {"add_field", "update_field", "remove_field", "move_field"}:
            errors.append(f"patch[{index}] has unsupported op: {op or '<empty>'}")
            continue
        if not field_path:
            errors.append(f"patch[{index}] has an invalid field_path")
            continue
        if field_path.split(".", 1)[0] not in approved_roots:
            errors.append(f"patch[{index}] uses unapproved root: {field_path}")
            continue
        if op == "add_field" and field_path in simulated_paths:
            errors.append(f"patch[{index}] add_field already exists: {field_path}")
            continue
        if op in {"update_field", "remove_field", "move_field"} and field_path not in simulated_paths:
            errors.append(f"patch[{index}] target does not exist: {field_path}")
            continue
        if op == "update_field":
            changes = operation.get("changes")
            if not isinstance(changes, dict) or not changes:
                errors.append(f"patch[{index}] update_field needs non-empty changes")
                continue
            unknown = sorted(
                set(changes) - PATCH_CHANGE_KEYS - PATCH_NESTED_CHANGE_KEYS
            )
            if unknown:
                errors.append(f"patch[{index}] changes unsupported keys: {unknown}")
                continue
        concept_ids = []
        if op == "add_field" and isinstance(operation.get("field"), dict):
            concept_ids = operation["field"].get("concept_ids", []) or []
        elif op == "update_field" and isinstance(operation.get("changes"), dict):
            concept_ids = operation["changes"].get("concept_ids", []) or []
        invalid_concept_ids = sorted(
            {
                str(concept_id)
                for concept_id in concept_ids
                if str(concept_id) not in valid_concept_ids
            }
        )
        if invalid_concept_ids:
            errors.append(
                f"patch[{index}] concept_ids are not in requirement_contract: "
                f"{invalid_concept_ids}"
            )
            continue
        if op == "move_field":
            target_path = _canonical_patch_path(operation.get("target_path"))
            if not target_path:
                errors.append(f"patch[{index}] move_field has an invalid target_path")
                continue
            if target_path.split(".", 1)[0] not in approved_roots:
                errors.append(f"patch[{index}] move_field uses unapproved target root: {target_path}")
                continue
            if target_path in simulated_paths and target_path != field_path:
                errors.append(f"patch[{index}] move_field target already exists: {target_path}")
                continue
            operation["target_path"] = target_path
            simulated_paths.remove(field_path)
            simulated_paths.add(target_path)
        elif op == "add_field":
            simulated_paths.add(field_path)
        elif op == "remove_field":
            simulated_paths.remove(field_path)
        operation["field_path"] = field_path
        normalized.append(operation)
    return normalized, errors


def schema_coverage_report_for_state(schema, state):
    shared = state.get("shared_context") or {}
    outputs = state.get("module_outputs") or {}
    subjective = outputs.get("subjective_supervisor_module") or {}
    requirement_contract = section_agent.normalize_requirement_contract(shared, subjective)
    entity_registry = section_agent.normalize_entity_registry(
        subjective,
        requirement_contract,
    )
    field_plan = outputs.get("field_planning_module") or {}
    return section_agent.build_coverage_report(
        {
            "requirement_contract": requirement_contract,
            "entity_registry": entity_registry,
            "schema_definition": schema or {},
            "reference_field_contract": shared.get("reference_field_contract") or {},
            "reference_field_audit": field_plan.get("reference_field_audit") or {},
            "field_plan_traceability": section_agent.build_field_plan_traceability(
                field_plan
            ),
        }
    )


def schema_contract_regressions(before_schema, after_schema, state):
    """Return losses against the run's own task contract, never external gold."""
    before = schema_coverage_report_for_state(before_schema, state)
    after = schema_coverage_report_for_state(after_schema, state)
    comparisons = {
        "required_concepts": (
            before.get("required_concept_coverage", {}).get("missing_concept_ids", []),
            after.get("required_concept_coverage", {}).get("missing_concept_ids", []),
        ),
        "entity_owners": (
            before.get("entity_owner_coverage", {}).get("missing_entity_ids", []),
            after.get("entity_owner_coverage", {}).get("missing_entity_ids", []),
        ),
        "object_contracts": (
            before.get("structured_object_completeness", {}).get("incomplete_fields", []),
            after.get("structured_object_completeness", {}).get("incomplete_fields", []),
        ),
        "evidence_contracts": (
            before.get("evidence_contract_coverage", {}).get("missing_concept_ids", []),
            after.get("evidence_contract_coverage", {}).get("missing_concept_ids", []),
        ),
        "query_requirements": (
            before.get("query_requirement_traceability", {}).get("missing_requirement_ids", []),
            after.get("query_requirement_traceability", {}).get("missing_requirement_ids", []),
        ),
        "reference_fields": (
            before.get("reference_field_coverage", {}).get("missing_applicable_paths", []),
            after.get("reference_field_coverage", {}).get("missing_applicable_paths", []),
        ),
    }
    regressions = []
    for label, (before_items, after_items) in comparisons.items():
        newly_missing = sorted(set(map(str, after_items)) - set(map(str, before_items)))
        if newly_missing:
            regressions.append(f"{label}:{','.join(newly_missing)}")
    return regressions


def apply_schema_patch_operations(state, operations, source="structured_schema_patch"):
    outputs = deepcopy(state.get("module_outputs", {}))
    schema = canonicalize_schema_definition(outputs.get("schema_design_module") or {})
    normalized, errors = validate_schema_patch_operations(schema, operations, state)
    if errors:
        return merge_update(
            state,
            {
                "validation_errors": [f"schema_patch_rejected:{item}" for item in errors],
                "status": "awaiting_supervisor_decision",
                "last_error_node": "schema_design",
                "last_error_type": "schema_validation_failure",
            },
        )

    registry = [
        deepcopy(item)
        for item in schema.get("field_registry") or []
        if isinstance(item, dict) and item.get("field_path")
    ]
    by_path = {str(item["field_path"]): item for item in registry}
    applied = []
    for operation in normalized:
        op = operation["op"]
        field_path = operation["field_path"]
        reason = str(operation.get("reason") or source)
        if op == "add_field":
            field = _normalize_patch_field(
                operation.get("field"),
                field_path,
                state,
                reason,
            )
            registry.append(field)
            by_path[field_path] = field
        elif op == "update_field":
            field = by_path[field_path]
            changes = deepcopy(operation["changes"])
            for key, value in changes.items():
                if key in PATCH_NESTED_CHANGE_KEYS:
                    parent, child = key.split(".", 1)
                    nested = deepcopy(field.get(parent))
                    if not isinstance(nested, dict):
                        nested = {}
                    nested[child] = value
                    field[parent] = nested
                else:
                    field[key] = value
            field["field_name"] = str(field.get("field_name") or field_path.rsplit(".", 1)[-1])
            field["section_id"] = section_agent.canonical_section_id(
                field.get("section_id") or infer_section_from_field_path(field_path)
            )
            if field_path.startswith("section5."):
                field["section_id"] = "section5"
            field["reason"] = reason
            if "object" in str(field.get("data_type") or "").lower():
                field["object_contract"] = merge_structured_object_contract(
                    field.get("object_contract"),
                    field_path,
                    [],
                )
            if "figure" in (field.get("source_basis") or []) and not field.get("figure_constraint"):
                field["figure_constraint"] = infer_rebuild_figure_constraint(
                    field_path,
                    field["section_id"],
                    field.get("source_basis") or [],
                )
        elif op == "remove_field":
            registry = [item for item in registry if item.get("field_path") != field_path]
            by_path.pop(field_path, None)
        elif op == "move_field":
            target_path = operation["target_path"]
            field = by_path.pop(field_path)
            field["field_path"] = target_path
            field["field_name"] = target_path.rsplit(".", 1)[-1]
            target_root = target_path.split(".", 1)[0]
            if target_path == field_path and section_agent.is_safe_entity_owner_key(target_root):
                field["section_id"] = target_root
            else:
                field["section_id"] = infer_section_from_field_path(target_path)
            field["reason"] = reason
            by_path[target_path] = field
        applied.append(
            {
                "op": op,
                "field_path": field_path,
                "target_path": operation.get("target_path", ""),
                "reason": reason,
            }
        )

    schema["field_registry"] = compact_field_registry(registry)
    schema = synchronize_top_level_keys(schema)
    contract_regressions = schema_contract_regressions(
        outputs.get("schema_design_module") or {},
        schema,
        state,
    )
    if contract_regressions:
        rejected_hashes = list(state.get("rejected_critic_patch_hashes") or [])
        patch_hash = critic_patch_hash(normalized)
        if patch_hash not in rejected_hashes:
            rejected_hashes.append(patch_hash)
        return merge_update(
            state,
            {
                "validation_errors": [
                    f"schema_patch_contract_regression:{item}"
                    for item in contract_regressions
                ],
                "rejected_critic_patch_hashes": rejected_hashes,
                "status": "awaiting_supervisor_decision",
                "last_error_node": "schema_design",
                "last_error_type": "schema_validation_failure",
            },
        )
    schema.setdefault("schema_patch_audit", []).append(
        {
            "source": source,
            "base_schema_revision": int(state.get("schema_revision", 0) or 0),
            "applied_operations": applied,
        }
    )
    outputs["schema_design_module"] = schema
    attempts = deepcopy(state.get("module_attempts", {}))
    prior = list(attempts.get("schema_design_module") or [])
    attempts["schema_design_module"] = [
        *prior,
        {
            "round": len(prior) + 1,
            "prompt": source,
            "raw_response": json.dumps(
                {"patch_operations": normalized, "schema": schema},
                ensure_ascii=False,
            ),
        },
    ]
    module_errors = deepcopy(state.get("module_errors", {}))
    module_errors["schema_design_module"] = []
    return merge_update(
        state,
        {
            "module_outputs": outputs,
            "module_attempts": attempts,
            "module_errors": module_errors,
            "schema_revision": int(state.get("schema_revision", 0) or 0) + 1,
            "critic_reviewed_schema_revision": -1,
            "validation_errors": [],
        },
    )


def _error_field_paths(error, prefix):
    text = str(error)
    if not text.startswith(prefix):
        return []
    payload = text[len(prefix) :]
    return list(
        dict.fromkeys(
            re.findall(
                r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+",
                payload,
            )
        )
    )


def structured_patch_operations_from_errors(state):
    outputs = state.get("module_outputs") or {}
    schema = outputs.get("schema_design_module") or {}
    registry = {
        str(item.get("field_path")): item
        for item in schema.get("field_registry") or []
        if isinstance(item, dict) and item.get("field_path")
    }
    operations = []
    for error in state.get("repair_instructions") or state.get("validation_errors") or []:
        for field_path in _error_field_paths(error, "coverage:incomplete_object_contracts:"):
            field = registry.get(field_path)
            if not field:
                continue
            operations.append(
                {
                    "op": "update_field",
                    "field_path": field_path,
                    "changes": {
                        "data_type": field.get("data_type")
                        if "object" in str(field.get("data_type") or "").lower()
                        else "object",
                        "object_contract": merge_structured_object_contract(
                            field.get("object_contract"),
                            field_path,
                            [],
                        ),
                    },
                    "reason": "Coverage gate required a complete executable object contract.",
                }
            )
        for field_path in _error_field_paths(error, "coverage:missing_reference_fields:"):
            if field_path in registry:
                continue
            reference_spec = next(
                (
                    item
                    for item in (state.get("shared_context") or {})
                    .get("reference_field_contract", {})
                    .get("leaf_fields", [])
                    if isinstance(item, dict)
                    and section_agent.normalize_reference_field_path(item.get("path"))
                    == section_agent.normalize_reference_field_path(field_path)
                ),
                {},
            )
            operations.append(
                {
                    "op": "add_field",
                    "field_path": field_path,
                    "field": {
                        "field_path": field_path,
                        "section_id": infer_section_from_field_path(field_path),
                        "field_name": field_path.rsplit(".", 1)[-1],
                        "data_type": reference_spec.get("data_type") or infer_data_type_from_field_path(field_path),
                        "description": reference_spec.get("description_hint") or "",
                        "extraction_notes": reference_spec.get("extraction_notes") or "",
                        "required": False,
                        "source_basis": infer_source_basis_from_field_path(field_path),
                        "concept_ids": [],
                        "figure_constraint": None,
                        "reason": "Applicable reference-field coverage gate required this exact leaf.",
                    },
                    "reason": "Applicable reference-field coverage gate required this exact leaf.",
                }
            )
        for field_path in _error_field_paths(error, "coverage:redundant_fields:"):
            if field_path not in registry:
                continue
            operations.append(
                {
                    "op": "remove_field",
                    "field_path": field_path,
                    "reason": "Coverage gate identified an exact semantic-contract duplicate.",
                }
            )
    return operations


def deterministic_schema_repair(state):
    """Apply only version-bound structured patches; never infer fields from prose."""
    schema = deepcopy((state.get("module_outputs") or {}).get("schema_design_module") or {})
    unapproved_roots = schema_has_unapproved_roots(schema, state)
    if unapproved_roots:
        return compile_schema_from_field_planning(
            state,
            f"unapproved schema roots required a clean rebuild: {unapproved_roots}",
        )

    critic = (state.get("module_outputs") or {}).get("specialization_critic_module") or {}
    current_revision = int(state.get("schema_revision", 0) or 0)
    reviewed_revision = int(
        critic.get("reviewed_schema_revision", state.get("critic_reviewed_schema_revision", -1))
        or 0
    )
    operations = []
    if reviewed_revision == current_revision:
        candidate_operations = list(critic.get("patch_operations") or [])
        if critic_patch_hash(candidate_operations) not in set(
            state.get("rejected_critic_patch_hashes") or []
        ):
            operations = candidate_operations
    if not operations:
        operations = structured_patch_operations_from_errors(state)
    if operations:
        return apply_schema_patch_operations(state, operations)
    return merge_update(
        state,
        {
            "validation_errors": [
                "schema_repair_requires_model: no safe structured patch was available; "
                "repair the current schema without rebuilding it from field planning"
            ],
            "status": "awaiting_schema_model_repair",
            "last_error_node": "schema_design_repair",
            "last_error_type": "schema_model_repair_required",
        },
    )


def schema_design_repair_node(state):
    if (
        not state.get("module_outputs", {}).get("schema_design_module")
        and state.get("module_outputs", {}).get("field_planning_module")
    ):
        update = compile_schema_from_field_planning(
            state,
            "schema design model output was empty after a failed or timed-out call",
        )
        update = clear_downstream_outputs(
            update,
            ["specialization_critic_module", "aggregation"],
        )
        update["validation_errors"] = []
        update["repair_instructions"] = []
        update["status"] = "running"
        update["next_node"] = "specialization_critic"
        update["last_error_node"] = ""
        update["last_error_type"] = ""
        update = pause_for_schema_inspection_if_due(update, "specialization_critic")
        return write_state_snapshot(
            update,
            "schema_design_repair",
            update["next_node"],
        )

    if state.get("last_error_type") == "schema_validation_failure" and state.get("module_outputs", {}).get("schema_design_module"):
        update = deterministic_schema_repair(state)
        if update.get("last_error_type") == "schema_model_repair_required":
            update = call_module_node(
                state,
                "schema_design_module",
                build_schema_repair_prompt(state),
                "schema_design_repair",
                "specialization_critic",
            )
        if update.get("validation_errors"):
            update["next_node"] = "supervisor_router"
            return write_state_snapshot(
                update,
                "schema_design_repair",
                "supervisor_router",
            )
        update = clear_downstream_outputs(
            update,
            ["specialization_critic_module", "aggregation"],
        )
        update["validation_errors"] = []
        update["repair_instructions"] = []
        update["status"] = "running"
        update["next_node"] = "specialization_critic"
        update["last_error_node"] = ""
        update["last_error_type"] = ""
        update = pause_for_schema_inspection_if_due(update, "specialization_critic")
        return write_state_snapshot(update, "schema_design_repair", update["next_node"])

    update = call_module_node(
        state,
        "schema_design_module",
        build_schema_repair_prompt(state),
        "schema_design_repair",
        "specialization_critic",
    )
    if not update.get("validation_errors"):
        update = clear_downstream_outputs(
            update,
            ["specialization_critic_module", "aggregation"],
        )
        update["validation_errors"] = []
        update["repair_instructions"] = []
        update["status"] = "running"
        update["next_node"] = "specialization_critic"
        update["last_error_node"] = ""
        update["last_error_type"] = ""
        update = pause_for_schema_inspection_if_due(update, "specialization_critic")
    return write_state_snapshot(update, "schema_design_repair", update.get("next_node", "specialization_critic"))


def _legacy_local_schema_coverage_sufficient(schema_result, critic_result, state=None):
    unapproved_roots = schema_has_unapproved_roots(schema_result or {}, state or {})
    if unapproved_roots:
        return False, [f"unapproved schema roots present: {unapproved_roots}"]
    registry = schema_result.get("field_registry") or []
    field_paths = [str(item.get("field_path") or "").lower() for item in registry]
    if not field_paths:
        return False, ["field_registry is empty"]

    critic_text = " ".join(
        [
            *map(str, critic_result.get("missing_concepts") or []),
            *map(str, critic_result.get("redo_directives") or []),
            *map(str, critic_result.get("structural_weaknesses") or []),
        ]
    ).lower()
    candidate_parents = set()
    scalar_or_evidence_terms = {
        "magnetic_ordering_type",
        "material_family",
        "sample_identifier",
        "measurement_conditions",
        "evidence",
        "source_figure",
        "source_table",
        "source_text",
        "confidence",
        "field_registry",
    }
    for parent, _child in re.findall(
        r"\b([a-z][a-z0-9_]*)\.(value|raw_value|unit|temperature|magnetic_field|source_figure|source_text|figure|raw_data|conditions|measurement_conditions)\b",
        critic_text,
    ):
        if is_repair_parent_candidate(parent) and parent not in scalar_or_evidence_terms:
            candidate_parents.add(parent)
    for parent in re.findall(r"\b([a-z][a-z0-9_]*)\s+(?:with|must have|should have)\s+(?:explicit\s+)?child fields", critic_text):
        if is_repair_parent_candidate(parent) and parent not in scalar_or_evidence_terms:
            candidate_parents.add(parent)
    for parent in re.findall(r"\b([a-z][a-z0-9_]*)\s*\(", critic_text):
        if is_repair_parent_candidate(parent) and parent not in scalar_or_evidence_terms:
            candidate_parents.add(parent)

    missing = []
    for parent in sorted(candidate_parents):
        child_paths = [path for path in field_paths if f".{parent}." in path]
        parent_paths = [path for path in field_paths if path.endswith(f".{parent}") or path == parent]
        if not child_paths and parent_paths:
            # Medium-granularity schemas keep value/unit/conditions/evidence as
            # the object contract for the parent field instead of flattening
            # every leaf into field_registry.
            continue
        if not child_paths:
            missing.append(f"{parent}: no child paths")
            continue
        if "section1" in " ".join(child_paths):
            required_any = [
                ("value", [".value", ".raw_value"]),
                ("evidence", [".source_figure", ".source_table", ".source_text"]),
                ("conditions", [".measurement_conditions", ".conditions"]),
            ]
            for label, options in required_any:
                if not any(any(option in path for option in options) for path in child_paths):
                    missing.append(f"{parent}: missing {label} child")

    if "source_figure" in critic_text and not any(".source_figure" in path for path in field_paths):
        missing.append("no source_figure fields")
    has_condition_contract = any(
        ".measurement_conditions" in path
        or ".conditions" in path
        or path.startswith("condition_set_info.")
        or path.startswith("condition_entry_info.")
        or path.endswith("_condition_set_id")
        for path in field_paths
    )
    if "measurement_conditions" in critic_text and not has_condition_contract:
        missing.append("no measurement condition fields")
    return not missing, missing


def local_schema_coverage_sufficient(schema_result, critic_result, state=None):
    """Perform deterministic structural checks without interpreting critic prose."""
    state = state or {}
    unapproved_roots = schema_has_unapproved_roots(schema_result or {}, state)
    if unapproved_roots:
        return False, [f"unapproved schema roots present: {unapproved_roots}"]
    registry = [
        item
        for item in (schema_result or {}).get("field_registry") or []
        if isinstance(item, dict) and item.get("field_path")
    ]
    if not registry:
        return False, ["field_registry is empty"]
    missing = []
    for field in registry:
        slots = section_agent.object_contract_missing_slots(field)
        if slots:
            missing.append(
                f"{field.get('field_path')}: incomplete object contract ({', '.join(sorted(slots))})"
            )
        if "figure" in (field.get("source_basis") or []) and not isinstance(
            field.get("figure_constraint"), dict
        ):
            missing.append(f"{field.get('field_path')}: missing figure_constraint")

    if critic_result.get("redo_needed") or critic_result.get("is_generic"):
        operations = critic_result.get("patch_operations")
        if not isinstance(operations, list) or not operations:
            missing.append("critic requested redesign without structured patch_operations")
        else:
            _normalized, patch_errors = validate_schema_patch_operations(
                schema_result,
                operations,
                state,
            )
            missing.extend(f"critic patch invalid: {item}" for item in patch_errors)
            if not patch_errors:
                missing.append("critic supplied pending structured patch_operations")
    return not missing, missing


def specialization_critic_node(state):
    outputs = state["module_outputs"]
    schema_result = outputs.get("schema_design_module") or {}
    registry = schema_result.get("field_registry") or []
    human_advice_text = str(
        state.get("human_advice")
        or state.get("applied_human_advice")
        or state.get("shared_context", {}).get("human_advice")
        or ""
    )
    if (
        human_advice_text
        and registry
        and bool(state.get("allow_local_critic_bypass_after_human_advice", False))
    ):
        critic_result = {
            "specialization_status": "local_structural_review_after_human_advice",
            "redo_needed": False,
            "is_generic": False,
            "missing_concepts": [human_advice_text],
            "redo_directives": [],
            "structural_weaknesses": [],
            "field_utility_audit": {
                "reviewed_field_count": len(registry),
                "decision": "pass",
                "unsupported_fields": [],
                "redundant_fields": [],
                "alias_or_unit_variant_fields": [],
                "derivable_duplicate_fields": [],
                "rationale": (
                    "Local structural fallback reviewed traceability only; "
                    "official benchmark use remains disabled for this path."
                ),
            },
            "patch_operations": [],
            "reviewed_schema_revision": int(state.get("schema_revision", 0) or 0),
        }
        coverage_ok, coverage_missing = local_schema_coverage_sufficient(
            schema_result,
            critic_result,
            state,
        )
        critic_result["local_structural_validation"] = {
            "status": "passed" if coverage_ok else "failed",
            "missing": coverage_missing[:20],
            "reason": (
                "Large schema repaired from expert advice was checked locally to avoid blocking the graph on an oversized critic call."
                if coverage_ok
                else "Large schema repaired from expert advice still misses explicit child paths requested by expert advice."
            ),
        }
        update = merge_update(
            state,
            {
                "module_outputs": {**outputs, "specialization_critic_module": critic_result},
                "validation_errors": [],
                "status": "running",
                "next_node": "aggregation",
            },
        )
        if not coverage_ok:
            update["validation_errors"] = [
                "specialization_critic local structural review requested schema redesign",
                *coverage_missing[:20],
            ]
            update["status"] = "awaiting_supervisor_decision"
            update["last_error_node"] = "schema_design"
            update["last_error_type"] = "schema_validation_failure"
            update["next_node"] = "supervisor_router"
            return write_state_snapshot(update, "specialization_critic", "supervisor_router")
        return write_state_snapshot(update, "specialization_critic", "aggregation")

    update = call_module_node(
        state,
        "specialization_critic_module",
        build_specialization_critic_prompt(
            state["shared_context"],
            outputs["mechanism_requirement_module"],
            outputs["query_semantics_module"],
            outputs["evidence_model_module"],
            outputs["subjective_supervisor_module"],
            outputs["section_partition_module"],
            outputs["field_planning_module"],
            outputs["schema_design_module"],
            repair_feedback=state.get("repair_instructions") or [],
        ),
        "specialization_critic",
        "aggregation",
    )
    if update.get("validation_errors"):
        return update

    critic_result = deepcopy(
        update["module_outputs"].get("specialization_critic_module") or {}
    )
    utility_errors = section_agent.validate_critic_field_utility(
        critic_result,
        update["module_outputs"].get("schema_design_module") or {},
    )
    if utility_errors:
        update["validation_errors"] = utility_errors
        update["status"] = "awaiting_supervisor_decision"
        update["last_error_node"] = "specialization_critic"
        update["last_error_type"] = "schema_validation_failure"
        update["next_node"] = "supervisor_router"
        return write_state_snapshot(
            update,
            "specialization_critic",
            "supervisor_router",
        )
    reviewed_revision = int(update.get("schema_revision", 0) or 0)
    critic_result["reviewed_schema_revision"] = reviewed_revision
    update["module_outputs"]["specialization_critic_module"] = critic_result
    update["critic_reviewed_schema_revision"] = reviewed_revision
    if critic_result.get("redo_needed") or critic_result.get("is_generic"):
        patch_operations = critic_result.get("patch_operations") or []
        patch_hash = critic_patch_hash(patch_operations)
        _normalized, patch_errors = validate_schema_patch_operations(
            update["module_outputs"].get("schema_design_module") or {},
            patch_operations,
            update,
        )
        if patch_errors:
            rejected_hashes = list(update.get("rejected_critic_patch_hashes") or [])
            if patch_hash not in rejected_hashes:
                rejected_hashes.append(patch_hash)
            critic_result["patch_validation"] = {
                "status": "rejected",
                "patch_hash": patch_hash,
                "errors": list(patch_errors),
            }
            update["module_outputs"]["specialization_critic_module"] = critic_result
            feedback = [f"schema_patch_rejected:{item}" for item in patch_errors]
            update["rejected_critic_patch_hashes"] = rejected_hashes
            update["validation_errors"] = feedback
            update["repair_instructions"] = feedback
            update["status"] = "awaiting_supervisor_decision"
            update["last_error_node"] = "specialization_critic"
            update["last_error_type"] = "schema_validation_failure"
            update["next_node"] = "supervisor_router"
            return write_state_snapshot(
                update,
                "specialization_critic",
                "supervisor_router",
            )
        preview = apply_schema_patch_operations(
            update,
            patch_operations,
            source="critic_internal_task_contract_preview",
        )
        contract_errors = [
            str(error)
            for error in preview.get("validation_errors") or []
            if str(error).startswith("schema_patch_contract_regression:")
        ]
        if contract_errors:
            rejected_hashes = list(update.get("rejected_critic_patch_hashes") or [])
            if patch_hash not in rejected_hashes:
                rejected_hashes.append(patch_hash)
            critic_result["patch_validation"] = {
                "status": "rejected",
                "patch_hash": patch_hash,
                "errors": contract_errors,
                "contract_scope": "run_internal_task_contract_not_external_gold",
            }
            update["module_outputs"]["specialization_critic_module"] = critic_result
            update["rejected_critic_patch_hashes"] = rejected_hashes
            update["validation_errors"] = contract_errors
            update["repair_instructions"] = contract_errors
            update["status"] = "awaiting_supervisor_decision"
            update["last_error_node"] = "specialization_critic"
            update["last_error_type"] = "schema_validation_failure"
            update["next_node"] = "supervisor_router"
            return write_state_snapshot(
                update,
                "specialization_critic",
                "supervisor_router",
            )
        coverage_ok, coverage_missing = local_schema_coverage_sufficient(
            update["module_outputs"].get("schema_design_module") or {},
            critic_result,
            update,
        )
        if coverage_ok:
            critic_result = deepcopy(critic_result)
            critic_result["specialization_status"] = "pass_with_local_structural_validation"
            critic_result["redo_needed"] = False
            critic_result["is_generic"] = False
            critic_result["local_structural_validation"] = {
                "status": "passed",
                "reason": "Supervisor local validator found explicit child field paths and provenance links requested by critic.",
            }
            update["module_outputs"]["specialization_critic_module"] = critic_result
            update["validation_errors"] = []
            update["status"] = "running"
            update["next_node"] = "aggregation"
            return write_state_snapshot(update, "specialization_critic", "aggregation")
        critic_result["local_structural_validation"] = {
            "status": "failed",
            "missing": coverage_missing[:20],
        }
        update["module_outputs"]["specialization_critic_module"] = critic_result
        directives = critic_result.get("redo_directives") or []
        findings = []
        findings.extend(critic_result.get("missing_concepts") or [])
        findings.extend(critic_result.get("structural_weaknesses") or [])
        errors = [
            "specialization_critic requested schema redesign",
            *[str(item) for item in directives[:8]],
            *[str(item) for item in findings[:8]],
        ]
        update["validation_errors"] = errors
        update["status"] = "awaiting_supervisor_decision"
        update["last_error_node"] = "schema_design"
        update["last_error_type"] = "schema_validation_failure"
        update["next_node"] = "supervisor_router"
        return write_state_snapshot(update, "specialization_critic", "supervisor_router")

    return write_state_snapshot(update, "specialization_critic", "aggregation")


def aggregation_node(state):
    outputs = state["module_outputs"]
    args = dict_to_namespace(state["args"])
    schema_revision = int(state.get("schema_revision", 0) or 0)
    critic_revision = int(
        (outputs.get("specialization_critic_module") or {}).get(
            "reviewed_schema_revision",
            state.get("critic_reviewed_schema_revision", -1),
        )
        or 0
    )
    if critic_revision != schema_revision:
        errors = [
            "schema_revision_mismatch: specialization critic did not review the current schema revision "
            f"({critic_revision} != {schema_revision})"
        ]
        update = merge_update(
            state,
            {
                "validation_errors": errors,
                "status": "awaiting_supervisor_decision",
                "last_error_node": "specialization_critic",
                "last_error_type": "schema_validation_failure",
                "next_node": "supervisor_router",
            },
        )
        return write_state_snapshot(update, "aggregation", "supervisor_router")

    if getattr(args, "use_llm_aggregation", False):
        update = call_module_node(
            state,
            "aggregation",
            build_aggregation_prompt(
                state["shared_context"],
                outputs["locating_module"],
                outputs["mechanism_requirement_module"],
                outputs["query_semantics_module"],
                outputs["evidence_model_module"],
                outputs["subjective_supervisor_module"],
                outputs["topic_adaptation_module"],
                outputs["section_partition_module"],
                outputs["field_planning_module"],
                outputs["supervisor_module"],
                outputs["figure_classification_module"],
                outputs["schema_design_module"],
                outputs["specialization_critic_module"],
            ),
            "aggregation",
            "write_output",
        )
        if update.get("validation_errors"):
            return update
        aggregation_result = update["module_outputs"]["aggregation"]
    else:
        aggregation_result = section_agent.assemble_final_result_from_modules(
            state["shared_context"],
            outputs["locating_module"],
            outputs["query_semantics_module"],
            outputs["subjective_supervisor_module"],
            outputs["topic_adaptation_module"],
            outputs["section_partition_module"],
            outputs["field_planning_module"],
            outputs["supervisor_module"],
            outputs["figure_classification_module"],
            outputs["schema_design_module"],
            outputs["specialization_critic_module"],
        )
        module_outputs = deepcopy(outputs)
        module_attempts = deepcopy(state.get("module_attempts", {}))
        module_errors = deepcopy(state.get("module_errors", {}))
        module_outputs["aggregation"] = aggregation_result
        module_attempts["aggregation"] = [
            {
                "round": 1,
                "prompt": "deterministic_aggregation",
                "raw_response": json.dumps(aggregation_result, ensure_ascii=False),
            }
        ]
        module_errors["aggregation"] = []
        update = merge_update(
            state,
            {
                "module_outputs": module_outputs,
                "module_attempts": module_attempts,
                "module_errors": module_errors,
            },
        )

    final_result = section_agent.finalize_result(
        state["shared_context"],
        aggregation_result,
    )
    validation_errors = section_agent.validate_result(final_result)
    validation_errors.extend(
        section_agent.validate_specialization(
            state["shared_context"],
            update["module_outputs"],
            final_result,
        )
    )
    update["result"] = final_result
    update["validation_errors"] = validation_errors
    update["status"] = "success" if not validation_errors else "needs_review"
    if validation_errors:
        update["last_error_node"] = "aggregation"
        update["last_error_type"] = "schema_validation_failure"
    update["next_node"] = "human_advice_gate"
    return write_state_snapshot(update, "aggregation", "human_advice_gate")


def write_output_node(state):
    args = dict_to_namespace(state["args"])
    shared = state.get("shared_context", {})
    query_requirements = shared.get("query_requirements", [])
    module_outputs = state.get("module_outputs", {})
    output_inputs = {
        "task_contract": dict(section_agent.MATERIAL_LITERATURE_TASK_CONTRACT),
        "database_goal": args.database_goal,
        "discipline": args.discipline,
        "query_requirements": query_requirements,
        "key_description_path": str(Path(args.key_description_path)),
        "reference_papers": args.reference_papers,
        "human_expert_review_after_internal_pass": bool(
            getattr(args, "require_human_advice_before_supervisor", False) or state.get("human_advice")
        ),
        "human_advice_provided": bool(state.get("human_advice")),
        "structured_protocol_enabled": bool(shared.get("structured_protocol_enabled")),
        "structured_concept_seeding_only": bool(shared.get("structured_concept_seeding_only")),
    }
    errors = state.get("validation_errors", [])
    result = state.get("result")
    if result is None and module_outputs.get("schema_design_module"):
        try:
            result = section_agent.finalize_result(
                shared,
                section_agent.assemble_final_result_from_modules(
                    shared,
                    module_outputs["locating_module"],
                    module_outputs["query_semantics_module"],
                    module_outputs["subjective_supervisor_module"],
                    module_outputs["topic_adaptation_module"],
                    module_outputs["section_partition_module"],
                    module_outputs["field_planning_module"],
                    module_outputs["supervisor_module"],
                    module_outputs["figure_classification_module"],
                    module_outputs["schema_design_module"],
                    module_outputs.get("specialization_critic_module", {}),
                ),
            )
        except KeyError:
            result = None
    output_status = state.get("status", "needs_review")
    if requested_stop_after_module(state) and not errors:
        output_status = "partial_success"
    provenance = state.get("execution_provenance")
    if not isinstance(provenance, dict):
        completed_modules = [
            name
            for name in section_agent.REQUIRED_PIPELINE_MODULES
            if isinstance(module_outputs.get(name), dict)
            and not (state.get("module_errors", {}).get(name) or [])
        ]
        all_required_modules_completed = len(completed_modules) == len(
            section_agent.REQUIRED_PIPELINE_MODULES
        )
        provenance = {
            "mode": "online_full_pipeline"
            if all_required_modules_completed
            else "online_partial_pipeline",
            "fallback_used": False,
            "all_required_modules_completed": all_required_modules_completed,
            "completed_module_count": len(completed_modules),
            "required_module_count": len(section_agent.REQUIRED_PIPELINE_MODULES),
            "request_timeout_seconds": getattr(args, "request_timeout", None),
            "module_max_tokens": dict(section_agent.MODULE_MAX_TOKENS),
        }
    output = {
        "step": "step8_section_design_agent_langgraph_human_gate",
        "framework": STEP8_FRAMEWORK,
        "run_identity": state.get("run_identity"),
        "schema_revision": int(state.get("schema_revision", 0) or 0),
        "critic_reviewed_schema_revision": int(
            state.get("critic_reviewed_schema_revision", -1) or 0
        ),
        "inputs": output_inputs,
        "reference_paper_context_used": shared.get("reference_paper_context", ""),
        "agent_flow": section_agent.build_agent_flow_trace(
            output_inputs,
            module_outputs,
            result,
            errors,
        ),
        "protocol_messages": section_agent.build_step8_protocol_messages(
            output_inputs,
            module_outputs,
            result,
            errors,
            state.get("module_errors", {}),
        ),
        "module_outputs": module_outputs,
        "module_errors": state.get("module_errors", {}),
        "result": result,
        "validation_errors": errors,
        "status": output_status,
        "execution_provenance": provenance,
        "attempts": {
            "modules": state.get("module_attempts", {}),
            "final_redo": [],
        },
    }
    output["protocol_validation"] = section_agent.agent_protocol.validate_message_list(output["protocol_messages"])
    output_path = Path(args.output)
    run_artifact_guard.atomic_write_json(
        output_path,
        output,
        run_identity=state.get("run_identity"),
    )
    write_state_snapshot(
        merge_update(state, {"output": str(output_path), "status": output_status}),
        "write_output",
        "__end__",
    )
    print(f"Saved result to {output_path}")
    print(f"Status: {output['status']}")
    return {"output": str(output_path), "status": output["status"]}


def build_human_gate_graph():
    graph = StateGraph(SectionDesignGraphState)
    graph.add_node("resume_router", resume_router_node)
    graph.add_node("prepare", prepare_state)
    graph.add_node("locating", locating_node)
    graph.add_node("mechanism", mechanism_node)
    graph.add_node("query_semantics", query_semantics_node)
    graph.add_node("evidence_model", evidence_model_node)
    graph.add_node("subjective_supervisor", subjective_supervisor_node)
    graph.add_node("topic_adaptation", topic_adaptation_node)
    graph.add_node("section_partition", section_partition_node)
    graph.add_node("field_planning", field_planning_node)
    graph.add_node("human_advice_gate", human_advice_gate_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("figure_classification", figure_classification_node)
    graph.add_node("schema_design", schema_design_node)
    graph.add_node("schema_design_repair", schema_design_repair_node)
    graph.add_node("supervisor_router", supervisor_router_node)
    graph.add_node("specialization_critic", specialization_critic_node)
    graph.add_node("aggregation", aggregation_node)
    graph.add_node("write_output", write_output_node)

    graph.add_edge(START, "resume_router")
    graph.add_conditional_edges("resume_router", interactive_resume_node)
    graph.add_edge("prepare", "locating")
    graph.add_conditional_edges("supervisor_router", supervisor_next_node)
    graph.add_conditional_edges("locating", lambda s: stop_if_errors(s, "mechanism"))
    graph.add_conditional_edges("mechanism", lambda s: stop_if_errors(s, "query_semantics"))
    graph.add_conditional_edges("query_semantics", lambda s: stop_if_errors(s, "evidence_model"))
    graph.add_conditional_edges("evidence_model", lambda s: stop_if_errors(s, "subjective_supervisor"))
    graph.add_conditional_edges("subjective_supervisor", lambda s: stop_if_errors(s, "topic_adaptation"))
    graph.add_conditional_edges("topic_adaptation", lambda s: stop_if_errors(s, "section_partition"))
    graph.add_conditional_edges("section_partition", lambda s: stop_if_errors(s, "field_planning"))
    graph.add_conditional_edges("field_planning", lambda s: stop_if_errors(s, "supervisor"))
    graph.add_conditional_edges("human_advice_gate", lambda s: stop_if_errors(s, "write_output"))
    graph.add_conditional_edges("supervisor", lambda s: stop_if_errors(s, "figure_classification"))
    graph.add_conditional_edges("figure_classification", lambda s: stop_if_errors(s, "schema_design"))
    graph.add_conditional_edges("schema_design", lambda s: stop_if_errors(s, "specialization_critic"))
    graph.add_conditional_edges("schema_design_repair", lambda s: stop_if_errors(s, "specialization_critic"))
    graph.add_conditional_edges("specialization_critic", lambda s: stop_if_errors(s, "aggregation"))
    graph.add_conditional_edges("aggregation", lambda s: stop_if_errors(s, "human_advice_gate"))
    graph.add_edge("write_output", END)
    return graph.compile(checkpointer=MemorySaver())


def build_parser():
    parser = section_agent.build_parser()
    for action in parser._actions:
        if action.dest in {
            "database_goal",
            "discipline",
            "query_requirements",
            "key_description_path",
        }:
            action.required = False
    parser.description = (
        "Run Step 8 Section Design through LangGraph. The system first repairs "
        "field design until internal critic/supervisor acceptance, then applies "
        "optional human expert review."
    )
    parser.add_argument(
        "--thread-id",
        default="section-design-human-gate",
        help="LangGraph checkpoint thread id for this run.",
    )
    parser.add_argument(
        "--interactive-human-gate",
        action="store_true",
        help="Ask for human advice on stdin when the graph interrupts.",
    )
    parser.add_argument(
        "--resume-from-state",
        default="",
        help="Resume from a previously written *.state.json snapshot.",
    )
    parser.add_argument(
        "--max-supervisor-retries",
        type=int,
        default=1,
        help="Maximum supervisor-directed retries per distinct blocker fingerprint.",
    )
    parser.add_argument(
        "--max-total-supervisor-repairs",
        type=int,
        default=12,
        help="Hard safety budget across all distinct supervisor repair cycles.",
    )
    parser.add_argument(
        "--schema-inspection-interval",
        type=int,
        default=10,
        help=(
            "Pause for convergence and critic-strictness inspection at each interval "
            "of unresolved schema revisions."
        ),
    )
    parser.add_argument(
        "--repeated-blocker-inspection-threshold",
        type=int,
        default=3,
        help=(
            "Pause when an identical blocker repeats this many times without a schema "
            "revision, so protocol defects or over-strict rules can be inspected early."
        ),
    )
    parser.add_argument(
        "--schema-inspection-approved-through",
        type=int,
        default=0,
        help="Resume-only approval boundary after the preserved history was inspected.",
    )
    parser.add_argument(
        "--skip-human-expert-review",
        action="store_true",
        help="Explicitly skip the post-internal-pass human expert review gate.",
    )
    parser.add_argument(
        "--structured-protocol",
        action="store_true",
        help=(
            "Apply structured expert schema concepts before design and expose protocol mode "
            "in the supervisor-visible run contract."
        ),
    )
    return parser


def explicit_cli_destinations(parser, argv):
    """Return only argparse destinations explicitly present on the command line."""
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    destinations = set()
    for token in argv:
        option = str(token).split("=", 1)[0]
        destination = option_to_dest.get(option)
        if destination:
            destinations.add(destination)
    return destinations


def inject_resume_human_advice(initial_state, human_advice):
    advice = str(human_advice or "").strip()
    shared = deepcopy(initial_state.get("shared_context", {}))
    shared["human_advice"] = advice
    post_design_review = bool(
        initial_state.get("next_node") == "human_advice_gate"
        or initial_state.get("current_node") == "human_advice_gate"
        or initial_state.get("status") == "waiting_for_human_advice"
    )
    if not post_design_review:
        shared["human_advice_available_before_design"] = bool(advice)
    initial_state["shared_context"] = shared
    initial_state["human_advice"] = advice
    if post_design_review:
        initial_state["post_design_human_advice"] = advice
    return initial_state


def _legacy_main_without_clean_run_guard():
    parser = build_parser()
    args = parser.parse_args()
    explicit_overrides = explicit_cli_destinations(parser, sys.argv[1:])
    if not args.resume_from_state:
        missing = [
            name
            for name in [
                "database_goal",
                "discipline",
                "query_requirements",
                "key_description_path",
            ]
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(
                "Missing required arguments for a new run: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )

    app = build_human_gate_graph()
    config = {"configurable": {"thread_id": args.thread_id}}

    args_dict = namespace_to_dict(args)
    thread_id = args_dict.pop("thread_id")
    interactive = args_dict.pop("interactive_human_gate")
    resume_from_state = args_dict.pop("resume_from_state")

    if resume_from_state:
        loaded_snapshot = load_state_snapshot(resume_from_state)
        if loaded_snapshot.get("step") == "step8_section_design_agent_checkpoint":
            initial_state = migrate_legacy_modular_checkpoint(loaded_snapshot, args_dict)
            snapshot_args = initial_state["args"]
        else:
            initial_state = loaded_snapshot
            snapshot_args = deepcopy(initial_state.get("args", {}))
            for key in (
                "api_key",
                "base_url",
                "model",
                "llm_backend",
                "request_timeout",
                "temperature",
                "max_retries",
                "max_supervisor_retries",
                "skip_human_expert_review",
                "checkpoint_output",
                "stop_after_module",
            ):
                if key in explicit_overrides:
                    snapshot_args[key] = getattr(args, key, None)
            initial_state["args"] = snapshot_args
        if args.human_advice or args.human_advice_path:
            human_advice = args.human_advice or section_agent.load_optional_text(args.human_advice_path)
            snapshot_args["human_advice"] = human_advice
            snapshot_args["human_advice_path"] = ""
            initial_state["args"] = snapshot_args
            initial_state = inject_resume_human_advice(initial_state, human_advice)
            if initial_state.get("next_node") == "human_advice_gate":
                initial_state["next_node"] = "human_advice_gate"
    else:
        initial_state = {"args": args_dict, "next_node": "prepare"}

    result = app.invoke(initial_state, config=config)
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if interrupts and not interactive:
        print("Paused before supervisor_module. Human advice is required.")
        print_console_safe(interrupts)
        return

    if interrupts and interactive:
        print("Paused before supervisor_module. Enter human advice, then press Enter:")
        advice = input("> ").strip()
        result = app.invoke(Command(resume=advice), config=config)

    if isinstance(result, dict):
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "output": result.get("output"),
                    "current_node": result.get("current_node"),
                    "next_node": result.get("next_node"),
                    "reference_papers": initial_state.get("args", {}).get("reference_papers", []),
                },
                ensure_ascii=False,
            )
        )
    else:
        print_console_safe(result)
    if thread_id:
        print(f"Thread id: {thread_id}")


def build_graph_step8_input_identity(args_dict):
    immutable_args = deepcopy(args_dict)
    # Human advice is intentionally mutable on explicit resume. The scientific
    # task, corpus, provider/model, and protocol mode remain immutable.
    immutable_args["human_advice"] = ""
    immutable_args["human_advice_path"] = ""
    return section_agent.build_step8_input_identity(dict_to_namespace(immutable_args))


def main():
    parser = build_parser()
    args = parser.parse_args()
    explicit_overrides = explicit_cli_destinations(parser, sys.argv[1:])
    if not args.resume_from_state:
        missing = [
            name
            for name in (
                "database_goal",
                "discipline",
                "query_requirements",
                "key_description_path",
            )
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(
                "Missing required arguments for a new run: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )

    args_dict = namespace_to_dict(args)
    requested_thread_id = args_dict.pop("thread_id")
    interactive = args_dict.pop("interactive_human_gate")
    resume_from_state = args_dict.pop("resume_from_state")

    if resume_from_state:
        initial_state = load_state_snapshot(resume_from_state)
        if initial_state.get("step") == "step8_section_design_agent_checkpoint":
            raise run_artifact_guard.RunIdentityError(
                "Legacy Step8 checkpoints cannot be resumed in clean-run mode. "
                "Start a fresh run with new artifact paths."
            )
        snapshot_args = deepcopy(initial_state.get("args", {}))
        for key in (
            "api_key",
            "base_url",
            "model",
            "llm_backend",
            "request_timeout",
            "temperature",
            "max_retries",
            "max_supervisor_retries",
            "max_total_supervisor_repairs",
            "schema_inspection_interval",
            "repeated_blocker_inspection_threshold",
            "skip_human_expert_review",
            "stop_after_module",
        ):
            if key in explicit_overrides:
                snapshot_args[key] = getattr(args, key, None)
        input_identity = build_graph_step8_input_identity(snapshot_args)
        run_identity = run_artifact_guard.validate_resume_identity(
            initial_state,
            pipeline="step8_section_design_langgraph",
            input_identity=input_identity,
            output_path=snapshot_args.get("output", ""),
        )
        initial_state["args"] = snapshot_args
        initial_state["run_identity"] = run_identity
        if args.schema_inspection_approved_through:
            initial_state["schema_inspection_approved_through"] = int(
                args.schema_inspection_approved_through
            )
        if args.human_advice or args.human_advice_path:
            human_advice = args.human_advice or section_agent.load_optional_text(
                args.human_advice_path
            )
            snapshot_args["human_advice"] = human_advice
            snapshot_args["human_advice_path"] = ""
            initial_state["args"] = snapshot_args
            initial_state = inject_resume_human_advice(initial_state, human_advice)
        run_artifact_guard.update_run_status(
            run_identity,
            "running",
            resumed_from=str(Path(resume_from_state).resolve(strict=False)),
        )
    else:
        input_identity = build_graph_step8_input_identity(args_dict)
        output_path = Path(args_dict["output"])
        checkpoint_path = state_path_from_args(dict_to_namespace(args_dict))
        human_context_path = human_gate_context_path_from_args(
            dict_to_namespace(args_dict)
        )
        run_identity = run_artifact_guard.reserve_fresh_run(
            pipeline="step8_section_design_langgraph",
            output_path=output_path,
            input_identity=input_identity,
            artifact_paths=[output_path, checkpoint_path, human_context_path],
        )
        initial_state = {
            "args": args_dict,
            "next_node": "prepare",
            "run_identity": run_identity,
        }

    app = build_human_gate_graph()
    config = {
        "configurable": {
            "thread_id": f"{requested_thread_id}:{run_identity['run_id']}"
        }
    }
    try:
        result = app.invoke(initial_state, config=config)
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if interrupts and not interactive:
            run_artifact_guard.update_run_status(
                run_identity,
                "interrupted",
                pause_reason="waiting_for_human_advice",
            )
            print("Paused before supervisor_module. Human advice is required.")
            print_console_safe(interrupts)
            return
        if interrupts and interactive:
            print("Paused before supervisor_module. Enter human advice, then press Enter:")
            advice = input("> ").strip()
            result = app.invoke(Command(resume=advice), config=config)
    except BaseException as exc:
        run_artifact_guard.update_run_status(
            run_identity,
            "interrupted",
            error_type=type(exc).__name__,
        )
        raise

    result_status = result.get("status") if isinstance(result, dict) else "unknown"
    run_artifact_guard.update_run_status(
        run_identity,
        "completed" if result_status in {"success", "partial_success"} else "needs_review",
        result_status=result_status,
    )
    if isinstance(result, dict):
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "output": result.get("output"),
                    "current_node": result.get("current_node"),
                    "next_node": result.get("next_node"),
                    "run_id": run_identity["run_id"],
                    "reference_papers": initial_state.get("args", {}).get(
                        "reference_papers", []
                    ),
                },
                ensure_ascii=False,
            )
        )
    else:
        print_console_safe(result)
    if requested_thread_id:
        print(f"Thread id: {requested_thread_id}:{run_identity['run_id']}")


if __name__ == "__main__":
    main()
