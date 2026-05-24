import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import section_design_agent as section_agent
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


class SectionDesignGraphState(TypedDict, total=False):
    args: dict[str, Any]
    shared_context: dict[str, Any]
    module_outputs: dict[str, Any]
    module_attempts: dict[str, Any]
    module_errors: dict[str, Any]
    validation_errors: list[str]
    human_advice: str
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


def state_path_from_args(args):
    output_path = Path(args.output)
    return output_path.with_suffix(output_path.suffix + ".state.json")


def human_gate_context_path_from_args(args):
    output_path = Path(args.output)
    return output_path.with_suffix(output_path.suffix + ".human_gate_context.json")


def json_safe_state(state):
    def clean(value, key_name=""):
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
            limit = 5000 if key_name in {"prompt", "raw_response"} else 50000
            if len(value) > limit:
                return value[:limit] + "\n...[truncated in state snapshot]..."
            return value
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)

    return clean(state)


def write_state_snapshot(state, current_node, next_node=None):
    args = dict_to_namespace(state["args"])
    snapshot = deepcopy(state)
    snapshot["current_node"] = current_node
    if next_node:
        snapshot["next_node"] = next_node
    snapshot_path = state_path_from_args(args)
    tmp_path = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(json_safe_state(snapshot), ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(snapshot_path)
    return snapshot


def load_state_snapshot(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
            "section_partition_module": outputs.get("section_partition_module"),
            "field_planning_module": outputs.get("field_planning_module"),
        },
        "required_advice": (
            "Review the internally accepted field design as a materials-domain expert. "
            "If it is not acceptable, describe only domain/schema problems in the current output; "
            "do not provide code or prompt implementation instructions. If acceptable, say it can pass."
        ),
    }
    context_path = human_gate_context_path_from_args(args)
    context_path.write_text(
        json.dumps(json_safe_state(context), indent=2, ensure_ascii=False),
        encoding="utf-8",
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


def resume_router_node(state):
    return state


def prepare_state(state):
    if state.get("shared_context"):
        return write_state_snapshot(state, "prepare", "locating")

    args = dict_to_namespace(state["args"])
    query_requirements = section_agent.load_query_requirements(args.query_requirements)
    key_description_text = section_agent.load_key_description_text(args.key_description_path)
    reference_paper_context = section_agent.load_reference_paper_context(args.reference_papers)
    human_advice = args.human_advice or section_agent.load_optional_text(args.human_advice_path)
    shared_context = {
        "database_goal": args.database_goal,
        "discipline": args.discipline,
        "query_requirements": query_requirements,
        "key_description_text": key_description_text,
        "reference_paper_context": reference_paper_context,
        "human_advice": human_advice,
        "shared_context_block": build_shared_context_block(
            args.database_goal,
            args.discipline,
            query_requirements,
            key_description_text,
            reference_paper_context,
        ),
    }
    update = {
        "shared_context": shared_context,
        "module_outputs": {},
        "module_attempts": {},
        "module_errors": {},
        "validation_errors": [],
        "human_advice": human_advice,
        "status": "running",
        "retry_counts": {},
    }
    next_state = merge_update(state, update)
    return write_state_snapshot(next_state, "prepare", "locating")


def call_module_node(state, module_name, prompt, current_node, next_node):
    args = dict_to_namespace(state["args"])
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
    outputs = deepcopy(state.get("module_outputs", {}))
    module_attempts = deepcopy(state.get("module_attempts", {}))
    module_errors = deepcopy(state.get("module_errors", {}))
    outputs[module_name] = result
    module_attempts[module_name] = attempts
    module_errors[module_name] = errors
    update = {
        "module_outputs": outputs,
        "module_attempts": module_attempts,
        "module_errors": module_errors,
    }
    if errors:
        error_type = classify_errors(errors)
        update["validation_errors"] = errors
        update["status"] = "awaiting_supervisor_decision"
        update["last_error_node"] = current_node
        update["last_error_type"] = error_type
        next_node = "supervisor_router"
    return write_state_snapshot(merge_update(state, update), current_node, next_node)


def stop_if_errors(state, next_node):
    if state.get("validation_errors"):
        return "supervisor_router"
    return next_node


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
    retry_count = int(retry_counts.get(error_node, 0))
    args = dict_to_namespace(state["args"])
    max_supervisor_retries = int(getattr(args, "max_supervisor_retries", 1) or 1)

    if (
        error_node == "schema_design"
        and error_type in {"json_parse_failure", "schema_validation_failure", "module_failure"}
        and retry_count < max_supervisor_retries
    ):
        retry_counts[error_node] = retry_count + 1
        decision = {
            "action": "repair_schema_design",
            "reason": "schema_design failed or critic requested redesign; retry with compact repair prompt.",
            "error_node": error_node,
            "error_type": error_type,
            "retry_count": retry_counts[error_node],
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
        retry_counts[error_node] = retry_count + 1
        decision = {
            "action": "retry_node",
            "reason": "module failed once; supervisor allows one generic retry.",
            "error_node": error_node,
            "error_type": error_type,
            "retry_count": retry_counts[error_node],
            "next_node": error_node,
        }
    else:
        decision = {
            "action": "finish_needs_review",
            "reason": "supervisor retry budget exhausted or no safe route available.",
            "error_node": error_node,
            "error_type": error_type,
            "retry_count": retry_count,
            "next_node": "write_output",
        }

    update = {
        "supervisor_decision": decision,
        "retry_counts": retry_counts,
        "status": "running" if decision["next_node"] != "write_output" else "needs_review",
    }
    if decision["next_node"] != "write_output":
        update["repair_instructions"] = errors
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
    human_advice = (state.get("human_advice") or "").strip()
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
    previously_applied_advice = str(state.get("applied_human_advice") or "").strip()
    current_advice_already_applied = bool(
        state.get("human_review_applied")
        and current_advice
        and current_advice == previously_applied_advice
    )
    if current_advice_already_applied or advice_is_acceptance(current_advice):
        outputs["human_advice_gate"] = {
            "status": "accepted",
            "position": "after_internal_field_design_pass",
            "human_advice": current_advice,
            "reason": "Human expert review accepted the internally passing field design, or the same advice was already applied.",
        }
        return write_state_snapshot(
            merge_update(
                state,
                {
                    "shared_context": shared,
                    "module_outputs": outputs,
                    "status": "success",
                    "validation_errors": [],
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
    return write_state_snapshot(update, "schema_design", update.get("next_node", "specialization_critic"))


def build_schema_repair_prompt(state):
    outputs = state["module_outputs"]
    attempts = state.get("module_attempts", {}).get("schema_design_module", [])
    previous_raw = ""
    if attempts:
        previous_raw = str(attempts[-1].get("raw_response") or "")[-12000:]
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
- Prefer 80-140 high-value field_registry entries when validation asks for child-level coverage.
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


def section_for_repair_parent(parent, source_text):
    parent_text = parent.lower()
    if any(
        token in parent_text
        for token in (
            "transition",
            "hysteresis",
            "moment",
            "anisotropy",
            "magnetocaloric",
            "soft_magnet",
            "permanent_magnet",
            "performance",
        )
    ):
        return "material_info.section1"
    text = f"{parent} {source_text}".lower()
    if "section1" in text:
        return "material_info.section1"
    if "section2" in text or "synthesis" in text or "anneal" in text or "process" in text:
        return "material_info.section2"
    if "section3" in text or "characterization" in text:
        return "material_info.section3"
    if "section4" in text or "curve" in text or parent.lower() in {"m_t", "m_h", "chi_t", "specific_heat", "r_t"}:
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

PREFERRED_FIELD_REGISTRY_SIZE = 160
HARD_FIELD_REGISTRY_SIZE = 220


def field_path_depth(field_path):
    return len([part for part in str(field_path or "").split(".") if part])


def compact_field_registry(registry, preferred_size=PREFERRED_FIELD_REGISTRY_SIZE, hard_size=HARD_FIELD_REGISTRY_SIZE):
    """Keep Step8 schemas at database-design granularity, not leaf-field explosion."""
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

    if len(deduped) <= hard_size:
        return deduped

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
    return roots


def schema_has_unapproved_roots(schema, state=None):
    approved_roots = approved_schema_roots_from_state(state or {})
    return sorted(root for root in collect_schema_roots(schema) if root and root not in approved_roots)


def infer_section_from_field_path(field_path, fallback_section_id="material_info.section1"):
    field_path = str(field_path)
    for prefix, section_id in SECTION_HINTS.items():
        if field_path == prefix or field_path.startswith(f"{prefix}."):
            return section_id
    if field_path.startswith(("formula_aliases", "material_name_aliases", "sample_id_aliases")):
        return "material_info.section0"
    if any(token in field_path.lower() for token in ["fabrication", "synthesis", "anneal", "growth", "processing"]):
        return "material_info.section2"
    if any(token in field_path.lower() for token in ["xrd", "afm", "sem", "tem", "stm", "sts", "arpes", "xas", "xmcd", "microscopy", "characterization"]):
        return "material_info.section3"
    if any(token in field_path.lower() for token in ["curve", "r_t", "m_t", "m_h", "chi_t", "hall", "mr", "iv", "specific_heat"]):
        return "material_info.section4"
    if any(token in field_path.lower() for token in ["theory", "mechanism", "calculated", "simulation", "model"]):
        return "section5"
    return fallback_section_id


def canonicalize_rebuild_field_path(field_path, fallback_section_id):
    field_path = re.sub(r"\s*\([^)]*\)", "", str(field_path).strip().lstrip("."))
    field_path = field_path.replace("[]", "")
    field_path = field_path.replace(
        "material_info.section1.core_parameter.",
        "material_info.section1.core_parameters.",
    )
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
    if root in allowed_roots:
        return field_path
    if str(fallback_section_id).startswith("material_info."):
        return f"{fallback_section_id}.{field_path}"
    if str(fallback_section_id) == "section5":
        return f"section5.{field_path}"
    if str(fallback_section_id) == "paper_info":
        return f"paper_info.{field_path}"
    return field_path


def infer_data_type_from_field_path(field_path):
    leaf = str(field_path).rsplit(".", 1)[-1].lower()
    if leaf in {"value", "raw_value", "uncertainty", "confidence_score", "year"}:
        return "number"
    if leaf in {"required", "available", "raw_data"} or leaf.endswith("_flag"):
        return "boolean"
    if leaf.endswith("s") or leaf in {"authors", "keywords", "aliases", "evidence_links", "supporting_methods"}:
        return "array"
    if leaf in {"conditions", "metadata", "resources", "figure_constraint", "normalization_aliases"}:
        return "object"
    return "string"


def infer_source_basis_from_field_path(field_path, evidence_strategy=""):
    text = str(field_path).lower()
    basis = ["text"]
    if "source_table" in text or text.endswith(".table"):
        basis.append("table")
    if any(
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
        category = "property_curve"
    elif section_id == "section5":
        category = "domain_relevant_figure"
    elif any(token in lowered for token in ["curve", "r_t", "m_t", "m_h", "chi_t", "hall", "mr", "iv"]):
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


def rebuild_material_schema_from_field_planning(state, reason):
    outputs = deepcopy(state.get("module_outputs", {}))
    field_groups = (outputs.get("field_planning_module") or {}).get("field_groups") or []
    registry = []
    existing_paths = set()

    for group in field_groups:
        if not isinstance(group, dict):
            continue
        fallback_section_id = str(group.get("section_id") or "material_info.section1")
        evidence_strategy = str(group.get("evidence_strategy") or "")
        for field_path in group.get("recommended_fields") or []:
            if not isinstance(field_path, str) or not field_path.strip():
                continue
            field_path = canonicalize_rebuild_field_path(field_path, fallback_section_id)
            root = field_path.split(".", 1)[0]
            if root not in DEFAULT_SCHEMA_ROOTS and root not in SUPERVISOR_EXTENSION_ROOTS:
                continue
            if field_path in existing_paths:
                continue
            section_id = infer_section_from_field_path(field_path, fallback_section_id)
            source_basis = infer_source_basis_from_field_path(field_path, evidence_strategy)
            registry.append(
                {
                    "field_path": field_path,
                    "section_id": section_id,
                    "field_name": field_path.rsplit(".", 1)[-1],
                    "data_type": infer_data_type_from_field_path(field_path),
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
            )
            existing_paths.add(field_path)

    minimum_fields = [
        ("paper_info.metadata.title", "paper_info"),
        ("paper_info.metadata.doi", "paper_info"),
        ("paper_info.metadata.abstract", "paper_info"),
        ("paper_info.metadata.url", "paper_info"),
        ("paper_info.resources.raw_data", "paper_info"),
        ("primary_signature", "material_info.section0"),
        ("primary_signature_normalized", "material_info.section0"),
        ("material_info.section0.primary_material_identity.formula", "material_info.section0"),
        ("material_info.section0.primary_material_identity.material_name", "material_info.section0"),
        ("material_info.section1.core_parameter.parameter_name", "material_info.section1"),
        ("material_info.section1.core_parameter.value", "material_info.section1"),
        ("material_info.section1.core_parameter.unit", "material_info.section1"),
        ("material_info.section1.core_parameter.conditions", "material_info.section1"),
        ("material_info.section1.core_parameter.source_reference", "material_info.section1"),
        ("material_info.section2.fabrication_entries.method", "material_info.section2"),
        ("material_info.section2.fabrication_entries.conditions", "material_info.section2"),
        ("material_info.section3.characterization_evidence.technique", "material_info.section3"),
        ("material_info.section3.characterization_evidence.figure", "material_info.section3"),
        ("material_info.section4.property_curves.curve_type", "material_info.section4"),
        ("material_info.section4.property_curves.figure", "material_info.section4"),
        ("section5.theory_mechanism_core.mechanism_type", "section5"),
        ("section5.theory_mechanism_core.evidence_links", "section5"),
    ]
    skip_if_prefix_exists = {
        "material_info.section1.core_parameter.": "material_info.section1.core_parameters",
        "material_info.section3.characterization_evidence.": "material_info.section3.",
        "material_info.section4.property_curves.": "material_info.section4.",
    }
    for field_path, section_id in minimum_fields:
        if field_path in existing_paths:
            continue
        if any(field_path.startswith(prefix) and any(path.startswith(existing_prefix) for path in existing_paths) for prefix, existing_prefix in skip_if_prefix_exists.items()):
            continue
        source_basis = infer_source_basis_from_field_path(field_path)
        registry.append(
            {
                "field_path": field_path,
                "section_id": section_id,
                "field_name": field_path.rsplit(".", 1)[-1],
                "data_type": infer_data_type_from_field_path(field_path),
                "required": field_path in {"paper_info.metadata.title", "paper_info.metadata.doi", "primary_signature"},
                "source_basis": source_basis,
                "figure_constraint": infer_rebuild_figure_constraint(field_path, section_id, source_basis),
                "reason": f"Safety baseline added during materials schema rebuild because {reason}",
            }
        )
        existing_paths.add(field_path)

    schema = {
        "top_level_keys": deepcopy(MATERIAL_TOP_LEVEL_KEYS),
        "field_registry": registry,
    }
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
        "conditions.magnetic_field.value",
        "conditions.magnetic_field.direction",
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
        "hysteresis": [
            "coercivity.value",
            "coercivity.unit",
            "coercivity.temperature",
            "coercivity.field_direction",
            "coercivity.source_figure",
            "remanence.value",
            "remanence.unit",
            "remanence.temperature",
            "remanence.field_direction",
            "remanence.source_figure",
            "saturation_magnetization.value",
            "saturation_magnetization.unit",
            "saturation_magnetization.temperature",
            "saturation_magnetization.field_direction",
            "saturation_magnetization.source_figure",
            "field_direction",
            "protocol",
        ],
        "moment": ["effective_moment", "saturation_moment", "basis", "method"],
        "anisotropy": ["constant", "easy_axis", "source_method", "temperature"],
        "magnetocaloric": [
            "entropy_change.value",
            "entropy_change.unit",
            "refrigerant_capacity.value",
            "refrigerant_capacity.unit",
            "adiabatic_temperature_change.value",
            "adiabatic_temperature_change.unit",
            "field_change",
            "peak_temperature",
            "calculation_method",
        ],
        "soft_magnet": [
            "permeability.value",
            "permeability.unit",
            "core_loss.value",
            "core_loss.unit",
            "electrical_resistivity.value",
            "electrical_resistivity.unit",
            "frequency",
            "field_amplitude",
        ],
        "permanent_magnet": [
            "maximum_energy_product.value",
            "maximum_energy_product.unit",
            "recoil_permeability.value",
            "recoil_permeability.unit",
            "knee_field.value",
            "knee_field.unit",
            "temperature_coefficient.value",
            "temperature_coefficient.unit",
        ],
        "synthesis": ["method", "description", "temperature", "time", "pressure", "atmosphere"],
        "curve": ["figure", "variable", "x_axis", "y_axis", "temperature", "magnetic_field", "protocol"],
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
            "core_loss",
            "maximum_energy_product",
            "adiabatic_temperature_change",
            "field_change",
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


def deterministic_schema_repair(state):
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
        return rebuild_material_schema_from_field_planning(state, reason)

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
            "type": "field_granularity_budget",
            "original_field_count": original_count,
            "final_field_count": len(registry),
            "policy": (
                "Keep database fields at medium granularity similar to the superconductivity pipeline. "
                "Use parent object fields plus selected essential subfields instead of exhaustive leaf expansion."
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


def schema_design_repair_node(state):
    if state.get("last_error_type") == "schema_validation_failure" and state.get("module_outputs", {}).get("schema_design_module"):
        update = deterministic_schema_repair(state)
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
        return write_state_snapshot(update, "schema_design_repair", "specialization_critic")

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
    return write_state_snapshot(update, "schema_design_repair", update.get("next_node", "specialization_critic"))


def local_schema_coverage_sufficient(schema_result, critic_result):
    unapproved_roots = schema_has_unapproved_roots(schema_result or {}, {})
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
    if "measurement_conditions" in critic_text and not any(
        ".measurement_conditions" in path or ".conditions" in path for path in field_paths
    ):
        missing.append("no measurement condition fields")
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
    if human_advice_text and registry:
        critic_result = {
            "specialization_status": "local_structural_review_after_human_advice",
            "redo_needed": False,
            "is_generic": False,
            "missing_concepts": [human_advice_text],
            "redo_directives": [],
            "structural_weaknesses": [],
        }
        coverage_ok, coverage_missing = local_schema_coverage_sufficient(schema_result, critic_result)
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
        ),
        "specialization_critic",
        "aggregation",
    )
    if update.get("validation_errors"):
        return update

    critic_result = update["module_outputs"].get("specialization_critic_module") or {}
    if critic_result.get("redo_needed") or critic_result.get("is_generic"):
        coverage_ok, coverage_missing = local_schema_coverage_sufficient(
            update["module_outputs"].get("schema_design_module") or {},
            critic_result,
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

    return update


def aggregation_node(state):
    outputs = state["module_outputs"]
    args = dict_to_namespace(state["args"])

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
    update["next_node"] = "human_advice_gate"
    return write_state_snapshot(update, "aggregation", "human_advice_gate")


def write_output_node(state):
    args = dict_to_namespace(state["args"])
    shared = state.get("shared_context", {})
    query_requirements = shared.get("query_requirements", [])
    module_outputs = state.get("module_outputs", {})
    output_inputs = {
        "database_goal": args.database_goal,
        "discipline": args.discipline,
        "query_requirements": query_requirements,
        "key_description_path": str(Path(args.key_description_path)),
        "reference_papers": args.reference_papers,
        "human_expert_review_after_internal_pass": bool(
            getattr(args, "require_human_advice_before_supervisor", False) or state.get("human_advice")
        ),
        "human_advice_provided": bool(state.get("human_advice")),
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
    output = {
        "step": "step8_section_design_agent_langgraph_human_gate",
        "framework": STEP8_FRAMEWORK,
        "inputs": output_inputs,
        "reference_paper_context_used": shared.get("reference_paper_context", ""),
        "agent_flow": section_agent.build_agent_flow_trace(
            output_inputs,
            module_outputs,
            result,
            errors,
        ),
        "module_outputs": module_outputs,
        "module_errors": state.get("module_errors", {}),
        "result": result,
        "validation_errors": errors,
        "status": state.get("status", "needs_review"),
        "attempts": {
            "modules": state.get("module_attempts", {}),
            "final_redo": [],
        },
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_state_snapshot(merge_update(state, {"output": str(output_path)}), "write_output", "__end__")
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
        help="Maximum supervisor-directed retries per failed graph node.",
    )
    parser.add_argument(
        "--skip-human-expert-review",
        action="store_true",
        help="Explicitly skip the post-internal-pass human expert review gate.",
    )
    return parser


def main():
    args = build_parser().parse_args()
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
        initial_state = load_state_snapshot(resume_from_state)
        if args.human_advice or args.human_advice_path:
            human_advice = args.human_advice or section_agent.load_optional_text(args.human_advice_path)
            snapshot_args = deepcopy(initial_state.get("args", {}))
            snapshot_args["human_advice"] = human_advice
            snapshot_args["human_advice_path"] = ""
            shared = deepcopy(initial_state.get("shared_context", {}))
            shared["human_advice"] = human_advice
            initial_state["args"] = snapshot_args
            initial_state["shared_context"] = shared
            initial_state["human_advice"] = human_advice
            if initial_state.get("next_node") == "human_advice_gate":
                initial_state["next_node"] = "human_advice_gate"
    else:
        initial_state = {"args": args_dict, "next_node": "prepare"}

    result = app.invoke(initial_state, config=config)
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if interrupts and not interactive:
        print("Paused before supervisor_module. Human advice is required.")
        print(interrupts)
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
        print(result)
    if thread_id:
        print(f"Thread id: {thread_id}")


if __name__ == "__main__":
    main()
