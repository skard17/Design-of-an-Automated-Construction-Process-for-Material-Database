import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import prompt_quality_code_agent as prompt_agent
import downstream_extraction_runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_PIPELINE = str(PROJECT_ROOT / "reference_pipelines" / "pure_extraction_pipeline")


class Step9State(TypedDict, total=False):
    args: dict[str, Any]
    step8_output: dict[str, Any]
    prompt_output: dict[str, Any]
    workflow_plan: dict[str, Any]
    judgement: dict[str, Any]
    quality_review: dict[str, Any]
    code_agent: dict[str, Any]
    extraction_test: dict[str, Any]
    extraction_eval: dict[str, Any]
    supervisor_decision: dict[str, Any]
    validation_errors: list[str]
    human_advice: list[dict[str, Any]]
    retry_counts: dict[str, int]
    repair_history: list[dict[str, Any]]
    failed_node: str
    schema_feedback: dict[str, Any]
    status: str
    current_node: str
    next_node: str
    output: str


SECTION_ORDER = [
    "paper_info",
    "material_info.section0",
    "material_info.section2",
    "material_info.section3",
    "figure_classification",
    "material_info.section4",
    "material_info.section1",
    "section5",
]

THEORY_SECTION_ALIASES = {
    "section5",
    "theory_mechanism",
    "theory_and_mechanism",
    "mechanism_theory",
    "material_info.section5",
    "material_info.theory_mechanism",
}


REFERENCE_STAGE_HINTS = [
    {
        "name": "classification_gate",
        "paths": ["IE_0_classification", "IE_0_single"],
        "use_when": "The corpus may include review, non-experimental, non-target, or multi-material papers.",
        "strategy": "Gate papers before expensive section extraction; route single-material and multi-material papers differently.",
    },
    {
        "name": "section5_and_resources_early",
        "paths": ["IE_1_part0(s5+resource)"],
        "use_when": "Theory/mechanism and paper-level resources are useful global context for later extraction.",
        "strategy": "Extract section5 and paper resources before section-specific value extraction.",
    },
    {
        "name": "figure_classification_before_s4_s5",
        "paths": ["IE_1_part2(fig_classify)"],
        "use_when": "Figure-heavy sections may confuse property curves with theory, mechanism, or characterization evidence.",
        "strategy": "Classify figures/tables/panels first; route section4 curves and section5/theory evidence using the classification result.",
    },
    {
        "name": "section2_two_pass",
        "paths": ["IE_1_part1(s0+s1+s2_1)", "IE_2_part1(s2_2)"],
        "use_when": "Fabrication, processing, treatment, or measurement-preparation fields combine method identity, process sequence, and detailed conditions.",
        "strategy": "First extract coarse section2 method/process information; then run a second fine pass for method-specific conditions and mapping.",
    },
    {
        "name": "section3_section4_pair",
        "paths": ["IE_2_part2(s3+s4)"],
        "use_when": "Characterization evidence and macroscopic curves need coordinated figure ownership.",
        "strategy": "Extract section3 and section4 as separate sections while sharing figure classification context.",
    },
    {
        "name": "json_check_and_schema_cleanup",
        "paths": ["IE_3_check", "IE_4_preprocess(others)", "IE_4_preprocess1(s2)", "IE_4_preprocess2(s2)"],
        "use_when": "Raw extraction JSON contains invalid structure, section drift, or section2 method naming inconsistency.",
        "strategy": "Run JSON repair, schema cleanup, section2 merge, and method mapping before final normalization.",
    },
    {
        "name": "unicode_semantic_normalization",
        "paths": ["IE_5_unicode"],
        "use_when": "Extracted strings contain Unicode variants, mojibake, inconsistent symbols, or semantically equivalent wording.",
        "strategy": "Normalize strings with schema-aware prompts after section JSON is structurally valid.",
    },
    {
        "name": "final_merge",
        "paths": ["IE_6_merge"],
        "use_when": "All section outputs are available and normalized.",
        "strategy": "Merge section outputs into one final JSON with primary_signature, material_info, section5, and paper_info.",
    },
    {
        "name": "multi_material_manifest_fact_match",
        "paths": ["BM_multi_match_pipeline"],
        "use_when": "A paper contains multiple candidate material targets or the single-material gate is uncertain.",
        "strategy": "Use manifest -> fact_candidates -> matcher -> aggregate instead of forcing all facts into one material record.",
    },
]


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def dict_to_namespace(values):
    return argparse.Namespace(**values)


def state_path_from_args(args):
    output_path = Path(args.output)
    return output_path.with_suffix(output_path.suffix + ".state.json")


def write_state_snapshot(state, current_node, next_node=None):
    args = dict_to_namespace(state["args"])
    snapshot = deepcopy(state)
    snapshot["current_node"] = current_node
    if next_node:
        snapshot["next_node"] = next_node
    state_path_from_args(args).write_text(json_dumps(snapshot), encoding="utf-8")
    return snapshot


def load_json_if_exists(path):
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8-sig"))


def load_human_advice(args):
    advice_items = []
    for item in getattr(args, "human_advice", []) or []:
        text = str(item).strip()
        if text:
            advice_items.append({"source": "cli", "advice": text})
    advice_file = getattr(args, "human_advice_file", "") or ""
    if advice_file and Path(advice_file).exists():
        content = Path(advice_file).read_text(encoding="utf-8")
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        advice_items.append({"source": "file", **item})
                    elif str(item).strip():
                        advice_items.append({"source": "file", "advice": str(item).strip()})
            elif isinstance(parsed, dict):
                advice_items.append({"source": "file", **parsed})
        except json.JSONDecodeError:
            advice_items.append({"source": "file", "advice": content.strip()})
    return advice_items[: max(0, int(getattr(args, "max_human_advice_rounds", 2) or 2))]


def advice_text(human_advice):
    chunks = []
    for item in human_advice or []:
        if isinstance(item, dict):
            text = str(item.get("advice") or item)
        else:
            text = str(item)
        if text.strip():
            chunks.append(text)
    return "\n".join(chunks)


def advice_is_acceptance(human_advice):
    text = advice_text(human_advice).lower()
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


def infer_human_advice_route(human_advice):
    text = advice_text(human_advice).lower()
    if not text:
        return ""
    if any(token in text for token in ("workflow", "split", "拆", "分批", "分段", "拓扑", "顺序", "依赖")):
        return "workflow_repair"
    if any(token in text for token in ("schema", "field", "字段", "step8", "粒度", "section")):
        return "schema_feedback_to_step8"
    if any(token in text for token in ("prompt", "提示", "json", "missing", "evidence", "证据")):
        return "prompt_repair"
    return ""


def build_reference_strategy_library(reference_pipeline):
    if not reference_pipeline:
        return {"available": False, "reason": "No --reference-pipeline was provided.", "strategies": []}
    root = Path(reference_pipeline)
    if not root.exists():
        return {"available": False, "reason": f"Reference pipeline not found: {root}", "strategies": []}
    strategies = []
    for hint in REFERENCE_STAGE_HINTS:
        existing_paths = [str(root / rel_path) for rel_path in hint["paths"] if (root / rel_path).exists()]
        prompt_files = []
        runner_files = []
        for path in existing_paths:
            path_obj = Path(path)
            prompt_files.extend(str(item) for item in path_obj.rglob("prompts/*.md"))
            runner_files.extend(str(item) for item in path_obj.glob("run*.py"))
        strategies.append(
            {
                "name": hint["name"],
                "available": bool(existing_paths),
                "paths": existing_paths,
                "prompt_files": prompt_files[:20],
                "runner_files": runner_files[:20],
                "use_when": hint["use_when"],
                "strategy": hint["strategy"],
            }
        )
    return {
        "available": True,
        "root": str(root),
        "strategies": strategies,
        "integration_policy": [
            "Use these as workflow and prompt/code design references, not as a mandatory hardcoded flow.",
            "The initial generated workflow should still be section-wise and schema-driven.",
            "During optimization, the supervisor may adopt reference strategies such as two-pass section2 extraction, figure classification before section4/section5, JSON repair, semantic normalization, voting, or multi-material manifest/fact/matcher aggregation.",
        ],
    }


def merge_update(state, update):
    merged = deepcopy(state)
    merged.update(update or {})
    return merged


def append_repair_history(state, item):
    history = deepcopy(state.get("repair_history", []))
    history.append(item)
    return history


def field_registry_from_step8(step8_output):
    result = step8_output.get("result") or step8_output
    schema = result.get("schema_definition") or {}
    return [field for field in schema.get("field_registry", []) or [] if isinstance(field, dict)]


def normalize_section_id(section_id, field_path=""):
    section_id = str(section_id or "")
    field_path = str(field_path or "")
    if section_id in THEORY_SECTION_ALIASES:
        return "section5"
    if section_id.startswith("material_info.") or section_id in {"paper_info", "section5"}:
        return section_id
    if section_id in {"section0", "section1", "section2", "section3", "section4"}:
        return f"material_info.{section_id}"
    if field_path.startswith("paper_info."):
        return "paper_info"
    if (
        field_path.startswith("section5.")
        or field_path.startswith("theory_mechanism.")
        or field_path.startswith("material_info.section5.")
        or field_path.startswith("material_info.theory_mechanism.")
    ):
        return "section5"
    for section in ("section0", "section1", "section2", "section3", "section4"):
        if f"material_info.{section}." in field_path:
            return f"material_info.{section}"
    return section_id or "material_info.section1"


def group_fields_by_section(field_registry):
    groups = {}
    for field in field_registry:
        section_id = normalize_section_id(field.get("section_id"), field.get("field_path"))
        groups.setdefault(section_id, []).append(field)
    return groups


def is_figure_field(field):
    path = str(field.get("field_path", "")).lower()
    source_basis = [str(item).lower() for item in field.get("source_basis", []) or []]
    return ".figure" in path or "figure" in source_basis or field.get("figure_constraint") is not None


def build_workflow_plan(step8_output, prompt_output=None, reference_library=None):
    field_registry = field_registry_from_step8(step8_output)
    groups = group_fields_by_section(field_registry)
    figure_fields = [field for field in field_registry if is_figure_field(field)]
    has_section4 = bool(groups.get("material_info.section4"))
    has_section5 = bool(groups.get("section5"))
    has_section2 = bool(groups.get("material_info.section2"))

    section_test_plan = []
    for section_id in SECTION_ORDER:
        if section_id == "figure_classification":
            if has_section4 and has_section5 and figure_fields:
                section_test_plan.append(
                    {
                        "stage_id": "figure_classification",
                        "section_id": "figure_classification",
                        "purpose": "Classify figures before section4/section5 extraction to reduce curve/mechanism ownership errors.",
                        "depends_on": ["material_info.section3"],
                        "test_focus": ["figure ownership", "allowed sections", "section4 versus section5 boundary"],
                    }
                )
            continue
        if section_id not in groups and section_id != "paper_info":
            continue
        depends_on = []
        if section_id != "paper_info":
            depends_on.append("paper_info")
        if section_id in {"material_info.section1", "material_info.section2", "material_info.section3", "material_info.section4", "section5"}:
            depends_on.append("material_info.section0")
        if section_id == "material_info.section4" and has_section4 and has_section5 and figure_fields:
            depends_on.append("figure_classification")
        if section_id == "section5" and has_section4 and figure_fields:
            depends_on.append("figure_classification")
        if section_id == "material_info.section1" and has_section4:
            depends_on.append("material_info.section4")
        if section_id == "material_info.section2" and has_section2:
            section_test_plan.append(
                {
                    "stage_id": "material_info.section2.method_pass",
                    "section_id": "material_info.section2",
                        "purpose": "First pass for fabrication/processing: extract method, route, sample geometry, and process sequence.",
                    "depends_on": list(dict.fromkeys(depends_on)),
                    "field_count": len(groups.get(section_id, [])),
                    "test_focus": ["method coverage", "process sequence", "sample geometry", "json validity"],
                }
            )
            section_test_plan.append(
                {
                    "stage_id": "material_info.section2.conditions_pass",
                    "section_id": "material_info.section2",
                        "purpose": "Second pass for fabrication/processing: extract temperature, time, pressure, atmosphere, environment, treatment, and other condition details using the method pass as context.",
                    "depends_on": list(dict.fromkeys([*depends_on, "material_info.section2.method_pass"])),
                    "field_count": len(groups.get(section_id, [])),
                    "test_focus": ["condition coverage", "unit preservation", "method-condition alignment", "missing_reason"],
                }
            )
            continue
        section_test_plan.append(
            {
                "stage_id": section_id,
                "section_id": section_id,
                "purpose": f"Extract and validate {section_id} fields.",
                "depends_on": list(dict.fromkeys(depends_on)),
                "field_count": len(groups.get(section_id, [])),
                "test_focus": ["json validity", "field coverage", "evidence provenance", "missing_reason"],
            }
        )

    workflow_repairs = []
    if has_section2:
        workflow_repairs.append(
            {
                "trigger": "section2 method and condition extraction are misaligned after the default two-pass workflow",
                "action": "repair_section2_two_pass_prompts",
                "implementation": [
                    "strengthen method_pass to preserve process sequence and sample geometry",
                    "strengthen conditions_pass to bind temperature/time/pressure/atmosphere to the correct method step",
                ],
            }
        )
    if has_section4 and has_section5 and figure_fields:
        workflow_repairs.append(
            {
                "trigger": "section4 curves and section5 mechanism/theory evidence are confused",
                "action": "add_figure_classification_before_section4_section5",
                "implementation": [
                    "classify every figure/panel by evidence type and allowed section",
                    "route curve figures to material_info.section4 and theory/simulation/mechanism figures to section5 or section3 as defined by Step8",
                ],
            }
        )

    return {
        "workflow_name": "step9_extraction_build_agent_system",
        "objective": "Generate prompts and code, run section-wise extraction tests, and repair prompt/code/workflow based on measured extraction quality.",
        "field_count": len(field_registry),
        "sections": [
            {"section_id": section_id, "field_count": len(fields)}
            for section_id, fields in sorted(groups.items())
        ],
        "execution_order": [item["stage_id"] for item in section_test_plan],
        "section_test_plan": section_test_plan,
        "dynamic_workflow_repairs": workflow_repairs,
        "reference_strategy_library": reference_library or {"available": False, "strategies": []},
        "supervisor_routes": [
            "accept",
            "prompt_repair",
            "code_repair",
            "workflow_repair",
            "schema_feedback_to_step8",
            "needs_human_review",
        ],
    }


def build_prompt_package_from_step8(step8_output, workflow_plan):
    field_registry = field_registry_from_step8(step8_output)
    figure_fields = [field for field in field_registry if is_figure_field(field)]
    field_index = [
        {
            "field_path": field.get("field_path"),
            "section_id": normalize_section_id(field.get("section_id"), field.get("field_path")),
            "data_type": field.get("data_type"),
            "source_basis": field.get("source_basis", []),
        }
        for field in field_registry
    ]
    prompts = {}
    for stage in workflow_plan.get("section_test_plan", []):
        section_id = stage["section_id"]
        prompts[stage["stage_id"]] = {
            "prompt": (
                f"Run stage {stage['stage_id']} for {section_id} according to the Step8 schema. "
                "Return valid JSON only. Preserve evidence provenance with source_text, source_table, "
                "source_figure, confidence, and missing_reason when values are absent. "
                f"Respect dependencies: {', '.join(stage.get('depends_on', [])) or 'none'}. "
                "If this stage repeatedly fails, consult workflow_plan.reference_strategy_library for known extraction patterns."
            ),
            "output_contract": {
                "section_id": section_id,
                "json_only": True,
                "fields": [
                    item["field_path"]
                    for item in field_index
                    if item["section_id"] == section_id
                ][:200],
            },
        }

    return {
        "step": "step9_prompt_generation",
        "status": "success",
        "validation_errors": [],
        "shared_prompt_context": {
            "field_index": field_index,
            "figure_fields": figure_fields,
            "workflow_plan": workflow_plan,
        },
        "module_outputs": {
            "prompt_supervisor_module": {
                "status": "success",
                "workflow_position": "Step9 prompt/code/test closed loop",
                "section_test_policy": "section-wise tests with dependency topology",
            },
            "classification_prompt_module": prompts.get("paper_info", {}),
            "section_extraction_prompt_module": {
                "section_extraction_prompts": prompts,
                "field_coverage_index": field_index,
            },
            "figure_extraction_prompt_module": {
                "prompt": "Classify figures/panels by allowed section before extracting section4 curves and section5 theory/mechanism evidence.",
                "figure_field_coverage_index": figure_fields,
            },
            "postprocess_repair_prompt_module": {
                "prompt": "Repair invalid JSON, missing fields, section ownership conflicts, figure ownership errors, and provenance gaps. Return JSON only.",
                "repair_types": ["json", "missing", "ownership", "figure", "provenance"],
            },
            "prompt_package_aggregation": {
                "execution_order": workflow_plan.get("execution_order", []),
            },
        },
        "result": {
            "execution_order": ["classification", "section_extraction", "figure_extraction", "postprocess_repair"],
            "workflow_execution_order": workflow_plan.get("execution_order", []),
            "reference_strategy_library": workflow_plan.get("reference_strategy_library", {}),
            "prompt_modules": {
                "classification": prompts.get("paper_info", {}),
                "section_extraction": prompts,
                "figure_extraction": "figure_extraction_prompt_module",
                "postprocess_repair": "postprocess_repair_prompt_module",
            },
        },
    }


def load_inputs_node(state):
    args = dict_to_namespace(state["args"])
    step8_output = load_json_if_exists(args.step8_output)
    prompt_output = load_json_if_exists(args.prompt_output)
    human_advice = load_human_advice(args)
    errors = []
    if not step8_output and not prompt_output:
        errors.append("Either --step8-output or --prompt-output must point to an existing JSON file.")
    update = {
        "step8_output": step8_output,
        "prompt_output": prompt_output,
        "human_advice": human_advice,
        "validation_errors": errors,
        "retry_counts": {},
        "repair_history": [],
        "status": "running" if not errors else "needs_human_review",
    }
    return write_state_snapshot(merge_update(state, update), "load_inputs", "supervisor_router" if errors else "workflow_plan")


def workflow_plan_node(state):
    args = dict_to_namespace(state["args"])
    step8_output = state.get("step8_output") or {}
    prompt_output = state.get("prompt_output") or {}
    if not step8_output and prompt_output:
        step8_output = {"result": {"schema_definition": {"field_registry": (prompt_output.get("shared_prompt_context") or {}).get("field_index", [])}}}
    reference_library = build_reference_strategy_library(args.reference_pipeline)
    workflow_plan = build_workflow_plan(step8_output, prompt_output, reference_library)
    return write_state_snapshot(
        merge_update(state, {"workflow_plan": workflow_plan, "status": "running"}),
        "workflow_plan",
        "prompt_generate",
    )


def prompt_generate_node(state):
    prompt_output = deepcopy(state.get("prompt_output") or {})
    if not prompt_output:
        prompt_output = build_prompt_package_from_step8(state.get("step8_output") or {}, state["workflow_plan"])
    else:
        prompt_output.setdefault("shared_prompt_context", {})
        prompt_output["shared_prompt_context"]["workflow_plan"] = state["workflow_plan"]
    return write_state_snapshot(
        merge_update(state, {"prompt_output": prompt_output}),
        "prompt_generate",
        "prompt_quality_review",
    )


def prompt_quality_review_node(state):
    judgement = prompt_agent.judge_prompt_package(state["prompt_output"])
    update = {"judgement": judgement}
    if judgement.get("status") == "needs_code_action":
        update["validation_errors"] = ["Prompt package has high-severity quality or format blockers."]
        update["status"] = "awaiting_supervisor_decision"
        update["failed_node"] = "prompt_quality_review"
    else:
        update["validation_errors"] = []
        update["status"] = "running"
    return write_state_snapshot(
        merge_update(state, update),
        "prompt_quality_review",
        "supervisor_router" if update.get("validation_errors") else "code_generate_or_patch",
    )


def prompt_repair_node(state):
    prompt_output = deepcopy(state.get("prompt_output") or {})
    judgement = state.get("judgement") or {}
    workflow_plan = state.get("workflow_plan") or {}
    eval_result = state.get("extraction_eval") or {}
    human_advice_text = advice_text(state.get("human_advice", [])) if state.get("human_review_applied") else ""
    repair_note = {
        "source": "prompt_repair_agent",
        "reason": "Supervisor routed prompt package through deterministic repair.",
        "judgement_status": judgement.get("status"),
        "issue_count": (judgement.get("summary") or {}).get("issue_count", 0),
        "warning_count": (judgement.get("summary") or {}).get("warning_count", 0),
        "evaluation_targets": eval_result.get("optimization_targets", []),
        "human_advice_used": bool(human_advice_text),
    }

    prompt_output.setdefault("shared_prompt_context", {})
    prompt_output["shared_prompt_context"]["workflow_plan"] = workflow_plan
    prompt_output["shared_prompt_context"].setdefault("repair_notes", []).append(repair_note)

    modules = prompt_output.setdefault("module_outputs", {})
    supervisor_module = modules.setdefault("prompt_supervisor_module", {})
    supervisor_module["repair_policy"] = [
        "Run extraction section by section using workflow_plan.execution_order.",
        "Treat dependency outputs as context, not as fields to overwrite.",
        "If a field is missing, emit missing_reason instead of inventing values.",
        "If section4/section5 figure ownership is ambiguous, use figure_classification before extraction.",
        "If section2 method and condition details are misaligned, use method_pass then conditions_pass.",
        "If JSON output is truncated or invalid, reduce per-call field scope or split the stage into smaller extraction batches before retrying.",
        "If extracted values are mostly null, narrow the prompt to only evidence-backed fields and require missing_reason outside extracted_fields.",
        "If evaluation reports unicode_normalization_required, insert or preserve an explicit unicode_normalization postprocess stage; do not hide this behavior inside extraction prompts.",
    ]
    if eval_result.get("failure_diagnoses"):
        supervisor_module["latest_extraction_eval_diagnoses"] = eval_result.get("failure_diagnoses")
    if human_advice_text:
        supervisor_module["human_advice"] = human_advice_text
    repair_module = modules.setdefault("postprocess_repair_prompt_module", {})
    repair_module["prompt"] = (
        str(repair_module.get("prompt") or "")
        + "\n\nRepair policy: validate JSON, enforce section ownership, preserve source_text/source_table/source_figure, "
        "repair missing_reason fields, and do not merge outputs across different section stages unless the workflow plan allows it."
    ).strip()
    prompt_output["status"] = "success"
    prompt_output["validation_errors"] = []

    update = {
        "prompt_output": prompt_output,
        "validation_errors": [],
        "failed_node": "",
        "status": "running",
        "repair_history": append_repair_history(state, repair_note),
    }
    return write_state_snapshot(merge_update(state, update), "prompt_repair", "prompt_quality_review")


def code_generate_or_patch_node(state):
    args = dict_to_namespace(state["args"])
    if args.skip_code_generation:
        code_agent = {
            "status": "skipped",
            "reason": "Code generation skipped by --skip-code-generation.",
            "code_change_plan": [],
            "generated_files": [],
        }
    else:
        prompt = prompt_agent.build_code_agent_prompt(
            state["prompt_output"],
            state["judgement"],
            args.target_files or ["code/downstream_extraction_runner.py"],
        )
        try:
            raw_response = prompt_agent.litellm_chat(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            code_agent = prompt_agent.parse_llm_json(raw_response)
            if args.write_generated_files:
                prompt_agent.write_generated_files(code_agent, args.generated_output_dir)
        except Exception as exc:
            code_agent = {
                "status": "llm_timeout_or_error",
                "reason": str(exc),
                "fallback": "Continue with the connected downstream extraction runner; extraction_eval will decide whether code repair is still required.",
                "code_change_plan": [],
                "generated_files": [],
            }
    return write_state_snapshot(
        merge_update(state, {"code_agent": code_agent, "status": "running"}),
        "code_generate_or_patch",
        "extraction_test_run",
    )


def code_repair_node(state):
    args = dict_to_namespace(state["args"])
    workflow_plan = state.get("workflow_plan") or {}
    eval_result = state.get("extraction_eval") or {}
    reference_library = workflow_plan.get("reference_strategy_library") or {}
    code_agent = {
        "status": "needs_runner_implementation",
        "diagnosis": [
            "Step9 workflow, prompt package, and section-wise test plan exist, but no live downstream extraction runner is connected.",
            "The runner should execute workflow_plan.section_test_plan in dependency order and write per-stage JSON artifacts.",
        ],
        "recommended_reference_strategies": [
            item["name"]
            for item in reference_library.get("strategies", [])
            if item.get("available")
            and item.get("name")
            in {
                "classification_gate",
                "section2_two_pass",
                "figure_classification_before_s4_s5",
                "json_check_and_schema_cleanup",
                "unicode_semantic_normalization",
                "final_merge",
            }
        ],
        "code_change_plan": [
            {
                "file_path": "code/downstream_extraction_runner.py",
                "change_type": "add",
                "reason": "Execute Step9 section stages, pass dependency outputs as context, and persist per-section JSON outputs.",
                "implementation_notes": [
                    "Read step9_extraction_build_output.json or prompt package JSON.",
                    "Run stages in workflow_plan.execution_order.",
                    "For each stage, load prompt_output.result.prompt_modules.section_extraction[stage_id].",
                    "Validate JSON, field coverage, evidence provenance, and missing_reason.",
                    "Write stage outputs under a run directory so extraction_eval can inspect them.",
                ],
            },
            {
                "file_path": "code/step9_extraction_build_graph.py",
                "change_type": "modify",
                "reason": "Connect extraction_test_run to the live downstream runner once implemented.",
                "implementation_notes": [
                    "Replace runner_missing with subprocess or direct function call.",
                    "Feed section results to extraction_eval.",
                ],
            },
        ],
        "blocked_reason": eval_result.get("blockers", []),
        "write_generated_files": bool(getattr(args, "write_generated_files", False)),
    }
    update = {
        "code_agent": code_agent,
        "status": "needs_code_action",
        "repair_history": append_repair_history(
            state,
            {
                "source": "code_repair_agent",
                "action": "planned_downstream_runner",
                "reason": "No live extraction runner was connected.",
            },
        ),
    }
    return write_state_snapshot(merge_update(state, update), "code_repair", "write_output")


def extraction_test_run_node(state):
    args = dict_to_namespace(state["args"])
    workflow_plan = state.get("workflow_plan") or {}
    test_docs = [str(path) for path in args.test_documents]
    extraction_results = load_json_if_exists(args.extraction_results)
    section_results = []
    if extraction_results:
        section_results = extraction_results.get("section_results", [])
        test_status = extraction_results.get("status", "completed")
        reason = "Loaded external extraction test results."
    elif not args.dry_run and test_docs:
        extraction_test = downstream_extraction_runner.run_extraction_bench(
            workflow_plan=workflow_plan,
            prompt_output=state.get("prompt_output") or {},
            documents=test_docs,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.extraction_max_tokens,
            output_dir=args.extraction_run_dir,
            max_documents=args.max_test_documents,
            max_document_chars=args.max_document_chars,
            max_fields_per_stage=args.max_fields_per_stage,
        )
        return write_state_snapshot(
            merge_update(state, {"extraction_test": extraction_test, "status": "running"}),
            "extraction_test_run",
            "extraction_eval",
        )
    else:
        for stage in workflow_plan.get("section_test_plan", []):
            section_results.append(
                {
                    "stage_id": stage["stage_id"],
                    "section_id": stage["section_id"],
                    "depends_on": stage.get("depends_on", []),
                    "status": "planned" if args.dry_run else "not_implemented",
                    "test_documents": test_docs,
                    "checks": ["dependency_inputs_available", "json_validity", "field_coverage", "evidence_provenance"],
                }
            )
        test_status = "dry_run_planned" if args.dry_run else "runner_missing"
        reason = "Section-wise extraction tests are planned; plug in downstream_extraction_runner for live execution."
    extraction_test = {
        "status": test_status,
        "reason": reason,
        "section_results": section_results,
    }
    return write_state_snapshot(
        merge_update(state, {"extraction_test": extraction_test, "status": "running"}),
        "extraction_test_run",
        "extraction_eval",
    )


def diagnose_section_result(section_result, unicode_normalized=False):
    stage_id = section_result.get("stage_id", "")
    status = section_result.get("status", "")
    summary = section_result.get("summary") or {}
    doc_results = section_result.get("document_results", []) or []
    diagnoses = []

    invalid_docs = [
        item for item in doc_results if isinstance(item, dict) and item.get("status") == "invalid_json"
    ]
    if invalid_docs:
        diagnoses.append(
            {
                "type": "invalid_json",
                "severity": "blocking",
                "stage_id": stage_id,
                "documents": [item.get("document") for item in invalid_docs],
                "evidence": [
                    (item.get("checks") or {}).get("error", "invalid json")
                    for item in invalid_docs[:3]
                ],
                "likely_cause": "The stage output is too long or insufficiently constrained, causing truncated or malformed JSON.",
                "recommended_repair": "Split this stage into smaller field batches or add a JSON repair/retry pass for only failed documents.",
                "route_hint": "workflow_repair" if int(section_result.get("field_count_tested", 0) or 0) > 30 else "prompt_repair",
            }
        )

    extracted_total = int(summary.get("extracted_total", 0) or 0)
    is_postprocess_stage = stage_id == "unicode_normalization" or str(section_result.get("section_id", "")).startswith("postprocess.")
    if status == "low_quality" or extracted_total == 0 and stage_id != "figure_classification" and not is_postprocess_stage:
        diagnoses.append(
            {
                "type": "low_extraction_yield",
                "severity": "warning",
                "stage_id": stage_id,
                "documents": [item.get("document") for item in doc_results if isinstance(item, dict)],
                "likely_cause": "The prompt/schema may be too broad, too vague, or mismatched to the documents.",
                "recommended_repair": "Tighten the stage prompt around evidence-backed fields and push absent fields into missing_fields.",
                "route_hint": "prompt_repair",
            }
        )

    null_like_total = sum(int((item.get("checks") or {}).get("null_like_count", 0) or 0) for item in doc_results if isinstance(item, dict))
    extracted_with_nulls = sum(int((item.get("checks") or {}).get("extracted_count", 0) or 0) for item in doc_results if isinstance(item, dict))
    if extracted_with_nulls and null_like_total / max(1, extracted_with_nulls) > 0.35:
        diagnoses.append(
            {
                "type": "null_values_in_extracted_fields",
                "severity": "warning",
                "stage_id": stage_id,
                "ratio": round(null_like_total / max(1, extracted_with_nulls), 3),
                "likely_cause": "The model is placing unavailable fields into extracted_fields instead of missing_fields.",
                "recommended_repair": "Prompt should forbid null-valued extracted_fields and require missing_reason entries instead.",
                "route_hint": "prompt_repair",
            }
        )

    if status == "missing_dependency":
        diagnoses.append(
            {
                "type": "missing_dependency",
                "severity": "blocking",
                "stage_id": stage_id,
                "depends_on": section_result.get("depends_on", []),
                "likely_cause": "Workflow topology allowed a stage to run without required upstream context.",
                "recommended_repair": "Enforce dependency gating or reorder the affected stage.",
                "route_hint": "workflow_repair",
            }
        )

    unicode_candidate_total = int((summary or {}).get("unicode_candidate_total", 0) or 0)
    if unicode_candidate_total and stage_id != "unicode_normalization" and not unicode_normalized:
        diagnoses.append(
            {
                "type": "unicode_normalization_required",
                "severity": "blocking",
                "stage_id": stage_id,
                "unicode_candidate_total": unicode_candidate_total,
                "likely_cause": "Extracted values contain formula digits, LaTeX-style notation, or inconsistent scientific Unicode rendering.",
                "recommended_repair": "Insert a schema-aware unicode_normalization postprocess stage after section extraction and before final merge.",
                "route_hint": "workflow_repair",
            }
        )

    return diagnoses


def build_extraction_judgement(section_results, extraction_test, workflow_plan, human_advice):
    diagnoses = []
    unicode_normalized = any(
        isinstance(item, dict)
        and item.get("stage_id") == "unicode_normalization"
        and item.get("status") == "passed"
        for item in section_results
    )
    for section_result in section_results:
        if isinstance(section_result, dict):
            diagnoses.extend(diagnose_section_result(section_result, unicode_normalized=unicode_normalized))

    blocking = [item for item in diagnoses if item.get("severity") == "blocking"]
    route_votes = [item.get("route_hint") for item in diagnoses if item.get("route_hint")]
    if extraction_test.get("status") == "runner_missing":
        recommended = "code_repair"
    elif any(route == "workflow_repair" for route in route_votes):
        recommended = "workflow_repair"
    elif diagnoses:
        recommended = "prompt_repair"
    else:
        recommended = "accept_live_test_result" if extraction_test.get("status") == "completed" else "accept_dry_run_plan"

    optimization_targets = []
    for diagnosis in diagnoses:
        optimization_targets.append(
            {
                "stage_id": diagnosis.get("stage_id"),
                "target": diagnosis.get("type"),
                "repair_route": diagnosis.get("route_hint"),
                "recommended_repair": diagnosis.get("recommended_repair"),
            }
        )

    return {
        "status": "failed" if blocking else "passed_with_warnings" if diagnoses else "passed",
        "recommended_next_action": recommended,
        "failure_diagnoses": diagnoses,
        "optimization_targets": optimization_targets,
        "human_advice": human_advice,
        "workflow_repair_candidates": workflow_plan.get("dynamic_workflow_repairs", []),
    }


def extraction_eval_node(state):
    extraction_test = state.get("extraction_test") or {}
    section_results = extraction_test.get("section_results", [])
    workflow_plan = state.get("workflow_plan") or {}
    human_advice = state.get("human_advice", []) if state.get("human_review_applied") else []
    judgement = build_extraction_judgement(section_results, extraction_test, workflow_plan, human_advice)
    blockers = []
    if extraction_test.get("status") == "runner_missing":
        blockers.append("No live extraction runner is connected.")
    if not section_results:
        blockers.append("No section-wise tests were planned.")
    failed_sections = [
        item
        for item in section_results
        if isinstance(item, dict) and item.get("status") in {"failed", "needs_repair", "invalid_json", "low_quality"}
    ]
    missing_dependency_sections = [
        item
        for item in section_results
        if isinstance(item, dict) and item.get("status") == "missing_dependency"
    ]
    if failed_sections:
        blockers.append("One or more section extraction stages failed quality checks.")
    if missing_dependency_sections:
        blockers.append("One or more section extraction stages missed required dependency outputs.")
    if judgement.get("status") == "failed":
        blockers.append("Extraction judgement failed; supervisor repair is required.")
    recommended_next_action = judgement.get("recommended_next_action", "accept_live_test_result")
    eval_result = {
        "status": "needs_code_action" if blockers else "ready_for_live_test",
        "blockers": blockers,
        "section_test_count": len(section_results),
        "failed_sections": failed_sections,
        "missing_dependency_sections": missing_dependency_sections,
        "workflow_repairs_available": workflow_plan.get("dynamic_workflow_repairs", []),
        "recommended_next_action": recommended_next_action,
        "extraction_judgement": judgement,
        "failure_diagnoses": judgement.get("failure_diagnoses", []),
        "optimization_targets": judgement.get("optimization_targets", []),
        "human_advice": human_advice,
    }
    update = {"extraction_eval": eval_result}
    if blockers:
        update["validation_errors"] = blockers
        update["status"] = "awaiting_supervisor_decision"
        update["failed_node"] = "extraction_eval"
        update["next_node"] = "supervisor_router"
    else:
        update["validation_errors"] = []
        update["status"] = "success"
        update["next_node"] = "human_expert_review"
    return write_state_snapshot(
        merge_update(state, update),
        "extraction_eval",
        "supervisor_router" if blockers else "human_expert_review",
    )


def human_expert_review_node(state):
    args = dict_to_namespace(state["args"])
    human_advice = state.get("human_advice", []) or []
    if not human_advice and getattr(args, "skip_human_expert_review", False):
        return write_state_snapshot(
            merge_update(
                state,
                {
                    "human_expert_review": {
                        "status": "skipped",
                        "reason": "Human expert review was explicitly skipped by --skip-human-expert-review.",
                    },
                    "status": "success",
                },
            ),
            "human_expert_review",
            "write_output",
        )
    if not human_advice:
        return write_state_snapshot(
            merge_update(
                state,
                {
                    "human_expert_review": {
                        "status": "waiting_for_human_advice",
                        "reason": "Internal Step9 evaluation passed. Human expert review is required unless --skip-human-expert-review is set.",
                        "required_advice": (
                            "As the task-domain materials expert, either approve the current Step9 extraction/test result "
                            "or describe concrete domain-output problems. Do not provide code or prompt implementation instructions."
                        ),
                    },
                    "status": "waiting_for_human_advice",
                    "next_node": "write_output",
                },
            ),
            "human_expert_review",
            "write_output",
        )

    current_advice_text = advice_text(human_advice)
    previously_applied_advice = str(state.get("applied_human_advice") or "").strip()
    current_advice_already_applied = bool(
        state.get("human_review_applied")
        and current_advice_text
        and current_advice_text == previously_applied_advice
    )
    if current_advice_already_applied or advice_is_acceptance(human_advice):
        return write_state_snapshot(
            merge_update(
                state,
                {
                    "human_expert_review": {
                        "status": "accepted",
                        "advice": human_advice,
                        "reason": "Human expert review accepted the internally passing Step9 output, or the same advice was already applied once.",
                    },
                    "status": "success",
                    "validation_errors": [],
                },
            ),
            "human_expert_review",
            "write_output",
        )

    route = infer_human_advice_route(human_advice) or "prompt_repair"
    retry_counts = deepcopy(state.get("retry_counts", {}))
    retry_counts.pop("extraction_eval", None)
    update = {
        "human_review_applied": True,
        "human_expert_review": {
            "status": "expert_revision_requested",
            "advice": human_advice,
            "route_hint": route,
            "policy": "Expert advice is applied only after prompt/code/extraction evaluation passed internally.",
        },
        "validation_errors": [
            "human expert review requested Step9 revision after internal acceptance",
            current_advice_text,
        ],
        "applied_human_advice": current_advice_text,
        "retry_counts": retry_counts,
        "status": "awaiting_supervisor_decision",
        "failed_node": "extraction_eval",
    }
    return write_state_snapshot(merge_update(state, update), "human_expert_review", "supervisor_router")


def workflow_repair_node(state):
    workflow_plan = deepcopy(state.get("workflow_plan") or {})
    eval_result = state.get("extraction_eval") or {}
    failed_sections = eval_result.get("failed_sections", []) or []
    missing_dependencies = eval_result.get("missing_dependency_sections", []) or []
    diagnoses = eval_result.get("failure_diagnoses", []) or []
    human_advice_text = advice_text(state.get("human_advice", [])) if state.get("human_review_applied") else ""
    repair_actions = []

    stage_ids = " ".join(str(item.get("stage_id", "")) for item in [*failed_sections, *missing_dependencies])
    invalid_large_stages = [
        item.get("stage_id")
        for item in diagnoses
        if item.get("type") == "invalid_json" and item.get("route_hint") == "workflow_repair"
    ]
    unicode_stages = [
        item.get("stage_id")
        for item in diagnoses
        if item.get("type") == "unicode_normalization_required"
    ]
    if unicode_stages:
        repair_actions.append("insert_unicode_normalization_stage")
        workflow_plan["unicode_normalization_policy"] = {
            "enabled": True,
            "source": "workflow_repair_agent",
            "reason": "Extraction evaluator detected inconsistent formula/Unicode notation.",
            "trigger_stages": sorted(set(unicode_stages)),
            "mode": "builtin_unicode_latex_formula_normalization",
            "target_field_patterns": [
                "primary_signature",
                "nominal_formula",
                "material_system",
                "sample_variants",
                "substitution_series",
                "interfaces.layer_stack",
                "value",
                "unit",
                "doi",
            ],
        }
        existing = {stage.get("stage_id") for stage in workflow_plan.get("section_test_plan", [])}
        if "unicode_normalization" not in existing:
            workflow_plan.setdefault("section_test_plan", []).append(
                {
                    "stage_id": "unicode_normalization",
                    "section_id": "postprocess.unicode_normalization",
                    "purpose": "Normalize scientific Unicode, LaTeX fragments, formula subscripts/superscripts, DOI punctuation, and selected schema fields after raw extraction.",
                    "depends_on": [stage.get("stage_id") for stage in workflow_plan.get("section_test_plan", [])],
                    "field_count": len(workflow_plan["unicode_normalization_policy"]["target_field_patterns"]),
                    "test_focus": ["unicode consistency", "formula rendering", "schema-aware target selection"],
                    "repair_notes": ["Inserted because evaluator detected Unicode/formula normalization candidates."],
                }
            )
    if invalid_large_stages:
        repair_actions.append("split_large_invalid_json_stages")
        workflow_plan.setdefault("stage_batching_policy", {})
        for stage_id in invalid_large_stages:
            workflow_plan["stage_batching_policy"][stage_id] = {
                "reason": "Invalid/truncated JSON on long stage output.",
                "max_fields_per_call": 20,
                "retry_failed_documents_only": True,
                "merge_policy": "merge batch outputs by field_path and material_system after JSON validation",
            }
            for stage in workflow_plan.get("section_test_plan", []):
                if stage.get("stage_id") == stage_id:
                    stage.setdefault("repair_notes", []).append(
                        "Split this stage into smaller field batches before retrying because invalid JSON indicates oversized output."
                    )
    if "section2" in stage_ids:
        repair_actions.append("strengthen_section2_two_pass_dependency")
        for stage in workflow_plan.get("section_test_plan", []):
            if stage.get("stage_id") == "material_info.section2.conditions_pass":
                stage["depends_on"] = list(
                    dict.fromkeys([*stage.get("depends_on", []), "material_info.section2.method_pass"])
                )
                stage.setdefault("repair_notes", []).append(
                    "conditions_pass must consume method_pass output and bind each condition to the correct fabrication/processing step."
                )
    if "figure_classification" in stage_ids or "section4" in stage_ids or "section5" in stage_ids:
        repair_actions.append("strengthen_figure_classification_route")
        existing = {stage.get("stage_id") for stage in workflow_plan.get("section_test_plan", [])}
        if "figure_classification" not in existing:
            workflow_plan.setdefault("section_test_plan", []).insert(
                0,
                {
                    "stage_id": "figure_classification",
                    "section_id": "figure_classification",
                    "purpose": "Repair-added figure classification before figure-heavy section extraction.",
                    "depends_on": ["material_info.section3"],
                    "test_focus": ["figure ownership", "allowed sections", "section4 versus section5 boundary"],
                    "repair_notes": ["Added because extraction evaluation found figure or section4/section5 confusion."],
                },
            )
    if missing_dependencies:
        repair_actions.append("enforce_dependency_context")
        workflow_plan["dependency_policy"] = {
            "mode": "strict",
            "rule": "A stage may run only after all depends_on outputs are available or explicitly marked unavailable with a missing_reason.",
        }
    if human_advice_text:
        repair_actions.append("apply_human_workflow_advice")
        workflow_plan.setdefault("human_advice", []).append(human_advice_text)

    workflow_plan["execution_order"] = [stage["stage_id"] for stage in workflow_plan.get("section_test_plan", [])]
    workflow_plan.setdefault("workflow_repair_history", []).append(
        {
            "actions": repair_actions or ["no_structural_change"],
            "source_eval_status": eval_result.get("status"),
            "blockers": eval_result.get("blockers", []),
            "diagnoses": diagnoses,
            "human_advice_used": bool(human_advice_text),
        }
    )
    update = {
        "workflow_plan": workflow_plan,
        "validation_errors": [],
        "failed_node": "",
        "status": "running",
        "repair_history": append_repair_history(
            state,
            {
                "source": "workflow_repair_agent",
                "actions": repair_actions or ["no_structural_change"],
            },
        ),
    }
    return write_state_snapshot(merge_update(state, update), "workflow_repair", "prompt_repair")


def schema_feedback_node(state):
    eval_result = state.get("extraction_eval") or {}
    judgement = state.get("judgement") or {}
    human_advice = state.get("human_advice", []) if state.get("human_review_applied") else []
    schema_feedback = {
        "status": "needs_step8_review",
        "reason": "Step9 could not repair extraction quality through prompt/code/workflow changes.",
        "signals": {
            "validation_errors": state.get("validation_errors", []),
            "extraction_eval": eval_result,
            "judgement_summary": judgement.get("summary"),
            "human_advice": human_advice,
        },
        "recommended_step8_actions": [
            "Check whether field granularity is too fine or too vague for reliable extraction.",
            "Check whether section ownership rules are ambiguous.",
            "Add missing evidence/provenance fields if extraction repeatedly lacks traceability.",
            "Simplify or regroup fields that cannot be extracted section-wise from paper text.",
            "If Step9 reports oversized invalid JSON for a section, split the section fields into smaller extraction groups or add a section-level batching rule.",
        ],
    }
    update = {
        "schema_feedback": schema_feedback,
        "status": "needs_step8_review",
        "repair_history": append_repair_history(
            state,
            {"source": "schema_feedback_agent", "action": "feedback_to_step8"},
        ),
    }
    return write_state_snapshot(merge_update(state, update), "schema_feedback", "write_output")


def supervisor_router_node(state):
    errors = state.get("validation_errors") or []
    retry_counts = deepcopy(state.get("retry_counts", {}))
    failed_node = state.get("failed_node") or state.get("current_node") or ""
    retry_count = int(retry_counts.get(failed_node, 0))
    args = dict_to_namespace(state["args"])
    max_retries = int(getattr(args, "max_supervisor_retries", 1) or 1)

    eval_result = state.get("extraction_eval") or {}
    recommended_action = eval_result.get("recommended_next_action", "")
    human_route = infer_human_advice_route(state.get("human_advice", [])) if state.get("human_review_applied") else ""

    if not errors:
        decision = {"action": "continue", "next_node": state.get("next_node") or "write_output"}
    elif failed_node == "prompt_quality_review" and retry_count < max_retries:
        retry_counts[failed_node] = retry_count + 1
        decision = {
            "action": "prompt_repair",
            "reason": "Prompt package quality checks failed; regenerate prompt package with workflow plan attached.",
            "next_node": "prompt_repair",
        }
    elif failed_node == "extraction_eval" and human_route == "workflow_repair" and retry_count < max_retries:
        retry_counts[failed_node] = retry_count + 1
        decision = {
            "action": "workflow_repair",
            "reason": "Human advice requested workflow/topology or batching adjustment.",
            "next_node": "workflow_repair",
        }
    elif failed_node == "extraction_eval" and human_route == "prompt_repair" and retry_count < max_retries:
        retry_counts[failed_node] = retry_count + 1
        decision = {
            "action": "prompt_repair",
            "reason": "Human advice requested prompt/output-contract adjustment.",
            "next_node": "prompt_repair",
        }
    elif failed_node == "extraction_eval" and human_route == "schema_feedback_to_step8":
        decision = {
            "action": "schema_feedback_to_step8",
            "reason": "Human advice indicates Step8 schema/field design should be reviewed.",
            "next_node": "schema_feedback",
        }
    elif failed_node == "extraction_eval" and recommended_action == "workflow_repair" and retry_count < max_retries:
        retry_counts[failed_node] = retry_count + 1
        decision = {
            "action": "workflow_repair",
            "reason": "Section-wise extraction evaluation indicates dependency or workflow topology failure.",
            "next_node": "workflow_repair",
        }
    elif failed_node == "extraction_eval" and recommended_action == "prompt_repair" and retry_count < max_retries:
        retry_counts[failed_node] = retry_count + 1
        decision = {
            "action": "prompt_repair",
            "reason": "Section-wise extraction evaluation indicates prompt-level quality failure.",
            "next_node": "prompt_repair",
        }
    elif "No live extraction runner is connected." in errors:
        decision = {
            "action": "code_repair",
            "reason": "The Step9 graph needs a downstream extraction runner to execute live tests.",
            "next_node": "code_repair",
        }
    elif failed_node == "extraction_eval":
        decision = {
            "action": "schema_feedback_to_step8",
            "reason": "Prompt/workflow retries were exhausted; Step8 schema may need revision.",
            "next_node": "schema_feedback",
        }
    else:
        decision = {
            "action": "needs_human_review",
            "reason": "No safe automatic route remains.",
            "next_node": "write_output",
        }

    update = {
        "supervisor_decision": decision,
        "retry_counts": retry_counts,
        "status": "running" if decision["next_node"] != "write_output" else state.get("status", "needs_human_review"),
    }
    if decision["next_node"] != "write_output":
        update["validation_errors"] = []
    return write_state_snapshot(merge_update(state, update), "supervisor_router", decision["next_node"])


def write_output_node(state):
    args = dict_to_namespace(state["args"])
    human_expert_review = state.get("human_expert_review")
    extraction_eval = state.get("extraction_eval") or {}
    if (
        human_expert_review is None
        and state.get("status") in {"success", "waiting_for_human_advice"}
        and extraction_eval.get("status") == "ready_for_live_test"
    ):
        if state.get("status") == "waiting_for_human_advice":
            human_expert_review = {
                "status": "waiting_for_human_advice",
                "reason": "Internal Step9 evaluation passed, but human expert review has not been supplied.",
                "source": "write_output_audit_fallback",
            }
        elif state.get("human_advice"):
            human_expert_review = {
                "status": "accepted",
                "advice": state.get("human_advice", []),
                "reason": "Internal Step9 evaluation passed and human expert review advice was supplied.",
                "source": "write_output_audit_fallback",
            }
        else:
            human_expert_review = {
                "status": "skipped",
                "reason": "Human expert review was explicitly skipped or not required for this run.",
                "source": "write_output_audit_fallback",
            }
    output = {
        "step": "step9_extraction_build_agent_system",
        "status": state.get("status", "needs_human_review"),
        "inputs": {
            "step8_output": args.step8_output,
            "prompt_output": args.prompt_output,
            "test_documents": args.test_documents,
            "dry_run": args.dry_run,
        },
        "workflow_plan": state.get("workflow_plan"),
        "prompt_output": state.get("prompt_output"),
        "judgement": state.get("judgement"),
        "code_agent": state.get("code_agent"),
        "extraction_test": state.get("extraction_test"),
        "extraction_eval": extraction_eval,
        "schema_feedback": state.get("schema_feedback"),
        "supervisor_decision": state.get("supervisor_decision"),
        "human_expert_review": human_expert_review,
        "human_advice": state.get("human_advice", []),
        "repair_history": state.get("repair_history", []),
        "validation_errors": state.get("validation_errors", []),
    }
    output_path = Path(args.output)
    output_path.write_text(json_dumps(output), encoding="utf-8")
    print(f"Saved Step9 result to {output_path}")
    print(f"Status: {output['status']}")
    return write_state_snapshot(merge_update(state, {"output": str(output_path)}), "write_output", "__end__")


def next_node(state):
    return (state.get("supervisor_decision") or {}).get("next_node", state.get("next_node") or "write_output")


def build_step9_graph():
    graph = StateGraph(Step9State)
    graph.add_node("load_inputs", load_inputs_node)
    graph.add_node("workflow_plan", workflow_plan_node)
    graph.add_node("prompt_generate", prompt_generate_node)
    graph.add_node("prompt_quality_review", prompt_quality_review_node)
    graph.add_node("prompt_repair", prompt_repair_node)
    graph.add_node("code_generate_or_patch", code_generate_or_patch_node)
    graph.add_node("code_repair", code_repair_node)
    graph.add_node("extraction_test_run", extraction_test_run_node)
    graph.add_node("extraction_eval", extraction_eval_node)
    graph.add_node("human_expert_review", human_expert_review_node)
    graph.add_node("workflow_repair", workflow_repair_node)
    graph.add_node("schema_feedback", schema_feedback_node)
    graph.add_node("supervisor_router", supervisor_router_node)
    graph.add_node("write_output", write_output_node)

    graph.add_edge(START, "load_inputs")
    graph.add_conditional_edges("load_inputs", lambda state: state.get("next_node", "workflow_plan"))
    graph.add_edge("workflow_plan", "prompt_generate")
    graph.add_edge("prompt_generate", "prompt_quality_review")
    graph.add_conditional_edges("prompt_quality_review", lambda state: state.get("next_node", "code_generate_or_patch"))
    graph.add_edge("prompt_repair", "prompt_quality_review")
    graph.add_edge("code_generate_or_patch", "extraction_test_run")
    graph.add_edge("code_repair", "write_output")
    graph.add_edge("extraction_test_run", "extraction_eval")
    graph.add_conditional_edges("extraction_eval", lambda state: state.get("next_node", "write_output"))
    graph.add_conditional_edges("human_expert_review", lambda state: state.get("next_node", "write_output"))
    graph.add_edge("workflow_repair", "prompt_repair")
    graph.add_edge("schema_feedback", "write_output")
    graph.add_conditional_edges("supervisor_router", next_node)
    graph.add_edge("write_output", END)
    return graph.compile(checkpointer=MemorySaver())


def build_parser():
    parser = argparse.ArgumentParser(description="Run Step9 extraction build as a LangGraph sub-agent system.")
    parser.add_argument("--step8-output", default="")
    parser.add_argument("--prompt-output", default="")
    parser.add_argument("--output", default="step9_extraction_build_output.json")
    parser.add_argument(
        "--reference-pipeline",
        default=DEFAULT_REFERENCE_PIPELINE,
        help="Optional existing extraction pipeline used as a workflow/prompt/code strategy reference.",
    )
    parser.add_argument("--test-documents", nargs="*", default=[])
    parser.add_argument("--extraction-results", default="", help="Optional JSON file with live section-wise extraction test results.")
    parser.add_argument("--dry-run", action="store_true", help="Plan section-wise extraction tests without executing a live runner.")
    parser.add_argument("--skip-code-generation", action="store_true", help="Skip LLM code generation and only build/evaluate the workflow plan.")
    parser.add_argument("--base-url", default=prompt_agent.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=prompt_agent.DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--extraction-max-tokens", type=int, default=4096)
    parser.add_argument("--extraction-run-dir", default="step9_extraction_runs")
    parser.add_argument("--max-test-documents", type=int, default=4)
    parser.add_argument("--max-document-chars", type=int, default=26000)
    parser.add_argument("--max-fields-per-stage", type=int, default=80)
    parser.add_argument("--human-advice", nargs="*", default=[], help="Optional human suggestions used by the Step9 supervisor/repair agents.")
    parser.add_argument("--human-advice-file", default="", help="Optional text or JSON file containing human suggestions.")
    parser.add_argument("--max-human-advice-rounds", type=int, default=2)
    parser.add_argument("--target-files", nargs="*", default=[])
    parser.add_argument("--write-generated-files", action="store_true")
    parser.add_argument("--generated-output-dir", default="generated_code")
    parser.add_argument("--max-supervisor-retries", type=int, default=1)
    parser.add_argument(
        "--skip-human-expert-review",
        action="store_true",
        help="Explicitly skip the post-internal-pass human expert review gate.",
    )
    parser.add_argument("--thread-id", default="step9-extraction-build")
    return parser


def main():
    args = build_parser().parse_args()
    app = build_step9_graph()
    initial_state = {"args": vars(args), "next_node": "load_inputs"}
    result = app.invoke(initial_state, config={"configurable": {"thread_id": args.thread_id}})
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "output": result.get("output"),
                "current_node": result.get("current_node"),
                "next_node": result.get("next_node"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
