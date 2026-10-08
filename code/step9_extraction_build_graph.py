import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import prompt_quality_code_agent as prompt_agent
import downstream_extraction_runner
import materials_agent_protocol as agent_protocol
import run_artifact_guard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_PIPELINE = str(PROJECT_ROOT / "reference_pipelines" / "pure_extraction_pipeline")
MATERIAL_LITERATURE_TASK_CONTRACT = agent_protocol.materials_literature_task_contract()


class Step9State(TypedDict, total=False):
    args: dict[str, Any]
    step8_output: dict[str, Any]
    prompt_output: dict[str, Any]
    workflow_plan: dict[str, Any]
    judgement: dict[str, Any]
    quality_review: dict[str, Any]
    code_agent: dict[str, Any]
    code_generation_attempt_count: int
    extraction_test: dict[str, Any]
    extraction_attempt_count: int
    extraction_eval: dict[str, Any]
    supervisor_decision: dict[str, Any]
    validation_errors: list[str]
    human_advice: list[dict[str, Any]]
    applied_human_advice: str
    human_review_applied: bool
    retry_counts: dict[str, int]
    repair_history: list[dict[str, Any]]
    failed_node: str
    schema_feedback: dict[str, Any]
    status: str
    current_node: str
    next_node: str
    output: str
    human_advice_available_before_design: bool
    paper_metadata_by_document: dict[str, dict[str, Any]]
    paper_metadata_report: dict[str, Any]
    human_expert_review: dict[str, Any]
    feedback_classification: dict[str, Any]
    impact_manifest: dict[str, Any]
    impact_supervisor_decision: dict[str, Any]
    impact_manifest_input: dict[str, Any]
    protocol_messages: list[dict[str, Any]]
    run_identity: dict[str, Any]


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

DYNAMIC_SECTION_RANK = {
    "sample_info": 10,
    "process_info": 20,
    "measurement_info": 30,
    "property_observation_info": 40,
    "evidence_info": 50,
    "claim_info": 60,
    "interpretation_info": 70,
}


PAPER_METADATA_FIELD_SPECS = [
    ("paper_info.metadata.title", "string", True),
    ("paper_info.metadata.authors", "array of strings", False),
    ("paper_info.metadata.doi", "string", False),
    ("paper_info.metadata.arxiv_id", "string", False),
    ("paper_info.metadata.abstract", "string", False),
    ("paper_info.metadata.publication_date", "string", False),
    ("paper_info.metadata.journal", "string", False),
    ("paper_info.metadata.url", "string", False),
    ("paper_info.metadata.pdf_url", "string", False),
    ("paper_info.metadata.keywords", "array of strings", False),
    ("paper_info.metadata.source", "string", False),
    ("paper_info.metadata.retrieved_at", "string", False),
]

PAPER_METADATA_FIELD_ALIASES = {
    "publication_date": {"publication_date", "published_date", "date"},
    "url": {"url", "paper_url", "landing_url"},
    "keywords": {"keywords", "subjects"},
}

PAPER_METADATA_DESCRIPTIONS = {
    "title": "Published or preprint title of this source document, not a material name.",
    "authors": "Ordered author names credited on this source document.",
    "doi": "Digital Object Identifier of this source document, excluding cited works.",
    "arxiv_id": "arXiv identifier and version of this source preprint.",
    "abstract": "Author-provided abstract of this document, not a generated summary.",
    "publication_date": "Reported publication or preprint release date of this document.",
    "journal": "Journal or publication venue associated with this document.",
    "url": "Canonical landing-page URL for this source document.",
    "pdf_url": "Download URL of the source document PDF, not a cited attachment.",
    "keywords": "Source-provided keywords or subject classifications for this document.",
    "source": "Retrieval provider or repository from which this document was obtained.",
    "retrieved_at": "Timestamp of document retrieval, distinct from its publication date.",
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


def literature_task_contract(step8_output):
    result = step8_output.get("result") if isinstance(step8_output, dict) else {}
    candidate = result.get("task_contract") if isinstance(result, dict) else {}
    contract = dict(MATERIAL_LITERATURE_TASK_CONTRACT)
    if isinstance(candidate, dict):
        for key in ("record_scope", "lineage_contract_version"):
            if candidate.get(key):
                contract[key] = candidate[key]
    return contract


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
    snapshot_for_file = deepcopy(snapshot)
    if isinstance(snapshot_for_file.get("args"), dict) and snapshot_for_file["args"].get("api_key"):
        snapshot_for_file["args"]["api_key"] = "[REDACTED]"
    run_artifact_guard.atomic_write_json(
        state_path_from_args(args),
        snapshot_for_file,
        run_identity=state.get("run_identity"),
    )
    return snapshot


def append_protocol_message(
    state,
    update,
    *,
    sender,
    receiver="step9_supervisor",
    phase,
    status,
    payload_refs=None,
    decision=None,
    requested_actions=None,
    produced_artifacts=None,
    next_route="",
    evidence=None,
):
    messages = deepcopy(state.get("protocol_messages") or [])
    messages.append(
        agent_protocol.make_message(
            sender=sender,
            receiver=receiver,
            phase=phase,
            status=status,
            task_contract=literature_task_contract(state.get("step8_output") or {}),
            payload_refs=payload_refs,
            decision=decision,
            requested_actions=requested_actions,
            produced_artifacts=produced_artifacts,
            next_route=next_route,
            evidence=evidence,
        )
    )
    update["protocol_messages"] = messages
    return update


def load_json_if_exists(path):
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8-sig"))


def _metadata_records(payload):
    if isinstance(payload, list):
        records = []
        for item in payload:
            records.extend(_metadata_records(item))
        return records
    if not isinstance(payload, dict):
        return []
    identity_keys = {"paper_id", "arxiv_id", "source_id", "title", "doi"}
    if identity_keys.intersection(payload):
        return [payload]
    records = []
    for key in ("papers", "records", "items", "results"):
        value = payload.get(key)
        if isinstance(value, dict):
            for item in value.values():
                records.extend(_metadata_records(item))
        elif isinstance(value, list):
            records.extend(_metadata_records(value))
    return records


def _identity_token(value):
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _record_identity_tokens(record):
    external_ids = record.get("external_ids") if isinstance(record.get("external_ids"), dict) else {}
    raw_values = [
        record.get("paper_id"),
        record.get("arxiv_id"),
        record.get("source_id"),
        external_ids.get("arxiv"),
        Path(str(record.get("file_path") or "")).stem,
    ]
    source = str(record.get("source") or "").strip()
    source_id = str(record.get("source_id") or "").strip()
    if source and source_id:
        raw_values.append(f"{source}:{source_id}")
    return {token for token in (_identity_token(value) for value in raw_values) if token}


def _document_identity_tokens(document):
    stem = Path(document).stem
    tokens = {_identity_token(stem)}
    if "__" in stem:
        tokens.add(_identity_token(stem.replace("__", ":", 1)))
    return {token for token in tokens if token}


def discover_paper_metadata_paths(documents, configured_paths=None):
    paths = []
    seen = set()

    def add(candidate):
        candidate = Path(candidate)
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if candidate.exists() and key not in seen:
            seen.add(key)
            paths.append(candidate)

    for configured in configured_paths or []:
        add(configured)
    for document in documents or []:
        document_path = Path(document)
        ancestors = [document_path.parent, *list(document_path.parents)[:4]]
        for ancestor in ancestors:
            add(ancestor / "paper_archive" / "metadata")
            if ancestor.name.casefold() in {"paper_archive", "archive"}:
                add(ancestor / "metadata")
    return paths


def load_paper_metadata_for_documents(documents, configured_paths=None):
    sources = discover_paper_metadata_paths(documents, configured_paths)
    records = []
    invalid_files = []
    for source in sources:
        files = [source] if source.is_file() else sorted(source.rglob("*.json"))
        for metadata_file in files:
            try:
                payload = json.loads(metadata_file.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as exc:
                invalid_files.append({"path": str(metadata_file), "error": str(exc)})
                continue
            for record in _metadata_records(payload):
                enriched = deepcopy(record)
                enriched["_metadata_source_path"] = str(metadata_file)
                records.append(enriched)

    matches = {}
    unmatched_documents = []
    for document in documents or []:
        document_tokens = _document_identity_tokens(document)
        ranked = []
        for record in records:
            overlap = document_tokens.intersection(_record_identity_tokens(record))
            if overlap:
                ranked.append((len(overlap), record))
        if ranked:
            ranked.sort(key=lambda item: item[0], reverse=True)
            matches[str(Path(document))] = ranked[0][1]
        else:
            unmatched_documents.append(str(Path(document)))
    report = {
        "configured_paths": [str(path) for path in configured_paths or []],
        "discovered_paths": [str(path) for path in sources],
        "records_loaded": len(records),
        "matched_documents": len(matches),
        "unmatched_documents": unmatched_documents,
        "invalid_files": invalid_files,
    }
    return matches, report


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


def inject_resume_human_advice(initial_state, human_advice):
    advice_items = deepcopy(list(human_advice or []))
    if not advice_items:
        return initial_state

    initial_state["human_advice"] = advice_items
    human_review = initial_state.get("human_expert_review") or {}
    resume_expert_review = (
        initial_state.get("status") in {"waiting_for_human_advice", "needs_human_review"}
        and human_review.get("status") in {"waiting_for_human_advice", "accepted"}
    )
    if resume_expert_review:
        initial_state["status"] = "running"
        initial_state["next_node"] = "human_expert_review"
    return initial_state


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


def _feedback_values(human_advice, key):
    values = []
    for item in human_advice or []:
        if not isinstance(item, dict):
            continue
        raw = item.get(key)
        if raw is None:
            continue
        candidates = raw if isinstance(raw, list) else [raw]
        values.extend(str(value).strip() for value in candidates if str(value).strip())
    return list(dict.fromkeys(values))


def build_feedback_classification_and_impact(state):
    """Classify expert/system feedback and produce a bounded literature reprocessing plan."""
    human_advice = state.get("human_advice", []) or []
    advice_route = infer_human_advice_route(human_advice)
    explicit_types = [value.lower() for value in _feedback_values(human_advice, "feedback_type")]
    advice = advice_text(human_advice).lower()
    schema_feedback = state.get("schema_feedback") or {}
    human_review = state.get("human_expert_review") or {}
    diagnoses = (state.get("extraction_eval") or {}).get("failure_diagnoses", []) or []

    if human_review.get("status") in {"accepted", "skipped"} and not schema_feedback and not diagnoses:
        feedback_type = "none"
        target_components = []
        reason = "The expert gate accepted the internally passing result, so explanatory review text is not corrective feedback."
    elif any(value in {"record_exception", "instance_error", "current_record"} for value in explicit_types) or any(
        token in advice for token in ("current record only", "instance only", "只改当前", "当前记录", "个例")
    ):
        feedback_type = "record_exception"
        target_components = ["record"]
        reason = "Expert feedback identifies a literature-record-specific exception."
    elif any(value in {"systematic_schema_or_rule", "schema_error", "rule_error"} for value in explicit_types) or schema_feedback or advice_route:
        feedback_type = "systematic_schema_or_rule"
        route_component = {
            "schema_feedback_to_step8": "schema",
            "prompt_repair": "prompt",
            "workflow_repair": "workflow",
        }.get(advice_route, "schema")
        target_components = [route_component]
        reason = "Feedback can affect the schema, extraction prompt/rule, validator, or workflow used by multiple literature records."
    elif any(value in {"unresolved", "uncertain"} for value in explicit_types) or human_review.get("status") in {
        "waiting_for_human_advice",
        "needs_revision",
    }:
        feedback_type = "unresolved"
        target_components = ["human_review"]
        reason = "The available literature evidence or review decision is not sufficient for a reliable automatic repair."
    else:
        feedback_type = "none"
        target_components = []
        reason = "No corrective expert or system feedback requires historical reprocessing."

    affected_documents = _feedback_values(human_advice, "documents")
    affected_stage_ids = _feedback_values(human_advice, "stage_ids")
    affected_field_paths = _feedback_values(human_advice, "field_paths")
    for diagnosis in diagnoses:
        if not isinstance(diagnosis, dict):
            continue
        stage_id = str(diagnosis.get("stage_id") or "").strip()
        if stage_id:
            affected_stage_ids.append(stage_id)
        for document in diagnosis.get("documents") or []:
            if str(document).strip():
                affected_documents.append(str(document).strip())

    stage_to_section = {
        str(item.get("stage_id") or ""): str(item.get("section_id") or "")
        for item in (state.get("workflow_plan") or {}).get("section_test_plan", []) or []
        if isinstance(item, dict) and item.get("stage_id")
    }
    affected_sections = {stage_to_section.get(stage_id, stage_id) for stage_id in affected_stage_ids}
    field_index = ((state.get("prompt_output") or {}).get("shared_prompt_context") or {}).get("field_index", [])
    for field in field_index or []:
        if not isinstance(field, dict):
            continue
        if affected_sections and field.get("section_id") not in affected_sections:
            continue
        if affected_sections and field.get("field_path"):
            affected_field_paths.append(str(field["field_path"]))

    affected_documents = sorted(set(affected_documents))
    affected_stage_ids = sorted(set(affected_stage_ids))
    affected_field_paths = sorted(set(affected_field_paths))
    has_bounded_scope = bool(affected_documents or affected_stage_ids or affected_field_paths)
    if feedback_type == "none":
        impact_status = "no_reprocessing_required"
    elif feedback_type == "record_exception":
        impact_status = "current_record_only" if affected_documents else "needs_record_scope_confirmation"
    elif feedback_type == "unresolved":
        impact_status = "pending_expert_scope_decision"
    else:
        impact_status = "targeted_reprocessing_required" if has_bounded_scope else "needs_impact_scope_confirmation"

    feedback_classification = {
        "feedback_type": feedback_type,
        "target_components": target_components,
        "reason": reason,
        "source": "human_and_step9_supervisor",
        "preserve_original_artifacts": True,
    }
    impact_manifest = {
        "status": impact_status,
        "task_type": "automated_materials_database_construction",
        "source_scope": "scientific_literature_only",
        "affected_documents": affected_documents,
        "affected_stage_ids": affected_stage_ids,
        "affected_field_paths": affected_field_paths,
        "selection_keys": [
            "literature_document_id",
            "field_rule_id_and_version",
            "prompt_version",
            "validator_version",
            "applicability_conditions",
        ],
        "reprocess_policy": "rerun_only_affected_documents_stages_and_field_paths",
        "preservation_policy": "retain_original_result_expert_edit_reason_impact_scope_and_reprocessed_result",
    }
    return feedback_classification, impact_manifest


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
    fields = [deepcopy(field) for field in schema.get("field_registry", []) or [] if isinstance(field, dict)]
    existing_metadata_names = set()
    for field in fields:
        path = str(field.get("field_path") or "").casefold()
        if not path.startswith("paper_info."):
            continue
        leaf = path.rsplit(".", 1)[-1]
        existing_metadata_names.add(leaf)
    for field_path, data_type, required in PAPER_METADATA_FIELD_SPECS:
        leaf = field_path.rsplit(".", 1)[-1]
        aliases = PAPER_METADATA_FIELD_ALIASES.get(leaf, {leaf})
        if existing_metadata_names.intersection(aliases):
            continue
        fields.append(
            {
                "field_path": field_path,
                "section_id": "paper_info",
                "field_name": leaf,
                "data_type": data_type,
                "required": required,
                "source_basis": ["upstream_metadata"],
                "concept_ids": [f"paper_metadata_{leaf}"],
                "description": PAPER_METADATA_DESCRIPTIONS[leaf],
                "extraction_notes": "Use authoritative retrieval/download metadata for this exact document. Preserve supplied values; leave unavailable metadata missing, and never use a cited paper or model-generated value as a substitute.",
                "reason": "Canonical bibliographic field populated from retrieval/download metadata.",
            }
        )
    for field in fields:
        field_path = str(field.get("field_path") or "")
        is_metadata = field_path.startswith("paper_info.")
        field.setdefault("field_rule_id", "field." + safe_contract_id(field_path))
        field.setdefault("core_field", bool(field.get("required")))
        field.setdefault("core_field_source", "step8_system_design")
        field.setdefault(
            "inclusion_rule",
            (
                "Populate from authoritative download-time bibliographic metadata."
                if is_metadata
                else "Populate only from direct scientific-literature evidence matching the exact field semantics."
            ),
        )
        field.setdefault(
            "absence_rule",
            "Use missing when absent and unresolved when relevant evidence is ambiguous; never invent a value.",
        )
        field.setdefault(
            "evidence_requirements",
            {
                "direct_support_required": not is_metadata,
                "locator_required": True,
                "allowed_source_types": list(field.get("source_basis") or ["text"]),
                "metadata_handoff_allowed": is_metadata,
            },
        )
        field.setdefault(
            "relation_constraints",
            {
                "entity_binding_required": not is_metadata,
                "condition_binding_required": False,
                "separate_instances": False,
            },
        )
        if not str(field.get("field_rule_version") or "").startswith("sha256:"):
            version_payload = {
                key: field.get(key)
                for key in (
                    "field_path",
                    "section_id",
                    "description",
                    "extraction_notes",
                    "data_type",
                    "required",
                    "source_basis",
                    "inclusion_rule",
                    "absence_rule",
                    "evidence_requirements",
                    "relation_constraints",
                )
            }
            digest = hashlib.sha256(
                json.dumps(version_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest().upper()
            field["field_rule_version"] = f"sha256:{digest}"
    return fields


def safe_contract_id(value):
    token = "".join(character if character.isalnum() else "_" for character in str(value or "").lower())
    return "_".join(part for part in token.split("_") if part) or "unnamed"


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


def dynamic_section_sort_key(section_id):
    section_id = str(section_id or "")
    root = section_id.split(".", 1)[0]
    return (DYNAMIC_SECTION_RANK.get(root, 100), section_id)


def workflow_section_order(groups):
    dynamic_sections = sorted(
        (
            section_id
            for section_id in groups
            if section_id not in SECTION_ORDER and section_id != "figure_classification"
        ),
        key=dynamic_section_sort_key,
    )
    ordered = []
    for section_id in SECTION_ORDER:
        ordered.append(section_id)
        if section_id == "figure_classification":
            ordered.extend(dynamic_sections)
    return ordered


def default_section_dependencies(section_id, groups, figure_classification_enabled=False):
    dependencies = []
    if section_id != "paper_info" and "paper_info" in groups:
        dependencies.append("paper_info")
    if (
        section_id not in {"paper_info", "material_info.section0"}
        and "material_info.section0" in groups
    ):
        dependencies.append("material_info.section0")

    root = str(section_id or "").split(".", 1)[0]
    if root in {"process_info", "measurement_info"} and "sample_info" in groups:
        dependencies.append("sample_info")
    if root == "property_observation_info" and "measurement_info" in groups:
        dependencies.append("measurement_info")

    section_fields = groups.get(section_id, []) or []
    if (
        figure_classification_enabled
        and section_id not in {"material_info.section3", "figure_classification"}
        and any(is_figure_field(field) for field in section_fields)
    ):
        dependencies.append("figure_classification")
    return list(dict.fromkeys(dependencies))


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
    has_figure_classification = bool(figure_fields)

    section_test_plan = []
    for section_id in workflow_section_order(groups):
        if section_id == "figure_classification":
            if has_figure_classification:
                classification_dependencies = []
                if groups.get("material_info.section3"):
                    classification_dependencies.append("material_info.section3")
                elif groups.get("material_info.section0"):
                    classification_dependencies.append("material_info.section0")
                elif groups.get("paper_info"):
                    classification_dependencies.append("paper_info")
                section_test_plan.append(
                    {
                        "stage_id": "figure_classification",
                        "section_id": "figure_classification",
                        "purpose": "Classify figures before section4/section5 extraction to reduce curve/mechanism ownership errors.",
                        "depends_on": classification_dependencies,
                        "test_focus": ["figure ownership", "allowed sections", "section4 versus section5 boundary"],
                    }
                )
            continue
        if section_id not in groups and section_id != "paper_info":
            continue
        depends_on = default_section_dependencies(
            section_id,
            groups,
            figure_classification_enabled=has_figure_classification,
        )
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
        "objective": "Construct queryable materials-database records from parsed scientific literature, run section-wise extraction tests, and repair prompt/code/workflow based on measured extraction quality.",
        "task_contract": literature_task_contract(step8_output),
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


def stage_prompt_definition(stage, field_specs):
    stage_id = stage["stage_id"]
    section_id = stage["section_id"]
    return {
        "prompt": (
            f"Run stage {stage_id} for {section_id} according to the Step8 schema. "
            "The fixed task is automated materials-database construction from parsed scientific literature; do not reinterpret it as generic scientific-text conversion. "
            "Return valid JSON only. Preserve evidence provenance with source_text, source_table, "
            "source_figure, confidence, and missing_reason when values are absent. "
            "Use output_contract.field_specs as the authoritative field semantics and data contract. "
            "Only emit field_path values listed there, and account for every allowed field path exactly once. "
            f"Respect dependencies: {', '.join(stage.get('depends_on', [])) or 'none'}. "
            "If this stage repeatedly fails, consult workflow_plan.reference_strategy_library for known extraction patterns."
        ),
        "output_contract": {
            "section_id": section_id,
            "json_only": True,
            "fields": [item["field_path"] for item in field_specs],
            "field_specs": deepcopy(field_specs),
        },
    }


def synchronize_prompt_field_contracts(prompt_output, workflow_plan):
    """Reconcile every schema field with an executable stage contract."""
    prompt_output = deepcopy(prompt_output or {})
    workflow_plan = deepcopy(workflow_plan or {})
    context = prompt_output.setdefault("shared_prompt_context", {})
    field_index = [
        field
        for field in context.get("field_index", []) or []
        if isinstance(field, dict) and field.get("field_path")
    ]
    groups = group_fields_by_section(field_index)
    stages = workflow_plan.setdefault("section_test_plan", [])
    existing_sections = {
        str(stage.get("section_id") or "")
        for stage in stages
        if isinstance(stage, dict)
    }
    figure_classification_enabled = any(
        stage.get("stage_id") == "figure_classification"
        for stage in stages
        if isinstance(stage, dict)
    )
    added_stages = []
    for section_id in sorted(groups, key=dynamic_section_sort_key):
        if section_id in existing_sections:
            continue
        stage = {
            "stage_id": section_id,
            "section_id": section_id,
            "purpose": f"Extract and validate {section_id} fields.",
            "depends_on": default_section_dependencies(
                section_id,
                groups,
                figure_classification_enabled=figure_classification_enabled,
            ),
            "field_count": len(groups.get(section_id, [])),
            "test_focus": ["json validity", "field coverage", "evidence provenance", "missing_reason"],
            "repair_notes": ["Added by deterministic field-contract synchronization."],
        }
        stages.append(stage)
        existing_sections.add(section_id)
        added_stages.append(section_id)

    modules = prompt_output.setdefault("module_outputs", {})
    section_module = modules.setdefault("section_extraction_prompt_module", {})
    prompts = section_module.setdefault("section_extraction_prompts", {})
    synchronized_stages = []
    for stage in stages:
        if not isinstance(stage, dict) or not stage.get("stage_id") or not stage.get("section_id"):
            continue
        stage_id = stage["stage_id"]
        section_id = stage["section_id"]
        field_specs = groups.get(section_id, [])
        existing_prompt = prompts.get(stage_id)
        if not isinstance(existing_prompt, dict):
            existing_prompt = stage_prompt_definition(stage, field_specs)
            prompts[stage_id] = existing_prompt
        output_contract = existing_prompt.setdefault("output_contract", {})
        output_contract["section_id"] = section_id
        output_contract["json_only"] = True
        output_contract["fields"] = [item["field_path"] for item in field_specs]
        output_contract["field_specs"] = deepcopy(field_specs)
        synchronized_stages.append(stage_id)

    section_module["field_coverage_index"] = deepcopy(field_index)
    workflow_plan["execution_order"] = [
        stage["stage_id"]
        for stage in stages
        if isinstance(stage, dict) and stage.get("stage_id")
    ]
    context["workflow_plan"] = workflow_plan
    result = prompt_output.setdefault("result", {})
    result["workflow_execution_order"] = list(workflow_plan["execution_order"])
    result.setdefault("prompt_modules", {})["section_extraction"] = prompts
    return prompt_output, workflow_plan, {
        "field_count": len(field_index),
        "added_stages": added_stages,
        "synchronized_stages": synchronized_stages,
    }


def build_prompt_package_from_step8(step8_output, workflow_plan):
    field_registry = field_registry_from_step8(step8_output)
    schema_definition = (
        ((step8_output.get("result") or {}).get("schema_definition") or {})
        if isinstance(step8_output, dict)
        else {}
    )
    shared_record_contracts = schema_definition.get("shared_record_contracts") or {}
    figure_fields = [field for field in field_registry if is_figure_field(field)]
    field_index = [
        {
            "field_path": field.get("field_path"),
            "section_id": normalize_section_id(field.get("section_id"), field.get("field_path")),
            "data_type": field.get("data_type"),
            "description": field.get("description") or field.get("reason") or "",
            "extraction_notes": field.get("extraction_notes") or "",
            "required": bool(field.get("required")),
            "core_field": bool(field.get("core_field")),
            "core_field_source": field.get("core_field_source", "step8_system_design"),
            "source_basis": field.get("source_basis", []),
            "inclusion_rule": field.get("inclusion_rule", ""),
            "absence_rule": field.get("absence_rule", ""),
            "field_rule_id": field.get("field_rule_id", ""),
            "field_rule_version": field.get("field_rule_version", ""),
            "evidence_requirements": field.get("evidence_requirements", {}),
            "relation_constraints": field.get("relation_constraints", {}),
            "concept_ids": field.get("concept_ids", []),
            "object_contract": field.get("object_contract"),
            "contract_refs": field.get("contract_refs", []),
            "shared_contracts": {
                ref: shared_record_contracts[ref]
                for ref in field.get("contract_refs", []) or []
                if ref in shared_record_contracts
            },
            "figure_constraint": field.get("figure_constraint"),
            "reason": field.get("reason", ""),
        }
        for field in field_registry
    ]
    prompts = {}
    for stage in workflow_plan.get("section_test_plan", []):
        section_id = stage["section_id"]
        stage_field_specs = [
            item for item in field_index if item["section_id"] == section_id
        ]
        prompts[stage["stage_id"]] = stage_prompt_definition(stage, stage_field_specs)

    return {
        "step": "step9_prompt_generation",
        "status": "success",
        "validation_errors": [],
        "shared_prompt_context": {
            "task_contract": literature_task_contract(step8_output),
            "field_index": field_index,
            "shared_record_contracts": shared_record_contracts,
            "figure_fields": figure_fields,
            "workflow_plan": workflow_plan,
            "lineage_contract": (
                ((step8_output.get("result") or {}).get("lineage_contract") or {})
                if isinstance(step8_output, dict)
                else {}
            ),
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
    impact_manifest_input = load_json_if_exists(getattr(args, "impact_manifest_input", ""))
    if impact_manifest_input.get("impact_manifest") and not impact_manifest_input.get("status"):
        impact_manifest_input = impact_manifest_input["impact_manifest"]
    human_advice = load_human_advice(args)
    paper_metadata_by_document, paper_metadata_report = load_paper_metadata_for_documents(
        args.test_documents,
        getattr(args, "paper_metadata", []) or [],
    )
    errors = []
    if not step8_output and not prompt_output:
        errors.append("Either --step8-output or --prompt-output must point to an existing JSON file.")
    if impact_manifest_input:
        if impact_manifest_input.get("task_type") != MATERIAL_LITERATURE_TASK_CONTRACT["task_type"]:
            errors.append("Impact manifest task_type is outside automated materials-database construction.")
        if impact_manifest_input.get("source_scope") != MATERIAL_LITERATURE_TASK_CONTRACT["source_scope"]:
            errors.append("Impact manifest source_scope must be scientific_literature_only.")
    update = {
        "step8_output": step8_output,
        "prompt_output": prompt_output,
        "human_advice": human_advice,
        "human_advice_available_before_design": bool(human_advice),
        "paper_metadata_by_document": paper_metadata_by_document,
        "paper_metadata_report": paper_metadata_report,
        "impact_manifest_input": impact_manifest_input,
        "validation_errors": errors,
        "retry_counts": {},
        "repair_history": [],
        "status": "running" if not errors else "needs_human_review",
    }
    update = append_protocol_message(
        state,
        update,
        sender="step9_input_loader",
        phase="load_inputs",
        status="rejected" if errors else "completed",
        payload_refs={
            "step8_output": "step8_output",
            "prompt_output": "prompt_output",
            "impact_manifest_input": "impact_manifest_input",
        },
        decision={"accepted_by_supervisor": not errors},
        requested_actions=errors,
        produced_artifacts=[{"ref": "paper_metadata_report"}],
        next_route="supervisor_router" if errors else "workflow_plan",
        evidence=errors,
    )
    return write_state_snapshot(merge_update(state, update), "load_inputs", "supervisor_router" if errors else "workflow_plan")


def workflow_plan_node(state):
    args = dict_to_namespace(state["args"])
    step8_output = state.get("step8_output") or {}
    prompt_output = state.get("prompt_output") or {}
    if not step8_output and prompt_output:
        step8_output = {"result": {"schema_definition": {"field_registry": (prompt_output.get("shared_prompt_context") or {}).get("field_index", [])}}}
    reference_library = build_reference_strategy_library(args.reference_pipeline)
    workflow_plan = build_workflow_plan(step8_output, prompt_output, reference_library)
    update = append_protocol_message(
        state,
        {"workflow_plan": workflow_plan, "status": "running"},
        sender="workflow_planning_agent",
        phase="workflow_plan",
        status="completed",
        payload_refs={"step8_schema": "step8_output.result.schema_definition"},
        decision={"accepted_by_supervisor": True, "stage_count": len(workflow_plan.get("section_test_plan", []))},
        produced_artifacts=[{"ref": "workflow_plan"}],
        next_route="prompt_generate",
    )
    return write_state_snapshot(
        merge_update(state, update),
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
    update = append_protocol_message(
        state,
        {"prompt_output": prompt_output},
        sender="prompt_generation_agent",
        phase="prompt_generate",
        status="completed",
        payload_refs={"workflow_plan": "workflow_plan", "field_index": "prompt_output.shared_prompt_context.field_index"},
        decision={"accepted_by_supervisor": True},
        produced_artifacts=[{"ref": "prompt_output"}],
        next_route="prompt_quality_review",
    )
    return write_state_snapshot(
        merge_update(state, update),
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
    next_route = "supervisor_router" if update.get("validation_errors") else "code_generate_or_patch"
    update = append_protocol_message(
        state,
        update,
        sender="prompt_quality_reviewer",
        phase="prompt_quality_review",
        status="rejected" if update.get("validation_errors") else "accepted",
        payload_refs={"prompt_package": "prompt_output", "judgement": "judgement"},
        decision={"accepted_by_supervisor": not bool(update.get("validation_errors"))},
        requested_actions=judgement.get("issues", []) if isinstance(judgement, dict) else [],
        produced_artifacts=[{"ref": "judgement"}],
        next_route=next_route,
        evidence=update.get("validation_errors", []),
    )
    return write_state_snapshot(merge_update(state, update), "prompt_quality_review", next_route)


def prompt_repair_node(state):
    prompt_output = deepcopy(state.get("prompt_output") or {})
    judgement = state.get("judgement") or {}
    workflow_plan = state.get("workflow_plan") or {}
    eval_result = state.get("extraction_eval") or {}
    human_advice_text = advice_text(state.get("human_advice", [])) if state.get("human_review_applied") else ""
    prompt_output, workflow_plan, field_contract_sync = synchronize_prompt_field_contracts(
        prompt_output,
        workflow_plan,
    )
    repair_note = {
        "source": "prompt_repair_agent",
        "reason": "Supervisor routed prompt package through deterministic repair.",
        "judgement_status": judgement.get("status"),
        "issue_count": (judgement.get("summary") or {}).get("issue_count", 0),
        "warning_count": (judgement.get("summary") or {}).get("warning_count", 0),
        "evaluation_targets": eval_result.get("optimization_targets", []),
        "human_advice_used": bool(human_advice_text),
        "field_contract_sync": field_contract_sync,
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
        "workflow_plan": workflow_plan,
        "validation_errors": [],
        "failed_node": "",
        "status": "running",
        "repair_history": append_repair_history(state, repair_note),
    }
    update = append_protocol_message(
        state,
        update,
        sender="prompt_repair_agent",
        phase="prompt_repair",
        status="completed",
        payload_refs={"previous_prompt": "prompt_output", "evaluation": "extraction_eval"},
        decision={"accepted_by_supervisor": True, "action": "revalidate_prompt"},
        produced_artifacts=[{"ref": "prompt_output"}],
        next_route="prompt_quality_review",
    )
    return write_state_snapshot(merge_update(state, update), "prompt_repair", "prompt_quality_review")


def next_fresh_attempt_dir(root, completed_count):
    attempt_count = int(completed_count or 0) + 1
    root = Path(root)
    attempt_dir = root / f"attempt_{attempt_count:03d}"
    while attempt_dir.exists():
        attempt_count += 1
        attempt_dir = root / f"attempt_{attempt_count:03d}"
    return attempt_count, attempt_dir


def code_generate_or_patch_node(state):
    args = dict_to_namespace(state["args"])
    code_generation_attempt_count, generated_attempt_dir = next_fresh_attempt_dir(
        args.generated_output_dir,
        state.get("code_generation_attempt_count", 0),
    )
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
                prompt_agent.write_generated_files(code_agent, generated_attempt_dir)
        except Exception as exc:
            code_agent = {
                "status": "llm_timeout_or_error",
                "reason": str(exc),
                "fallback": "Continue with the connected downstream extraction runner; extraction_eval will decide whether code repair is still required.",
                "code_change_plan": [],
                "generated_files": [],
            }
    code_agent["generation_attempt"] = code_generation_attempt_count
    code_agent["generated_output_dir"] = str(generated_attempt_dir)
    update = append_protocol_message(
        state,
        {
            "code_agent": code_agent,
            "code_generation_attempt_count": code_generation_attempt_count,
            "status": "running",
        },
        sender="code_generation_agent",
        phase="code_generate_or_patch",
        status=code_agent.get("status", "completed"),
        payload_refs={"prompt_package": "prompt_output", "judgement": "judgement"},
        decision={"accepted_by_supervisor": code_agent.get("status") not in {"llm_timeout_or_error", "failed"}},
        produced_artifacts=[{"ref": "code_agent"}],
        next_route="extraction_test_run",
    )
    return write_state_snapshot(
        merge_update(state, update),
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
    update = append_protocol_message(
        state,
        update,
        sender="code_repair_agent",
        phase="code_repair",
        status="needs_code_action",
        payload_refs={"diagnosis": "code_agent", "evaluation": "extraction_eval"},
        decision={"accepted_by_supervisor": False, "action": "hold_and_report_impact"},
        requested_actions=code_agent.get("code_change_plan", []),
        produced_artifacts=[{"ref": "code_agent"}],
        next_route="feedback_classification",
        evidence=code_agent.get("blocked_reason", []),
    )
    return write_state_snapshot(merge_update(state, update), "code_repair", "feedback_classification")


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
        extraction_attempt_count, extraction_attempt_dir = next_fresh_attempt_dir(
            args.extraction_run_dir,
            state.get("extraction_attempt_count", 0),
        )
        extraction_test = downstream_extraction_runner.run_extraction_bench(
            workflow_plan=workflow_plan,
            prompt_output=state.get("prompt_output") or {},
            documents=test_docs,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.extraction_max_tokens,
            output_dir=str(extraction_attempt_dir),
            max_documents=args.max_test_documents,
            max_document_chars=args.max_document_chars,
            max_fields_per_stage=args.max_fields_per_stage,
            max_workers=getattr(args, "extraction_workers", 4),
            paper_metadata_by_document=state.get("paper_metadata_by_document") or {},
            impact_manifest=state.get("impact_manifest_input") or {},
            run_identity=state.get("run_identity"),
            reuse_existing_outputs=False,
        )
        update = append_protocol_message(
            state,
            {
                "extraction_test": extraction_test,
                "extraction_attempt_count": extraction_attempt_count,
                "status": "running",
            },
            sender="downstream_extraction_runner",
            phase="extraction_test_run",
            status=extraction_test.get("status", "completed"),
            payload_refs={"workflow_plan": "workflow_plan", "impact_manifest": "impact_manifest_input"},
            decision={"accepted_by_supervisor": extraction_test.get("status") == "completed"},
            produced_artifacts=[{"ref": "extraction_test.section_results"}],
            next_route="extraction_eval",
        )
        return write_state_snapshot(
            merge_update(state, update),
            "extraction_test_run",
            "extraction_eval",
        )
    else:
        selective_scope = downstream_extraction_runner.build_selective_reprocess_scope(
            workflow_plan,
            state.get("prompt_output") or {},
            test_docs[: args.max_test_documents],
            state.get("impact_manifest_input") or {},
        )
        selected_stage_ids = set(selective_scope.get("selected_stage_ids", []))
        selected_fields_by_stage = selective_scope.get("selected_field_paths_by_stage", {})
        for stage in workflow_plan.get("section_test_plan", []):
            if stage["stage_id"] not in selected_stage_ids:
                continue
            field_count = len(downstream_extraction_runner.stage_field_specs(state.get("prompt_output") or {}, stage["stage_id"]))
            if stage["stage_id"] in selected_fields_by_stage:
                field_count = len(selected_fields_by_stage[stage["stage_id"]])
            section_results.append(
                {
                    "stage_id": stage["stage_id"],
                    "section_id": stage["section_id"],
                    "depends_on": stage.get("depends_on", []),
                    "status": "planned" if args.dry_run else "not_implemented",
                    "test_documents": selective_scope.get("selected_documents", []),
                    "field_count_tested": field_count,
                    "checks": ["dependency_inputs_available", "json_validity", "field_coverage", "evidence_provenance"],
                }
            )
        test_status = "dry_run_planned" if args.dry_run else "runner_missing"
        if selective_scope.get("errors"):
            test_status = "selection_rejected"
        reason = "Section-wise extraction tests are planned; plug in downstream_extraction_runner for live execution."
    extraction_test = {
        "status": test_status,
        "reason": reason,
        "section_results": section_results,
        "selective_reprocess": locals().get("selective_scope", {}),
    }
    update = append_protocol_message(
        state,
        {"extraction_test": extraction_test, "status": "running"},
        sender="downstream_extraction_runner",
        phase="extraction_test_run",
        status=test_status,
        payload_refs={"workflow_plan": "workflow_plan", "impact_manifest": "impact_manifest_input"},
        decision={"accepted_by_supervisor": test_status in {"completed", "dry_run_planned"}},
        produced_artifacts=[{"ref": "extraction_test.section_results"}],
        next_route="extraction_eval",
    )
    return write_state_snapshot(
        merge_update(state, update),
        "extraction_test_run",
        "extraction_eval",
    )


def diagnose_section_result(section_result, unicode_normalized=False):
    stage_id = section_result.get("stage_id", "")
    status = section_result.get("status", "")
    summary = section_result.get("summary") or {}
    doc_results = section_result.get("document_results", []) or []
    diagnoses = []

    if status == "metadata_missing":
        diagnoses.append(
            {
                "type": "metadata_handoff_missing",
                "severity": "blocking",
                "stage_id": stage_id,
                "documents": summary.get("metadata_missing_docs", []),
                "evidence": ["No retrieval/download metadata record matched one or more test documents."],
                "likely_cause": "The download manifest or paper archive metadata was not handed off, or document and paper identifiers do not match.",
                "recommended_repair": "Provide --paper-metadata or preserve paper_id/arXiv/DOI identifiers when converting PDF to Markdown.",
                "route_hint": "workflow_repair",
            }
        )

    transport_docs = [
        item for item in doc_results if isinstance(item, dict) and item.get("status") == "transport_error"
    ]
    if transport_docs:
        diagnoses.append(
            {
                "type": "retryable_transport_error",
                "severity": "blocking",
                "stage_id": stage_id,
                "documents": [item.get("document") for item in transport_docs],
                "evidence": [
                    (item.get("checks") or {}).get("error", "transport error")
                    for item in transport_docs[:3]
                ],
                "likely_cause": "The configured model endpoint rejected or temporarily failed one or more API calls.",
                "recommended_repair": "Retry the failed extraction in a fresh attempt with HTTP backoff and lower bounded concurrency.",
                "route_hint": "transport_retry",
            }
        )

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
    extraction_outcome = str(summary.get("extraction_outcome") or "")
    field_coverage_ratio = float(summary.get("field_coverage_ratio", 0) or 0)
    is_postprocess_stage = stage_id == "unicode_normalization" or str(section_result.get("section_id", "")).startswith("postprocess.")
    complete_negative_result = extraction_outcome == "no_values_found" and field_coverage_ratio >= 1.0
    complete_unresolved_result = extraction_outcome == "unresolved_requires_review" and field_coverage_ratio >= 1.0
    if status == "low_quality" or (
        extracted_total == 0
        and stage_id != "figure_classification"
        and status not in {"invalid_json", "transport_error"}
        and not is_postprocess_stage
        and not complete_negative_result
        and not complete_unresolved_result
    ):
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
    elif any(route == "transport_retry" for route in route_votes):
        recommended = "transport_retry"
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
    if extraction_test.get("status") == "selection_rejected":
        blockers.extend(
            (extraction_test.get("selective_reprocess") or {}).get("errors", [])
            or ["Selective reprocessing scope was rejected."]
        )
    if not section_results and extraction_test.get("status") != "completed_noop":
        blockers.append("No section-wise tests were planned.")
    failed_sections = [
        item
        for item in section_results
        if isinstance(item, dict)
        and item.get("status") in {"failed", "needs_repair", "invalid_json", "transport_error", "low_quality", "metadata_missing"}
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
        update["schema_feedback"] = None
        update["supervisor_decision"] = None
    update = append_protocol_message(
        state,
        update,
        sender="extraction_evaluation_agent",
        phase="extraction_eval",
        status="rejected" if blockers else "accepted",
        payload_refs={"test_results": "extraction_test.section_results", "judgement": "judgement"},
        decision={
            "accepted_by_supervisor": not blockers,
            "recommended_next_action": recommended_next_action,
        },
        requested_actions=eval_result.get("optimization_targets", []),
        produced_artifacts=[{"ref": "extraction_eval"}],
        next_route="supervisor_router" if blockers else "human_expert_review",
        evidence=blockers,
    )
    return write_state_snapshot(
        merge_update(state, update),
        "extraction_eval",
        "supervisor_router" if blockers else "human_expert_review",
    )


def human_expert_review_node(state):
    args = dict_to_namespace(state["args"])
    human_advice = state.get("human_advice", []) or []
    if not human_advice and getattr(args, "skip_human_expert_review", False):
        update = {
            "human_expert_review": {
                "status": "skipped",
                "reason": "Human expert review was explicitly skipped by --skip-human-expert-review.",
            },
            "status": "success",
        }
        update = append_protocol_message(
            state,
            update,
            sender="human_expert_review_gate",
            phase="human_expert_review",
            status="skipped",
            payload_refs={"extraction_eval": "extraction_eval"},
            decision={"accepted_by_supervisor": True, "explicit_skip": True},
            next_route="feedback_classification",
        )
        return write_state_snapshot(
            merge_update(state, update),
            "human_expert_review",
            "feedback_classification",
        )
    if not human_advice:
        update = {
            "human_expert_review": {
                "status": "waiting_for_human_advice",
                "reason": "Internal Step9 evaluation passed. Human expert review is required unless --skip-human-expert-review is set.",
                "required_advice": (
                    "As the task-domain materials expert, either approve the current Step9 extraction/test result "
                    "or describe concrete domain-output problems. Do not provide code or prompt implementation instructions."
                ),
            },
            "status": "waiting_for_human_advice",
            "next_node": "feedback_classification",
        }
        update = append_protocol_message(
            state,
            update,
            sender="human_expert_review_gate",
            phase="human_expert_review",
            status="waiting_for_human_advice",
            payload_refs={"extraction_eval": "extraction_eval"},
            decision={"accepted_by_supervisor": False, "action": "hold_for_expert"},
            requested_actions=[update["human_expert_review"]["required_advice"]],
            next_route="feedback_classification",
        )
        return write_state_snapshot(
            merge_update(state, update),
            "human_expert_review",
            "feedback_classification",
        )

    current_advice_text = advice_text(human_advice)
    previously_applied_advice = str(state.get("applied_human_advice") or "").strip()
    current_advice_already_applied = bool(
        state.get("human_advice_available_before_design")
        or (
            state.get("human_review_applied")
            and current_advice_text
            and current_advice_text == previously_applied_advice
        )
    )
    if current_advice_already_applied or advice_is_acceptance(human_advice):
        update = {
            "human_expert_review": {
                "status": "accepted",
                "advice": human_advice,
                "reason": "Human expert advice was available before Step9 design, accepted the internally passing output, or the same advice was already applied once.",
            },
            "status": "success",
            "validation_errors": [],
            "schema_feedback": None,
            "supervisor_decision": None,
        }
        update = append_protocol_message(
            state,
            update,
            sender="human_expert_review_gate",
            phase="human_expert_review",
            status="accepted",
            payload_refs={"expert_advice": "human_advice", "extraction_eval": "extraction_eval"},
            decision={"accepted_by_supervisor": True},
            produced_artifacts=[{"ref": "human_expert_review"}],
            next_route="feedback_classification",
        )
        return write_state_snapshot(
            merge_update(state, update),
            "human_expert_review",
            "feedback_classification",
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
    update = append_protocol_message(
        state,
        update,
        sender="human_expert_review_gate",
        phase="human_expert_review",
        status="revision_requested",
        payload_refs={"expert_advice": "human_advice", "extraction_eval": "extraction_eval"},
        decision={"accepted_by_supervisor": False, "route_hint": route},
        requested_actions=human_advice,
        next_route="supervisor_router",
    )
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
    update = append_protocol_message(
        state,
        update,
        sender="workflow_repair_agent",
        phase="workflow_repair",
        status="completed",
        payload_refs={"previous_workflow": "workflow_plan", "evaluation": "extraction_eval"},
        decision={"accepted_by_supervisor": True, "actions": repair_actions or ["no_structural_change"]},
        produced_artifacts=[{"ref": "workflow_plan"}],
        next_route="prompt_repair",
    )
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
    update = append_protocol_message(
        state,
        update,
        sender="schema_feedback_agent",
        phase="schema_feedback",
        status="needs_step8_review",
        payload_refs={"evaluation": "extraction_eval", "feedback": "schema_feedback"},
        decision={"accepted_by_supervisor": False, "action": "feedback_to_step8"},
        requested_actions=schema_feedback.get("recommended_step8_actions", []),
        produced_artifacts=[{"ref": "schema_feedback"}],
        next_route="feedback_classification",
    )
    return write_state_snapshot(merge_update(state, update), "schema_feedback", "feedback_classification")


def feedback_classification_node(state):
    classification, _ = build_feedback_classification_and_impact(state)
    update = append_protocol_message(
        state,
        {"feedback_classification": classification},
        sender="feedback_classification_agent",
        phase="feedback_classification",
        status="completed",
        payload_refs={
            "human_review": "human_expert_review",
            "schema_feedback": "schema_feedback",
            "evaluation": "extraction_eval",
        },
        decision={"accepted_by_supervisor": True, "feedback_type": classification.get("feedback_type")},
        produced_artifacts=[{"ref": "feedback_classification"}],
        next_route="impact_analysis",
        evidence=[classification.get("reason", "")],
    )
    return write_state_snapshot(merge_update(state, update), "feedback_classification", "impact_analysis")


def impact_analysis_node(state):
    classification, impact_manifest = build_feedback_classification_and_impact(state)
    impact_manifest["executable_selector"] = {
        "runner_argument": "--impact-manifest-input",
        "documents": impact_manifest.get("affected_documents", []),
        "stage_ids": impact_manifest.get("affected_stage_ids", []),
        "field_paths": impact_manifest.get("affected_field_paths", []),
        "include_stage_dependencies": True,
        "force_targeted_outputs": True,
        "preserve_unaffected_outputs": True,
    }
    update = append_protocol_message(
        state,
        {"feedback_classification": classification, "impact_manifest": impact_manifest},
        sender="impact_analysis_agent",
        phase="impact_analysis",
        status=impact_manifest.get("status", "completed"),
        payload_refs={"feedback_classification": "feedback_classification", "workflow_plan": "workflow_plan"},
        decision={
            "bounded_scope": bool(
                impact_manifest.get("affected_documents")
                or impact_manifest.get("affected_stage_ids")
                or impact_manifest.get("affected_field_paths")
            )
        },
        produced_artifacts=[{"ref": "impact_manifest"}],
        next_route="impact_supervisor",
    )
    return write_state_snapshot(merge_update(state, update), "impact_analysis", "impact_supervisor")


def impact_supervisor_node(state):
    manifest = deepcopy(state.get("impact_manifest") or {})
    status = manifest.get("status")
    bounded = bool(
        manifest.get("affected_documents")
        or manifest.get("affected_stage_ids")
        or manifest.get("affected_field_paths")
    )
    task_locked = (
        manifest.get("task_type") == MATERIAL_LITERATURE_TASK_CONTRACT["task_type"]
        and manifest.get("source_scope") == MATERIAL_LITERATURE_TASK_CONTRACT["source_scope"]
    )
    if not task_locked:
        decision = {
            "status": "rejected",
            "action": "hold_invalid_task_scope",
            "reason": "Impact scope does not preserve the materials-database scientific-literature task contract.",
        }
    elif status == "no_reprocessing_required":
        decision = {
            "status": "approved",
            "action": "no_reprocessing",
            "reason": "No feedback changes a record, schema, rule, prompt, validator, or workflow.",
        }
    elif status in {"current_record_only", "targeted_reprocessing_required"} and bounded:
        decision = {
            "status": "approved",
            "action": "approve_selective_reprocess",
            "reason": "The impact manifest provides bounded document, stage, or field selectors.",
        }
    else:
        decision = {
            "status": "needs_human_scope_confirmation",
            "action": "hold_unbounded_reprocess",
            "reason": "Historical reprocessing is not allowed until its document, stage, or field scope is bounded.",
        }
    manifest["supervisor_approval"] = decision
    final_status = state.get("status", "needs_human_review")
    if decision["status"] == "needs_human_scope_confirmation" and final_status != "waiting_for_human_advice":
        final_status = "needs_human_review"
    update = {
        "impact_manifest": manifest,
        "impact_supervisor_decision": decision,
        "status": final_status,
    }
    update = append_protocol_message(
        state,
        update,
        sender="impact_supervisor",
        receiver="step9_output_writer",
        phase="impact_supervision",
        status=decision["status"],
        payload_refs={"impact_manifest": "impact_manifest", "feedback_classification": "feedback_classification"},
        decision=decision,
        requested_actions=["bound affected documents, stages, or field paths"]
        if decision["status"] == "needs_human_scope_confirmation"
        else [],
        produced_artifacts=[{"ref": "impact_supervisor_decision"}],
        next_route="write_output",
        evidence=[decision["reason"]],
    )
    return write_state_snapshot(merge_update(state, update), "impact_supervisor", "write_output")


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
    elif failed_node == "extraction_eval" and recommended_action == "transport_retry" and retry_count < max_retries:
        retry_counts[failed_node] = retry_count + 1
        updated_args = deepcopy(state["args"])
        previous_workers = max(1, int(updated_args.get("extraction_workers", 1) or 1))
        updated_args["extraction_workers"] = max(1, previous_workers // 2)
        decision = {
            "action": "transport_retry",
            "reason": "Retryable API transport or rate-limit failures require a fresh extraction attempt, not prompt repair.",
            "previous_extraction_workers": previous_workers,
            "next_extraction_workers": updated_args["extraction_workers"],
            "next_node": "extraction_test_run",
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
            "next_node": "feedback_classification",
        }

    if decision.get("next_node") == "write_output":
        decision["next_node"] = "feedback_classification"

    update = {
        "supervisor_decision": decision,
        "retry_counts": retry_counts,
        "status": "running" if decision["next_node"] != "feedback_classification" else state.get("status", "needs_human_review"),
    }
    if decision.get("action") == "transport_retry":
        update["args"] = updated_args
    if decision["next_node"] != "feedback_classification":
        update["validation_errors"] = []
    update = append_protocol_message(
        state,
        update,
        sender="step9_supervisor",
        receiver=decision["next_node"],
        phase="supervisor_router",
        status="routed",
        payload_refs={"validation_errors": "validation_errors", "evaluation": "extraction_eval"},
        decision=decision,
        requested_actions=[decision.get("action")],
        next_route=decision["next_node"],
        evidence=errors,
    )
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
    feedback_classification = state.get("feedback_classification")
    impact_manifest = state.get("impact_manifest")
    if not feedback_classification or not impact_manifest:
        feedback_classification, impact_manifest = build_feedback_classification_and_impact(
            {**state, "human_expert_review": human_expert_review}
        )
    protocol_messages = state.get("protocol_messages") or []
    output = {
        "step": "step9_extraction_build_agent_system",
        "status": state.get("status", "needs_human_review"),
        "run_identity": state.get("run_identity"),
        "task_contract": literature_task_contract(state.get("step8_output") or {}),
        "inputs": {
            "step8_output": args.step8_output,
            "prompt_output": args.prompt_output,
            "test_documents": args.test_documents,
            "paper_metadata": args.paper_metadata,
            "dry_run": args.dry_run,
            "impact_manifest_input": getattr(args, "impact_manifest_input", ""),
        },
        "paper_metadata_report": state.get("paper_metadata_report", {}),
        "workflow_plan": state.get("workflow_plan"),
        "prompt_output": state.get("prompt_output"),
        "judgement": state.get("judgement"),
        "code_agent": state.get("code_agent"),
        "extraction_test": state.get("extraction_test"),
        "extraction_eval": extraction_eval,
        "schema_feedback": state.get("schema_feedback"),
        "feedback_classification": feedback_classification,
        "impact_manifest": impact_manifest,
        "impact_manifest_input": state.get("impact_manifest_input", {}),
        "impact_supervisor_decision": state.get("impact_supervisor_decision"),
        "supervisor_decision": state.get("supervisor_decision"),
        "human_expert_review": human_expert_review,
        "human_advice": state.get("human_advice", []),
        "repair_history": state.get("repair_history", []),
        "validation_errors": state.get("validation_errors", []),
        "protocol_messages": protocol_messages,
        "protocol_validation": agent_protocol.validate_message_list(protocol_messages),
    }
    output_path = Path(args.output)
    impact_path = output_path.with_suffix(output_path.suffix + ".impact.json")
    output["impact_manifest_path"] = str(impact_path)
    run_artifact_guard.atomic_write_json(
        output_path,
        output,
        run_identity=state.get("run_identity"),
    )
    run_artifact_guard.atomic_write_json(
        impact_path,
        impact_manifest,
        run_identity=state.get("run_identity"),
    )
    print(f"Saved Step9 result to {output_path}")
    print(f"Status: {output['status']}")
    return write_state_snapshot(merge_update(state, {"output": str(output_path)}), "write_output", "__end__")


def next_node(state):
    return (state.get("supervisor_decision") or {}).get("next_node", state.get("next_node") or "write_output")


def resume_router_node(state):
    return state


def resume_next_node(state):
    node = state.get("next_node") or "load_inputs"
    return "write_output" if node == "__end__" else node


def build_step9_graph():
    graph = StateGraph(Step9State)
    graph.add_node("resume_router", resume_router_node)
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
    graph.add_node("feedback_classification", feedback_classification_node)
    graph.add_node("impact_analysis", impact_analysis_node)
    graph.add_node("impact_supervisor", impact_supervisor_node)
    graph.add_node("supervisor_router", supervisor_router_node)
    graph.add_node("write_output", write_output_node)

    graph.add_edge(START, "resume_router")
    graph.add_conditional_edges("resume_router", resume_next_node)
    graph.add_conditional_edges("load_inputs", lambda state: state.get("next_node", "workflow_plan"))
    graph.add_edge("workflow_plan", "prompt_generate")
    graph.add_edge("prompt_generate", "prompt_quality_review")
    graph.add_conditional_edges("prompt_quality_review", lambda state: state.get("next_node", "code_generate_or_patch"))
    graph.add_edge("prompt_repair", "prompt_quality_review")
    graph.add_edge("code_generate_or_patch", "extraction_test_run")
    graph.add_edge("code_repair", "feedback_classification")
    graph.add_edge("extraction_test_run", "extraction_eval")
    graph.add_conditional_edges("extraction_eval", lambda state: state.get("next_node", "write_output"))
    graph.add_conditional_edges("human_expert_review", lambda state: state.get("next_node", "write_output"))
    graph.add_edge("workflow_repair", "prompt_repair")
    graph.add_edge("schema_feedback", "feedback_classification")
    graph.add_edge("feedback_classification", "impact_analysis")
    graph.add_edge("impact_analysis", "impact_supervisor")
    graph.add_edge("impact_supervisor", "write_output")
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
    parser.add_argument(
        "--paper-metadata",
        nargs="*",
        default=[],
        help="Optional metadata JSON files/directories. When omitted, Step9 auto-discovers paper_archive/metadata near test documents.",
    )
    parser.add_argument("--extraction-results", default="", help="Optional JSON file with live section-wise extraction test results.")
    parser.add_argument(
        "--impact-manifest-input",
        default="",
        help="Optional approved Step9 impact manifest used to rerun only affected literature documents, stages, and fields.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan section-wise extraction tests without executing a live runner.")
    parser.add_argument("--skip-code-generation", action="store_true", help="Skip LLM code generation and only build/evaluate the workflow plan.")
    parser.add_argument("--base-url", default=prompt_agent.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.getenv("CODE_AGENT_API_KEY"))
    parser.add_argument("--model", default=prompt_agent.DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=384000)
    parser.add_argument("--extraction-max-tokens", type=int, default=384000)
    parser.add_argument("--extraction-run-dir", default="step9_extraction_runs")
    parser.add_argument("--max-test-documents", type=int, default=4)
    parser.add_argument("--max-document-chars", type=int, default=200000)
    parser.add_argument(
        "--max-fields-per-stage",
        type=int,
        default=80,
        help=(
            "Maximum fields sent in one extraction API call. Larger stages are fully batched; "
            "this never truncates or caps the schema field inventory."
        ),
    )
    parser.add_argument(
        "--extraction-workers",
        type=int,
        default=4,
        help=(
            "Maximum concurrent document extraction calls within one dependency stage. "
            "Stages remain ordered by their dependency topology."
        ),
    )
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
    parser.add_argument(
        "--resume-from-state",
        default="",
        help="Resume from a previously written Step9 *.state.json snapshot.",
    )
    return parser


def _legacy_main_without_clean_run_guard():
    args = build_parser().parse_args()
    app = build_step9_graph()
    args_dict = vars(args)
    resume_from_state = args_dict.pop("resume_from_state")
    if resume_from_state:
        initial_state = load_json_if_exists(resume_from_state)
        if not initial_state:
            raise FileNotFoundError(f"Step9 resume state not found or empty: {resume_from_state}")
        initial_state["args"] = args_dict
    else:
        initial_state = {"args": args_dict, "next_node": "load_inputs"}
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


def build_step9_input_identity(args_dict):
    input_paths = [
        args_dict.get("step8_output", ""),
        args_dict.get("prompt_output", ""),
        args_dict.get("reference_pipeline", ""),
        args_dict.get("extraction_results", ""),
        args_dict.get("impact_manifest_input", ""),
        *list(args_dict.get("test_documents") or []),
        *list(args_dict.get("paper_metadata") or []),
        *list(args_dict.get("target_files") or []),
    ]
    immutable_inputs = {
        key: deepcopy(args_dict.get(key))
        for key in (
            "step8_output",
            "prompt_output",
            "reference_pipeline",
            "test_documents",
            "paper_metadata",
            "extraction_results",
            "impact_manifest_input",
            "dry_run",
            "skip_code_generation",
            "model",
            "temperature",
            "max_tokens",
            "extraction_max_tokens",
            "max_test_documents",
            "max_document_chars",
            "max_fields_per_stage",
            "extraction_workers",
            "target_files",
            "write_generated_files",
        )
    }
    return run_artifact_guard.build_input_identity(
        "step9_extraction_build",
        immutable_inputs,
        input_paths,
    )


def main():
    args = build_parser().parse_args()
    app = build_step9_graph()
    args_dict = vars(args).copy()
    requested_thread_id = args_dict.pop("thread_id")
    resume_from_state = args_dict.pop("resume_from_state")

    if resume_from_state:
        initial_state = load_json_if_exists(resume_from_state)
        if not initial_state:
            raise FileNotFoundError(
                f"Step9 resume state not found or empty: {resume_from_state}"
            )
        snapshot_args = deepcopy(initial_state.get("args") or {})
        if args.api_key:
            snapshot_args["api_key"] = args.api_key
        input_identity = build_step9_input_identity(snapshot_args)
        run_identity = run_artifact_guard.validate_resume_identity(
            initial_state,
            pipeline="step9_extraction_build",
            input_identity=input_identity,
            output_path=snapshot_args.get("output", ""),
        )
        initial_state["args"] = snapshot_args
        initial_state["run_identity"] = run_identity
        if args.human_advice or args.human_advice_file:
            snapshot_args["human_advice"] = list(args.human_advice or [])
            snapshot_args["human_advice_file"] = args.human_advice_file
            initial_state["args"] = snapshot_args
            resume_advice = load_human_advice(dict_to_namespace(snapshot_args))
            initial_state = inject_resume_human_advice(initial_state, resume_advice)
        run_artifact_guard.update_run_status(
            run_identity,
            "running",
            resumed_from=str(Path(resume_from_state).resolve(strict=False)),
        )
    else:
        input_identity = build_step9_input_identity(args_dict)
        output_path = Path(args_dict["output"])
        state_path = state_path_from_args(dict_to_namespace(args_dict))
        impact_path = output_path.with_suffix(output_path.suffix + ".impact.json")
        artifacts = [
            output_path,
            state_path,
            impact_path,
            args_dict.get("extraction_run_dir", ""),
        ]
        if args_dict.get("write_generated_files"):
            artifacts.append(args_dict.get("generated_output_dir", ""))
        run_identity = run_artifact_guard.reserve_fresh_run(
            pipeline="step9_extraction_build",
            output_path=output_path,
            input_identity=input_identity,
            artifact_paths=artifacts,
        )
        initial_state = {
            "args": args_dict,
            "next_node": "load_inputs",
            "run_identity": run_identity,
        }

    config = {
        "configurable": {
            "thread_id": f"{requested_thread_id}:{run_identity['run_id']}"
        }
    }
    try:
        result = app.invoke(initial_state, config=config)
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
        "completed" if result_status == "success" else "needs_review",
        result_status=result_status,
    )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "output": result.get("output"),
                "current_node": result.get("current_node"),
                "next_node": result.get("next_node"),
                "run_id": run_identity["run_id"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
