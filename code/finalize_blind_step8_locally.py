#!/usr/bin/env python3
"""Finish a blinded Step8 run from a saved field-planning checkpoint.

This fallback is intentionally narrow: it compiles the model-produced field plan,
runs the normal deterministic validators, and writes the standard Step8 artifact.
It never reads a manual schema, core-field mapping, or extraction gold file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import section_design_agent as section_agent
import section_design_langgraph_human_gate as graph
import run_artifact_guard


FALLBACK_PIPELINE = "step8_blind_local_fallback"


def _payload_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def structured_identifier(value: object) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(value or ""))
    return section_agent.stable_concept_id(text)


def infer_protocol_object_kind(label: str, value_kind: str, entity_id: str) -> str:
    text = section_agent.stable_concept_id(f"{label} {value_kind} {entity_id}")
    if any(token in text for token in ("process", "synthesis", "fabrication", "treatment")):
        return "process"
    if any(token in text for token in ("evidence", "provenance", "locator", "fragment")):
        return "evidence_collection"
    if any(token in text for token in ("status", "category", "qualifier", "presence")):
        return "classification"
    if any(
        token in text
        for token in ("quantity", "property", "value", "range", "uncertainty", "condition", "observation")
    ):
        return "measurement"
    return "scalar"


def hydrate_checkpoint_state(state: dict, output_path: Path) -> dict:
    """Normalize raw pipeline checkpoints into the graph-state write contract."""

    state = deepcopy(state)
    inputs = state.get("inputs") if isinstance(state.get("inputs"), dict) else {}
    args = state.setdefault("args", {})
    for key in ("database_goal", "discipline", "key_description_path", "reference_papers"):
        if key not in args and key in inputs:
            args[key] = inputs[key]
    args.setdefault("database_goal", "Construct a materials database from scientific literature.")
    args.setdefault("discipline", "Materials science")
    args.setdefault("key_description_path", "")
    args.setdefault("reference_papers", [])
    args.setdefault("require_human_advice_before_supervisor", False)
    args["output"] = str(output_path)

    shared = state.setdefault("shared_context", {})
    raw_queries = shared.get("query_requirements") or inputs.get("query_requirements") or []
    if isinstance(raw_queries, str):
        query_requirements = [
            item.strip() for item in raw_queries.split("||") if item.strip()
        ]
    else:
        query_requirements = [str(item).strip() for item in raw_queries if str(item).strip()]
    shared.setdefault("task_contract", dict(section_agent.MATERIAL_LITERATURE_TASK_CONTRACT))
    shared.setdefault("database_goal", str(args["database_goal"]))
    shared.setdefault("discipline", str(args["discipline"]))
    shared["query_requirements"] = query_requirements
    shared.setdefault("key_description_text", "")
    shared.setdefault("reference_paper_context", "")
    shared.setdefault("reference_paper_count", len(args.get("reference_papers") or []))
    shared.setdefault("human_advice", str(state.get("human_advice") or ""))
    shared.setdefault(
        "shared_context_block",
        section_agent.build_shared_context_block(
            shared["database_goal"],
            shared["discipline"],
            query_requirements,
            shared["key_description_text"],
            shared["reference_paper_context"],
        ),
    )
    return state


def remove_reference_field_contract(state: dict) -> dict:
    """Remove a placeholder policy document that is not a field catalog."""

    state = deepcopy(state)
    shared = state.setdefault("shared_context", {})
    shared["reference_field_contract"] = {
        "available": False,
        "source_path": "",
        "leaf_fields": [],
        "leaf_count": 0,
        "reason": "blind benchmark intentionally supplied no reference field catalog",
    }
    field_plan = (
        state.setdefault("module_outputs", {}).get("field_planning_module") or {}
    )
    field_plan.pop("reference_field_audit", None)
    return state


def apply_structured_advice(state: dict, advice_text: str) -> tuple[dict, dict]:
    payload = json.loads(advice_text)
    if not isinstance(payload, dict):
        raise ValueError("structured advice must be a JSON object")
    state = deepcopy(state)
    state["human_advice"] = advice_text
    shared = state.setdefault("shared_context", {})
    shared["human_advice"] = advice_text
    shared["human_advice_available_before_design"] = True
    shared["structured_concept_seeding_only"] = True
    subjective = state.setdefault("module_outputs", {}).setdefault(
        "subjective_supervisor_module", {}
    )
    raw_entities = (
        payload.get("entity_registry")
        or payload.get("entities")
        or payload.get("revised_entities")
        or []
    )
    explicit_entities = []
    entity_owner_keys = {}
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        entity_id = structured_identifier(
            item.get("entity_id") or item.get("name") or item.get("label")
        )
        if not entity_id:
            continue
        owner_key = str(
            item.get("owner_key")
            or ("paper_info" if entity_id == "publication" else f"{entity_id}_info")
        )
        entity_owner_keys[entity_id] = owner_key
        explicit_entities.append(
            {
                **item,
                "entity_id": entity_id,
                "label": str(item.get("label") or item.get("name") or entity_id),
                "owner_key": owner_key,
                "required": bool(item.get("required", True)),
                "independent_owner": bool(item.get("independent_owner", True)),
            }
        )
    if explicit_entities:
        subjective["entity_registry"] = explicit_entities
        subjective["structured_entity_registry"] = True
    explicit_concepts = []
    raw_concepts = payload.get("schema_concepts") or payload.get("revised_schema_concepts") or []
    for item in raw_concepts:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("name") or "").strip()
        concept_id = section_agent.stable_concept_id(item.get("concept_id") or label)
        if not concept_id or not label:
            continue
        raw_owner_value = item.get("entity_id") or item.get("owner_entity") or "material"
        if isinstance(raw_owner_value, list):
            raw_owner_value = next((value for value in raw_owner_value if str(value).strip()), "material")
        raw_owner = str(raw_owner_value)
        entity_id = structured_identifier(raw_owner.split("|", 1)[0]) or "material"
        owner_key = str(
            item.get("owner_key")
            or entity_owner_keys.get(entity_id)
            or ("material_info" if entity_id == "material" else f"{entity_id}_info")
        )
        value_kind = str(item.get("value_kind") or item.get("data_type") or "")
        explicit_concepts.append(
            {
                **item,
                "concept_id": concept_id,
                "label": label,
                "entity_id": entity_id,
                "owner_key": owner_key,
                "object_kind": str(
                    item.get("object_kind")
                    or infer_protocol_object_kind(label, value_kind, entity_id)
                ),
                "condition_requirements": list(
                    item.get("condition_requirements") or item.get("required_context") or []
                ),
                "evidence_types": list(item.get("evidence_types") or ["text"]),
                "data_type": value_kind,
                "definition": str(item.get("definition") or item.get("description") or ""),
                "source_requirement_ids": list(
                    dict.fromkeys([*(item.get("source_requirement_ids") or []), "human_advice"])
                ),
            }
        )
    if explicit_concepts:
        subjective["requirement_contract"] = {"concepts": explicit_concepts}
        subjective["must_have_concepts"] = [
            item.get("concept_id") or item.get("label") for item in explicit_concepts
        ]
    if "relationship_contracts" not in payload:
        payload["relationship_contracts"] = list(
            payload.get("relationships") or payload.get("revised_relationships") or []
        )
    if not state["module_outputs"].get("query_semantics_module"):
        query_objects = []
        for family in payload.get("query_families") or []:
            if not isinstance(family, dict):
                continue
            examples = family.get("examples") or [family.get("name")]
            for example in examples:
                if not example:
                    continue
                query_objects.append(
                    {
                        "query_requirement": str(example),
                        "object_type": str(family.get("name") or "materials_database_query"),
                        "recommended_field_groups": [],
                        "comparison_axes": [],
                        "anti_generic_warning": "Preserve entity, condition, evidence, and provenance binding.",
                    }
                )
        for counterfactual in payload.get("counterfactual_queries") or []:
            if not isinstance(counterfactual, dict) or not counterfactual.get("query"):
                continue
            query_objects.append(
                {
                    "query_requirement": str(counterfactual["query"]),
                    "object_type": str(counterfactual.get("category") or "counterfactual_query"),
                    "recommended_field_groups": [],
                    "comparison_axes": [],
                    "anti_generic_warning": str(counterfactual.get("failure_exposed") or ""),
                }
            )
        if query_objects:
            state["module_outputs"]["query_semantics_module"] = {
                "query_objects": query_objects
            }
    return state, payload


def _fallback_concept(
    concept_id: str,
    label: str,
    *,
    owner_key: str = "material_info",
    entity_id: str = "material",
    object_kind: str = "scalar",
    evidence_types: list[str] | None = None,
) -> dict:
    return {
        "concept_id": section_agent.stable_concept_id(concept_id),
        "label": str(label),
        "required": True,
        "entity_id": entity_id,
        "owner_key": owner_key,
        "object_kind": object_kind,
        "condition_requirements": [],
        "evidence_types": list(evidence_types or ["text"]),
        "source_requirement_ids": ["query_1"],
    }


def build_blind_subjective_fallback(state: dict) -> dict:
    """Derive a bounded requirement contract from completed, gold-blind modules."""

    outputs = state.get("module_outputs") or {}
    concepts = [
        _fallback_concept("paper_title", "paper title", owner_key="paper_info", entity_id="paper"),
        _fallback_concept("paper_doi", "paper DOI", owner_key="paper_info", entity_id="paper"),
        _fallback_concept("paper_authors", "paper authors", owner_key="paper_info", entity_id="paper"),
        _fallback_concept("publication_year", "publication year", owner_key="paper_info", entity_id="paper"),
        _fallback_concept("material_identity", "material identity"),
        _fallback_concept("chemical_formula", "material identity chemical formula"),
        _fallback_concept("sample_identity", "sample identity"),
        _fallback_concept("measurement_value", "measurement value", object_kind="measurement"),
        _fallback_concept("measurement_unit", "measurement unit", object_kind="measurement"),
        _fallback_concept("measurement_uncertainty", "measurement uncertainty", object_kind="measurement"),
        _fallback_concept("measurement_conditions", "measurement conditions", object_kind="measurement"),
        _fallback_concept("measurement_method", "measurement method", object_kind="measurement"),
        _fallback_concept("synthesis_method", "synthesis method", object_kind="process"),
        _fallback_concept("synthesis_conditions", "synthesis conditions", object_kind="process"),
        _fallback_concept("missingness_status", "explicit missingness status"),
        _fallback_concept("missingness_reason", "explicit missingness reason"),
        _fallback_concept(
            "source_figure",
            "source figure evidence",
            object_kind="evidence_collection",
            evidence_types=["text", "figure"],
        ),
        _fallback_concept(
            "source_table",
            "source table evidence",
            object_kind="evidence_collection",
            evidence_types=["text", "table"],
        ),
        _fallback_concept(
            "evidence_quote",
            "evidence quote and locator",
            object_kind="evidence_collection",
        ),
        _fallback_concept("provenance", "data provenance", object_kind="evidence_collection"),
        _fallback_concept("confidence_grade", "evidence confidence grade", object_kind="classification"),
    ]

    mechanism = outputs.get("mechanism_requirement_module") or {}
    for raw in mechanism.get("must_have_concepts") or []:
        concept_id = section_agent.stable_concept_id(raw)
        concepts.append(
            _fallback_concept(
                concept_id,
                concept_id.replace("_", " "),
                object_kind="classification" if "symmetry" in concept_id else "scalar",
            )
        )

    query_semantics = outputs.get("query_semantics_module") or {}
    for query_object in query_semantics.get("query_objects") or []:
        if not isinstance(query_object, dict):
            continue
        for raw in query_object.get("comparison_axes") or []:
            concept_id = section_agent.stable_concept_id(raw)
            concepts.append(_fallback_concept(concept_id, str(raw)))
        for raw in query_object.get("recommended_field_groups") or []:
            raw = str(raw)
            if "*" in raw:
                continue
            concept_id = section_agent.stable_concept_id(raw.rsplit(".", 1)[-1])
            concepts.append(_fallback_concept(concept_id, concept_id.replace("_", " ")))

    evidence_model = outputs.get("evidence_model_module") or {}
    for layer in evidence_model.get("evidence_layers") or []:
        if not isinstance(layer, dict) or not layer.get("layer_name"):
            continue
        concept_id = section_agent.stable_concept_id(f"{layer['layer_name']} evidence layer")
        concepts.append(
            _fallback_concept(
                concept_id,
                str(layer["layer_name"]),
                object_kind="evidence_collection",
                evidence_types=["text", "table", "figure"],
            )
        )

    deduped = []
    seen = set()
    for concept in concepts:
        concept_id = concept["concept_id"]
        if not concept_id or concept_id in seen:
            continue
        seen.add(concept_id)
        deduped.append(concept)

    existing = outputs.get("subjective_supervisor_module") or {}
    entity_registry = list(existing.get("entity_registry") or [])
    if not entity_registry:
        entity_registry = [
            {
                "entity_id": "material",
                "label": "Material and sample record",
                "required": True,
                "owner_key": "material_info",
                "independent_owner": False,
            },
            {
                "entity_id": "paper",
                "label": "Source paper",
                "required": True,
                "owner_key": "paper_info",
                "independent_owner": True,
            },
        ]
    return {
        "database_nature": "Evidence-backed materials database constructed from scientific literature",
        "modeling_position": "Material and sample records own condition-bound properties and linked evidence.",
        "must_have_concepts": [item["concept_id"] for item in deduped],
        "requirement_contract": {"concepts": deduped},
        "entity_registry": entity_registry,
        "must_not_become": ["generic document index", "unstructured property bag"],
        "red_flags": ["umbrella fields", "unbound values", "missing provenance", "implicit missingness"],
        "approved_section_strategy": [
            "material identity and tuning",
            "task-relevant properties",
            "preparation and processing",
            "microscopic characterization",
            "macroscopic measurements",
            "theory and mechanisms",
        ],
        "redo_directives": [],
        "fallback_source": "completed gold-blind Step8 modules",
    }


def ensure_blind_architecture_modules(state: dict) -> dict:
    state = deepcopy(state)
    outputs = state.setdefault("module_outputs", {})
    if not outputs.get("query_semantics_module"):
        raw_requirements = state.get("shared_context", {}).get("query_requirements") or []
        if isinstance(raw_requirements, str):
            raw_requirements = [raw_requirements]
        query_requirements = [str(item) for item in raw_requirements if str(item).strip()]
        if not query_requirements:
            query_requirements = [
                "Retrieve and compare material records with sample, condition, evidence, and provenance binding."
            ]
        outputs["query_semantics_module"] = {
            "query_objects": [
                {
                    "query_requirement": requirement,
                    "object_type": "materials_database_query",
                    "recommended_field_groups": [],
                    "comparison_axes": [],
                    "anti_generic_warning": "Do not flatten sample-bound or condition-bound assertions.",
                }
                for requirement in query_requirements
            ]
        }
    subjective = outputs.get("subjective_supervisor_module") or {}
    if not (subjective.get("requirement_contract") or {}).get("concepts"):
        outputs["subjective_supervisor_module"] = build_blind_subjective_fallback(state)
    outputs.setdefault(
        "topic_adaptation_module",
        {
            "topic_type": "task-adaptive materials literature database",
            "adaptation_principles": [
                "derive domain fields from distilled literature concepts",
                "bind values to samples, conditions, evidence, and provenance",
            ],
            "avoid_generic_template": ["property bag", "document-only index"],
            "topic_specific_adjustments": ["preserve the domain concepts found by upstream modules"],
        },
    )
    standard_sections = [
        section_agent.section_descriptor(section_id)
        for section_id in (
            "material_info.section0",
            "material_info.section1",
            "material_info.section2",
            "material_info.section3",
            "material_info.section4",
            "section5",
        )
    ]
    outputs.setdefault(
        "section_partition_module",
        {"core_sections": standard_sections, "non_core_sections": []},
    )
    return state


def ensure_deterministic_planning_modules(state: dict, *, replace_field_plan: bool) -> dict:
    state = ensure_blind_architecture_modules(state)
    outputs = state.setdefault("module_outputs", {})
    shared = state.get("shared_context") or {}
    subjective = outputs.get("subjective_supervisor_module") or {}
    contract = section_agent.normalize_requirement_contract(shared, subjective)
    concepts = contract.get("concepts") or []
    if replace_field_plan or not outputs.get("field_planning_module"):
        grouped = {}
        for concept in concepts:
            owner = graph.concept_owner_prefix(concept)
            grouped.setdefault(owner, []).append(f"{owner}.{concept['concept_id']}")
        outputs["field_planning_module"] = {
            "planning_status": "deterministic_from_structured_requirement_contract",
            "field_groups": [
                {
                    "group_name": owner,
                    "section_id": owner,
                    "purpose": f"Executable fields owned by {owner}",
                    "evidence_strategy": "field-level direct evidence with stable locators",
                    "recommended_fields": paths,
                }
                for owner, paths in sorted(grouped.items())
            ],
            "red_flag_fixes": [
                "one executable field per atomic concept",
                "condition and evidence binding retained in field contracts",
            ],
            "gold_available": False,
        }
    outputs.setdefault(
        "supervisor_module",
        {
            "review_stage": "post_section_partition",
            "enable_figure_classification": False,
            "routing_decision": "deterministic_blind_fallback",
            "decision_summary": "Compile the completed gold-blind module contract locally.",
            "risk_signals": ["online downstream module unavailable"],
            "risk_assessment": "preserve failure evidence and validate the compiled schema",
            "trigger_sections": [],
            "expected_figure_fields": ["material_info.section4.source_figure"],
            "skip_reason": "figure constraints are inferred deterministically from field paths",
            "approved_schema_roots": sorted(
                {
                    graph.concept_owner_prefix(concept).split(".", 1)[0]
                    for concept in concepts
                }
            ),
            "decision_basis": "structured expert contract plus deterministic validators",
        },
    )
    outputs.setdefault(
        "figure_classification_module",
        {
            "enable_figure_classification": False,
            "routing_status": "skipped",
            "classification_strategy": [],
            "section_figure_plan": [],
            "skip_reason": "deterministic field-level figure constraints are applied",
        },
    )
    return state


def structured_relationship_errors(result: dict) -> list[str]:
    entities = {
        structured_identifier(item.get("entity_id") or item.get("label"))
        for item in result.get("entity_registry") or []
        if isinstance(item, dict)
    }
    errors = []
    seen_ids = set()
    for index, relationship in enumerate(result.get("relationship_contracts") or [], start=1):
        if not isinstance(relationship, dict):
            errors.append(f"relationship:{index}:not_an_object")
            continue
        relationship_id = structured_identifier(
            relationship.get("relationship_id") or f"relationship_{index}"
        )
        if relationship_id in seen_ids:
            errors.append(f"relationship:{relationship_id}:duplicate_id")
        seen_ids.add(relationship_id)
        source = relationship.get("source_entity") or relationship.get("source")
        target = relationship.get("target_entity") or relationship.get("target")
        if isinstance(source, list) or isinstance(target, list) or any(
            token in str(value or "") for value in (source, target) for token in ("|", "/")
        ):
            errors.append(f"relationship:{relationship_id}:non_atomic_endpoint")
            continue
        source_id = structured_identifier(source)
        target_id = structured_identifier(target)
        if source_id not in entities:
            errors.append(f"relationship:{relationship_id}:unknown_source:{source_id}")
        if target_id not in entities:
            errors.append(f"relationship:{relationship_id}:unknown_target:{target_id}")
    return errors


def apply_structured_protocol_release_gate(result: dict, outputs: dict, advice_payload: dict) -> None:
    """Remove compatibility-only fields and retain the typed protocol registry."""

    released_entities = [
        item
        for item in result.get("entity_registry") or []
        if isinstance(item, dict) and item.get("primary_key") and item.get("owner_key")
    ]
    entity_roots = {str(item["owner_key"]) for item in released_entities}
    schema_definition = result.setdefault("schema_definition", {})
    fields = [
        field
        for field in schema_definition.get("field_registry") or []
        if isinstance(field, dict)
        and str(field.get("field_path") or "").split(".", 1)[0] in entity_roots
    ]
    for field in fields:
        figure_constraint = field.get("figure_constraint")
        if isinstance(figure_constraint, dict):
            figure_constraint["uses_figure_classification"] = False
            figure_constraint["why_needed"] = (
                "Figure and table evidence resolves through DocumentFragment and EvidenceItem locators; "
                "no separate classifier is required."
            )
        object_contract = field.get("object_contract")
        if isinstance(object_contract, dict):
            required = list(object_contract.get("required_subfields") or [])
            if "confidence" in required:
                required = [
                    "extraction_confidence" if item == "confidence" else item
                    for item in required
                ]
            object_contract["required_subfields"] = list(dict.fromkeys(required))
            contract_fields = object_contract.get("fields")
            if isinstance(contract_fields, dict) and "confidence" in contract_fields:
                contract_fields["extraction_confidence"] = contract_fields.pop("confidence")
    schema_definition["field_registry"] = fields
    outputs.setdefault("schema_design_module", {})["field_registry"] = fields
    schema_definition["top_level_keys"] = [
        {
            "key": str(entity["owner_key"]),
            "description": str(entity.get("role") or entity.get("label") or entity["entity_id"]),
        }
        for entity in released_entities
    ]
    result["section_design"] = {
        "core_sections": [
            {
                "section_id": str(entity["owner_key"]),
                "section_name": str(entity.get("label") or entity["entity_id"]),
                "purpose": str(entity.get("role") or "Typed scientific record owner"),
            }
            for entity in released_entities
        ],
        "non_core_sections": [],
        "architecture": "typed_entity_records_from_structured_protocol",
    }

    raw_concepts = advice_payload.get("schema_concepts") or advice_payload.get("revised_schema_concepts") or []
    explicit_concept_ids = {
        section_agent.stable_concept_id(
            item.get("concept_id") or item.get("label") or item.get("name")
        )
        for item in raw_concepts
        if isinstance(item, dict)
    }
    requirement_contract = result.get("requirement_contract") or {}
    referenced_concept_ids = {
        str(concept_id)
        for field in fields
        for concept_id in field.get("concept_ids") or []
        if concept_id
    }
    requirement_contract["concepts"] = [
        concept
        for concept in requirement_contract.get("concepts") or []
        if isinstance(concept, dict)
        and concept.get("concept_id") in (explicit_concept_ids | referenced_concept_ids)
    ]
    keep_schema_support = bool(referenced_concept_ids - explicit_concept_ids)
    if not keep_schema_support:
        requirement_contract["source_requirements"] = [
            item
            for item in requirement_contract.get("source_requirements") or []
            if isinstance(item, dict) and item.get("requirement_id") != "schema_support"
        ]
    result["coverage_report"] = section_agent.build_coverage_report(result)
    section_agent.ensure_literature_field_rule_contract(result, fields)
    result["structured_protocol_release_gate"] = {
        "typed_entity_root_count": len(entity_roots),
        "released_field_count": len(fields),
        "legacy_compatibility_roots_removed": ["material_info", "section5"],
        "figure_evidence_route": "document_fragment_and_evidence_item",
    }


def finalize(
    state: dict,
    output_path: Path,
    reason: str,
    *,
    ignore_reference_contract: bool = False,
    structured_advice_text: str = "",
    replace_model_field_plan: bool = False,
) -> dict:
    state = hydrate_checkpoint_state(state, output_path)
    output_path = output_path.expanduser().resolve(strict=False)
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".state.json")
    state.setdefault("args", {})["output"] = str(output_path)
    state["args"]["checkpoint_output"] = str(checkpoint_path)
    input_identity = run_artifact_guard.build_input_identity(
        FALLBACK_PIPELINE,
        {
            "source_state_sha256": _payload_digest(state),
            "structured_advice_sha256": _payload_digest(structured_advice_text),
            "reason": str(reason),
            "ignore_reference_contract": bool(ignore_reference_contract),
            "replace_model_field_plan": bool(replace_model_field_plan),
            "official_benchmark_eligible": False,
        },
    )
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline=FALLBACK_PIPELINE,
        output_path=output_path,
        input_identity=input_identity,
        artifact_paths=[output_path, checkpoint_path],
    )
    state["run_identity"] = run_identity
    try:
        return _finalize_reserved(
            state,
            output_path,
            reason,
            ignore_reference_contract=ignore_reference_contract,
            structured_advice_text=structured_advice_text,
            replace_model_field_plan=replace_model_field_plan,
        )
    except BaseException as exc:
        run_artifact_guard.update_run_status(
            run_identity,
            "interrupted",
            error_type=type(exc).__name__,
            error_message=str(exc)[:1000],
        )
        raise


def _finalize_reserved(
    state: dict,
    output_path: Path,
    reason: str,
    *,
    ignore_reference_contract: bool = False,
    structured_advice_text: str = "",
    replace_model_field_plan: bool = False,
) -> dict:
    advice_payload = {}
    if structured_advice_text:
        state, advice_payload = apply_structured_advice(state, structured_advice_text)
    if ignore_reference_contract:
        state = remove_reference_field_contract(state)
    state = ensure_deterministic_planning_modules(
        state,
        replace_field_plan=replace_model_field_plan,
    )
    state = graph.compile_schema_from_field_planning(state, reason)
    outputs = deepcopy(state.get("module_outputs") or {})
    registry = (outputs.get("schema_design_module") or {}).get("field_registry") or []
    if not registry:
        raise ValueError("field planning compiled to an empty schema")

    critic = {
        "specialization_status": "pass",
        "is_generic": False,
        "missing_concepts": [],
        "structural_weaknesses": [],
        "redo_needed": False,
        "redo_directives": [],
        "field_utility_audit": {
            "reviewed_field_count": len(registry),
            "decision": "pass",
            "unsupported_fields": [],
            "redundant_fields": [],
            "alias_or_unit_variant_fields": [],
            "derivable_duplicate_fields": [],
            "rationale": (
                "Deterministic local fallback checked structural traceability only; "
                "this artifact is not eligible as an official benchmark result."
            ),
        },
        "patch_operations": [],
        "reviewed_schema_revision": int(state.get("schema_revision", 0) or 0),
        "review_scope": "deterministic structure and requirement traceability only",
        "gold_available": False,
    }
    outputs["specialization_critic_module"] = critic
    state["critic_reviewed_schema_revision"] = int(state.get("schema_revision", 0) or 0)
    attempts = deepcopy(state.get("module_attempts") or {})
    attempts["specialization_critic_module"] = [
        {
            "round": 1,
            "prompt": "local_blind_structural_review_after_schema_timeout",
            "raw_response": json.dumps(critic, ensure_ascii=False),
        }
    ]
    errors_by_module = deepcopy(state.get("module_errors") or {})
    errors_by_module["specialization_critic_module"] = []

    shared = state.get("shared_context") or {}
    assembled = section_agent.assemble_final_result_from_modules(
        shared,
        outputs["locating_module"],
        outputs["query_semantics_module"],
        outputs["subjective_supervisor_module"],
        outputs["topic_adaptation_module"],
        outputs["section_partition_module"],
        outputs["field_planning_module"],
        outputs["supervisor_module"],
        outputs["figure_classification_module"],
        outputs["schema_design_module"],
        critic,
    )
    result = section_agent.finalize_result(shared, assembled)
    if advice_payload:
        result["relationship_contracts"] = list(
            advice_payload.get("relationship_contracts") or []
        )
        result["structured_expert_contract"] = {
            "round": advice_payload.get("round"),
            "verdict": advice_payload.get("verdict"),
            "schema_concept_count": len(advice_payload.get("schema_concepts") or []),
            "gold_available": False,
        }
        apply_structured_protocol_release_gate(result, outputs, advice_payload)
    validation_errors = section_agent.validate_result(result)
    validation_errors.extend(
        section_agent.validate_specialization(shared, outputs, result)
    )
    if advice_payload:
        validation_errors.extend(structured_relationship_errors(result))

    state.update(
        {
            "module_outputs": outputs,
            "module_attempts": attempts,
            "module_errors": errors_by_module,
            "result": result,
            "validation_errors": validation_errors,
            "status": "fallback_needs_online_completion",
            "execution_provenance": {
                "mode": "blind_deterministic_fallback",
                "fallback_used": True,
                "all_required_modules_completed": False,
                "fallback_reason": reason,
                "source_checkpoint_stage": state.get("stage") or state.get("current_node"),
                "validation_error_count": len(validation_errors),
                "official_benchmark_eligible": False,
                "ineligibility_reason": (
                    "Local deterministic fallback did not complete the online API critic and supervisor gates."
                ),
            },
            "current_node": "local_blind_schema_finalize",
            "next_node": "write_output",
        }
    )
    state.setdefault("args", {})["output"] = str(output_path)
    graph.write_output_node(state)
    run_artifact_guard.update_run_status(
        state.get("run_identity"),
        "needs_review",
        official_benchmark_eligible=False,
        output_status=state["status"],
    )
    return {
        "status": state["status"],
        "field_count": len((result.get("schema_definition") or {}).get("field_registry") or []),
        "validation_errors": validation_errors,
        "output": str(output_path),
        "run_id": state["run_identity"]["run_id"],
        "official_benchmark_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reason",
        default="online schema generation exceeded the benchmark wall-time limit",
    )
    parser.add_argument("--ignore-reference-contract", action="store_true")
    parser.add_argument("--structured-advice-path")
    parser.add_argument("--replace-model-field-plan", action="store_true")
    args = parser.parse_args()
    state_path = Path(args.state).resolve()
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    structured_advice_text = (
        Path(args.structured_advice_path).read_text(encoding="utf-8-sig")
        if args.structured_advice_path
        else ""
    )
    summary = finalize(
        state,
        Path(args.output).resolve(),
        args.reason,
        ignore_reference_contract=args.ignore_reference_contract,
        structured_advice_text=structured_advice_text,
        replace_model_field_plan=args.replace_model_field_plan,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
