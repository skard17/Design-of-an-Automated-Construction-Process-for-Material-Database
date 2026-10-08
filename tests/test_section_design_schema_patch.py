import json
import sys
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import section_design_agent as agent  # noqa: E402
import section_design_agent_prompt as prompts  # noqa: E402
import section_design_langgraph_human_gate as graph  # noqa: E402


def field(path, *, data_type="string", concept_ids=None):
    return {
        "field_path": path,
        "section_id": graph.infer_section_from_field_path(path),
        "field_name": path.rsplit(".", 1)[-1],
        "data_type": data_type,
        "required": False,
        "source_basis": ["text"],
        "concept_ids": concept_ids or [],
        "figure_constraint": None,
        "extraction_notes": "Extract explicit evidence only.",
        "reason": "test",
    }


def utility_audit(field_count, **overrides):
    audit = {
        "reviewed_field_count": field_count,
        "decision": "pass",
        "unsupported_fields": [],
        "redundant_fields": [],
        "alias_or_unit_variant_fields": [],
        "derivable_duplicate_fields": [],
        "rationale": "Every field adds a distinct query or evidence capability.",
    }
    audit.update(overrides)
    return audit


def base_state():
    return {
        "shared_context": {"query_requirements": ["record coercive field"]},
        "module_outputs": {
            "subjective_supervisor_module": {
                "requirement_contract": {
                    "concepts": [
                        {
                            "concept_id": "coercive_field",
                            "label": "Coercive field",
                            "required": True,
                            "entity_id": "material",
                            "owner_key": "material_info",
                            "object_kind": "measurement",
                            "source_requirement_ids": ["query_1"],
                        }
                    ]
                },
                "entity_registry": [],
            },
            "field_planning_module": {
                "field_groups": [
                    {
                        "section_id": "material_info.section1",
                        "recommended_fields": ["material_info.section1.coercive_field"],
                        "evidence_strategy": "text",
                    }
                ]
            },
            "schema_design_module": {
                "top_level_keys": [{"key": "material_info"}],
                "field_registry": [
                    field(
                        "material_info.section1.coercive_field",
                        concept_ids=["coercive_field"],
                    )
                ],
            },
        },
        "module_attempts": {},
        "module_errors": {},
        "schema_revision": 2,
        "critic_reviewed_schema_revision": 2,
        "validation_errors": [],
    }


def test_structured_add_field_completes_object_contract_and_revision():
    state = base_state()
    patched = graph.apply_schema_patch_operations(
        state,
        [
            {
                "op": "add_field",
                "field_path": "material_info.section1.magnetic_order",
                "field": {
                    "data_type": "object",
                    "concept_ids": ["coercive_field"],
                    "source_basis": ["text"],
                },
                "reason": "Represent a structured classification with evidence.",
            }
        ],
    )

    schema = patched["module_outputs"]["schema_design_module"]
    added = next(
        item
        for item in schema["field_registry"]
        if item["field_path"] == "material_info.section1.magnetic_order"
    )
    assert not agent.object_contract_missing_slots(added)
    assert patched["schema_revision"] == 3
    assert patched["critic_reviewed_schema_revision"] == -1
    assert [item["key"] for item in schema["top_level_keys"]] == ["material_info"]


def test_critic_patch_rejects_invented_requirement_concept_id():
    state = base_state()

    patched = graph.apply_schema_patch_operations(
        state,
        [
            {
                "op": "add_field",
                "field_path": "material_info.section1.exchange_coupling",
                "field": {
                    "data_type": "number",
                    "concept_ids": ["invented_exchange_coupling"],
                    "source_basis": ["text"],
                },
                "reason": "Represent a proposed quantity.",
            }
        ],
    )

    assert patched["schema_revision"] == state["schema_revision"]
    assert patched["validation_errors"] == [
        "schema_patch_rejected:patch[0] concept_ids are not in "
        "requirement_contract: ['invented_exchange_coupling']"
    ]


def test_patch_accepts_legacy_section5_alias_and_persists_canonical_path():
    state = base_state()
    schema = state["module_outputs"]["schema_design_module"]
    schema["field_registry"].append(
        {
            **field("material_info.section5.mechanism_interpretation"),
            "section_id": "material_info.section5",
        }
    )

    patched = graph.apply_schema_patch_operations(
        state,
        [
            {
                "op": "remove_field",
                "field_path": "material_info.section5.mechanism_interpretation",
                "reason": "Replace the scalar field with a structured record.",
            },
            {
                "op": "add_field",
                "field_path": "material_info.section5.mechanism_interpretation",
                "field": {
                    "section_id": "material_info.section5",
                    "data_type": "array<object>",
                    "source_basis": ["text"],
                },
                "reason": "Preserve per-mechanism evidence and provenance.",
            },
        ],
    )

    assert patched["validation_errors"] == []
    fields = patched["module_outputs"]["schema_design_module"]["field_registry"]
    replacement = next(
        item for item in fields if item["field_path"] == "section5.mechanism_interpretation"
    )
    assert replacement["section_id"] == "section5"
    assert not any(
        item["field_path"].startswith("material_info.section5.") for item in fields
    )


def test_invalid_patch_is_atomic_and_does_not_self_approve_root():
    state = base_state()
    original = json.dumps(state["module_outputs"]["schema_design_module"], sort_keys=True)

    patched = graph.apply_schema_patch_operations(
        state,
        [
            {
                "op": "add_field",
                "field_path": "invented_info.value",
                "field": {"data_type": "string"},
            }
        ],
    )

    assert patched["validation_errors"]
    assert json.dumps(
        patched["module_outputs"]["schema_design_module"], sort_keys=True
    ) == original


def test_same_path_move_reassigns_field_to_its_canonical_owner():
    state = base_state()
    state["module_outputs"]["subjective_supervisor_module"]["entity_registry"].append(
        {"entity_id": "measurement", "owner_key": "measurement_info"}
    )
    schema = state["module_outputs"]["schema_design_module"]
    schema["top_level_keys"].append({"key": "measurement_info"})
    schema["field_registry"].append(
        {
            **field("measurement_info.measurement_temperature"),
            "section_id": "material_info.section3",
        }
    )

    patched = graph.apply_schema_patch_operations(
        state,
        [
            {
                "op": "move_field",
                "field_path": "measurement_info.measurement_temperature",
                "target_path": "measurement_info.measurement_temperature",
                "reason": "Reassign the field to its canonical measurement owner.",
            }
        ],
    )

    moved = [
        item
        for item in patched["module_outputs"]["schema_design_module"]["field_registry"]
        if item["field_path"] == "measurement_info.measurement_temperature"
    ]
    assert len(moved) == 1
    assert moved[0]["section_id"] == "measurement_info"
    assert patched["validation_errors"] == []
    assert patched["schema_revision"] == 3


def test_move_to_another_existing_field_remains_a_conflict():
    state = base_state()
    schema = state["module_outputs"]["schema_design_module"]
    schema["field_registry"].append(field("material_info.section1.magnetic_order"))
    original = json.dumps(schema, sort_keys=True)

    patched = graph.apply_schema_patch_operations(
        state,
        [
            {
                "op": "move_field",
                "field_path": "material_info.section1.coercive_field",
                "target_path": "material_info.section1.magnetic_order",
                "reason": "This must remain invalid because the target is occupied.",
            }
        ],
    )

    assert patched["validation_errors"] == [
        "schema_patch_rejected:patch[0] move_field target already exists: "
        "material_info.section1.magnetic_order"
    ]
    assert json.dumps(
        patched["module_outputs"]["schema_design_module"], sort_keys=True
    ) == original


def test_nested_object_contract_update_preserves_and_merges_contract():
    state = base_state()
    target = state["module_outputs"]["schema_design_module"]["field_registry"][0]
    target["data_type"] = "object"
    target["object_contract"] = {
        "object_kind": "measurement",
        "required_subfields": ["value"],
        "fields": {"value": "reported value"},
    }

    patched = graph.apply_schema_patch_operations(
        state,
        [
            {
                "op": "update_field",
                "field_path": "material_info.section1.coercive_field",
                "changes": {
                    "object_contract.required_subfields": [
                        "value",
                        "unit",
                        "sample_id",
                        "evidence",
                    ]
                },
                "reason": "Bind the measurement to its sample and evidence.",
            }
        ],
    )

    updated = patched["module_outputs"]["schema_design_module"]["field_registry"][0]
    contract = updated["object_contract"]
    assert contract["object_kind"] == "measurement"
    assert set(contract["required_subfields"]) >= {
        "value",
        "unit",
        "sample_id",
        "evidence",
        "conditions",
        "entity_ref",
        "source_type",
        "extraction_confidence",
    }
    assert contract["fields"]["value"] == "reported value"
    assert patched["validation_errors"] == []


def test_unknown_nested_update_key_is_rejected_atomically():
    state = base_state()
    original = json.dumps(
        state["module_outputs"]["schema_design_module"], sort_keys=True
    )

    patched = graph.apply_schema_patch_operations(
        state,
        [
            {
                "op": "update_field",
                "field_path": "material_info.section1.coercive_field",
                "changes": {"object_contract.unapproved_slot": "value"},
            }
        ],
    )

    assert "object_contract.unapproved_slot" in patched["validation_errors"][0]
    assert json.dumps(
        patched["module_outputs"]["schema_design_module"], sort_keys=True
    ) == original


def test_real_critic_protocol_errors_are_rejected_with_exact_keys():
    state = base_state()
    state["module_outputs"]["subjective_supervisor_module"]["entity_registry"].append(
        {"entity_id": "observation", "owner_key": "observations"}
    )
    schema = state["module_outputs"]["schema_design_module"]
    schema["field_registry"].extend(
        [
            field("material_info.section3.characterization_measurements.evidence"),
            field("observations.property_uncertainty"),
        ]
    )

    _normalized, errors = graph.validate_schema_patch_operations(
        schema,
        [
            {
                "op": "add_field",
                "field_path": "material_info.section1.coercive_field",
                "field": {"data_type": "string"},
            },
            {
                "op": "update_field",
                "field_path": "material_info.section3.characterization_measurements.evidence",
                "changes": {
                    "data_type": "object",
                    "object_contract_ref": "object_contract_004",
                    "fields": {"document_id": "required_semantic_slot"},
                },
            },
            {
                "op": "update_field",
                "field_path": "observations.property_uncertainty",
                "changes": {
                    "data_type": "object",
                    "required_subfields": ["type", "value", "unit", "confidence"],
                },
            },
        ],
        state,
    )

    assert errors == [
        "patch[0] add_field already exists: material_info.section1.coercive_field",
        "patch[1] changes unsupported keys: ['fields', 'object_contract_ref']",
        "patch[2] changes unsupported keys: ['required_subfields']",
    ]


def test_complete_object_contract_update_remains_valid():
    state = base_state()
    target = state["module_outputs"]["schema_design_module"]["field_registry"][0]

    patched = graph.apply_schema_patch_operations(
        state,
        [
            {
                "op": "update_field",
                "field_path": target["field_path"],
                "changes": {
                    "data_type": "object",
                    "object_contract": {
                        "object_kind": "uncertainty",
                        "required_subfields": ["type", "value", "unit", "confidence"],
                        "fields": {
                            "type": "uncertainty type",
                            "value": "uncertainty value",
                            "unit": "uncertainty unit",
                            "confidence": "reported confidence",
                        },
                    },
                },
                "reason": "Represent uncertainty as a queryable object.",
            }
        ],
    )

    assert patched["validation_errors"] == []
    assert patched["schema_revision"] == 3
    updated = patched["module_outputs"]["schema_design_module"]["field_registry"][0]
    assert updated["object_contract"]["object_kind"] == "measurement"
    assert {"type", "value", "unit", "confidence"}.issubset(
        set(updated["object_contract"]["required_subfields"])
    )


def test_stale_critic_patch_is_not_applied():
    state = base_state()
    state["module_outputs"]["specialization_critic_module"] = {
        "reviewed_schema_revision": 1,
        "patch_operations": [
            {
                "op": "add_field",
                "field_path": "material_info.section1.stale_field",
                "field": {"data_type": "string"},
            }
        ],
    }

    repaired = graph.deterministic_schema_repair(state)
    paths = {
        item["field_path"]
        for item in repaired["module_outputs"]["schema_design_module"]["field_registry"]
    }

    assert "material_info.section1.stale_field" not in paths
    assert "material_info.section1.coercive_field" in paths


def test_aggregation_rejects_critic_from_previous_schema_revision(monkeypatch):
    state = base_state()
    state["args"] = {"use_llm_aggregation": False, "output": "unused.json"}
    state["module_outputs"]["specialization_critic_module"] = {
        "specialization_status": "pass",
        "reviewed_schema_revision": 1,
    }
    monkeypatch.setattr(
        graph,
        "write_state_snapshot",
        lambda state, current_node, next_node=None: {
            **state,
            "current_node": current_node,
            "next_node": next_node,
        },
    )

    result = graph.aggregation_node(state)

    assert result["next_node"] == "supervisor_router"
    assert result["last_error_node"] == "specialization_critic"
    assert result["validation_errors"][0].startswith("schema_revision_mismatch:")


def test_module_attempt_history_is_appended_across_supervisor_retries(monkeypatch):
    state = base_state()
    state["args"] = {
        "base_url": "https://example.invalid/v1",
        "api_key": "secret",
        "llm_backend": "openai",
        "request_timeout": 30,
        "model": "test",
        "temperature": 0,
        "max_retries": 0,
        "output": "unused.json",
    }
    state["module_attempts"] = {
        "schema_design_module": [{"round": 1, "raw_response": "old"}]
    }
    monkeypatch.setattr(graph.section_agent, "get_client", lambda **kwargs: object())
    monkeypatch.setattr(
        graph.section_agent,
        "call_module",
        lambda *args, **kwargs: (
            state["module_outputs"]["schema_design_module"],
            [],
            [{"round": 1, "raw_response": "new"}],
        ),
    )
    monkeypatch.setattr(graph, "write_state_snapshot", lambda state, *args: state)

    result = graph.call_module_node(
        state,
        "schema_design_module",
        "prompt",
        "schema_design",
        "specialization_critic",
    )

    attempts = result["module_attempts"]["schema_design_module"]
    assert [item["round"] for item in attempts] == [1, 2]
    assert [item["raw_response"] for item in attempts] == ["old", "new"]


def test_critic_prompt_is_compact_and_requires_structured_repairs():
    huge_text = "LONG_REFERENCE_SENTINEL " * 20000
    schema = {
        "top_level_keys": [{"key": "material_info"}],
        "field_registry": [
            {
                **field(f"material_info.section1.property_{index}"),
                "description": huge_text,
                "reason": huge_text,
            }
            for index in range(240)
        ],
    }

    prompt = prompts.build_specialization_critic_prompt(
        {
            "database_goal": "materials database",
            "discipline": "materials science",
            "query_requirements": ["compare properties"],
            "reference_paper_context": huge_text,
            "reference_field_contract": {"available": False},
        },
        {},
        {},
        {},
        {},
        {},
        {},
        schema,
    )

    assert "patch_operations" in prompt
    assert "field_utility_audit" in prompt
    assert "Do not infer a numeric ideal" in prompt
    assert "`section5.*` is the canonical material-record-owned" in prompt
    assert "Do not replace independently queryable task-specific scientific quantities" in prompt
    assert "`fields` must be a JSON object" in prompt
    assert "LONG_REFERENCE_SENTINEL" not in prompt
    assert len(prompt) < 180000


def test_critic_prompt_deduplicates_contracts_without_dropping_field_paths():
    shared_object_contract = {
        "object_kind": "measurement",
        "required_subfields": ["value", "unit", "conditions"],
        "fields": {"marker": "SHARED_OBJECT_CONTRACT_SENTINEL"},
    }
    shared_figure_constraint = {
        "allowed_sections": ["Results"],
        "allowed_figure_categories": ["property_plot"],
        "why_needed": "SHARED_FIGURE_CONSTRAINT_SENTINEL",
    }
    schema = {
        "top_level_keys": [{"key": "material_info"}],
        "field_registry": [
            {
                **field(
                    f"material_info.section1.property_{index}",
                    data_type="object",
                ),
                "object_contract": shared_object_contract,
                "figure_constraint": shared_figure_constraint,
            }
            for index in range(240)
        ],
    }

    prompt = prompts.build_specialization_critic_prompt(
        {
            "database_goal": "materials database",
            "discipline": "materials science",
            "query_requirements": ["compare properties"],
            "reference_field_contract": {"available": False},
        },
        {},
        {},
        {},
        {},
        {},
        {},
        schema,
    )

    assert "material_info.section1.property_0" in prompt
    assert "material_info.section1.property_239" in prompt
    assert prompt.count("SHARED_OBJECT_CONTRACT_SENTINEL") == 1
    assert prompt.count("SHARED_FIGURE_CONSTRAINT_SENTINEL") == 1
    assert '"object_contract_ref":"object_contract_001"' in prompt
    assert '"figure_constraint_ref":"figure_constraint_001"' in prompt


def test_critic_prompt_includes_path_whitelist_and_retry_feedback():
    valid_path = "material_info.section1.coercive_field"
    invalid_path = "material_info.section1.coercive_field_alias"
    feedback = [
        "specialization_critic field_utility_audit references unknown fields: "
        + invalid_path,
        "specialization_critic utility findings lack executable patch operations: "
        + valid_path,
    ]

    prompt = prompts.build_specialization_critic_prompt(
        {
            "database_goal": "materials database",
            "discipline": "materials science",
            "query_requirements": ["compare coercive fields"],
            "reference_field_contract": {"available": False},
        },
        {},
        {},
        {},
        {},
        {},
        {},
        {
            "top_level_keys": [{"key": "material_info"}],
            "field_registry": [field(valid_path)],
        },
        repair_feedback=feedback,
    )

    assert '"field_path_whitelist":["material_info.section1.coercive_field"]' in prompt
    assert invalid_path in prompt
    assert "Supervisor validation feedback from the immediately preceding critic attempt" in prompt
    assert "copied exactly from that whitelist" in prompt
    assert "Correct every listed protocol error" in prompt
    assert "concept_id nested only inside object_contract.fields does not satisfy" in prompt
    assert "object_contract_ref and figure_constraint_ref are read-only" in prompt
    assert "Never place bare fields or required_subfields" in prompt
    assert "add_field on an existing path is invalid" in prompt


def test_critic_node_feeds_supervisor_validation_errors_into_retry_prompt(monkeypatch):
    state = base_state()
    for module_name in (
        "mechanism_requirement_module",
        "query_semantics_module",
        "evidence_model_module",
        "section_partition_module",
    ):
        state["module_outputs"][module_name] = {}
    state["repair_instructions"] = [
        "specialization_critic field_utility_audit references unknown fields: bad.path"
    ]
    captured = {}

    def fake_build(*args, **kwargs):
        captured["repair_feedback"] = kwargs.get("repair_feedback")
        return "critic prompt"

    def fake_call(current_state, *args, **kwargs):
        current_state["module_outputs"]["specialization_critic_module"] = {
            "specialization_status": "pass",
            "is_generic": False,
            "missing_concepts": [],
            "structural_weaknesses": [],
            "redo_needed": False,
            "redo_directives": [],
            "field_utility_audit": utility_audit(1),
            "patch_operations": [],
        }
        current_state["validation_errors"] = []
        return current_state

    monkeypatch.setattr(graph, "build_specialization_critic_prompt", fake_build)
    monkeypatch.setattr(graph, "call_module_node", fake_call)

    def fake_snapshot(current_state, current_node, next_node):
        current_state["current_node"] = current_node
        current_state["next_node"] = next_node
        return current_state

    monkeypatch.setattr(graph, "write_state_snapshot", fake_snapshot)

    result = graph.specialization_critic_node(state)

    assert captured["repair_feedback"] == state["repair_instructions"]
    assert result["next_node"] == "aggregation"


def test_supervisor_applies_current_revision_critic_patch_before_figure_rerun(monkeypatch):
    state = base_state()
    state.update(
        {
            "args": {
                "max_supervisor_retries": 10,
                "max_total_supervisor_repairs": 60,
            },
            "validation_errors": [
                "material_info.section1.coercive_field: missing figure_constraint",
                "critic supplied pending structured patch_operations",
            ],
            "last_error_node": "schema_design",
            "last_error_type": "schema_validation_failure",
            "retry_counts": {},
            "blocker_history": [],
        }
    )
    state["module_outputs"]["specialization_critic_module"] = {
        "reviewed_schema_revision": state["schema_revision"],
        "patch_operations": [
            {
                "op": "remove_field",
                "field_path": "material_info.section1.coercive_field",
                "reason": "Remove a confirmed duplicate before reevaluating figure coverage.",
            }
        ],
    }

    def fake_snapshot(current_state, current_node, next_node):
        current_state["current_node"] = current_node
        current_state["next_node"] = next_node
        return current_state

    monkeypatch.setattr(graph, "write_state_snapshot", fake_snapshot)

    result = graph.supervisor_router_node(state)

    assert result["supervisor_decision"]["action"] == "repair_schema_design"
    assert result["next_node"] == "schema_design_repair"


def test_supervisor_routes_invalid_current_critic_patch_back_to_critic(monkeypatch):
    state = base_state()
    state.update(
        {
            "args": {
                "max_supervisor_retries": 10,
                "max_total_supervisor_repairs": 60,
            },
            "validation_errors": [
                "schema_patch_rejected:patch[0] add_field already exists: "
                "material_info.section1.coercive_field"
            ],
            "last_error_node": "schema_design",
            "last_error_type": "schema_validation_failure",
            "retry_counts": {},
            "blocker_history": [],
            "rejected_critic_patch_hashes": [],
        }
    )
    operations = [
        {
            "op": "add_field",
            "field_path": "material_info.section1.coercive_field",
            "field": {"data_type": "string"},
        }
    ]
    state["module_outputs"]["specialization_critic_module"] = {
        "reviewed_schema_revision": state["schema_revision"],
        "patch_operations": operations,
    }

    def fake_snapshot(current_state, current_node, next_node):
        current_state["current_node"] = current_node
        current_state["next_node"] = next_node
        return current_state

    monkeypatch.setattr(graph, "write_state_snapshot", fake_snapshot)

    result = graph.supervisor_router_node(state)
    rejected_hash = graph.critic_patch_hash(operations)

    assert result["supervisor_decision"]["action"] == "retry_critic_patch"
    assert result["next_node"] == "specialization_critic"
    assert result["repair_instructions"] == state["validation_errors"]
    assert rejected_hash in result["rejected_critic_patch_hashes"]
    assert not graph.has_current_revision_critic_patch(
        result,
        "schema_design",
        "schema_validation_failure",
    )


def test_specialization_critic_rejects_invalid_patch_before_schema_repair(monkeypatch):
    state = base_state()
    for module_name in (
        "mechanism_requirement_module",
        "query_semantics_module",
        "evidence_model_module",
        "section_partition_module",
    ):
        state["module_outputs"][module_name] = {}
    invalid_critic = {
        "specialization_status": "needs_redesign",
        "is_generic": False,
        "missing_concepts": ["structured uncertainty"],
        "structural_weaknesses": [],
        "redo_needed": True,
        "redo_directives": ["Represent uncertainty as an object."],
        "field_utility_audit": utility_audit(1),
        "patch_operations": [
            {
                "op": "update_field",
                "field_path": "material_info.section1.coercive_field",
                "changes": {
                    "data_type": "object",
                    "required_subfields": ["value", "unit"],
                },
            }
        ],
    }

    def fake_call(current_state, *args, **kwargs):
        outputs = dict(current_state["module_outputs"])
        outputs["specialization_critic_module"] = invalid_critic
        return {
            **current_state,
            "module_outputs": outputs,
            "validation_errors": [],
        }

    def fake_snapshot(current_state, current_node, next_node):
        current_state["current_node"] = current_node
        current_state["next_node"] = next_node
        return current_state

    monkeypatch.setattr(graph, "build_specialization_critic_prompt", lambda *a, **k: "prompt")
    monkeypatch.setattr(graph, "call_module_node", fake_call)
    monkeypatch.setattr(graph, "write_state_snapshot", fake_snapshot)

    result = graph.specialization_critic_node(state)

    assert result["next_node"] == "supervisor_router"
    assert result["last_error_node"] == "specialization_critic"
    assert result["validation_errors"] == [
        "schema_patch_rejected:patch[0] changes unsupported keys: ['required_subfields']"
    ]
    patch_hash = graph.critic_patch_hash(invalid_critic["patch_operations"])
    assert patch_hash in result["rejected_critic_patch_hashes"]
    assert (
        result["module_outputs"]["specialization_critic_module"]["patch_validation"]["status"]
        == "rejected"
    )


def test_supervisor_still_routes_pure_figure_error_upstream(monkeypatch):
    state = base_state()
    state.update(
        {
            "args": {
                "max_supervisor_retries": 10,
                "max_total_supervisor_repairs": 60,
            },
            "validation_errors": [
                "material_info.section1.coercive_field: missing figure_constraint"
            ],
            "last_error_node": "schema_design",
            "last_error_type": "schema_validation_failure",
            "retry_counts": {},
            "blocker_history": [],
        }
    )

    def fake_snapshot(current_state, current_node, next_node):
        current_state["current_node"] = current_node
        current_state["next_node"] = next_node
        return current_state

    monkeypatch.setattr(graph, "write_state_snapshot", fake_snapshot)

    result = graph.supervisor_router_node(state)

    assert result["supervisor_decision"]["action"] == "repair_coverage"
    assert result["next_node"] == "figure_classification"


def test_critic_redesign_requires_executable_patch_operations():
    errors = agent.validate_module_result(
        "specialization_critic_module",
        {
            "specialization_status": "needs_redesign",
            "is_generic": True,
            "missing_concepts": ["magnetic order"],
            "structural_weaknesses": [],
            "redo_needed": True,
            "redo_directives": ["add magnetic order"],
            "field_utility_audit": utility_audit(1),
            "patch_operations": [],
        },
    )

    assert any("must be non-empty" in error for error in errors)


def test_critic_utility_audit_must_match_exact_schema_and_cover_findings():
    schema = {
        "field_registry": [
            field(
                "material_info.section1.coercive_field",
                concept_ids=["coercive_field"],
            )
        ]
    }
    critic = {
        "field_utility_audit": utility_audit(
            2,
            decision="needs_pruning",
            redundant_fields=["material_info.section1.coercive_field"],
        ),
        "patch_operations": [],
    }

    errors = agent.validate_critic_field_utility(critic, schema)

    assert any("reviewed_field_count does not match" in error for error in errors)
    assert any("lack executable patch operations" in error for error in errors)


def test_critic_utility_audit_accepts_issue_with_structured_removal():
    field_path = "material_info.section1.coercive_field_alias"
    schema = {"field_registry": [field(field_path, concept_ids=["coercive_field"])]}
    critic = {
        "field_utility_audit": utility_audit(
            1,
            decision="needs_pruning",
            alias_or_unit_variant_fields=[field_path],
        ),
        "patch_operations": [
            {
                "op": "remove_field",
                "field_path": field_path,
                "reason": "Alias belongs in normalization metadata.",
            }
        ],
    }

    assert agent.validate_critic_field_utility(critic, schema) == []


def test_exact_semantic_duplicates_are_blocking_without_a_field_count_cap():
    first = field(
        "material_info.section1.coercive_field_reported",
        concept_ids=["coercive_field"],
    )
    second = {
        **first,
        "field_path": "material_info.section1.coercive_field_duplicate",
        "field_name": "coercive_field_duplicate",
    }
    result = {
        "schema_definition": {
            "top_level_keys": [{"key": "material_info"}],
            "field_registry": [first, second],
        },
        "requirement_contract": {
            "concepts": [
                {
                    "concept_id": "coercive_field",
                    "required": True,
                    "source_requirement_ids": ["query_1"],
                }
            ],
            "source_requirements": [
                {"requirement_id": "query_1", "text": "record coercive field"}
            ],
        },
        "entity_registry": [],
    }

    report = agent.build_coverage_report(result)
    errors = agent.validate_coverage_report(report)

    assert report["semantic_redundancy"]["redundant_field_count"] == 1
    assert any(error.startswith("coverage:redundant_fields:") for error in errors)


def test_common_slots_under_distinct_scientific_quantities_are_not_duplicates():
    result = {
        "schema_definition": {
            "top_level_keys": [{"key": "material_info"}],
            "field_registry": [
                field(
                    "material_info.section1.lower_critical_field.value",
                    concept_ids=["critical_field"],
                ),
                field(
                    "material_info.section1.upper_critical_field.value",
                    concept_ids=["critical_field"],
                ),
                field(
                    "material_info.section1.lower_critical_field.evidence_record",
                    concept_ids=["critical_field"],
                ),
                field(
                    "material_info.section1.upper_critical_field.evidence_record",
                    concept_ids=["critical_field"],
                ),
            ],
        },
        "requirement_contract": {
            "concepts": [
                {
                    "concept_id": "critical_field",
                    "required": True,
                    "source_requirement_ids": ["query_1"],
                }
            ],
            "source_requirements": [
                {"requirement_id": "query_1", "text": "record critical fields"}
            ],
        },
        "entity_registry": [],
    }

    report = agent.build_coverage_report(result)

    assert report["semantic_redundancy"]["redundant_field_count"] == 0
    assert report["semantic_redundancy"]["redundant_fields"] == []


def test_schema_module_has_no_numeric_field_count_floor():
    result = {
        "top_level_keys": [{"key": "material_info"}],
        "field_registry": [
            field(
                "material_info.section1.task_specific_property",
                concept_ids=["task_specific_property"],
            )
        ],
    }

    errors = agent.validate_module_result("schema_design_module", result)

    assert not any("enough materials-database fields" in error for error in errors)
    assert not any("no materials-database fields" in error for error in errors)


def test_thousand_unjustified_fields_are_blocked_by_utility_not_a_numeric_cap():
    fields = [
        field(
            f"material_info.section1.speculative_property_{index}",
            concept_ids=[],
        )
        for index in range(1000)
    ]
    result = {
        "schema_definition": {
            "top_level_keys": [{"key": "material_info"}],
            "field_registry": fields,
        },
        "requirement_contract": {"concepts": [], "source_requirements": []},
        "entity_registry": [],
    }

    report = agent.build_coverage_report(result)
    errors = agent.validate_coverage_report(report)

    assert report["unrelated_domain_field_count"] == 1000
    assert any(error.startswith("coverage:unmapped_domain_fields:") for error in errors)
    assert not any("field_count" in error or "field count" in error for error in errors)


def test_redundancy_repair_removes_only_the_reported_duplicate():
    state = base_state()
    original = state["module_outputs"]["schema_design_module"]["field_registry"][0]
    duplicate = {
        **original,
        "field_path": "material_info.section1.coercive_field_duplicate",
        "field_name": "coercive_field_duplicate",
    }
    state["module_outputs"]["schema_design_module"]["field_registry"].append(duplicate)
    state["repair_instructions"] = [
        "coverage:redundant_fields:material_info.section1.coercive_field_duplicate"
    ]

    repaired = graph.deterministic_schema_repair(state)
    paths = {
        item["field_path"]
        for item in repaired["module_outputs"]["schema_design_module"]["field_registry"]
    }

    assert "material_info.section1.coercive_field" in paths
    assert "material_info.section1.coercive_field_duplicate" not in paths


def test_critic_patch_cannot_drop_an_internal_required_concept_without_replacement():
    state = base_state()
    operations = [
        {
            "op": "remove_field",
            "field_path": "material_info.section1.coercive_field",
            "reason": "critic considered the field redundant",
        }
    ]

    rejected = graph.apply_schema_patch_operations(state, operations)

    assert rejected["validation_errors"] == [
        "schema_patch_contract_regression:required_concepts:coercive_field"
    ]
    assert graph.critic_patch_hash(operations) in rejected["rejected_critic_patch_hashes"]


def test_critic_patch_may_replace_a_required_concept_in_the_same_patch():
    state = base_state()
    replacement_path = "material_info.section1.coercive_field_measurement"
    patched = graph.apply_schema_patch_operations(
        state,
        [
            {
                "op": "remove_field",
                "field_path": "material_info.section1.coercive_field",
                "reason": "replace an ambiguous field",
            },
            {
                "op": "add_field",
                "field_path": replacement_path,
                "field": field(replacement_path, concept_ids=["coercive_field"]),
                "reason": "retain the required capability with clearer semantics",
            },
        ],
    )

    assert patched["validation_errors"] == []
    assert replacement_path in {
        item["field_path"]
        for item in patched["module_outputs"]["schema_design_module"]["field_registry"]
    }


def test_rejected_contract_patch_is_not_reapplied_by_schema_repair():
    state = base_state()
    operations = [
        {
            "op": "remove_field",
            "field_path": "material_info.section1.coercive_field",
            "reason": "remove without replacement",
        }
    ]
    state["module_outputs"]["specialization_critic_module"] = {
        "reviewed_schema_revision": state["schema_revision"],
        "patch_operations": operations,
    }
    state["rejected_critic_patch_hashes"] = [graph.critic_patch_hash(operations)]

    repaired = graph.deterministic_schema_repair(state)

    assert repaired["last_error_type"] == "schema_model_repair_required"
    assert repaired["schema_revision"] == state["schema_revision"]
    assert repaired["module_outputs"]["schema_design_module"] == (
        state["module_outputs"]["schema_design_module"]
    )


def test_critic_contract_regression_is_rejected_before_supervisor_applies_it(monkeypatch):
    state = base_state()
    for module_name in (
        "mechanism_requirement_module",
        "query_semantics_module",
        "evidence_model_module",
        "section_partition_module",
    ):
        state["module_outputs"][module_name] = {}
    field_path = "material_info.section1.coercive_field"
    operations = [
        {
            "op": "remove_field",
            "field_path": field_path,
            "reason": "critic considered it redundant",
        }
    ]
    critic = {
        "specialization_status": "needs_redesign",
        "is_generic": False,
        "missing_concepts": [],
        "structural_weaknesses": ["one redundant field"],
        "redo_needed": True,
        "redo_directives": ["remove the redundant field"],
        "field_utility_audit": utility_audit(
            1,
            decision="needs_pruning",
            redundant_fields=[field_path],
        ),
        "patch_operations": operations,
    }

    def fake_call(current, *args, **kwargs):
        return {
            **current,
            "module_outputs": {
                **current["module_outputs"],
                "specialization_critic_module": critic,
            },
            "validation_errors": [],
        }

    monkeypatch.setattr(graph, "call_module_node", fake_call)
    monkeypatch.setattr(graph, "build_specialization_critic_prompt", lambda *a, **k: "prompt")
    monkeypatch.setattr(
        graph,
        "write_state_snapshot",
        lambda current, current_node, next_node: {
            **current,
            "current_node": current_node,
            "next_node": next_node,
        },
    )

    result = graph.specialization_critic_node(state)

    assert result["last_error_node"] == "specialization_critic"
    assert result["validation_errors"] == [
        "schema_patch_contract_regression:required_concepts:coercive_field"
    ]
    assert graph.critic_patch_hash(operations) in result["rejected_critic_patch_hashes"]


def test_missing_concept_repair_uses_current_schema_model_prompt_not_field_plan(monkeypatch):
    state = base_state()
    state.update(
        {
            "last_error_type": "schema_validation_failure",
            "repair_instructions": ["coverage:missing_concepts:critical_temperature"],
            "validation_errors": [],
        }
    )
    called = {}

    monkeypatch.setattr(graph, "build_schema_repair_prompt", lambda current: "repair-current-schema")

    def fake_call(current, module_name, prompt, current_node, next_node):
        called.update(
            module_name=module_name,
            prompt=prompt,
            current_node=current_node,
            next_node=next_node,
        )
        return {**current, "validation_errors": [], "next_node": next_node}

    monkeypatch.setattr(graph, "call_module_node", fake_call)
    monkeypatch.setattr(
        graph,
        "write_state_snapshot",
        lambda current, current_node, next_node: {
            **current,
            "current_node": current_node,
            "next_node": next_node,
        },
    )

    result = graph.schema_design_repair_node(state)

    assert called == {
        "module_name": "schema_design_module",
        "prompt": "repair-current-schema",
        "current_node": "schema_design_repair",
        "next_node": "specialization_critic",
    }
    assert result["next_node"] == "specialization_critic"


def test_supervisor_pauses_at_each_ten_revision_inspection_boundary(monkeypatch):
    state = base_state()
    state.update(
        {
            "args": {
                "max_supervisor_retries": 20,
                "max_total_supervisor_repairs": 100,
                "schema_inspection_interval": 10,
            },
            "schema_revision": 10,
            "validation_errors": ["coverage:missing_concepts:critical_temperature"],
            "last_error_node": "aggregation",
            "last_error_type": "schema_validation_failure",
            "retry_counts": {},
            "blocker_history": [],
        }
    )
    monkeypatch.setattr(
        graph,
        "write_state_snapshot",
        lambda current, current_node, next_node: {
            **current,
            "current_node": current_node,
            "next_node": next_node,
        },
    )

    result = graph.supervisor_router_node(state)

    assert result["status"] == "needs_review"
    assert result["next_node"] == "write_output"
    assert result["supervisor_decision"]["action"] == "schema_inspection_required"
    assert result["supervisor_decision"]["inspection_revision"] == 10
    assert "critic strictness" in result["supervisor_decision"]["reason"]
    assert any(
        "too strict" in item
        for item in result["supervisor_decision"]["inspection_summary"][
            "inspection_requirements"
        ]
    )


def test_successful_tenth_schema_revision_pauses_before_critic(monkeypatch):
    state = base_state()
    state.update(
        {
            "args": {"schema_inspection_interval": 10},
            "schema_revision": 9,
            "schema_inspection_approved_through": 0,
            "force_deterministic_schema_compilation": True,
        }
    )
    monkeypatch.setattr(
        graph,
        "write_state_snapshot",
        lambda current, current_node, next_node: {
            **current,
            "current_node": current_node,
            "next_node": next_node,
        },
    )

    result = graph.schema_design_node(state)

    assert result["schema_revision"] == 10
    assert result["next_node"] == "supervisor_router"
    assert result["schema_inspection_resume_node"] == "specialization_critic"
    assert result["validation_errors"] == [
        "schema_inspection_boundary_reached:10"
    ]


def test_approved_immediate_schema_boundary_resumes_at_critic():
    state = base_state()
    state.update(
        {
            "args": {"schema_inspection_interval": 10},
            "schema_revision": 10,
            "schema_inspection_approved_through": 10,
            "schema_inspection_resume_node": "specialization_critic",
            "validation_errors": ["schema_inspection_boundary_reached:10"],
            "repair_instructions": [],
            "supervisor_decision": {
                "action": "schema_inspection_required",
                "inspection_revision": 10,
                "next_node": "write_output",
            },
        }
    )

    resumed = graph.resume_router_node(state)

    assert resumed["status"] == "running"
    assert resumed["next_node"] == "specialization_critic"
    assert resumed["validation_errors"] == []
    assert resumed["repair_instructions"] == []


def test_supervisor_pauses_after_three_identical_blockers_without_revision(monkeypatch):
    state = base_state()
    errors = [
        "schema_patch_rejected:patch[29] target does not exist: section5.mechanism_interpretation"
    ]
    fingerprint = graph.blocker_fingerprint(
        "schema_design", "schema_validation_failure", errors
    )
    state.update(
        {
            "args": {
                "max_supervisor_retries": 10,
                "max_total_supervisor_repairs": 60,
                "schema_inspection_interval": 10,
                "repeated_blocker_inspection_threshold": 3,
            },
            "schema_revision": 1,
            "validation_errors": errors,
            "last_error_node": "schema_design",
            "last_error_type": "schema_validation_failure",
            "retry_counts": {},
            "blocker_history": [
                {"fingerprint": fingerprint},
                {"fingerprint": fingerprint},
            ],
        }
    )
    monkeypatch.setattr(
        graph,
        "write_state_snapshot",
        lambda current, current_node, next_node: {
            **current,
            "current_node": current_node,
            "next_node": next_node,
        },
    )

    result = graph.supervisor_router_node(state)

    decision = result["supervisor_decision"]
    assert result["status"] == "needs_review"
    assert decision["action"] == "repeated_blocker_inspection_required"
    assert decision["repeated_blocker_count"] == 3
    assert decision["inspection_summary"]["strictness_review_required"] is True
    assert any(
        "too strict" in item
        for item in decision["inspection_summary"]["inspection_requirements"]
    )


def test_approved_tenth_revision_resumes_and_next_gate_is_twenty(monkeypatch):
    state = base_state()
    state.update(
        {
            "args": {"schema_inspection_interval": 10},
            "schema_revision": 10,
            "schema_inspection_approved_through": 10,
            "validation_errors": ["coverage:missing_concepts:critical_temperature"],
            "repair_instructions": ["coverage:missing_concepts:critical_temperature"],
            "last_error_node": "aggregation",
            "supervisor_decision": {
                "action": "schema_inspection_required",
                "inspection_revision": 10,
                "next_node": "write_output",
            },
        }
    )

    resumed = graph.resume_router_node(state)

    assert resumed["status"] == "running"
    assert resumed["next_node"] == "schema_design_repair"
    assert resumed["supervisor_decision"]["action"] == "schema_inspection_approved"

    resumed.update(
        {
            "schema_revision": 19,
            "validation_errors": ["coverage:missing_concepts:critical_temperature"],
            "last_error_type": "schema_validation_failure",
            "retry_counts": {},
            "blocker_history": [],
        }
    )
    monkeypatch.setattr(
        graph,
        "write_state_snapshot",
        lambda current, current_node, next_node: {
            **current,
            "current_node": current_node,
            "next_node": next_node,
        },
    )
    before_boundary = graph.supervisor_router_node(resumed)
    assert before_boundary["next_node"] == "schema_design_repair"

    before_boundary.update(
        {
            "schema_revision": 20,
            "validation_errors": ["coverage:missing_concepts:critical_temperature"],
        }
    )
    at_boundary = graph.supervisor_router_node(before_boundary)
    assert at_boundary["supervisor_decision"]["action"] == "schema_inspection_required"
    assert at_boundary["supervisor_decision"]["inspection_revision"] == 20
