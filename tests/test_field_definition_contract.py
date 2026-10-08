import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, os.environ.get("FIELD_DEFINITION_CODE_DIR", str(Path(__file__).resolve().parents[1] / "code")))
import field_definition_contract as contract
import section_design_agent as agent
import section_design_langgraph_human_gate as graph
import step9_extraction_build_graph as step9


def definition(path="material_info.section1.yield_strength[].value"):
    return {
        "field_path": path,
        "description": "Stress at the explicitly reported yield criterion; excludes ultimate tensile strength.",
        "extraction_notes": "Bind specimen, temperature, strain rate and yield criterion; preserve reported stress units.",
        "data_type": "number", "required": False, "source_basis": ["text", "table"],
        "inclusion_rule": "Populate only an explicitly reported yield stress.",
        "absence_rule": "Use missing for unreported stress; unresolved for ambiguous criterion.",
        "relation_constraints": {"entity_binding_required": True, "condition_binding_required": True, "separate_instances": True},
        "evidence_requirements": {"direct_support_required": True, "locator_required": True, "allowed_source_types": ["text", "table"]},
    }


def plan():
    item = definition()
    return {"field_definition_contract_version": contract.VERSION, "field_groups": [{
        "section_id": "material_info.section1", "group_name": "mechanical properties",
        "purpose": "Compare mechanical response", "evidence_strategy": "text and table",
        "recommended_fields": [item["field_path"]], "field_definitions": [item],
    }]}


def test_complete_definition_validates():
    assert contract.validate_definitions(plan()) == []


@pytest.mark.parametrize("key", contract.TEXT_KEYS)
def test_missing_semantics_rejected(key):
    candidate = plan()
    candidate["field_groups"][0]["field_definitions"][0].pop(key)
    assert any(key in error for error in contract.validate_definitions(candidate))


def test_missing_extra_and_duplicate_paths_rejected():
    candidate = plan()
    candidate["field_groups"][0]["field_definitions"].append(definition("material_info.section1.extra.value"))
    assert any("exactly match" in error for error in contract.validate_definitions(candidate))
    candidate = plan()
    candidate["field_groups"][0]["field_definitions"].append(definition())
    assert any("duplicate" in error for error in contract.validate_definitions(candidate))


def test_group_purpose_and_lost_multiplicity_rejected():
    candidate = plan()
    item = candidate["field_groups"][0]["field_definitions"][0]
    item["description"] = candidate["field_groups"][0]["purpose"]
    item["relation_constraints"]["separate_instances"] = False
    errors = contract.validate_definitions(candidate)
    assert any("own scientific definition" in error for error in errors)
    assert any("repeated ancestor" in error for error in errors)


def test_legacy_plan_remains_readable_but_fresh_run_requires_definitions(monkeypatch):
    monkeypatch.delenv("FIELD_DEFINITIONS_REQUIRED", raising=False)
    assert contract.validate_definitions({"field_groups": []}) == []
    monkeypatch.setenv("FIELD_DEFINITIONS_REQUIRED", "1")
    assert contract.validate_definitions({"field_groups": []})


@pytest.mark.parametrize("compiler_name", ["compile_schema_from_field_planning", "rebuild_material_schema_from_field_planning"])
def test_definition_survives_compilation_and_extraction_handoff(compiler_name):
    if not hasattr(graph, compiler_name):
        pytest.skip("This maintained version has no deterministic compiler")
    compiler = getattr(graph, compiler_name)
    planned = plan()
    state = {"shared_context": {}, "module_outputs": {
        "field_planning_module": planned,
        "subjective_supervisor_module": {"requirement_contract": {"concepts": []}, "entity_registry": []},
    }, "module_attempts": {}, "module_errors": {}}
    result = compiler(state, "definition regression")
    registry = result["module_outputs"]["schema_design_module"]["field_registry"]
    field = next(row for row in registry if row["field_path"] == contract.path_key(definition()["field_path"]))
    for key in contract.TEXT_KEYS:
        assert field[key] == definition()[key]
    assert "object_contract" not in field
    assert field["relation_constraints"]["repeatable_ancestors"] == ["material_info.section1.yield_strength[]"]
    output = {"result": {"schema_definition": {"field_registry": registry}}}
    if not hasattr(agent, "ensure_literature_field_rule_contract"):
        return
    agent.ensure_literature_field_rule_contract(output["result"], registry)
    package = step9.build_prompt_package_from_step8(output, {"section_test_plan": [{
        "stage_id": "material_info.section1", "section_id": "material_info.section1", "focus": "mechanical response",
    }]})
    handed = next(row for row in package["shared_prompt_context"]["field_index"] if row["field_path"] == field["field_path"])
    for key in ("description", "extraction_notes", "relation_constraints", "evidence_requirements", "data_type"):
        assert handed[key] == field[key]


def test_nested_component_ancestors_retained():
    candidate = plan()
    item = definition("material_info.section0.stacks[].components[].thickness")
    candidate["field_groups"][0].update(recommended_fields=[item["field_path"]], field_definitions=[item])
    registry = [{"field_path": contract.path_key(item["field_path"])}]
    contract.apply_definitions(registry, candidate)
    assert registry[0]["relation_constraints"]["repeatable_ancestors"] == [
        "material_info.section0.stacks[]", "material_info.section0.stacks[].components[]",
    ]


def test_compiler_cannot_drop_a_defined_field():
    with pytest.raises(ValueError, match="dropped planned fields"):
        contract.apply_definitions([], plan())


def test_oversized_definition_is_rejected():
    candidate = plan()
    candidate["field_groups"][0]["field_definitions"][0]["description"] = "x" * 1201
    assert any("exceeds 1200" in error for error in contract.validate_definitions(candidate))
