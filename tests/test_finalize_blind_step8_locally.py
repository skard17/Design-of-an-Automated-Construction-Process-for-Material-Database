import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import finalize_blind_step8_locally as finalizer


def test_fallback_source_does_not_reference_evaluation_gold():
    source = Path(finalizer.__file__).read_text(encoding="utf-8")
    assert "schema2_store" not in source
    assert "step9_gold" not in source
    assert "core_field" not in source


def test_structured_identifier_normalizes_camel_case_entity_names():
    assert finalizer.structured_identifier("CharacterizationEvent") == "characterization_event"


def test_relationship_validator_rejects_union_and_unknown_endpoints():
    result = {
        "entity_registry": [{"entity_id": "sample"}, {"entity_id": "observation"}],
        "relationship_contracts": [
            {
                "relationship_id": "bad_union",
                "source_entity": "Sample|MaterialSystem",
                "target_entity": "Observation",
            },
            {
                "relationship_id": "bad_target",
                "source_entity": "Sample",
                "target_entity": "MissingEntity",
            },
        ],
    }

    errors = finalizer.structured_relationship_errors(result)

    assert "relationship:bad_union:non_atomic_endpoint" in errors
    assert "relationship:bad_target:unknown_target:missing_entity" in errors


def test_fallback_requires_nonempty_compiled_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(
        finalizer.graph,
        "compile_schema_from_field_planning",
        lambda state, reason: {**state, "module_outputs": {"schema_design_module": {"field_registry": []}}},
    )
    try:
        finalizer.finalize({"module_outputs": {}}, tmp_path / "out.json", "test")
    except ValueError as exc:
        assert "empty schema" in str(exc)
    else:
        raise AssertionError("empty schema must be rejected")


def test_fallback_is_marked_incomplete_for_online_scoring(monkeypatch, tmp_path):
    result = {
        "schema_definition": {
            "field_registry": [
                {"field_path": "material_info.section0.material_id", "data_type": "string"}
            ]
        }
    }
    prepared = {
        "module_outputs": {
            "locating_module": {},
            "query_semantics_module": {},
            "subjective_supervisor_module": {},
            "topic_adaptation_module": {},
            "section_partition_module": {},
            "field_planning_module": {},
            "supervisor_module": {},
            "figure_classification_module": {},
            "schema_design_module": result["schema_definition"],
        },
        "shared_context": {},
        "args": {},
    }
    monkeypatch.setattr(finalizer, "hydrate_checkpoint_state", lambda state, output: prepared)
    monkeypatch.setattr(finalizer, "ensure_deterministic_planning_modules", lambda state, **kwargs: state)
    monkeypatch.setattr(finalizer.graph, "compile_schema_from_field_planning", lambda state, reason: state)
    monkeypatch.setattr(finalizer.section_agent, "assemble_final_result_from_modules", lambda *args: result)
    monkeypatch.setattr(finalizer.section_agent, "finalize_result", lambda shared, value: value)
    monkeypatch.setattr(finalizer.section_agent, "validate_result", lambda value: [])
    monkeypatch.setattr(finalizer.section_agent, "validate_specialization", lambda *args: [])
    monkeypatch.setattr(finalizer.graph, "write_output_node", lambda state: None)

    outcome = finalizer.finalize({}, tmp_path / "out.json", "timeout")

    assert outcome["status"] == "fallback_needs_online_completion"
    assert prepared["execution_provenance"]["fallback_used"] is True
    assert prepared["execution_provenance"]["official_benchmark_eligible"] is False
    assert outcome["official_benchmark_eligible"] is False


def test_blind_mode_removes_placeholder_reference_contract():
    state = {
        "shared_context": {
            "reference_field_contract": {
                "available": True,
                "leaf_fields": [{"path": "schema_policy"}],
            }
        },
        "module_outputs": {
            "field_planning_module": {
                "reference_field_audit": {"unresolved_leaf_paths": ["schema_policy"]}
            }
        },
    }

    cleaned = finalizer.remove_reference_field_contract(state)

    assert not cleaned["shared_context"]["reference_field_contract"]["available"]
    assert cleaned["shared_context"]["reference_field_contract"]["leaf_fields"] == []
    assert "reference_field_audit" not in cleaned["module_outputs"]["field_planning_module"]


def test_structured_advice_builds_clean_entity_field_plan():
    state = {
        "shared_context": {"query_requirements": [], "reference_paper_count": 2},
        "module_outputs": {"subjective_supervisor_module": {}},
    }
    advice = """{
      "round": 3,
      "verdict": "revise",
      "entity_registry": [
        {"entity_id": "sample", "owner_key": "sample_info", "required": true}
      ],
      "schema_concepts": [
        {
          "concept_id": "sample_id",
          "label": "sample identifier",
          "entity_id": "sample",
          "owner_key": "sample_info",
          "object_kind": "scalar",
          "data_type": "string"
        }
      ]
    }"""

    prepared, payload = finalizer.apply_structured_advice(state, advice)
    prepared = finalizer.ensure_deterministic_planning_modules(
        prepared,
        replace_field_plan=True,
    )

    field_plan = prepared["module_outputs"]["field_planning_module"]
    assert field_plan["field_groups"][0]["recommended_fields"] == [
        "sample_info.sample_id"
    ]
    entity = prepared["module_outputs"]["subjective_supervisor_module"]["entity_registry"][0]
    assert entity["entity_id"] == payload["entity_registry"][0]["entity_id"]
    assert entity["owner_key"] == payload["entity_registry"][0]["owner_key"]
    assert "recommendation" not in str(field_plan)


def test_early_checkpoint_builds_blind_architecture_and_field_plan():
    state = {
        "shared_context": {"query_requirements": ["Compare materials"], "reference_paper_count": 28},
        "module_outputs": {
            "mechanism_requirement_module": {
                "must_have_concepts": ["pairing_mechanism", "critical_temperature"]
            },
            "query_semantics_module": {
                "query_objects": [
                    {
                        "comparison_axes": ["chemical composition"],
                        "recommended_field_groups": ["section0.sample_id"],
                    }
                ]
            },
            "evidence_model_module": {
                "evidence_layers": [{"layer_name": "Direct Experimental"}]
            },
        },
    }

    prepared = finalizer.ensure_deterministic_planning_modules(
        state,
        replace_field_plan=False,
    )

    outputs = prepared["module_outputs"]
    assert outputs["subjective_supervisor_module"]["requirement_contract"]["concepts"]
    assert len(outputs["section_partition_module"]["core_sections"]) == 6
    planned = {
        path
        for group in outputs["field_planning_module"]["field_groups"]
        for path in group["recommended_fields"]
    }
    assert any(path.endswith(".pairing_mechanism") for path in planned)
    assert any(path.endswith(".source_figure") for path in planned)


def test_name_based_protocol_advice_is_normalized_and_keeps_governance_out_of_fields():
    state = {
        "shared_context": {"query_requirements": [], "reference_paper_count": 28},
        "module_outputs": {"locating_module": {"target_database_nature": "materials"}},
    }
    advice = """{
      "round": 1,
      "verdict": "proceed",
      "entities": [
        {"name": "Sample", "role": "specimen", "type": "experimental_subject"}
      ],
      "relationships": [
        {"source": "Sample", "predicate": "has", "target": "PropertyAssertion"}
      ],
      "query_families": [
        {"name": "condition_bound_property", "examples": ["Compare properties by condition"]}
      ],
      "counterfactual_queries": [
        {"category": "condition_binding", "query": "Can two conditions coexist?", "failure_exposed": "global conditions"}
      ],
      "schema_concepts": [
        {
          "name": "property_assertion",
          "description": "A context-bound property result.",
          "owner_entity": "Sample|MaterialSystem",
          "value_kind": "quantity_range_category_or_relation",
          "required_context": ["condition"],
          "evidence_expectation": "Direct source evidence."
        }
      ],
      "governance_rules": [{"rule": "Do not flatten", "instruction": "Keep records typed."}]
    }"""

    prepared, payload = finalizer.apply_structured_advice(state, advice)
    prepared = finalizer.ensure_deterministic_planning_modules(prepared, replace_field_plan=True)

    subjective = prepared["module_outputs"]["subjective_supervisor_module"]
    assert subjective["entity_registry"][0]["entity_id"] == "sample"
    assert subjective["requirement_contract"]["concepts"][0]["concept_id"] == "property_assertion"
    assert prepared["module_outputs"]["query_semantics_module"]["query_objects"]
    assert payload["relationship_contracts"] == payload["relationships"]
    field_plan_text = str(prepared["module_outputs"]["field_planning_module"])
    assert "property_assertion" in field_plan_text
    assert "do_not_flatten" not in field_plan_text


def test_locating_only_checkpoint_receives_query_semantics_fallback():
    state = {
        "shared_context": {"query_requirements": ["Compare materials"]},
        "module_outputs": {"locating_module": {"target_database_nature": "materials"}},
    }

    prepared = finalizer.ensure_deterministic_planning_modules(state, replace_field_plan=True)

    assert prepared["module_outputs"]["query_semantics_module"]["query_objects"]
    assert prepared["module_outputs"]["field_planning_module"]["field_groups"]


def test_raw_pipeline_checkpoint_is_hydrated_for_graph_output(tmp_path):
    state = {
        "inputs": {
            "database_goal": "Build a materials database",
            "discipline": "Materials science",
            "query_requirements": "Compare samples||Preserve evidence",
            "key_description_path": "no-reference.txt",
            "reference_papers": ["one.md", "two.md"],
        },
        "module_outputs": {"locating_module": {"target_database_nature": "materials"}},
    }

    prepared = finalizer.hydrate_checkpoint_state(state, tmp_path / "out.json")

    assert prepared["args"]["database_goal"] == "Build a materials database"
    assert prepared["args"]["reference_papers"] == ["one.md", "two.md"]
    assert prepared["shared_context"]["query_requirements"] == [
        "Compare samples",
        "Preserve evidence",
    ]
    assert prepared["shared_context"]["reference_paper_count"] == 2


def test_revised_protocol_aliases_preserve_primary_keys_and_object_contracts():
    state = {
        "shared_context": {"query_requirements": ["Compare samples under conditions"]},
        "module_outputs": {"locating_module": {"target_database_nature": "materials"}},
    }
    advice = """{
      "round": 2,
      "verdict": "revise",
      "revised_entities": [
        {
          "entity_id": "sample",
          "label": "Sample",
          "owner_key": "sample_records",
          "primary_key": "sample_id",
          "record_type": "experimental_subject"
        }
      ],
      "revised_relationships": [
        {
          "relationship_id": "sample_has_property",
          "source_entity": "Sample",
          "predicate": "has_property",
          "target_entity": "PropertyAssertion"
        }
      ],
      "revised_schema_concepts": [
        {
          "name": "sample_identity",
          "description": "Stable sample identity.",
          "owner_entity": ["Sample", "MaterialSystem"],
          "value_kind": "structured_record",
          "required_context": ["publication"],
          "object_contract": {
            "fields": {"sample_id": "string", "source_label": "string"},
            "required_subfields": ["sample_id"]
          }
        }
      ]
    }"""

    prepared, payload = finalizer.apply_structured_advice(state, advice)
    prepared = finalizer.ensure_deterministic_planning_modules(prepared, replace_field_plan=True)
    contract = finalizer.section_agent.normalize_requirement_contract(
        prepared["shared_context"],
        prepared["module_outputs"]["subjective_supervisor_module"],
    )
    entities = finalizer.section_agent.normalize_entity_registry(
        prepared["module_outputs"]["subjective_supervisor_module"], contract
    )

    assert payload["relationship_contracts"] == payload["revised_relationships"]
    assert contract["concepts"][0]["object_contract"]["required_subfields"] == ["sample_id"]
    assert entities[0]["primary_key"] == "sample_id"
    assert [item["concept_id"] for item in contract["concepts"]] == ["sample_identity"]
    merged = finalizer.graph.merge_structured_object_contract(
        contract["concepts"][0]["object_contract"],
        "sample_records.sample_identity",
        contract["concepts"],
    )
    assert merged["object_kind"] == "entity_descriptor"
    assert {"identity", "evidence", "sample_id"} <= set(merged["required_subfields"])
