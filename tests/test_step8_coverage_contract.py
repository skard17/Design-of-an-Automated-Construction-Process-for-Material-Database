import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import section_design_agent as agent  # noqa: E402
import section_design_agent_prompt as prompts  # noqa: E402
import section_design_langgraph_human_gate as graph  # noqa: E402


def base_result(fields, *, concepts=None, entities=None):
    return {
        "database_positioning": {
            "database_goal": "test database",
            "discipline": "materials science",
            "query_requirements": ["compare A, B, and C"],
            "retrieval_unit": "record",
            "design_rationale": "test",
        },
        "section_design": {
            "core_sections": [
                {"section_id": "material_info.section0"},
                {"section_id": "material_info.section1"},
            ],
            "non_core_sections": [],
        },
        "schema_definition": {
            "top_level_keys": [
                {"key": "material_info", "description": "material records"},
                {"key": "paper_info", "description": "paper records"},
            ],
            "field_registry": fields,
        },
        "requirement_contract": {"concepts": concepts or []},
        "entity_registry": entities or [],
        "quality_check": {
            "topic_specific_adjustments": ["test"],
            "coverage_check": ["test"],
        },
    }


def scalar_field(path, concept_ids):
    return {
        "field_path": path,
        "section_id": "material_info.section1",
        "field_name": path.rsplit(".", 1)[-1],
        "data_type": "string",
        "required": False,
        "source_basis": ["text"],
        "concept_ids": concept_ids,
        "reason": "test",
    }


class Step8CoverageContractTests(unittest.TestCase):
    def test_subjective_module_requires_contract_and_entities(self):
        legacy_output = {
            "database_nature": "materials database",
            "modeling_position": "domain specific",
            "must_have_concepts": ["A"],
            "must_not_become": ["generic"],
            "red_flags": ["missing A"],
            "approved_section_strategy": ["use backbone"],
            "redo_directives": [],
        }

        errors = agent.validate_module_result("subjective_supervisor_module", legacy_output)

        self.assertTrue(any("requirement_contract" in error for error in errors))
        self.assertTrue(any("entity_registry" in error for error in errors))

    def test_missing_required_concept_is_blocking(self):
        concepts = [
            {"concept_id": "a", "label": "A", "required": True, "entity_id": "material"},
            {"concept_id": "b", "label": "B", "required": True, "entity_id": "material"},
            {"concept_id": "c", "label": "C", "required": True, "entity_id": "material"},
        ]
        result = base_result(
            [scalar_field("material_info.section1.a", ["a"]), scalar_field("material_info.section1.c", ["c"])],
            concepts=concepts,
        )

        report = agent.build_coverage_report(result)
        errors = agent.validate_coverage_report(report)

        self.assertEqual(2 / 3, report["required_concept_coverage"]["ratio"])
        self.assertIn("b", report["required_concept_coverage"]["missing_concept_ids"])
        self.assertTrue(any(error.startswith("coverage:missing_concepts:") for error in errors))

    def test_structured_object_without_contract_is_blocking(self):
        field = scalar_field("material_info.section1.parameter", ["parameter"])
        field["data_type"] = "array of objects"
        result = base_result(
            [field],
            concepts=[
                {
                    "concept_id": "parameter",
                    "label": "Parameter",
                    "required": True,
                    "entity_id": "material",
                    "object_kind": "measurement",
                }
            ],
        )

        report = agent.build_coverage_report(result)

        self.assertIn("material_info.section1.parameter", report["structured_object_completeness"]["incomplete_fields"])
        self.assertTrue(agent.validate_coverage_report(report))

    def test_required_device_entity_needs_owner(self):
        result = base_result(
            [scalar_field("material_info.section1.response", ["device_response"])],
            concepts=[
                {
                    "concept_id": "device_response",
                    "label": "Device response",
                    "required": True,
                    "entity_id": "device",
                }
            ],
            entities=[
                {
                    "entity_id": "device",
                    "label": "Device",
                    "required": True,
                    "owner_key": "device_info",
                    "independent_owner": True,
                }
            ],
        )

        report = agent.build_coverage_report(result)

        self.assertIn("device", report["entity_owner_coverage"]["missing_entity_ids"])
        self.assertTrue(any(error.startswith("coverage:missing_entity_owners:") for error in agent.validate_coverage_report(report)))

    def test_registered_entity_identity_is_not_domain_leakage(self):
        identity = scalar_field("device_info.identity", [])
        identity["section_id"] = "device_info"
        result = base_result(
            [identity],
            entities=[
                {
                    "entity_id": "device",
                    "label": "Device",
                    "required": True,
                    "owner_key": "device_info",
                    "independent_owner": True,
                }
            ],
        )
        result["schema_definition"]["top_level_keys"].append(
            {"key": "device_info", "description": "device records"}
        )

        report = agent.build_coverage_report(result)

        self.assertEqual([], report["unrelated_domain_fields"])

    def test_human_advice_is_required_input_to_contract_prompt(self):
        shared = {
            "database_goal": "general device materials database",
            "query_requirements": ["compare directional response"],
            "human_advice": "Preserve forward and reverse critical responses as separate concepts.",
        }
        prompt = prompts.build_subjective_supervisor_prompt(shared, {}, {}, {}, {})

        self.assertIn("forward and reverse critical responses", prompt)
        self.assertIn('"requirement_contract"', prompt)
        self.assertIn('"entity_registry"', prompt)

    def test_normalized_contract_backfills_source_requirement_traceability(self):
        contract = agent.normalize_requirement_contract(
            {
                "query_requirements": ["record transition temperature"],
                "human_advice": "Keep the measurement criterion.",
            },
            {
                "requirement_contract": {
                    "concepts": [
                        {
                            "concept_id": "transition_temperature",
                            "label": "Transition temperature",
                            "required": True,
                        },
                        {
                            "concept_id": "measurement_criterion",
                            "label": "Measurement criterion",
                            "required": True,
                        },
                    ]
                }
            },
        )
        traced = {
            requirement_id
            for concept in contract["concepts"]
            for requirement_id in concept["source_requirement_ids"]
        }

        self.assertEqual({"query_1", "human_advice"}, traced)

    def test_query_requirements_are_atomized_when_model_omits_items(self):
        contract = agent.normalize_requirement_contract(
            {"query_requirements": ["compare A, B, and C"]},
            {
                "requirement_contract": {
                    "concepts": [
                        {
                            "concept_id": "a",
                            "label": "A",
                            "required": True,
                            "source_requirement_ids": ["query_1"],
                        }
                    ]
                }
            },
        )

        concept_ids = {concept["concept_id"] for concept in contract["concepts"]}

        self.assertTrue({"a", "b", "c"}.issubset(concept_ids))

    def test_schema_prompt_does_not_cap_required_fields(self):
        prompt = prompts.build_schema_design_prompt(*({} for _ in range(11)))

        self.assertNotIn("60-90", prompt)
        self.assertIn("required concept", prompt)

    def test_unmapped_domain_field_is_reported_as_unrelated(self):
        result = base_result(
            [scalar_field("material_info.section1.Tc", [])],
            concepts=[
                {
                    "concept_id": "coercive_field",
                    "label": "Coercive field",
                    "required": True,
                    "entity_id": "material",
                }
            ],
        )

        report = agent.build_coverage_report(result)

        self.assertEqual(1, report["unrelated_domain_field_count"])

    def test_legacy_result_without_contract_remains_readable(self):
        result = base_result([scalar_field("material_info.section1.a", [])])
        result.pop("requirement_contract")
        result.pop("entity_registry")

        report = agent.build_coverage_report(result)

        self.assertTrue(report["legacy_mode"])
        self.assertEqual([], agent.validate_coverage_report(report))

    def test_assembled_result_preserves_normalized_contract(self):
        subjective = {
            "must_have_concepts": ["A"],
            "requirement_contract": {
                "concepts": [
                    {
                        "concept_id": "a",
                        "label": "A",
                        "required": True,
                        "entity_id": "material",
                        "source_requirement_ids": ["query_1"],
                    }
                ]
            },
            "entity_registry": [
                {
                    "entity_id": "material",
                    "label": "Material",
                    "required": True,
                    "owner_key": "material_info",
                    "independent_owner": False,
                }
            ],
            "red_flags": [],
        }
        assembled = agent.assemble_final_result_from_modules(
            {"database_goal": "broad database", "query_requirements": ["compare A"], "reference_paper_count": 1},
            {"database_goal": "broad database", "discipline": "materials", "retrieval_unit": "record"},
            {"query_objects": []},
            subjective,
            {"topic_specific_adjustments": []},
            {"core_sections": [], "non_core_sections": []},
            {},
            {},
            {},
            {"top_level_keys": [], "field_registry": []},
            {},
        )

        self.assertEqual("a", assembled["requirement_contract"]["concepts"][0]["concept_id"])
        self.assertEqual("broad_goal_with_single_reference", assembled["requirement_contract"]["reference_representativeness"]["risk"])
        self.assertEqual("material", assembled["entity_registry"][0]["entity_id"])

    def test_retry_candidate_must_improve_coverage_without_new_blockers(self):
        old_report = {
            "legacy_mode": False,
            "required_concept_coverage": {"ratio": 0.5, "missing_concept_ids": ["b"]},
            "entity_owner_coverage": {"missing_entity_ids": []},
            "structured_object_completeness": {"incomplete_fields": []},
            "evidence_contract_coverage": {"missing_concept_ids": []},
            "query_requirement_traceability": {"missing_requirement_ids": []},
            "unrelated_domain_fields": [],
        }
        improved = {
            **old_report,
            "required_concept_coverage": {"ratio": 1.0, "missing_concept_ids": []},
        }
        regressed = {
            **improved,
            "entity_owner_coverage": {"missing_entity_ids": ["device"]},
        }

        self.assertTrue(agent.is_coverage_improvement(old_report, improved))
        self.assertFalse(agent.is_coverage_improvement(old_report, regressed))

    def test_coverage_errors_route_to_owning_module(self):
        self.assertEqual(
            "subjective_supervisor",
            graph.coverage_repair_target(["coverage:missing_entity_owners:device"]),
        )
        self.assertEqual(
            "field_planning",
            graph.coverage_repair_target(["coverage:missing_concepts:critical_current"]),
        )
        self.assertEqual(
            "schema_design",
            graph.coverage_repair_target(["coverage:incomplete_object_contracts:device_info.response"]),
        )
        self.assertEqual(
            "schema_design",
            graph.coverage_repair_target(["coverage:untraced_query_requirements:query_1"]),
        )
        self.assertEqual(
            "figure_classification",
            graph.coverage_repair_target(["field x is figure-based but missing figure_constraint"]),
        )

    def test_large_field_plan_uses_full_deterministic_compilation(self):
        recommended_fields = [f"material_info.section1.parameter_{index}" for index in range(105)]
        state = {
            "shared_context": {"query_requirements": ["cover parameter zero"]},
            "module_outputs": {
                "subjective_supervisor_module": {
                    "requirement_contract": {
                        "concepts": [
                            {
                                "concept_id": "parameter_0",
                                "label": "Parameter zero",
                                "required": True,
                                "entity_id": "material",
                                "owner_key": "material_info",
                                "object_kind": "measurement",
                                "source_requirement_ids": ["query_1"],
                            }
                        ]
                    },
                    "entity_registry": [
                        {
                            "entity_id": "material",
                            "label": "Material",
                            "required": True,
                            "owner_key": "material_info",
                            "independent_owner": False,
                        }
                    ],
                },
                "field_planning_module": {
                    "field_groups": [
                        {
                            "section_id": "material_info.section1",
                            "recommended_fields": recommended_fields,
                            "evidence_strategy": "text and table",
                        }
                    ]
                },
            },
            "module_attempts": {},
            "module_errors": {},
        }

        self.assertTrue(graph.should_compile_schema_deterministically(state))
        compiled = graph.compile_schema_from_field_planning(state, "large plan")
        registry = compiled["module_outputs"]["schema_design_module"]["field_registry"]
        paths = {field["field_path"] for field in registry}

        self.assertTrue(set(recommended_fields).issubset(paths))
        self.assertTrue(any("parameter_0" in field.get("concept_ids", []) for field in registry))

    def test_legacy_checkpoint_migrates_to_schema_compilation(self):
        migrated = graph.migrate_legacy_modular_checkpoint(
            {
                "step": "step8_section_design_agent_checkpoint",
                "inputs": {
                    "database_goal": "legacy goal",
                    "discipline": "materials",
                    "query_requirements": "compare A; compare B",
                    "reference_papers": [],
                },
                "module_outputs": {"field_planning_module": {"field_groups": []}},
                "module_errors": {},
            },
            {
                "output": "new-output.json",
                "key_description_path": str(REPO_ROOT / "tests" / "fixtures" / "generic_step8_context.yaml"),
                "reference_papers": [],
                "human_advice": "",
                "human_advice_path": "",
            },
        )

        self.assertEqual("schema_design", migrated["next_node"])
        self.assertTrue(migrated["force_deterministic_schema_compilation"])
        self.assertEqual(["compare A; compare B"], migrated["shared_context"]["query_requirements"])
        self.assertEqual("new-output.json", migrated["args"]["output"])

    def test_compiled_object_field_has_executable_contract(self):
        state = {
            "shared_context": {"query_requirements": ["record processing conditions"]},
            "module_outputs": {
                "subjective_supervisor_module": {
                    "requirement_contract": {
                        "concepts": [
                            {
                                "concept_id": "processing_conditions",
                                "label": "Processing conditions",
                                "required": True,
                                "entity_id": "material",
                                "owner_key": "material_info",
                                "object_kind": "process",
                                "source_requirement_ids": ["query_1"],
                            }
                        ]
                    },
                    "entity_registry": [],
                },
                "field_planning_module": {
                    "field_groups": [
                        {
                            "section_id": "material_info.section2",
                            "recommended_fields": ["material_info.section2.processing_conditions"],
                            "evidence_strategy": "text",
                        }
                    ]
                },
            },
            "module_attempts": {},
            "module_errors": {},
        }

        compiled = graph.compile_schema_from_field_planning(state, "object contract")
        field = next(
            item
            for item in compiled["module_outputs"]["schema_design_module"]["field_registry"]
            if item["field_path"] == "material_info.section2.processing_conditions"
        )

        self.assertIn("object", field["data_type"])
        self.assertEqual("process", field["object_contract"]["object_kind"])
        self.assertFalse(agent.object_contract_missing_slots(field))

    def test_postprocess_standard_object_fields_remain_executable(self):
        result = base_result(
            [scalar_field("material_info.section1.parameter", ["parameter"])],
            concepts=[
                {
                    "concept_id": "parameter",
                    "label": "Parameter",
                    "required": True,
                    "entity_id": "material",
                    "source_requirement_ids": ["query_1"],
                }
            ],
        )

        finalized = agent.finalize_result({}, result)
        report = finalized["coverage_report"]

        self.assertEqual([], report["structured_object_completeness"]["incomplete_fields"])
        figure_fields = [
            field
            for field in finalized["schema_definition"]["field_registry"]
            if "figure" in (field.get("source_basis") or [])
        ]
        self.assertTrue(figure_fields)
        self.assertTrue(all(isinstance(field.get("figure_constraint"), dict) for field in figure_fields))


if __name__ == "__main__":
    unittest.main()
