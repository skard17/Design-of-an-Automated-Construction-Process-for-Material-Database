import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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
    def test_long_context_defaults_do_not_force_short_schema_generation(self):
        self.assertGreaterEqual(agent.DEFAULT_REQUEST_TIMEOUT_SECONDS, 1200)
        self.assertEqual(384000, agent.DEFAULT_MODULE_MAX_TOKENS)
        self.assertTrue(agent.MODULE_MAX_TOKENS)
        self.assertTrue(
            all(value == 384000 for value in agent.MODULE_MAX_TOKENS.values())
        )
        self.assertEqual(2000000, prompts.MAX_REFERENCE_CONTEXT_CHARS)
        self.assertEqual(500000, prompts.MAX_KEY_DESCRIPTION_CHARS)

    def test_field_registry_is_not_silently_truncated_by_a_fixed_count_cap(self):
        registry = [
            scalar_field(f"material_info.section1.property_{index}", [f"property_{index}"])
            for index in range(300)
        ]

        compacted = graph.compact_field_registry(registry)

        self.assertEqual(300, len(compacted))

    def test_state_snapshot_preserves_complete_supported_corpus_context(self):
        reference_context = "x" * 600000

        snapshot = graph.json_safe_state(
            {"shared_context": {"reference_paper_context": reference_context}}
        )

        self.assertEqual(
            reference_context,
            snapshot["shared_context"]["reference_paper_context"],
        )

    def test_local_schema_review_accepts_protocol_approved_entity_owner(self):
        schema = {
            "top_level_keys": [{"key": "sample_info"}],
            "field_registry": [scalar_field("sample_info.sample_id", ["sample_id"])],
        }
        state = {
            "module_outputs": {
                "subjective_supervisor_module": {
                    "entity_registry": [
                        {"entity_id": "sample", "owner_key": "sample_info"}
                    ]
                }
            }
        }

        ok, missing = graph.local_schema_coverage_sufficient(schema, {}, state)

        self.assertTrue(ok)
        self.assertEqual([], missing)

    def test_local_schema_review_rejects_unapproved_entity_owner(self):
        schema = {
            "top_level_keys": [{"key": "invented_info"}],
            "field_registry": [scalar_field("invented_info.value", ["value"])],
        }

        ok, missing = graph.local_schema_coverage_sufficient(schema, {}, {})

        self.assertFalse(ok)
        self.assertIn("unapproved schema roots present: ['invented_info']", missing)

    def test_local_schema_review_does_not_treat_evidence_container_as_measurement(self):
        schema = {
            "top_level_keys": [{"key": "material_info"}],
            "field_registry": [
                scalar_field(
                    "material_info.section1.evidence.source_text",
                    ["source_text"],
                )
            ],
        }
        critic = {
            "missing_concepts": ["evidence with explicit child fields"],
            "redo_directives": [],
            "structural_weaknesses": [],
        }

        ok, missing = graph.local_schema_coverage_sufficient(schema, critic, {})

        self.assertTrue(ok)
        self.assertEqual([], missing)

    def test_local_schema_review_accepts_normalized_condition_entities(self):
        schema = {
            "top_level_keys": [
                {"key": "condition_set_info"},
                {"key": "condition_entry_info"},
                {"key": "observation_info"},
            ],
            "field_registry": [
                scalar_field("condition_set_info.condition_set_id", ["condition_set_id"]),
                scalar_field("condition_entry_info.condition_normalized_value", ["condition_normalized_value"]),
                scalar_field("observation_info.observation_condition_set_id", ["observation_condition_set_id"]),
            ],
        }
        state = {
            "module_outputs": {
                "subjective_supervisor_module": {
                    "entity_registry": [
                        {"entity_id": "condition_set", "owner_key": "condition_set_info"},
                        {"entity_id": "condition_entry", "owner_key": "condition_entry_info"},
                        {"entity_id": "observation", "owner_key": "observation_info"},
                    ]
                }
            }
        }
        critic = {
            "missing_concepts": ["measurement_conditions"],
            "redo_directives": [],
            "structural_weaknesses": [],
        }

        ok, missing = graph.local_schema_coverage_sufficient(schema, critic, state)

        self.assertTrue(ok)
        self.assertEqual([], missing)

    def test_query_semantics_budget_can_finish_reasoning_and_emit_json(self):
        self.assertEqual(384000, agent.MODULE_MAX_TOKENS["query_semantics_module"])

    def test_subjective_supervisor_budget_can_emit_full_requirement_contract(self):
        self.assertEqual(384000, agent.MODULE_MAX_TOKENS["subjective_supervisor_module"])

    def test_section_partition_budget_can_emit_full_structured_plan(self):
        self.assertEqual(384000, agent.MODULE_MAX_TOKENS["section_partition_module"])
        self.assertEqual(384000, agent.MODULE_MAX_TOKENS["figure_classification_module"])

    def test_openai_client_disables_hidden_transport_retries(self):
        with mock.patch.object(agent, "OpenAI") as openai_cls:
            client = agent.get_client(
                base_url="https://example.invalid/v1",
                api_key="test-key",
                timeout=42,
            )

        self.assertEqual(client["backend"], "openai")
        openai_cls.assert_called_once_with(
            base_url="https://example.invalid/v1",
            api_key="test-key",
            timeout=42,
            max_retries=0,
        )

    def test_openai_chat_streams_with_module_token_budget(self):
        completion_api = mock.Mock()
        completion_api.create.return_value = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content='{"status":'))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content='"ok"}'))]
            ),
        ]
        client = {
            "backend": "openai",
            "client": SimpleNamespace(
                chat=SimpleNamespace(completions=completion_api)
            ),
        }

        response = agent.chat(client, "test-model", "prompt", max_tokens=2048)

        self.assertEqual(response, '{"status":"ok"}')
        request = completion_api.create.call_args.kwargs
        self.assertTrue(request["stream"])
        self.assertEqual(request["max_tokens"], 2048)
        self.assertEqual(request["response_format"], {"type": "json_object"})

    def test_openai_empty_stream_reports_reasoning_and_finish_reason(self):
        completion_api = mock.Mock()
        completion_api.create.return_value = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content="thinking",
                        ),
                    )
                ]
            )
        ]
        client = {
            "backend": "openai",
            "client": SimpleNamespace(
                chat=SimpleNamespace(completions=completion_api)
            ),
        }

        with self.assertRaises(RuntimeError) as caught:
            agent.chat(client, "test-model", "prompt", max_tokens=2048)

        message = str(caught.exception)
        self.assertIn("reasoning_chars=8", message)
        self.assertIn("finish_reason=length", message)

    def test_qiniu_chat_uses_explicit_streaming_adapter(self):
        qiniu_client = mock.Mock()
        qiniu_client.chat_json.return_value = ({"status": "ok"}, {"elapsed_seconds": 1})
        client = {"backend": "qiniu", "client": qiniu_client}

        response = agent.chat(client, "test-model", "prompt", max_tokens=2048)

        self.assertEqual(response, '{"status": "ok"}')
        self.assertEqual(qiniu_client.model, "test-model")
        qiniu_client.chat_json.assert_called_once()
        self.assertEqual(qiniu_client.chat_json.call_args.kwargs["max_tokens"], 2048)

    def test_langgraph_honors_explicit_checkpoint_path(self):
        args = SimpleNamespace(
            output="fallback.json",
            checkpoint_output="explicit.state.json",
        )

        self.assertEqual(graph.state_path_from_args(args), Path("explicit.state.json"))

    def test_deterministic_routing_recognizes_generic_axis_pair(self):
        self.assertEqual(
            "material_info.section4",
            graph.infer_section_from_field_path("device_info.i_v"),
        )
        self.assertEqual(
            "material_info.section4",
            graph.section_for_repair_parent("i_v", "explicit response series"),
        )

    def test_reference_paper_context_balances_all_documents(self):
        context = "\n\n".join(
            f"Paper: paper_{index}.md\n" + (str(index) * 12000)
            for index in range(1, 4)
        )

        compact = prompts._balanced_reference_paper_context(context, 6000)

        self.assertLessEqual(len(compact), 6000)
        for index in range(1, 4):
            self.assertIn(f"Paper: paper_{index}.md", compact)

    def test_complete_superconductivity_reference_catalog_has_239_leaves(self):
        path = REPO_ROOT / "key_description.yaml"
        contract = agent.build_reference_field_contract(
            path.read_text(encoding="utf-8"),
            str(path),
        )

        self.assertTrue(contract["available"])
        self.assertEqual(292, contract["all_path_count"])
        self.assertEqual(239, contract["leaf_count"])
        self.assertIn(
            "section5.theoretical_keywords",
            {item["path"] for item in contract["leaf_fields"]},
        )
        self.assertIn(
            "paper_info.resources.cif.notes",
            {item["path"] for item in contract["leaf_fields"]},
        )
        theoretical_keywords = next(
            item
            for item in contract["leaf_fields"]
            if item["path"] == "section5.theoretical_keywords"
        )
        self.assertIn("strict fallback list", theoretical_keywords["extraction_notes"])

    def test_reference_catalog_treats_schema_policy_as_design_instruction_not_field(self):
        contract = agent.build_reference_field_contract(
            "schema_policy: derive atomic fields from the task and papers\n",
            "context.yaml",
        )

        self.assertFalse(contract["available"])
        self.assertEqual(0, contract["leaf_count"])
        self.assertEqual([], contract["leaf_fields"])
        self.assertEqual("schema_policy", contract["design_directives"][0]["path"])

    def test_reference_catalog_is_task_adaptive_not_a_mandatory_template(self):
        contract = {
            "available": True,
            "policy": "task_adaptive",
            "leaf_fields": [
                {"path": "material_info.section1.band_gap", "data_type": "number"},
                {"path": "material_info.section1.transition_temperature", "data_type": "number"},
                {"path": "section5.pairing_mechanism", "data_type": "string"},
            ],
        }
        audit = agent.normalize_reference_field_audit(
            contract,
            {
                "included_leaf_paths": ["material_info.section1.band_gap"],
                "excluded_path_prefixes": [
                    {
                        "path_prefix": "material_info.section1.transition_temperature",
                        "reason": "Not part of the semiconductor task or supplied evidence.",
                    },
                    {
                        "path_prefix": "section5.pairing_mechanism",
                        "reason": "Superconducting mechanism is outside this task.",
                    },
                ],
            },
        )

        self.assertEqual(["material_info.section1.band_gap"], audit["included_leaf_paths"])
        self.assertEqual([], audit["unresolved_leaf_paths"])
        self.assertEqual([], audit["invalid_decisions"])

    def test_reference_coverage_uses_applicable_leaves_not_full_catalog_denominator(self):
        result = base_result(
            [scalar_field("material_info.section1.band_gap", [])],
        )
        result["reference_field_contract"] = {
            "available": True,
            "policy": "task_adaptive",
            "leaf_fields": [
                {"path": "material_info.section1.band_gap"},
                {"path": "material_info.section1.transition_temperature"},
            ],
        }
        result["reference_field_audit"] = {
            "included_leaf_paths": ["material_info.section1.band_gap"],
            "excluded_path_prefixes": [
                {
                    "path_prefix": "material_info.section1.transition_temperature",
                    "reason": "Outside this task.",
                }
            ],
        }

        report = agent.build_coverage_report(result)

        self.assertEqual(1.0, report["reference_field_coverage"]["decision_coverage_ratio"])
        self.assertEqual(1.0, report["reference_field_coverage"]["applicable_coverage_ratio"])
        self.assertEqual(0.5, report["reference_field_coverage"]["full_reference_coverage_ratio"])
        self.assertEqual([], agent.validate_coverage_report(report))

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

    def test_subjective_supervisor_uses_distilled_modules_not_raw_paper_context(self):
        shared = {
            "database_goal": "materials database",
            "query_requirements": ["compare performance"],
            "reference_paper_count": 28,
            "reference_paper_context": "RAW_PAPER_TEXT" * 1000,
        }

        prompt = prompts.build_subjective_supervisor_prompt(shared, {}, {}, {}, {})

        self.assertNotIn("RAW_PAPER_TEXT", prompt)
        self.assertIn('"reference_paper_count": 28', prompt)

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
        self.assertEqual(
            {"transition_temperature", "measurement_criterion"},
            {concept["concept_id"] for concept in contract["concepts"]},
        )

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

    def test_assembled_failure_result_accepts_missing_specialization_critic(self):
        assembled = agent.assemble_final_result_from_modules(
            {"database_goal": "materials database", "query_requirements": []},
            {"database_goal": "materials database", "discipline": "materials"},
            {"query_objects": []},
            {},
            {"topic_specific_adjustments": []},
            {"core_sections": [], "non_core_sections": []},
            {},
            {},
            {},
            {"top_level_keys": [{"key": "material_info"}], "field_registry": []},
            None,
        )

        self.assertIn(
            "Specialization critic: unknown; generic-template risk: False",
            assembled["quality_check"]["topic_specific_adjustments"],
        )

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
            "specialization_critic",
            graph.coverage_repair_target(
                ["schema_patch_contract_regression:required_concepts:provenance"]
            ),
        )
        self.assertEqual(
            "subjective_supervisor",
            graph.coverage_repair_target(["coverage:missing_entity_owners:device"]),
        )
        self.assertEqual(
            "schema_design_repair",
            graph.coverage_repair_target(["coverage:missing_concepts:critical_current"]),
        )
        self.assertEqual(
            "schema_design_repair",
            graph.coverage_repair_target(["coverage:incomplete_object_contracts:device_info.response"]),
        )
        self.assertEqual(
            "schema_design_repair",
            graph.coverage_repair_target(["coverage:untraced_query_requirements:query_1"]),
        )
        self.assertEqual(
            "specialization_critic",
            graph.coverage_repair_target(
                ["coverage:unmapped_domain_fields:material_info.section1.extra"]
            ),
        )
        self.assertEqual(
            "specialization_critic",
            graph.coverage_repair_target(
                ["coverage:invalid_concept_references:section5.lambda=>invented"]
            ),
        )
        self.assertEqual(
            "query_semantics",
            graph.coverage_repair_target(
                ["coverage:untraced_care_counterfactuals:query_2"]
            ),
        )
        self.assertEqual(
            "figure_classification",
            graph.coverage_repair_target(["field x is figure-based but missing figure_constraint"]),
        )
        self.assertEqual(
            "field_planning",
            graph.coverage_repair_target(["coverage:unresolved_reference_fields:section5.x"]),
        )
        self.assertEqual(
            "schema_design_repair",
            graph.coverage_repair_target(["coverage:missing_reference_fields:material_info.section1.x"]),
        )

    def test_exact_current_field_plan_paths_are_utility_traced(self):
        planned = scalar_field("material_info.section2.annealing_temperature", [])
        speculative = scalar_field("material_info.section2.speculative_extra", [])
        result = base_result([planned, speculative])
        result["field_plan_traceability"] = agent.build_field_plan_traceability(
            {
                "field_groups": [
                    {
                        "section_id": "material_info.section2",
                        "group_name": "processing",
                        "recommended_fields": ["annealing_temperature"],
                    }
                ]
            }
        )

        report = agent.build_coverage_report(result)

        self.assertNotIn(
            "material_info.section2.annealing_temperature",
            report["unrelated_domain_fields"],
        )
        self.assertIn(
            "material_info.section2.speculative_extra",
            report["unrelated_domain_fields"],
        )
        self.assertEqual(
            ["material_info.section2.annealing_temperature"],
            report["field_plan_traceability"]["represented_planned_field_paths"],
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
        self.assertTrue(all(str(field.get("extraction_notes") or "").strip() for field in registry))

    def test_unapproved_root_repair_preserves_concept_contracts(self):
        state = {
            "shared_context": {"query_requirements": ["record anisotropy field"]},
            "module_outputs": {
                "subjective_supervisor_module": {
                    "requirement_contract": {
                        "concepts": [
                            {
                                "concept_id": "anisotropy_field",
                                "label": "Anisotropy field",
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
                            "group_name": "anisotropy",
                            "purpose": "Magnetic anisotropy parameters",
                            "recommended_fields": [
                                "material_info.section1.anisotropy_field.value"
                            ],
                            "evidence_strategy": "text and figure",
                        }
                    ]
                },
                "schema_design_module": {
                    "top_level_keys": [{"key": "evidence_collection"}],
                    "field_registry": [
                        {
                            "field_path": "evidence_collection.figure",
                            "section_id": "evidence_collection",
                        }
                    ],
                },
            },
            "module_attempts": {},
            "module_errors": {},
            "repair_instructions": ["unapproved schema root"],
        }

        repaired = graph.deterministic_schema_repair(state)
        registry = repaired["module_outputs"]["schema_design_module"]["field_registry"]
        field = next(
            item
            for item in registry
            if item["field_path"] == "material_info.section1.anisotropy_field.value"
        )

        self.assertIn("anisotropy_field", field["concept_ids"])
        self.assertIn("do not infer", field["extraction_notes"])

    def test_resume_router_compiles_coverage_repair_without_rewriting_contract(self):
        state = {
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
                            "group_name": "coercivity",
                            "purpose": "Coercive-field measurements",
                            "recommended_fields": [
                                "material_info.section1.coercive_field.value"
                            ],
                            "evidence_strategy": "text and figure",
                        }
                    ]
                },
                "schema_design_module": {"field_registry": []},
                "specialization_critic_module": {"specialization_status": "pass"},
                "aggregation": {"status": "failed"},
            },
            "module_attempts": {},
            "module_errors": {},
            "repair_instructions": ["coverage:missing_entity_owners:evidence"],
            "validation_errors": [],
            "last_error_node": "aggregation",
            "next_node": "subjective_supervisor",
        }

        repaired = graph.resume_router_node(state)

        self.assertEqual(repaired["next_node"], "specialization_critic")
        self.assertNotIn("aggregation", repaired["module_outputs"])
        registry = repaired["module_outputs"]["schema_design_module"]["field_registry"]
        self.assertTrue(
            any("coercive_field" in field.get("concept_ids", []) for field in registry)
        )

    def test_material_section5_path_is_canonicalized(self):
        self.assertEqual(
            graph.canonicalize_rebuild_field_path(
                "material_info.section5.model.parameter",
                "material_info.section5",
            ),
            "section5.model.parameter",
        )
        self.assertEqual(
            graph.canonicalize_rebuild_field_path("material_system.formula", "section0"),
            "material_info.section0.material_system.formula",
        )

    def test_requirement_contract_drops_orphans_and_includes_human_advice(self):
        contract = agent.normalize_requirement_contract(
            {
                "query_requirements": ["Record magnetic order"],
                "human_advice": "Include magnetocaloric entropy change",
            },
            {
                "requirement_contract": {
                    "concepts": [
                        {"concept_id": "supervisee", "label": "Supervisee"},
                        {
                            "concept_id": "magnetic_order",
                            "label": "Magnetic order",
                            "source_requirement_ids": ["query_1"],
                        },
                    ]
                }
            },
        )
        concept_ids = {item["concept_id"] for item in contract["concepts"]}

        self.assertNotIn("supervisee", concept_ids)
        self.assertIn("magnetic_order", concept_ids)
        self.assertTrue(
            any("magnetocaloric" in concept_id for concept_id in concept_ids)
        )

    def test_requirement_contract_keeps_data_concepts_not_workflow_directives(self):
        contract = agent.normalize_requirement_contract(
            {
                "query_requirements": [],
                "human_advice": (
                    "Treat this as a general database schema. "
                    "The current test documents were converted from the PDF text layer. "
                    "Record magnetic order."
                ),
            },
            {"requirement_contract": {"concepts": []}},
        )

        concept_ids = {item["concept_id"] for item in contract["concepts"]}

        self.assertEqual({"magnetic_order"}, concept_ids)

    def test_evidence_entity_is_not_an_unapproved_top_level_owner(self):
        entities = agent.normalize_entity_registry(
            {
                "entity_registry": [
                    {
                        "entity_id": "evidence",
                        "owner_key": "evidence_collection",
                        "required": True,
                        "independent_owner": True,
                    }
                ]
            },
            {
                "concepts": [{"entity_id": "evidence"}],
                "source_requirements": [
                    {"requirement_id": "query_1", "text": "Record evidence provenance"}
                ],
            },
        )

        self.assertEqual(entities[0]["owner_key"], "material_info")
        self.assertFalse(entities[0]["independent_owner"])

    def test_entity_registry_drops_untraced_supervisor_entities(self):
        entities = agent.normalize_entity_registry(
            {
                "entity_registry": [
                    {"entity_id": "supervisor", "owner_key": "supervisor_info"},
                    {"entity_id": "evaluation_record", "owner_key": "evaluation_info"},
                    {"entity_id": "material", "owner_key": "material_info"},
                ]
            },
            {
                "concepts": [{"entity_id": "material"}],
                "source_requirements": [
                    {"requirement_id": "query_1", "text": "Record magnetic order"}
                ],
            },
        )

        self.assertEqual([item["entity_id"] for item in entities], ["material"])

    def test_deterministic_compilation_preserves_adapted_reference_leaf_semantics(self):
        state = {
            "shared_context": {
                "query_requirements": ["record ionic conductivity"],
                "reference_field_contract": {
                    "available": True,
                    "policy": "task_adaptive",
                    "leaf_fields": [
                        {
                            "path": "material_info.section1.transport_value",
                            "data_type": "number with unit",
                            "description_hint": "A measured transport-property value.",
                            "extraction_notes": "Keep separate records for distinct measurement conditions.",
                        }
                    ],
                },
            },
            "module_outputs": {
                "subjective_supervisor_module": {"requirement_contract": {"concepts": []}},
                "field_planning_module": {
                    "field_groups": [],
                    "reference_field_audit": {
                        "adapted_leaf_mappings": [
                            {
                                "source_path": "material_info.section1.transport_value",
                                "target_path": "material_info.section1.ionic_conductivity",
                                "reason": "Adapt the generic transport leaf to the task property.",
                            }
                        ]
                    },
                },
            },
            "module_attempts": {},
            "module_errors": {},
        }

        compiled = graph.compile_schema_from_field_planning(state, "test adaptation")
        fields = compiled["module_outputs"]["schema_design_module"]["field_registry"]
        adapted = next(
            field
            for field in fields
            if field["field_path"] == "material_info.section1.ionic_conductivity"
        )

        self.assertEqual("number with unit", adapted["data_type"])
        self.assertEqual("A measured transport-property value.", adapted["description"])
        self.assertIn("distinct measurement conditions", adapted["extraction_notes"])

    def test_reference_leaf_type_is_not_replaced_by_object_concept_kind(self):
        state = {
            "shared_context": {
                "query_requirements": ["record transition temperature"],
                "reference_field_contract": {
                    "available": True,
                    "policy": "task_adaptive",
                    "leaf_fields": [
                        {
                            "path": "material_info.section1.Tc.value",
                            "data_type": "number with unit",
                            "description_hint": "Reported transition temperature.",
                            "extraction_notes": "Preserve the stated criterion.",
                        }
                    ],
                },
            },
            "module_outputs": {
                "subjective_supervisor_module": {
                    "requirement_contract": {
                        "concepts": [
                            {
                                "concept_id": "transition_temperature",
                                "label": "Transition temperature",
                                "object_kind": "measurement",
                                "required": True,
                                "source_requirement_ids": ["query_1"],
                            }
                        ]
                    }
                },
                "field_planning_module": {
                    "field_groups": [
                        {
                            "section_id": "material_info.section1",
                            "recommended_fields": ["material_info.section1.Tc.value"],
                            "evidence_strategy": "text and table",
                        }
                    ],
                    "reference_field_audit": {
                        "included_leaf_paths": ["material_info.section1.Tc.value"]
                    },
                },
            },
            "module_attempts": {},
            "module_errors": {},
        }

        compiled = graph.compile_schema_from_field_planning(state, "test reference type")
        field = next(
            item
            for item in compiled["module_outputs"]["schema_design_module"]["field_registry"]
            if item["field_path"] == "material_info.section1.Tc.value"
        )

        self.assertEqual("number with unit", field["data_type"])
        self.assertNotIn("object_contract", field)

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

    def test_postprocess_uses_shared_contracts_without_synthetic_fields(self):
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
        fields = finalized["schema_definition"]["field_registry"]

        self.assertEqual([], report["structured_object_completeness"]["incomplete_fields"])
        self.assertEqual(1, len(fields))
        self.assertIn("literature_provenance_v1", fields[0]["contract_refs"])
        self.assertIn("measurement_context_v1", fields[0]["contract_refs"])
        self.assertIn(
            "literature_provenance_v1",
            finalized["schema_definition"]["shared_record_contracts"],
        )

    def test_finalize_normalizes_sections_without_inventing_traceability(self):
        result = base_result(
            [
                {
                    "field_path": "material_info.section1.magnetic_order",
                    "section_id": "section1",
                    "field_name": "magnetic_order",
                    "data_type": "string",
                    "required": False,
                    "source_basis": ["text"],
                    "concept_ids": [],
                    "reason": "Supports magnetic phase comparison.",
                }
            ]
        )
        result["section_design"]["core_sections"][1]["section_id"] = "section1"

        finalized = agent.finalize_result({}, result)
        fields = finalized["schema_definition"]["field_registry"]
        magnetic_order = next(
            field
            for field in fields
            if field["field_path"] == "material_info.section1.magnetic_order"
        )
        self.assertEqual("material_info.section1", magnetic_order["section_id"])
        self.assertEqual([], magnetic_order["concept_ids"])
        self.assertTrue(magnetic_order["extraction_notes"])
        self.assertFalse(
            any(
                item.get("requirement_id") == "schema_support"
                for item in finalized["requirement_contract"].get(
                    "source_requirements", []
                )
            )
        )
        self.assertIn(
            "material_info.section1.magnetic_order",
            finalized["coverage_report"]["unrelated_domain_fields"],
        )
        self.assertEqual([], agent.validate_result(finalized))

    def test_finalize_recovers_section_architecture_but_not_concept_mapping(self):
        result = base_result(
            [scalar_field("material_info.section1.measurement_conditions.protocol", [])],
            concepts=[
                {
                    "concept_id": "measurement_conditions",
                    "label": "Measurement conditions",
                    "required": True,
                    "entity_id": "material",
                    "owner_key": "material_info",
                    "source_requirement_ids": ["query_1"],
                },
                {
                    "concept_id": "calculated_values_separate",
                    "label": "Calculated values separate",
                    "required": True,
                    "entity_id": "material",
                    "owner_key": "material_info",
                    "source_requirement_ids": ["query_1"],
                },
                {
                    "concept_id": "inferred_quantities_separate",
                    "label": "Inferred quantities separate",
                    "required": True,
                    "entity_id": "material",
                    "owner_key": "material_info",
                    "source_requirement_ids": ["query_1"],
                },
                {
                    "concept_id": "represent_i_v",
                    "label": "Represent I-V",
                    "required": True,
                    "entity_id": "material",
                    "owner_key": "material_info",
                    "object_kind": "scalar",
                    "source_requirement_ids": ["query_1"],
                }
            ],
        )
        result["section_design"]["core_sections"] = [
            {"section_id": "section0", "section_name": "Identity"}
        ]
        result["section_design"]["non_core_sections"] = [
            {"section_id": "sectionX", "section_name": "Placeholder"}
        ]

        finalized = agent.finalize_result({}, result)
        core_ids = {
            item["section_id"] for item in finalized["section_design"]["core_sections"]
        }
        non_core_ids = {
            item["section_id"]
            for item in finalized["section_design"]["non_core_sections"]
        }
        protocol = next(
            field
            for field in finalized["schema_definition"]["field_registry"]
            if field["field_path"]
            == "material_info.section1.measurement_conditions.protocol"
        )

        self.assertEqual(
            {"material_info.section0", "material_info.section1"}, core_ids
        )
        self.assertNotIn("sectionX", non_core_ids)
        self.assertEqual([], protocol["concept_ids"])
        self.assertEqual(
            ["material_info.section1.measurement_conditions.protocol"],
            [
                field["field_path"]
                for field in finalized["schema_definition"]["field_registry"]
            ],
        )
        self.assertEqual(
            {
                "calculated_values_separate",
                "inferred_quantities_separate",
                "measurement_conditions",
                "represent_i_v",
            },
            set(
                finalized["coverage_report"]["required_concept_coverage"][
                    "missing_concept_ids"
                ]
            ),
        )
        self.assertEqual([], agent.validate_result(finalized))


if __name__ == "__main__":
    unittest.main()
