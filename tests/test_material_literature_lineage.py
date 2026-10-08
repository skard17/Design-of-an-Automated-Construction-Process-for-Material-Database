import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import downstream_extraction_runner as runner  # noqa: E402
import materials_agent_protocol as protocol  # noqa: E402
import section_design_agent as step8  # noqa: E402
import step9_extraction_build_graph as step9  # noqa: E402


class MaterialLiteratureLineageTests(unittest.TestCase):
    def test_structured_protocol_is_deterministic_and_task_locked(self):
        first = protocol.make_message(
            sender="field_agent",
            receiver="step8_supervisor",
            phase="field_design",
            status="completed",
            payload_refs={"output": "module_outputs.field_agent"},
            next_route="step8_supervisor",
        )
        second = protocol.make_message(
            sender="field_agent",
            receiver="step8_supervisor",
            phase="field_design",
            status="completed",
            payload_refs={"output": "module_outputs.field_agent"},
            next_route="step8_supervisor",
        )

        self.assertEqual(first["message_id"], second["message_id"])
        self.assertEqual([], protocol.validate_message(first))
        self.assertEqual("automated_materials_database_construction", first["task_contract"]["task_type"])

    def test_legacy_step8_modules_emit_supervisor_protocol_messages(self):
        inputs = {"task_contract": protocol.materials_literature_task_contract()}
        messages = step8.build_step8_protocol_messages(
            inputs,
            {"locating_module": {"status": "success"}, "aggregation": {"status": "success"}},
            {"schema_definition": {"field_registry": []}},
            [],
            {},
        )
        validation = protocol.validate_message_list(messages)

        self.assertTrue(validation["valid"])
        self.assertEqual("step8_supervisor", messages[0]["receiver"])
        self.assertEqual("release_to_step9", messages[-1]["decision"]["action"])

    def test_step8_locks_task_scope_and_versions_field_rules(self):
        result = {
            "section_design": {
                "core_sections": [{"section_id": "material_info.section1"}],
                "non_core_sections": [],
            },
            "schema_definition": {},
        }
        fields = [
            {
                "field_path": "material_info.section1.transition_temperature",
                "section_id": "material_info.section1",
                "description": "Reported transition temperature.",
                "extraction_notes": "Keep each sample and criterion separate.",
                "data_type": "number with unit",
                "required": True,
                "source_basis": ["text", "table"],
                "object_contract": {"object_kind": "measurement"},
            }
        ]

        step8.ensure_literature_field_rule_contract(result, fields)

        self.assertEqual(
            "automated_materials_database_construction",
            result["task_contract"]["task_type"],
        )
        self.assertEqual("scientific_literature_only", result["task_contract"]["source_scope"])
        self.assertIn("arbitrary_scientific_text", result["task_contract"]["excluded_source_classes"])
        field = fields[0]
        self.assertTrue(field["core_field"])
        self.assertTrue(field["field_rule_id"].startswith("field."))
        self.assertTrue(field["field_rule_version"].startswith("sha256:"))
        self.assertTrue(field["relation_constraints"]["condition_binding_required"])

    def test_step9_enriches_legacy_step8_fields_without_changing_paths(self):
        legacy = {
            "result": {
                "schema_definition": {
                    "field_registry": [
                        {
                            "field_path": "material_info.section1.band_gap",
                            "section_id": "material_info.section1",
                            "data_type": "number with unit",
                            "source_basis": ["text"],
                        }
                    ]
                }
            }
        }

        fields = step9.field_registry_from_step8(legacy)
        band_gap = next(field for field in fields if field["field_path"].endswith("band_gap"))

        self.assertEqual("material_info.section1.band_gap", band_gap["field_path"])
        self.assertTrue(band_gap["field_rule_version"].startswith("sha256:"))
        self.assertIn("scientific-literature", band_gap["inclusion_rule"])

    def test_unresolved_is_a_first_class_covered_decision(self):
        payload = runner.sanitize_payload(
            {
                "stage_id": "material_info.section1",
                "section_id": "material_info.section1",
                "extracted_fields": [],
                "missing_fields": [
                    {"field_path": "material_info.section1.transition_temperature", "missing_reason": "not found"}
                ],
                "unresolved_fields": [
                    {
                        "field_path": "material_info.section1.transition_temperature",
                        "candidate_value": 20,
                        "evidence_text": "A feature near 20 K may be the transition.",
                        "source_hint": "Figure 2",
                        "unresolved_reason": "The criterion is not stated.",
                    }
                ],
            }
        )
        checks = runner.validate_stage_payload(payload)
        assessment = runner.assess_stage_yield(
            [checks],
            ["material_info.section1.transition_temperature"],
            stage_id="material_info.section1",
        )

        self.assertEqual([], payload["missing_fields"])
        self.assertEqual("unresolved", payload["unresolved_fields"][0]["decision_status"])
        self.assertEqual(1, checks["unresolved_count"])
        self.assertEqual("unresolved_requires_review", assessment["extraction_outcome"])

    def test_lineage_binds_decision_to_rule_prompt_model_and_validator_versions(self):
        payload = {
            "extracted_fields": [
                {
                    "field_path": "material_info.section1.band_gap",
                    "value": 2.1,
                    "unit": "eV",
                    "evidence_text": "The measured band gap is 2.1 eV.",
                    "source_hint": "Results",
                }
            ],
            "missing_fields": [],
            "unresolved_fields": [],
        }
        specs = [
            {
                "field_path": "material_info.section1.band_gap",
                "field_rule_id": "field.material_info_section1_band_gap",
                "field_rule_version": "sha256:ABC",
            }
        ]

        annotated = runner.annotate_payload_lineage(
            payload,
            specs,
            document_path="paper.md",
            stage_id="material_info.section1",
            model_version="test-model",
        )
        item = annotated["extracted_fields"][0]

        self.assertEqual("extracted", item["decision_status"])
        self.assertTrue(item["record_id"].startswith("record:"))
        self.assertEqual("sha256:ABC", item["lineage"]["field_rule_version"])
        self.assertTrue(item["lineage"]["prompt_version"].startswith("sha256:"))
        self.assertEqual("test-model", item["lineage"]["model_version"])
        self.assertEqual("scientific_literature_only", annotated["task_contract"]["source_scope"])

    def test_complete_unresolved_result_routes_to_review_without_prompt_repair(self):
        section_result = {
            "stage_id": "material_info.section1",
            "section_id": "material_info.section1",
            "status": "passed",
            "summary": {
                "extracted_total": 0,
                "missing_total": 0,
                "unresolved_total": 1,
                "field_coverage_ratio": 1.0,
                "extraction_outcome": "unresolved_requires_review",
            },
            "document_results": [],
        }

        self.assertEqual([], step9.diagnose_section_result(section_result))

    def test_systematic_feedback_builds_a_bounded_impact_manifest(self):
        state = {
            "human_advice": [
                {
                    "feedback_type": "schema_error",
                    "documents": ["paper-a.md"],
                    "field_paths": ["material_info.section1.band_gap"],
                    "advice": "The field definition needs revision.",
                }
            ],
            "workflow_plan": {"section_test_plan": []},
            "prompt_output": {"shared_prompt_context": {"field_index": []}},
            "extraction_eval": {},
        }

        classification, impact = step9.build_feedback_classification_and_impact(state)

        self.assertEqual("systematic_schema_or_rule", classification["feedback_type"])
        self.assertEqual("targeted_reprocessing_required", impact["status"])
        self.assertEqual(["paper-a.md"], impact["affected_documents"])
        self.assertEqual(
            ["material_info.section1.band_gap"],
            impact["affected_field_paths"],
        )
        self.assertEqual("scientific_literature_only", impact["source_scope"])

    def test_impact_supervisor_holds_unbounded_systematic_reprocessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "step9.json"
            state = {
                "args": {"output": str(output_path)},
                "status": "success",
                "impact_manifest": {
                    "status": "needs_impact_scope_confirmation",
                    "task_type": "automated_materials_database_construction",
                    "source_scope": "scientific_literature_only",
                    "affected_documents": [],
                    "affected_stage_ids": [],
                    "affected_field_paths": [],
                },
                "feedback_classification": {"feedback_type": "systematic_schema_or_rule"},
                "protocol_messages": [],
            }

            result = step9.impact_supervisor_node(state)

        self.assertEqual(
            "needs_human_scope_confirmation",
            result["impact_supervisor_decision"]["status"],
        )
        self.assertEqual("write_output", result["next_node"])
        self.assertFalse(result["protocol_messages"][-1]["decision"]["status"] == "approved")

    def test_approved_impact_manifest_drives_selective_rerun_and_preserves_unaffected_output(self):
        field_a = "material_info.section1.structure"
        field_b = "material_info.section1.band_gap"
        workflow_plan = {
            "section_test_plan": [
                {"stage_id": "paper_info", "section_id": "paper_info", "depends_on": []},
                {"stage_id": "stage_a", "section_id": "material_info.section0", "depends_on": ["paper_info"]},
                {"stage_id": "stage_b", "section_id": "material_info.section1", "depends_on": ["stage_a"]},
            ]
        }
        prompt_output = {
            "module_outputs": {
                "section_extraction_prompt_module": {
                    "section_extraction_prompts": {
                        "paper_info": {"output_contract": {"field_specs": [{"field_path": "paper_info.metadata.title"}]}},
                        "stage_a": {"output_contract": {"field_specs": [{"field_path": field_a}]}},
                        "stage_b": {"output_contract": {"field_specs": [{"field_path": field_b}]}},
                    }
                }
            }
        }
        impact = {
            "status": "targeted_reprocessing_required",
            "task_type": "automated_materials_database_construction",
            "source_scope": "scientific_literature_only",
            "affected_documents": ["paper-b.md"],
            "affected_stage_ids": [],
            "affected_field_paths": [field_b],
            "supervisor_approval": {"status": "approved", "action": "approve_selective_reprocess"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper_a = root / "paper-a.md"
            paper_b = root / "paper-b.md"
            paper_a.write_text("Paper A", encoding="utf-8")
            paper_b.write_text("The measured band gap is 2.1 eV.", encoding="utf-8")
            output_dir = root / "run"
            stage_a_dir = output_dir / "stage_a"
            stage_b_dir = output_dir / "stage_b"
            stage_a_dir.mkdir(parents=True)
            stage_b_dir.mkdir(parents=True)
            (stage_a_dir / "paper-b.json").write_text(
                json.dumps({"extracted_fields": [], "missing_fields": [{"field_path": field_a, "missing_reason": "not reported"}]}),
                encoding="utf-8",
            )
            unaffected = stage_b_dir / "paper-a.json"
            unaffected.write_text('{"marker":"preserve-me"}', encoding="utf-8")
            old_target = stage_b_dir / "paper-b.json"
            old_target.write_text('{"marker":"old-target"}', encoding="utf-8")
            llm_payload = json.dumps(
                {
                    "extracted_fields": [
                        {
                            "field_path": field_b,
                            "value": 2.1,
                            "unit": "eV",
                            "evidence_text": "The measured band gap is 2.1 eV.",
                            "source_hint": "Results",
                        }
                    ],
                    "missing_fields": [],
                    "unresolved_fields": [],
                }
            )
            with mock.patch.object(runner.prompt_agent, "litellm_chat", return_value=llm_payload):
                result = runner.run_extraction_bench(
                    workflow_plan,
                    prompt_output,
                    [str(paper_a), str(paper_b)],
                    base_url="http://unused",
                    api_key="unused",
                    model="test-model",
                    output_dir=str(output_dir),
                    max_documents=2,
                    paper_metadata_by_document={str(paper_b): {"title": "Paper B"}},
                    impact_manifest=impact,
                )

            target_payload = json.loads(old_target.read_text(encoding="utf-8"))
            self.assertEqual("completed", result["status"])
            self.assertEqual([str(paper_b)], result["selective_reprocess"]["selected_documents"])
            self.assertEqual(["paper_info", "stage_a"], result["selective_reprocess"]["dependency_stage_ids"])
            self.assertEqual([field_b], result["selective_reprocess"]["selected_field_paths_by_stage"]["stage_b"])
            self.assertEqual(2.1, target_payload["extracted_fields"][0]["value"])
            self.assertEqual('{"marker":"preserve-me"}', unaffected.read_text(encoding="utf-8"))
            self.assertTrue(target_payload["protocol_envelope"]["message_id"].startswith("message:"))

    def test_step9_graph_runs_new_supervisor_chain_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            step8_path = root / "step8.json"
            document_path = root / "paper.md"
            output_path = root / "step9.json"
            step8_path.write_text(
                json.dumps(
                    {
                        "result": {
                            "task_contract": protocol.materials_literature_task_contract(),
                            "schema_definition": {
                                "field_registry": [
                                    {
                                        "field_path": "material_info.section1.band_gap",
                                        "section_id": "material_info.section1",
                                        "data_type": "number with unit",
                                        "description": "Reported electronic band gap.",
                                        "source_basis": ["text", "table"],
                                    }
                                ]
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            document_path.write_text("The band gap is 2.1 eV.", encoding="utf-8")
            args = step9.build_parser().parse_args(
                [
                    "--step8-output",
                    str(step8_path),
                    "--output",
                    str(output_path),
                    "--test-documents",
                    str(document_path),
                    "--dry-run",
                    "--skip-code-generation",
                    "--skip-human-expert-review",
                ]
            )
            args_dict = vars(args).copy()
            args_dict.pop("resume_from_state")
            result = step9.build_step9_graph().invoke(
                {"args": args_dict, "next_node": "load_inputs"},
                config={"configurable": {"thread_id": "test-supervisor-chain"}},
            )
            output = json.loads(output_path.read_text(encoding="utf-8"))

        phases = [message["phase"] for message in output["protocol_messages"]]
        self.assertEqual("approved", output["impact_supervisor_decision"]["status"])
        self.assertIn("feedback_classification", phases)
        self.assertIn("impact_analysis", phases)
        self.assertIn("impact_supervision", phases)
        self.assertTrue(output["protocol_validation"]["valid"])
        self.assertEqual(str(output_path), result["output"])

    def test_new_protocol_preserves_magnetic_fields_without_superconducting_injection(self):
        magnetic_paths = {
            "material_info.section0.magnetic_order",
            "material_info.section1.Curie_temperature",
            "material_info.section1.coercive_field",
            "material_info.section4.magnetization_vs_field",
            "section5.magnetic_interactions",
        }
        step8_output = {
            "result": {
                "task_contract": protocol.materials_literature_task_contract(),
                "schema_definition": {
                    "field_registry": [
                        {
                            "field_path": path,
                            "section_id": path.rsplit(".", 1)[0],
                            "data_type": "structured measurement record",
                            "description": "Magnetic-material database field.",
                            "source_basis": ["text", "table", "figure"],
                        }
                        for path in sorted(magnetic_paths)
                    ]
                },
            }
        }
        workflow = step9.build_workflow_plan(
            step8_output,
            {},
            {"available": False, "strategies": []},
        )
        prompt_output = step9.build_prompt_package_from_step8(step8_output, workflow)
        generated_paths = {
            item["field_path"]
            for item in prompt_output["shared_prompt_context"]["field_index"]
        }

        self.assertTrue(magnetic_paths.issubset(generated_paths))
        self.assertFalse(any("superconduct" in path.casefold() for path in generated_paths))
        self.assertEqual(
            "scientific_literature_only",
            prompt_output["shared_prompt_context"]["task_contract"]["source_scope"],
        )


if __name__ == "__main__":
    unittest.main()
