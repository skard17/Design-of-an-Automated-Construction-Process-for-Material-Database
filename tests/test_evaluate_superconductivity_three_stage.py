import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import evaluate_superconductivity_three_stage as evaluator


def test_manual_schema_flattening_preserves_arrays():
    schema = {"material_info": {"section1": {"Tc": [{"value": None, "pressure": None}]}}}
    assert evaluator.flatten_schema(schema) == [
        "material_info.section1.Tc[].value",
        "material_info.section1.Tc[].pressure",
    ]


def test_generated_object_contract_expands_required_leaf_subfields():
    payload = {
        "result": {
            "schema_definition": {
                "field_registry": [
                    {
                        "field_path": "observation_records.observation_result",
                        "data_type": "array of objects",
                        "object_contract": {
                            "required_subfields": [
                                "value",
                                "unit",
                                "conditions",
                                "evidence",
                            ]
                        },
                    }
                ]
            }
        }
    }

    inventory = evaluator.extract_system_fields(payload)

    assert inventory["declared_field_count"] == 1
    assert inventory["comparison_basis"] == "field_registry_rows"
    assert inventory["comparison_leaf_count"] == 1
    assert inventory["comparison_leaf_fields"] == [
        "observation_records.observation_result"
    ]
    assert inventory["explicit_leaf_count"] == 4
    assert inventory["explicit_leaf_fields"] == [
        "observation_records.observation_result[].value",
        "observation_records.observation_result[].unit",
        "observation_records.observation_result[].conditions",
        "observation_records.observation_result[].evidence",
    ]


def test_field_coverage_uses_registry_rows_not_repeated_object_slots():
    payload = {
        "result": {
            "schema_definition": {
                "field_registry": [
                    {
                        "field_path": "material_info.section1.transition_temperature.value",
                        "data_type": "array of objects",
                        "object_contract": {
                            "required_subfields": ["value", "unit", "evidence"]
                        },
                    }
                ]
            }
        }
    }

    inventory = evaluator.extract_system_fields(payload)

    assert evaluator.comparison_field_paths(inventory) == [
        "material_info.section1.transition_temperature.value"
    ]
    assert inventory["explicit_leaf_count"] == 3


def test_schema_supervisor_report_preserves_failed_expert_gate(tmp_path):
    decision = tmp_path / "supervisor_decision.json"
    decision.write_text(
        json.dumps(
            {
                "decision": "freeze_for_measurement_with_failed_expert_gate",
                "expert_rounds_used": 3,
                "expert_gate_passed": False,
            }
        ),
        encoding="utf-8",
    )

    report = evaluator.schema_supervisor_report(decision, required=True)

    assert report["status"] == "failed"
    assert report["expert_gate_passed"] is False
    assert report["expert_rounds_used"] == 3


def test_alias_match_does_not_expand_generic_conditions():
    gold = [
        "material_info.section1.Tc[].value",
        "material_info.section1.Tc[].pressure",
    ]
    system = [
        "material_info.section1.transition_temperature[].value",
        "material_info.section1.transition_temperature[].conditions",
    ]
    report = evaluator.match_field_inventory(gold, system)
    assert report["covered_leaf_count"] == 1
    assert report["partial_leaf_count"] == 1
    assert report["missing_leaf_count"] == 0


def test_bidirectional_schema_metrics_penalize_unreviewed_extras():
    report = evaluator.match_field_inventory(
        ["material.name", "material.formula"],
        ["material.name", "material.formula", "material.notes"],
    )

    assert report["manual_reference_recall"] == 1.0
    assert report["manual_reference_overlap_precision"] == 2 / 3
    assert report["manual_reference_overlap_f1"] == 0.8
    assert report["manual_reference_jaccard"] == 2 / 3
    assert report["unresolved_extra_count"] == 1
    assert report["design_utility_precision"] is None
    assert not report["set_equality"]


def test_extra_utility_requires_independent_adjudication():
    report = evaluator.match_field_inventory(
        ["material.name"],
        ["material.name", "material.sample_id", "material.todo"],
        {
            "material.sample_id": "useful_supplement",
            "material.todo": "redundant",
        },
    )

    assert report["independently_adjudicated_extra_count"] == 2
    assert report["unresolved_extra_count"] == 0
    assert report["useful_extra_count"] == 1
    assert report["design_utility_precision"] == 2 / 3


def test_independent_semantic_correspondence_counts_one_to_one_equivalence_only():
    report = evaluator.match_field_inventory(
        ["material.Tc[].value", "material.Tc[].pressure"],
        [
            "observations.transition_temperature.value",
            "observations.transition_temperature.conditions",
        ],
        semantic_correspondence={
            "material.Tc[].value": {
                "classification": "semantic_equivalent",
                "system_path": "observations.transition_temperature.value",
                "rationale": "Same atomic value.",
            },
            "material.Tc[].pressure": {
                "classification": "partial_broader_system",
                "system_path": "observations.transition_temperature.conditions",
                "rationale": "Pressure is not explicit.",
            },
        },
    )

    assert report["covered_leaf_count"] == 1
    assert report["partial_leaf_count"] == 1
    assert report["missing_leaf_count"] == 0
    assert report["system_extra_count"] == 1


def test_reused_semantic_system_leaf_cannot_inflate_coverage():
    report = evaluator.match_field_inventory(
        ["manual.a", "manual.b"],
        ["system.generic_value"],
        semantic_correspondence={
            "manual.a": {
                "classification": "semantic_equivalent",
                "system_path": "system.generic_value",
                "rationale": "First claim.",
            },
            "manual.b": {
                "classification": "semantic_equivalent",
                "system_path": "system.generic_value",
                "rationale": "Second claim.",
            },
        },
    )

    assert report["covered_leaf_count"] == 1
    assert report["partial_leaf_count"] == 1


def test_set_equality_rejects_array_cardinality_conflict():
    report = evaluator.match_field_inventory(
        ["material.measurements[].value"],
        ["material.measurements.value"],
    )

    assert report["manual_reference_recall"] == 1.0
    assert report["cardinality_conflict_count"] == 1
    assert not report["set_equality"]


def test_generic_normalizer_has_no_gold_specific_rewrite():
    gold = {
        "paper_id": "p1",
        "concept_id": "transition_temperature",
        "record_key": "RhGe4",
        "value": 2.6,
        "unit": "K",
        "evidence_line": 50,
        "qualifiers": {"result_status": "reported"},
    }
    prediction = dict(gold)
    prediction["value"] = "2.6"
    prediction["evidence_line"] = 51
    assert evaluator.generic_normalized_fact_match(gold, prediction)
    prediction["value"] = "1.1"
    assert not evaluator.generic_normalized_fact_match(gold, prediction)


def test_value_unit_diagnostic_excludes_qualifier_and_evidence_failures():
    gold = {
        "paper_id": "p1",
        "concept_id": "transition_temperature",
        "record_key": "RhGe4",
        "value": 2.6,
        "unit": "K",
        "evidence_line": 50,
        "qualifiers": {"criterion": "reported Tc"},
    }
    prediction = {
        **gold,
        "evidence_line": 300,
        "qualifiers": {"criterion": "onset"},
    }
    assert evaluator.scientific_value_unit_match(gold, prediction)
    assert not evaluator.strict_fact_match(gold, prediction)


def test_semantic_qualifier_normalizer_handles_generic_scientific_variants():
    assert evaluator.qualifier_values_semantically_equal("temperature", "0 K", "zero-temperature")
    assert evaluator.qualifier_values_semantically_equal(
        "criterion", "90% Rn", "90% of normal-state resistance"
    )
    assert evaluator.qualifier_values_semantically_equal("model", "two-band", "two-band model")
    assert evaluator.qualifier_values_semantically_equal("uncertainty", 0.2, "±0.2")
    assert evaluator.qualifier_values_semantically_equal(
        "result_status", "author inference", "suggested"
    )


def test_semantic_qualifier_normalizer_does_not_collapse_distinct_provenance_or_methods():
    assert not evaluator.qualifier_values_semantically_equal(
        "result_status", "measured", "reported"
    )
    assert not evaluator.qualifier_values_semantically_equal(
        "method", "penetration-depth fit", "TDO magnetic penetration depth"
    )


def test_gold_evidence_tokens_allow_a_supported_alternative_source_line(tmp_path):
    pack = tmp_path / "paper-1.md"
    pack.write_text(
        "<!-- SOURCE_LINE: 10 -->\nLaFeAsO 0.89 F 0.11 is in SC1.\n"
        "<!-- SOURCE_LINE: 20 -->\nUnrelated text.\n",
        encoding="utf-8",
    )
    index = evaluator.load_evidence_index(tmp_path)
    gold = {
        "paper_id": "paper-1",
        "evidence_line": 5,
        "evidence_tokens": ["LaFeAsO", "F", "0.11", "SC1"],
    }
    prediction = {"paper_id": "paper-1", "evidence_line": 10}

    assert evaluator.gold_evidence_tokens_supported(gold, prediction, index)
    prediction["evidence_line"] = 20
    assert not evaluator.gold_evidence_tokens_supported(gold, prediction, index)


def test_core_fields_are_not_invented(tmp_path):
    manual = tmp_path / "manual.json"
    manual.write_text(json.dumps({"primary_signature": None}), encoding="utf-8")
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"facts": []}), encoding="utf-8")
    metric = tmp_path / "metric.json"
    metric.write_text(json.dumps({"concepts": []}), encoding="utf-8")
    step8 = tmp_path / "step8.json"
    step8.write_text(
        json.dumps(
            {
                "result": {
                    "schema_definition": {
                        "field_registry": [
                            {"field_path": "primary_signature", "data_type": "string"}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "manual_schema": str(manual),
                "gold_facts": str(gold),
                "metric_contract": str(metric),
                "stages": [{"stage_id": "s1", "step8_output": str(step8)}],
            }
        ),
        encoding="utf-8",
    )
    report = evaluator.build_report(config)
    assert report["evaluation_contract"]["core_field_policy"] == "not_scored_no_user_approved_mapping"
    assert report["stages"][0]["schema"]["core_fields"]["status"] == "not_scored_no_user_approved_mapping"


def test_failed_step8_can_be_audited_without_promoting_it_to_success():
    payload = {
        "status": "needs_review",
        "result": None,
        "module_outputs": {
            "schema_design_module": {
                "field_registry": [
                    {"field_path": "material_info.section4.R_T", "data_type": "array"}
                ]
            }
        },
    }
    inventory = evaluator.extract_system_fields(payload)
    assert inventory["inventory_source"] == "module_outputs.schema_design_module"
    assert inventory["declared_field_count"] == 1


def test_fallback_step8_is_audited_but_not_formally_scored(tmp_path):
    manual = tmp_path / "manual.json"
    manual.write_text(json.dumps({"material_info": {"section0": {"material_id": None}}}), encoding="utf-8")
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"facts": []}), encoding="utf-8")
    metric = tmp_path / "metric.json"
    metric.write_text(json.dumps({"concepts": []}), encoding="utf-8")
    step8 = tmp_path / "step8.json"
    step8.write_text(
        json.dumps(
            {
                "status": "fallback_needs_online_completion",
                "execution_provenance": {
                    "mode": "blind_deterministic_fallback",
                    "fallback_used": True,
                    "all_required_modules_completed": False,
                },
                "result": {
                    "schema_definition": {
                        "field_registry": [
                            {"field_path": "material_info.section0.material_id", "data_type": "string"}
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "manual_schema": str(manual),
                "gold_facts": str(gold),
                "metric_contract": str(metric),
                "stages": [{"stage_id": "fallback", "step8_output": str(step8)}],
            }
        ),
        encoding="utf-8",
    )

    schema = evaluator.build_report(config)["stages"][0]["schema"]

    assert schema["status"] == "audited_unscored_incomplete_step8"
    assert schema["step8_completion_audit"]["fallback_detected"] is True
    assert schema["manual_json_238"]["status"] == "not_scored_incomplete_step8"
    assert schema["prompt_complete_239"]["status"] == "not_scored_incomplete_step8"
    assert schema["core_fields"]["status"] == "not_scored_no_user_approved_mapping"
    assert "strict_or_alias_coverage" not in schema["manual_json_238"]


def test_fixed_field_contract_is_scored_without_claiming_step8_generation(tmp_path):
    manual = tmp_path / "manual.json"
    manual.write_text(
        json.dumps({"material_info": {"section1": {"Tc": [{"value": None}]}}}),
        encoding="utf-8",
    )
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"facts": []}), encoding="utf-8")
    metric = tmp_path / "metric.json"
    metric.write_text(json.dumps({"concepts": []}), encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text(manual.read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "manual_schema": str(manual),
                "gold_facts": str(gold),
                "metric_contract": str(metric),
                "stages": [{"stage_id": "legacy", "field_contract": str(contract)}],
            }
        ),
        encoding="utf-8",
    )

    stage = evaluator.build_report(config)["stages"][0]["schema"]

    assert stage["schema_source_type"] == "fixed_extraction_field_contract"
    assert stage["step8_execution_status"] == "fixed_contract_valid"
    assert stage["manual_json_238"]["strict_or_alias_coverage"] == 1.0


def test_manual_gold_cannot_be_reused_as_the_system_field_contract(tmp_path):
    manual = tmp_path / "manual.json"
    manual.write_text(json.dumps({"material_info": {"section1": {"tc": None}}}), encoding="utf-8")
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"facts": []}), encoding="utf-8")
    metric = tmp_path / "metric.json"
    metric.write_text(json.dumps({"concepts": []}), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "manual_schema": str(manual),
                "gold_facts": str(gold),
                "metric_contract": str(metric),
                "stages": [{"stage_id": "leaked", "field_contract": str(manual)}],
            }
        ),
        encoding="utf-8",
    )

    schema = evaluator.build_report(config)["stages"][0]["schema"]

    assert schema["status"] == "not_scored_reference_schema_leak"
    assert schema["step8_execution_status"] == "reference_schema_leak_rejected"


def test_care_report_accepts_exact_unique_responsibility_metric(tmp_path):
    path = tmp_path / "care.json"
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "case_count": 25,
                "exact_unique_responsibility_accuracy": 1.0,
                "scope": "controlled responsibility replay",
            }
        ),
        encoding="utf-8",
    )

    result = evaluator.care_report(path)

    assert result["gold_case_count"] == 25
    assert result["prediction_case_count"] == 25
    assert result["coverage"] == 1.0
    assert result["accuracy"] == 1.0


def test_component_audit_aligns_targets_and_decomposes_failures():
    gold = [
        {
            "fact_id": "fact-1",
            "paper_id": "paper-1",
            "record_key": "sample-a",
            "concept_id": "transition_temperature",
            "value": 2.6,
            "unit": "K",
            "qualifiers": {"criterion": "onset"},
            "evidence_line": 50,
        },
        {
            "fact_id": "fact-2",
            "paper_id": "paper-1",
            "record_key": "sample-b",
            "concept_id": "critical_field",
            "value": 1.2,
            "unit": "T",
            "qualifiers": {},
            "evidence_line": 80,
        },
    ]
    predictions = [
        {
            "target_id": "fact-1",
            "paper_id": "paper-1",
            "record_key": "sample-a",
            "concept_id": "transition_temperature",
            "value": 2.6,
            "unit": "K",
            "qualifiers": {"criterion": "midpoint", "extra": "allowed"},
            "evidence_line": 51,
        },
        {
            "target_id": "fact-2",
            "paper_id": "paper-1",
            "record_key": "sample-b",
            "concept_id": "critical_field",
            "value": 1.2,
            "unit": "T",
            "qualifiers": {"source": "text"},
            "evidence_line": 80,
        },
    ]

    report = evaluator.component_audit(gold, predictions)

    assert report["aligned_prediction_count"] == 2
    assert report["metrics"]["value"]["matched"] == 2
    assert report["metrics"]["required_qualifiers"]["matched"] == 1
    assert report["metrics"]["evidence_line_exact"]["matched"] == 1
    assert report["metrics"]["evidence_line_within_1"]["matched"] == 2
    assert report["metrics"]["scientific_components_excluding_evidence"]["matched"] == 1


def test_specialized_reports_keep_supervisor_and_core_gates_explicit(tmp_path):
    report = {
        "evaluation_contract": {
            "core_field_policy": "not_scored_no_user_approved_mapping",
            "gold_facts_path": "gold.json",
            "gold_facts_sha256": "ABC",
            "gold_fact_count": 1,
            "gold_paper_count": 1,
        },
        "stages": [
            {
                "stage_id": "stage-1",
                "label": "baseline",
                "ablation": {"care": False, "structured_protocol": False},
                "fresh_run_status": "complete",
                "schema": {
                    "status": "scored",
                    "step8_execution_status": "needs_review",
                    "manual_json_238": {
                        "missing_gold_paths": ["a"],
                        "partial": [],
                        "system_extras": [],
                    },
                    "prompt_complete_239": {},
                    "core_fields": {"status": "not_scored_no_user_approved_mapping"},
                    "system_inventory": {},
                },
                "extraction": {
                    "status": "scored",
                    "overall": {
                        "strict_exact": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                        "generic_normalized": {
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                        },
                    },
                    "component_audit": {
                        "failures": [],
                        "metrics": {
                            name: {"matched": 0, "total": 1, "rate": 0.0}
                            for name in (
                                "value",
                                "unit",
                                "required_qualifiers",
                                "evidence_line_exact",
                                "evidence_line_within_1",
                            )
                        },
                    },
                },
                "care": {"status": "not_applicable_or_not_replayed"},
                "protocol": {"status": "not_applicable"},
            }
        ],
    }

    payloads = evaluator.build_specialized_reports(report)
    paths, qc = evaluator.write_specialized_reports(report, tmp_path)

    assert payloads["QC_REPORT"]["status"] == "not_all_gates_passed"
    assert not qc["all_supervisor_and_evaluation_gates_passed"]
    assert not qc["stages"][0]["checks"]["step8_supervisor_passed"]
    assert len(paths) == 10
    assert all(path.exists() for path in paths)


def test_specialized_markdown_uses_na_for_terminal_unscored_stages():
    field_markdown = evaluator.render_specialized_markdown(
        "FIELD_COVERAGE_REPORT",
        {
            "stages": [
                {
                    "label": "baseline",
                    "step8_execution_status": "needs_review",
                    "manual_json": {"status": "not_scored_incomplete_step8"},
                    "core_fields": {"status": "not_scored_no_user_approved_mapping"},
                }
            ]
        },
    )
    extraction_markdown = evaluator.render_specialized_markdown(
        "EXTRACTION_QUALITY_REPORT",
        {
            "stages": [
                {
                    "label": "baseline",
                    "extraction": {"status": "not_scored_missing_predictions"},
                }
            ]
        },
    )

    assert "0.000" not in field_markdown
    assert "| n/a | n/a | n/a |" in field_markdown
    assert "not_scored_missing_predictions" in extraction_markdown
    assert "0.000" not in extraction_markdown

    failure_markdown = evaluator.render_specialized_markdown(
        "FAILURE_REPORT",
        {
            "stages": [
                {
                    "label": "baseline",
                    "step8_execution_status": "needs_review",
                    "field_evaluation_status": "audited_unscored_incomplete_step8",
                    "extraction_evaluation_status": "not_scored_missing_fresh_predictions",
                    "terminal_supervisor": {},
                    "missing_manual_fields": None,
                    "partial_manual_fields": None,
                    "system_extras": None,
                    "extraction_failures": None,
                }
            ]
        },
    )
    assert "Missing manual fields: n/a" in failure_markdown
    assert "System-only fields: n/a" in failure_markdown


def test_terminal_supervisor_report_preserves_graph_decision(tmp_path):
    state = tmp_path / "step8.state.json"
    state.write_text(
        json.dumps(
            {
                "current_node": "write_output",
                "next_node": "__end__",
                "last_error_node": "evidence_model",
                "last_error_type": "json_parse_failure",
                "retry_counts": {"evidence_model": 4},
                "supervisor_decision": {
                    "action": "finish_needs_review",
                    "reason": "retry budget exhausted",
                    "error_node": "evidence_model",
                    "error_type": "json_parse_failure",
                    "retry_count": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    report = evaluator.terminal_supervisor_report(state)

    assert report["status"] == "terminal_failure_preserved"
    assert report["failure_routed_to_supervisor"] is True
    assert report["terminal_action"] == "finish_needs_review"
    assert report["retry_count"] == 4
