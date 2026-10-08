import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import section_design_agent as agent


def concept_ids(contract):
    return {item["concept_id"] for item in contract["concepts"]}


def test_expert_review_prose_is_not_converted_into_fields():
    advice = json.dumps(
        {
            "round": 1,
            "verdict": "revise",
            "recommendations": [
                {
                    "issue_type": "entity_model",
                    "recommendation": "Use separate entities for publication and sample.",
                    "acceptance_criteria": ["Every observation references a publication."],
                }
            ],
            "forbidden_information_used": False,
        }
    )
    shared = {
        "query_requirements": ["experimental conditions; uncertainty; evidence"],
        "human_advice": advice,
    }
    subjective = {
        "requirement_contract": {
            "concepts": [
                {
                    "concept_id": "sample_identity",
                    "label": "sample identity",
                    "source_requirement_ids": ["human_advice"],
                }
            ]
        }
    }

    ids = concept_ids(agent.normalize_requirement_contract(shared, subjective))

    assert "sample_identity" in ids
    assert "round_1" not in ids
    assert "recommendations_issue_type_entity_model" not in ids
    assert "every_observation_references_a_publication" not in ids
    assert "forbidden_information_used_false" not in ids


def test_only_explicit_schema_concepts_can_be_added_directly_from_advice():
    advice = json.dumps(
        {
            "recommendations": [{"recommendation": "Preserve provenance."}],
            "schema_concepts": [
                {
                    "concept_id": "evidence_span",
                    "label": "evidence span",
                    "entity_id": "evidence",
                    "owner_key": "evidence_info",
                    "object_kind": "evidence_collection",
                }
            ],
        }
    )
    shared = {"query_requirements": [], "human_advice": advice}

    contract = agent.normalize_requirement_contract(shared, {"requirement_contract": {"concepts": []}})
    concepts = {item["concept_id"]: item for item in contract["concepts"]}

    assert set(concepts) == {"evidence_span"}
    assert concepts["evidence_span"]["owner_key"] == "evidence_info"
    assert concepts["evidence_span"]["source_requirement_ids"] == ["human_advice"]


def test_workflow_phrases_are_not_atomic_database_fields():
    shared = {
        "query_requirements": [
            "Support reliable comparison and retrieval across material systems; experimental conditions; "
            "derive the exact field inventory from the task and literature"
        ],
        "human_advice": "",
    }

    ids = concept_ids(agent.normalize_requirement_contract(shared, {}))

    assert "experimental_conditions" in ids
    assert "support_reliable_comparison" not in ids
    assert "retrieval_across_material_systems" not in ids
    assert "derive_the_exact_field_inventory_from_the_task" not in ids
    assert "literature" not in ids


def test_structured_protocol_still_atomizes_care_requirements():
    shared = {
        "query_requirements": [
            "Retrieve material identity.",
            "Can sample ownership be swapped? Required distinctions: sample owner, conditions.",
        ],
        "human_advice": json.dumps(
            {
                "care_counterfactual_enabled": True,
                "counterfactual_queries": [
                    {
                        "query_id": "care_binding_swap",
                        "query": "Can sample ownership be swapped?",
                        "required_distinctions": ["sample owner", "conditions"],
                    }
                ],
                "schema_concepts": [],
            }
        ),
        "structured_concept_seeding_only": True,
        "care_counterfactual_requirement_ids": ["query_2"],
    }

    contract = agent.normalize_requirement_contract(shared, {})

    assert "query_2" in {
        requirement_id
        for concept in contract["concepts"]
        for requirement_id in concept["source_requirement_ids"]
    }
    sources = {item["requirement_id"]: item["source"] for item in contract["source_requirements"]}
    assert sources["query_2"] == "care_counterfactual_query"


def test_coverage_reports_untraced_care_requirement_separately():
    result = {
        "requirement_contract": {
            "concepts": [],
            "source_requirements": [
                {
                    "requirement_id": "query_2",
                    "source": "care_counterfactual_query",
                    "text": "Preserve sample binding.",
                }
            ],
        },
        "schema_definition": {"top_level_keys": [], "field_registry": []},
    }

    report = agent.build_coverage_report(result)
    errors = agent.validate_coverage_report(report)

    assert report["care_counterfactual_traceability"]["missing_requirement_ids"] == [
        "query_2"
    ]
    assert "coverage:untraced_care_counterfactuals:query_2" in errors
