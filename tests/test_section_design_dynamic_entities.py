import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import section_design_agent as agent
import section_design_langgraph_human_gate as graph


def test_safe_declared_entity_owner_is_preserved():
    concept = {
        "concept_id": "sample_id",
        "label": "sample identifier",
        "owner_key": "sample_info",
        "object_kind": "scalar",
    }

    assert agent.is_safe_entity_owner_key("sample_info") is True
    assert graph.concept_owner_prefix(concept) == "sample_info"


def test_safe_plural_entity_owner_is_preserved():
    concept = {
        "concept_id": "observation_id",
        "label": "observation identifier",
        "owner_key": "observations",
        "object_kind": "identifier",
    }

    assert agent.is_safe_entity_owner_key("observations") is True
    assert graph.concept_owner_prefix(concept) == "observations"


def test_unsafe_owner_falls_back_to_material_owner():
    concept = {
        "concept_id": "sample_id",
        "label": "sample identifier",
        "owner_key": "../../unsafe",
        "object_kind": "scalar",
    }

    assert agent.is_safe_entity_owner_key("../../unsafe") is False
    assert graph.concept_owner_prefix(concept) == "material_info.section1"


def test_entity_registry_accepts_safe_independent_owner():
    contract = {
        "concepts": [
            {
                "concept_id": "evidence_span_id",
                "entity_id": "evidence_span",
                "owner_key": "evidence_info",
            }
        ],
        "source_requirements": [{"text": "evidence span"}],
    }
    subjective = {
        "entity_registry": [
            {
                "entity_id": "evidence_span",
                "owner_key": "evidence_info",
                "independent_owner": True,
            }
        ]
    }

    entities = agent.normalize_entity_registry(subjective, contract)

    assert entities[0]["owner_key"] == "evidence_info"
    assert entities[0]["independent_owner"] is True
