import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import adjudicate_schema_correspondence_local as local_adjudicator


def test_local_adjudicator_counts_only_atomic_synonyms_as_equivalent():
    manual = [
        "paper_info.metadata.title",
        "material_info.section1.Tc[].value",
        "paper_info.resources.software[]",
    ]
    system = [
        "paper_info.title",
        "material_info.section1.tc_onset.value",
        "paper_info.authors",
    ]

    items, extras = local_adjudicator.adjudicate(manual, system)

    assert [item["classification"] for item in items] == [
        "semantic_equivalent",
        "partial_narrower_system",
        "missing",
    ]
    assert {item["system_path"] for item in extras} == {
        "material_info.section1.tc_onset.value",
        "paper_info.authors",
    }


def test_local_adjudicator_does_not_reuse_one_leaf_for_two_equivalents():
    manual = ["material_info.section3.STM[].figure", "material_info.section3.STS[].figure"]
    system = ["material_info.section3.stm.figure_number"]

    items, _ = local_adjudicator.adjudicate(manual, system)

    assert items[0]["classification"] == "semantic_equivalent"
    assert items[1]["classification"] == "partial_broader_system"


def test_local_extra_classifier_separates_utility_from_workflow_artifacts():
    assert local_adjudicator.classify_extra("paper_info.authors")[0] == "useful_supplement"
    assert (
        local_adjudicator.classify_extra(
            "material_info.section1.experimental_or_computational_conditions"
        )[0]
        == "workflow_artifact"
    )
    assert (
        local_adjudicator.classify_extra("material_info.section1.tc_zero.criterion")[0]
        == "reasonable_specialization"
    )
