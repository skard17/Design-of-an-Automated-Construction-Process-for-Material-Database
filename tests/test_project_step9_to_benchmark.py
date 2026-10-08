import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import project_step9_to_benchmark as projector


def test_projection_maps_only_primary_executable_values():
    assert projector.project_concept("material_info.section1.tc_onset.value") == (
        "transition_temperature",
        1,
    )
    assert projector.project_concept("material_info.section1.tc_onset.unit") is None
    assert projector.project_concept("material_info.section4.resistivity_curve.figure_number") == (
        "resistance_temperature_curve",
        1,
    )


def test_projection_maps_task_adaptive_quantity_names_without_gold_answers():
    assert projector.project_concept(
        "material_info.section1.superconducting_transition_temperature"
    ) == ("transition_temperature", 1)
    assert projector.project_concept(
        "material_info.section1.upper_critical_field_Hc2_parallel_ab"
    ) == ("upper_critical_field", 1)
    assert projector.project_concept("section5.pairing_mechanism") == (
        "pairing_mechanism",
        1,
    )
    assert projector.project_concept("section5.software_used") == (
        "computational_method",
        1,
    )


def test_projection_qualifiers_come_from_schema_and_sibling_output():
    item = {
        "field_path": "material_info.section1.hc2_parallel.value",
        "value": 12,
        "evidence_text": "Hc2 was fitted using the WHH model.",
    }
    siblings = [
        item,
        {
            "field_path": "material_info.section1.hc2_parallel.criterion",
            "value": "90% Rn",
        },
    ]
    qualifiers = projector.build_qualifiers(
        item, siblings, {"direction", "criterion", "result_status"}
    )
    assert qualifiers == {
        "direction": "parallel",
        "criterion": "90% Rn",
        "result_status": "fitted",
    }


def test_projection_qualifiers_use_task_adaptive_embedded_conditions():
    item = {
        "field_path": "material_info.section1.superconducting_transition_temperature",
        "value": 2.9,
        "conditions": {"method": "specific heat", "pressure": "ambient"},
        "evidence_text": "Tc was estimated from specific heat.",
    }
    qualifiers = projector.build_qualifiers(item, [item], {"method", "pressure"})
    assert qualifiers == {"method": "specific heat", "pressure": "ambient"}


def test_evidence_locator_uses_source_text_without_gold_answers():
    line, score = projector.locate_evidence_line(
        ["unrelated", "The transition occurs at Tc = 2.6 K.", "tail"],
        "The transition occurs at Tc = 2.6 K.",
        "",
    )
    assert line == 2
    assert score == 1.0


def test_evidence_locator_uses_extracted_record_and_value_as_answer_free_anchors():
    line, score = projector.locate_evidence_line(
        [
            "A transition was observed in another sample.",
            "RhGe4, IrGe4 and IrSi4 have Tc values of about 2.6 K, 1.1 K and 2.5 K.",
        ],
        "Clear superconducting transitions were observed.",
        "",
        anchors=("RhGe4", 2.6, "K"),
    )
    assert line == 2
    assert score > 0.2


def test_evidence_locator_prefers_earliest_equal_support():
    line, _ = projector.locate_evidence_line(
        ["RhGe4 has Tc 2.6 K.", "middle", "RhGe4 has Tc 2.6 K."],
        "RhGe4 has Tc 2.6 K.",
        "",
        anchors=("RhGe4", 2.6, "K"),
    )
    assert line == 1
