import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_protocol_normalization_gate as gate


def test_generic_scientific_qualifier_canonicalization():
    cases = [
        ("temperature", "zero-temperature", "0 K"),
        ("temperature", "T->0", "0 K"),
        ("pressure", "ambient pressure", "ambient"),
        ("criterion", "Tc_onset", "Tc onset"),
        ("criterion", "90% of normal-state resistance", "90% Rn"),
        ("model", "two-band model", "two-band"),
        ("uncertainty", "±0.2", 0.2),
        ("result_status", "suggested", "author interpretation"),
        ("sample_form", "epitaxial film", "epitaxial thin film"),
    ]
    for key, before, expected in cases:
        after, rule = gate.canonicalize_qualifier(key, before)
        assert after == expected
        assert rule


def test_normalization_gate_preserves_noncanonical_scientific_detail():
    value = "TDO magnetic penetration depth"
    assert gate.canonicalize_qualifier("method", value) == (value, None)


def test_normalize_fact_never_changes_value_unit_or_evidence():
    fact = {
        "target_id": "t1",
        "value": 5.4,
        "unit": "T",
        "evidence_line": 10,
        "qualifiers": {"temperature": "zero-temperature"},
    }
    normalized, changes = gate.normalize_fact(fact)
    assert normalized["value"] == fact["value"]
    assert normalized["unit"] == fact["unit"]
    assert normalized["evidence_line"] == fact["evidence_line"]
    assert normalized["qualifiers"]["temperature"] == "0 K"
    assert changes[0]["rule"] == "temperature_zero"
