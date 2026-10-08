import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import adjudicate_schema_correspondence as adjudicator


def test_validate_items_requires_complete_manual_batch():
    items = adjudicator.validate_items(
        {
            "items": [
                {
                    "manual_path": "material.Tc[].value",
                    "classification": "semantic_equivalent",
                    "system_path": "material.transition_temperature.value",
                    "rationale": "Both store the transition-temperature value.",
                },
                {
                    "manual_path": "material.Tc[].pressure",
                    "classification": "partial_broader_system",
                    "system_path": "material.transition_temperature.conditions",
                    "rationale": "A generic condition container does not preserve pressure explicitly.",
                },
            ]
        },
        ["material.Tc[].value", "material.Tc[].pressure"],
        [
            "material.transition_temperature.value",
            "material.transition_temperature.conditions",
        ],
    )

    assert items[0]["classification"] == "semantic_equivalent"
    assert items[1]["classification"] == "partial_broader_system"


def test_duplicate_equivalent_claims_are_demoted_to_partial():
    items = adjudicator.demote_duplicate_equivalents(
        [
            {
                "manual_path": "manual.a",
                "classification": "semantic_equivalent",
                "system_path": "system.generic_value",
                "rationale": "Claim A.",
            },
            {
                "manual_path": "manual.b",
                "classification": "semantic_equivalent",
                "system_path": "system.generic_value",
                "rationale": "Claim B.",
            },
        ]
    )

    assert {item["classification"] for item in items} == {"partial_broader_system"}


def test_split_retry_batch_preserves_order_and_remainder():
    assert adjudicator.split_retry_batch(["a", "b", "c", "d", "e"], 2) == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_split_retry_batch_disabled_or_not_smaller_returns_original_batch():
    batch = ["a", "b"]
    assert adjudicator.split_retry_batch(batch, 0) == [batch]
    assert adjudicator.split_retry_batch(batch, 2) == [batch]
