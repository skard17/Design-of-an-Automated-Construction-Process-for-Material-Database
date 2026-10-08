"""Answer-free semantic contracts for targeted materials extraction.

The benchmark target manifest intentionally excludes expected values.  This
module enriches those targets with concept and qualifier semantics that a real
materials-database extraction worker would receive from Step 8.  The rules are
domain-neutral: superconductivity-specific labels come from the supplied task
contract, not from hard-coded answers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


QUALIFIER_GUIDANCE = {
    "approximate": "Boolean. True only when the cited value is explicitly approximate, rounded, or marked by words or symbols such as about, approximately, or ~.",
    "basis": "The experimental or computational basis used to support the value or claim. Copy a concise source-supported method or comparison basis.",
    "boundary": "The named phase or property boundary represented by the value. Preserve the direction or event, such as emergence, disappearance, onset, or crossover.",
    "component": "The explicitly named component, branch, band, gap, axis, layer, or contribution to which the value belongs.",
    "composition": "The exact composition or stoichiometric condition attached to the result. Do not replace it with a material-family label.",
    "coupling_regime": "The source-supported coupling classification. Preserve whether it is weak, intermediate, strong, conventional, or otherwise qualified.",
    "criterion": "The exact operational criterion used to define the reported value, such as onset, midpoint, zero response, a stated fraction of the normal-state value, or the paper's explicit reported criterion.",
    "doping": "The exact dopant identity and level bound to the record or measurement.",
    "excitation_frequency": "The excitation or drive frequency, including its unit, used for the stated response.",
    "field": "The magnetic, electric, or other applied field condition attached to the value, including magnitude and unit when stated.",
    "functional": "The explicitly named computational functional.",
    "method": "The measurement, fitting, synthesis, or calculation method that directly produced or supported the value.",
    "method_alias": "The concise conventional name or acronym for the method used in the evidence.",
    "model": "The exact fitting, extrapolation, theoretical, or computational model used to obtain or interpret the value.",
    "orientation": "The source-stated direction, crystallographic axis, parallel/perpendicular relation, or in-plane/out-of-plane orientation.",
    "phase": "The phase, regime, or branch to which the value is bound. Do not merge values across phases.",
    "pressure": "The pressure condition attached to the value, including unit and whether it is ambient when explicitly stated.",
    "purpose": "The scientific purpose for which the measurement or value is used in the source.",
    "quantity": "The physical quantity represented by the value or curve. Use the specific source-supported quantity rather than a broad property family.",
    "relation": "The explicit comparison operator or relation supported by the source, such as greater than, less than, approximately equal, or a bound.",
    "result_status": "The provenance/status of the result: for example directly measured, reported, fitted, calculated, estimated, derived, predicted, or author interpretation. Do not substitute confidence words such as confirmed unless that is the requested status and is explicit in the source.",
    "sample_form": "The physical form and geometry of the material or device, such as bulk, single crystal, polycrystal, thin film, flake, wire, junction, or heterostructure.",
    "scope": "The precise scientific scope or regime to which the statement applies.",
    "series": "The independent variable or series dimension over which a curve or collection varies.",
    "source": "The named source location or artifact, such as a figure, table, metadata record, or cited analysis.",
    "source_type": "The evidence modality: text, table, figure, metadata, calculation, or another explicit source type.",
    "temperature": "The temperature condition attached to the value, including unit and limiting value when stated.",
    "temperature_region": "The named temperature regime or relation to a transition in which the observation applies.",
    "uncertainty": "The numerical or textual uncertainty associated with the value. Do not include the central value.",
    "x_axis": "The physical quantity plotted or varied on the horizontal axis.",
}


def concept_guidance(concept_id: str, label: str) -> str:
    """Derive a concise extraction rule from a task-supplied concept label."""

    text = f"{concept_id} {label}".casefold()
    if any(token in text for token in ("identity", "formula", "composition")):
        return (
            "Return the exact named material, sample, device, formula, or composition bound to "
            "record_key. Do not return a broad material family, paper topic, or descriptive category."
        )
    if any(token in text for token in ("temperature", "critical_field", "critical_current", "gap", "length", "pressure")):
        return (
            "Return the smallest source-supported scalar or concise bound. Separate the unit and "
            "preserve the requested criterion, method, model, phase, orientation, and conditions."
        )
    if any(token in text for token in ("curve", "resistance", "magnetization", "susceptibility", "spectroscopy", "heat_capacity")):
        return (
            "Return a concise description of the measured relation or curve, including the varied "
            "quantity and fixed conditions requested by the qualifier contract."
        )
    if any(token in text for token in ("mechanism", "pairing", "symmetry", "topological", "interpretation")):
        return (
            "Return only the source's explicit claim or interpretation, and preserve whether it is "
            "measured, fitted, calculated, predicted, or author interpretation."
        )
    if any(token in text for token in ("method", "preparation", "synthesis", "computational")):
        return (
            "Return the explicit method or process and preserve material/sample binding, sequence, "
            "conditions, software, model, or functional when requested."
        )
    return (
        "Return the smallest source-supported value or concise scientific statement that answers "
        "this concept for record_key; preserve all requested qualifiers and do not infer missing facts."
    )


def concept_index(metric_contract: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("concept_id")): item
        for item in (metric_contract or {}).get("concepts") or []
        if isinstance(item, dict) and str(item.get("concept_id") or "").strip()
    }


def enrich_targets(
    targets: list[dict[str, Any]],
    metric_contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Attach answer-free Step8 semantics to each benchmark target."""

    concepts = concept_index(metric_contract)
    enriched = []
    for target in targets:
        item = deepcopy(target)
        concept_id = str(item.get("concept_id") or "")
        concept = concepts.get(concept_id) or {}
        label = str(concept.get("label") or concept_id.replace("_", " ")).strip()
        item["concept_contract"] = {
            "label": label,
            "tier": concept.get("tier"),
            "extraction_guidance": concept_guidance(concept_id, label),
        }
        item["qualifier_contract"] = {
            str(key): QUALIFIER_GUIDANCE.get(
                str(key),
                "Copy the smallest source-supported value for this named condition or provenance attribute.",
            )
            for key in item.get("qualifier_keys") or []
        }
        enriched.append(item)
    return enriched
