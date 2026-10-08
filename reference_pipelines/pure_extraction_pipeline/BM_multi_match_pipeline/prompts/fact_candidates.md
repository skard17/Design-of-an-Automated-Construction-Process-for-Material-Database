# Prompt for Paper-Level Fact Candidate Extraction

## Role
Act as a strict fact candidate extractor for multi-material scientific materials papers.

## Goal
Extract paper-level candidate facts first, without forcing final ownership too early.

Each candidate fact must preserve:

- the value
- the evidence
- the local nearby material mentions
- the source anchor
- a rough attribution hint

## Important principle

Do not force every fact to one material.

It is acceptable for a fact to remain:

- family-level
- multi-target
- unclear

That is better than contaminating a target.

This is a domain-general materials-database task. Extract facts for the material domain actually described by the paper. Superconducting, magnetic, electrochemical, catalytic, thermoelectric, mechanical, optical, electronic, structural, synthesis, and characterization facts are all eligible when they are explicitly supported.

## Output format

Return JSON only.

Schema:
{
  "paper_id": "string",
  "candidate_version": "v1",
  "paper_level_candidates": [
    {
      "candidate_id": "string",
      "fact_type": "material_identity | composition | structure | phase | synthesis_method | processing_condition | property_value | transition_temperature | critical_field | critical_current_density | carrier_concentration | defect_or_doping | secondary_phases | stack_descriptor | morphology | measurement_condition | characterization_result | computational_result | performance_metric | mechanism_or_interpretation | relation_or_trend | other_supported_fact",
      "property_name": "string or null",
      "property_category": "string or null",
      "value": "string or object",
      "unit": "string or null",
      "verbatim_evidence": "short verbatim quote",
      "source_anchor": {
        "section_label": "string or null",
        "figure": "string or null",
        "table": "string or null"
      },
      "local_material_mentions": ["string", "..."],
      "local_series_mentions": ["string", "..."],
      "attribution_hint": {
        "candidate_targets": ["canonical_name", "..."],
        "reason": "same_sentence | same_caption | same_table_column | nearby_paragraph | family_level | unclear"
      },
      "conditions": {
        "temperature": "string or null",
        "magnetic_field": "string or null",
        "pressure": "string or null",
        "composition": "string or null",
        "sample_form": "string or null",
        "atmosphere": "string or null",
        "time": "string or null",
        "frequency": "string or null",
        "voltage": "string or null",
        "current": "string or null",
        "measurement_method": "string or null",
        "direction": "string or null",
        "characteristics": "string or null"
      },
      "confidence": 0.0
    }
  ],
  "paper_level_unassigned_notes": ["string", "..."]
}

## Extraction rules

### 1. Facts to extract

Focus on:

- target-defining material information:
  - composition, formula, doping level, phase, crystal structure, morphology, sample form, stack/composite/device architecture
- synthesis and processing information:
  - preparation method, precursor, temperature, atmosphere, time, pressure, annealing, growth, deposition, calcination, sintering, post-treatment
- measured or computed material properties:
  - use `fact_type = "property_value"` for most scalar or categorical properties
  - set `property_name` to the paper's specific property name, e.g. `Tc`, `Curie_temperature`, `Neel_temperature`, `magnetization`, `coercivity`, `band_gap`, `ionic_conductivity`, `capacity`, `overpotential`, `Seebeck_coefficient`, `thermal_conductivity`, `hardness`
  - set `property_category` when useful, e.g. `superconducting`, `magnetic`, `electrochemical`, `catalytic`, `thermoelectric`, `mechanical`, `optical`, `electronic`, `structural`
- characterization and interpretation facts:
  - phase identification, structure refinement, spectroscopy/microscopy findings, computational results, trends, proposed mechanisms

### 1.1 Strict type boundaries

Only emit a candidate if it is explicitly supported and has enough local context to be useful for a database.

Boundary rules:

- Do not force a fact into a domain-specific label just because the old schema had that label. Use `property_value` plus `property_name` for new domains.
- Do not treat measurement settings as material properties unless the paper reports the setting as the studied variable or processing condition.
- Do not treat background comparison materials as target-specific facts unless the manifest marks them as targets.
- Do not treat family-level trends as target-specific values. Extract them as `relation_or_trend` with `attribution_hint.reason = "family_level"` when useful.
- Do not invent units, values, compositions, or target names.
- If the paper uses a domain-specific symbol, keep it as written in `property_name` and preserve the evidence quote.

Examples of allowed `property_name` values are illustrative only; they are not a closed vocabulary. The extraction should follow the paper and the downstream database purpose.

### 2. Local context only

For `local_material_mentions`, use the immediate context around the fact:

- same sentence
- same caption
- same table row or column header when explicit
- immediately adjacent sentence if needed

Do not use paper-wide intuition.

### 3. Candidate granularity

Create one candidate per atomic claim whenever possible.

If a sentence says "6.1 K and 4.8 K in K... and Rb..., respectively", create two candidates.

### 4. Family-level facts

If a statement is clearly family-level and not target-specific:

- still extract it if useful
- mark `attribution_hint.reason` as `family_level`
- keep `candidate_targets` broad or empty if necessary

### 5. Quotes

`verbatim_evidence` should be short and exact.

### 6. Better omission than wrong typing

If a value is scientifically important but does not fit the supported fact types, do not force it into a nearby type.
Leave it out and mention it in `paper_level_unassigned_notes`.

### 7. Example guidance

The examples below illustrate attribution behavior, not a fixed domain:

- Superconducting validation-style fact: `property_name = "Tc"`, `value = "6.1"`, `unit = "K"`, `property_category = "superconducting"`.
- Magnetic validation-style fact: `property_name = "Curie_temperature"`, `value = "365"`, `unit = "K"`, `property_category = "magnetic"`.
- General materials fact: `property_name = "band_gap"` or `property_name = "specific_capacity"` when those are the explicit values reported by the paper.

## Input

### Manifest
{{manifest_json}}

### Paper text
{{paper_text}}

### Supplementary information
{{supplementary_information}}
