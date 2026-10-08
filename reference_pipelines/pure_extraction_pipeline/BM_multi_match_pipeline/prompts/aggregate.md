# Prompt for Target-Level Aggregation

## Role
Act as a conservative aggregator that converts accepted matched facts into the final structured target JSON.

## Goal
Build one final target record for one material.

This is a domain-general materials-database aggregation task. Preserve superconducting, magnetic, electrochemical, catalytic, thermoelectric, mechanical, optical, electronic, structural, synthesis, and characterization facts when they are accepted and evidence-backed.

You must only use:

- the target definition
- accepted candidates
- ambiguous candidates only for quality-control notes

Do not resurrect rejected candidates.

## Output format

Return JSON only.

Schema:
{
  "paper_id": "string",
  "target_id": "string",
  "canonical_name": "string",
  "primary_signature": "string",
  "aliases_used": ["string", "..."],
  "excluded_siblings": ["string", "..."],
  "material_info": {
    "section0": {},
    "section1": {},
    "section2": [],
    "section3": {},
    "section4": {}
  },
  "section5": {},
  "provenance": {
    "accepted_candidate_ids": ["string", "..."],
    "ambiguous_candidate_ids": ["string", "..."]
  },
  "quality_control": {
    "has_target_specific_property_evidence": true,
    "ambiguity_flags": ["string", "..."],
    "omission_reasons": ["string", "..."]
  }
}

## Aggregation rules

### 1. Conservative population

Populate `section0` and `section1` only from accepted candidates.

General field mapping:

- `section0` stores target-defining and contextual material facts, such as:
  - material identity, composition, structure, phase, synthesis method, processing condition, morphology, stack descriptor, secondary phases, mechanisms, trends, and characterization summaries
- `section1` stores evidence-backed material property and performance records, keyed by `property_name` when available:
  - examples: `Tc`, `Curie_temperature`, `Neel_temperature`, `magnetization`, `coercivity`, `band_gap`, `ionic_conductivity`, `specific_capacity`, `overpotential`, `Seebeck_coefficient`, `thermal_conductivity`, `hardness`

Never place target identity, synthesis-only, processing-only, or background facts inside `section1`.
Never place measured/computed property values inside `section0` when they have a clear property name.

Normalize incoming candidate fact types to old single-pipeline field names:

- `tuning` -> `electronic_state_tuning_mechanism`
- `secondary_phase` -> `secondary_phases`
- if a candidate already uses the old single-pipeline field name, keep it as-is

For domain-general property candidates:

- If `property_name` exists, use it as the `section1` key after light normalization.
- If `property_name` is absent but `fact_type` itself is a specific property name, use `fact_type` as the key.
- Keep the original `fact_type`, `property_name`, and `property_category` inside each record when present.

### 2. Repeated facts

If multiple accepted candidates support the same fact type:

- keep multiple entries when they differ by conditions
- do not overwrite silently

### 3. Ambiguity tracking

Use ambiguous candidates only to populate:

- `ambiguity_flags`
- `omission_reasons`
- `provenance.ambiguous_candidate_ids`

### 4. Zero contamination policy

If a field cannot be filled cleanly from accepted candidates, omit it.

### 5. Preserve fact typing

Do not rename fact types.
Do not reinterpret a candidate into a different scientific quantity during aggregation.

## Input

### Current target
{{target_json}}

### Matched facts
{{matched_json}}

### Original fact candidates
{{fact_candidates_json}}
