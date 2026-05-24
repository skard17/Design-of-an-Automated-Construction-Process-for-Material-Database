# Prompt for Paper-Level Fact Candidate Extraction

## Role
Act as a strict fact candidate extractor for multi-material superconductivity papers.

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

## Output format

Return JSON only.

Schema:
{
  "paper_id": "string",
  "candidate_version": "v1",
  "paper_level_candidates": [
    {
      "candidate_id": "string",
      "fact_type": "Tc | Jc | Hc1 | Hc2 | Hc | P_sc | P_nsc | lambda | xi | electronic_state_tuning_mechanism | carrier_concentration | secondary_phases | stack_descriptor",
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

- `section0`-relevant facts
  - electronic_state_tuning_mechanism
  - carrier concentration
  - secondary phases
  - stack descriptor
- `section1`-relevant facts
  - `Tc`
  - `Jc`
  - `Hc1`
  - `Hc2`
  - `Hc`
  - `P_sc`
  - `P_nsc`
  - `lambda`
  - `xi`

### 1.1 Strict type boundaries

Only emit a candidate if it truly belongs to one of the supported fact types.

Forbidden misclassifications:

- normal-state resistivity is NOT `P_nsc`
- magnetoresistance is NOT `P_nsc`
- heat-capacity jump ratio is NOT `P_sc`
- RRR is NOT `P_sc` or `P_nsc`
- Sommerfeld coefficient is NOT `lambda` or `xi`
- dHc2/dT slope is allowed only as `Hc2` when explicitly tied to upper critical field analysis

Interpretation rules:

- `P_sc` means pressure values or pressure ranges where superconductivity exists
- `P_nsc` means pressure values or pressure ranges where superconductivity is absent or suppressed
- If the paper does not discuss pressure-driven appearance/disappearance of superconductivity, omit `P_sc` and `P_nsc`
- `Tc` is superconducting transition temperature only
- `Hc1/Hc2/Hc` must be magnetic critical fields only
- `Jc` must be critical current density only
- `lambda` and `xi` must be penetration depth or coherence length only

Additional exclusion rules for `section0`:

- Do NOT use `electronic_state_tuning_mechanism` for measurement settings such as field angle, field orientation, or criterion definitions
- Do NOT use `electronic_state_tuning_mechanism` for family-level comparison language such as ionic-size trends, chain-spacing commentary, or Uemura-classification discussion
- Do NOT use `electronic_state_tuning_mechanism` for superconducting gap ratios, `Tc/TF`, or other derived classification metrics
- Do NOT use `carrier_concentration` unless a numeric carrier density or concentration-like value is explicitly given
- Do NOT use `stack_descriptor` for generic crystal-structure labels, `alpha-W` / `bcc` type labels, or qualitative dimensionality statements such as `quasi-1D electronic structure`

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

## Input

### Manifest
{{manifest_json}}

### Paper text
{{paper_text}}

### Supplementary information
{{supplementary_information}}
