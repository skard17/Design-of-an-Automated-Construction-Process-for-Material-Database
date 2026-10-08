# Prompt for Multi-Material Manifest Discovery

## Role
Act as a strict material-target roster builder for scientific materials papers that may contain multiple studied materials, compositions, phases, stacks, or sample variants.

## Goal
Identify the material targets in the paper that should later receive separate fact matching and aggregation.

This is a domain-general materials-database task. Do not assume the paper is about superconductors, magnets, batteries, catalysts, thermoelectrics, structural alloys, semiconductors, or any other single field unless the paper itself says so.

## Output format

Return JSON only.

Schema:
{
  "paper_id": "string",
  "paper_title": "string or null",
  "manifest_version": "v1",
  "material_targets": [
    {
      "target_id": "stable short id",
      "canonical_name": "string",
      "signature_type": "formula_based | family_member | composition_variant | phase_variant | structure_variant | stack_based | elemental | composite_or_device",
      "aliases": ["string", "..."],
      "family_context": "string or null",
      "sibling_targets": ["canonical_name", "..."],
      "must_exclude_aliases": ["string", "..."],
      "evidence_quotes": ["short quote", "..."],
      "target_specificity": "explicit_target_specific | likely_target_specific | ambiguous"
    }
  ],
  "paper_scope_notes": ["string", "..."]
}

## Rules

### 1. Separate targets when target-specific scientific facts exist

Create separate targets if the paper gives target-specific materials facts for:

- different compounds
- different family members
- different substitution members
- different doping or composition levels when they are treated as distinct study objects
- different phases, polymorphs, crystal structures, or processing states
- different stack targets
- different composite/device/sample architectures when the material record would otherwise be mixed

### 2. Expand compact notation

Examples:

- `A = K, Rb` -> `K...` and `Rb...`
- `X = Ti, Zr, Hf` -> each explicit member
- `Ln = La, Nd, Sm, Gd` -> each explicit member if the paper treats them as study objects
- `x = 0.1, 0.2, 0.3` -> separate composition variants if target-specific properties or processing facts are reported
- `M = Fe, Co, Ni` -> each explicit member when the text uses them as separate materials

### 3. Alias rules

Include aliases that later stages may need:

- chemical shorthand
- normalized formula variants
- family shorthand used in the paper

Do not invent speculative aliases.

### 4. Exclusion rules

Each target's `must_exclude_aliases` should include alias variants of sibling targets that are easy to confuse locally.

### 5. Conservative handling

If the paper mentions a family but does not clearly separate all members with target-specific evidence, still list the likely study objects when the family members are central to the paper. Mark `target_specificity` as `ambiguous` or `likely_target_specific` where appropriate.

### 6. Domain-neutrality rules

- Do not promote a material to a target just because it appears as background, a reference material, a substrate, an electrode, a calibration standard, or a comparison-only example.
- Do not discard non-superconducting or non-magnetic papers. The roster is based on whether the paper reports material-specific facts, not on a fixed validation domain.
- When compact notation appears in property tables or figure captions, preserve enough aliases so later stages can attribute each property value to the correct target.

## Input

### Paper ID
{{paper_id}}

### Paper text
{{paper_text}}

### Supplementary information
{{supplementary_information}}
