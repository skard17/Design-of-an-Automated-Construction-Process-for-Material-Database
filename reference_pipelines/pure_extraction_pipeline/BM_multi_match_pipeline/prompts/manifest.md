# Prompt for Multi-Material Manifest Discovery

## Role
Act as a strict superconducting-material roster builder for papers that may contain multiple superconducting materials.

## Goal
Identify the material targets in the paper that should later receive separate fact matching and aggregation.

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
      "signature_type": "formula_based | family_member | stack_based | elemental",
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

### 1. Separate targets when target-specific superconducting values exist

Create separate targets if the paper gives target-specific superconducting results for:

- different compounds
- different family members
- different substitution members
- different stack targets

### 2. Expand compact notation

Examples:

- `A = K, Rb` -> `K...` and `Rb...`
- `X = Ti, Zr, Hf` -> each explicit member
- `Ln = La, Nd, Sm, Gd` -> each explicit member if the paper treats them as study objects

### 3. Alias rules

Include aliases that later stages may need:

- chemical shorthand
- normalized formula variants
- family shorthand used in the paper

Do not invent speculative aliases.

### 4. Exclusion rules

Each target's `must_exclude_aliases` should include alias variants of sibling targets that are easy to confuse locally.

### 5. Conservative handling

If the paper mentions a family but does not clearly separate all members with target-specific superconducting evidence, still list the likely study objects when the family members are central to the paper. Mark `target_specificity` as `ambiguous` or `likely_target_specific` where appropriate.

## Input

### Paper ID
{{paper_id}}

### Paper text
{{paper_text}}

### Supplementary information
{{supplementary_information}}
