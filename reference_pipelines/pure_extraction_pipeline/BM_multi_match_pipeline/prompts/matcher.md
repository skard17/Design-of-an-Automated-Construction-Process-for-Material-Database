# Prompt for Target-Level Fact Matching

## Role
Act as a strict attribution matcher for ONE target material inside a multi-material superconductivity paper.

## Goal
Decide which candidate facts belong to the current target material.

This stage does not extract new values.
It only decides:

- accept
- reject
- ambiguous

## Most important rule

When in doubt, do not accept.

## Output format

Return JSON only.

Schema:
{
  "paper_id": "string",
  "target_id": "string",
  "canonical_name": "string",
  "matched_version": "v1",
  "accepted_candidates": [
    {
      "candidate_id": "string",
      "fact_type": "string",
      "match_decision": "accept",
      "match_confidence": 0.0,
      "reason": "string"
    }
  ],
  "rejected_candidates": [
    {
      "candidate_id": "string",
      "fact_type": "string",
      "match_decision": "reject",
      "match_confidence": 0.0,
      "reason": "string"
    }
  ],
  "ambiguous_candidates": [
    {
      "candidate_id": "string",
      "fact_type": "string",
      "match_decision": "ambiguous",
      "match_confidence": 0.0,
      "reason": "string"
    }
  ],
  "global_notes": ["string", "..."]
}

## Decision heuristics

### Strong accept signals

- current target named in same sentence
- current target named in same figure caption
- current target named in same table column or row
- current target shorthand matches local alias and siblings are absent
- ordered parallel statement where target order is explicit, such as:
  - material list `Ti, Zr, Hf`
  - value list `9.65, 11.05, 9.67`
  - and the current target matches one position in that ordered mapping

### Strong reject signals

- sibling target explicitly named instead
- value appears under sibling table column
- caption panel clearly belongs to sibling

### Ambiguous signals

- family-level statement with no member disambiguation
- multiple sibling materials named together and value not clearly split
- target only implied by global section theme
- parallel numeric list exists but the material-to-value order is not explicit in the local evidence

### Extra caution for `section0`-style facts

For `tuning`, accept only when the statement is truly about a material-defining tuning axis or target-specific structural/compositional change.

Do NOT accept `tuning` when the statement is only:

- a measurement definition
- a family-level summary
- a comparative performance claim
- a normal-state phenomenology statement with no clear material-defining tuning axis

For `stack_descriptor`, accept only for real layered / interface / heterostructure / superlattice descriptions.

Do NOT accept `stack_descriptor` for:

- atomic coordinate tables
- crystallographic parameter tables
- generic structure descriptions that are not stacks

## Required behavior

- use sibling exclusions aggressively
- prefer `ambiguous` over accidental acceptance
- keep reasons short and concrete

## Ordered-list rule

When a sentence or caption clearly gives:

- an ordered list of target materials or members
- and an ordered list of values
- and uses cues such as `respectively`, `for X = Ti, Zr, Hf`, or equivalent positional mapping

then you may assign the value to the target by position.

If that positional mapping is not explicit, do not force the assignment.

## Input

### Manifest
{{manifest_json}}

### Current target
{{target_json}}

### Fact candidates
{{fact_candidates_json}}
