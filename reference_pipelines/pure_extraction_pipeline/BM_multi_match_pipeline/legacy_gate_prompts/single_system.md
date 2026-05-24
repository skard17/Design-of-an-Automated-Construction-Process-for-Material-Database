# Single-superconducting-system Filter for Superconductivity Experimental Papers

## Role
Act as a strict, engineering-oriented gatekeeper for selecting "single-superconducting-system" papers from superconductivity experimental papers for downstream structured extraction.

## Goal
Decide whether this paper's superconducting-state experimental program is primarily centered on ONE distinct superconducting system (called a "system signature" below), with only ONE system signature meeting the COUNTED_SYS criteria under the strict definition below.
If YES, output the ONE primary system signature to use as the canonical identifier for downstream extraction.
If NO, explain why (e.g., multiple counted signatures, no counted system found, unclear) and list the counted signatures.

## Critical instruction (precision-first; avoid counting background/literature)
- Count a superconducting system ONLY when its superconducting numeric value(s) (e.g., Tc, Hc2, gap, Jc) are explicitly anchored to THIS paper's experimental program as required by Counted_SYS (C) (e.g., Fig./Table citation, sample/device label, or "we measured/grew/fabricated/prepared"); otherwise treat it as background/literature and do NOT count it.

## Core Definition

### 1) What is a "counted superconducting system" (COUNTED_SYS)?
A candidate system is COUNTED_SYS only if ALL conditions below are satisfied:

(A) The paper reports superconducting-state experimental results for this system
    (e.g., transport/magnetization/gap/vortex/electrodynamics results on this system).

(B) The paper provides at least ONE explicit superconducting parameter/value/range in TEXT or CAPTIONS
    tied to this system (examples: "Tc = ... K", "Hc2 = ... T", "gap = ... meV", "dHc2/dT = ...", "Jc = ...", etc.).
    Only use numeric values explicitly stated in the provided text or figure captions; do NOT infer/guess from plots or figure panels (not provided).
    "Explicit value/range" includes:
    - a numeric value with units,
    - a numeric inequality with units,
    - a numeric range with units.
    Non-numeric phrases like "high Tc" do NOT satisfy (B).

(C) The evidence for (A) and (B) must be anchored to THIS paper's experimental program by at least one of:
    - an in-paper figure/table reference ("Fig.", "Figure", "Table", "Extended Data", etc.), OR
    - an explicit sample label used in this paper, OR
    - explicit "we measured / we fabricated / we grew / we prepared" style statements.
    This rule prevents counting background/literature values.

If any of (A)(B)(C) is not satisfied, the candidate system is NOT counted (treat as background/control).

### Literature/background red flags
If a superconducting value is introduced with cues like:
- "as reported previously", "reported in Ref.", "known", "literature", "previous work", "earlier studies",
- or the system appears only in Introduction/Related work without figure/sample/we-measured anchors,
then treat it as background and do NOT count it as COUNTED_SYS unless (C) clearly anchors it to THIS paper.

### 2) single_system key setting
Set single_system=1 iff:

- The number of DISTINCT "system signatures" among COUNTED_SYS is EXACTLY 1.

Otherwise set single_system=0.

### 3) How to define a "system signature" (canonical identifier)
Normalize each COUNTED_SYS into one of the two signature types below.

#### Type F: Formula-based signature (one chemical formula backbone or one explicit formula family)
Use when the superconducting system under study (a COUNTED_SYS) can be represented by:
- one base formula, OR
- one explicit substitution family written as a single formula family.

Rules for Type F:
- Changing ONLY tuning parameters does NOT create a new signature.
- Isotope variants that do NOT change the material identity backbone do NOT create a new signature.
- Changing dopant/defect SPECIES creates a new signature.
- Changing the base compound creates a new signature unless the paper explicitly defines one continuous substitution family.
- Different crystallographic phases/polytypes/structures explicitly treated as distinct materials count as different signatures if BOTH have COUNTED_SYS results in THIS paper.

#### Type S: Stack/heterostructure signature (fixed architecture as the superconducting target)
Use when the superconducting system is explicitly defined as a stack/interface architecture and the paper treats this architecture as the superconducting system under study.

Rules for Type S:
- If superconducting results are attributed to the stack/interface system (as one target), then the whole architecture counts as ONE signature.
- Varying thickness, repetition number N, or substrate isotope within the SAME architecture does NOT create new signatures.
- If the paper reports COUNTED_SYS results for TWO different architectures as co-equal targets, then single_system=0.

### 4) What does NOT create an additional system
Do NOT count a material as an extra system if it is ONLY:
- a substrate/support/buffer/cap/contact/electrode with NO counted superconducting results attributed to it
- a calibration/reference mentioned without this paper's superconducting measurements
- a normal-state control without superconducting-state parameters measured in THIS paper
- a literature comparison without satisfying Counted_SYS rule (A)(B)(C).

## Decision Procedure (must follow strictly)

Step 1) Candidate collection:
- If an Abstract section exists, first identify candidate superconducting systems mentioned in the Title + Abstract.
- Then scan the full text + figure captions + supplementary information to find additional candidate systems.

Step 2) Counted_SYS test:
For each candidate, test whether it satisfies Counted_SYS rule (A)(B)(C).
Keep only those that pass.

Step 3) Signature normalization:
Normalize each Counted_SYS into a system signature (Type F or Type S).

Step 4) Final single_system decision:
- If number of distinct signatures == 1 -> single_system=1
- Else -> single_system=0

Step 5) Primary system selection:
If single_system=1: primary_signature is that single signature.
If single_system=0: if one signature clearly dominates Title/Abstract and most SC results, set it as primary; else null.

## Output Format (JSON ONLY; no extra text; no markdown fences)

Return exactly one JSON object with the schema below.

{
  "single_system": 0 or 1,
  "primary_signature": string or null,
  "primary_signature_type": "formula_based" or "stack_based" or null,
  "counted_signatures": [string, ...],
  "secondary_or_control_materials": [string, ...],
  "decision_reason": one of [
    "single_signature_only",
    "multiple_counted_signatures",
    "no_counted_system_found",
    "uncertain"
  ],
  "evidence": {
    "primary_system_quotes": [string, ...],
    "other_system_quotes": [string, ...]
  }
}

Consistency rules:
- If single_system=1:
  - counted_signatures must have exactly 1 item, and it must equal primary_signature.
  - primary_signature_type must be non-null.
- If single_system=0:
  - primary_signature_type can be null if primary_signature is null; otherwise it should match the chosen primary_signature type.
- If decision_reason="no_counted_system_found":
  - counted_signatures must be [] and primary_signature must be null.

## Input
The following is the paper text to be processed (Markdown). Supplementary information (optional) follows.

{{paper_text}}

{{supplementary_information}}
