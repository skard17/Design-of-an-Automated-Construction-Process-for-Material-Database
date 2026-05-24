# Prompt for Section0: General Information Extraction

## Role
Act as an information extractor specializing in superconductivity.

## Task
Read the provided experimental superconducting-materials paper (including title, abstract, main text, figure captions, supplementary information if provided) and extract specific information according to the requirements below.

Inputs (provided verbatim in the `## Input` section below):
- Single-system identification result (JSON)
- Paper text (Markdown)
- Supplementary information (if provided)

Single-system identification result (JSON) rule: use it only for primary-system disambiguation; do NOT treat it as evidence for extracting any section0 key/value.

# Output Format and Constraints

## Output Structure
Output pure JSON text only. No explanations, no markdown code blocks (no ```json or ```).
Output exactly one JSON object with the top-level key `"section0"`.
- `"section0"` must be a JSON object describing the ONE primary superconducting material/system studied in this paper.
- The primary system should align with the provided single-system identification result (e.g., its `primary_signature`).
- **Always include `"section0"`**: if no section0 information is extractable, output `{"section0": {}}`.

## Output Content
All keys inside `"section0"` are **CONDITIONAL**:
- Output a key (or sub-key for nested objects) **only if** explicitly supported by the paper text (title/abstract/main text/figure captions/supplementary information). Do NOT infer, invent, summarize, or interpret.
- If a key is not supported, OMIT it entirely—no `null` or empty placeholders.
- Do NOT add keys beyond the schema. Key names must match exactly as defined in `JSON Keys and Extraction Rules` (case-sensitive).
- Interpretation of "REQUIRED": means "required only IF the parent is being output"; not a reason to infer unsupported parents.

Array-valued keys in this prompt:
- If an allowed key is defined as an array (e.g., `electronic_state_tuning_mechanism`, `secondary_phases`, `carrier_concentration`, `stack_descriptor`), then:
  - If you include the key, its value MUST be a JSON array, even if there is only ONE record (i.e., output `[ { ... } ]`, not `{ ... }`).
  - Do NOT output empty arrays. If there is no explicitly supported record, omit the key entirely.
- Each array item (one `{...}`) represents ONE distinct, self-consistent record about the primary system, as explicitly stated in the paper.
- Split into multiple array items ONLY when the paper explicitly provides multiple distinct records that cannot be represented as a single record without losing meaning (e.g., different methods, different conditions, or clearly different sample architectures).

## JSON Validity
Preserve the original text content (including LaTeX, special symbols, etc.) when extracting. Only escape characters that would break JSON validity:
- `\` → `\\`
- `"` → `\"`
- newline → `\n`

## JSON Keys and Extraction Rules

### 1) `electronic_state_tuning_mechanism`

**Definition**: Record the designed tuning mechanism(s) used for the primary system (chemical/defect tuning or carrier tuning by gating/interface).

**When to include the top-level key `electronic_state_tuning_mechanism`**:
- If the paper explicitly gives ANY compositional/defect parameter for the primary system (e.g., `x`, `δ`, `y`, nominal compositions, or ranges such as "A1-xBx...", "O7-δ", "x=0.05–0.20").
- OR if the paper explicitly describes tuning via `electrostatic_gating` or `interface_charge_transfer` (even if no `x/δ/y` is used).

**Sub-keys (each item is one mechanism record)**:
- **`tuning_type`** (string, REQUIRED if an `electronic_state_tuning_mechanism` item is included; enum): `cation_substitution | anion_substitution | oxygen_nonstoichiometry | vacancy_defect | interstitial | intercalation | isovalent_substitution | interface_charge_transfer | electrostatic_gating | unknown`
  - Unified classification principle: classify by the *physical tuning mechanism that changes the electronic state* of the primary system.
    - Chemical/defect tuning (composition/defect changes the lattice + electronic structure): `cation_substitution | anion_substitution | oxygen_nonstoichiometry | vacancy_defect | interstitial | intercalation | isovalent_substitution`
    - Carrier tuning without changing chemical composition: `electrostatic_gating | interface_charge_transfer`
  - Tie-breaker / priority rules to avoid overlaps:
    - If the paper explicitly uses gating language (e.g., gate voltage, EDLT, ionic liquid gating) -> `electrostatic_gating`
    - Else if the paper explicitly attributes tuning to interface charge transfer / charge reservoir / interface doping -> `interface_charge_transfer`
    - Else if oxygen is written as `O(7-δ)` / "oxygen deficiency δ" / "oxygen content y" -> `oxygen_nonstoichiometry` (preferred over `vacancy_defect`)
    - Else if the paper explicitly says "isovalent substitution" -> `isovalent_substitution` (preferred over cation/anion substitution)
    - Else if it is substitution on a positive-ion site -> `cation_substitution`; on a negative-ion site -> `anion_substitution`
    - Else if the paper explicitly says "vacancy/deficiency" of a species (not specifically framed as oxygen nonstoichiometry) -> `vacancy_defect`
    - Else if the paper explicitly says "intercalation"/"intercalated" (layered host with inserted guest species) -> `intercalation`
    - Else if the paper explicitly says "interstitial" -> `interstitial`
    - Otherwise -> `unknown` (do NOT guess).
- **`dopant_defect`** (string, optional): the dopant species or defect explicitly stated as the tuning agent; copy the paper's wording (e.g., "Sr", "Co", "F", "oxygen vacancy", "Fe vacancy"). Do NOT infer a dopant/defect from a formula if it is not explicitly stated.
- **`parameter`** (string, optional): the tuning parameter symbol/name exactly as written in the paper (e.g., `x`, `δ`, `y`, `Vg`, "gate voltage", "oxygen content"). Do NOT rename/standardize symbols; do NOT invent a parameter name.
- **`stoichiometry`** (string, optional): the explicit chemical formula/family/composition expression tied to this tuning mechanism, copied as written (include parameter values/ranges if provided), e.g., "La2−xSrxCuO4 (x=0.10–0.20)", "YBa2Cu3O7−δ", "FeSe1−xTex", "KxFe2−ySe2". Use this for composition/defect/intercalation mechanisms; for non-compositional tuning (e.g., electrostatic gating / interface charge transfer) OMIT `stoichiometry` unless the paper explicitly provides a chemical formula/composition for the tuned layer/sample. Do NOT rewrite into a different format.
- **`carrier_effect`** (string, optional; enum): `electron | hole | none_isovalent | mixed_or_ambiguous | unknown` (ONLY if explicitly stated; do NOT infer).

**Notes**:
- Do NOT output placeholder items where all sub-keys are missing.
- One item should correspond to one independent mechanism/dopant/defect description in the paper.

Example (format only; values shown are illustrative placeholders—do NOT copy them):
```json
{
  "electronic_state_tuning_mechanism": [
    {
      "tuning_type": "cation_substitution",
      "dopant_defect": "Sr",
      "parameter": "x",
      "stoichiometry": "La2-xSrxCuO4; x=0.10–0.20"
    },
    {
      "tuning_type": "electrostatic_gating",
      "parameter": "Vg (gate voltage)"
    }
  ]
}
```

### 2) `carrier_concentration`

**Definition**: Record any explicitly stated carrier level/density for the primary system (e.g., p/n, holes per Cu, Hall carrier density).

**When to include the top-level key `carrier_concentration`**:
- Only if the paper explicitly provides a carrier level/density (e.g., `p=...`, `n=...`, "holes/Cu", `cm^-3`, `cm^-2`). Do NOT infer from `x/δ/y`.

**Output format**:
- `carrier_concentration` MUST be a JSON array of objects.
- Each item is ONE explicitly stated carrier concentration record for the primary system.

**When to split into multiple items** (create multiple `{...}` inside the array):
- The paper explicitly reports multiple carrier concentrations that are distinct, for example:
  - different determination methods (e.g., Hall vs ARPES vs chemical-counting), OR
  - different conditions explicitly tied to the value (e.g., different temperatures, gate voltages, doping levels), OR
  - different bases/normalizations/definitions.
- If a single statement reports a range (e.g., `n = 1–3×10^21 cm^-3`), keep it as ONE item and record the range verbatim in `level`.

**Sub-keys**:
- **`level`** (number or string, REQUIRED in each `carrier_concentration` item): numeric level/density as written (e.g., `0.16`, `3×10^14`, `1.2×10^21`).
- **`unit`** (string, REQUIRED in each `carrier_concentration` item): unit as written (e.g., `holes/Cu`, `e/Fe`, `cm^-2`, `cm^-3`).
- **`basis`** (string, optional): the reference/definition for the carrier level (i.e., what it is normalized to, how it is defined, or how it is determined), only if explicitly stated (e.g., "CuO2 plane" / "per Cu in CuO2 plane", "per Fe", "Hall-derived carrier density", "sheet (2D) density", "bulk (3D) density").
- **`conditions`** (string, optional): the explicit condition(s) under which this carrier concentration value applies (ONLY if explicitly stated), copied verbatim as a short phrase. Examples: "T=20 K", "at 300 K", "Vg=3 V", "under 2 GPa", "at x=0.10".

Separation rule (keep it clean and non-overlapping):
- Put "what the number means" (definition/normalization/method-of-determination) in `basis`.
- Put "when/under what external state" (temperature/pressure/gate voltage/magnetic field/doping-state qualifier explicitly tied to the value) in `conditions`.

Minimality rule (when to actually output `conditions`):
- Prefer using `basis` alone to distinguish different `carrier_concentration` items whenever possible.
- Output `conditions` ONLY when you output MULTIPLE `carrier_concentration` items AND (without `conditions`) the items would be ambiguous or indistinguishable because their `basis` is identical/missing/insufficient to separate them.
- If `basis` already cleanly distinguishes the items, OMIT `conditions` even if conditions are mentioned in the paper.

Example (format only; values shown are illustrative placeholders—do NOT copy them):
```json
{
  "carrier_concentration": [
    {
      "level": 0.16,
      "unit": "holes/Cu",
      "basis": "CuO2 plane"
    }
  ]
}
```

Example (when `conditions` is needed to disambiguate; format only):
```json
{
  "carrier_concentration": [
    {
      "level": "3×10^14",
      "unit": "cm^-2",
      "basis": "Hall-derived sheet density",
      "conditions": "T=20 K"
    },
    {
      "level": "2×10^14",
      "unit": "cm^-2",
      "basis": "Hall-derived sheet density",
      "conditions": "T=300 K"
    }
  ]
}
```

### 3) `secondary_phases`

**Definition**: Record any explicitly stated secondary/additive/impurity phases within the sample (multiphase/composite/phase-separated).

**When to include the top-level key `secondary_phases`**:
- Only if the paper explicitly states the sample is multiphase/composite/phase-separated or contains a secondary/additive phase in the sample.

**Sub-keys (each item is one phase)**:
- **`phase`** (string, REQUIRED if a `secondary_phases` item is included): phase name/formula as written (e.g., "Y2BaCuO5 (211)", "BaCuO2", "Ag", "FeAs", "NiO").
- **`role`** (string, optional): role label ONLY if explicitly stated in the paper; copy the paper's wording when possible (e.g., "secondary phase", "impurity phase", "precipitate"). Omit `role` if the paper does not give a meaningful/explicit role label.
- **`fraction`** (object, optional): include only if an amount is explicitly stated. Only output `fraction.value/unit` if explicitly stated in the paper.
  - `value` (number or string, REQUIRED if `fraction` is included)
  - `unit` (string, REQUIRED if `fraction` is included; e.g., `wt.%`, `vol.%`, `mol.%`)

Example (format only; values shown are illustrative placeholders—do NOT copy them):
```json
{
  "secondary_phases": [
    {
      "phase": "Y2BaCuO5 (211)",
      "role": "secondary phase",
      "fraction": { "value": 10, "unit": "wt.%" }
    }
  ]
}
```

**Notes**:
- Do NOT list substrates/electrodes/caps/contacts as secondary phases. Only include them in `secondary_phases` if the paper explicitly labels them as a phase/component *within the sample* (e.g., explicitly calls it a "secondary phase", "second phase", "minor phase", "impurity phase", "parasitic phase", "additive", "composite component", "inclusion", or "precipitate", or uses an equivalent explicit role label).

### 4) `stack_descriptor`

**Definition**: Record any explicitly stated layered/stacked architecture for the primary superconducting system (film/substrate, heterostructure, interface, superlattice, multilayer).

**When to include the top-level key `stack_descriptor`**:
- Only if the paper explicitly describes a layered/stacked architecture.

**Output format**:
- `stack_descriptor` MUST be a JSON array of objects.
- Each item is ONE distinct sample/architecture description (one stack) for the primary system as explicitly stated.

**When to split into multiple items** (create multiple `{...}` inside the array):
- The paper explicitly studies multiple distinct architectures for the primary system, for example:
  - the same superconducting layer grown on different substrates, OR
  - different layer sequences / different superlattice periods, OR
  - clearly distinct heterostructure designs.
- If the paper describes only one architecture, output a one-item array: `[ { ... } ]`.

**Sub-keys**:
- **`architecture`** (string, REQUIRED in each `stack_descriptor` item): short description of the stack (e.g., "thin film on substrate", "A/B superlattice").
- **`layers`** (array, REQUIRED in each `stack_descriptor` item): list layers from top to bottom if explicit.
  - Each layer is an object with:
    - `material` (string, REQUIRED if a `layers` item is included)
    - `thickness` (number or string, optional)
    - `unit` (string, optional; e.g., `nm`, `uc`)
    - `role` (string, optional; e.g., "film", "substrate", "buffer", "cap")
- **`periodicity`** (string, optional): repetition info if explicit (e.g., "(m/n)×N").
- **`notes`** (string, optional): extra stack-specific details that cannot be placed into `architecture` / `layers` / `periodicity`. Keep it as a very short verbatim phrase (preferably <= 10 words) copied from the paper; omit `notes` if it would be generic, speculative, or not explicitly stated.

Example (format only; values shown are illustrative placeholders—do NOT copy them):
```json
{
  "stack_descriptor": [
    {
      "architecture": "YBa2Cu3O7/PrBa2Cu3O7 superlattice on SrTiO3 substrate",
      "layers": [
        { "material": "YBa2Cu3O7", "role": "superconducting layer", "thickness": 2, "unit": "uc" },
        { "material": "PrBa2Cu3O7", "role": "spacer layer", "thickness": 5, "unit": "uc" },
        { "material": "SrTiO3", "role": "substrate" }
      ],
      "periodicity": "(2 uc / 5 uc)×20",
      "notes": "c-axis oriented"
    }
  ]
}
```

## JSON Output Schema (overview)
Notes: The schema below is a comprehensive list of all top-level keys and sub-keys; it does NOT imply that every paper will contain extractable information for every key. Include a key only if explicitly supported by the paper text; if included, the key name MUST match the schema exactly (case- and space-sensitive).
```json
{
  "section0": {
    /* Include Section0 keys only if explicitly supported by the paper text. Omit missing keys/sub-keys; do NOT output null. */

    "electronic_state_tuning_mechanism": [
      {
        "tuning_type": "cation_substitution | anion_substitution | oxygen_nonstoichiometry | vacancy_defect | interstitial | intercalation | isovalent_substitution | interface_charge_transfer | electrostatic_gating | unknown",
        "dopant_defect": "string (optional)",
        "parameter": "string (optional)",
        "stoichiometry": "string (optional)",
        "carrier_effect": "electron | hole | none_isovalent | mixed_or_ambiguous | unknown (optional; only if explicitly stated)"
      }
      /* ... more mechanism records ... */
    ],

    "carrier_concentration": [
      {
        "level": "number or string",
        "unit": "string",
        "basis": "string (optional)",
        "conditions": "string (optional)"
      }
      /* ... more carrier concentration records ... */
    ],

    "secondary_phases": [
      {
        "phase": "string",
        "role": "string (optional)",
        "fraction": {
          "value": "number or string",
          "unit": "string"
        }
      }
      /* ... more phases ... */
    ],

    "stack_descriptor": [
      {
        "architecture": "string",
        "layers": [
          {
            "material": "string",
            "role": "string (optional)",
            "thickness": "number or string (optional)",
            "unit": "string (optional)"
          }
          /* ... more layers ... */
        ],
        "periodicity": "string (optional)",
        "notes": "string (optional)"
      }
      /* ... more stack records ... */
    ]
  }
}
```

## Input

Notes:
- The paper text to process is converted from PDF to Markdown via OCR/text conversion. Markdown formatting may reflect some document structure, but it can be noisy/unreliable; you may use it as a weak hint, but do NOT rely on it as a source of truth.
- Due to PDF-to-text conversion artifacts, the boundary between the main text and the references section may contain interleaved/crossed text; be cautious and avoid treating such artifacts as factual evidence.

**Key Explanations** for the single-system identification result JSON below (to help you identify the primary superconducting material system studied in this paper - do NOT use these as extraction evidence):
- `single_system`: 1 if paper focuses on one superconducting system, 0 if multiple
- `primary_signature`: The canonical identifier of the main superconducting system (e.g., "Bi2Sr2CuO6+δ")
- `primary_signature_type`: "formula_based" (chemical formula) or "stack_based" (interface/stack)
- `counted_signatures`: List of all confirmed superconducting systems in this paper
- `secondary_or_control_materials`: Non-superconducting materials mentioned (substrates, electrodes, etc.)
- `decision_reason`: Why this paper was classified as single/multi-system
- `evidence.primary_system_quotes`: Direct quotes confirming the primary system
- `evidence.other_system_quotes`: Quotes about other systems (if any)

### Single-system identification result (JSON)
{{single_system_result}}

### Paper text (Markdown) to process
{{paper_text}}

### Supplementary information (if provided)
{{supplementary_information}}