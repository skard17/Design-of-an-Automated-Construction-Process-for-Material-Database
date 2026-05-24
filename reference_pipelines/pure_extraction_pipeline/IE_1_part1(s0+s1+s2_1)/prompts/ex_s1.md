# Prompt for Section 1: Critical Values and Ginzburg-Landau Parameters Extraction

## Role
Act as an information extractor specializing in superconductivity.

## Task
Read the provided experimental superconducting-materials paper (including title, abstract, main text, figure captions, supplementary information if provided) and extract **critical values and Ginzburg-Landau parameters** according to the requirements below.

Inputs (provided verbatim in the `## Input` section below):
- Single-system identification result (JSON)
- Paper text (Markdown)
- Supplementary information (if provided)

Single-system identification result (JSON) rule: use it only for primary-system disambiguation; do NOT treat it as evidence for extracting any Section1 key/value.

# Output Format and Constraints

## Output Structure
Output pure JSON text only. No explanations, no markdown code blocks (no ```json or ```).
Output exactly one JSON object with the top-level key `"section1"`.
- `"section1"` must be a JSON object describing the ONE primary superconducting material/system studied in this paper.
- The primary system should align with the provided single-system identification result (e.g., its `primary_signature`).
- **Always include `"section1"`**: if no Section1 information is extractable, output `{"section1": {}}`.

## JSON Validity
Preserve the original text content (including LaTeX, special symbols, etc.) when extracting. Only escape characters that would break JSON validity:
- `\` → `\\`
- `"` → `\"`
- newline → `\n`

## Output Content
All keys inside "`section1`" are **CONDITIONAL**:
- Output a key (or sub-key for nested objects) **only if** explicitly supported by the paper text (title/abstract/main text/figure captions/supplementary information). Do NOT infer, invent, summarize, or interpret.
- If a key is not supported, OMIT it entirely—no `null` or empty placeholders.
- Do NOT add keys beyond the schema. Key names must match exactly as defined in `JSON Keys and Extraction Rules` (case-sensitive).
- Interpretation of "REQUIRED": means "required only IF the parent is being output"; not a reason to infer unsupported parents.

## Parameter object structure
For each parameter, use an **array of objects**. Each object contains:
- **`value`**: The parameter value (string) - **REQUIRED**
- **Structured condition keys** (only include if explicitly stated in article):
  - `temperature`: Measurement temperature
  - `magnetic_field`: Applied magnetic field
  - `pressure`: Applied pressure
  - `direction`: Measurement direction/orientation notation (copy as written)
  - `figure`: Figure reference
- **`characteristics`**: A string for additional distinguishing information not covered by the structured keys (optional)

**Notes**:
- For `Tc`: do NOT add a separate `temperature` condition key. `Tc.value` is itself the transition temperature.
- For `Hc1`, `Hc2`, `Hc`: do NOT add `magnetic_field` as a condition key.
- For `P_sc`, `P_nsc`: do NOT add `pressure` as a condition key. Pressure can only appear in `value`.
- For `lambda` and `xi` anisotropy: you may record anisotropy direction (e.g., `ab-plane` vs `c-axis`) inside `characteristics` as shown in the examples below.

## Figure ID recording rules (for the `figure` key)
When you include a `figure` value, record the figure ID in the paper's original style, using this priority order:
1) If the relevant figure caption is available in the provided text, use the figure ID surface form exactly as it appears in the caption (e.g., "Figure 2", "Fig. 2", "Fig. S1").
2) Otherwise (no caption available), use the figure ID surface form exactly as it appears at the supporting evidence location in the main text (the same sentence/paragraph near the info point).
3) Keep subfigure/panel designations only if explicitly stated (e.g., "Fig. 3(a)", "Fig. 3a"); do NOT invent panels.

## Parameter Keys

| Key | Parameter | Definition |
|-----|-----------|------------|
| `Tc` | Critical temperature | Superconducting transition temperature |
| `Jc` | Critical current density | Critical current density of the material |
| `Hc1` | Lower critical field | Threshold field for vortex penetration (Type-II superconductors) |
| `Hc2` | Upper critical field | Field at which superconductivity is completely suppressed |
| `Hc` | Thermodynamic critical field | For Type-I superconductors only |
| `P_sc` | Superconducting pressure | Pressure(s) at which material exhibits superconductivity |
| `P_nsc` | Non-superconducting pressure | Pressure(s) at which material loses superconductivity |
| `lambda` | Penetration depth | London penetration depth |
| `xi` | Coherence length | Superconducting coherence length |

```json
"Tc": [
  {
    "value": "38 K",
    "magnetic_field": "0 T",
    "pressure": "0 GPa",
    "figure": "Fig. 1",
    "characteristics": "doping: x=0.1, measurement: resistivity, criterion: onset"
  },
  {
    "value": "42 K",
    "magnetic_field": "0 T",
    "pressure": "0 GPa",
    "figure": "Fig. 1",
    "characteristics": "doping: x=0.15, measurement: resistivity, criterion: onset"
  }
]
```

### Key Definitions

| Key | Type | Description | When to include |
|-------|------|-------------|-----------------|
| `value` | string | The measured/calculated parameter value | **Always required** |
| `temperature` | string | Measurement temperature | Only if explicitly stated |
| `magnetic_field` | string | Applied magnetic field | Only if explicitly stated |
| `pressure` | string | Applied pressure | Only if explicitly stated |
| `direction` | string | Measurement direction/orientation notation | Only if explicitly stated |
| `figure` | string | Figure reference | Only if the text explicitly links this value to a specific figure |
| `characteristics` | string | Additional distinguishing info | Only if needed |

### What goes in `characteristics`

The `characteristics` key is for information that **cannot** be captured by the structured keys above:

| Category | Examples |
|----------|----------|
| Sample / Material Properties | `sample: film d180`, `sample: single crystal`, `sample: epitaxial NbN` |
| Composition / Tuning Parameters | `doping: x=0.3`, `oxygen deficiency: δ=0.1`, `composition: optimally doped` |
| Measurement Method | `measurement: specific heat`, `measurement: μSR`, `measurement: resistivity` |
| Data Criterion / Analysis Method | `criterion: onset`, `criterion: zero resistance`, `extrapolation: WHH` |

**Important**: 
- Do NOT include temperature, magnetic_field, pressure, direction, or figure in `characteristics` – use the structured keys instead.
- Exception: For `lambda` and `xi` anisotropy only, you MAY include `direction: ...` inside `characteristics` (because anisotropy direction is treated here as a material/property qualifier, not a measurement condition).

---

## JSON Keys and Extraction Rules

### 1. `Tc` (Critical Temperature)
- **Definition**: The superconducting transition temperature ($T_c$) of the material.
- **Example**: 
```json
"Tc": [
  {
    "value": "4.2 K",
    "magnetic_field": "1 T",
    "pressure": "0 GPa",
    "figure": "Fig. 1",
    "characteristics": "measurement: resistivity"
  },
  {
    "value": "4.5 K",
    "magnetic_field": "2 T",
    "pressure": "0 GPa",
    "figure": "Fig. 1",
    "characteristics": "measurement: resistivity"
  }
]
```
- **Notes**:
  - Record all $T_c$ values with their conditions.
  - **Exclude**: Curie temperature, Néel temperature, $T_c$ of reference materials or other materials not being studied.
  - Keep the original unit as used in the article.

### 2. `Jc` (Critical Current Density)
- **Definition**: The critical current density $J_c$ of the superconducting material.
- **Example**: 
```json
"Jc": [
  {
    "value": "2.6×10^2 A/cm^2",
    "magnetic_field": "1 T",
    "temperature": "10 K",
    "figure": "Fig. 3"
  },
  {
    "value": "3.2 MA/cm^2",
    "magnetic_field": "2 T",
    "temperature": "20 K",
    "figure": "Fig. 3"
  }
]
```
- **Notes**:
  - Record all $J_c$ values with magnetic field, temperature, and figure conditions.
  - Preserve the original unit as used in the article.

### 3. `Hc1` (Lower Critical Field)
- **Definition**: The lower critical magnetic field at which vortices first penetrate a Type-II superconductor, marking the transition from the Meissner state.
- **Example**: 
```json
"Hc1": [
  {
    "value": "0.02 T",
    "temperature": "2 K",
    "direction": "H // c",
    "figure": "Fig. 3"
  },
  {
    "value": "0.05 T",
    "temperature": "5 K",
    "direction": "H // c",
    "figure": "Fig. 3"
  }
]
```
- **Notes**:
  - **CRITICAL**: Hc1 and Hc2 must be separate keys. Never combine them.
  - Use the notation from the article: $H$, $\mu_0H$, or $B$ as written.
  - **Direction**: Copy exactly from article (e.g., `"H // c"`, `"H ⊥ c"`, `"c-axis"`, `"in-plane"`, `"normal to the film"`). Do NOT normalize or standardize the notation.

### 4. `Hc2` (Upper Critical Field)
- **Definition**: The upper critical magnetic field at which superconductivity is completely suppressed and the material returns to the normal state.
- **Example**: 
```json
"Hc2": [
  {
    "value": "50 T",
    "temperature": "0 K",
    "direction": "H // c",
    "figure": "Fig. 4",
    "characteristics": "extrapolation: WHH"
  },
  {
    "value": "~150 T",
    "temperature": "0 K",
    "direction": "H // ab",
    "figure": "Fig. 4",
    "characteristics": "extrapolation: WHH"
  }
]
```
- **Notes**:
  - **CRITICAL**: Hc2 and Hc1 must be separate keys. Never combine them.
  - Use the notation from the article: $H$, $\mu_0H$, or $B$ as written.
  - Record extrapolation method in `characteristics` if mentioned.
  - **Direction**: Copy exactly from article. Do NOT normalize or standardize the notation.

### 5. `Hc` (Thermodynamic Critical Field)
- **Definition**: The thermodynamic critical field for Type-I superconductors.
- **Example**: 
```json
"Hc": [
  {
    "value": "0.02 T",
    "temperature": "2 K",
    "figure": "Fig. 3"
  }
]
```
- **Notes**:
  - Use only for Type-I superconductors.

### 6. `P_sc` (Superconducting Pressure)
- **Definition**: Pressure conditions at which the material exhibits superconductivity.
- **Example**: 
```json
"P_sc": [
  {"value": "0 GPa", "temperature": "10 K", "figure": "Fig. 7"},
  {"value": "5 GPa", "temperature": "10 K", "figure": "Fig. 7"},
  {"value": "10 GPa", "temperature": "10 K", "figure": "Fig. 7"}
]
```
- **Notes**:
  - Record each superconducting pressure value as a separate object.
  - **Only record if explicitly mentioned** in the article. Do NOT assume or infer.
  - For ambient pressure explicitly stated in the article (e.g., "at ambient pressure", "at zero pressure"), record as `"0 GPa"`.

### 7. `P_nsc` (Non-Superconducting Pressure)
- **Definition**: Pressure conditions at which the material loses superconductivity.
- **Example**: 
```json
"P_nsc": [
  {"value": ">15 GPa", "temperature": "10 K", "figure": "Fig. 7"}
]
```
- **Notes**:
  - Record each non-superconducting pressure value.
  - **Only record if explicitly mentioned** in the article. Do NOT assume or infer.

### 8. `lambda` (Penetration Depth)
- **Definition**: London penetration depth $\lambda$, the characteristic distance over which an external magnetic field decays inside a superconductor.
- **Example** (isotropic): 
```json
"lambda": [
  {
    "value": "150 nm",
    "temperature": "10 K",
    "magnetic_field": "0 T",
    "characteristics": "measurement: μSR"
  }
]
```
- **Example** (anisotropic): 
```json
"lambda": [
  {
    "value": "150 nm",
    "temperature": "10 K",
    "characteristics": "direction: ab-plane, measurement: μSR"
  },
  {
    "value": "1500 nm",
    "temperature": "10 K",
    "characteristics": "direction: c-axis, measurement: μSR"
  }
]
```
- **Notes**:
  - For anisotropic superconductors, use `characteristics` to note the direction (since it's a material property, not a measurement condition).
  - For effective values in ultra-thin films, note as `effective lambda` in characteristics.
  - Keep the original unit as used in the article.

### 9. `xi` (Coherence Length)
- **Definition**: The superconducting coherence length $\xi$, which characterizes the size of a Cooper pair in the material.
- **Example** (isotropic): 
```json
"xi": [
  {
    "value": "3 nm",
    "characteristics": "calculated from Hc2"
  }
]
```
- **Example** (anisotropic): 
```json
"xi": [
  {
    "value": "3 nm",
    "characteristics": "direction: ab-plane, calculated from Hc2"
  },
  {
    "value": "0.5 nm",
    "characteristics": "direction: c-axis, calculated from Hc2"
  }
]
```
- **Notes**:
  - For anisotropic superconductors, use `characteristics` to note the direction (since it's a material property, not a measurement condition).
  - Preserve ranges as written in the article (e.g., `"3-10 nm"`).
  - Keep the original unit as used in the article.

---

## General Extraction Guidelines

1. **Faithfulness**: Extract only what is explicitly stated or shown in figures/tables. Do not infer or calculate values not directly provided.

2. **Units**: Preserve original units exactly as in the source text. Do not convert units.

3. **Approximate values**: Keep original notation exactly as written:
   - `~4.2 K` → `"~4.2 K"` (NOT `"4.2 K"`)
   - `≈150 nm` → `"≈150 nm"` (NOT `"150 nm"`)
   - `≃4 K` → `"≃4 K"`
   - `around 50 T` → `"around 50 T"` (NOT `"50 T"`)
   - `>100 GPa` → `">100 GPa"`
   - `<2 K` → `"<2 K"`
   - `∼24 K` → `"∼24 K"`

4. **Direction notation**: Copy exactly from article. Do NOT normalize or standardize:
   - `"H // c"` stays as `"H // c"` (NOT `"c"`)
   - `"in-plane"` stays as `"in-plane"` (NOT `"ab"`)
   - `"c-axis"` stays as `"c-axis"` (NOT `"c"`)
   - `"H ⊥ c"` stays as `"H ⊥ c"`
   - `"ab-plane"` stays as `"ab-plane"`
   - `"normal to the film"` stays as `"normal to the film"`

5. **Condition sub-key rules**: 
   - **Only include keys that are explicitly stated** in the article.
   - Do NOT assume `"0 T"` or `"0 GPa"` unless the article explicitly states "zero field", "ambient pressure", "at zero magnetic field", etc.
   - If a condition is not mentioned, **omit that key entirely** from the object.

6. **Figure reference**: Include the `"figure"` key only when the paper text explicitly links this extracted value to a specific figure. Follow the **Figure ID recording rules** above.

7. **Omission rule**: If a parameter is not mentioned in the article, omit the key entirely from the output.

8. **Single value simplification**: If a parameter has only one value and no additional characteristics or conditions needed, you can still omit the `characteristics` key.

---

## Critical Reminders

### NEVER Combine Hc1 and Hc2
```json
// WRONG - Never do this
"critical_magnetic_field": [
  {"Hc1": "0.02 T", "Hc2": "1.5 T"}
]

// CORRECT - Always separate
"Hc1": [{"value": "0.02 T", "temperature": "2 K"}],
"Hc2": [{"value": "1.5 T", "temperature": "2 K"}]
```

### Do NOT Add Assumed Values
```json
// WRONG - Don't assume conditions
{"value": "4.2 K", "magnetic_field": "0 T", "pressure": "0 GPa"}

// CORRECT - Only include if explicitly stated
{"value": "4.2 K"}
// OR if the article says "at ambient pressure"
{"value": "4.2 K", "pressure": "0 GPa"}
```

### Do NOT Duplicate Information
```json
// WRONG - temperature appears in both places
{
  "value": "50 T",
  "temperature": "0 K",
  "characteristics": "temperature: 0 K, extrapolation: WHH"
}

// CORRECT - temperature only in structured key
{
  "value": "50 T",
  "temperature": "0 K",
  "characteristics": "extrapolation: WHH"
}
```

---

## Common Mistakes to Avoid

| Mistake | Correction |
|---------|------------|
| Combining Hc1 and Hc2 in one key | Use separate keys: `Hc1` and `Hc2` as independent entries |
| Adding `"pressure": "0 GPa"` when pressure is not mentioned | Only include if the article explicitly states "ambient pressure" or "zero pressure" |
| Adding `"magnetic_field": "0 T"` when magnetic field is not mentioned | Only include if the article explicitly states "zero field" or "in zero magnetic field" |
| Normalizing `"c-axis"` to `"c"` | Keep original notation: `"direction": "c-axis"` |
| Normalizing `"in-plane"` to `"ab"` | Keep original notation: `"direction": "in-plane"` |
| Normalizing `"H // c"` to `"c"` | Keep original notation: `"direction": "H // c"` |
| Converting `~4.2 K` to `4.2 K` | Keep approximate symbol: `"value": "~4.2 K"` |
| Converting `≈150 nm` to `150 nm` | Keep approximate symbol: `"value": "≈150 nm"` |
| Converting `around 50 T` to `50 T` | Keep original wording: `"value": "around 50 T"` |
| Recording Curie temperature or Néel temperature as Tc | Only record superconducting transition temperature as Tc |
| Including temperature/magnetic_field/pressure/direction in characteristics | Use the structured keys instead |
| Adding `null` values for missing keys | Omit the key entirely |

---

## Example Output

```json
{
  "section1": {
      "Tc": [
        {
          "value": "38 K",
          "figure": "Fig. 1",
          "characteristics": "doping: x=0.1, measurement: resistivity, criterion: onset"
        },
        {
          "value": "42 K",
          "figure": "Fig. 1",
          "characteristics": "doping: x=0.15, measurement: resistivity, criterion: onset"
        }
      ],
      "Hc1": [
        {
          "value": "0.02 T",
          "temperature": "2 K",
          "direction": "H // c",
          "figure": "Fig. 3"
        }
      ],
      "Hc2": [
        {
          "value": "50 T",
          "temperature": "0 K",
          "direction": "H // c",
          "figure": "Fig. 4",
          "characteristics": "extrapolation: WHH"
        },
        {
          "value": "~150 T",
          "temperature": "0 K",
          "direction": "H // ab",
          "figure": "Fig. 4",
          "characteristics": "extrapolation: WHH"
        }
      ],
      "P_sc": [
        {"value": "0 GPa", "figure": "Fig. 6"},
        {"value": "5 GPa", "figure": "Fig. 6"},
        {"value": "10 GPa", "figure": "Fig. 6"}
      ],
      "lambda": [
        {
          "value": "≈150 nm",
          "temperature": "10 K",
          "characteristics": "measurement: μSR"
        }
      ],
      "xi": [
        {
          "value": "3 nm",
          "characteristics": "direction: ab-plane, calculated from Hc2"
        },
        {
          "value": "0.5 nm",
          "characteristics": "direction: c-axis, calculated from Hc2"
        }
      ]
  }
}
```

---

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