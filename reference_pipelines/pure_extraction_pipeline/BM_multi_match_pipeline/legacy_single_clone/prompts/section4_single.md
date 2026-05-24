# Task

You are an information extraction specialist in superconductivity research. Your task is to extract physical property curve information (Section 4) from a scientific article, utilizing figure classification results from an upstream language model as a reference for extraction guidance.

## Input Overview
 Two inputs will be provided. The full content of these inputs will be provided in the `# Input` section below.
1. Figure classification results: A JSON object from the upstream LLM containing the following keys:
   - `"section3"`: figures/subfigures classified as Section3 (microscopic characterization & electronic structure)
   - `"section4"`: figures/subfigures classified as Section4 (physical property curves)
   - `"others"`: figures/subfigures that are neither Section3 nor Section4
   - `"unsure"`: figures/subfigures that cannot be confidently classified into Section3/Section4/others based on text evidence
2. Article text (Markdown): The full article text of the scientific article (including title, abstract, main text, figure captions, and supplementary information (if provided)).

## Task Guideline:
- General principle: adhere to the upstream figure classification as a guideline; extract only information that satisfies the key definitions below.
- Primary focus: figures/subfigures classified as `"section4"`; also examine `"unsure"` classifications as a secondary scope.
- Other figures: examine `"section3"` and `"others"` classifications to prevent misclassification, but apply stricter evaluation criteria.
- Information sources: utilize figure captions and the relevant discussion in the main text (textual content only; no figure images).
- Scope: electrical transport (R–T/ρ–T, R–H/ρ–H, I–V), magnetic/susceptibility (M–T / DC susceptibility–T, χ–T), specific heat/thermal curves, and related phase-boundary curves (e.g., \(H_{c2}\), \(H_{irr}\)). If a superconductivity-relevant physical-property curve does not fit the predefined keys (e.g., M–H, \(J_c\), \(H_{c1}\), \(\Delta\lambda\), Seebeck), record it under `others`.

# Output Format

## Output Structure
Output pure JSON text only. No explanations, no markdown code blocks (no ```json or ```).
Output a single JSON object with one top-level key: `"section4"`.
- **Always include `"section4"`**: if no extractable Section4 information is found, output `{"section4": {}}`.


## JSON Validity
Preserve the original text content (including LaTeX, special symbols, etc.) when extracting.

Only apply JSON *string escaping* when necessary to keep the final output valid JSON (do not rewrite content otherwise). Escape the following characters inside JSON strings:
- `\` → `\\`
- `"` → `\"`
- newline (a literal line break) → `\n` (two characters: backslash + n)

## Key Data Type
- **Terminology (used throughout this prompt)**:
  - `top-level key`: the only top-level key in the output JSON is `"section4"`.
  - `main key`: the keys directly under `"section4"` (e.g., `R_T`, `R_H`, `M_T`, ...; including `others` when present).
  - `record object`: one `{...}` item that represents one extracted curve/measurement record (one entry) under a main key.
  - `sub-key` (leaf key): the keys inside one record object (e.g., `figure`, `magnetic_field`, `pressure`, ...).
- **Main key value type: object vs array**：
  - For each main key under `section4` (e.g., `R_T`, `R_H`, `M_T`, ...), output either one record object or an array of record objects following the pairing rules below. (`others` is always an array when present.)
  - One figure/subfigure → one record object.
  - Multiple figures with identical conditions → one record object with `"figure"` as a string array.
  - Different conditions → an array of record objects.
- **sub-key value type: string vs string array**：
  - Scope note: this rule does NOT apply to sub-keys under the main key `others`.
  - Default: use a JSON string.
  - Use a JSON string array ONLY when the caption/text explicitly enumerates two or more distinct values that all apply to this same record object/entry (do not infer or search for additional values beyond what is explicitly tied to this entry).
  - If different values belong to different entries (different curves/conditions), do NOT merge them into one record object; split them into multiple record objects under the same main key.
  - **Paired sub-keys: alignment within one record object**:
    - This pairing rule applies to these two pairs only:
      - (`magnetic_field`, `magnetic_direction`)
      - (`current`, `current_direction`)
    - If both values are single strings, they are treated as a 1-to-1 paired condition for this entry.
    - If one is a string and the other is a string array, treat it as 1-to-many (the single value applies to all items in the array).
    - If both are string arrays and you intend to express many-to-many pairing within this single entry, they MUST be index-aligned arrays of the same length: item `i` in the first array corresponds to item `i` in the second array.
      - If one direction corresponds to multiple `magnetic_field` values (or one `current_direction` corresponds to multiple `current` values), repeat that direction string in the corresponding positions so the arrays remain the same length.
      - If you cannot align the correspondence unambiguously from text/caption evidence, do NOT guess; instead, split into multiple record objects under the same main key.
  - Do not put `null` inside arrays.

## Output Content
- Include a main key/sub-key only if explicitly supported by the article text/captions (or explicitly stated raw_data availability).
- Do not invent values. Keep units/symbols/wording as written.
- Do not output `null` in the final JSON. Omit missing sub-keys entirely.
- Keys in the output JSON must match exactly the key names defined below (case-sensitive).
- Ambient pressure normalization:
  - If explicitly stated (e.g., ambient/atmospheric/1 bar/0 GPa/zero pressure), record as `"0 GPa"` (do not assume if unstated).

## Figure ID Recording Rules
- Prefer the figure ID style as written in the figure caption.
- If multiple surface forms refer to the same figure/subfigure, choose one canonical ID (caption-first; otherwise first in-text mention).
- Treat surface-form variants as equivalent (e.g., `"Figure 2"` ≈ `"Fig. 2"`; `"Fig. 3a"` ≈ `"Fig. 3(a)"`).
- Include subfigure identifier only if explicitly cited in text/caption.
- Extract figure IDs from captions/text only; do not invent or infer IDs.

# Schema Map (allowed keys)

This section lists the allowed main keys and sub-keys only. Definitions and constraints are in the  `# Key Definitions and Extraction Rules` section.

## `section4` top-level keys

`section4` is an object that may include any subset of the following keys:
`R_T`, `R_H`, `I_V`, `specific_heat`, `M_T`, `chi_T`, `PDO`, `thermal_conductivity`, `nernst_effect`, `others`

## Main keys and allowed sub-keys

**Common sub-keys** (available for most main keys):
- `figure`, `magnetic_field`, `magnetic_direction`, `magnetic_direction_reference`, `pressure`, `angles`, `angles_reference`, `current`, `current_direction`, `raw_data`

**Main-key-specific allowed sub-keys**:
- `R_T` (R–T / ρ–T): Common sub-keys only
- `R_H` (R–H / ρ–H): Common sub-keys + `temperature`
- `I_V` (I–V): Common sub-keys + `temperature`
- `specific_heat` (C–T / \(C_p\)/T–T etc.): Common sub-keys (except `current`, `current_direction`)
- `M_T` (M–T / DC susceptibility–T): Common sub-keys (except `current`, `current_direction`) + `field_cooling`, `magnetometer`
- `chi_T` (AC susceptibility–T): Common sub-keys (except `current`, `current_direction`) + `frequency`
- `PDO` (Δf vs T OR Δf vs \(\mu_0 H\)): Common sub-keys + `variable`, `temperature`
  - `variable`: `"temperature"` or `"magnetic_field"` (x-axis variable; include only if explicit in caption/text)
- `thermal_conductivity` (κ curves): Common sub-keys + `variable`, `temperature`
  - `variable`: `"temperature"` or `"magnetic_field"` (x-axis variable; include only if explicit in caption/text)
- `nernst_effect` (Nernst curves vs T or H): Common sub-keys + `signal`, `variable`, `temperature`
  - `signal`: y-axis Nernst quantity name as written (e.g., `e_N`, `ν`; include only if explicitly stated)
  - `variable`: `"temperature"` or `"magnetic_field"` (x-axis variable; include only if explicit in caption/text)
- `others` (array of objects for curves not covered above):
  - Each item: `name`, `figure`, `detail` (object with `independent_variable`, `dependent_variable`, `conditions`, `result`), optionally `raw_data`

# Key Definitions and Extraction Rules

## Common sub-keys
This section defines sub-keys that are shared across multiple main keys. Any sub-keys included in that record object must refer to that same entry; do not mix information from different entries/curves/conditions.

- `figure`: figure/subfigure ID(s) for the curve (follow `## Figure ID Recording Rules` and `## Key Data Type` for canonicalization and pairing).
- `magnetic_field`: magnetic_field value(s) used for this curve/entry (this record object). If `magnetic_direction` is also present and one/both are arrays, follow the paired sub-key alignment rules in `## Key Data Type`.
- `magnetic_direction`: the most direct and concise description of the applied magnetic-field orientation for this specific curve/entry (i.e., for the corresponding `magnetic_field` value(s) in this same record object), recorded exactly as written in the paper. Prioritize the clearest/shortest unambiguous expression (e.g., `H // c`, `perpendicular to the film plane`, `along c axis`). If multiple equivalent phrasings describe the same direction for this entry, choose the most precise one; if multiple distinct directions are explicitly tied to this same entry, record them as a string array.
  - Data type: one direction → JSON string; two or more distinct directions explicitly tied to this same entry → JSON string array (and if `magnetic_field` is also an array, follow the paired sub-key alignment rules in `## Key Data Type`).
- `magnetic_direction_reference` (optional): additional contextual information about the magnetic_field direction when clarification is needed. Include when the paper provides multiple descriptions of the direction, or when additional context about sample geometry or coordinate system is necessary to fully understand the primary description in `magnetic_direction`.
  - Examples of when to include (only when clarification is needed beyond `magnetic_direction`):
    - `The c axis is perpendicular to the ab plane.` (provides crystallographic coordinate system definition)
    - `For thin films, the c axis corresponds to the film normal.` (clarifies axis orientation in thin film geometry)
    - `H was applied parallel to the CuO2 planes.` (provides structural reference for superconducting material)
- `pressure`: pressure value(s) used for this curve/entry (this record object). If the paper explicitly states ambient pressure (e.g., ambient/atmospheric/1 bar/0 GPa/zero pressure), record it as `"0 GPa"`.
- `angles`: the angle setting(s) used for this specific curve, recorded exactly as written (e.g., `θ=30°`, `0–90°`), only if explicitly stated.
- `angles_reference` (optional): a short verbatim snippet (from the text/caption) that explains the angular coordinate system used in `angles`. Include when the paper defines what the angle variables (θ, φ, etc.) measure relative to, or specifies the zero-point/rotation geometry. Omit if no such explanatory context is provided.
  - Examples (snippets; copy verbatim if present):
    - `θ is the angle between H and the c axis.`
    - `φ is the in-plane azimuthal angle measured from the a axis.`
    - `θ=0° corresponds to H ∥ c.`
    - `Rotation angle measured from the sample normal.`
- Guardrail (for direction/angle keys): do not over-infer direction/geometry mappings; keep relations exactly as written (e.g., keep `H ⟂ c` as-is) unless the paper explicitly defines the mapping (record such definitions in `magnetic_direction_reference` / `angles_reference`).
- `raw_data`: indicates whether the paper explicitly states that the underlying/source data for this specific curve/figure is available (figure/curve-level only). Set to `"yes"` only if explicitly stated for the relevant figure/curve; otherwise omit. Do not infer this from a paper-level Data Availability statement (handle paper-level availability elsewhere).
  - Examples (figure/curve-level wording; copy verbatim if present):
    - `Source Data are provided for Fig. 3.`
    - `Source data for this figure are available.`
    - `raw_data underlying Fig. 3 are provided.`

## Main-key-specific sub-keys
This section defines additional sub-keys and rules that apply only to certain main keys (see Schema Map for where each applies).

- `variable` (optional): for keys such as `PDO`, `thermal_conductivity`, `nernst_effect`, indicates the x-axis being swept (`"temperature"` or `"magnetic_field"`), only if explicit in the caption/text. If present, treat it as the swept axis:
  - If `variable` is `"temperature"`: do not record swept temperature values as a `temperature` condition; record `magnetic_field` only when explicitly stated as a fixed condition.
  - If `variable` is `"magnetic_field"`: do not record swept magnetic-field values as a `magnetic_field` condition; record `temperature` only when explicitly stated as a fixed condition.
- `temperature`: temperature value(s) used for this curve/entry (this record object), only for main keys that list `temperature` as an allowed sub-key in the Schema Map.
- `signal` (optional): for `nernst_effect`, the y-axis Nernst quantity name as written (e.g., `e_N`, `ν`, `Nernst signal`), only if explicitly stated.
- `current`: transport current value(s), only if explicitly stated (keep units/wording as written). If `current_direction` is also present and one/both are arrays, follow the paired sub-key alignment rules in `## Key Data Type`.
- `current_direction`: transport current_direction/orientation, only if explicitly stated (keep wording as written).
  - Data type: one direction → JSON string; two or more distinct directions explicitly tied to this same entry → JSON string array (and if `current` is also an array, follow the paired sub-key alignment rules in `## Key Data Type`).
- `field_cooling`: for `M_T`, whether ZFC/FC protocols were used (record as a string array, e.g., `["ZFC", "FC"]`), only if explicitly stated.
- `magnetometer`: for `M_T`, the instrument name/type(s) used to obtain this curve/entry (e.g., `SQUID`, `VSM`), only if explicitly stated. One instrument → string; two or more distinct instruments → string array.
- `frequency`: AC excitation frequency value(s) for `chi_T`, only if explicitly stated (keep units/wording as written). One value → string; two or more distinct values → string array.
- `name`: for `others`, a short standardized curve name you define (e.g., `Hc2_T`, `Jc_H`).
- `detail`: for `others`, an object with `independent_variable`, `dependent_variable`, `conditions`, `result`.
- `independent_variable`: for `others.detail`, x-axis quantity and unit as written.
- `dependent_variable`: for `others.detail`, y-axis quantity and unit as written.
- `conditions`: for `others.detail`, a concise condition string explicitly stated for that curve (direction/pressure/criterion/fit, etc.).
- `result`: for `others.detail`, a concise main observation/conclusion for that curve.

## Main keys

Applies to every main key below:
- Allowed sub-keys are specified in `# Schema Map (allowed keys)` (Main keys and allowed sub-keys). Do not output any sub-key not listed there.
- Common sub-key meanings (`figure`, conditions, and `raw_data`) are defined in `## Common sub-keys` / `## Main-key-specific sub-keys`.
- Examples are illustrative; they may omit optional sub-keys. Include only what is explicitly stated for the specific curve/figure (omit missing sub-keys; do not output `null`).

### 1. `R_T`
Description: Resistance-temperature (R-T) or resistivity-temperature (ρ-T) measurements and their basic experimental conditions.

Example:
  
  `{"figure": "Fig. 3(a)", "magnetic_field": ["0 T", "1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°", "60°"], "angles_reference": "θ is the angle between H and the c axis", "current": ["1 mA", "10 mA"], "current_direction": "I // ab"}`

### 2. `R_H`
Description: Resistance-magnetic_field (R-H) or resistivity-magnetic_field (ρ-H) measurements and their basic experimental conditions. Sometimes the magnetic_field is written as \(\mu_0 H\).
  
Example:
  
  `{"figure": "Fig. 3(a)", "temperature": ["0.5 K", "1 K", "2 K"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°", "60°"], "angles_reference": "θ is the angle between H and the c axis", "current": ["1 mA", "10 mA"], "current_direction": "I // ab"}`

### 3. `I_V`
Description: Macroscopic current-voltage (I-V) measurements and their basic experimental conditions. This is not measured by scanning tunneling microscopy (STM).
Example:
  
  `{"figure": "Fig. 3(a)", "temperature": ["0.5 K", "1 K", "2 K"], "magnetic_field": ["0 T", "1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°", "60°"], "angles_reference": "θ is the angle between H and the c axis", "current_direction": "I // ab"}`

### 4. `specific_heat`

Description: Specific heat or heat capacity measurements presented as C_p/T vs T (or related forms), including experimental conditions.
Example: `{"figure": "Fig. 3(a)", "magnetic_field": ["0 T", "1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "pressure": "0 GPa", "angles": ["0°", "30°", "60°"], "angles_reference": "θ is the angle between H and the c axis"}`

### 5. `M_T`
Description: Magnetic moment-temperature (M-T), including experimental conditions.
Example: `{"figure": "Fig. 3(a)", "magnetic_field": "10 Oe", "field_cooling": ["ZFC", "FC"], "magnetometer": "SQUID"}`

### 6. `chi_T`
Description: AC magnetic susceptibility-temperature (χ-T), including experimental conditions.
Example: `{"figure": "Fig. 3(a)", "magnetic_field": "0 T", "frequency": "13.33 Hz"}`

### 7. `PDO`
Description: Proximity detection oscillator (PDO), e.g., Δf vs T or Δf vs \(\mu_0 H\).
Example: `{"figure": "Fig. 3(a)", "variable": "temperature", "magnetic_field": ["1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°"], "angles_reference": "θ is the angle between H and the c axis"}` (Δf vs T) or `{"figure": "Fig. 3(a)", "variable": "magnetic_field", "temperature": "1 K", "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "60°"], "angles_reference": "θ is the angle between H and the c axis"}` (Δf vs \(\mu_0 H\))

### 8. `thermal_conductivity`
Description: Thermal conductivity (κ), including experimental conditions.
Example: `{"figure": "Fig. 3(a)", "variable": "temperature", "magnetic_field": ["1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°"], "angles_reference": "θ is the angle between H and the c axis"}`

### 9. `nernst_effect`
Description: Nernst signal vs temperature or magnetic_field, including experimental conditions.
Example: `{"figure": "Fig. 3(a)", "signal": "e_N", "variable": "temperature", "magnetic_field": ["1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°"], "angles_reference": "θ is the angle between H and the c axis"}` (signal vs T) or `{"figure": "Fig. 3(a)", "signal": "e_N", "variable": "magnetic_field", "temperature": "1 K", "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "60°"], "angles_reference": "θ is the angle between H and the c axis"}` (signal vs \(\mu_0 H\))

### 10. `others`
Description: Physical-property curves not covered by the predefined main keys under `section4` (i.e., not `R_T`, `R_H`, `I_V`, `specific_heat`, `M_T`, `chi_T`, `PDO`, `thermal_conductivity`, or `nernst_effect`).
Example:
  `[{"name": "Hc2_T", "figure": "Fig. 5(a)", "detail": {"independent_variable": "temperature (K)", "dependent_variable": "µ0Hc2 (T)", "conditions": "H // c; pressure = 0 GPa; criterion = 90% p_n; fit = WHH", "result": "Hc2(T) is nearly linear near Tc; WHH fit is consistent with data."}}]`
Notes:
- Output an array of objects (one object per distinct curve type/figure).
- `name`: define a short, standardized name (e.g., `Hc2_T`, `Hc1_T`, `Jc_H`).
- `conditions`: include only conditions explicitly given for that curve.
- Do not duplicate content already captured under other keys.
- If no such curves exist, omit `others`.

# Extraction Procedure and Guardrails
Recommended extraction procedure (use the upstream figure classification as the primary guide):
1. Process figures/subfigures in this priority order:
   - Primary: `"section4"`
   - Secondary: `"unsure"`
   - Cross-check (strict): `"section3"` and `"others"` (record under `section4` only if the evidence clearly matches the key definitions and Schema Map).
2. For each selected figure/subfigure, choose the main key (`R_T`, `R_H`, `I_V`, `specific_heat`, `M_T`, `chi_T`, `PDO`, `thermal_conductivity`, `nernst_effect`; otherwise `others`) using the definitions above.
3. For that main key, extract only the conditions explicitly tied to that figure/subfigure (do not mix conditions across different measurements).
4. If axes are normalized/derived (e.g., R/Rn, Δf/f0), still treat it as a valid curve and record it under the correct key (or `others`).

# Example Output
Note: this example is illustrative only, showing the expected JSON structure and format. In actual extraction, include ONLY what is explicitly stated in the paper text/captions. Some keys like `"raw_data": "yes"` are shown for completeness but should only be included when explicitly stated in the paper.

(The example below may be shown in a code block for readability, but your actual response MUST be pure JSON text with no Markdown fences.)

```json
{
  "section4": {
    "R_T": {"figure": ["Fig. 3(a)", "Fig. 3(b)"], "magnetic_field": ["0 T", "1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°", "60°"], "angles_reference": "θ is the angle between H and the c axis", "current": ["1 mA", "10 mA"], "current_direction": "I // ab", "raw_data": "yes"},
    "R_H": {"figure": "Fig. 3(a)", "temperature": ["0.5 K", "1 K", "2 K"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°", "60°"], "angles_reference": "θ is the angle between H and the c axis", "current": ["1 mA", "10 mA"], "current_direction": "I // ab"},
    "I_V": {"figure": "Fig. 3(a)", "temperature": ["0.5 K", "1 K", "2 K"], "magnetic_field": ["0 T", "1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°", "60°"], "angles_reference": "θ is the angle between H and the c axis", "current_direction": "I // ab"},
    "specific_heat": {"figure": "Fig. 3(a)", "magnetic_field": ["0 T", "1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "pressure": "0 GPa", "angles": ["0°", "30°", "60°"], "angles_reference": "θ is the angle between H and the c axis"},
    "M_T": {"figure": "Fig. 3(a)", "magnetic_field": "10 Oe", "field_cooling": ["ZFC", "FC"], "magnetometer": "SQUID"},
    "chi_T": {"figure": "Fig. 3(a)", "magnetic_field": "0 T", "frequency": "13.33 Hz"},
    "PDO": {"figure": "Fig. 3(a)", "variable": "temperature", "magnetic_field": ["1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°"], "angles_reference": "θ is the angle between H and the c axis"},
    "thermal_conductivity": {"figure": "Fig. 3(a)", "variable": "temperature", "magnetic_field": ["1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°"], "angles_reference": "θ is the angle between H and the c axis"},
    "nernst_effect": {"figure": "Fig. 3(a)", "signal": "e_N", "variable": "temperature", "magnetic_field": ["1 T", "2 T"], "magnetic_direction": "H // c", "magnetic_direction_reference": "H was applied along the crystallographic c axis (out-of-plane)", "angles": ["0°", "30°"], "angles_reference": "θ is the angle between H and the c axis"},
    "others": [{"name": "Hc2_T", "figure": "Fig. 5(a)", "detail": {"independent_variable": "temperature (K)", "dependent_variable": "µ0Hc2 (T)", "conditions": "H // c; criterion = 90% p_n; fit = WHH", "result": "Hc2(T) is nearly linear near Tc; WHH fit is consistent with data."}, "raw_data": "yes"}]
  }
}
```

# Input

Notes:
- The paper text to process is converted from PDF to Markdown via OCR/text conversion. Markdown formatting may reflect some document structure, but it can be noisy/unreliable; you may use it as a weak hint, but do NOT rely on it as a source of truth.
- Due to PDF-to-text conversion artifacts, the boundary between the main text and the references section may contain interleaved/crossed text; be cautious and avoid treating such artifacts as factual evidence.

### Figure classification result (JSON) from upstream
{{figure_classification}}

### Paper text (Markdown) to process
{{paper_text}}

### Supplementary information (if provided)
{{supplementary_information}}