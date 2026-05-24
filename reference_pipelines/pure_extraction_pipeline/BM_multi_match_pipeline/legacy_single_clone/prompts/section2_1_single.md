# Role
Act as an information extractor specializing in superconductivity.

# Task
Identify fabrication-related information according to the instructions below.

Inputs (provided verbatim in the `## Input` section below):
- Single-system identification result (JSON)
- Paper text (Markdown)
- Supplementary information (if provided)

Single-system identification result (JSON) rule: use it ONLY to lock the primary system identity (e.g., `primary_signature` / `counted_signatures` as system-identifier keys). Do NOT use any specific numeric values or evidence quotes inside that JSON for extracting any fabrication key/value.
In this extraction, the primary system you focus on must be the one specified by the provided single-system identification result (e.g.,`primary_signature`).

# Instructions
1. **Adhere to the Source Text**: Use the original terminology and phrasing from the paper text and supplementary information (if provided). Do not infer or add information not explicitly stated.
2. **In-paper anchoring (Exclude background/literature)**: Record fabrication entries only when the paper clearly anchors them to THIS paper's experimental program (e.g., "we grew/prepared/fabricated", in-paper figure/table reference, explicit sample/device label, or "our sample/device" wording). Do NOT record fabrication information that is mentioned only as background/literature comparison (e.g., "reported previously", "known", "as reported in Ref.").
3. **Method-entry list structure (Most Important)**: The value of `"section2_1"` is a JSON array. Each array element is one fabrication method entry (a JSON object). Each method entry MUST include:
   - `entry_id`: the 1-based index of this entry in the output array (1, 2, 3, ...)
   - `method`: one of the allowed method categories
   - `description`: copied verbatim from the paper text (preferably one full sentence; no paraphrase)
   Multiple entries may have the same `method` value.
4. **When to create a new entry**: Create a new method entry when (a) the `method` differs, or (b) within the same `method`, the set of explicit quantitative parameters (represented by `conditions`) differs. If `conditions` is omitted for an entry, treat it as an empty set for this comparison.
5. **Geometry within each method entry**: For each method entry, record the geometry of the superconducting sample(s) of the primary system prepared/measured in this paper for that entry (omit substrates/targets/contacts/controls unless the paper explicitly treats them as part of preparing the primary superconducting sample). If the `method` and `conditions` are the same (i.e., a batch of samples processed/prepared in exactly the same way) but multiple different sample geometries are reported, keep a single entry and list all geometries in the `geometry` array.
6. **Conditions within each method entry**: For each method entry, list only the parameter names that have explicit values in the text (number/range/inequality with units). If no explicit quantitative parameters are given for that entry, omit the `conditions` key for that entry.

# Output Format and Constraints

## Output Structure
Output pure JSON text only. No explanations, no markdown code blocks (no ```json or ```).
Output exactly one valid JSON object with the top-level key `"section2_1"`.
- `"section2_1"` must be a JSON array containing fabrication method entry objects.
- **Always include `"section2_1"`**: if no in-paper-anchored fabrication information is found for the primary system, output `{"section2_1": []}`.

## Output Content
- Only output an entry if the paper text (and supplementary information if provided) explicitly supports an in-paper-anchored fabrication record for the primary system.
- If you output an entry, `entry_id`, `method`, and `description` are required as the minimal structure for that entry (`entry_id` is the 1-based index you assign).
- All other keys (e.g., `geometry`, `conditions`) are **CONDITIONAL** and should be omitted unless explicitly supported by the text.
- Do NOT output `null` / empty placeholders.
- Do NOT add extra keys beyond the schema.
- Key names must match exactly as defined below (case-sensitive).

## JSON Validity
Preserve the original text content (including LaTeX, special symbols, etc.) when extracting. Only escape characters that would break JSON validity:
- `\` → `\\`
- `"` → `\"`
- newline → `\n`

# JSON Keys and Extraction Rules

## 1. entry_id
- **Description**: (Within each entry) a stable identifier for aligning with stage 2.
- **Extraction Rules**:
  - `entry_id` MUST be the 1-based index of this entry in the output array (1, 2, 3, ...).

## 2. method
- **Description**: (Within each entry) the fabrication method category key for this entry.
- **Extraction Rules**:
  - Must be exactly one of the allowed method categories listed below.

## 3. description (Fabrication Process Description)
- **Description**: (Within each entry) a `description` string that describes the fabrication/process steps for this entry.
- **Extraction Rules**:
  - Must be copied verbatim from the paper text (preferably one full sentence). Do NOT paraphrase.

## 4. geometry
- **Description**: (Within each method entry) the geometry of the superconducting sample(s) for the primary superconducting material/system, including (when explicitly stated) sample form/shape and/or quantitative dimensions (thickness/length/width/diameter, etc.).
- **Extraction Rules**:
  - Extract descriptions as a list of strings.
  - Each list item may contain: (i) qualitative form/shape (e.g., thin film, pellet, flake, rectangular bar) and/or (ii) explicit quantitative dimensions (with units). If both are available in the text, include both in the same item.
  - If multiple samples of the same type (e.g., thin films) are prepared for the primary system, describe them collectively in a single list item. If multiple kinds of superconducting samples for the primary system are mentioned (e.g., bulk pellets and thin films), list them as separate items in the order they appear.
  - Describe as much detail as possible.
  - **Crucial**: If there is no explicit description of geometry in the paper, **omit this key entirely** from the JSON output.
- **Example**: `"geometry": ["100 nm thick thin films on SrTiO3", "Polycrystalline pellets of 10 mm diameter"]`

Notes:
- Only include methods used for preparing the primary superconducting sample/system (ignore methods solely for substrates/targets/contacts/controls unless the paper explicitly treats them as part of preparing the primary superconducting sample).
- **Allowed method categories (for the `method` key)**:
  - `"CVD"`, `"CVT"`, `"MBE"`, `"PVD"`, `"Solution"`, `"SSR"`, `"Flux"`, `"Melt"`, `"Nano"`

- **Definitions**:
  - **CVD**: Chemical Vapor Deposition (Standard, PECVD, ALD).
  - **CVT**: Chemical Vapor Transport (using transport agents).
  - **MBE**: Molecular Beam Epitaxy (Laser-MBE, Hybrid-MBE).
  - **PVD**: Physical Vapor Deposition (PLD, Sputtering, Evaporation).
  - **Solution**: Solution-based (Sol-gel, Hydrothermal, Spin-coating).
  - **SSR**: Solid-State Reaction (Sintering, HPHT). *Distinct from Flux.*
  - **Flux**: Flux Growth (Self-flux, solvent growth).
  - **Melt**: Melt Growth (Arc Melting, Floating Zone, Bridgman).
  - **Nano**: Nanofabrication & Exfoliation (Mechanical exfoliation, Transfer, Lithography, Device fabrication).

## 5. conditions (Fabrication Conditions)
- **Description**: (Within each method entry) the specific parameter names that have explicit quantitative values in the text for that entry.
- **Extraction Rules**:
  - Extract as an **array of strings** (parameter names).
  - Include **only** parameter names (from the list below) that have explicit values in the text for that entry.
  - Do not include any parameter name unless the text explicitly provides a value (number/range/inequality, with units) for that parameter.
  - Focus on parameters for the primary superconducting sample/system; do not list parameter names that only apply to substrates/targets/contacts/controls unless the paper explicitly treats them as part of preparing the primary superconducting sample.

- **Allowed Parameters per Method Category**:
  *Select only from these lists based on the identified method:*

  - **CVD**: reactor_type, process_gases, substrate_temperature, chamber_pressure, deposition_time, deposition_rate, gas_ratios, pulse_sequence, rf_power, substrate, patterning_method, post_treatment
  - **CVT**: source_material, transport_agent, transport_agent_loading, ampoule_material, ampoule_dimensions, charge_conditions, source_temperature, growth_temperature, temperature_gradient, growth_duration, atmosphere, cooling_procedure, crystal_morphology, post_treatment
  - **MBE**: chamber_base_pressure, effusion_cells, gas_sources, substrate_temperature, growth_rate, growth_monitoring, shutter_sequence, growth_sequence, laser_parameters, plasma_parameters, cooling_rate, patterning_method, post_treatment
  - **PVD**: base_pressure, deposition_gas, deposition_pressure, substrate_temperature, target_substrate_distance, deposition_rate, deposition_duration, energy_source_parameters, laser_energy, laser_frequency, patterning_method, post_treatment
  - **Solution**: solvents, solution_concentration, additives, stirring_conditions, ph_adjustment, drying_conditions, pyrolysis_conditions, annealing_conditions, autoclave_conditions, spin_coating_parameters, electrodeposition_parameters, electrolyte_composition, epd_parameters, spray_parameters, post_treatment
  - **SSR**: precursor_preparation, starting_composition, calcination_conditions, sintering_conditions, heating_rate, cooling_rate, pressure_technique, applied_pressure, encapsulation, post_treatment
  - **Flux**: flux_material, flux_composition_ratio, crucible_material, encapsulation, max_temperature, soak_time, cooling_rate, decanting_temperature, removal_method, post_treatment
  - **Melt**: method_variant, heat_source, atmosphere, crucible_material, growth_rate, rotation_speed, feed_rod_preparation, homogenization_steps, post_treatment
  - **Nano**: starting_material, exfoliation_method, substrate, transfer_method, encapsulation, patterning_method, contact_fabrication, post_treatment

# Example Output

```json
{
  "section2_1": [
  {
    "entry_id": 1,
    "method": "PVD",
    "description": "Thin films were grown by pulsed laser deposition (PLD)",
    "geometry": ["Epitaxial thin film (~30 nm) on SrTiO3(001)", "Epitaxial thin film (~30 nm) on LSAT(001)", "Epitaxial thin film (~60 nm) on SrTiO3(001)"],
    "conditions": ["substrate_temperature", "deposition_pressure", "deposition_gas", "laser_energy", "laser_frequency", "target_substrate_distance", "deposition_duration", "post_treatment"]
  },
  {
    "entry_id": 2,
    "method": "Nano",
    "description": "Hall bar devices were patterned from the thin film and contacted with metal electrodes",
    "geometry": ["Hall bar device patterned from the epitaxial thin film (~30 nm) on SrTiO3(001)"],
    "conditions": ["substrate", "patterning_method", "contact_fabrication", "post_treatment"]
  }
]
}
```

---

# Input

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

## Single-system identification result (JSON)
{{single_system_result}}

## Paper text (Markdown) to process
{{paper_text}}

## Supplementary information (if provided)
{{supplementary_information}}