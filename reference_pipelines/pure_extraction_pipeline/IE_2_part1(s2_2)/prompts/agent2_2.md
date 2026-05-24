# Example Output

The output must be a JSON object with a single key `"section2_2"` containing an array where each element corresponds to an entry from the first-stage fabrication extraction result.
Each entry must include the exact `entry_id` from the input, the `method` from the input, and the detailed fabrication conditions for that entry's method.
Below is an example of the expected output format:

```json
{
  "section2_2": [
    {
      "entry_id": 1,
      "method": "PVD",
      "conditions": {
        "fabrication_method": "PLD",
        "chamber_base_pressure": "1e-6 mBar",
        "deposition_pressure": "1e-4 mBar",
        "deposition_atmosphere": "O2",
        "substrate_temperature": ["780 °C","800 °C","820 °C"],
        "laser_energy": "300 mJ",
        "laser_frequency": "5 Hz",
        "target_to_substrate_distance": "50 mm",
        "deposition_time": "10 min",
        "post_annealing_conditions": "500 °C in 1 atm O2 for 1 hour"
      }
    },
    {
      "entry_id": 2,
      "method": "Nano",
      "conditions": {
        "starting_material": "Bulk NbSe2 single crystal",
        "exfoliation_method": "Mechanical exfoliation",
        "substrate": "Si/SiO2 (300 nm)",
        "transfer_method": "Dry transfer",
        "patterning_method": "E-beam lithography",
        "contact_fabrication": "Ti/Au contacts via e-beam evaporation",
        "post_treatment": "Vacuum annealing at 200 °C"
      }
    }
  ]
}
```

# Input

Notes:
- The paper text to process is converted from PDF to Markdown via OCR/text conversion. Markdown formatting may reflect some document structure, but it can be noisy/unreliable; you may use it as a weak hint, but do NOT rely on it as a source of truth.
- Due to PDF-to-text conversion artifacts, the boundary between the main text and the references section may contain interleaved/crossed text; be cautious and avoid treating such artifacts as factual evidence.
- The first-stage fabrication extraction result (JSON) provides the structure you must follow: each output entry must have the same `entry_id` as the corresponding input entry, and extract detailed conditions for that entry's fabrication method.
- **Key Explanations** for the first-stage fabrication extraction result JSON (to help you identify what each entry represents and extract appropriate detailed parameters):
  - `entry_id`: Unique identifier for each fabrication entry (integer, e.g., 1, 2, 3...)
  - `method`: Fabrication method category (string, e.g., "PVD", "CVD", "Solution", "SSR", "Flux", "Melt", "Nano")
  - `description`: Verbatim description of the fabrication process from the paper text (string)
  - `geometry`: Array of sample shapes/dimensions mentioned (array of strings)
  - `conditions`: Array of parameter names that the upstream LLM judged to have explicit quantitative values in the paper for this entry (array of strings, e.g., ["substrate_temperature", "deposition_pressure"]) - **FOR REFERENCE ONLY**. Your primary task is to correctly identify and align with these upstream entries, then extract detailed condition values for each parameter listed here.
- **Important**: Multiple input entries may have the same `method` value if they represent different fabrication processes (e.g., same PVD method but different parameter sets). Each output entry must correspond exactly to one input entry based on `entry_id`.

### First-stage fabrication extraction result (JSON)
{{fabrication_extracton1}}

### Paper text (Markdown) to process
{{paper_text}}

### Supplementary information (if provided)
{{supplementary_information}}