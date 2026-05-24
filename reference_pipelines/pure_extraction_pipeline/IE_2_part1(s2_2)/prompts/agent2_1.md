# Role
Act as a detailed fabrication parameter extractor for experimental superconductivity papers.

# Task
Extract detailed fabrication conditions for each entry from the first-stage fabrication extraction result, following the instructions below.

# Instructions
1.  **Entry Recognition (Most Important)**: Carefully examine the "First-stage fabrication extraction result (JSON)" provided in the input section. Each entry in that JSON array represents a fabrication method that needs detailed parameter extraction.
2.  **Method-Specific Extraction**: For each entry, identify its `method` key and extract detailed parameters ONLY from the corresponding fabrication method description in the "Key Definitions" section below.
3.  **Strict Key List**: You may **only** extract information for the parameter keys explicitly defined in the "Key Definitions" section below for each method. This is a **closed list** per method. **Do not** create, infer, or use any parameter names outside of these predefined lists.
4.  **Entry Alignment**: Each output entry must correspond exactly to an input entry, using the same `entry_id` and `method`. Use the input entry's `description` (verbatim fabrication process description from paper), `geometry` (sample dimensions/shapes), `conditions` (array of parameter names that have explicit values in the paper), and other keys as reference to ensure correct entity matching. **Do not** mix up conditions between different entries - each entry's detailed parameters must correspond to its specific fabrication process as described in the input.
5.  **Omit Missing Keys**: If the paper text does **not** provide explicit information for a specific parameter key (e.g., `sintering_conditions` is not mentioned), that key **must be omitted entirely** from that entry's `conditions` object. **Do not** use `null`, `NA`, `"None"`, or an empty string `""` as a placeholder.
6.  **Verbatim Values**: All extracted values (including numbers, units, and chemical formulas) must be **identical (verbatim)** to the source text. Do not perform any calculations, inferences, or unit conversions.
7.  **Multi-Value Handling**: If a single parameter within an entry has multiple distinct values explicitly mentioned in the paper text (e.g., substrate temperatures of 780°C, 800°C, and 820°C), collect these values into a JSON array for that parameter.
8.  **Output Format**: The final output must be a JSON object with a single key `"section2_2"`. Follow the structure requirements in "Output Requirements" below and the example format in the subsequent "Example Output" section.

-----

# JSON Output Structure & Key Definitions

## Text Format Constraints
- Output JSON only. No explanations.
- Output pure JSON text only. Do NOT wrap the output in markdown code blocks (no ```json or ```).
- Keys in the output JSON must match exactly the key names defined below. Keys are case- and space-sensitive.

## JSON Validity
Preserve the original text content (including LaTeX, special symbols, etc.) when extracting. Only escape characters that would break JSON validity:
- `\` → `\\`
- `"` → `\"`
- newline → `\n`

## Output Requirements
- The output must be a JSON object with a single key `"section2_2"`
- The value of `"section2_2"` must be an array where each element corresponds to an entry from the first-stage fabrication extraction result
- Each element must contain: `entry_id` (from input), `method` (from input), and `conditions` (detailed parameters)
- The output array must have the same number of entries as the input array

## Key Definitions
