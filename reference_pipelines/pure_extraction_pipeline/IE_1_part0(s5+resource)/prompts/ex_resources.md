# Role
Act as an information extractor specializing in superconductivity.

# Task
Read the provided experimental superconducting-material paper (including title, abstract, main text, figure captions, and supplementary information if provided) and extract information for the **resource-related** keys as defined in the **JSON Keys and Extraction Rules** section below.

## Fundamental Guidelines
1. **Identify Relevant Data** — Carefully read the provided text to locate information corresponding to each key defined in the **JSON Keys and Extraction Rules** section.  
2. **Handle Missing Data** — If information for a key (or an object subkey) is not explicitly stated/supported in the text, omit that key/subkey from the output. Do NOT output `null`, `NA`, `none`, or any placeholder values.
3. **Strict Adherence** — Do not infer, invent, summarize, or interpret information. Extract only what is explicitly present in the text, preserving original software/package names, repository/archive names, accession/deposition identifiers, URLs, and other resource-related identifiers/links exactly as written.
4. **URL/Identifier Fidelity and De-duplication** — Do not guess, repair, or expand incomplete URLs/identifiers. If the same software/resource/link appears multiple times, de-duplicate it in the final output:
   - Treat items as duplicates only when they are the exact same string/URL, or when the text explicitly indicates they refer to the same resource.
   - If multiple variants of the same resource are explicitly provided in the text, keep the most complete/specific one as written (e.g., a full URL rather than a shortened/truncated one); otherwise keep the first occurrence.

## Output Format and Constraints
Output exactly one JSON object as the entire output (i.e., the output starts with `{` and ends with `}`), with **no extra text** before/after it. The JSON object must have exactly one top-level key: `"resources"`. Do NOT wrap the output in markdown code blocks (no ```json or ```). See **Example Output**.

- The value of `"resources"` must be a JSON object.
- Inside `"resources"`, **output only the keys defined in `JSON Keys and Extraction Rules`**, and match key names exactly (case-sensitive).
- If no resource information is explicitly stated, output an empty object for `"resources"`: `{"resources": {}}`.
- **Use proper double quotes** around all keys and string values.
- **JSON string escaping**: when writing extracted text into JSON string values, preserve the original meaning while ensuring valid JSON by escaping special characters as needed (e.g., `"` → `\"`, `\` → `\\`, newlines → `\n`).

# JSON Keys and Extraction Rules
General rule: avoid duplicate recording across keys. A resource should be recorded in only one of `software`, `code`, `raw_data`, or `cif` based on the most specific matching rule.

## 1. `software`
- **Description**: Record software/packages explicitly used for numerical calculation, simulation, or theoretical modeling in the paper.
- **Examples**: `"Quantum Espresso"`, `"VASP"`, `"WIEN2k"`, `"ABINIT"`, `"LAMMPS"`, `"CASTEP"`, `"Wannier90"`, `"WannierTools"`
- **Extraction Rules**:
  - Extract only software/packages explicitly stated as being used in the work.
  - If mentioned only as background/related work and not actually used, do not record it.
  - If both full name and abbreviation are provided, use the full name.
  - If multiple software/packages are used, record them as an array of strings.
  - Do not record general-purpose programming libraries (e.g., "numpy", "pandas", "matplotlib") unless the paper explicitly frames them as the software used for the scientific calculation/simulation.
  - Do not duplicate items in `code` or `raw_data`; this key is only for software names, not availability statements or URLs.
  - If nothing relevant is mentioned, omit this key.

## 2. `code`
- **Description**: Record whether the authors made code (including analysis scripts) available, and any corresponding link(s).
- **Extraction Rules**:
  - Emit this key **only if** the text explicitly states code is available; include subkeys only when explicitly supported.
  - Represent as a **single object** when only one code item is mentioned; use a **list of objects** when multiple code items are mentioned.
  - Each item may include (omit any subkey not explicitly supported):
    - `"type"`: short phrase describing what the code is (e.g., analysis scripts, simulation code), **only if** explicitly stated.
    - `"availability"`: set to `"yes"` only when the text states code is available.
    - `"link"`: a valid repository or archive URL (e.g., GitHub, GitLab, Zenodo, Figshare, OSF). If none, omit `"link"`.
    - `"notes"`: brief context about where the code is referenced (e.g., "Code availability" statement, supplementary info). Omit if not explicitly stated.
  - Set `"availability"` to `"yes"` only when the text states code is available.
  - `"link"` should be a valid repository or archive URL (e.g., GitHub, GitLab, Zenodo, Figshare, OSF).
  - If multiple valid links are provided, record them as an array of strings.
  - If the text explicitly states code is available (so `"availability": "yes"`), but no link can be extracted, omit the `"link"` subkey and still output the `code` object/item.
  - Do **not** use irrelevant URLs (e.g., DOI, publisher landing page, general institution homepage) as `"link"`.
  - Do not duplicate links or statements that are strictly about data or CIF; those belong to `raw_data` or `cif`.
  - Weak hint: relevant information often appears around section headings or phrases such as "Code availability", "Data and code availability", "Availability of data and materials", "Methods", "Supplementary information", or "Author information". You must still search the full text and extract only what is explicitly stated.

## 3. `raw_data`
- **Description**: Record explicitly stated **raw data / data underlying the findings / source data**. Typical types include raw/underlying data behind curves/plots, diffraction/structure-related raw data, or other source data.
- **Extraction Rules**:
  - Emit this key **only if** the text explicitly states raw data is available; include subkeys only when explicitly supported.
  - Represent as a **single object** when only one raw data item is mentioned; use a **list of objects** when multiple items are mentioned.
  - Each item should include:
    - `"type"`: the raw data type (short explicit phrase from the text; do not infer).
    - `"availability"`: set to `"yes"` only when the text states availability.
    - `"link"`: a valid dataset/archive URL (e.g., Zenodo, Figshare, Dryad, OSF) or an explicit dataset landing page provided by the authors. If none, omit `"link"`.
    - `"notes"`: brief context on where the raw data is referenced (e.g., specific figure/caption, supplementary information, or a paper-level data availability statement). Omit if not explicitly stated.
  - If the data described is a CIF/crystallographic structure file, record it under `cif` instead of `raw_data`.
  - If multiple valid links are provided for one item, record `"link"` as an array of strings.
  - Do **not** use irrelevant URLs (e.g., DOI, publisher landing page) as `"link"`.
  - Weak hint: relevant information often appears around "Data availability", "Data availability statement", "Availability of data and materials", "Supplementary information", or "Author information". You must still search the full text and extract only what is explicitly stated.

## 4. `cif`
- **Description**: Record whether a CIF file (or crystallographic structure file) is provided or deposited, including repository name and deposition identifier(s) when present.
- **Extraction Rules**:
  - Emit this key **only if** the text explicitly states CIF availability/deposition (e.g., CIF deposited to CCDC/ICSD or deposition numbers provided); include subkeys only when explicitly supported.
  - Represent as a **single object** when only one CIF/deposition item is mentioned; use a **list of objects** when multiple CIF/deposition items are mentioned (e.g., different repositories, or clearly separate deposition sets).
  - Each item should include (omit any subkey not explicitly supported):
    - `"availability"`: set to `"yes"` only when the text states CIF/structure file is available/deposited.
    - `"repository"`: the named repository/database if present (e.g., `"CCDC"`, `"ICSD"`).
    - `"identifier"`: deposition numbers / accession IDs exactly as written (string or array if multiple).
    - `"link"`: explicit repository landing page URL if provided; if not provided, omit `"link"`.
    - `"notes"`: brief context on where the CIF is referenced (e.g., "Data availability" statement, supplementary info). Omit if not explicitly stated.
  - Set `"availability"` to `"yes"` only when the text states CIF/structure file is available/deposited.
  - `"repository"` should be the named repository/database if present (e.g., `"CCDC"`, `"ICSD"`).
  - `"identifier"` should capture deposition numbers / accession IDs exactly as written (string or array if multiple).
  - `"link"` should be the explicit repository landing page URL if provided; if not provided, omit the `"link"` subkey.
  - Do **not** fabricate identifiers or links.

# Example Output

This is an **illustrative** example showing the JSON format (including complete keys). The example values below are **fabricated**. In real outputs, you should **omit any key/subkey** that you cannot explicitly support from the provided text, and you must **not copy** the example values—extract values from the provided text only.

```json
{
  "resources": {
  "software": ["Quantum Espresso", "VASP"],
  "code": [{"type": "analysis scripts", "availability": "yes", "link": "https://github.com/abcxyz123"}, {"availability": "yes", "link": "https://zenodo.org/record/7654321", "notes": "Code availability"}],
  "raw_data": {"type": "source data for Fig. 2", "availability": "yes", "link": "https://zenodo.org/record/1234567", "notes": "Figure 2 caption"},
  "cif": [{"availability": "yes", "repository": "CCDC", "identifier": ["1234567", "1234568"]}, {"availability": "yes", "repository": "ICSD", "identifier": "987654"}]
  }
}
```

# Input

Notes:
- The paper text to process is converted from PDF to Markdown via OCR/text conversion. Markdown formatting may reflect some document structure, but it can be noisy/unreliable; you may use it as a weak hint, but do NOT rely on it as a source of truth.
- Due to PDF-to-text conversion artifacts, the boundary between the main text and the references section may contain interleaved/crossed text; be cautious and avoid treating such artifacts as factual evidence.

## Paper text (Markdown) to process
{{paper_text}}

## Supplementary information (if provided)
{{supplementary_information}}