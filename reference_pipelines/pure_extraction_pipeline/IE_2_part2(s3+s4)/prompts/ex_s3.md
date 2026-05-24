# Task Overview

You are an information extractor specializing in superconductivity. Your task is to extract microscopic characterization and electronic structure information from a scientific article, using figure classification results provided by an upstream LLM as a reference to assist your extraction.

**Input:** You will receive two inputs:
1. Figure classification results: A JSON object from the upstream LLM containing the following keys:
   - `"section3"`: figures/panels classified as Section3 (microscopic characterization & electronic structure)
   - `"section4"`: figures/panels classified as Section4 (physical property curves)
   - `"others"`: figures/panels that are neither Section3 nor Section4
   - `"unsure"`: figures/panels that cannot be confidently classified into Section3/Section4/others based on text evidence
2. Article text: The full paper text of the scientific article (including title, abstract, main text, figure captions; supplementary information may be included)

The specific content of these inputs will be provided in the **Input** section below.

**Task:**
*   **General principle:** Generally follow the upstream LLM's figure classification results as a guideline. Extract information that satisfies the key definitions and requirements listed below.
*   **Primary focus:** For figures classified as `"section3"` and `"unsure"` by the upstream LLM, extract all information that meets the key definitions and requirements.
*   **Other figures:** Also check figures in `"section4"` and `"others"` to guard against upstream misclassification/omissions, but apply stricter criteria. Record information only if it strictly satisfies all key definitions and requirements; be more cautious and vigilant when evaluating these figures.
*   **Information sources:** Extract information from the figure captions and from the relevant parts of the article text that discuss these figures (note: you only have access to the text, not the actual figure images).

# Output Format

Construct the extracted information into a JSON object containing a unique key: `"section3"`.

**Hard requirement (even when nothing is found)**:
- The output MUST always include the top-level key `"section3"`.
- If no extractable Section3 information is found, output an empty object: `{"section3": {}}`.

## Text Format Constraints
When outputting:
- Output JSON only. No explanations.
- Output pure JSON text only. Do NOT wrap the output in markdown code blocks (no ```json or ```).
- Keys in the output JSON must match exactly the key names defined above. Keys are case- and space-sensitive.
- Do NOT output an empty `section3` object if the paper text contains extractable Section3 information.

The keys within the `"section3"` object are specified in the **JSON Keys and Extraction Rules** section below.

General rule for keys inside `"section3"`:
- Include a key only if it is explicitly supported by the article text/captions.
- Do NOT output `null` values in the final JSON; omit missing keys entirely.
- **Strictly adhere to the original intent of the article**; do not fabricate or infer any information not explicitly stated.
*   **Figure Naming Convention:**
    *   **Goal:** Keep figure IDs in the paper's original style (prefer the FIGURE CAPTION's wording).
    *   **Canonical choice (caption-first):** If the same figure/panel is referenced with multiple surface forms, choose ONE canonical ID string:
        *   Prefer the ID string as written in the figure caption.
        *   If the caption is not available, use the first in-text mention as the canonical form.
    *   **Deduplication:** Treat equivalent surface forms as the same figure/panel and record only ONE canonical ID. Examples of equivalent forms: `"Figure 2"` vs `"Fig. 2"`; `"Fig. 3a"` vs `"Fig. 3(a)"`.
    *   **Panels/subfigures:** If the paper distinguishes panels (a/b/c...), include the panel in the ID using the paper's style (e.g., `"Fig. 2(b)"` or `"Fig. 2b"`). Treat `"2b"` and `"2(b)"` as the same panel; output whichever style is used in the caption (or first mention if caption missing).
    *   **Source requirement:** Extract figure IDs directly from the article text (captions and/or in-text citations). Do NOT invent IDs.
    *   **Prohibition:** Do not create figure IDs based on concepts mentioned in the text (e.g., if the text mentions "phase diagram" but does not reference a specific figure ID, do not create "Fig. phase"). Do not infer or guess figure IDs that are not explicitly stated in the article.
*   Single figures are represented as strings. If a key corresponds to multiple figures, represent them as a string array.
    *   **Example:** `"XRD": {"figure": "Figure 2"}`; `"STM": {"figure": ["Fig. 3(a)", "Fig. 3(b)"]}`

*   **Per-key object format (for every key under `"section3"`):**
    *   Each key's value should be an object with:
        *   `"figure"`: a string or string array of figure/panel IDs.
        *   `"raw_data"` (optional): set to `"yes"` only if the paper explicitly states figure/curve-level underlying/source data availability for the relevant figure(s) (e.g., “Source Data for Fig. 3”). If not explicitly stated for that figure/key, omit this sub-key entirely (do not output `"no"` or `null`). Do not infer this from a paper-level Data Availability statement.

# JSON Keys and Extraction Rules

### 1. crystal_structure_diagram
*   **Description:** Record figure ID(s) for an explicit crystal-structure schematic (not experimental data), e.g., ball-and-stick or polyhedral models illustrating atomic arrangement.
*   **Note:** Record figure ID(s) only when the caption/text explicitly links a specific figure to a crystal-structure schematic (e.g., ball-and-stick or polyhedral model). Do NOT confuse this with measurement-result figures defined below (e.g., diffraction patterns for `XRD`/neutron or microscope micrographs for `SEM`/`TEM`).

### 2. XRD
*   **Description:** Record figure ID(s) for X-ray Diffraction (XRD) measurement results (if present).
*   **Note:** Record figure ID(s) only when the caption/text explicitly links a specific figure to XRD data/results (e.g., XRD pattern / intensity–2θ plot). Do NOT record XRD if it is mentioned without citing any specific figure or presenting data/results. Common confusion cases: crystal-structure schematics (record under `crystal_structure_diagram`, not `XRD`).

### 3. NMR
*   **Description:** Record figure ID(s) for Nuclear Magnetic Resonance (NMR) measurement results (if present).
*   **Note:** Record figure ID(s) only when the caption/text explicitly links a specific figure to NMR/NQR data/results (e.g., NMR spectra/lineshape, Knight shift, \(1/T_1\), \(T_1\), \(1/T_1T\)). Do NOT assume \(T_1\) or \(1/T_1\) is NMR unless NMR/NQR is explicitly mentioned. Do NOT record NMR if it is mentioned without citing any specific figure or presenting data/results.

### 4. AFM
*   **Description:** Record figure ID(s) for Atomic Force Microscopy (AFM) measurement results (if present).
*   **Note:** Record figure ID(s) only when the caption/text explicitly links a specific figure to AFM data/results (e.g., topography images, height profiles). Do NOT record AFM if it is mentioned without citing any specific figure or presenting data/results.

### 5. SEM
*   **Description:** Record figure ID(s) for Scanning Electron Microscopy (SEM) measurement results (if present).
*   **Note:** Record figure ID(s) only when the caption/text explicitly links a specific figure to SEM data/results (e.g., surface morphology images, cross-section micrographs). Do NOT record SEM if it is mentioned without citing any specific figure or presenting data/results. Common confusion cases: EDS/EDX mapping figures, optical photos, or TEM micrographs (do not count as SEM unless explicitly labeled as SEM).

### 6. TEM
*   **Description:** Record figure ID(s) for Transmission Electron Microscopy (TEM) measurement results (if present).
*   **Note:** Record figure ID(s) only when the caption/text explicitly links a specific figure to TEM data/results (e.g., TEM/HRTEM/STEM micrographs, SAED patterns). Do NOT record TEM if it is mentioned without citing any specific figure or presenting data/results. Common confusion cases: SEM images, EDS/EDX mapping figures, or optical photos (do not count as TEM unless explicitly labeled as TEM/STEM/SAED).

### 7. STM
*   **Description:** Record figure ID(s) for Scanning Tunneling Microscopy (STM) topography results (if present).
*   **Note:** Record figure ID(s) only when the caption/text explicitly links a specific figure to STM topography data/results (e.g., STM topographic images). Do NOT record STM if it is mentioned without citing any specific figure or presenting data/results. Common confusion cases: STS/dI/dV spectroscopy figures (record under `STS`, not `STM`).

### 8. STS (Scanning Tunneling Spectroscopy)
*   **Description:** Record figure ID(s) for Scanning Tunneling Spectroscopy (STS) results (e.g., dI/dV spectra) used to probe the local density of states (DoS) (if present).
*   **Note:** Record figure ID(s) only when the caption/text explicitly links a specific figure to STS spectroscopy data/results (e.g., dI/dV or differential conductance spectra/curves, tunneling spectra, gap spectra). Do NOT record STS if it is mentioned without citing any specific figure or presenting data/results. Do NOT confuse STS with STM topography images.

### 9. band_structure (Energy Band Structure)
*   **Description:** Record figure ID(s) for energy band-structure plots. Here, a band-structure plot is defined as an $E-E_F$ vs. $k$ relation plot covering high-symmetry momentum paths (theoretical or experimentally extracted).
*   **Note:**
    *   Band structures can come from first-principles calculations (e.g., DFT, DMFT) or experimentally extracted high-symmetry path dispersions (e.g., from ARPES).
    *   The horizontal axis should follow high-symmetry k-paths (e.g., $\Gamma–X–M–\Gamma$) or be explicitly described as a high-symmetry path in the caption/text. If the plot is not along a high-symmetry path, it may be more likely `band_dispersion` than `band_structure`.
    *   If the paper indicates both theoretical and experimental band-structure results (across different panels or different figures), record all corresponding figure ID(s) as an array.

### 10. band_dispersion (Band Dispersion)
*   **Description:** Record figure ID(s) for Angle-Resolved Photoemission Spectroscopy (ARPES) band-dispersion data (e.g., E–k dispersion plots/maps) (if present).
*   **Note:**
    *   Unlike `band_structure`, band dispersion data from ARPES does not necessarily follow high-symmetry paths and is often presented as raw or fitted E–k intensity maps.

### 11. EDC (Energy Distribution Curve)
*   **Description:** Record figure ID(s) for Energy Distribution Curve (EDC) plots. An EDC is photoemission intensity vs. binding energy at fixed momentum (typically via ARPES).
*   **Note:** Record figure ID(s) only when the caption/text explicitly links a specific figure to EDC data/results (e.g., explicitly says "EDC" or describes an ARPES intensity vs binding-energy curve at fixed momentum). Do NOT confuse EDC with STS dI/dV spectra or generic DOS/lineshape plots unless ARPES/EDC is explicitly indicated.

### 12. MDC (Momentum Distribution Curve)
*   **Description:** Record figure ID(s) for Momentum Distribution Curve (MDC) plots. An MDC is photoemission intensity vs. momentum at fixed energy (typically via ARPES).
*   **Note:** Record figure ID(s) only when the caption/text explicitly links a specific figure/panel to MDC data/results (e.g., explicitly says "MDC"/"momentum distribution curve", or explicitly says MDC curves are extracted at fixed energy such as $E_F$). Do NOT infer MDC from ARPES dispersion mentions alone: if the caption/text only describes an "ARPES band dispersion"/"E–k map"/"intensity map", record it under `band_dispersion` (not `MDC`).

### 13. fermi_surface (Fermi Surface)
*   **Description:** Record figure ID(s) for Fermi-surface maps. Here, a Fermi-surface map shows electronic states at the Fermi energy ($E_F$) in momentum space.
*   **Note:** Record figure ID(s) only when the caption/text explicitly links a specific figure/panel to a Fermi-surface map (e.g., explicitly says "Fermi surface"/"FS map" or describes a momentum-space map at $E_F$). Do NOT infer a Fermi-surface map from generic ARPES intensity/map wording unless $E_F$/Fermi surface is explicitly indicated.

### 14. others
*   **Description:** Record other structure-related characterization or electronic-structure figure(s) used in the article that do not fit the standard keys above (e.g., EDS, RHEED, Raman, neutron diffraction, XPS/XAS/EELS, DOS curves, QPI/FT-STS maps, DMFT spectral functions, etc.).
*   **Example:** Single figure: `{"name": "EDS", "figure": "Fig. 5"}`; Multiple figures: `{"name": "EDS", "figure": ["Fig. 1", "Fig. 2", "Fig. 3"]}`; Optional raw_data: `{"name": "Raman", "figure": "Fig. 6(a)", "raw_data": "yes"}`
*   **Note:**
    *   Use `others` only for additional microscopic characterization and electronic structure methods/results that do not fit any predefined key above. If something fits a predefined key, do NOT put it in `others`.
    *   If multiple methods are used, record each method as a separate object in the array (e.g., `"others": [{"name": "EDS", "figure": "Fig. 9"}, {"name": "RHEED", "figure": "Fig. 6"}, {"name": "Raman", "figure": ["Fig. 6(a)", "Fig. 6(b)", "Fig. 6(c)"]}]`). If a single method records multiple figures, treat it as one object with figures recorded as a string array.
    *   Only record the method if the article text or figure captions indicate actual measurement results (e.g., Raman spectra figures, diffraction data mentioned in captions or text).
    *   If a method is merely mentioned (e.g., "Film quality was checked by TEM") but no figure caption or data description is provided, do not include that method.
    *   Strictly retain the method name from the article. If both full name and abbreviation are provided, use the abbreviation.
    *   Each method should be an object in the array, with the method name written exactly as in the article, accompanied by actual figure ID(s). Do not fabricate figure ID(s).
    *   Optional `"raw_data"`: set to `"yes"` only if the paper explicitly states figure/curve-level underlying/source data availability for the corresponding figure(s); otherwise omit.
    *   If no such additional methods exist, do not output `others` at all (omit the key).

# Extraction Guidelines

1.  **Focus on classified figures:** Use figures/panels classified as `"section3"` by the upstream LLM as your primary scope. Also review `"unsure"` figures/panels as a secondary scope to catch potential upstream ambiguity/misclassification. Check figures in `"section4"` and `"others"` with stricter criteria. Read the relevant sections of the article (excluding references) that discuss these figures.
2.  **Text-only & evidence rule:** You only have access to text (including figure captions), not images. Extract/record only what is explicitly stated in the caption/text and linked to a specific figure ID; do not match keys with values from unrelated contexts.
3.  **Key separation rule:** If figure-related information fits any predefined key, record it under that key (do NOT put it in `others`). If it does not fit any predefined key but is still figure-related microscopic characterization and electronic structure information, record it under `others` (do NOT create a new top-level key).
4.  For methods already covered by predefined keys: if the paper mentions the full name (e.g., "X-ray diffraction", "Scanning Electron Microscopy"), still output using the predefined key name (`XRD`, `SEM`, etc.). Do NOT put such cases into `others`.

## Self-check before output
- Self-check that you followed the **Extraction Guidelines** above (especially the text-only evidence rule and the key-separation rule).
- Key relationship check (panel-aware):
  - Overlap is allowed across different panels/figures (e.g., one multi-panel figure may contain `band_dispersion`, `EDC`, `MDC`, and `fermi_surface` in different panels).
  - For the SAME figure/panel ID, try to assign the most appropriate single key and keep the following pairs mutually exclusive (i.e., do NOT record the same figure/panel ID under both keys). Assign multiple keys to the SAME figure/panel ID only with explicit textual support (see below).
    - `STM` (topography) vs `STS` (dI/dV spectroscopy)
    - `EDC` (intensity vs binding energy at fixed momentum) vs `MDC` (intensity vs momentum at fixed energy)
    - `band_structure` (high-symmetry k-path band structure) vs `band_dispersion` (ARPES dispersion/intensity map not explicitly along a high-symmetry path)
  - If the text does not clearly distinguish panels, prefer the most specific key that is explicitly indicated, and do NOT duplicate the same figure ID across multiple keys by guessing.
  - A single panel may contain multiple plot types; assign multiple keys to the SAME figure/panel ID only if the caption/text explicitly supports it (do not guess). Example: "Fig. 3(b) shows an ARPES dispersion map with extracted MDC curves" → record `band_dispersion` and `MDC` for `"Fig. 3(b)"`.

# Example Output

```json
{
 "section3": {
  "crystal_structure_diagram": {"figure": "Fig. 1"},
  "XRD": {"figure": "Fig. 2", "raw_data": "yes"},
  "NMR": {"figure": "Fig. 3"},
  "AFM": {"figure": "Fig. 4"},
  "SEM": {"figure": "Fig. 5(c)"},
  "TEM": {"figure": "Fig. 5(a)"},
  "STM": {"figure": ["Fig. 11", "Fig. 12", "Fig. 13", "Fig. 14"]},
  "band_structure": {"figure": ["Fig. 15(b)", "Fig. 15(c)"]},
  "band_dispersion": {"figure": "Fig. 16(b)", "raw_data": "yes"},
  "EDC": {"figure": "Fig. 16(c)"},
  "MDC": {"figure": "Fig. 16(d)"},
  "fermi_surface": {"figure": "Fig. 17(c)"},
  "STS": {"figure": ["Fig. 18", "Fig. 19", "Fig. 20"], "raw_data": "yes"},
  "others": [
    {
      "name": "EDS",
      "figure": "Fig. 9"
    },
    {
      "name": "Raman",
      "figure": ["Fig. 6(a)", "Fig. 6(b)", "Fig. 6(c)"],
      "raw_data": "yes"
    },
    {
      "name": "momentum resolved density of states",
      "figure": "Fig. 5"
    }
  ]
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
