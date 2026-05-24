# Figure Classification

## Role
Act as a figure classifier for experimental superconductivity papers.  

## Task
Given the full paper text `paper_text` (from .tex or .pdf-to-markdown), identify all figure IDs and classify each figure (or subfigure panel) into:
- section3: Microscopic characterization & electronic structure
- section4: Physical property curves
- others: neither section3 nor section4
- unsure: not enough evidence to decide between section3, section4, or others
Additionally, tag phase diagram figures as `phase_diagram` using the STRICT rule below (this is a tag, not a mutually exclusive category).

Important: Do NOT treat tables as figures. Ignore any "Table"/"Tab." identifiers (e.g., "Table 1", "Table S1") and do not include them in any output list.

From the provided `paper_text`, do the following:
1) **Figure identification**: list all figure identifiers appearing in captions and/or main-text references.
2) **Figure classification**: decide whether each figure belongs to section3, section4, others, or unsure.
3) **Phase diagram tagging (STRICT)**: tag figures/panels as phase diagrams ONLY when the text explicitly describes them as a "phase diagram".

### What you must focus on
- Figure labels/IDs (e.g., "Fig. 1", "Figure 2", "Fig. 3a")
- Figure captions
- Main-text sentences that cite figures ("…as shown in Fig. 2(b)…") and any description of what that figure shows

### What you do NOT have
- Actual images. You must infer figure type only from caption + text mentions.

---

## Figure ID recording rules
Goal: Record figure IDs in the paper's original style (prefer the caption's wording), while still deduplicating equivalent mentions.

### Core rule
- Prefer the figure ID string exactly as it appears in the FIGURE CAPTION (e.g., "Figure 2", "Fig. 2", "Extended Data Fig. 1", "Fig. S1").
- If the caption is not available in the provided text, use the first in-text mention as the canonical form.
- If neither caption nor a stable in-text form is available, fall back to a reasonable default (e.g., "Fig. 2").

### Deduplication rule
The same figure is often referenced in multiple surface forms (e.g., "Figure 2", "Fig. 2", "Fig.2", "Fig 2").
You MUST treat these as the same figure and output only ONE canonical ID string (chosen by the core rule above).

### Panel / subfigure rule
- If the paper distinguishes panels, keep that panel designation in the ID using the paper's style (e.g., "Fig. 3a" or "Fig. 3(a)").
- Treat "3a" and "3(a)" as the same panel; output whichever style is used in the caption (or first mention if caption missing).

Do NOT invent figure numbers or letters. Only include IDs that appear in the text/captions.

---

## Granularity rule
Prefer the finest granularity that is supported by explicit evidence in the text/caption.

- If the caption/main text explicitly points to a specific panel (e.g., "Fig. 2(b)", "Fig. 2b", "(b) shows ..."), you MUST record/classify at panel level (e.g., `"Fig. 2(b)"`).
- If the paper does not provide enough information to reliably map content to a specific panel, you MUST record/classify at whole-figure level (e.g., `"Fig. 2"`). Do NOT guess a panel.

Mutual exclusivity constraint:
- Each ID must belong to exactly ONE of {`section3`, `section4`, `others`, `unsure`}.
- The `phase_diagram` tag is separate and may overlap with any of the above.

---

## Section definitions for classification

### section3 = Microscopic characterization & electronic structure
Classify as section3 if the caption/text clearly indicates microscopic characterization or electronic structure information. Strong signals include:
- **Structure / microstructure / morphology**: XRD/diffraction (incl. neutron diffraction, Rietveld refinement, lattice-parameter plots), (S)TEM/SEM/AFM (incl. SAED), EDS/EDX mapping, Raman used for characterization, XPS/XAS/EELS/EXAFS, μSR, etc.
- **Electronic structure**: ARPES (band dispersion/EDC/MDC/FS), STM/STS (topography or dI/dV), QPI/FT-STS, band structure / DOS / spectral-function plots (incl. DFT/DMFT).

If uncertain, put it into `unsure` (do not guess).

### section4 = Physical property curves
Classify as section4 if the caption/text clearly indicates macroscopic physical property measurements/curves. Strong signals include:
- **Electrical transport**: R–T / ρ–T, R–H / ρ–H (magnetoresistance), Hall (ρxy–H, RH vs T), I–V / dV/dI / dI/dV (macroscopic transport, NOT STM/STS).
- **Magnetization / susceptibility**: M–T, M–H, χ–T (AC/DC), ZFC/FC, shielding fraction, SQUID/VSM traces.
- **Thermodynamics / thermal transport**: specific heat (Cp, C/T vs T), thermal conductivity κ(T,H), Nernst signal, PDO frequency shift curves.
- **Derived superconducting property summary plots** based on the above (still section4): Hc2(T), superconducting H–T phase diagram, Jc(H)/Ic(T,H), Tc vs pressure/doping/field when explicitly tied to transport/magnetization/heat-capacity.

If uncertain, put it into `unsure` (do not guess).

### others = neither section3 nor section4
Examples (put into others unless clearly tied to section3 or section4 rules above):
- Experimental setup schematics, device schematics, fabrication flow charts
- Photos of apparatus, wiring diagrams
- Pure theory cartoons/mechanism schematics without concrete electronic-structure plots
- Chemical synthesis route diagrams not showing characterization results
Note: Tables are not figures for this task; ignore them.

---

## Phase diagram tagging rule (STRICT)
Rule: Add an ID to `phase_diagram` ONLY if either:
- The figure caption explicitly calls it a "phase diagram", or
- An in-text sentence that cites that figure/panel explicitly calls it a "phase diagram"
(case-insensitive; treat "phase-diagram" as the same).

Do NOT tag it if any of the following is true:
- The text only mentions "phase boundary"/"phase boundaries"/"phase line" (or similar) but never explicitly says "phase diagram".
- The figure merely looks like a phase diagram from axis choices (e.g., T vs doping/pressure/field) but the text does not explicitly call it a phase diagram.
If unsure, do NOT tag it.

This is a tag (not a mutually exclusive category): an ID may appear in `phase_diagram` and also in any of `section3`, `section4`, `others`, or `unsure`.

---

## Procedure
Follow this procedure internally, but output ONLY the final JSON.
1) Scan the text for figure captions and figure citations:
   - captions often start with "Fig." / "Figure" and contain a descriptive sentence
   - also include figure IDs cited in main text even if captions are missing in the provided text
   - ignore any table references (e.g., "Table 1", "Table S1", "Tab. 2")
2) Build a unique, deduplicated set of figure IDs (use the figure ID recording rules above; panel-aware when possible).
3) For each ID, use caption + any in-text mentions that cite that ID to decide which category it belongs to:
   - section3 vs section4 vs others vs unsure
4) Also decide whether each ID should be tagged as `phase_diagram` using the STRICT rule above.
5) Ensure the categories/lists are consistent:
   - `all_figures` contains every unique ID found
   - `section3`, `section4`, `others`, and `unsure` must be pairwise disjoint (no overlaps)
   - Every ID in `all_figures` must appear in exactly ONE of {`section3`, `section4`, `others`, `unsure`} (i.e., their union equals `all_figures`)
   - `phase_diagram` must be a subset of `all_figures`
6) Deduplicate and sort each OUTPUT list (`all_figures`, `section3`, `section4`, `others`, `unsure`, `phase_diagram`) for readability and stability:
   - Sort by figure index in ascending order (1, 2, 3, ...), and then by panel letter when applicable (a, b, c, ...); keep each ID string in the paper's style.
   - If an ID cannot be reliably sorted, keep it in the order of first appearance.

---

## Output format
Output ONLY one valid JSON object with EXACTLY these six keys:
- "all_figures": string array
- "section3": string array
- "section4": string array
- "others": string array
- "unsure": string array
- "phase_diagram": string array

**Format Requirements:**
- No extra keys. No explanations. No markdown.
- Output pure JSON text only. Do NOT wrap the output in markdown code blocks (no ```json or ```).
- Keys in the output JSON must match exactly the key names defined above. Keys are case- and space-sensitive.

**Example output:**
```json
{
  "all_figures": ["Fig. 1", "Fig. 2(a)", "Fig. 2(b)", "Fig. 3"],
  "section3": ["Fig. 1", "Fig. 2(a)"],
  "section4": ["Fig. 2(b)", "Fig. 3"],
  "others": [],
  "unsure": [],
  "phase_diagram": ["Fig. 3"]
}
```
---

## Input
Notes:
- The paper text to process is converted from PDF to Markdown via OCR/text conversion. Markdown formatting may reflect some document structure, but it can be noisy/unreliable; you may use it as a weak hint, but do NOT rely on it as a source of truth.
- Due to PDF-to-text conversion artifacts, the boundary between the main text and the references section may contain interleaved/crossed text; be cautious and avoid treating such artifacts as factual evidence.

You will be given:
- paper_text: the full paper text (from .tex or pdf-to-markdown; includes title/abstract/main text/figure captions; supplementary information may be included)

Classify based only on what is explicitly written in `paper_text`.

### Paper text (Markdown) to process
{{paper_text}}

### Supplementary information (if provided)
{{supplementary_information}}
