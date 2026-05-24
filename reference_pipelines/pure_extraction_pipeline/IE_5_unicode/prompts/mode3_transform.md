# Task

Normalize the provided text into a clean Unicode-rendered form while preserving the original meaning exactly.

## Primary Goal

Produce a character-level normalization, not a rewrite.

You may:
- convert LaTeX-style symbols into Unicode when safe
- normalize super/subscripts when they are clearly part of formulas or notations
- fix escaped Unicode sequences
- preserve already-correct Unicode text

You must not:
- paraphrase
- summarize
- translate
- expand abbreviations
- invent missing content
- reorder information
- change domain meaning

## Hard Constraints

1. Keep the original meaning exactly unchanged.
2. Keep the original language unchanged.
3. Keep punctuation, spacing, and casing as close as possible unless a normalization requires a small local change.
4. If the input is already clean, return it unchanged.
5. Do not hallucinate corrections when uncertain.
6. Prefer conservative output over aggressive rewriting.

## Safe Normalizations

### 1. Unicode escape cleanup

- Convert escaped Unicode sequences like `\\u03b1` into their intended Unicode characters when clearly intended.

### 2. LaTeX symbol normalization

Convert common LaTeX-like commands into Unicode when safe, such as:

- `\\alpha` -> `\\u03b1`
- `\\beta` -> `\\u03b2`
- `\\gamma` -> `\\u03b3`
- `\\Delta` -> `\\u0394`
- `\\times` -> `\\u00d7`
- `\\pm` -> `\\u00b1`
- `\\approx` -> `\\u2248`
- `\\leq` -> `\\u2264`
- `\\geq` -> `\\u2265`
- `\\rightarrow` -> `\\u2192`
- `\\degree` or `\\circ` -> `\\u00b0`

If a command is ambiguous or not clearly a symbol, keep it unchanged.

### 2.1 Inline LaTeX fragments in running text

If the text contains short LaTeX fragments embedded inside normal prose, normalize the fragment and keep the surrounding sentence unchanged.

Typical cases:

- `K$_2$Cr$_3$As$_3$` -> `K₂Cr₃As₃`
- `La$_3$Ni$_2$O$_7$` -> `La₃Ni₂O₇`
- `FeSe$_{0.5}$Te$_{0.5}$` -> `FeSe₀.₅Te₀.₅`

Rules:

- remove formatting-only `$...$` wrappers when they are only marking inline math
- preserve the surrounding natural-language text exactly
- if the fragment is clearly a chemical formula, render indices as Unicode subscripts
- if the fragment is already normalized, keep it unchanged

### 3. Super/subscript normalization

Convert superscripts and subscripts into Unicode only when they are clearly notation, such as:

- `10^3` -> `10` plus superscript 3
- `cm^2` -> `cm` plus superscript 2
- `H_2O` -> `H` plus subscript 2 plus `O`

Be conservative with variable names. Do not over-convert ordinary identifiers.

### 4. Chemical formula normalization

For chemical formulas, convert stoichiometric indices into Unicode subscripts when safe.

Examples:
- `H2O` -> water formula with subscript 2
- `La3Ni2O7` -> formula with 3, 2, 7 rendered as subscripts
- `Nd0.8Sr0.2NiO2` -> decimal stoichiometric values rendered with subscript digits
- `FeSe0.5Te0.5` -> decimal stoichiometric values rendered with subscript digits
- `K$_2$Cr$_3$As$_3$` -> `K₂Cr₃As₃`
- `BaFe$_{2-x}$Co$_x$As$_2$` -> keep the variable structure, but normalize obvious subscripts safely
- `Sr$_2$RuO$_4$` -> `Sr₂RuO₄`

If the formula is already correct, keep it unchanged.

### 5. Unit and symbol cleanup

Normalize common scientific units and symbols only when safe:

- malformed degree expressions may become degree sign plus unit when clearly intended
- preserve `K`, `T`, `Pa`, `eV`, `meV`, `GHz`, `nm`, `um`, `cm-1` when already valid

Do not rewrite units semantically.

## Must-Preserve Content

The following should usually be preserved exactly unless they only need character decoding:

- DOI
- arXiv ID
- URL
- file names
- record IDs
- bibliographic identifiers
- paper type labels
- years
- venue abbreviations that are already correct

Examples:
- `10.1103/PhysRevB.75.195116` must stay the same
- `https://doi.org/...` must stay a valid URL
- `0704.0093v1` must stay the same

## Metadata-Specific Guidance

If the input is a title, venue, or short bibliographic field:

- normalize symbols and encoding issues
- preserve wording exactly
- do not rewrite title style
- do not expand abbreviations like `Phys. Rev. B`
- do not change capitalization unless the input is clearly corrupted
- if a title contains an inline chemical formula, normalize the formula but keep the title wording unchanged

Example:

- `Evidence of P-wave Pairing in K$_2$Cr$_3$As$_3$ Superconductors from Phase-sensitive Measurement`
  should become
  `Evidence of P-wave Pairing in K₂Cr₃As₃ Superconductors from Phase-sensitive Measurement`

## Ambiguity Rule

If you are not sure whether a transformation is safe, keep the original text.

## Input

Field key: `{{key_name}}`
Part: `{{part}}`
Field path: `{{field_path}}`

```json
{{input_pair_json}}
```

## Output Format

- Return only one valid JSON object.
- Use exactly the same key name: `{{key_name}}`
- Do not output markdown, comments, explanations, or code fences.

Expected shape:

```json
{"{{key_name}}": "...normalized unicode text..."}
```
