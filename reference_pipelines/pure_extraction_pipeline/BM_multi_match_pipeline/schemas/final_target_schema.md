# Final Target Schema

```json
{
  "primary_signature": "string",
  "material_info": {
    "section0": {},
    "section1": {},
    "section2": [],
    "section3": {},
    "section4": {}
  },
  "section5": {},
  "paper_info": {
    "metadata": {
      "paper_id": "string",
      "target_id": "string",
      "canonical_name": "string",
      "aliases_used": ["string"],
      "excluded_siblings": ["string"],
      "multi_material_provenance": {
        "accepted_candidate_ids": ["string"],
        "ambiguous_candidate_ids": ["string"]
      },
      "quality_control": {
        "has_target_specific_superconducting_evidence": true,
        "ambiguity_flags": ["string"],
        "omission_reasons": ["string"]
      }
    },
    "resources": {}
  }
}
```

The final target output intentionally follows the single-material final record shape:

- `primary_signature`
- `material_info`
- `section5`
- `paper_info`

Multi-material bookkeeping is kept inside `paper_info.metadata`, because each target emitted by this pipeline is still one material record.
