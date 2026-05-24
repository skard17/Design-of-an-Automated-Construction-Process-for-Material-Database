# Fact Candidates Schema

```json
{
  "paper_id": "string",
  "candidate_version": "v1",
  "paper_level_candidates": [
    {
      "candidate_id": "string",
      "fact_type": "Tc | Jc | Hc1 | Hc2 | Hc | P_sc | P_nsc | lambda | xi | electronic_state_tuning_mechanism | carrier_concentration | secondary_phases | stack_descriptor",
      "value": "string or object",
      "unit": "string or null",
      "verbatim_evidence": "string",
      "source_anchor": {
        "section_label": "string or null",
        "figure": "string or null",
        "table": "string or null"
      },
      "local_material_mentions": ["string"],
      "local_series_mentions": ["string"],
      "attribution_hint": {
        "candidate_targets": ["string"],
        "reason": "same_sentence | same_caption | same_table_column | nearby_paragraph | family_level | unclear"
      },
      "conditions": {
        "temperature": "string or null",
        "magnetic_field": "string or null",
        "pressure": "string or null",
        "direction": "string or null",
        "characteristics": "string or null"
      },
      "confidence": 0.0
    }
  ],
  "paper_level_unassigned_notes": ["string"]
}
```

## Notes

- This stage extracts candidate facts without final ownership.
- `value` can be string or object depending on fact type.
- `local_material_mentions` should come from the immediate local context only.
