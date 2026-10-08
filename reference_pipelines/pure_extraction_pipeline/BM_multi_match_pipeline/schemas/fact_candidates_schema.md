# Fact Candidates Schema

```json
{
  "paper_id": "string",
  "candidate_version": "v1",
  "paper_level_candidates": [
    {
      "candidate_id": "string",
      "fact_type": "material_identity | composition | structure | phase | synthesis_method | processing_condition | property_value | transition_temperature | critical_field | critical_current_density | carrier_concentration | defect_or_doping | secondary_phases | stack_descriptor | morphology | measurement_condition | characterization_result | computational_result | performance_metric | mechanism_or_interpretation | relation_or_trend | other_supported_fact",
      "property_name": "string or null",
      "property_category": "string or null",
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
        "composition": "string or null",
        "sample_form": "string or null",
        "atmosphere": "string or null",
        "time": "string or null",
        "frequency": "string or null",
        "voltage": "string or null",
        "current": "string or null",
        "measurement_method": "string or null",
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
- `property_name` carries the domain-specific property label when `fact_type` is generic, such as `Tc`, `Curie_temperature`, `band_gap`, or `specific_capacity`.
- `property_category` is optional and should describe the domain family only when the paper supports it.
- `local_material_mentions` should come from the immediate local context only.
