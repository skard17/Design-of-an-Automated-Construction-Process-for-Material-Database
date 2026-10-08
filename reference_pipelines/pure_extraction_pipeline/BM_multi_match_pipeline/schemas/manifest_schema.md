# Manifest Schema

```json
{
  "paper_id": "string",
  "paper_title": "string or null",
  "manifest_version": "v1",
  "material_targets": [
    {
      "target_id": "string",
      "canonical_name": "string",
      "signature_type": "formula_based | family_member | composition_variant | phase_variant | structure_variant | stack_based | elemental | composite_or_device",
      "aliases": ["string"],
      "family_context": "string or null",
      "sibling_targets": ["string"],
      "must_exclude_aliases": ["string"],
      "evidence_quotes": ["string"],
      "target_specificity": "explicit_target_specific | likely_target_specific | ambiguous"
    }
  ],
  "paper_scope_notes": ["string"]
}
```

## Notes

- `material_targets` is the authoritative roster for later stages.
- `must_exclude_aliases` is crucial for anti-mixing.
- `target_id` should be stable and filesystem-safe.
- Targets are domain-general materials records. They may be compounds, composition variants, phases, stacks, composites, devices, or processing states when the paper reports target-specific evidence.
