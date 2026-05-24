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
      "signature_type": "formula_based | family_member | stack_based | elemental",
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
