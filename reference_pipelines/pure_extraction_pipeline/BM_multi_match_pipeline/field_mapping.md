# General Material Field Mapping

This file describes the active multi-material extraction contract after the prompt generalization.

The pipeline is no longer a superconductivity-specific field mapper. It is a material-database attribution framework:

1. build a target roster for all material systems that deserve separate records;
2. extract paper-level candidate facts without assigning ownership too early;
3. match each candidate fact to a target material conservatively;
4. aggregate accepted facts into one evidence-backed material record per target.

## Output Skeleton

Each target still uses the existing single-material-style skeleton:

- `primary_signature`
- `material_info.section0`
- `material_info.section1`
- `material_info.section2`
- `material_info.section3`
- `material_info.section4`
- `section5`
- `paper_info.metadata`
- `paper_info.resources`

The meaning is now domain-general:

- `section0`: target identity, composition, structure, phase, synthesis, processing, morphology, stack, mechanism, relation, and characterization context.
- `section1`: target-specific material properties and performance metrics keyed by extracted `property_name`.
- `paper_info.metadata`: provenance, aliases, excluded siblings, and quality-control information.

## Candidate Fact Contract

Candidate facts are not limited to a fixed property vocabulary.

Core fields:

- `fact_type`: broad category such as `composition`, `structure`, `synthesis_method`, `property_value`, `performance_metric`, `characterization_result`, or `relation_or_trend`.
- `property_name`: the paper-specific property label when the fact is a property value, such as `Tc`, `Curie_temperature`, `band_gap`, `specific_capacity`, or `thermal_conductivity`.
- `property_category`: optional domain family such as `superconducting`, `magnetic`, `electrochemical`, `catalytic`, `thermoelectric`, `mechanical`, `optical`, `electronic`, or `structural`.
- `value` and `unit`: the extracted value and unit when present.
- `verbatim_evidence`: short evidence quote.
- `source_anchor`: section, figure, or table location.
- `local_material_mentions`: nearby material mentions used for attribution.
- `attribution_hint`: candidate target names and the local attribution reason.
- `conditions`: temperature, pressure, field, atmosphere, sample form, composition, direction, measurement method, and other explicitly stated conditions.

## Aggregation Mapping

### `section0`

The aggregator maps these fact types into `material_info.section0`:

- `material_identity`
- `composition`
- `structure`
- `phase`
- `synthesis_method`
- `processing_condition`
- `defect_or_doping`
- `morphology`
- `measurement_condition`
- `characterization_result`
- `computational_result`
- `mechanism_or_interpretation`
- `relation_or_trend`
- `carrier_concentration`
- `secondary_phases`
- `stack_descriptor`

Historical aliases are still normalized:

- `tuning` -> `electronic_state_tuning_mechanism`
- `secondary_phase` -> `secondary_phases`

### `section1`

The aggregator maps accepted property and performance facts into `material_info.section1`.

Rules:

- If `property_name` exists, it becomes the section key after light normalization.
- If `property_name` is absent but `fact_type` is itself a property name, `fact_type` becomes the section key.
- Legacy superconducting property labels such as `Tc`, `Jc`, `Hc1`, `Hc2`, `Hc`, `P_sc`, `P_nsc`, `lambda`, and `xi` remain supported as property names for backward compatibility.
- New domains should use `property_value` plus `property_name` instead of adding a hard-coded field for every possible property.

Examples:

- `property_name = "Tc"` -> `material_info.section1.Tc`
- `property_name = "Curie_temperature"` -> `material_info.section1.Curie_temperature`
- `property_name = "band_gap"` -> `material_info.section1.band_gap`
- `property_name = "specific_capacity"` -> `material_info.section1.specific_capacity`

## Quality Control

Current quality-control fields:

- `has_target_specific_property_evidence`
- `ambiguity_flags`
- `omission_reasons`

The old name `has_target_specific_superconducting_evidence` may still be read by compatibility utilities, but new outputs should use `has_target_specific_property_evidence`.

## Interpretation

The framework should be validated on superconducting and magnetic examples because those are the currently available real test cases, but the prompt and schema should remain usable for general materials databases.
