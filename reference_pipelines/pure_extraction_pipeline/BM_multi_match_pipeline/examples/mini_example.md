# Mini Example

These examples are format and attribution checks for the general materials pipeline. They are not a closed domain list.

## Pattern 1: ordered property values in a composition family

Paper pattern:

- `A2Cr3As3 (A = K, Rb)` is studied as two target materials
- `Tc = 6.1 K` for `K2Cr3As3`
- `Tc = 4.8 K` for `Rb2Cr3As3`
- one family-level statement about quasi-1D character

Expected behavior:

- manifest has two targets
- candidate facts include two `property_value` entries with `property_name = "Tc"` and one family-level note
- matcher accepts each `Tc` only for the correct target
- aggregator writes the accepted values under `material_info.section1.Tc`
- aggregator keeps family-level ambiguity out of target-specific fields unless clearly attributed

## Pattern 2: magnetic validation-style values

Paper pattern:

- `La0.7Sr0.3MnO3` and `La0.7Ca0.3MnO3` are compared as two target materials
- `Curie temperatures are 365 K and 250 K, respectively`
- a shared paragraph discusses double-exchange as a family-level interpretation

Expected behavior:

- manifest has two composition targets
- candidate facts include two `property_value` entries with `property_name = "Curie_temperature"` and `property_category = "magnetic"`
- matcher maps the two values by the explicit `respectively` order
- aggregator writes target-specific values under `material_info.section1.Curie_temperature`
- the family-level interpretation is retained only as ambiguous/context unless the paper explicitly attributes it to one target

## Pattern 3: non-validation-domain property values

Paper pattern:

- `LiFePO4` and `NaFePO4` are compared as two target materials
- specific capacity, synthesis temperature, and phase-identification results are reported separately

Expected behavior:

- manifest has two targets
- capacity values become `property_value` candidates with `property_name = "specific_capacity"`
- synthesis temperatures become `processing_condition` or `synthesis_method` candidates
- phase-identification results become `phase` or `characterization_result` candidates
- matcher keeps each fact with the locally supported target rather than assuming all facts apply to both materials
