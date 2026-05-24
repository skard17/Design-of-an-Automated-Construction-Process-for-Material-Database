# Mini Example

Paper pattern:

- `A2Cr3As3 (A = K, Rb)`
- `Tc = 6.1 K` for `K2Cr3As3`
- `Tc = 4.8 K` for `Rb2Cr3As3`
- one family-level statement about quasi-1D character

Expected behavior:

- manifest has two targets
- candidate facts include two `Tc` entries and one family-level note
- matcher accepts each `Tc` only for the correct target
- aggregator keeps family-level ambiguity out of target-specific fields unless clearly attributed
