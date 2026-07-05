# Risk register (live; reviewed at every gate)
| # | Risk | LxI | Mitigation / trigger |
|---|---|---|---|
| 1 | Submatrix purification accuracy insufficient for full-DFT H | MxH | error-compensated truncation (20-math/22); gate GB1; fallback OMM/FOE |
| 2 | Mixed-precision instability near degeneracies | MxH | f64e refinement sweeps; gap monitor escalates precision; nightly A/B budget |
| 3 | NAO basis ceiling (BSSE, anions, polarizabilities) | MxM | TZP+diffuse; counterpoise tooling; PAW decision M36 |
| 4 | XL-BOMD fragility at vanishing gaps | LxM | KSA kernel + fractional occupations; per-step SCF fallback |
| 5 | k-point complex-tile cost | MxM | Gamma-supercell for large N; complex tiles only R0/R1 |
| 6 | Competitor movement (ABACUS-GPU, xQC, DFT-FE/PAW-FE, SPARC) | HxM | orthogonal differentiators; quarterly re-aim of bars |
| 7 | Hardware shift (INT8 phase-out; FP8-centric) | HxL | Ozaki layer abstracts slice type; FP16 and FP8 variants implemented |
| 8 | Key-person loss | MxH | contract tests; docs-with-derivations; cross-WP weeks |
| 9 | Scope creep | HxM | non-goals list; RFC + explicit descope required |
| 10 | Cluster access slips past M31 | MxM | Phase B validates all algorithms <=3x10^4 atoms on owned cards; only scaling waits |
