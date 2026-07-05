# Solver broker (regime dispatch)
Inputs: N_atoms, BCs, gap estimate (cheap: few Lanczos steps on H from a superposition density),
T_e, available VRAM, user override.
Routes: R0 batch-dense (<=~200 atoms, many systems) | R1 ChFSI/dense (<=~2k atoms) |
R2 SP2-submatrix (gapped, large) | R3 FOE/SQ (metallic/finite-Te, large).
`tides tune` measures crossovers on the actual machine at install and caches a per-device table;
"fastest across the spectrum" is thereby enforced empirically per machine, not assumed.
Fallback chain on failure: R2->R3 (raise Te) -> R1 (if memory allows) with logged reason.
Broker decisions are logged in the run record for reproducibility.
