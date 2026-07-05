# Precision policy (op-by-op; enforced by descriptors from WP1 T1.7)
| Operation | Store | Compute | Notes |
|---|---|---|---|
| H, S, P tiles | BF16/FP16 + tile scale | FP32 accum | escalate to FP32 store if gap monitor trips |
| SpGEMM chains (SP2/FOE/filter) | BF16/FP16 | FP32 accum | filter ledger logged |
| Grid rho/v maps | FP32 | FP32 | FP64 path exists for rungs 1–3 tests |
| Poisson (FFT/ISF) | FP32 | FP32 (twiddle FP64) | FP64 variant for verification |
| Traces, total E, mixer dots, Fermi search, level shifts | — | f64e (Ozaki) | MANDATORY |
| Eigensolver (R0/R1) | FP32 | FP32 + f64e RR refresh | dense FP64 fallback for validation |
| Forces accumulation | FP32 | f64e final reduce | FD ladder gates it |
Escalation: gap < gap_min OR A/B drift > 0.5 meV/atom => auto-promote stage precision, log event.
Determinism mode: ordered reductions, fixed seeds; bitwise reproducible; used for all debugging.
