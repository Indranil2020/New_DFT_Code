# WP4 — Mid-range solvers R0/R1 (owner S4)
Purpose: batched dense eigensolves (R0) and ChFSI (R1) + dense bridges + the broker.
Math: 20-math/24. Phase A.

### T4.1 — Batched dense eigensolver path (R0)
- Observables: (1) residuals <=1e-9 at n<=400; (2) eigensolves/s at n=200 recorded per device and
  >=80% of raw cuSOLVER syevjBatched. Effort 4 pw. Depends T1.1.

### T4.2 — R0 batching driver (structure-of-arrays over molecules; one graph per sweep)
- Problem: run 10^3–10^4 small molecules concurrently — the small-end throughput headline.
- Observables: (1) >=5x10^3 DZP-PBE single-points/hour on RTX for the 30-atom set (record H100);
  (2) results identical to serial path <=1e-8 Ha. Effort 6 pw. Depends T4.1, T6.1. Unblocks GA2.

### T4.3 — ChFSI core (filter, Rayleigh–Ritz, locking, subspace reuse)
- Observables: (1) residuals <=1e-9; (2) SCF iterations within +2 of dense on gauntlet;
  (3) subspace reuse cuts filter applications >=2x along an MD trajectory. Effort 6 pw. Depends T1.2.

### T4.4 — ELPA / cuSOLVERMp bridge (single node)
- Observables: parity harness on 10k–40k matrices; used as validation oracle. Effort 3 pw. Depends –.

### T4.5 — OMM direct minimization (optional insulator alternative)
- Observables: E vs diagonalization <=1e-8 Ha; converges on stretched H2O where naive mixing fails.
  Effort 5 pw. Depends T1.3.

### T4.6 — Broker + `tides tune`
- Observables: per-device crossover table emitted; broker choice within 10% of best regime on the
  calibration set. Effort 4 pw. Depends T4.2, T4.3. Unblocks all-regime UX.
