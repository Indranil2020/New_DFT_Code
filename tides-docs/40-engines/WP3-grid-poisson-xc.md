# WP3 — Grids, Poisson, XC (owner S3; QTT with S5)
Purpose: rho(P)->grid, v(grid)->H tiles, Hartree under all BCs, XC. Physics: 10-physics/12,13.
Model: memory-bound; ~0.5 MB traffic/atom/build => grid phase never dominates above ~10^3 atoms.

### T3.1 — Dual-grid layout + decomposition structs
- Observables: index-map unit tests; halo spec documented. Effort 3 pw. Depends –.

### T3.2 — rho builder (P -> n(r))
- Problem: the #1 memory-bound kernel; tile-batched orbital products onto the fine grid.
- Observables: (1) vs CPU <=1e-9; (2) integral(n)=N_e <=1e-10; (3) >=60% HBM roofline on RTX.
  Effort 6 pw. Depends T1.2, T2.5, T3.1. Unblocks T3.3, T6.1, T7.2.

### T3.3 — v -> H adjoint map
- Observables: adjointness |<Av,w>-<v,A+w>| <=1e-12 (FP64 path) on 100 random pairs.
  Effort 4 pw. Depends T3.2.

### T3.4 — Poisson: FFT (periodic) + ISF (free/wire/slab)
- Observables: (1) Gaussian-charge analytic <=1e-10 Ha under all four BCs; (2) timing table
  128^3–512^3 per device recorded. Effort 6 pw. Depends T3.1. Unblocks T6.1.

### T3.5 — libxc integration (LDA/GGA, collinear spin)
- Observables: He/Ne totals vs PySCF <=1e-8 Ha; spin-polarized O atom correct ground state.
  Effort 3 pw. Depends T3.2.

### T3.6 — Grid force + stress terms
- Observables: terms pass WP6 FD isolation harness (rung 4). Effort 4 pw. Depends T3.3, T3.5.

### T3.7 — QTT-rho prototype (research flag)
- Observables: bond-dimension vs error curve on 64-H2O density; memory ratio vs FP32 grid reported.
  Effort 5 pw. Depends T3.2.

### T3.8 — QTT-Poisson prototype (research flag)
- Observables: accuracy/speed vs T3.4 on identical payloads; feeds gate R-1 (M30).
  Effort 5 pw. Depends T3.7.

### T3.9 — ESP/prolate Ewald backend (periodic long-range candidate)
- Problem: test the prolate-spheroidal Ewald/ESP idea as an optional accelerator for periodic
  long-range electrostatics when FFT communication or reciprocal-grid size dominates.
- Start: read `10-physics/13-electrostatics-boundary-conditions.md` and
  `10-physics/s41467-026-73232-8_reference.pdf`.
- Observables: (1) energy, forces, stress, and neutralization match T3.4/T6.2 within rung-2/4
  tolerances on neutral and charged-compensated periodic cells; (2) timing and communication table
  vs FFT/Ewald is recorded; (3) backend stays off by default unless it wins at equal accuracy.
  Effort 4 pw. Depends T3.4, T6.2.
