# WP7 — Hybrids, dispersion, PAW memo (owner S7)
Purpose: surface-chemistry-grade functionals. Physics: 10-physics/12,15. Phase B.

### T7.1 — D3(BJ)/D4 integration (E, F, stress)
- Observables: vs reference libraries <=1e-10 on a 10-dimer set. Effort 3 pw. Depends T6.3.

### T7.2 — ISDF interpolation points (randomized) + fit
- Observables: exchange-energy error vs rank curve for benzene; rank for 0.1 mHa recorded.
  Effort 5 pw. Depends T3.2.

### T7.3 — ACE construction + hybrid SCF
- Observables: PBE0 H2O and benzene vs PySCF <=0.1 mHa. Effort 6 pw. Depends T7.2, T6.1.

### T7.4 — Short-range HSE screening in tiles  [gate GB3 evidence]
- Observables: 500-atom TiO2 slab HSE06 <=4x own-PBE time/SCF at matched accuracy, on A40.
  Effort 5 pw. Depends T7.3, T1.3.

### T7.5 — Hybrid forces
- Observables: FD <=1e-4 Ha/Bohr (mixed) on 5 systems. Effort 4 pw. Depends T7.3, T6.3.

### T7.6 — PAW feasibility memo (input to M36 decision)
- Observables: written DOF/accuracy/effort analysis incl. PAW-FE-2026 evidence; council-reviewed.
  Effort 3 pw. Depends T2.4.
