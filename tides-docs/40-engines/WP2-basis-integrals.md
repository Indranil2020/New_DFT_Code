# WP2 — Basis & integrals (owner S2)
Purpose: NAO generation; S, T, V_nl assembly as tiles; derivative streams. Consumes TileMat.
Provides S,H0,dS/dR,dH0/dR to WP3/WP6. Physics: 10-physics/10,11. Phase A core; T2.8 Phase B.

### T2.1 — Radial confined-atom solver (log grid, FP64, CPU)
- Problem: solve the confined atomic KS problem per element to generate NAO radial functions.
- Start: core/basis/atomgen/; read 10-physics/10 refs.
- Observables: (1) hydrogenic eigenvalues <=1e-10 Ha; (2) Ne LDA total energy vs NIST atomic
  reference <=1e-6 Ha. Effort 5 pw. Depends –. Unblocks T2.2, T2.4.

### T2.2 — NAO generation & optimization (DZP/TZP recipes; file format)
- Observables: (1) monotone DZP->TZP atomization convergence on the 10-molecule set; (2) zero ghost
  states via T2.3 detector; (3) HDF5 basis format with recipe hash, documented. Effort 6 pw. Depends T2.1, T2.3.

### T2.3 — ONCV readers (UPF2/PSML) + validators + ghost detector
- Observables: (1) full PseudoDojo standard table parses; (2) known-ghost test cases flagged;
  (3) checksums match published hashes. Effort 4 pw. Depends –.

### T2.4 — Two-center tables (S, T, V_nl in KB form) + splines + rotations
- Problem: tabulate pair integrals vs distance on splines; assemble via Slater–Koster-style rotations.
- Observables: (1) vs PySCF matched-basis elements <=1e-8 Ha; (2) rotation invariance <=1e-12;
  (3) spline error mapped vs r and bounded in tolerances.yaml. Effort 6 pw. Depends T2.1, T2.3.

### T2.5 — GPU tile assembly of S, H0
- Observables: (1) equals CPU path <=1e-7 rel; (2) throughput (atom-pairs/s) recorded on RTX and
  entered in the model ledger. Effort 5 pw. Depends T1.1, T2.4. Unblocks T3.2, T6.1.

### T2.6 — dS/dR, dH0/dR derivative streams
- Observables: 5-point FD on random displacements <=1e-8 (FP64 path). Effort 4 pw. Depends T2.4. Unblocks T6.3.

### T2.7 — Basis library release (H–Kr DZP + selected TZP) with docs
- Observables: library published with per-element convergence notes; gauntlet-10 runs off it only.
  Effort 4 pw. Depends T2.2.

### T2.8 — Bloch-phase (complex) tiles for periodic R0/R1 (Phase B)
- Observables: k-mesh result equals Gamma-supercell reference <=1e-8 Ha/atom on Si. Effort 4 pw. Depends T2.5.
