# WP6 — SCF, XL-BOMD, forces, dynamics (owner S6)
Purpose: the production loop. Physics: 10-physics/14,16; math: 20-math/25. Phase A core; T6.5+ Phase B.

### T6.1 — SCF driver + mixers (Pulay/Kerker/Broyden)
- Observables: converges gauntlet-10 within a documented iteration budget; charged/open-shell UKS
  cases preserve charge, spin multiplicity, spin moment, and integrated spin density and match
  PySCF/ABACUS references within rung-3 budgets; charged slabs/cells exercise the T6.2 neutralization
  path; restart-safe via HDF5.
  Effort 5 pw. Depends T2.5, T3.2–T3.5. Unblocks T4.2, T6.2.

### T6.2 — Total energy assembly + Ewald/neutralization
- Observables: component-wise match vs PySCF (molecules) and ABACUS (bulk Si) <=1e-6 Ha/atom;
  all sums via f64e. Effort 4 pw. Depends T6.1, T1.4.

### T6.3 — Analytic forces (HF + Pulay + grid + dispersion)  [gate GA1 evidence]
- Problem: one P-based force path valid for R0–R3 and fractional occupations.
- Observables: (1) 5-point FD <=1e-6 Ha/Bohr (FP64 path) and <=1e-4 (production mixed) on gauntlet;
  (2) per-term FD isolation harness green nightly. Effort 6 pw. Depends T2.6, T3.6, T6.2.

### T6.4 — Stress tensor
- Observables: FD vs cell strain <=1e-6 Ha (FP64 path) on Si and NaCl. Effort 4 pw. Depends T6.3, T2.8.

### T6.5 — XL-BOMD integrator (KSA kernel, thermostats)  [gate GB2 evidence]
- Observables: (1) NVE 64-H2O 0.5 fs 100 ps drift <=30 uHa/atom/ps at ~1 solve/step on RTX;
  (2) RDF vs converged SCF-MD, KS-test p>0.05; (3) per-step SCF fallback switch works. Effort 6 pw.
  Depends T6.3 and (T4.3 or T5.3).

### T6.6 — ASPC warm starts + optimizers (FIRE, L-BFGS)
- Observables: relaxation solve count reduced >=3x vs cold on a 20-structure set. Effort 4 pw. Depends T6.3.

### T6.7 — NEB (climbing image)
- Observables: barriers vs an established open code <=0.02 eV on 3 reactions. Effort 4 pw. Depends T6.6.

### T6.8 — MD throughput record vs anchors
- Observables: steps/s vs DFTB+ on 64/512-H2O per device, published with accuracy line; target
  within 5–20x. Effort 3 pw. Depends T6.5.
