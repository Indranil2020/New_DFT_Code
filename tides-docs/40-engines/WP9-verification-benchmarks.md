# WP9 — Verification & benchmarks, the red team (owner S9; release veto)
Purpose: nothing is true until this WP says so. Sources: 50-verification/*, 60-benchmarks/*.

### T9.1 — tolerances.yaml + runner framework
- Observables: every ladder rung executable by one command; budgets loaded from the single file.
  Effort 4 pw. Depends –. Unblocks all DoD checks.

### T9.2 — Reference data curation
- Observables: ACWF subset, S22, W4-11 subset, surface DB ingested with DOI/license/uncertainty per
  entry, plus charged/open-shell UKS molecules and charged-compensated cells/slabs with charge,
  N_alpha/N_beta, spin moment, diffuse-basis convergence, integrated spin density, and density-tail
  diagnostics (52-reference-data). Effort 4 pw. Depends –.

### T9.3 — Nightly mixed-vs-FP64 A/B harness
- Observables: <=0.5 meV/atom budget enforced on gauntlet-10; alarm + bisect pointer on breach.
  Effort 3 pw. Depends T1.7.

### T9.4 — Nightly FD force checks
- Observables: rung-4 green nightly; per-term isolation reports archived. Effort 3 pw. Depends T6.3.

### T9.5 — Competitor farm (containers + parsers)
- Problem: build and drive ABACUS, SIESTA, CP2K, SPARC, GPAW, GPU4PySCF, DFT-FE, DFTB+ reproducibly.
- Observables: each builds in a pinned container, runs 3 payloads, and its outputs parse into the
  dashboard schema. Effort 6 pw. Depends –. Unblocks all piecewise rows.

### T9.6 — Regression dashboard + energy metering
- Observables: sqlite + plots from JSON-lines logs; NVML kWh in every record; alarm on >10%
  regression. Effort 4 pw. Depends T8.5.

### T9.7 — Campaign runner + reproducibility archiver
- Observables: one command re-runs a published campaign from its archive (inputs+containers+commits);
  DOI minting scripted. Effort 4 pw. Depends T9.5, T9.6.
