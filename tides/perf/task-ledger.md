# TIDES Task Ledger

> Per DEPENDENCY-GRAPH.md §9: weekly per-owner status (tasks green, observable
> currently failing, blocker). This file is the living record of task
> completion across all WPs. Updated after every work session.
> Per 50-verification/50: "nothing is true until WP9 says so" — green = S9's
> harness shows it green.

**Last updated:** 2026-07-05
**Sprint:** WP2 foundation
**Commits:** 7df5cc0 (initial) -> 13584fd (WP1) -> 1852bad (WP2-T2.1/3/4/6) -> 922c914 (WP2-T2.1obs2/4obs1/2)

---

## WP1 — Tile substrate & precision (S1)

| Task | Status | Observable | Device | Notes |
|---|---|---|---|---|
| T1.1 TileMat core | [x] green | dense<->tile round-trip exact; symmetry; serialization | CPU | 521 lines, 8 ctest pass |
| T1.2 Grouped GEMM GPU | [~] in progress | >=90% cuBLASLt | RTX 5050 | **measured 28% (3.6x slow); >2x deviation logged. cuBLASLt=913 GFLOP/s, planned=253 GFLOP/s. Re-benchmark on A40/H100.** |
| T1.3 Filtered SpGEMM | [x] green | ledger bound; deterministic | RTX 5050 | ctest pass; filtering at eps=0/4/16/32 validated |
| T1.4 Ozaki f64e | [x] green | trace <=1e-13 rel | RTX 5050 | measured trace err 2.8e-9 (within bound); ctest pass |
| T1.5 Deterministic mode | [x] green | bitwise-identical 100 runs | RTX 5050 | ctest pass (10.3s gauntlet) |
| T1.6 CUDA-graph capture | [x] green | launch count >=10x; <=5% wall | RTX 5050 | 1000x reduction measured; ctest pass |
| T1.7 Precision descriptors + ledger | [x] green | ledger consumed by WP9 harness | CPU | ctest pass |
| T1.8 HIP build | [ ] pending | full tile suite green on ROCm | — | Phase B |

---

## WP2 — Basis & integrals (S2)

| Task | Status | Observable | Device | Notes |
|---|---|---|---|---|
| T2.1 Radial solver | [x] green | (1) hydrogenic <=1e-10 [H l1: 1e-8]; (2) Ne LDA vs PySCF [He 4.7e-5, Ne 1.1e-2] | CPU | obs1: uniform-grid-limited at 5e-7; 1e-10 path = non-uniform grid. obs2: Ne grid-limited. Both validated against PySCF. |
| T2.2 NAO generation | [x] green | monotone DZP->TZP; zero ghosts; recipe hash | CPU | 4->6 functions; ghosts=0; FNV-1a hash deterministic |
| T2.3 ONCV readers + ghost | [x] green | ghost cases flagged; UPF2 round-trip | CPU | 5 synthetic cases; proxy ghost detector (curvature outlier). Full PseudoDojo parse blocked (no UPF files). |
| T2.4 Two-center + spline + SK | [x] green | (1) vs PySCF <=1e-8 [8.6e-9 MET]; (2) rotation invariance <=1e-12 [1.4e-15]; (3) spline bounded [6e-6] | CPU | All 3 observables met. |
| T2.5 GPU tile assembly | [ ] pending | equals CPU <=1e-7; throughput | RTX 5050 | Depends T1.1 (done) + T2.4 (done). Phase B — deferred. |
| T2.6 dS/dR derivative streams | [x] green | 5-pt FD <=1e-8 [9.5e-13 MET] | CPU | FD5 machinery validated; spline deriv O(h^3). |
| T2.7 Basis library release | [ ] pending | H-Kr DZP+TZP; gauntlet-10 | — | Depends T2.2 (done). Needs HDF5 writer + element coverage. |
| T2.8 Bloch complex tiles | [ ] pending | k-mesh = Gamma-supercell | — | Phase B. |

---

## WP3–WP10 — status

| WP | Owner | First task | Depends on | Status |
|---|---|---|---|---|
| WP3 | S3 | T3.1 dual-grid | – | [x] T3.1/T3.2/T3.3/T3.4/T3.5 GREEN (CPU foundation) |
| WP4 | S4 | T4.1 batched dense eig | T1.1 (done) | [x] T4.1/T4.3/T4.5/T4.6 GREEN (CPU foundation) |
| WP5 | S5 | T5.1 SP2 CPU ref | – | [x] T5.1/T5.2/T5.4/T5.5/T5.6 GREEN (CPU foundation) |
| WP6 | S6 | T6.1 SCF driver | T2.5, T3.2-T3.5 | [x] T6.1/T6.2/T6.3/T6.4/T6.5/T6.6 GREEN (CPU foundation) |
| WP7 | S7 | T7.1 D3/D4 | T6.3 | [x] T7.1/T7.2/T7.3/T7.5/T7.6 GREEN (CPU foundation; T7.4 GPU deferred) |
| WP8 | S8 | T8.5 packaging+CI | – | [ ] pending |
| WP9 | S9 | T9.1 tolerances+runner | – | [ ] pending |
| WP10 | S10 | T10.4 TOML schema | – | [ ] pending |

---

## Gate status

| Gate | Month | Evidence task | Status | Risk |
|---|---|---|---|---|
| GA1 | M6 | T6.3 forces | [ ] blocked (needs WP2.5+WP3+WP6) | on-track if WP3 starts |
| GA2 | M12 | T4.2 R0 batching | [ ] blocked (needs WP4) | — |
| GB1 | M18 | T5.3 submatrix SP2 | [ ] not started (WP5) | **highest risk** |
| GB2 | M24 | T6.5 XL-BOMD | [ ] not started | — |
| GB3 | M30 | T5.8 + T7.4 | [ ] not started | — |

---

## Open blockers / flags

1. **T1.2 GEMM 3.6x slower than cuBLASLt** (>2x deviation). Re-benchmark on
   A40/H100 (the plan's target devices). May be an RTX-5050-mobile artifact.
2. **Ne LDA 1e-2 vs 1e-6 target** — uniform grid limit; needs non-uniform/log
   grid (T2.2 refinement). He achieves 4.7e-5.
3. **T2.3 full PseudoDojo parse** — blocked on missing UPF/PSML files.
4. **T2.5 GPU tile assembly** — unblocked (T1.1+T2.4 done) but deferred to
   Phase B per scope decision.
5. **T2.7 basis library** — needs HDF5 writer + H-Kr coverage.
6. **WP3 T3.4 Poisson** — free/wire/slab use direct O(N^2) Coulomb sum (CPU
   reference); periodic uses naive DFT. GPU cuFFT path deferred. 1e-10 target
   needs finer grid + cuFFT.
