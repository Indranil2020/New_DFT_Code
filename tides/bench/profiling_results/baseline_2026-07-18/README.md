# Day-0 baseline — 2026-07-18 (for ROADMAP-2026-07-18-correctness-to-1000x)

All phase exit criteria in `tides-docs/ROADMAP-2026-07-18-correctness-to-1000x.md` are measured against the numbers in this directory.

## How this was produced

```
cd tides/bench
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmkl_core.so:/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so:/usr/lib/x86_64-linux-gnu/libiomp5.so \
  python3 profiling_results/baseline_2026-07-18/run_baseline.py
```

- Wrapper `run_baseline.py` imports `benchmark_fair_comparison.py` unmodified and truncates the ladder to CH4→C10H22 (32 atoms, 8 rungs). **Why truncated:** the full-ladder attempt on the morning of 2026-07-18 (`tides/bench/benchmark_fair_output.log`) stalled inside C14H30 (44 atoms) and never reached the summary/JSON dump; the stall itself is a baseline fact (44+ atoms is past the practical ceiling until roadmap Phase 3).
- The `LD_PRELOAD` is mandatory on this machine (`audit_ledger_gga_screen_2026-07-17.md:155`); without it numpy/MKL aborts at import.
- Protocol (from `benchmark_fair_comparison.py`, unchanged): B3LYP, grid_h=0.5, margin 6.0, tol 1e-7, TIDES NAO-DZP (AE) / NAO-DZP+PseudoDojo (PP) vs PySCF def2-svp (AE) / gth-dzvp+gth-pbe (PP); TIDES cold+warm; `TIDES_LEVEL_SHIFT` 0.2 baked in for >4 atoms (0.3 for >20), PP max_iter 250 for ≥12 atoms.
- Hardware: RTX 3060 12 GB, 8-core CPU. PySCF 2.11.0, gpu4pyscf 1.5.0.

## Files

- `fair_baseline_output.log` — full run log (also the summary tables at the end).
- `fair_benchmark_baseline_2026-07-18.json` — per-molecule structured results (copy of the script's hard-coded output `../fair_benchmark_2026-07-17.json` from THIS run; the file name embeds the script's old date, the content is 2026-07-18).
- `run_baseline.py` — the wrapper.

## Headline day-0 numbers

Warm wall time, TIDES vs gpu4pyscf (ratio >1 = TIDES faster):

| Mol | N | TIDES-AE | GPU-AE | AE ratio | TIDES-PP | GPU-PP | PP ratio |
|---|---|---|---|---|---|---|---|
| CH4 | 5 | 0.97 s | 1.74 s | 1.79× | 1.07 s | 2.25 s | 2.09× |
| H2O | 3 | 0.39 s | 0.84 s | 2.13× | 0.49 s | 0.81 s | 1.65× |
| NH3 | 4 | 1.95 s | 1.30 s | 0.67× | 0.99 s | 1.20 s | 1.21× |
| C2H6 | 8 | 4.80 s | 3.40 s | 0.71× | 1.93 s | 2.97 s | 1.54× |
| C6H6 | 12 | 9.93 s | 8.43 s | 0.85× | 29.15 s | 8.22 s | 0.28× |
| C4H10 | 14 | 12.20 s | 9.05 s | 0.74× | 21.42 s | 9.56 s | 0.45× |
| C8H18 | 26 | 92.52 s | 37.66 s | 0.41× | 124.04 s | 39.67 s | 0.32× |
| C10H22 | 32 | 133.80 s | 57.19 s | 0.43× | 225.48 s | 60.69 s | 0.27× |

SCF iterations (T = TIDES warm, G = gpu4pyscf; **bold = hit max_iter, not converged**):

| Mol | AE T/G | PP T/G |
|---|---|---|
| CH4 | 16 / 5 | 17 / 6 |
| H2O | 13 / 6 | 13 / 6 |
| NH3 | **100** / 6 | 31 / 6 |
| C2H6 | **100** / 6 | 26 / 6 |
| C6H6 | 70 / 6 | **250** / 6 |
| C4H10 | 60 / 7 | 128 / 8 |
| C8H18 | **120** / 7 | 182 / 8 |
| C10H22 | **120** / 7 | **240** / 8 |

Total-energy deltas, TIDES − gpu4pyscf (warm; sub-Ha differences are expected from basis/PP mismatch — multi-Ha and positive totals are broken physics):

| Mol | AE Δ (Ha) | PP Δ (Ha) |
|---|---|---|
| CH4 | −11.04 | −0.64 |
| H2O | −5.46 | −3.19 |
| NH3 | **+45.79** | −1.18 |
| C2H6 | +17.99 | −0.50 |
| C6H6 | +26.30 | +17.45 |
| C4H10 | +15.58 | +11.01 |
| C8H18 | −26.71 | **+127.10** (E_total = +70.9 Ha!) |
| C10H22 | −13.47 | **+210.72** (E_total = +140.7 Ha!) |

## What this baseline establishes (feeds the roadmap)

1. **Per-iteration GPU machinery is already competitive.** build_H at C10H22 is 820 ms/iter vs gpu4pyscf's ~7.6 s/cycle; at ≤5 atoms TIDES warm even beats gpu4pyscf on wall time. The rho/poisson/vmat substeps are ≤0.2 ms/iter — earlier optimizations (GEMM rho, cached cuFFT, batched screened vmat) worked.
2. **Iteration count / non-convergence is the dominant failure at every size** (10–30× gpu4pyscf, 6 of 16 TIDES routes hit max_iter even with the baked-in level shift). Roadmap Phase 1 (real SAD + DIIS quality + automatic stabilizers) is the single biggest lever, and Phase 0 is a precondition (part of the non-convergence is broken physics, not weak mixing).
3. **AE energy errors are large and of BOTH signs** (−26.7 … +45.8 Ha), varying with geometry — the signature of the non-convergent bare −Z/r grid quadrature diagnosed in roadmap §2.2, not a systematic shift.
4. **xc_eval is now ~100% of build_H** (e.g. 820.1 of 820.7 ms at C10H22) — the per-iteration optimization target for Phase 2. (Instrumentation caveat: vmat shows 0.0 ms — vmat GEMM time appears to be attributed to the xc bucket in the fused GGA path; re-check timers before optimizing.)
5. **GPU pipeline held to 32 atoms at grid_h=0.5** (no CPU fallback on any rung). The >12-atom VRAM collapse (BUG-6) bites at finer grids (the four-route protocol uses h=0.3, ~3.4× more points) — the Phase 3 memory work is still mandatory for production grids.
6. **44+ atoms is the practical ceiling**: the same-day full-ladder attempt stalled at C14H30 (V_nl assembly alone >100 s, SCF grinding at n=496 with no convergence in sight).

## Relation to the four-route ground report (same day)

`four_route_ground_report_2026-07-18.md` uses a different protocol (PBE, grid_h=0.3, CH4/H2O only, CPU routes via TIDES_DISABLE_GPU=1). The two datasets are complementary and consistent in conclusions: PP CPU and AE routes physics-broken; PP GPU healthiest at small N; TIDES iteration counts well above PySCF. Wall times are NOT directly comparable between the two protocols (different functional and ~3.4× grid-point count).
