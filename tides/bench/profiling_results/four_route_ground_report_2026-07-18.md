# Four-route ground report: TIDES vs PySCF (AE / PP, CPU / GPU)

**Date:** 2026-07-18
**Hardware:** NVIDIA RTX 3060 12 GB, 8-core CPU, MKL-preloaded
**Functional:** PBE
**PySCF version:** 2.11.0, gpu4pyscf 1.5.0
**TIDES commit:** current working tree
**Geometry:** RDKit MMFF-optimized (CH4, H2O)
**TIDES grid:** grid_h=0.3, grid_margin=6.0
**PySCF basis:** def2-SVP; AE = no ECP, PP = ccECP
**TIDES CPU path:** `TIDES_DISABLE_GPU=1` (OpenMP multi-core, no MPI distribution yet)
**TIDES GPU path:** CUDA device pipeline with GPU S/T enabled (`TIDES_USE_GPU_ST=1` default)

## Executive summary

- **TIDES PP GPU** is the only route that is both **converged and physics-plausible** on both molecules. It is ~5–15× slower than the equivalent PySCF GPU PP run, with setup (S/T, V_ext, V_nl) dominating the wall time.
- **TIDES PP CPU** is broken on CH4 and H2O: CH4 energy is ~4.6 Ha too negative vs the GPU route, and H2O fails to converge in 50 iterations. The CPU PP `BuildAnalyticVextPP` / KB non-local projector path must be fixed before any PP CPU optimization.
- **TIDES AE CPU/GPU** are internally consistent (CPU vs GPU energies agree to <0.3 Ha) but CH4 is ~7.5 Ha too negative compared with PySCF AE. H2O is within ~0.5 Ha. This points to an all-electron C-atom / V_ext regularization issue, not a random GPU/CPU divergence.
- **PySCF** CPU and GPU routes are robust and fast; the GPU PP route is 0.3–0.4 s for these systems.
- **MPI** in the current `NaoDriver` is not actually distributed; the CPU path is OpenMP only. True MPI route data will require wiring `MpiWorld` into `NaoDriver`.

## Raw numbers

### CH4 (5 atoms, n_basis=40, 8 valence e in PP, 10 e in AE)

| Route | E_total (Ha) | Converged | Iters | Wall (s) | build_H/iter (ms) | xc_eval/iter (ms) | poisson/iter (ms) | Notes |
|-------|--------------|-----------|-------|----------|-------------------|-------------------|-------------------|-------|
| TIDES AE CPU | -47.758 | Yes | 15 | 53.7 | 2991 | 2009 | 527 | GPU pipeline: no |
| TIDES AE GPU | -47.916 | Yes | 12 | 12.06 | 114 | 99.4 | 8.7 | GPU pipeline: yes |
| TIDES PP CPU | -12.925 | Yes | 16 | 54.1 | 2991 | 2009 | 527 | **Energy non-physical vs GPU/PySCF** |
| TIDES PP GPU | -8.342 | Yes | 13 | 6.29 | 109 | 102.2 | 4.5 | Closest to PySCF PP |
| PySCF AE CPU | -40.4146 | Yes | — | 7.16 | — | — | — | def2-SVP |
| PySCF AE GPU | -40.4146 | Yes | — | 1.39 | — | — | — | def2-SVP |
| PySCF PP CPU | -7.9459 | Yes | — | 6.98 | — | — | — | def2-SVP + ccECP |
| PySCF PP GPU | -7.9459 | Yes | — | 0.40 | — | — | — | def2-SVP + ccECP |

### H2O (3 atoms, n_basis=24, 8 valence e in PP, 10 e in AE)

| Route | E_total (Ha) | Converged | Iters | Wall (s) | build_H/iter (ms) | xc_eval/iter (ms) | poisson/iter (ms) | Notes |
|-------|--------------|-----------|-------|----------|-------------------|-------------------|-------------------|-------|
| TIDES AE CPU | -75.757 | Yes | 17 | 32.9 | 1406 | 914 | 300 | GPU pipeline: no |
| TIDES AE GPU | -76.013 | Yes | 13 | 9.84 | 44 | 36 | 5 | GPU pipeline: yes |
| TIDES PP CPU | -13.438 | **No** | 50 | 73.3 | 1393 | 902 | 302 | **Diverged / non-physical** |
| TIDES PP GPU | -17.947 | Yes | 13 | 3.72 | 39 | 36 | 3.2 | Closest to PySCF PP |
| PySCF AE CPU | -76.2725 | Yes | — | 7.99 | — | — | — | def2-SVP |
| PySCF AE GPU | -76.2725 | Yes | — | 0.67 | — | — | — | def2-SVP |
| PySCF PP CPU | -16.9454 | Yes | — | 7.32 | — | — | — | def2-SVP + ccECP |
| PySCF PP GPU | -16.9454 | Yes | — | 0.34 | — | — | — | def2-SVP + ccECP |

## Bottleneck analysis

### Common setup phases (TIDES)

For all four TIDES routes the dominant fixed costs in a cold run are:

1. **NAO basis generation** — 1.7–2.9 s (cold, CPU only, cached after first call).
2. **S/T assembly** — 2.0–5.1 s. The cuBLAS GPU dgemm is fast, but CPU gradient computation + `memcpy` flattening is still a large fraction.
3. **V_ext assembly** — 0.7–2.0 s (GPU path) to 2.0–6.0 s (CPU path).
4. **V_nl (KB) assembly** (PP only) — 0.5–1.3 s on CPU for all routes (not GPU-accelerated).

After setup, the per-SCF-iteration cost breaks down as follows.

### Route 1: TIDES AE CPU

- **build_H per iter:** ~1.4 s (H2O) to ~3.0 s (CH4).
- **Dominant sub-step:** `xc_eval` 900–2000 ms, then `poisson` 300–500 ms.
- **GPU not used:** all grids/integrals run on CPU.
- **Physics block:** CH4 energy is ~7.5 Ha too negative vs PySCF. H2O is close. This is the highest-priority accuracy bug for the AE route.

### Route 2: TIDES AE GPU

- **build_H per iter:** ~40–110 ms, dominated by `xc_eval` (~35–100 ms) and `poisson` (~5–9 ms).
- **Setup bottleneck:** S/T assembly still costs ~2–4 s, and V_ext/V_nl are on CPU.
- **GPU speedup vs CPU:** ~25–30× on build_H per iteration.
- **Physics block:** same CH4 energy error as AE CPU.

### Route 3: TIDES PP CPU

- **build_H per iter:** ~1.4–3.0 s, same CPU XC/Poisson bottleneck as AE CPU.
- **Extra setup bottleneck:** `V_nl (KB)` ~0.6–1.3 s, `V_ext` analytic PP ~2.0–6.0 s.
- **Physics block:** CH4 energy is ~4.6 Ha too negative vs GPU/PySCF; H2O does not converge. `BuildAnalyticVextPP` semi-on-site correction and/or KB projector assembly is the likely root cause.

### Route 4: TIDES PP GPU

- **build_H per iter:** ~40–110 ms; `xc_eval` is the largest remaining GPU step (~35–100 ms), with `poisson` <5 ms.
- **Setup bottleneck:** S/T assembly (2–4 s), V_ext assembly (0.5–0.7 s), V_nl assembly (0.5 s).
- **Convergence:** stable for both CH4 (13 iters) and H2O (13 iters).
- **Physics:** energies within ~0.4–1.0 Ha of PySCF PP, consistent with different pseudopotentials / basis sets.
- **This is the only route ready for throughput optimization.**

### PySCF reference routes

- **PySCF AE CPU:** ~7–8 s, limited by CPU 2-electron integrals and grid XC.
- **PySCF AE GPU:** ~0.7–1.4 s, dominated by grid XC on GPU.
- **PySCF PP CPU:** ~7 s; ECP integrals are CPU only.
- **PySCF PP GPU:** ~0.3–0.4 s, fastest route overall.

## Relative performance vs PySCF

| System | TIDES vs PySCF (GPU PP) | TIDES vs PySCF (CPU PP) | TIDES vs PySCF (GPU AE) | TIDES vs PySCF (CPU AE) |
|--------|-------------------------|--------------------------|--------------------------|--------------------------|
| CH4 | 6.3 / 0.40 = **15.8× slower** | 54.1 / 6.98 = **7.7× slower** | 12.1 / 1.39 = **8.7× slower** | 53.7 / 7.16 = **7.5× slower** |
| H2O | 3.7 / 0.34 = **10.9× slower** | 73.3 / 7.32 = **10.0× slower** | 9.84 / 0.67 = **14.7× slower** | 32.9 / 7.99 = **4.1× slower** |

Numbers for PP CPU are unreliable because the route is broken; they are shown only to highlight the need for a fix.

## Key route-specific findings

1. **TIDES PP GPU is the production-quality route today.**
   - Correct convergence and plausible energies.
   - Main remaining work: reduce setup time (S/T, V_ext, V_nl) and per-iter `xc_eval`.

2. **TIDES PP CPU is unusable for benchmarking until fixed.**
   - `BuildAnalyticVextPP` + KB assembly path gives wrong energies and/or divergence.
   - Do **not** optimize CPU PP code paths until the analytic correction is validated against the GPU grid path or PySCF.

3. **TIDES AE has a C-atom / all-electron V_ext issue.**
   - H2O is close to PySCF; CH4 is ~7.5 Ha too negative.
   - Suggests the all-electron on-site / erf smoothing regularization is not handling carbon core states correctly.
   - Also blocks AE CPU/GPU fair comparison.

4. **MPI CPU route is not actually MPI-distributed.**
   - `TIDES_ENABLE_MPI` is on, but `NaoDriver::Run` does not call `MpiWorld` partition / halo exchange.
   - The "MPI CPU" route is effectively the OpenMP CPU path. True MPI scaling requires integrating the parallel layer into `NaoDriver`.

## Recommended next steps (route by route)

1. **Fix TIDES PP CPU accuracy** (`BuildAnalyticVextPP` semi-on-site + KB projector). Make CPU PP reproduce GPU PP/PySCF PP energies.
2. **Fix TIDES AE all-electron accuracy** for carbon (or generally for systems with core electrons). Re-benchmark AE routes once fixed.
3. **Optimize TIDES PP GPU** once accuracy is locked: the lever is setup time (S/T, V_nl) and `xc_eval` per iteration.
4. **Design true MPI distribution** separately; it is not a quick win on the current `NaoDriver` path.
