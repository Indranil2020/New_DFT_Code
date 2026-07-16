# TIDES External-Code Benchmark Ledger — 2026-07-16

## Ledger status

- Recorded: `2026-07-16T07:21:44+02:00`
- Source commit: `6cd39046234e7f1c51abd6586715193b3e3b4b10`
- Commit subject: `feat(bench): unified benchmarking with per-step SCF profiling`
- Classification: **exploratory, not publication-grade, not an apples-to-apples accuracy benchmark**
- Workload: fixed-geometry single-point SCF only
- Geometry optimization: **not run**
- Lowest-energy structure search: **not run**

## Machine and software

| Item | Value |
|---|---|
| CPU | Intel Core i5-10400F, 6 cores / 12 threads, 2.90 GHz |
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB |
| NVIDIA driver | 575.51.03 |
| Python | 3.10.12 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| PySCF | 2.11.0 |
| gpu4pyscf | 1.5.0 |
| Quantum ESPRESSO | PWSCF 7.4, CPU installation |
| Explicit thread controls | None recorded; `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` were unset |

## Commands represented by the saved results

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmkl_def.so:/usr/lib/x86_64-linux-gnu/libmkl_sequential.so:/usr/lib/x86_64-linux-gnu/libmkl_core.so python3 tides/bench/unified_benchmark.py --max-atoms 8 --repeats 3 --basis def2-svp --output bench/profiling_results/unified_benchmark_3rep.json
```

```bash
python3 tides/bench/qe_benchmark.py --max-atoms 8 --repeats 1 --ecut 25 --output bench/profiling_results/qe_benchmark_1rep.json
```

Raw structured results:

- `unified_benchmark_3rep.json`
- `qe_benchmark_1rep.json`

## Protocol actually used

| Engine | XC | Core treatment | Basis | Repeats | Reported time |
|---|---|---|---|---:|---|
| TIDES NaoDriver | LDA-PW92 | PseudoDojo ONCV pseudopotential loaded by `PpLoader` | generated NAO/DZP-like basis | 3 | minimum end-to-end wall time |
| PySCF CPU | `LDA,VWN` | all-electron | def2-SVP GTO | 3 | minimum SCF wall time after molecule construction |
| gpu4pyscf | `LDA,VWN` | all-electron | def2-SVP GTO | 3 | minimum SCF wall time after molecule construction |
| QE | PBE | PSLibrary PP | plane waves, 25 Ry | 1 | end-to-end `pw.x` process time |

This protocol does not match XC, pseudopotential, basis convergence, setup boundaries, cold/warm treatment, or repeat count across all engines. Absolute-energy differences between TIDES, PySCF, and QE are therefore not accuracy errors and must not be used to rank correctness.

## Saved single-point results

Times are milliseconds. Slowdown is TIDES time divided by competitor time. PySCF iterations were measured. gpu4pyscf iterations are **unknown**, because the original harness read an attribute not provided by gpu4pyscf and stored zero; zero is not a valid iteration count.

| Molecule | Atoms | TIDES ms | TIDES iter | PySCF ms | PySCF iter | gpu4pyscf ms | gpu4 iter | T/PySCF | T/GPU4 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| H2 | 2 | 13,321.7 | 16 | 2,642.9 | 4 | 402.9 | unknown | 5.04× | 33.07× |
| H2O | 3 | 12,270.9 | 2 | 3,140.5 | 7 | 810.1 | unknown | 3.91× | 15.15× |
| CH4 | 5 | 39,220.3 | 27 | 6,366.7 | 8 | 897.0 | unknown | 6.16× | 43.72× |
| NH3 | 4 | 91,282.8 | 95 | 4,059.9 | 7 | 789.1 | unknown | 22.48× | 115.68× |
| C2H6 | 8 | 94,888.5 | 49 | 5,864.7 | 7 | 1,390.6 | unknown | 16.18× | 68.24× |
| C2H4 | 6 | 91,727.5 | 70 | 4,721.0 | 6 | 1,092.1 | unknown | 19.43× | 83.99× |

The worst observed saved slowdown was **115.68×**, not 95,000×. The value `94,888.5` was milliseconds for C2H6, not a slowdown factor.

## Saved energies

| Molecule | TIDES Ha | PySCF CPU Ha | gpu4pyscf Ha | QE PBE/PP Ha |
|---|---:|---:|---:|---:|
| H2 | -1.0822886461 | -1.1315907756 | -1.1315907756 | -1.1613391550 |
| H2O | -1.9658054153 | -75.7951904935 | -75.7951904936 | -17.0936553700 |
| CH4 | -7.8234682101 | -39.9961354396 | -39.9961354397 | -8.5042514050 |
| NH3 | -11.1797207989 | -56.0004008016 | -56.0004008017 | -11.9421062200 |
| C2H6 | -12.6288807427 | -78.7233019560 | -78.7233019563 | failed |
| C2H4 | -12.8241517079 | -77.7636148134 | -77.7636148135 | failed |

## TIDES SCF-loop profile

The original label `build_H` covers the complete density-to-Hamiltonian callback, not one operation. It includes density construction, Poisson, Hartree projection, XC, XC projection, transfers/synchronizations, and Hamiltonian assembly.

| Molecule | Basis | build_H ms/iter | eigensolve ms/iter | SCF-loop total ms | Approx. build_H share |
|---|---:|---:|---:|---:|---:|
| H2 | 16 | 615.239 | 6.968 | 9,957.4 | 98.9% |
| H2O | 24 | 736.165 | 1.498 | 1,475.5 | 99.8% |
| CH4 | 40 | 1,050.985 | 3.278 | 28,475.3 | 99.7% |
| NH3 | 32 | 864.479 | 2.415 | 82,376.2 | 99.7% |
| C2H6 | 64 | 1,542.934 | 5.780 | 75,943.4 | 99.6% |
| C2H4 | 48 | 1,118.848 | 3.201 | 78,581.3 | 99.7% |

## Known validity defects requiring correction

1. TIDES used PseudoDojo ONCV PP while PySCF/gpu4pyscf were all-electron.
2. TIDES used LDA-PW92, PySCF used VWN LDA, and QE used PBE.
3. Basis/grid/cutoff convergence was not demonstrated to a shared energy or force tolerance.
4. TIDES timing includes basis generation and grid setup; PySCF/gpu4pyscf timing begins after molecule construction.
5. Cold and warm runs were not explicitly separated despite taking three repeats.
6. CPU thread counts and affinity were not fixed.
7. gpu4pyscf SCF iteration count was not captured.
8. QE used one repeat and failed for C2H4/C2H6; these are capability results, not wins.
9. No geometry optimization or force-convergence comparison was run.
10. The TIDES `build_H` profile needs synchronized substep timings before optimization claims.

## Corrected benchmark policy for the next run

1. Use PBE for all engines.
2. Use pseudopotentials for all engines and publish the exact PP family/file/hash and valence partition. Exact shared UPF physics is possible for TIDES versus QE. PySCF/gpu4pyscf require a supported ECP/GTH route; if the identical PP cannot be represented, report that comparison as matched valence/PBE but not identical PP.
3. Use identical geometries, charge, spin, boundary conditions, SCF energy/density thresholds, and force threshold.
4. Establish basis/grid/cutoff convergence independently for each engine against its own converged reference before timing.
5. Separate setup, cold SCF, and warm SCF timings; fix CPU threads and GPU clocks/state where possible.
6. Record actual SCF cycles for every engine with callbacks.
7. Treat single-point and geometry-optimization campaigns as separate tables.
8. Archive command lines, stdout/stderr, JSON, git commit, environment, PP hashes, and convergence traces.

## Optimization gate

No speedup claim is accepted unless:

- the dominant `build_H` suboperation is identified with synchronized CPU/GPU timing;
- energy, density, and symmetry checks pass against the pre-optimization result;
- three cold and three warm repeats are archived;
- before/after timings use the same executable, inputs, thread count, and hardware state.
