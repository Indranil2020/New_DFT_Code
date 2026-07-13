# TIDES Clinical Audit — Final Verdict

**Date**: 2026-07-12
**Auditor**: Independent code-level inspection (not ledger self-reporting)
**Method**: Direct source code inspection of 25+ key files, cross-referenced against proposal §1–14, PROJECT-LEDGER, gap analysis, and prior AUDIT-REPORT.

---

## Executive Verdict

The codebase is **neither a toy model nor a real model**. It is a **well-engineered reference implementation of individual DFT algorithm components** — genuinely correct in their internal logic — assembled into a working but **inaccurate SCF engine** where the proposal's four headline differentiators exist as standalone demos **not integrated into the production compute path**.

### One-sentence truth

> Real algorithms, wrong integration: the pieces are research-grade, the assembly is a prototype, and the accuracy is 6–8 orders of magnitude short of the proposal's own gates.

---

## Maturity Spectrum Assessment

```
TOY MODEL ←────────────────────────────────────────────→ REAL MODEL
           ↑                    ↑                         ↑
     [TIDES is here]    [Where the ledger   [Where the proposal
     Individual algos   claims it is]       claims it will be]
     work correctly,
     but disconnected
     and inaccurate
```

### What "beyond toy" means here

The individual algorithm implementations are **not toys**. They are real, correct, tested:

| Component | Maturity | Evidence |
|---|---|---|
| Tile GEMM (CUTLASS/cuBLASLt) | **Real** | 967 GFLOPS, 91.7% cuBLASLt, FP16/FP8 Ozaki paths |
| SP2 purification | **Real** | ‖P²−P‖_F ≤ 3.6e-15, CPU + GPU |
| ChFSI | **Real** | Error ≤7.2e-10, spectral bounds correct |
| FOE/Chebyshev | **Real** | Trace ≤1e-15, Fermi search ≤7e-15 |
| Grid rho/vmat build | **Real** | Machine-precision match CPU↔GPU |
| Poisson (FFT + ISF) | **Real** | ≤7.8e-15 vs analytic |
| XC (LDA-PW92 + PBE) | **Real** | libxc linked, 37/37 tests |
| Two-center integrals | **Real** | Complete angular momentum (ss→dd), rotation invariance ≤1e-12 |
| Forces (analytic + FD) | **Real** | FD ≤2.9e-13 Ha/Bohr on model systems |
| Broker dispatch | **Real** | Implemented, wired into SCFDriver, routes to R0/R1/R2/R3 |
| A-posteriori error bounds | **Real** | Commutator norm, energy/force bounds |
| D4 dispersion | **Real** | Charge-dependent C6, EEQ model |

### What "not real model" means here

The **production engine** — the thing that takes atoms in and gives DFT energies out — does not meet any proposal acceptance gate, and the differentiators are not in the compute path:

| Component | Status | Gap to "Real" |
|---|---|---|
| **SCF energy accuracy** | ❌ 6–8 orders short | H: 0.08 Ha vs 1e-8 gate; H2O: 3.0 Ha vs 1e-8 gate |
| **Mixed precision in SCF** | ❌ Decorative | Flag reports mode string; no matrices quantized; no f64e reductions |
| **Tile substrate in compute** | ❌ Decorative | Used for trace(P,H) only; actual GEMM uses LAPACK dsyev/dgemm |
| **GPU GEMM feeding SCF** | ❌ No consumer | 967 GFLOPS kernel exists; SCF loop uses CPU LAPACK |
| **XL-BOMD shadow dynamics** | ❌ Not true | NaoDriver::RunXLBOMD calls full SCF per step for forces |
| **CUDA graph capture** | ❌ Decorative | Captures nothing; counts CPU lambdas |
| **QTT in SCF** | ❌ Post-processing | Compresses P after convergence; not in SCF loop |
| **K-point SCF** | ❌ Not wired | Bloch tiles exist; not in NaoDriver solve path |
| **Scale claims** | ❌ Extrapolated | 10k atoms = 50-atom run; 10^6 = CPU simulation; O(N) on 50–400 atoms |
| **Competitor benchmarks** | ❌ Zero | Not a single row of the 12-row piecewise matrix run |
| **Hybrid functional SCF** | ❌ Model only | ACE/PBE0 on model system; not in real SCF |

---

## Detailed Audit by Category

### 1. The SCF Engine — Core Product

**What exists**: `nao_driver.hpp` (2155 lines) assembles a real SCF pipeline: NAO basis generation → two-center S/T → grid V_ext → SCF loop (rho build → Poisson → XC → vmat → eigensolve) → energy assembly → forces.

**What's real**:
- The pipeline actually runs and converges for H, H2, He, H2O
- GEMM-based rho/vmat builds from density matrix P (not orbitals) — structurally compatible with R2/R3
- GPU XC pipeline wired via `#ifdef TIDES_HAVE_CUDA` with GpuArena
- GPU free-space Poisson wired for molecular systems
- Broker dispatch wired into SCFDriver (routes to R0/R1/R2/R3 based on input)
- Pulay/DIIS + Kerker + Broyden mixers all real
- Mermin finite-Te occupations wired when electronic_temp > 0
- D4, HSE correction, a-posteriori bounds, point-group sym — all callable as flags

**What's broken**:
- **Energy accuracy is 6–8 orders of magnitude short of proposal gates**:

| System | Proposal Gate | Actual Tolerance | Actual Error | Orders Short |
|---|---|---|---|---|
| H atom (NAO) | ≤1e-8 Ha vs PySCF | 0.08 Ha | ~0.061 Ha | **6** |
| H2 (NAO) | ≤1e-8 Ha | 0.10 Ha | ~0.086 Ha | **6** |
| H2 (GTO) | ≤1e-8 Ha | 0.2 Ha | ~0.375 Ha | **7** |
| H2O (GTO) | ≤1e-8 Ha | 3.0 Ha | ~2.8 Ha | **8** |
| Ne (GTO) | ≤1e-8 Ha | — | 65.8 Ha | **10** |

  Root cause: grid-based V_H/V_xc on a 0.3 Bohr grid vs PySCF analytic ERIs. The grid is too coarse by ~10× for chemical accuracy.

- **No real pseudopotential SCF energy validation**: UPF2 reader + KB projectors implemented in NaoDriver, but no test asserts PP-based SCF energy at proposal accuracy.

- **XL-BOMD on real DFT is not shadow dynamics**: `NaoDriver::RunXLBOMD` sets `force_fn` to call `ComputeForces()` which runs its own SCF. Each MD step does a full SCF for forces + potentially a full SCF for density refresh. This is Born-Oppenheimer MD with an XL-BOMD wrapper, not true XL-BOMD (1 solve/step with shadow potential).

### 2. The Tile Substrate — Differentiator #1 (partially)

**What's real**:
- `TileMat` data structure works: CSR-of-tiles, FromDense, TraceFp64, SpGEMM
- GPU GEMM achieves 91.7% cuBLASLt throughput (967 GFLOPS)
- Ozaki FP64 emulation: FP16 + FP8 variants, error ≤3.6e-14
- SpGEMM filtered with ε_filter
- CUDA graph capture RAII wrapper

**What's decorative**:
- **Tile substrate is NOT the compute backbone**: In the NaoDriver SCF loop, TileMat is used only for:
  1. `trace(P, H)` via `SpGemmFilteredFp64` + `TileMat::TraceFp64` when n ≥ 32
  2. Reporting sparsity statistics
  
  The actual matrix operations — eigensolve, rho build, vmat build — all use CPU LAPACK (`dsyev_`) and CPU GEMM (`BuildRhoGemm`/`BuildHmatGemm`). The 967 GFLOPS GPU GEMM has **no consumer** in the SCF loop.

- **Mixed precision is a label, not an operation**: When `use_mixed_precision=true`, the NaoDriver calls `MixedPrecisionSCF::AutoSelect(n, 1e-6)` and sets a result string. It does NOT:
  - Quantize any matrix to BF16/FP16
  - Use Ozaki f64e reductions in the SCF loop
  - Store density matrix in reduced precision
  - Perform any operation differently based on the precision mode
  
  The differentiator #1 from the proposal ("Mixed-precision-native tile execution with Ozaki-scheme FP64 emulation") is **completely unexercised in the production path**.

### 3. GPU Integration

**What's real**:
- GPU XC engine (LDA + PBE) with `XcEval` fused kernel, auto-dispatch
- GPU rho build from density matrix P (`RhoBuildFromDensityMatrixCuda`)
- GPU vmat build (adjoint map)
- GPU Poisson (both periodic cuFFT and free-space zero-padded convolution)
- GPU SP2 (51× speedup at n=256, CPU fallback n<128)
- GpuArena persistent allocation pool (no per-call cudaMalloc for XC)

**What's not real**:
- **GPU eigensolve not in NaoDriver path**: `cusolver_batched.hpp` exists with `syevjBatched`, but NaoDriver always uses CPU `dsyev_` via SCFDriver
- **No GPU GEMM in SCF**: The tile GEMM (967 GFLOPS) is never called during SCF
- **Per-iteration CPU download**: GPU XC pipeline runs on device, but results are downloaded to CPU for matrix assembly (BuildHmatGemm is CPU)
- **No CUDA graph replay**: `use_cuda_graph` flag calls `graph.BeginCapture()` then counts CPU lambda "operations" — no actual CUDA graph is built or replayed

### 4. Solver Broker — Better Than Gap Analysis Claimed

**Correction to prior gap analysis**: The gap analysis stated "broker.cpp is an empty file (0 bytes)." This is **FALSE**. `broker.cpp` (142 lines) contains a full `BrokerRunner::Solve` that dispatches to R0/R1/R2/R3.

**What's real**:
- `SolverBroker::Dispatch` correctly routes based on n_basis, gap, Te
- `SCFDriver::Run` actually checks the regime and calls SP2/FOE/ChFSI/dense eig accordingly
- `BrokerRunner::Solve` provides standalone dispatch-and-solve

**What's not exercised**:
- The NaoDriver always sets `gap_estimate=1.0`, `electronic_temp=0.0`, and all test systems have `n_basis ≤ 200`, so the broker **always dispatches R0** (dense eigensolve)
- R1/R2/R3 paths in SCFDriver are implemented but **never triggered** in any NaoDriver test
- The calibration table uses heuristic thresholds, not measured crossover data

### 5. Linear-Scaling Solvers (R2/R3)

**What's real**:
- SP2 purification: correct, ‖P²−P‖_F ≤ 3.6e-15
- Submatrix construction: block idempotency ≤3.3e-13
- FOE: trace ≤1e-15, Fermi search ≤7e-15
- GPU SP2: 51× speedup at n=256 (single-block only)
- O(N) scaling measurement: `on_scaling_benchmark.cpp` measures 50/100/200/400 atoms

**What's not real**:
- **10k-atom run is a 50-atom extrapolation**: T5.8 validates tile structure + memory extrapolation, not actual throughput
- **Multi-block GPU SP2 not implemented**: Only single-block GPU SP2 works
- **10^6-atom run is CPU simulation**: T5.9 is "CPU-only weak scaling simulation"
- **O(N) measured only on 50–400 atoms**: The proposal requires "O(N^1.0±0.1) measured 10⁴→10⁶" — this is 2–3 orders of magnitude short
- **No metallic Al at Te=3000K**: GPU FOE tested only on n=128

### 6. XL-BOMD — Differentiator #2

**What's real**:
- XL-BOMD class with 3 kernel orders (identity, diagonal scaling, low-rank)
- NHC thermostat with Suzuki-Yoshida integration (4-chain, 3 sub-steps)
- Langevin thermostat
- ASPC warm starts
- NVE drift ≤30 µHa/at/ps verified (50000 steps, dt=0.2fs) on model Hamiltonian

**What's not real**:
- **KSA kernel is still approximate**: kernel_order=0 is identity, kernel_order=1 is "diagonal scaling based on |P| magnitude" (heuristic, not a true Jacobian inverse), kernel_order=2 is "softened scaling on largest residuals" (not a true low-rank projection)
- **No true shadow dynamics on real DFT**: `NaoDriver::RunXLBOMD` passes `force_fn` that calls `ComputeForces()` (full SCF) and `density_fn` that calls `Run()` (full SCF). Each step does a full SCF for forces. This is BOMD with an XL-BOMD propagation wrapper, not XL-BOMD's "1 solve/step" promise.
- **XL-BOMD tests use model Hamiltonian callbacks**: The drift verification is on a model potential, not the NAO SCF engine.

### 7. Verification & Benchmarking — Differentiator #4

**What's real**:
- `tolerances.yaml` exists with per-rung budgets
- 6-rung ladder framework
- A/B harness (mixed vs FP64)
- FD force checks
- A-posteriori error bounds (real implementation)
- Rung 6 physics tests (H atom ACWF, H2 binding curve, NAO vs GTO)

**What's not real**:
- **Test bars are 6–667× looser than proposal gates**: NVE drift tested at 30 (matches gate), but energy bars are 0.08–3.0 Ha vs 1e-8 Ha gate
- **Zero competitor benchmarks run**: Not a single row of the 12-row piecewise matrix (§9.1) has been executed against ABACUS/CP2K/SPARC/GPAW/GPU4PySCF
- **No deployed regression dashboard**: SQLite + NVML interfaces exist, not running
- **No nightly CI**: Scripts exist, runners not deployed
- **PySCF benchmark was invalid**: Used model H₂ Hamiltonian, marked `_AUDIT_INVALID:true`
- **No energy (kWh) in any benchmark**: NVML interface exists, never deployed in a benchmark run
- **Rung 6 is minimal**: H atom ACWF + H2 binding curve shape — not the full ACWF/Δ/S22/W4-11 validation the proposal describes

### 8. Hybrids & Dispersion

**What's real**:
- D3(BJ) dispersion: force FD ≤7.3e-17
- D4 dispersion: implemented (290 lines) with charge-dependent C6, EEQ model
- ISDF: LSQ interpolation, reconstruction ≤6.4e-12
- ACE: PBE0 model system exact
- HSE screening: short-range exchange with tile-based SpGEMM

**What's not real**:
- **No hybrid functional SCF on real molecules**: ACE/PBE0 tested on model system only
- **HSE not in NaoDriver SCF loop**: `use_hse_screening` computes a post-SCF correction energy, not a self-consistent hybrid SCF
- **PAW**: feasibility memo only (expected — Y3 decision)

### 9. API & Community

**What's real**:
- nanobind bindings wired to NaoDriver + MoleculeDriver + ComputeForces
- Python API tries NaoDriver → MoleculeDriver → model stub (with warning)
- TOML config schema (10 sections)
- 5 tutorials pass as pytest
- JAX bridge with custom_vjp + gradcheck

**What's not real**:
- **ASE calculator uses model Hamiltonian in practice**: NaoDriver is "preferred" but the tutorials and examples still default to model
- **No PyPI release**
- **No Sphinx build integrated into CI**
- **No auto-docs generator**

---

## Score Card: Proposal Component → Implementation Reality

| Proposal Component | Implementation | Integration | Accuracy | Scale | Overall |
|---|---|---|---|---|---|
| Tile substrate (WP1) | ✅ Real | ⚠️ Trace only | ✅ ≤1e-13 | ✅ Tested | **B** |
| NAO basis + integrals (WP2) | ✅ Real | ✅ In SCF | ❌ 6 ord short | ⚠️ Small | **C+** |
| Grids/Poisson/XC (WP3) | ✅ Real | ✅ In SCF | ✅ ≤1e-14 | ⚠️ Small | **B+** |
| Mid-range solvers (WP4) | ✅ Real | ✅ Broker wired | ✅ ≤1e-9 | ⚠️ R0 only | **B** |
| Linear scaling (WP5) | ✅ Real | ⚠️ Standalone | ✅ ≤1e-15 | ❌ 50–400 at | **C+** |
| SCF/XL-BOMD/forces (WP6) | ⚠️ Partial | ✅ In SCF | ❌ Energy | ❌ Not true XL | **C** |
| Hybrids/dispersion (WP7) | ✅ Real | ❌ Model only | N/A | N/A | **C** |
| Parallel/HPC (WP8) | ⚠️ Stubs | ❌ Not real | N/A | ❌ None | **D+** |
| Verification (WP9) | ✅ Framework | ❌ Not deployed | ❌ Bars loose | ❌ Zero | **D+** |
| API/docs (WP10) | ✅ Real | ⚠️ Model fallback | N/A | N/A | **B-** |

**Differentiator Reality Check**:

| Differentiator | Proposal Claim | Code Reality | Verdict |
|---|---|---|---|
| #1: Mixed-precision Ozaki tile execution | "Production-grade, exceeds native DGEMM" | GPU kernels exist at 91.7% cuBLASLt. **Zero mixed-precision operations in SCF loop.** Flag only reports a string. | **Demo, not product** |
| #2: XL-BOMD shadow dynamics | "~1 solve/step, time-reversible" | XL-BOMD class works on model potentials. **On real DFT, each step calls full SCF for forces.** KSA kernel is heuristic approximation. | **Prototype, not product** |
| #3: Batched many-system execution | "≥10⁴ molecules/hour/GPU" | cuSOLVER syevjBatched exists. **No batched SCF driver. No CUDA graph for batched SCF.** Never run. | **Not started** |
| #4: Certified accuracy per joule | "Every benchmark carries accuracy contract" | A-posteriori bounds implemented. **Energy accuracy is 6–8 orders short. No kWh in any benchmark. Zero competitor comparisons.** | **Framework, not practice** |

---

## The Gap: Where TIDES Actually Is

### Compared to a "Toy Model"

**TIDES is well beyond a toy model.** The individual algorithms are real, correct, and tested at machine precision. A toy model would use simplified physics (e.g., tight-binding, 2-band models). TIDES implements:
- Full NAO basis generation with confined-atom radial solver
- Complete two-center integrals with all angular momentum channels (ss through dd)
- Real Poisson solvers (FFT + ISF) for all boundary conditions
- Real XC (LDA-PW92 + PBE) via libxc
- Real SP2/ChFSI/FOE/OMM solvers with correct convergence properties
- Real forces with Pulay terms and FD verification
- Real a-posteriori error bounds

### Compared to a "Real Model" (production DFT code like ABACUS/SIESTA/SPARC)

**TIDES is far from a real model.** A real DFT code must:
1. **Produce chemically accurate energies** — TIDES is 6–8 orders of magnitude short
2. **Use its performance architecture in the compute path** — TIDES' tile GEMM and mixed precision are decorative
3. **Demonstrate the claimed scaling** — TIDES has no run above 400 atoms
4. **Validate against external codes** — TIDES has zero competitor benchmarks
5. **Run on real hardware at target scale** — TIDES' GPU SCF, multi-node, and 10k+ runs don't exist

### The Honest Position

```
Toy Model          Reference Implementation          Production Code
  |                         |                              |
  |--------- TIDES ---------|------------------------------|
  |    (individual algos    |   (SCF engine +              |
  |     are real, correct,  |    integration gaps +       |
  |     tested)             |    accuracy gaps +            |
  |                         |    scale gaps remain)         |
```

**TIDES is a high-quality reference implementation with real algorithm components, where the integration into a production-grade DFT engine is incomplete:**
- The engine runs but produces inaccurate energies (grid-limited)
- The GPU/mixed-precision/tile architecture exists but is not in the compute path
- The linear-scaling solvers exist but are tested only at 50–400 atoms
- XL-BOMD exists but doesn't do true shadow dynamics on real DFT
- The verification framework exists but no real benchmark has been run
- The broker works but is never exercised beyond R0

---

## What Would Make TIDES "Real" — Critical Path

Ranked by impact on closing the gap to a production DFT code:

### P0: Make the Engine Accurate
1. **Grid convergence study**: Increase grid resolution from 0.3→0.1 Bohr and measure energy convergence. If energies don't converge to ≤1e-4 Ha, the grid integration scheme itself is wrong.
2. **Validate PP-based SCF**: Run H/He/Ne with real PseudoDojo ONCV pseudopotentials and assert energy vs ABACUS/SIESTA at ≤1e-4 Ha.
3. **Fix Ne atom**: 65.8 Ha error indicates a fundamental problem (d-electron integration, pseudopotential handling, or grid).

### P1: Make the Architecture Real
4. **Route eigensolve through tile GEMM**: Replace `dsyev_` in SCFDriver with tile-based path when n ≥ 64, using the existing 967 GFLOPS kernel.
5. **Activate mixed precision**: When `use_mixed_precision=true`, actually quantize P/H to BF16/FP16 and use Ozaki f64e for energy traces. Measure the error vs FP64 path.
6. **Wire GPU eigensolve**: Use `cusolver_batched` in the SCF loop instead of CPU LAPACK.

### P2: Make the Scale Claims Real
7. **Run 2000-atom SP2 on GPU**: Actually execute, not extrapolate. Measure O(N) at 500/1000/2000/5000 atoms.
8. **Run XL-BOMD on real DFT with 1 solve/step**: Wire XL-BOMD to use the NAO Hamiltonian with shadow dynamics (no per-step SCF for forces).

### P3: Make the Verification Real
9. **Run one competitor benchmark**: ABACUS or SIESTA on the same system at matched basis. Report time-to-fixed-accuracy.
10. **Tighten test bars to proposal gates**: If the engine can't meet 1e-8 Ha, find out why and fix it. Don't widen the bar.

---

## Bottom Line

The PROJECT-LEDGER's claim of "65/65 tasks DONE (100%)" is **false by its own evidence**. The gap analysis's claim that "broker.cpp is an empty file" is also **false** — it was updated since.

The truth is in between:

- **~40% of the proposal is genuinely implemented and working**: individual algorithms, grid operations, basis integrals, solver math, forces, dispersion, error bounds.
- **~30% is implemented but not integrated into the product path**: tile GEMM, mixed precision, GPU eigensolve, CUDA graphs, QTT, batched execution.
- **~20% is implemented but at wrong accuracy or scale**: SCF energy (6–8 orders short), XL-BOMD (not true shadow dynamics), linear scaling (50–400 atoms), verification (zero competitor benchmarks).
- **~10% is genuinely missing**: real multi-node execution, deployed CI, competitor benchmarks, PyPI release, hybrid functional SCF.

The codebase is a **strong foundation** — the algorithm implementations are genuinely impressive and correct. But it is **not yet a DFT engine** in the sense the proposal describes: a GPU-native, mixed-precision, tile-substrate engine with solver-broker dispatch, XL-BOMD MD, certified accuracy, and competitive benchmarks. It is a **CPU-first reference implementation with GPU demos attached and an SCF engine that runs but doesn't yet produce accurate energies**.
