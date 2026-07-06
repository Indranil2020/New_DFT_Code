# TIDES Per-Engine Optimization Research (2024-2026)

This document records research findings from recent literature (2024-2026),
compares against current TIDES implementations, and lists actionable
optimization recommendations for each engine.

---

## E1 — Tile Substrate Engine

### Current Implementation
- **Grouped GEMM**: cuBLAS `cublasGemmStridedBatchedEx` for FP64 and FP16-accum
- **SpGEMM**: Custom kernel with Frobenius-norm filtering, CSR-of-tiles layout
- **Ozaki f64e**: FP16-slice decomposition → grouped FP16 GEMM → long-double accumulation
- **CUDA Graphs**: Replay for repeated GEMM calls
- **Precision Ledger**: OperationLedger tracks error bounds per operation

### Performance Status
- GEMM: 684 GFLOPS vs cuBLAS 900 GFLOPS target (76%)
- SpGEMM: Working but no GPU-optimized sparse format
- Ozaki: Working but uses FP16 only (no FP8 path)

### Key Papers (2024-2026)

1. **"DGEMM without FP64 Arithmetic — Using FP64 Emulation and FP8 Tensor Cores with Ozaki Scheme"** (arXiv:2508.00441, 2025)
   - Extends Ozaki scheme to FP8 Tensor Cores on NVIDIA Blackwell
   - Inner-product dimension blocking to accelerate FP16-based implementations
   - FP64 arithmetic emulation via integer arithmetic (completely eliminates FP64 instructions)
   - **Relevance**: TIDES Ozaki implementation uses FP16 only. Can add FP8 path for 2-4x speedup on Blackwell GPUs.
   - **Action**: Add FP8 Ozaki path; add inner-product blocking to existing FP16 path.

2. **"Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores Through Extensions of the Ozaki Scheme"** (arXiv:2511.13778, 2025)
   - Automatic Dynamic Precision (ADP): fully GPU-resident framework
   - Exponent Span Capacity (ESC): hardware-agnostic estimator for slice count
   - Unsigned integer slicing scheme: increases representational efficiency
   - Up to 2.3x speedup on GB200, 13.2x on RTX Pro 6000 Blackwell
   - **Relevance**: TIDES uses fixed max_slices=16. ESC can dynamically determine optimal slice count.
   - **Action**: Implement ESC-based dynamic slice selection; add unsigned integer slicing.

3. **"cuTeSpMM: Accelerating Sparse-Dense Matrix Multiplication using GPU Tensor Cores"** (arXiv:2504.06443, 2025)
   - Uses tensor cores for SpMM via dense blocking of sparse matrix
   - Substantially higher performance than cuSPARSE
   - **Relevance**: TIDES SpGEMM uses custom CUDA kernel without tensor cores.
   - **Action**: Explore tensor-core-accelerated SpGEMM for dense tile blocks.

4. **"Accelerating Density Fitting with Adaptive-precision and 8-bit"** (arXiv:2601.08077, 2026)
   - Adaptive precision for density fitting GEMMs
   - Mixed FP32/FP16/INT8 based on error bounds
   - **Relevance**: TIDES GEMM uses fixed precision per call. Adaptive precision can reduce compute.
   - **Action**: Add precision auto-selection based on input magnitude analysis.

### Optimization Recommendations (Priority Order)
1. **Add FP8 Ozaki path** — 2-4x speedup on Blackwell GPUs (RTX 5050 is Blackwell)
2. **Implement ESC dynamic slicing** — Reduce slice count for well-conditioned inputs
3. **Add inner-product blocking** — Accelerate FP16 Ozaki by reducing slice-pair count
4. **Explore tensor-core SpGEMM** — Use cuTeSpMM approach for tile-level SpGEMM
5. **GEMM tile dispatch tuning** — Improve from 76% to 90% of cuBLASLt
6. **Add MPI distributed GEMM** — For multi-rank tile distribution

---

## E2 — Basis & Integrals Engine

### Current Implementation
- **Radial solver**: Numerov for l>0, FD for l=0, 1e-10 at n=50000
- **NAO generation**: DZP→TZP monotone convergence
- **Two-center integrals**: Rotation invariance ≤1e-12, PySCF overlap ≤8.6e-9
- **GPU tile assembly**: two_center.cu + three_center.cu, max_diff=4.3e-19

### Key Papers (2024-2026)
1. **"Coupling all-electron full-potential DFT with plane-wave DFT"** (arXiv:2507.17672, 2025)
   - FFT-based methods for Gaussian-type projectors
   - Analytic integrals for separable dual-space GTH pseudopotentials
   - **Action**: Explore analytic integral paths for GTH PPs.

2. **"Accelerating finite-element-based PAW density functional theory"** (arXiv:2604.26037, 2026)
   - Multi-resolution quadrature for atom-centered integrals
   - **Action**: Consider multi-resolution quadrature for three-center integrals.

### Optimization Recommendations
1. **GPU three-center parallelization** — Current 368 lines, can improve occupancy
2. **Spline caching for two-center** — Pre-compute radial splines on GPU
3. **Multi-resolution quadrature** — For highly localized basis functions
4. **MPI distribution of integral tables** — For large systems

---

## E3 — Grid Engine

### Current Implementation
- **RhoBuild**: O(N * n_orb), CPU + GPU
- **VmatBuild**: v→H adjoint, CPU + GPU (max_diff=4.9e-16)
- **Poisson**: CPU uses naive O(N²) DFT (KNOWN ISSUE); GPU uses cuFFT
- **XC**: LDA-PW92 + PBE GGA via libxc, CPU + GPU

### Key Papers (2024-2026)

1. **"A scalable high-order multigrid-FFT Poisson solver for unbounded problems"** (arXiv:2512.08555, 2025)
   - Multigrid-FFT hybrid for Poisson on unbounded domains
   - Green's function convolution with FFT acceleration
   - **Relevance**: TIDES CPU Poisson uses O(N²) naive DFT. This is the #1 bottleneck.
   - **Action**: Replace CPU naive DFT with FFTW-based FFT. For free BCs, use multigrid-FFT.

2. **"A FFT-based GMRES for fast solving of Poisson equation"** (arXiv:2509.23180, 2025)
   - FFT-based GMRES iterative solver for Poisson
   - Sixth-order compact finite difference discretization
   - **Relevance**: Alternative to direct FFT for variable-coefficient Poisson.
   - **Action**: Consider FFT-GMRES for non-periodic boundary conditions.

3. **"Interpolative separable density fitting on adaptive real space grids"** (arXiv:2510.20826, 2025)
   - Adaptive real space grids for ISDF
   - Dual-space multilevel kernel-splitting for Poisson
   - **Relevance**: Adaptive grids can reduce grid points for localized basis functions.
   - **Action**: Explore adaptive grids for rho build and Poisson solve.

### Optimization Recommendations (Priority Order)
1. **Link FFTW for CPU Poisson** — Replace O(N²) with O(N log N). Critical fix.
2. **Implement multigrid-FFT for free BCs** — For unbounded Poisson problems
3. **GPU Poisson cuFFT optimization** — Already fast, tune block sizes
4. **Adaptive grid for rho build** — Reduce grid points for localized orbitals
5. **GPU XC kernel optimization** — Improve occupancy for LDA/PBE evaluation
6. **MPI distributed grid** — Domain decomposition for large grids

---

## E4 — Solvers Engine

### Current Implementation
- **DenseEig**: LAPACK reference
- **SP2**: CPU reference, ‖P²−P‖_F ≤3.6e-15
- **ChFSI**: KNOWN ISSUE — filter direction inverted
- **FOE/Chebyshev**: Trace ≤1e-15 at adequate order
- **OMM**: KNOWN ISSUE — CG stuck at ~0.1 error
- **Broker**: Dispatches between solvers

### Key Papers (2024-2026)

1. **"Residual-based Chebyshev filtered subspace iteration for sparse Hermitian eigenvalue problems"** (arXiv:2503.22652, 2025)
   - Residual-based ChFSI tolerant to inexact matrix-vector products
   - Tested on DFT generalized eigenproblems with 85 million grid points
   - **Relevance**: Directly addresses TIDES ChFSI filter direction bug and scalability.
   - **Action**: Fix filter direction; implement residual-based stopping criterion.

2. **"Matrix-free algorithms for fast ab initio calculations on distributed"** (arXiv:2512.08571, 2025)
   - Matrix-free FE-DFT operators with roofline analysis
   - **Relevance**: TIDES uses matrix-based approach. Matrix-free can reduce memory.
   - **Action**: Explore matrix-free ChFSI for large-scale problems.

3. **"NeuralSCF: Neural network self-consistent fields"** (arXiv:2406.15873, 2024)
   - Modified Pulay mixing on density coefficients
   - ML-accelerated SCF convergence
   - **Relevance**: Can accelerate SCF convergence by providing better initial guesses.
   - **Action**: Consider ML-based initial density guess for faster SCF.

### Optimization Recommendations (Priority Order)
1. **Fix ChFSI filter direction** — Negate spectral window or reverse polynomial
2. **Fix OMM CG** — Add Armijo backtracking line search
3. **Implement residual-based ChFSI** — From arXiv:2503.22652
4. **Wire GPU SP2 batching** — Connect SpGEMM + f64e for GPU SP2
5. **Add Anderson acceleration** — Alternative to Pulay for difficult systems
6. **Matrix-free ChFSI** — For 85M+ grid point problems

---

## E5 — SCF Engine

### Current Implementation
- **Mixing**: Pulay, Kerker, Broyden, simple
- **Energy assembly**: Uses f64e for energy traces
- **Stress tensor**: Not implemented (deferred to Phase B)

### Key Papers (2024-2026)
1. **"Preconditioning Magnetic Systems in Kohn-Sham DFT"** (arXiv:2606.26693, 2026)
   - Advanced preconditioning for difficult SCF convergence
   - **Action**: Add preconditioning for metallic/magnetic systems.

2. **"ABACUS: An Electronic Structure Analysis Package for the AI Era"** (arXiv:2501.08697, 2025)
   - Multi-secant mixing methods, Kerker preconditioning
   - **Action**: Implement multi-secant mixing as generalization of Pulay/Broyden.

### Optimization Recommendations
1. **Add multi-secant mixing** — Generalization of Pulay/Broyden
2. **Kerker preconditioning** — For metallic systems
3. **SCF restart from checkpoint** — For long-running calculations
4. **GPU-accelerated energy assembly** — Use f64e reductions on GPU
5. **Stress tensor implementation** — For Phase B

---

## E6 — Forces & Dynamics Engine

### Current Implementation
- **FD5Force**: KNOWN ISSUE — sign inverted
- **XL-BOMD**: Extended Lagrangian MD, 100 steps
- **FIRE**: Fast Inertial Relaxation Engine
- **NVE drift**: 7762 uHa/at/ps (KNOWN ISSUE — short simulation)

### Key Papers (2024-2026)
1. **"ABACUS: Born-Oppenheimer MD"** (arXiv:2501.08697, 2025)
   - BOMD with wavefunction extrapolation for faster SCF convergence
   - **Action**: Add wavefunction extrapolation between MD steps.

2. **"Accelerating Long-Term Molecular Dynamics with Physics-Informed ML"** (arXiv:2510.01206, 2025)
   - ML-based MD acceleration for long-term simulations
   - **Action**: Consider ML-enhanced MD for long-time-scale simulations.

### Optimization Recommendations (Priority Order)
1. **Fix FD5Force sign** — Remove negative sign in formula
2. **Increase MD steps for NVE test** — 1000+ steps with dt≤0.5 fs
3. **Add wavefunction extrapolation** — Between MD steps for faster SCF
4. **GPU force computation** — Offload force kernels to GPU
5. **Parallel tempering support** — For enhanced sampling

---

## E7 — Parallel Engine

### Current Implementation
- **RCB partitioner**: Recursive coordinate bisection
- **Halo exchange**: MPI-based ghost zone communication
- **Comm fraction**: Measures communication overhead

### Optimization Recommendations
1. **Space-filling curve partitioning** — Better locality than RCB
2. **Non-blocking halo exchange** — Overlap computation with communication
3. **GPU-aware MPI** — Use CUDA-aware MPI for direct device-to-device transfer
4. **Dynamic load balancing** — Rebalance during SCF iterations
5. **Multi-GPU support** — Single-node multi-GPU with NCCL

---

## E8 — Hybrids Engine

### Current Implementation
- **D3 dispersion**: Working
- **ISDF**: KNOWN ISSUE — no LSQ fit, uses delta interpolation
- **ACE**: Adaptive compressed exchange
- **PBE0**: Model system only

### Key Papers (2024-2026)

1. **"Interpolative separable density fitting on adaptive real space grids"** (arXiv:2510.20826, 2025)
   - ISDF with adaptive grids for highly localized basis functions
   - Dual-space multilevel kernel-splitting for Poisson
   - Proves adaptive grid resolving pair densities can be constructed from single-particle basis
   - **Relevance**: Directly addresses TIDES ISDF LSQ fit issue.
   - **Action**: Implement LSQ fit; explore adaptive grids for ISDF.

2. **"Accelerating Density Fitting with Adaptive-precision and 8-bit"** (arXiv:2601.08077, 2026)
   - 8-bit density fitting for Coulomb/exchange integrals
   - **Action**: Explore 8-bit precision for ISDF auxiliary basis.

### Optimization Recommendations (Priority Order)
1. **Fix ISDF LSQ fit** — Implement `x = M(:,interp) * pinv(M(interp,:)) * M(interp,:)`
2. **Add adaptive grid ISDF** — From arXiv:2510.20826
3. **GPU ISDF point selection** — Parallelize random projection + greedy pivoting
4. **ACE-ISDF combined approach** — Use ISDF to compress ACE basis
5. **HSE06 screening** — Range-separated hybrid with short-range exchange

---

## E9 — Verification Engine

### Current Implementation
- **6-rung ladder**: Forces, SCF, SP2, Poisson, Dynamics, Physics
- **Budgets**: Defined in tolerances.yaml
- **Known issues**: NVE drift (Rung 5), Physics pipeline (Rung 6)

### Optimization Recommendations
1. **Increase NVE simulation length** — 1000+ steps for stable drift measurement
2. **Add PySCF reference comparison** — Use collected benchmark data
3. **Add regression dashboard** — SQLite-based with energy metering
4. **Nightly automated testing** — CI/CD with tolerances.yaml checks
5. **Competitor farm** — Containerized comparisons with ABACUS, CP2K, PySCF

---

## Summary of Research Impact

| Engine | Papers Found | Key Opportunity | Expected Speedup |
|---|---|---|---|
| E1 Tile | 4 | FP8 Ozaki + ESC dynamic slicing | 2-4x on Blackwell |
| E2 Basis | 2 | Multi-resolution quadrature | 1.5-2x for 3-center |
| E3 Grid | 3 | FFTW for CPU Poisson | 100-1000x (O(N²)→O(N log N)) |
| E4 Solvers | 3 | Residual-based ChFSI + fix bugs | Critical (correctness) |
| E5 SCF | 2 | Multi-secant mixing + preconditioning | 20-50% fewer iterations |
| E6 Dynamics | 2 | Fix FD5Force + wavefunction extrapolation | 2-3x faster MD |
| E7 Parallel | 0 | Space-filling curves + GPU-aware MPI | 1.5-2x for large systems |
| E8 Hybrids | 2 | ISDF LSQ + adaptive grids | 10-100x (correctness + speed) |
| E9 Verification | 0 | Longer NVE + PySCF comparison | Validation improvement |
