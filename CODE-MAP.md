# TIDES — Complete Code Map

*Generated July 2026. All files and directories in the repository.*

---

## Root Directory

```
New_DFT_Code/
├── .vscode/
│   └── settings.json
├── .git/
├── TIDES_5yr_proposal.md              # 5-year proposal (master document, 44 KB)
├── tides-docs-README.md               # Documentation pack overview & reading order
├── tides-docs-pack.zip                # Zipped docs archive
├── sample-WP1-tile-substrate.md       # Sample WP task breakdown
├── CODE-MAP.md                        # THIS FILE
├── tides/                             # Main codebase (see below)
├── tides-docs/                        # Documentation (see below)
└── build-cuda/                        # CUDA build artifacts (540 items, generated)
```

---

## `tides/` — Main Codebase

```
tides/
├── CMakeLists.txt                     # 648-line build system (CMake 3.24+, C++20)
├── CMakePresets.json                  # Build presets (debug/release/cuda)
├── pyproject.toml                     # Python package config (pip install -e .)
├── AUDIT-REPORT.md                    # Full project audit (584 lines, July 2026)
├── CHANGELOG.md                       # Keep-a-Changelog format
├── CITATION.cff                       # Citation file format
├── CONTRIBUTING.md                    # Contribution guidelines
├── GOVERNANCE.md                      # Project governance
├── LICENSE                            # Apache-2.0
├── progress.txt                       # Per-engine progress tracker (151 lines)
│
├── core/                              # ══ C++20 core engine (no Python deps) ══
│   │
│   ├── common/                        # Shared infrastructure
│   │   ├── status.hpp                 #   Typed Status/Result<T> (no exceptions)
│   │   ├── config.hpp                 #   Physical constants, SCFConfig, GridConfig, etc.
│   │   ├── units.hpp                  #   Hartree atomic units conversions
│   │   └── logging.hpp                #   Logging utilities
│   │
│   ├── tile/                          # WP1: Tile substrate & precision
│   │   ├── layout.hpp                 #   TileMat: CSR-of-tiles block-sparse matrix (522 lines)
│   │   ├── ops.hpp                    #   Tile operations (axpy, trace, norms)
│   │   ├── precision.hpp              #   PrecisionDescriptor, NumericFormat, OperationLedger
│   │   ├── f64e_reference.hpp         #   FP64-emulation CPU reference (long-double oracle)
│   │   ├── graphs.hpp                 #   CUDA graph capture/replay utilities
│   │   ├── gemm_grouped.hpp           #   Grouped GEMM header (CUTLASS/cuBLASLt wrapper)
│   │   ├── gemm_grouped.cu            #   GPU grouped GEMM kernel (62 KB, largest CUDA file)
│   │   ├── spgemm_filtered.hpp        #   Filtered sparse tile GEMM (CPU + error ledger)
│   │   ├── spgramm_filtered_cuda.hpp  #   CUDA SpGEMM header
│   │   ├── spgemm_filtered.cu         #   GPU filtered SpGEMM kernel
│   │   ├── ozaki.hpp                  #   Ozaki FP16/FP8 slicing + GEMM reference (483 lines)
│   │   ├── ozaki.cu                   #   GPU Ozaki f64e GEMM kernel
│   │   ├── reduce_f64e.hpp            #   f64e reduction header
│   │   ├── reduce_f64e.cpp            #   f64e deterministic reductions
│   │   └── tests/                     #   25 test files
│   │       ├── tilemat_tests.cpp              # TileMat invariants & round-trip
│   │       ├── tilemat_layout_probe.cpp       # Layout probe
│   │       ├── tilemat_dump_fixture.cpp       # HDF5 dump fixture generator
│   │       ├── tilemat_hdf5_roundtrip_test.py # HDF5 round-trip (Python)
│   │       ├── spgemm_filtered_tests.cpp      # SpGEMM analytical bounds
│   │       ├── spgemm_filter_probe.cpp        # SpGEMM probe
│   │       ├── ops_tests.cpp                  # Tile ops contracts
│   │       ├── tile_ops_probe.cpp             # Tile ops probe
│   │       ├── precision_tests.cpp            # Precision descriptor & ledger
│   │       ├── f64e_reference_tests.cpp       # f64e adversarial numerics
│   │       ├── f64e_probe.cpp                 # f64e probe
│   │       ├── ozaki_slice_tests.cpp          # Ozaki FP16 slice planning
│   │       ├── ozaki_gemm_probe.cpp           # Ozaki GEMM probe
│   │       ├── deterministic_gauntlet_tests.cpp # Deterministic substrate gauntlet
│   │       ├── cuda_gemm_tests.cpp            # CUDA GEMM tests
│   │       ├── cuda_gemm_probe.cpp            # CUDA GEMM probe (24 KB)
│   │       ├── cuda_spgemm_tests.cpp          # CUDA SpGEMM tests
│   │       ├── cuda_spgemm_probe.cpp          # CUDA SpGEMM probe
│   │       ├── cuda_ozaki_gemm_tests.cpp      # CUDA Ozaki GEMM tests
│   │       ├── cuda_reduce_f64e_tests.cpp     # CUDA f64e reduction tests
│   │       ├── cuda_reduce_f64e_probe.cpp     # CUDA f64e reduction probe
│   │       ├── cuda_graph_tests.cpp           # CUDA graph capture tests
│   │       ├── cuda_graph_probe.cpp           # CUDA graph probe
│   │       ├── cuda_determinism_tests.cpp     # CUDA determinism tests
│   │       └── cuda_runtime_link_probe.cpp    # CUDA runtime link probe
│   │
│   ├── basis/                         # WP2: NAO basis & integrals
│   │   ├── nao_generator.hpp          #   NAO generation from confined-atom solver
│   │   ├── two_center_integrals.hpp   #   Two-center integral framework
│   │   ├── two_center_gpu.hpp         #   GPU two-center kernel header
│   │   ├── two_center.cu              #   GPU two-center integrals (overlap, T, V_ext)
│   │   ├── three_center_gpu.hpp       #   GPU three-center kernel header
│   │   ├── three_center.cu            #   GPU three-center KB projectors
│   │   ├── atomgen/                   #   Atomic generator (confined-atom solver)
│   │   │   ├── radial_grid.hpp        #     Log/linear radial grid
│   │   │   ├── radial_solver.hpp      #     Radial Schrödinger solver (Numerov)
│   │   │   ├── numerov_solver.hpp     #     Numerov method implementation
│   │   │   ├── atomic_lda.hpp         #     Atomic LDA SCF solver
│   │   │   ├── lda_xc.hpp             #     LDA exchange-correlation
│   │   │   ├── dense_sym_eig.hpp      #     Dense symmetric eigensolver
│   │   │   ├── symmetric_eigensolver.hpp #  Symmetric eigensolver (Jacobi)
│   │   │   ├── tridiag_eig.hpp        #     Tridiagonal eigensolver
│   │   │   ├── selective_tridiag_eig.hpp #  Selective tridiagonal eigensolver
│   │   │   └── tests/                 #     7 test files
│   │   │       ├── atomic_lda_tests.cpp
│   │   │       ├── atomic_lda_oracle.py
│   │   │       ├── eigensolver_tests.cpp
│   │   │       ├── hydrogenic_tests.cpp
│   │   │       ├── hydrogenic_oracle.py
│   │   │       ├── hydrogenic_r_oracle.py
│   │   │       └── lda_xc_tests.cpp
│   │   ├── pseudo/                    #   Pseudopotential readers
│   │   │   ├── pseudopotential.hpp    #     Pseudopotential abstraction
│   │   │   ├── upf2_reader.hpp        #     UPF2 format reader (ONCV/PseudoDojo)
│   │   │   └── tests/
│   │   │       └── pseudo_tests.cpp
│   │   ├── paw/                       #   PAW (flag-gated, empty)
│   │   └── tests/                     #   7 WP2 test files
│   │       ├── nao_tests.cpp
│   │       ├── two_center_tests.cpp
│   │       ├── cuda_two_center_tests.cpp
│   │       ├── cuda_three_center_tests.cpp
│   │       ├── derivative_tests.cpp
│   │       ├── pyscf_overlap_tests.cpp
│   │       └── profile_wp2.cpp
│   │
│   ├── grid/                          # WP3: Grid, Poisson, XC
│   │   ├── dual_grid.hpp              #   Dual real-space grid (coarse+fine, 4 BCs)
│   │   ├── poisson.hpp                #   Poisson solver interface
│   │   ├── poisson_fft_gpu.hpp        #   GPU Poisson FFT header
│   │   ├── poisson_fft.cu             #   GPU Poisson FFT kernel (cuFFT + FFTW3)
│   │   ├── rho_build.hpp              #   Density builder interface
│   │   ├── rho_build_gpu.hpp          #   GPU rho build header
│   │   ├── rho_build.cu               #   GPU density construction kernel
│   │   ├── vmat_build.hpp             #   Potential matrix builder interface
│   │   ├── vmat_build_gpu.hpp         #   GPU V_mat build header
│   │   ├── vmat_build.cu              #   GPU V_H + V_xc matrix build kernel
│   │   ├── xc.hpp                     #   XC functional interface
│   │   ├── xc_gpu.hpp                 #   GPU XC header
│   │   ├── xc.cu                      #   GPU XC kernel (LDA PW92 + PBE via libxc)
│   │   ├── libxc_wrapper.hpp          #   libxc C API wrapper
│   │   ├── poisson_qtt/               #   QTT Poisson (flag-gated, empty)
│   │   └── tests/                     #   8 test files
│   │       ├── dual_grid_tests.cpp
│   │       ├── grid_ops_tests.cpp
│   │       ├── poisson_tests.cpp
│   │       ├── cuda_poisson_fft_tests.cpp
│   │       ├── cuda_rho_build_tests.cpp
│   │       ├── cuda_vmat_build_tests.cpp
│   │       ├── cuda_xc_tests.cpp
│   │       └── libxc_pbe_tests.cpp
│   │
│   ├── solvers/                       # WP4/WP5: All solver regimes
│   │   ├── broker.hpp                 #   SolverBroker: regime dispatch logic
│   │   ├── broker.cpp                 #   (empty — logic in header)
│   │   ├── dense/                     #   R0: Batched dense eigensolver
│   │   │   └── batched_eig.hpp        #     LAPACK dsygv_ / cuSOLVER syevjBatched
│   │   ├── chfsi/                     #   R1: Chebyshev-filtered subspace iteration
│   │   │   └── chfsi.hpp              #     ChFSI + Lanczos spectral bounds
│   │   ├── omm/                       #   OMM: Orbital minimization method
│   │   │   └── omm.hpp               #     OMM solver (fallback for R2)
│   │   ├── sp2_submatrix/             #   R2: SP2 purification + submatrix
│   │   │   ├── sp2.hpp                #     SP2 CPU reference (Niklasson recursion)
│   │   │   ├── sp2_gpu.hpp            #     SP2 GPU header
│   │   │   ├── sp2_gpu.cu             #     SP2 GPU kernel (51x speedup at n=256)
│   │   │   ├── submatrix.hpp          #     Submatrix extraction & assembly
│   │   │   └── truncation.hpp         #     Error-compensated truncation policy
│   │   ├── foe_sq/                    #   R3: Fermi-operator expansion
│   │   │   ├── foe.hpp                #     FOE Chebyshev expansion of Fermi function
│   │   │   └── fermi_search.hpp       #     Chemical potential search (bracketed Newton)
│   │   └── tests/                     #   3 test files
│   │       ├── wp4_tests.cpp          #     WP4 solver tests
│   │       ├── wp5_tests.cpp          #     WP5 linear-scaling tests
│   │       └── cuda_sp2_tests.cpp     #     CUDA SP2 tests
│   │
│   ├── scf/                           # WP6: SCF driver & energy
│   │   ├── scf_driver.hpp             #   SCF loop with DIIS/Pulay + Kerker fallback
│   │   ├── energy_assembly.hpp        #   Total energy: E_kin + E_ne + E_H + E_xc + E_ion
│   │   ├── stress.hpp                 #   Stress tensor (periodic)
│   │   └── tests/
│   │       └── wp6_tests.cpp
│   │
│   ├── dynamics/                      # WP6: Molecular dynamics
│   │   ├── xlbomd/
│   │   │   └── xlbomd.hpp             #   XL-BOMD shadow dynamics (Verlet + kernel)
│   │   ├── optimizers/
│   │   │   └── optimizers.hpp         #   FIRE + L-BFGS geometry optimizers
│   │   ├── md_driver/                 #   MD driver (empty, planned)
│   │   └── neb/                       #   NEB (empty, planned)
│   │
│   ├── forces/                        # WP6: Analytic forces
│   │   └── analytic_forces.hpp        #   HF forces + 5-point FD validation
│   │
│   ├── hybrids/                       # WP7: Hybrid functionals & dispersion
│   │   ├── d3_dispersion.hpp          #   D3/D4 dispersion correction
│   │   ├── ace/
│   │   │   └── ace.hpp                #   ACE (asymptotically corrected exchange)
│   │   ├── isdf/
│   │   │   └── isdf.hpp               #   ISDF (interpolative separable density fitting)
│   │   └── tests/
│   │       └── wp7_tests.cpp
│   │
│   ├── parallel/                      # WP8: Parallelism & I/O
│   │   ├── graph_partitioner.hpp      #   Recursive coordinate bisection partitioner
│   │   ├── halo_exchange.hpp          #   Halo exchange framework (NCCL)
│   │   └── tests/
│   │       └── wp8_tests.cpp
│   │
│   ├── io/
│   │   └── stage_dump.hpp             #   HDF5 stage-dump schema (bisect-the-physics)
│   │
│   ├── ham/                           #   Hamiltonian assembly (empty, planned)
│   │
│   ├── verification/                  # WP9: Verification ladder
│   │   ├── ladder_runner.hpp          #   Test ladder runner
│   │   └── tests/
│   │       └── wp9_tests.cpp
│   │
│   └── tests/                         # Cross-cutting integration tests
│       ├── benchmark.cpp              #   End-to-end benchmark suite
│       ├── physics_tests.cpp          #   Physics-level integration tests
│       └── gpu_regression_tests.cpp   #   GPU regression tests
│
├── api/                               # ══ User-facing API layer ══
│   ├── python/
│   │   ├── ase_calculator.py          #   (empty — ASE calculator is in tides/ subpkg)
│   │   ├── tides/                     #   Python package
│   │   │   ├── __init__.py            #     Package init, exports
│   │   │   ├── status.py              #     Status/Result Python mirrors
│   │   │   ├── config.py              #     TidesConfig + TOML schema (14 KB)
│   │   │   ├── core.py                #     TidesCalculator facade (20 KB, model Hamiltonian)
│   │   │   ├── ase_calculator.py      #     ASE-compatible calculator
│   │   │   ├── cli.py                 #     CLI: run, tune, bench, verify (10 KB)
│   │   │   ├── _native.cpp            #     nanobind binding stubs
│   │   │   └── __pycache__/           #     (generated)
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── test_wp10.py           #     Python API integration tests
│   │   │   └── test_pyscf_crosscheck.py #   PySCF cross-validation
│   │   └── tides_dft.egg-info/        #     (generated, pip install -e .)
│   ├── jax_bridge/
│   │   ├── __init__.py
│   │   └── tides_jax.py               #   JAX bridge: energy_and_forces custom VJP
│   └── cli/                           #   CLI (empty — logic in python/tides/cli.py)
│
├── tests/                             # ══ Per-engine test profiles ══
│   └── per_engine/
│       ├── e1_tile/e1_test_profile.cpp        # E1: Tile substrate profiling
│       ├── e2_basis/e2_test_profile.cpp       # E2: Basis & integrals profiling
│       ├── e3_grid/e3_test_profile.cpp        # E3: Grid/Poisson/XC profiling
│       ├── e4_solvers/e4_test_profile.cpp     # E4: Solvers profiling
│       ├── e5_scf/e5_test_profile.cpp         # E5: SCF profiling
│       ├── e6_dynamics/e6_test_profile.cpp    # E6: Dynamics profiling
│       ├── e7_parallel/e7_test_profile.cpp    # E7: Parallel profiling
│       ├── e8_hybrids/e8_test_profile.cpp     # E8: Hybrids profiling
│       └── e9_verification/e9_test_profile.cpp # E9: Verification profiling
│
├── bench/                             # ══ Benchmarking & profiling ══
│   ├── comprehensive_benchmark.py     #   Full benchmark suite (23 KB)
│   ├── pyscf_benchmark.py             #   PySCF comparison benchmark
│   ├── pyscf_vs_tides_profile.py      #   Side-by-side profiling
│   ├── pyscf_benchmark_results.json   #   PySCF results data
│   ├── profiling_ledger.json          #   GPU profiling measurements
│   ├── ENGINE_OPTIMIZATION_RESEARCH.md #  Optimization research notes (14 KB)
│   └── optimization/                  #   Per-engine optimization ledgers
│       ├── comprehensive_benchmark.json       # Full benchmark data
│       ├── comprehensive_benchmark.md         # Summary report
│       ├── pyscf_vs_tides_ledger.json         # PySCF vs TIDES comparison
│       ├── e1_tile/optimization_ledger.json
│       ├── e2_basis/optimization_ledger.json
│       ├── e3_grid/optimization_ledger.json
│       ├── e4_solvers/optimization_ledger.json
│       ├── e5_scf/optimization_ledger.json
│       ├── e6_dynamics/optimization_ledger.json
│       ├── e7_parallel/optimization_ledger.json
│       ├── e8_hybrids/optimization_ledger.json
│       └── e9_verification/optimization_ledger.json
│
├── perf/                              # ══ Performance tracking ══
│   ├── README.md                      #   Performance methodology
│   ├── model-ledger.md                #   Analytical performance models
│   ├── task-ledger.md                 #   Task-level performance tracking
│   ├── summarize_cpu.py               #   CPU profiling summarizer
│   └── logs/
│       ├── wp1_gpu_20260705T192428Z.txt  # WP1 GPU profiling log
│       └── wp2_cpu_20260705T191837Z.jsonl # WP2 CPU profiling log
│
├── verification/                      # ══ Verification framework ══
│   ├── tolerances.yaml                #   Per-kernel ULP/abs tolerances (8 KB)
│   ├── references/
│   │   └── gauntlet10.yaml            #   10-molecule gauntlet reference data
│   └── runners/                       #   (empty, planned)
│
├── ci/                                # ══ Continuous integration ══
│   ├── .gitlab-ci.yml                 #   GitLab CI pipeline
│   ├── nightly.sh                     #   Nightly test script
│   ├── setup.sh                       #   CI environment setup
│   └── spack/
│       └── package.py                 #   Spack package recipe
│
├── cmake/                             # ══ CMake modules ══
│   ├── cuda.cmake                     #   CUDA detection (empty stub)
│   └── hip.cmake                      #   HIP detection (empty stub)
│
├── docs/                              # ══ In-repo documentation ══
│   ├── theory-manual.md               #   Theory manual with derivations
│   └── P4_benchmark_report.md         #   Phase 4 benchmark report
│
├── examples/                          # ══ Tutorials (double as integration tests) ══
│   ├── tutorial_01_single_point.py    #   Single-point SCF
│   ├── tutorial_02_forces_optimization.py # Forces + geometry optimization
│   ├── tutorial_03_xlbomd_md.py       #   XL-BOMD molecular dynamics
│   ├── tutorial_04_toml_input.py      #   TOML input file usage
│   └── tutorial_05_solver_broker.py   #   Solver broker regime dispatch
│
├── tools/
│   └── tilemat_hdf5.py                #   TileMat HDF5 bridge tool (12 KB)
│
├── benchmarks/                        #   (empty, planned)
├── external/                          #   (empty, for vendored deps)
└── build/                             #   Build artifacts (generated, 992 items)
```

---

## `tides-docs/` — Documentation Pack

```
tides-docs/
├── README.md                          # Documentation overview & reading order
├── AGENT.md                           # RALPH Protocol (Reconnaissance→Architecture→Logic→Proof→Handoff)
├── DEPENDENCY-GRAPH.md                # 65-task dependency graph, critical path, milestone gates (485 lines)
│
├── 00-project/                        # Project-level documents
│   ├── 00-vision-scope-claims.md      #   Vision, honest claims table, non-goals
│   ├── 01-hardware-strategy.md        #   Workstation-first: RTX 24GB primary, A40 48GB, H100 occasional
│   ├── 02-roadmap-phases-milestones.md #  4 phases / 60 months: A (molecules), B (extended), C (scale-out), D (v1.0)
│   ├── 03-team-raci-interfaces.md     #   Team organization, RACI matrix
│   ├── 04-risk-register.md            #   Risk register with mitigations
│   ├── 05-governance-license.md       #   Apache-2.0, governance principles
│   ├── 06-task-management-howto.md    #   Task tracking conventions
│   └── 99-original-onefile-proposal.md #  Original proposal (same as TIDES_5yr_proposal.md)
│
├── 10-physics/                        # Physics model documents
│   ├── 10-nao-basis.md                #   NAO basis: φ=R_nl·Y_lm, confined atom, DZP/TZP
│   ├── 11-pseudopotentials.md         #   ONCV from PseudoDojo, UPF2 format
│   ├── 12-xc-dispersion.md            #   LDA/GGA/hybrid XC, D3/D4 dispersion
│   ├── 13-electrostatics-boundary-conditions.md # Dual grid, cuFFT, ISF kernels, 4 BCs
│   ├── 14-finite-temperature-metals.md #  Mermin finite-Te DFT
│   ├── 15-hybrid-functionals.md       #   HSE06, ISDF, ACE
│   ├── 16-forces-stress.md            #   HF + Pulay forces, stress tensor
│   ├── 15_jax_xc_exchange_correlation_fu.pdf  # JAX XC reference
│   ├── s41467-026-73232-8_reference.pdf # Prolate Ewald reference
│   ├── ankh-a-generalized-o(n)-interpolated-ewald-strategy-...pdf # Interpolated Ewald
│   ├── ci6c00123_si_001.pdf           #   Supporting info reference
│   └── fast-fourier-transform-...pdf  #   FFT dihedral parametrization reference
│
├── 20-math/                           # Mathematical methods
│   ├── 20-tile-algebra.md             #   TileMat: CSR-of-tiles, SpGEMM, filter error bounds
│   ├── 21-mixed-precision-ozaki.md    #   Ozaki FP64 emulation, error model, escalation
│   ├── 22-purification-submatrix.md   #   SP2 + submatrix method, error-compensated truncation
│   ├── 23-foe-spectral-quadrature.md  #   FOE Chebyshev, SQ, mu search
│   ├── 24-chfsi-and-dense.md          #   R0 batched eig, R1 ChFSI, subspace reuse
│   ├── 25-xlbomd.md                   #   XL-BOMD shadow dynamics, KSA kernel, thermostats
│   ├── 26-qtt-research.md             #   QTT compression (flag-gated research thrust)
│   └── 27-error-control.md            #   Certified accuracy, a-posteriori bounds
│
├── 30-architecture/                   # Architecture documents
│   ├── 30-repo-layout.md              #   Monorepo layout, directory = owner rule
│   ├── 31-data-contracts.md           #   TileMat, GridArray, HDF5 stage-dump schema
│   ├── 32-solver-broker.md            #   Regime dispatch, `tides tune` calibration
│   ├── 33-precision-policy.md         #   Op-by-op precision table, escalation rules
│   ├── 34-parallelism-io.md           #   Phase A/B/C parallelism, NCCL, MPI/NVSHMEM
│   └── 35-coding-standards.md         #   C++20, CUDA, Python standards, DCO, linters
│
├── 40-engines/                        # Work package task decompositions
│   ├── WP1-tile-substrate.md          #   T1.1–T1.8: TileMat, GEMM, SpGEMM, Ozaki, graphs, HIP
│   ├── WP2-basis-integrals.md         #   T2.1–T2.8: NAO, two/three-center, derivatives, pseudo
│   ├── WP3-grid-poisson-xc.md         #   T3.1–T3.8: Dual grid, rho, V_mat, Poisson, XC
│   ├── WP4-midrange-solvers.md        #   T4.1–T4.6: R0 batched, R1 ChFSI, broker
│   ├── WP5-linear-scaling.md          #   T5.1–T5.6: SP2, submatrix, FOE, Fermi search
│   ├── WP6-scf-xlbomd-forces.md       #   T6.1–T6.6: SCF, energy, forces, XL-BOMD, optimizers
│   ├── WP7-hybrids-dispersion.md      #   T7.1–T7.4: D3, ISDF, ACE, PAW feasibility
│   ├── WP8-parallel-hpc.md            #   T8.1–T8.4: Partitioner, halos, HDF5, CI
│   ├── WP9-verification-benchmarks.md #   T9.1–T9.4: Tolerances, references, ladder, dashboard
│   └── WP10-api-docs-community.md     #   T10.1–T10.8: Python API, ASE, CLI, TOML, JAX, packaging
│
├── 50-verification/                   # Verification framework
│   ├── 50-test-ladder.md              #   6 rungs: kernel → operator → energy → force → dynamics → physics
│   ├── 51-tolerances.md               #   Tolerance framework documentation
│   └── 52-reference-data.md           #   Reference data management
│
└── 60-benchmarks/                     # Benchmarking
    ├── 60-protocol.md                 #   Benchmark protocol (accuracy contract mandatory)
    ├── 61-piecewise-matrix.md         #   Piecewise matrix benchmark specification
    └── 62-campaigns.md                #   Benchmark campaign plans
```

---

## File Type Summary

| Type | Count | Purpose |
|---|---|---|
| `.hpp` / `.h` | ~35 | C++20 headers (header-only library) |
| `.cu` | 10 | CUDA GPU kernels |
| `.cpp` | ~30 | C++ test/probe/benchmark sources |
| `.py` | ~20 | Python API, tests, examples, tools |
| `.cmake` | 2 | CMake modules (cuda, hip) |
| `.yaml` / `.yml` | 4 | CI, tolerances, reference data |
| `.json` | ~15 | Benchmark results, optimization ledgers, profiling |
| `.md` | ~40 | Documentation, proposals, reports |
| `.pdf` | 4 | Reference papers |
| `.toml` | 1 | Python packaging (pyproject.toml) |

## Key Source Files by Function

### Core Engine (C++20, header-only)
| File | Lines | Role |
|---|---|---|
| `core/tile/layout.hpp` | 522 | TileMat — CSR-of-tiles block-sparse matrix |
| `core/tile/ozaki.hpp` | 483 | Ozaki FP16/FP8 slicing + f64e GEMM reference |
| `core/tile/gemm_grouped.cu` | ~2000 | GPU grouped GEMM (largest CUDA file, 63 KB) |
| `core/solvers/broker.hpp` | 131 | Solver regime dispatch (R0/R1/R2/R3) |
| `core/solvers/chfsi/chfsi.hpp` | 258 | Chebyshev-filtered subspace iteration |
| `core/solvers/sp2_submatrix/sp2.hpp` | 208 | SP2 density-matrix purification |
| `core/solvers/foe_sq/foe.hpp` | 167 | Fermi-operator expansion (Chebyshev) |
| `core/solvers/dense/batched_eig.hpp` | 208 | Batched dense eigensolver (LAPACK) |
| `core/scf/scf_driver.hpp` | 244 | SCF loop with DIIS/Pulay mixing |
| `core/scf/energy_assembly.hpp` | 95 | Total energy decomposition |
| `core/dynamics/xlbomd/xlbomd.hpp` | 155 | XL-BOMD shadow dynamics |
| `core/forces/analytic_forces.hpp` | 101 | HF forces + FD5 validation |
| `core/basis/nao_generator.hpp` | 211 | NAO basis generation |
| `core/common/status.hpp` | 90 | Typed Status/Result (no exceptions) |
| `core/common/config.hpp` | 70 | Constants, SCFConfig, GridConfig, XCConfig |

### Python API
| File | Lines | Role |
|---|---|---|
| `api/python/tides/core.py` | 569 | TidesCalculator facade (SCF, forces, MD) |
| `api/python/tides/config.py` | ~400 | TidesConfig + TOML schema |
| `api/python/tides/cli.py` | ~300 | CLI: run, tune, bench, verify |
| `api/python/tides/ase_calculator.py` | ~200 | ASE-compatible calculator |
| `api/python/tides/status.py` | ~85 | Status/Result Python mirrors |
| `api/jax_bridge/tides_jax.py` | ~160 | JAX differentiable bridge |

### GPU CUDA Kernels (10 files)
| File | Size | Role |
|---|---|---|
| `core/tile/gemm_grouped.cu` | 63 KB | Grouped GEMM (CUTLASS/cuBLASLt) |
| `core/tile/spgemm_filtered.cu` | 12 KB | Filtered sparse tile SpGEMM |
| `core/tile/ozaki.cu` | 15 KB | Ozaki f64e GEMM (FP16+FP8) |
| `core/basis/two_center.cu` | 13 KB | Two-center integrals |
| `core/basis/three_center.cu` | 14 KB | Three-center KB projectors |
| `core/grid/rho_build.cu` | 11 KB | Density construction |
| `core/grid/vmat_build.cu` | 7 KB | V_H + V_xc matrix build |
| `core/grid/poisson_fft.cu` | 12 KB | Poisson FFT (cuFFT + FFTW3) |
| `core/grid/xc.cu` | 13 KB | XC functional evaluation |
| `core/solvers/sp2_submatrix/sp2_gpu.cu` | 11 KB | SP2 purification on GPU |

### Build & Config
| File | Role |
|---|---|
| `CMakeLists.txt` | 648-line build system, 50+ test targets |
| `CMakePresets.json` | Debug/release/cuda presets |
| `pyproject.toml` | Python package (nanobind, ASE, numpy deps) |
| `cmake/cuda.cmake` | CUDA detection (stub) |
| `cmake/hip.cmake` | HIP detection (stub) |
| `ci/.gitlab-ci.yml` | CI pipeline |
| `ci/spack/package.py` | Spack package recipe |
