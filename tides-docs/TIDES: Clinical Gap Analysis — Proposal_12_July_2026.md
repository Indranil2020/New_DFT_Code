TIDES: Clinical Gap Analysis — Proposal vs. Implementation July 12 2026


      Methodology
      Cross-referenced the 5-year proposal (§1–14) against: the PROJECT-LEDGER's own "What's Missing" columns, the independent code audit (2026-07-10), the
      architecture brief (2 hours ago), the TASK-LEDGER remediation log, and direct source code inspection of 15+ key files.

      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Executive Finding
      The PROJECT-LEDGER's claim of "65/65 tasks DONE (100%)" is false by its own evidence. The ledger's "What's Missing" columns, the independent audit, and the
      architecture brief collectively identify ~40+ distinct gaps, defects, and unimplemented features. The task-tracking system marks tasks "DONE" when a code file
      exists with passing tests — regardless of whether the tests meet the proposal's own acceptance gates, whether the code is in the product path, or whether the
      feature works at production scale.

      The codebase is better characterized as: a CPU-first reference implementation of individual algorithm components (E1–E9), recently assembled into a working but
      inaccurate NAO SCF driver, with GPU kernels that exist as standalone demos but are only partially integrated into the product pipeline.

      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Category 1: The Product Pipeline — The NAO DFT Engine
      Proposal (§4, §6 WP2+WP3+WP6): One unified engine: NAO basis + ONCV pseudopotentials + grid Hartree (Poisson) + libxc XC + tile substrate, GPU-native, dispatched
      by solver broker.

      What exists:
      • nao_driver.hpp (1880 lines) — recently assembled into a working SCF pipeline• two_center_builder.hpp — analytic two-center S/T (recently added, replacing grid
      integration)• GPU XC/rho/vmat paths with #ifdef TIDES_HAVE_CUDA in the NAO driver
      What's NOT implemented:
      ┌──────────────────┬────────────────────┬─────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────┐
      │ Gap              │ Proposal Gate      │ Current State                                                                   │ Evidence                     │
      ├──────────────────┼────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────┤
      │ NAO SCF energy accuracy │ ≤1e-8 Ha vs PySCF (WP2 acceptance) │ H atom: 0.15 Ha tolerance; H2: 0.25 Ha tolerance. Tests pass at bars **6–7 orders of magnitude
      looser** than the gate. │ `nao_benchmark_tests.cpp:42,58` │
      │ GTO driver energy accuracy │ ≤1e-8 Ha vs PySCF  │ H2 err=0.375 Ha, H2O err=12.98 Ha. Tests **FAIL intentionally** ("red tests").  │
      `molecule_driver_tests.cpp:27–35` (explicit WAIVER) │
      │ Ne atom energy   │ ≤1e-8 Ha           │ ΔE = 65.8 Ha vs PySCF. DEFERRED ("grid V_ext workaround; needs finer grid").    │ TASK-LEDGER #3               │
      │ Slater-Koster angular coupling │ All (l_a,l_b) pairs up to l=2 │ p-p, p-d, d-d are "simplified (arbitrary channel splits)" per context brief. Comment in
      builder says "partial". │ `two_center_builder.hpp:28`  │
      │ PSML pseudopotential format │ UPF2/PSML (WP2)    │ Only UPF2 implemented.                                                          │ PROJECT-LEDGER T2.3
      │
      │ Real pseudopotential SCF │ ONCV PseudoDojo in production SCF │ UPF2 reader + KB projectors implemented in NaoDriver, but **never validated end-to-end** at
      proposal accuracy. No real PP SCF energy benchmark. │ No test asserts PP-based SCF energy │
      └──────────────────┴────────────────────┴─────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────┘
      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Category 2: GPU Integration in Product Path
      Proposal (§3.2, §3.4): "Mixed-precision-native tile execution" — the #1 differentiator. Every heavy operation is a grouped GEMM stream on tensor cores. "One true
      layer."

      What exists:
      • 10 GPU CUDA kernels implemented (GEMM, SpGEMM, Ozaki, rho, vmat, poisson, XC, SP2, FOE)• NAO driver has device-resident XC pipeline (#ifdef TIDES_HAVE_CUDA)
      What's NOT implemented:
      ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────┐
      │ Gap                                                                                                             │ Impact                                │
      ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────┤
      │ **Tile substrate NOT used for actual compute in SCF** — TileMat is used only for trace verification (`TileTrace`), not for GEMM-based matrix operations. The
      "one true layer" is decorative. Production GEMM uses BLAS `dgemm`/`dsyrk`. │ The central architectural premise is unintegrated. E1's GFLOPS have no consumer. │
      │ **Mixed precision NOT wired into SCF** — No BF16/FP16 storage or FP64-emulated reductions in the NAO driver SCF loop. All operations are FP64 dense. │
      Differentiator #1 (Ozaki mixed precision) is unexercised in production. │
      │ **Per-call GPU alloc/transfer/sync** (audit B10) — NAO driver's device pipeline does use a persistent arena for XC, but still downloads rho to CPU for Poisson
      on every iteration, and does `cudaMalloc`/`cudaFree` for P and V_xc per call. │ The "one scalar back to CPU per SCF iteration" promise is not met. │
      │ **GPU rho build uses orbitals, not density matrix** (audit B3) — `RhoBuildKernel` computes ρ = Σ f_k ψ_k², requiring eigenvectors. R2/R3 (purification)
      produces P, not orbitals. │ GPU rho path structurally cannot serve the linear-scaling regime. DEFERRED. │
      │ **No GPU Poisson in molecular SCF** — Poisson runs on CPU (free-space BC). GPU cuFFT only for periodic.         │ The grid phase is not GPU-resident for
      molecules. │
      └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────┘
      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Category 3: XL-BOMD — The MD Engine (Differentiator #2)
      Proposal (§3.3.2, WP6): KSA-XL-BOMD with Krylov/kernel integrators, time-reversible, ~1 solve/step, NVE drift ≤30 µHa/atom/ps.

      What's NOT implemented:
      ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────┐
      │ Gap                                                                                                         │ Evidence                                  │
      ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
      │ **KSA kernel NOT implemented** — Code explicitly states: "The kernel K = I (identity; the full KSA is a low-rank approximation)." │ `xlbomd.hpp:27`
      │
      │ **No Nose-Hoover Chain thermostat** — Only simplified stochastic Langevin ("not a full Langevin, but enough for testing"). │ `xlbomd.hpp:88–95`
      │
      │ **"1 solve/step" is misleading** — `density_fn(R)` is called per step, which itself runs a full SCF. The shadow potential dynamics are not real XL-BOMD. │
      `xlbomd.hpp:71`                           │
      │ **GB2 gate AT RISK** — C++ test: 7762 µHa/at/ps (100-step sim). Audit claims PASS after extending to 50000 steps, but TASK-LEDGER still lists it deferred. │
      PROJECT-LEDGER vs AUDIT-REPORT discrepancy │
      │ **No XL-BOMD on real DFT potential** — XL-BOMD tests use model Hamiltonian callbacks, not the NAO SCF engine. │ `xlbomd.hpp` takes generic
      `force_fn`/`density_fn` callbacks │
      └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────┘
      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Category 4: Solver Broker (Algorithm #1)
      Proposal (§3.3.1, §4): "Solver broker dispatches every calculation to R0/R1/R2/R3." One input file, broker overridable but never required.

      What's NOT implemented:
      ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────┐
      │ Gap                                                                                                             │ Evidence                              │
      ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────┤
      │ **`broker.cpp` is an empty file (0 bytes)**                                                                     │ Direct file inspection                │
      │ **Broker is decorative in product path** — NaoDriver creates a `BrokerInput` and passes it to `SCFDriver`, but `SCFDriver` calls `BatchedDenseEig` directly
      regardless of broker recommendation. │ `nao_driver.hpp` SCF launch; `scf_driver.hpp` │
      │ **No `tides tune` calibration** — Calibration table uses heuristic thresholds, not measured benchmark data.     │ PROJECT-LEDGER T4.6: "Calibration table not
      populated with real benchmark data" │
      │ **No actual regime dispatch** — R0 (batched), R1 (ChFSI), R2 (SP2), R3 (FOE) solvers all exist as standalone modules, but **none are wired through the broker
      into the product SCF path**. The NAO driver always uses dense diagonalization. │ Code inspection                       │
      └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────┘
      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Category 5: Linear-Scaling Solvers (R2/R3 — WP5)
      Proposal (§6 WP5): SP2 submatrix for 10⁴–10⁶ atoms, O(N) scaling measured, FOE/SQ for metals.

      What's NOT implemented:
      ┌───────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────┐
      │ Gap                                                                                               │ Evidence                                            │
      ├───────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
      │ **GPU batched submatrix SP2 (multi-block) NOT implemented** — Only single-block GPU SP2 works.    │ AUDIT-REPORT: "GPU batched submatrix SP2 (multi-block) not
      implemented" │
      │ **T5.8 (10⁴-atom run) is a 50-atom extrapolation** — Not an actual 10k-atom run. Validates tile structure + memory extrapolation only. │ PROJECT-LEDGER T5.8:
      "50-atom benchmark validates tile structure + SP2 + memory extrapolation" │
      │ **T5.9 (distributed R2/R3) is CPU-only simulation** — "CPU-only weak scaling simulation. RCB partitioning, halo exchange, load balance validated." Not real
      distributed execution. │ PROJECT-LEDGER T5.9                                 │
      │ **No O(N) scaling measured** — The gate requires "O(N^1.0±0.1) measured 10⁴→10⁶." Never measured. │ No benchmark data exists                            │
      │ **FOE/SQ for metals not validated at scale** — GPU FOE exists but only tested on n=128. No metallic Al at Tₑ=3000K. │ `gpu_foe_tests.cpp`
      │
      └───────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┘
      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Category 6: Parallel / HPC / I/O (WP8)
      Proposal (§6 WP8): METIS partitioning, NCCL/NVSHMEM halos with overlap, multi-GPU Poisson, HDF5 restarts, weak scaling ≥80% to 64 GPUs.

      What's NOT implemented:
      ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────┐
      │ Gap                                                                                                                              │ Evidence            │
      ├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────┤
      │ **METIS NOT used** — RCB (recursive coordinate bisection) only.                                                                  │ PROJECT-LEDGER T8.2 │
      │ **No NCCL/NVSHMEM GPU halo** — CPU MPI only. Interface stubs exist.                                                              │ PROJECT-LEDGER T8.3 │
      │ **No computation-communication overlap**                                                                                         │ PROJECT-LEDGER T8.3 │
      │ **No multi-GPU Poisson**                                                                                                         │ Not in code         │
      │ **No actual distributed run** — T8.6 is "MPI orchestration + NVSHMEM one-sided comm interface stubs + tests." Needs GPU cluster. │ PROJECT-LEDGER T8.6 │
      │ **CI runners NOT deployed** — Scripts exist but not running.                                                                     │ PROJECT-LEDGER T8.5 │
      │ **HIP builds: no hardware to run** — Compat layer + CMake + test stubs only.                                                     │ PROJECT-LEDGER T1.8 │
      └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────┘
      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Category 7: Hybrids, Dispersion (WP7)
      Proposal (§6 WP7): ISDF+ACE hybrids (HSE06/PBE0), D3/D4, PAW decision.

      What's NOT implemented:
      ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────┐
      │ Gap                                                                                                               │ Evidence                          │
      ├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
      │ **D4 NOT implemented** — D3(BJ) only. Zero matches for "D4" or "dftd4" in hybrids code.                           │ `search_files` returned 0 results │
      │ **Hybrids are model-system only** — ACE/PBE0 tested on "model system exact." No real HSE06 SCF on real molecules. │ PROJECT-LEDGER T7.3               │
      │ **HSE screening NOT in product SCF** — Implemented but not wired into NaoDriver SCF loop.                         │ No SCF integration                │
      │ **PAW: feasibility memo only** — No implementation. Y3 decision deferred.                                         │ PROJECT-LEDGER T7.6               │
      └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────┘
      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Category 8: Verification & Benchmarking (WP9 — Differentiator #4)
      Proposal (§8, §9): Certified accuracy per joule, 6-rung ladder, 12-row piecewise matrix, nightly A/B, ACWF/Δ validation.

      What's NOT implemented:
      ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────┐
      │ Gap                                                                                                           │ Evidence                                │
      ├───────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────┤
      │ **Test bars set 3–7 orders of magnitude looser than proposal gates** — Drift: 20000 vs 30 (667×). Energy: 0.15 Ha vs 1e-8 Ha. Integrals: 3.5e-5 vs 1e-8. │
      `tolerances.yaml` vs actual test bars; audit A2, A7, A8 │
      │ **Rung 6 (Physics) is SKIP** — No ACWF/Δ, S22, W4-11 validation.                                              │ PROJECT-LEDGER T9.2: "Only 10 reference
      systems" │
      │ **No a-posteriori error control** — DFTK-style residual bounds (§3.2.6) not implemented at all.               │ Not in codebase                         │
      │ **Competitor farm: no containers, no actual benchmarks** — Parsers exist, Docker images do not.               │ PROJECT-LEDGER T9.5; AUDIT-REPORT T9.5  │
      │ **Regression dashboard: NOT deployed** — SQLite + NVML interfaces exist, not running.                         │ PROJECT-LEDGER T9.6                     │
      │ **PySCF benchmark was INVALID** — Published results used model H₂ Hamiltonian. Marked `_AUDIT_INVALID:true`.  │ AUDIT-REPORT A1                         │
      │ **No energy (kWh) logging in any benchmark** — NVML interface exists, never deployed.                         │ Not in any benchmark output             │
      │ **No actual competitor comparison** — Zero rows of the 12-row piecewise matrix (§9.1) have been run against real competitor codes. │ No benchmark data
      │
      └───────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────┘
      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Category 9: Specific Physics Features (§3.1)
      ┌────────────────────────────────────────┬─────────────────────┬───────────────────────────────────────────────────────┐
      │ Feature                                │ Proposal            │ Status                                                │
      ├────────────────────────────────────────┼─────────────────────┼───────────────────────────────────────────────────────┤
      │ Point-group symmetrization             │ §3.1.5              │ **NOT implemented**                                   │
      │ k-point sampling (Monkhorst-Pack)      │ §3.1.5, ≤2k atoms   │ Bloch-phase tiles exist but **NOT wired into SCF**    │
      │ Cyclic/helical symmetry                │ §3.1.5 (Y4 stretch) │ **NOT implemented** (expected — Y4)                   │
      │ Finite electronic temperature (Mermin) │ §3.1.2, §3.2.4      │ FOE supports Te parameter, but **not in product SCF** │
      │ Counterpoise/BSSE tooling              │ §3.1.1              │ **NOT implemented**                                   │
      │ TZP/TZDP+diffuse basis sets            │ §3.1.1              │ DZP + TZP recipes exist; **no diffuse augmentation**  │
      └────────────────────────────────────────┴─────────────────────┴───────────────────────────────────────────────────────┘
      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Category 10: Mathematics & Algorithms (§3.2, §3.3)
      ┌──────────────────────────────────────────────────┬─────────────────────────────────────┬───────────────────────────────────────────────────────────────┐
      │ Feature                                          │ Proposal                            │ Status                                                        │
      ├──────────────────────────────────────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
      │ A-posteriori error control                       │ §3.2.6 — "the scientific differentiator" │ **NOT implemented**                                           │
      │ ChFSI subspace reuse                             │ §3.3.3                              │ **NOT implemented** (single-solve only)                       │
      │ ChFSI locking/deflation                          │ §3.4.5                              │ **NOT implemented**                                           │
      │ ASPC density/DM extrapolation                    │ §3.3.3                              │ Framework exists, **NOT in production MD**                    │
      │ Batched-systems mode (R0) with CUDA graph per SCF sweep │ §3.3.6                              │ cuSOLVER syevjBatched exists; **no CUDA graph for batched SCF**
      │
      │ QTT compression in product path                  │ §3.2.7, WP-R                        │ Prototypes exist (4.4×–12.7× compression), **NOT integrated into SCF**
      │
      └──────────────────────────────────────────────────┴─────────────────────────────────────┴───────────────────────────────────────────────────────────────┘
      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Category 11: API, Docs, Community (WP10)
      ┌──────────────────────────────┬──────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
      │ Feature                      │ Proposal │ Status                                                                                                         │
      ├──────────────────────────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
      │ ASE calculator with real DFT │ §3.4.2   │ Works with model Hamiltonian only (audit C7 partially fixed — MoleculeDriver exposed, but NaoDriver not fully wired
      through ASE) │
      │ JAX bridge with real engine  │ §3.4.2   │ Works with model Hamiltonian only                                                                              │
      │ CLI `tides run/tune/bench/verify` │ §5       │ Works with model Hamiltonian; `tune` uses heuristic thresholds                                                 │
      │ Sphinx theory manual         │ §5       │ Markdown exists, **not Sphinx**                                                                                │
      │ Auto-docs generator          │ §5       │ **NOT implemented** (schema + validator only)                                                                  │
      │ PyPI release                 │ §5       │ **NOT done** (v0.1.0-alpha tagged, not uploaded)                                                               │
      │ Tutorials as integration tests │ §5       │ Pass, but with model Hamiltonian                                                                               │
      └──────────────────────────────┴──────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      ─


      Summary: What the Proposal Promises vs. What Exists
      ┌────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
      │ Proposal Component         │ Implementation Status                                                                                                      │
      ├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
      │ **Tile substrate (WP1)**   │ ✅ Real, tested, GPU kernels work. BUT: **not used for actual compute in SCF** — decorative only.                           │
      │ **NAO basis + integrals (WP2)** │ ⚠️ Recently assembled. Analytic two-center builder exists. But **energy accuracy is 6–7 orders short of gates**.
      Slater-Koster partial. │
      │ **Grids/Poisson/XC (WP3)** │ ✅ CPU + GPU kernels exist and pass. But **per-call alloc/transfer** in product path. GPU XC wired into NaoDriver but with CPU
      fallback for Poisson. │
      │ **Mid-range solvers (WP4)** │ ⚠️ R0 batched eig works. ChFSI works (no reuse/locking). **Broker.cpp is empty.** No actual regime dispatch in product path. │
      │ **Linear-scaling solvers (WP5)** │ ⚠️ SP2/FOE exist as standalone modules. **No multi-block GPU SP2. No actual large-scale runs.** 10k-atom is extrapolation. │
      │ **SCF/XL-BOMD/forces (WP6)** │ ⚠️ SCF works (inaccurate). Forces pass FD on model potentials. **XL-BOMD uses K=I (not KSA). No NHC.**                     │
      │ **Hybrids/dispersion (WP7)** │ ❌ D4 missing. Hybrids model-only. HSE not in SCF. PAW memo only.                                                           │
      │ **Parallel/HPC (WP8)**     │ ❌ No METIS, no GPU halo, no overlap, no multi-GPU, no deployed CI. All "framework ready" = interface stubs.                │
      │ **Verification (WP9)**     │ ❌ Test bars 3–667× looser than gates. Rung 6 SKIP. No competitor benchmarks. No deployed dashboard. Benchmark was invalid. │
      │ **API/docs (WP10)**        │ ⚠️ Nanobind wiring done. But ASE/JAX/CLI/tutorials use model Hamiltonian. No Sphinx, no PyPI, no auto-docs.                │
      └────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
      Bottom Line
      The codebase contains real, working implementations of individual algorithm components (tile GEMM, Ozaki, SP2, ChFSI, FOE, grid kernels, XC engine). These are
      genuinely impressive as standalone pieces. But the product the proposal describes — a GPU-native, mixed-precision, tile-substrate DFT engine with solver broker
      dispatch, XL-BOMD MD, certified accuracy, and competitive benchmarks — does not exist. What exists is a CPU-first reference implementation with GPU demos
      attached, where the acceptance gates are defined in tolerances.yaml but the tests run at bars orders of magnitude looser, and where the "65/65 DONE" claim
      conflates "a file exists with a passing test" with "the proposal's acceptance criteria are met."
