# TIDES Codebase Audit — 2026-07-10
## Independent review of code, test reports, and benchmark claims (post "engine tested" commits)

**Scope:** working tree at commit `c1195d2` ("feat(scf): BLAS dsyrk density matrix, WMMA tensor
cores, nanobind fixes"). Reviewed: `core/scf`, `core/grid`, `core/tile`, `core/dynamics`,
`api/python`, `tests/per_engine`, `bench/*`, `progress.txt`, `AUDIT-REPORT.md`.

**Verdict in one paragraph:** the individual engines (E1–E9) contain real, working code and the
per-engine tests do run — but **the product the proposal describes does not exist yet, and the
reports overstate what does.** The end-to-end SCF is a CPU-only, all-electron GTO/STO-3G toy with
O(n⁴) analytic ERIs — not the NAO + pseudopotential + grid pipeline of the proposal. **Zero GPU
code is called anywhere in the SCF loop.** The published "TIDES vs PySCF" benchmark ran a model
H₂-like Hamiltonian (Ne "energy" −3.4 Ha vs. the true −128.16 Ha) and its timing claims are
therefore void. Several tests pass only because their acceptance bars are set orders of magnitude
looser than the proposal's own criteria (worst case: NVE drift bar 20,000 µHa/atom/ps vs. the
stated budget of 30). None of this is unusual for a bootstrap phase — but per Rule 12, "51/51
PASS / release-ready" is the wrong summary of this state, and optimization effort is currently
flowing into a pipeline (GTO toy path) that the plan says must not be the product.

---

## A. Test-report claims vs. reality

| # | Claim (progress.txt / AUDIT-REPORT.md / bench JSON) | Reality | Evidence |
|---|---|---|---|
| A1 | "TIDES 0.25 ms vs PySCF 1.4 s" (He); benchmark JSON published | The Python `TidesCalculator` evaluates a **hard-coded model H₂-like Hamiltonian** even on the "native" backend (`_build_model_h`); `n_iterations: 0`; He = −2.28 Ha (true −2.8267), **Ne = −3.4 Ha (true −128.156)**. Every TIDES number in `bench/pyscf_benchmark_results.json` is physically meaningless; all speed comparisons from it are void. | `api/python/tides/core.py:96–118, 287–296, 325`; `bench/pyscf_benchmark_results.json` |
| A2 | "E6 XL-BOMD NVE: **ALL PASS**, drift 7762 µHa/atom/ps" | The proposal budget and the code's own header say **≤ 30 µHa/atom/ps**. The test bar is `drift < 20000.0` — 667× looser than the acceptance criterion, so the test cannot fail. Measured drift is **259× over budget**. "Short simulation inflates drift" is a hypothesis, not a verification — rerun longer instead of passing it. | `tests/per_engine/e6_dynamics/e6_test_profile.cpp:160`; `core/dynamics/xlbomd/xlbomd.hpp:29` |
| A3 | "Poisson GPU: **43,000× speedup**" | Baseline was a naive O(N²) direct-sum CPU DFT (progress.txt itself lists this as known issue #6). Speedup vs. a strawman. Honest baseline: FFTW CPU (now installed) — report that number instead; expect ~5–50×, not 43,000×. | `progress.txt` E3 + issue #6 |
| A4 | "Grouped GEMM 967 GFLOPS = **91.7% of cuBLASLt** — exceeds 90% target" | Ratio is real but both numbers are ~1–2% of RTX-class FP16 tensor-core peak (tens of TFLOP/s); the workload is tiny fixed tiles, and the WMMA kernel loads fragments straight from global memory with no shared-memory staging/double-buffering. The proposal's ≥90% target was meant against a realistic H-matrix tile mix; passing a ratio of two slow numbers hides the absolute gap. | `core/tile/gemm_grouped.cu:180–215`; `progress.txt` E1 |
| A5 | "XC-LDA GPU accuracy 6e-12 — rung-1 PASS vs FP64 oracle" | The GPU kernel is compared against a **CPU copy of the same formulas** (both use the same finite-difference PW92 derivative). This validates consistency, not correctness. The claimed oracle (libxc/PySCF pointwise) is not in the loop; the only pointwise libxc check is a handful of PW92 values. | `tests/per_engine/e3_grid/e3_test_profile.cpp:290–330`; `core/grid/xc.cu:41–115` |
| A6 | "RhoBuild/VmatBuild GPU 5× speedup at 48³" | Measured against naive O(n²·N_grid) CPU triple loops, with per-call `cudaMalloc` + PCIe transfers included on the GPU side, at toy size. Neither side is near roofline (WP3 target: ≥60% of HBM roofline — never measured). | `core/grid/rho_build.cu`, `vmat_build.cu`; `bench/profiling_ledger.json` |
| A7 | "E2 ALL PASS: two-center 3.5e-5, three-center symmetry 3.8e-3" | WP2 acceptance is **≤1e-8 Ha vs PySCF / ≤1e-10 relative** (rung 2). Current accuracy is 3–5 orders of magnitude short of the gate; a 3.8e-3 symmetry violation in an assembled operator is a correctness defect, not noise. Tests pass because bars were set at what the code achieves. | `progress.txt` E2 |
| A8 | "Rung 3 (Energy) 5e-9 PASS … full ladder release-ready" | No C++ test asserts a molecular SCF total energy against an external reference. The T3.5 observable — "He/Ne totals vs PySCF ≤ 1e-8 Ha" — is unmet end-to-end (only pointwise PW92 values are checked; the sole end-to-end external comparison is A1's broken benchmark). Rung 6 is SKIP. "Release-ready" is not a defensible summary. | `core/tests/physics_tests.cpp:291–311`; grep: no `-75.85`/`-2.8267` assertions anywhere |
| A9 | "SCF Pulay converges in 13–129 iters" | 129 iterations for tiny closed-shell systems signals weak mixing; but see B7/C1 — each iteration also costs ~3 Hamiltonian builds + 2 eigensolves, so the wall-time per iteration is inflated ~3× before mixing quality even matters. | `core/scf/scf_driver.hpp:84–118`; `molecule_driver.hpp:141–193` |
| A10 | "E1.2 Grouped GEMM **FP16-accum**" | The WMMA accumulator fragment is `float` (correct per the proposal!) — the test name, comment, and its tolerance model ("mantissa ~10 bits") describe FP16 accumulation that isn't happening. Mislabeled precision ledger entries poison the A/B audit trail the proposal depends on. | `core/tile/gemm_grouped.cu:209`; `tests/per_engine/e1_tile/e1_test_profile.cpp:187–230` |

## B. Correctness defects (file:line)

| # | Defect | Location |
|---|---|---|
| B1 | `XCEvalPbeCuda` = CPU libxc evaluation + GPU-only energy reduction, with **two PCIe round-trips** — the exact gpu4pyscf anti-pattern TIDES exists to eliminate. Additionally its returned `vxc` omits the −2∇·(v_σ∇ρ) GGA term (wrapper comment defers it "to the SCF loop"; no SCF code computes it). The PBE path is both slow and wrong; it is currently dead code (driver is LDA-only) — delete or fix, don't leave it. | `core/grid/xc.cu:288–370`; `core/grid/libxc_wrapper.hpp:185–189` |
| B2 | Device PW92 correlation potential uses **central finite differences** for dε_c/dr_s (h=1e-6). The analytic derivative is ~5 lines, exact, and 3× cheaper. FD noise (~1e-11) also contaminates the "oracle" comparisons in A5, since CPU and GPU share it. | `core/grid/xc.cu:67–75` |
| B3 | GPU `RhoBuildKernel` computes ρ = Σ_k f_k ψ_k² from **orbitals** while the CPU path computes ρ = Σ_ij P_ij φ_i φ_j from the **density matrix**. Different inputs, different formula; the GPU path needs eigenvectors, so it structurally cannot serve R2/R3 (purification produces P, not orbitals), and produces no ∇ρ (GGA impossible). | `core/grid/rho_build.cu:44–59` vs `core/grid/vmat_build.hpp:30–46` |
| B4 | Molecule driver sets **periodic BCs for isolated molecules**. Latent: harmless only because the driver bypasses the Poisson solver entirely (Hartree via analytic ERIs); the moment grid Hartree is enabled this silently gives wrong electrostatics. | `core/scf/molecule_driver.hpp:110–112` |
| B5 | `energy_fn` recomputes band energy by **re-diagonalizing H[P_new]** while the double-counting terms use P_new from H[P_old] — an inconsistent (P, H) pairing during the SCF path. Converges to the same fixed point, but the per-iteration energy history is not variational and misleads convergence diagnostics (and costs an extra eigensolve + full H build, see C1). | `core/scf/molecule_driver.hpp:163–189` |
| B6 | `eps_xc_mat`: builds a full n×n matrix via `BuildHmat` (O(n²·N_grid)) to obtain the scalar E_xc = Tr(P·M), when the identical quantity is the existing O(N_grid) grid dot product `XCGridEvaluator::XCEnergy`. Wasteful and a second code path for the same observable (the two can silently diverge). | `core/scf/molecule_driver.hpp:174,186`; `core/grid/xc.hpp:54–62` |
| B7 | Per SCF iteration the Hamiltonian is built **3×** (once in `build_H` from the SCF loop; `energy_fn` then rebuilds Coulomb+XC+matrices *and calls `build_H` again*), plus 2 eigensolves. ~3× wall-time inflation of every reported SCF timing. | `core/scf/scf_driver.hpp:88,117`; `molecule_driver.hpp:141–189` |
| B8 | `GTOIntegrals::CoulombMatrix`: 8-nested shell/component loops; **no 8-fold ERI permutational symmetry, no Schwarz screening, and no caching** — the ERI tensor is constant across SCF iterations but recomputed from scratch 3× per iteration (see B7). | `core/scf/gto_integrals.hpp:151–180` |
| B9 | STO-3G table covers H, He, C, N, O, F only (comment claims Li; no branch exists); Cartesian p-shells only, no d — silently returns an empty basis for any other element. Must be a loud `Status` failure, not empty shells. | `core/scf/molecule_driver.hpp:203–341` |
| B10 | Every GPU wrapper (`XCEvalLdaCuda`, rho/vmat/poisson GPU paths) does `cudaMalloc` → H2D → kernel → D2H → `cudaFree` → `cudaDeviceSynchronize` **per call**. No persistent device buffers, no streams, full-device stalls. This is the residency defect the whole XC design (Design doc §0) exists to fix. | `core/grid/xc.cu:203–265` and analogous |

## C. Architectural deviations from the proposal (surface, don't average — Rule 7)

| # | Proposal | Code today | Consequence |
|---|---|---|---|
| C1 | NAO basis + ONCV pseudopotentials + grid Hartree (Poisson) + libxc XC; one shared Hamiltonian builder in `core/ham/` | End-to-end SCF is all-electron **GTO/STO-3G** with analytic ERIs; **`core/ham/` is an empty directory**; Poisson and NAO modules exist but are orphaned from the SCF product | The pipeline being profiled and optimized is not the product. Optimization effort (dsyrk, DIIS tuning) is flowing into a path the plan says must be replaced. Decision needed: either bless the GTO driver *explicitly* as a bootstrap oracle (fine!) or freeze it. It must not be the benchmark subject. |
| C2 | Tile substrate (CSR-of-tiles, grouped GEMM) is "the one true layer" every regime runs on | Grid engine GPU kernels are hand-written dense loops over `[n_orb][N_grid]` arrays; WP1 TileMat is not used by grid, SCF, or solvers in the product path | The central architectural premise is unintegrated; E1's GFLOPS have no consumer. rho_build/vmat_build must become tile-batched GEMM (this is also the fix for A6/B3). |
| C3 | SCF/XC/grid all GPU-resident; "one scalar back to CPU per SCF iteration" | **`core/scf/` contains zero CUDA**; all GPU kernels are standalone demos; XC in SCF = CPU `XCGridEvaluator::EvaluateLDA` | The user-visible product is CPU-only. The 20× XC win and everything in the Design doc applies to a pipeline that still needs to be assembled. |
| C4 | libxc as the XC production path from Y1 (proposal §3.1); PBE in Phase A | SCF hard-codes LDA-PW92; PBE exists only as the broken B1 path; no spin polarization in the driver | XC coverage gate for Y1Q3 ("LDA/GGA SCF for molecules") is not met even on CPU. |
| C5 | Solver broker dispatches every calculation | Molecule driver calls `BatchedDenseEig` directly | Broker exists (E4/nanobind) but is decorative in the product path. |
| C6 | Forces: Hellmann–Feynman + Pulay, one code path, FD-validated nightly | Molecule driver computes no forces; force machinery (E6) validated only on model potentials | Rung-4 "PASS" does not cover the real SCF. |
| C7 | Python API: nanobind exposes the real engine; ASE calculator runs DFT | `_native.cpp` exposes Status/broker/forces wrappers only — **no MoleculeDriver binding**; `TidesCalculator` falls back to the model Hamiltonian in both backends | A1 follows directly. Wiring `MoleculeDriver` through nanobind is ~50 lines and would make every Python benchmark honest. |
| C8 | Dual grid (coarse orbital / fine density) | Single uniform grid everywhere in the driver | Acceptable for bootstrap; flag so it's a decision, not drift. |

## D. Performance bottlenecks, ranked by impact on the current product path

1. **3× Hamiltonian rebuild + 2× eigensolve per SCF iteration** (B5/B6/B7) — restructure
   `SCFDriver` so `build_H` returns H once per iteration and the energy is assembled from the
   *same* (P, H) pair; compute E_xc on the grid. Immediate ~3× on every SCF timing, zero risk.
2. **Uncached, unscreened, symmetry-blind ERIs** (B8) — cache the ERI tensor across iterations
   (it is geometry-constant): turns 3 O(n⁴) evaluations/iter into 1 O(n⁴) at setup + O(n⁴)
   contraction… then add 8-fold symmetry (8×) and Schwarz screening. Only worth doing at all if
   the GTO driver is blessed as the bootstrap oracle (C1 decision).
3. **CPU dense O(n²·N_grid) rho/vmat with dense per-orbital grid storage** (C2) — replace with
   the tile-batched GEMM formulation on GPU (this is the Design doc's Kernel A/C; it is also the
   only route to the proposal's ≥60% roofline target).
4. **Per-call GPU alloc/transfer/sync** (B10) — arena + streams + persistence (Design doc §4).
5. **XC on CPU inside SCF** (C3/C4) — the Tier-0 fused engine (Design doc §2, execution plan in
   `TIDES_XC_Tier0_Execution_Plan.md`).
6. **WMMA kernel without shared-memory staging** (A4) — either adopt CUTLASS grouped GEMM as the
   proposal specifies (recommended; stop hand-writing what CUTLASS does better) or stage
   fragments through shared memory with double buffering.

## E. Test-suite quality audit (Rule 9: tests must encode intent)

- **Bars must come from `verification/tolerances.yaml`, not from the achieved value.** Violations
  found: A2 (drift 20000 vs 30), A7 (integral accuracy vs rung-2 gates), A4 (ratio-to-slow-baseline
  vs roofline intent). Action: single source of truth for every threshold; a test that needs a
  looser bar than the proposal must carry a written waiver in the test file.
- **Self-consistency masquerading as validation:** A5 (GPU vs CPU same formula), A8 (no external
  energy reference anywhere in ctest). Action: pointwise XC oracle = libxc sweep (rung 0 of the
  XC plan); end-to-end oracle = pinned PySCF reference values asserted in C++ tests with matched
  basis/grid (STO-3G LDA references are computable in PySCF in seconds — match the basis, then
  1e-8 Ha is a fair gate).
- **Benchmarks must refuse to run on stubs:** the PySCF comparison script should hard-fail when
  `backend != native-full-scf` instead of publishing model-Hamiltonian numbers labeled "TIDES".
- **Mislabeled precision entries** (A10) undermine the OperationLedger's purpose. Ledger labels
  must be generated from the actual template parameters, not hand-typed strings.

## F. Priority-ordered fix plan

**P0 — truth in reporting (days):**
1. Mark `bench/pyscf_benchmark_results.json` and derived claims as INVALID in `progress.txt` /
   `AUDIT-REPORT.md`; regenerate only after C7 wiring. Benchmarks hard-fail on stub backends.
2. Retighten test bars to `tolerances.yaml` values (drift 30, rung-2 integrals); let the tests
   FAIL and track the failures as open defects — red tests that tell the truth beat green tests
   that don't (Rule 12).
3. Fix A10 ledger labels; report Poisson speedup vs FFTW.

**P1 — make the real pipeline exist (1–2 weeks):**
4. Decision (project owner): bless GTO driver as *bootstrap oracle only* (recommended) — then
   apply the cheap fixes B5/B6/B7 (+ B8 ERI caching) so it's a usable reference, and freeze it.
5. Wire `MoleculeDriver` through nanobind (C7); regenerate the PySCF benchmark honestly (CPU vs
   CPU, matched STO-3G basis, energies asserted to ≤1e-6 Ha as a start).
6. Assemble the **NAO product driver**: NAO basis (E2) + grid V_ext/Hartree (E3 Poisson, free-BC)
   + XC + `SCFDriver` — the pipeline the proposal actually describes, CPU-first, validated
   against the GTO oracle and PySCF.

**P2 — GPU residency + XC Tier 0 (the Design doc's M0/M1, ~4 weeks):**
7. B10 arena/streams; B2 analytic derivative; delete/fix B1; then the fused Tier-0 XC engine per
   `TIDES_XC_Tier0_Execution_Plan.md` (LDA+PBE first), integrated into the P1.6 driver.
8. rho_build/vmat_build as tile-batched GEMM on the WP1 substrate (C2) with ∇ρ output (unblocks
   GGA end-to-end and the ≥60% roofline measurement).

**P3 — then and only then, performance claims:** rooflines on the real pipeline, XL-BOMD drift
rerun at ≥10 ps, competitor benchmarks per §9 of the proposal.

---
*Method note: every finding above was verified by direct code inspection at the cited lines; test
and benchmark claims were checked against `progress.txt`, `AUDIT-REPORT.md`,
`bench/profiling_ledger.json`, and `bench/pyscf_benchmark_results.json`. Items I could not fully
verify in this pass (ChFSI/OMM/ISDF "fixed" claims, HIP status, MPI scaling) are explicitly out of
scope and should be re-audited when they enter the product path.*
