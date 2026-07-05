# TIDES — TIle-based Democratic Electronic-Structure suite
### A 5-year plan for the fastest open-source Kohn–Sham DFT engine across the full system-size spectrum (10 → 10⁶ atoms)
*Working title; proposal v1.0 — July 2026. Prepared for a 10-scientist collaborative team.*

---

## 1. Thesis and the "middle way"

No single algorithm is fastest at both 10 atoms and 10⁶ atoms — that is a mathematical fact, not an engineering failure (dense linear algebra wins small; sparsity/nearsightedness wins large; the crossover is real). The middle way is therefore **not one algorithm but one substrate**:

> **Every regime of DFT — molecular, slab, bulk, insulating, metallic, 10 atoms or 10⁶ — is reduced to the same primitive: batched dense operations on small fixed-size matrix tiles, executed in mixed precision on GPU tensor cores, with FP64 recovered where needed by Ozaki-scheme emulation.**

On top of that single substrate sits a **solver broker** that dispatches each problem to the asymptotically optimal algorithm for its regime, while every algorithm shares the same Hamiltonian builder, grid engine, force engine, I/O, and test harness. One code, one representation, one force expression, one input format — different asymptotics per regime. This is how "fastest across the spectrum" becomes a coherent engineering target instead of a contradiction.

The representation that makes this possible is the **numeric atom-centered orbital (NAO)** basis with a dual real-space grid:
- Compact enough (~13–30 functions/atom at DZP–TZP) to be competitive with Gaussian codes at the small end.
- Strictly localized, so H, S, and the density matrix are block-sparse → tiles → tensor cores → linear scaling at the large end.
- Works for molecules, wires, slabs, and crystals with all boundary conditions; all elements.
- Proven lineage across the entire spectrum (FHI-aims, SIESTA, CONQUEST, OpenMX, ABACUS) — we are not inventing an unproven discretization; we are giving the proven one a 2026-native execution model that none of them has.

### 1.1 The four differentiators (what nobody has combined)

1. **Mixed-precision-native tile execution with Ozaki-scheme FP64 emulation.** GPUs are abandoning fast native FP64 (Blackwell-class hardware, FP8-centric roadmaps); FP64-equivalent GEMM via error-free transformations on low-precision tensor cores is now production-grade and can *exceed* native DGEMM (≈2.3× on GB200-class parts; DOE "MFP64" endorsement, 2025). No DFT code is designed around this. We will be.
2. **XL-BOMD shadow dynamics as the default MD engine.** Extended-Lagrangian Born–Oppenheimer MD (Niklasson lineage) removes the SCF loop: one Hamiltonian build + one density-matrix solve per timestep, time-reversible, with kernel/Krylov stabilization for small-gap systems, already demonstrated on tensor cores and in graph-based linear-scaling form. This — not wishful architecture — is what makes "DFTB-comparable MD throughput at DFT accuracy" an honest sentence.
3. **Batched many-system execution at the small end.** At <200 atoms a single GPU is idle 95% of the time; the meaningful small-end metric is throughput (screening, batched MD, batched Hessians), where running 10³–10⁴ molecules concurrently on one device beats any per-molecule-latency champion by an order of magnitude.
4. **Certified accuracy per joule.** Every benchmark is reported as time-to-fixed-accuracy (meV/atom against converged references) plus measured energy (kWh), with a public reproducibility archive. Speed claims without an accuracy contract are marketing; ours carry one.

Plus one **high-risk/high-reward research thrust** (flag-gated, with go/no-go reviews): **quantics tensor-train (QTT) compression** of the density/Hartree pipeline — the genuinely new mathematics (tensorized orbitals, Jolly–Núñez-Fernández–Waintal, PRB 111, 2025) that provides multiscale adaptivity as dense, regular, GPU-friendly linear algebra instead of irregular adaptive meshes.

### 1.2 Honest claims table (what we promise, what we don't)

| Regime | Claim (target, at fixed accuracy) | What we do NOT claim |
|---|---|---|
| 10–200 atoms, single molecule, cold start | Within ~2× of the fastest GPU Gaussian codes (GPU4PySCF / xQC-class) on B3LYP/def2-TZVPP-equivalent single points | Beating specialized analytic-integral codes on a single cold 10-atom single point (that fight is launch-latency-bound and scientifically uninteresting) |
| 10–200 atoms, throughput (screening/batched) | ≥10× the per-GPU molecules/hour of current GPU Gaussian codes via native batching | — |
| 10–500 atoms, MD trajectories | Within 5–20× of DFTB+ step rate at DZP-PBE quality via XL-BOMD (today's gap: 10²–10³×) | Literal DFTB step rates at full DFT accuracy — that gap is definitional (DFTB pre-tabulates exactly what DFT computes) |
| 10³–10⁴ atoms | Fastest open-source time-to-solution at ≤1 meV/atom on a single GPU; single-GPU capability to ~10⁵ atoms (gapped, DZP, mixed precision) | — |
| 10⁵–10⁶ atoms | 10⁶-atom gapped systems on 8–32 GPUs (one modern node to a small partition); metallic 10⁵–10⁶ at finite Tₑ via Fermi-operator/spectral-quadrature methods | 10⁶ atoms on ONE GPU — impossible on 2026 memory (see §2.2), for us and everyone |
| Accuracy | Systematically improvable to the basis-set limit (DZP→TZP→TZDP+diffuse; PAW thrust Y3); ACWF/Δ-verification reported at every release | Beyond-DFT accuracy; ML shortcuts (excluded by design constraint) |

---

## 2. Design constraints and the physics/hardware limits we respect

### 2.1 Constraints (from the project owner)
- Full Kohn–Sham DFT accuracy: no tight-binding parameterization, no ML potentials, no ML exchange–correlation. (Hooks are kept so *others* can plug ML XC in — the core never depends on it. Open ≠ opinionated.)
- One uniform open-source code across the spectrum: no separate "small code" and "large code".
- GPU-native, and *democratic*: first-class support for consumer GPUs (RTX-class), which is precisely what a mixed-precision-native design enables — gaming cards have crippled FP64 but monstrous tensor cores; Ozaki emulation turns them into research instruments.
- No `try`/`except` control flow in the Python layer: explicit status objects and assertions instead (team coding standard, §8.5).

### 2.2 Hard limits we will state out loud (so no reviewer states them for us)
**Memory at 10⁶ atoms.** DZP (n_b≈15/atom, light elements), density-matrix truncation radius giving m≈200 neighbor blocks/atom, FP16 tile storage: per atom per matrix ≈ m·n_b²·2 B ≈ 90 KB. With S, H, P, two XL-BOMD auxiliaries, and workspace (~6 matrix-equivalents): ≈ 0.5 MB/atom ⇒ **~0.5 TB for 10⁶ atoms**, plus ~0.1 TB of real-space grids. That is 8× GH200/H200-class devices or a GB200 partition — not one GPU. Anyone claiming 10⁶-atom self-consistent DFT on a single 2026 GPU is describing a different problem. Corollary: **~10⁵ atoms on ONE GPU** (~50–70 GB) is the achievable single-device headline, and it is enough to lead the field.

**Metals.** T=0 metals have no nearsightedness; linear scaling for metals is bought with finite electronic temperature (Mermin functional). We adopt that trade openly: the metallic large-N path (R3, §4) is exact Mermin-DFT at the user's Tₑ, with polynomial order growing as ~β·(spectral width) — the same physics SPARC's Spectral Quadrature uses to reach million-atom systems.

**The DFTB gap is definitional.** DFTB is 10²–10³× faster than DFT because it replaces integral evaluation and self-consistent XC with lookup tables. We close the gap by removing *redundant* work (SCF loops via XL-BOMD, FP64 via mixed precision, idle GPU via batching, cubic solves via purification) — the honest end state is "within one order of magnitude of DFTB for MD", not parity.

---

## 3. The four attack angles

### 3.1 Physics
1. **NAO basis, systematically improvable.** On-the-fly generated confined-atom orbitals (DZP default; TZP/TZDP + diffuse augmentation for benchmarks and anions/surfaces), ONCV pseudopotentials from PseudoDojo (shared with SPARC/QE/ABACUS/GPAW ⇒ apples-to-apples benchmarking). Basis-set superposition handled with counterpoise tooling; PAW-in-FE-style multi-resolution augmentation is a Y3 decision informed by the 2026 PAW-FE results (arXiv:2604.26037) showing large DOF reductions.
2. **Finite electronic temperature as a first-class citizen** (Mermin free energy, fractional occupations everywhere) — required for metals at scale, for XL-BOMD stability at small gaps, and for warm-dense-matter reach.
3. **Hybrid functionals for surface chemistry** via ISDF (interpolative separable density fitting) + ACE (adaptively compressed exchange): HSE06/PBE0 at ≤3–4× the PBE cost per SCF-equivalent, because surface chemistry at GGA level is often not publishable-grade.
4. **Dispersion**: DFT-D3/D4 (analytic, cheap, open implementations) wired into energies/forces/stress from Y1.
5. **Symmetry**: point-group symmetrization for molecules; Γ-supercell strategy for large systems; Monkhorst–Pack k-sampling for ≤~2,000-atom periodic cells (complex tiles). Cyclic/helical symmetry adaptation (Cyclix-style) is an optional Y4 stretch for nanotubes/2D bending — it buys orders of magnitude for that class and SPARC has shown the math works.

### 3.2 Mathematics
1. **Tile algebra as the universal object.** Block-sparse matrices stored as CSR-of-tiles with micro-tile edge ∈ {16, 32, 64} (padded), per-tile block scaling (MX-style) for dynamic range. Every heavy operation — SpGEMM, purification polynomials, subspace projections, grid↔orbital maps — is a grouped GEMM stream.
2. **Precision as an algorithmic variable with an error budget.** Storage BF16/FP16 (+per-tile scale) → accumulate FP32 → critical reductions (traces, total energies, level shifts, Fermi-level search) in FP64 obtained by Ozaki-scheme emulation on tensor cores (FP8/FP16-based variants, since INT8 paths are being phased out on 2025+ hardware). A written perturbation model links tile precision ε and gap g to ΔE/atom; nightly A/B (mixed vs full FP64) enforces ≤0.5 meV/atom drift. This is the single largest "free" hardware tailwind of 2026–2031 and no incumbent is built for it.
3. **Density-matrix purification via the submatrix method** for the gapped large-N regime: transform the sparse sign/SP2 iteration into many *small dense* per-atom-neighborhood problems — exactly the shape tensor cores want, exactly the method behind the 100-million-atom NOLSM demonstrations (CP2K lineage), here upgraded from tight-binding-quality to full-DFT Hamiltonians with error compensation.
4. **Fermi-operator expansion / Spectral Quadrature** (Chebyshev/Gauss quadrature of the Fermi function) for the metallic large-N regime — diagonalization-free, same tile substrate, polynomial order set by β·ΔH.
5. **Chebyshev-filtered subspace iteration (ChFSI) + dense eigensolver bridge** (cuSOLVERMp / ELPA-GPU) for the mid-range, where cubic-but-small is simply fastest.
6. **A-posteriori error control** (DFTK-style residual bounds adapted to NAO): every result can carry a certified accuracy estimate — the scientific differentiator that turns "fast" into "trustworthy fast".
7. **QTT research thrust**: quantics tensor-train representation of ρ(r), v_H(r) and the Poisson solve on 2ⁿ virtual grids; adaptivity lives in bond dimensions (dense algebra), not meshes. Gate criteria in §10.

### 3.3 Algorithms
1. **Solver broker** (regime dispatch): inputs (N, boundary conditions, estimated gap from a cheap pre-pass, Tₑ, available VRAM) → route to R0 batch-dense / R1 dense-ChFSI / R2 SP2-submatrix / R3 FOE-SQ. One input file; the broker is overridable but never required.
2. **XL-BOMD everywhere**: shadow-potential dynamics with Krylov/kernel integrators (KSA-XL-BOMD) → no SCF loop in production MD; ~1 solve/step; validated NVE drift budget.
3. **Warm-start discipline**: ASPC density/DM extrapolation across MD/relaxation steps; subspace reuse in ChFSI; geometry-optimizer coupling.
4. **O(N) Hamiltonian construction**: two-center integrals from splined radial tables (S, T, V_nl in Kleinman–Bylander form); three-center (local potential) via grid integration with cell tasking; sparsity from basis cutoffs.
5. **Communication**: atom/tile domain decomposition via graph partitioning (METIS/KaHIP); NCCL/NVSHMEM one-sided halo exchange; computation–communication overlap; Γ-point large systems avoid all-to-alls except the (coarse-grid) Poisson FFT, which is replaceable by the ISF/QTT path.
6. **Batched-systems mode (R0)**: structure-of-arrays over systems; one CUDA graph per SCF sweep for thousands of small molecules; batched syev for the per-system diagonalizations.

### 3.4 Computational language & engineering
1. **Core: modern C++20 + CUDA**, with a thin device-abstraction layer and HIP builds for AMD (portability validated quarterly on MI300-class); kernels built on CUTLASS/cuBLASLt grouped GEMM; a small number (≤10) of hand-written hot kernels (grid↔orbital scatter/gather, tile SpGEMM filtering, submatrix assembly). **Not** whole-code JAX: XLA's static-shape, dense-array model fits neither block-sparsity nor the broker (established in our landscape review); we take autodiff frameworks' *lessons* (pure functions, explicit state, graph capture) without their constraints.
2. **Python API via nanobind** + an ASE calculator; optional **JAX bridge** exposing `energy_and_forces(R) → (E, F)` as a custom call with analytic-gradient VJP, so differentiable-workflow users get TIDES as a primitive without TIDES being written in JAX.
3. **Engineering discipline**: reproducible builds (CMake + Spack recipe + container), deterministic mode (ordered reductions) for debugging, GPU CI on both a datacenter card and a consumer RTX card (the "democratic" gate), performance-regression CI with a tracked dashboard, every kernel unit-tested against an FP64 CPU oracle with ULP budgets.
4. **Python layer style**: explicit `Status`/result objects and assertions; no `try`/`except` control flow (team standard, matching the project owner's requirement); errors surface as typed status codes from the C++ core.

---

## 4. Architecture overview

```
                            ┌─────────────────────────────────────────────┐
                            │              Python API (nanobind)          │
                            │   ASE calculator · CLI · optional JAX bridge│
                            └──────────────────────┬──────────────────────┘
                                                   │
                     ┌─────────────────────────────┴─────────────────────────────┐
                     │                       SOLVER BROKER                       │
                     │  inputs: N, BCs, gap estimate, T_e, VRAM, user override   │
                     └──┬───────────────┬───────────────┬───────────────┬────────┘
                        │               │               │               │
              ┌─────────▼───┐  ┌────────▼──────┐  ┌─────▼─────────┐  ┌──▼───────────┐
              │ R0 BATCH    │  │ R1 DENSE      │  │ R2 SPARSE-GAP │  │ R3 SPARSE-   │
              │ ≤200 atoms  │  │ ≤~2k atoms    │  │ 2k–10^6 atoms │  │ METALLIC     │
              │ 10^3–10^4   │  │ ChFSI +       │  │ SP2 purificat.│  │ FOE / SQ,    │
              │ systems/GPU │  │ cuSOLVERMp/   │  │ via submatrix │  │ Mermin T_e   │
              │ batched syev│  │ ELPA bridge   │  │ method        │  │ Chebyshev    │
              └─────────┬───┘  └────────┬──────┘  └─────┬─────────┘  └──┬───────────┘
                        └───────────────┴───────┬───────┴────────────────┘
                                                │  all regimes share ↓
      ┌────────────────────────────────────────┬┴───────────────────────────────────────┐
      │ HAMILTONIAN & FORCES (shared)          │ MD OVERLAY (shared)                     │
      │ NAO integrals · dual-grid ρ/V · Poisson│ XL-BOMD shadow dynamics (KSA kernel),   │
      │ (FFT/ISF, QTT research) · libxc ·      │ thermostats, ASPC warm starts,          │
      │ hybrids (ISDF+ACE) · D3/D4 · stress    │ geometry opt / NEB                      │
      ├────────────────────────────────────────┴────────────────────────────────────────┤
      │                    TILE SUBSTRATE (the one true layer)                           │
      │  CSR-of-tiles block-sparse · grouped GEMM (CUTLASS/cuBLASLt) · tile SpGEMM ·     │
      │  per-tile scaling · BF16/FP16 store, FP32 accumulate, FP64-emulated reductions   │
      │  (Ozaki) · CUDA graphs · deterministic mode · NCCL/NVSHMEM halos                 │
      └──────────────────────────────────────────────────────────────────────────────────┘
```

Regime boundaries are defaults, not walls: the broker's crossover points (atom counts, gap thresholds) are measured on the actual hardware during install (`tides tune`) and cached, so "fastest across the spectrum" is enforced empirically per machine, not assumed.

---

## 5. Codebase structure (monorepo)

```
tides/
├── CMakeLists.txt · LICENSE (Apache-2.0) · CITATION.cff · CONTRIBUTING.md · GOVERNANCE.md
├── cmake/                        # toolchains: cuda.cmake, hip.cmake; sanitizer & LTO presets
├── external/                     # pinned submodules: libxc, CUTLASS, nanobind, METIS, spglib,
│                                 #   simple-dftd3, dftd4, HDF5 (find-package), GoogleTest
├── core/                         # C++20; zero Python dependencies
│   ├── common/
│   │   ├── status.hpp            # typed Status codes (no exceptions across API boundary)
│   │   ├── units.hpp             # CODATA-2022; single source of truth
│   │   ├── config.hpp            # input schema (TOML), validated; every key documented
│   │   └── logging.hpp           # structured logs (JSON lines) → benchmark harness reads these
│   ├── tile/                     # WP1 — THE substrate
│   │   ├── layout.hpp            # CSR-of-tiles; micro-tiles 16/32/64; per-tile scale factors
│   │   ├── gemm_grouped.cu       # CUTLASS grouped GEMM: bf16/fp16/tf32 paths + fp32 accum
│   │   ├── spgemm_filtered.cu    # tile SpGEMM with on-the-fly norm filtering (ε_filter)
│   │   ├── ozaki.cu              # FP64-equivalent GEMM/dot via error-free slicing (FP16/FP8)
│   │   ├── reduce_f64e.cu        # deterministic FP64-emulated reductions (traces, norms)
│   │   ├── graphs.hpp            # CUDA-graph capture of solver sweeps
│   │   └── tests/                # ULP budgets vs FP64 CPU oracle; adversarial spectra
│   ├── basis/                    # WP2
│   │   ├── atomgen/              # confined-atom radial solver (log grid, spectral); basis optimizer
│   │   ├── two_center.cu         # splined S, T, V_nl(KB) pair tables → tiles
│   │   ├── three_center.cu       # ⟨φ|v_loc+v_H+v_xc|φ⟩ via grid, cell-tasked, O(N)
│   │   ├── pseudo/               # ONCV(PseudoDojo) readers: UPF2/PSML; sanity validators
│   │   └── paw/                  # Y3 thrust (flag-gated)
│   ├── grid/                     # WP3
│   │   ├── dual_grid.hpp         # coarse(orbital)/fine(density) grids; domain decomposition
│   │   ├── rho_build.cu          # P·φφ → n(r): tile-batched, the #1 memory-bound kernel
│   │   ├── vmat_build.cu         # adjoint map v(r) → H tiles
│   │   ├── poisson_fft.cu        # cuFFT/ hipFFT; ISF kernels for free/wire/slab BCs
│   │   ├── poisson_qtt/          # WP-R (research, flag-gated): quantics TT Poisson + ρ compression
│   │   └── xc.cu                 # libxc batched LDA/GGA/mGGA; stress terms
│   ├── ham/                      # assembly orchestration: S, H(ρ), gradients wrt R (WP2+WP3)
│   ├── solvers/
│   │   ├── broker.cpp            # regime dispatch + `tides tune` calibration tables
│   │   ├── dense/                # batched syev (R0); cuSOLVERMp / ELPA bridge (R1)
│   │   ├── chfsi/                # Chebyshev filter, RR projection, subspace reuse (R1)
│   │   ├── omm/                  # orbital-minimization direct solver (insulators; optional R2 alt)
│   │   ├── sp2_submatrix/        # WP5: NOLSM-style purification; error compensation; halo build
│   │   └── foe_sq/               # WP5: Fermi-operator expansion / spectral quadrature (R3)
│   ├── scf/                      # mixers (Pulay/Kerker/Broyden), smearing, Fermi search (f64e)
│   ├── dynamics/                 # WP6: xlbomd/ (KSA kernel, thermostats), md_driver, optimizers, NEB
│   ├── forces/                   # Hellmann–Feynman + Pulay + grid + D3/D4 + stress; one code path
│   ├── hybrids/                  # WP7: isdf/ (randomized sampling), ace/, short-range HSE tiles
│   └── parallel/                 # WP8: partitioner (METIS), halos (NCCL/NVSHMEM), HDF5 I/O, restart
├── api/
│   ├── python/                   # nanobind bindings; ase_calculator.py; result/Status objects
│   ├── jax_bridge/               # custom_call: energy_and_forces with analytic-grad VJP
│   └── cli/                      # `tides run`, `tides tune`, `tides bench`, `tides verify`
├── verification/                 # WP9 — correctness, not speed
│   ├── references/               # ACWF/Δ curated subset, S22, W4-11 subset, surface DB (with DOIs)
│   ├── tolerances.yaml           # single source of truth for every acceptance threshold
│   └── runners/                  # nightly: mixed-vs-FP64 A/B, force FD checks, NVE drift
├── benchmarks/                   # WP9 — speed, always at fixed accuracy
│   ├── piecewise/                # per-module harnesses: vs_abacus/ vs_cp2k/ vs_sparc/ vs_dftfe/
│   │                             #   vs_gpaw/ vs_gpu4pyscf/ vs_bigdft_psolver/ vs_ntpoly/ vs_dftbplus/
│   ├── end2end/                  # campaign definitions (systems, protocols, repeats)
│   ├── energy_meter/             # NVML / rocm-smi kWh logging, bundled into every report
│   └── report/                   # sqlite regression DB, auto-plots, nightly diff alarms
├── ci/                           # GitHub Actions + self-hosted runners: 1 datacenter GPU + 1 RTX
├── docs/                         # Sphinx: theory manual WITH derivations; developer guide; tutorials
└── examples/                     # scripts (Python, no try/except); each doubles as an integration test
```

---

## 6. Module specifications (contracts, owners, acceptance)

Each work package (WP) below lists: purpose → key algorithms/refs → interface contract → precision policy → acceptance tests → piecewise competitor → owner → effort (person-months over 5 y, of ~55 PM/person available).

**WP1 · Tile substrate & precision (S1).** Purpose: the one layer everything runs on. Algorithms: CSR-of-tiles; CUTLASS grouped GEMM; filtered tile SpGEMM; Ozaki-scheme FP64-equivalent GEMM/reductions (FP16/FP8-slice variants); CUDA-graph capture. Contract: `TileMat` create/convert/spgemm/axpy/trace/norm with explicit precision descriptors; deterministic mode. Acceptance: ≥90% of cuBLASLt grouped-GEMM throughput on tile-size mix drawn from real H matrices; FP64-emulated trace matches FP64 to ≤1e-13 relative on adversarial spectra; zero race reports under compute-sanitizer. Competitor probe: DBCSR (CP2K) and NTPoly on identical sparsity patterns. Effort: 45 PM.

**WP2 · Basis & integrals (S2).** Purpose: NAO generation and S/T/V_nl/3-center assembly. Algorithms: confined-atom spectral radial solver; spline pair tables; KB nonlocal; O(N) cell-tasked grid integration. Contract: `build_S/H0(R) → TileMat`, `dH/dR` streams for forces. Precision: tables FP64; assembled tiles BF16+scale with FP32 path for validation. Acceptance: overlap/kinetic vs PySCF (matched STO-tabulated basis) ≤1e-8 Ha; vs ABACUS same-basis ≤1e-10 relative; force ingredients pass 5-point FD at 1e-6 Ha/Bohr (FP32 path). Competitor: ABACUS GPU-NAO builder (arXiv:2409.09399), SIESTA. Effort: 50 PM.

**WP3 · Grids, Poisson, XC (S3; QTT rotation with S5).** Purpose: ρ(r) build, v(r)→H map, electrostatics for all BCs, XC. Algorithms: dual grid; cuFFT + interpolating-scaling-function kernels (free/wire/slab); libxc; QTT Poisson/ρ (research). Contract: `rho(P)→grid`, `vmat(grid)→TileMat`, `hartree(rho, BC)→grid`. Acceptance: Poisson vs analytic Gaussians ≤1e-10 Ha; ρ-build hits ≥60% HBM roofline; egg-box force error <1e-4 Ha/Bohr at production grid spacing. Competitor: BigDFT PSolver (BC-flexible electrostatics), GPAW/SPARC grid ops. Effort: 50 PM.

**WP4 · Mid-range solvers (S4).** Purpose: R0/R1. Algorithms: batched syevj (R0); ChFSI with subspace reuse; bridge to cuSOLVERMp and ELPA-GPU; OMM direct minimization option. Acceptance: 5,000-atom Mo (ONCV, PBE, Mermin smearing) time/SCF within 1.5× of DFT-FE at matched ≤1 meV/atom; R0 sustains ≥10⁴ single-points/hour/GPU on a 30-atom organic set. Competitor: DFT-FE ChFSI, SPARC CheFSI, ELPA. Effort: 45 PM.

**WP5 · Linear-scaling solvers (S5).** Purpose: R2/R3. Algorithms: SP2 purification through the submatrix method (per-atom-neighborhood dense problems, batched); truncation error compensation; FOE/SQ with Chebyshev order control; Fermi-level search in f64e. Acceptance: 10⁴-atom a-Si:H on one GPU, ≤0.5 meV/atom vs R1 reference on a 2,000-atom control; O(N^1.0±0.1) measured 10⁴→10⁶; metallic Al at Tₑ=3000 K matches R1 free energy ≤1 meV/atom on control. Competitor: CP2K NOLSM/DBCSR, NTPoly, SPARC-SQ, CheSS. Effort: 55 PM.

**WP6 · SCF, XL-BOMD, forces, dynamics (S6).** Purpose: the production loop. Algorithms: Pulay/Kerker mixing (SCF mode); KSA-XL-BOMD shadow dynamics with fractional occupations (default MD mode); ASPC warm starts; FIRE/L-BFGS optimizers; NEB. Acceptance: NVE 64-H₂O, 0.5 fs, 100 ps: energy drift ≤30 μHa/atom/ps with ~1 solve/step; forces vs FD ≤1e-6 (FP32 path) / ≤1e-4 (production mixed) Ha/Bohr; XL-BOMD vs converged-SCF trajectories: pair-correlation functions statistically indistinguishable. Competitor: CP2K MD, GPAW; anchors DFTB+ and tblite (speed only). Effort: 50 PM.

**WP7 · Hybrids, dispersion, PAW thrust (S7).** Algorithms: ISDF (randomized column sampling) + ACE; short-range HSE tiles; simple-dftd3/dftd4; Y3 PAW-in-NAO decision. Acceptance: 500-atom TiO₂ slab HSE06 ≤4× own-PBE time/SCF at matched accuracy; S22 with TZP+D4 MAD ≤0.35 kcal/mol (counterpoise-corrected). Competitor: CP2K ADMM, QE ACE. Effort: 45 PM.

**WP8 · Parallel, HPC, I/O (S8).** Algorithms: METIS tile partitioning; NCCL/NVSHMEM halos with overlap; multi-GPU Poisson; HDF5 restarts; container/Spack packaging. Acceptance: weak scaling ≥80% to 64 GPUs on 10⁵→10⁶ a-Si; strong scaling ≥60% 8→64 GPUs at 10⁵ atoms; HIP build passes full verification on MI300-class quarterly. Effort: 50 PM.

**WP9 · Verification, benchmarking, CI — the red team (S9).** Owns `verification/`, `benchmarks/`, `ci/`, `tolerances.yaml`; runs nightly mixed-vs-FP64 A/B, FD force checks, NVE drift, ACWF subset; maintains competitor builds and the regression dashboard; has veto on releases. Effort: 50 PM.

**WP10 · API, docs, community, releases (S10).** nanobind API, ASE calculator, JAX bridge, CLI, theory manual with derivations, tutorials, release engineering, community calls, RFC process. Acceptance: a new user reproduces the S22 benchmark from docs alone in <1 hour; API-frozen surface from Y5Q1. Effort: 45 PM.

**WP-R · QTT research thrust (S3+S5 at 20%, flag-gated).** Quantics-TT compression of ρ and the Hartree solve (tensor cross interpolation; multiscale/patched QTT). Gates in §10. Effort: 25 PM inside WP3/WP5 budgets.

---

## 7. Analytical performance model (design targets, to be validated at M-gates)

**Assumption ledger.** DZP light-element basis n_b=15 (TM: 25); fine grid h=0.15 Å ⇒ g_f≈6,000 points/atom (Si density); density-matrix block neighbors m: 60 (molecular), 200 (bulk gapped), 300 (metallic FOE); SP2 iterations 30–40; SpGEMM fill×filter factor f=0.3; effective sustained tensor-core rate 200–300 TFLOP/s (H100-class, tile mix, FP16/BF16-store); HBM sustained 2.5–3 TB/s; all figures per GPU.

**Grid phase (memory-bound).** Traffic ≈ 20 passes × g_f × 4 B ≈ 0.5 MB/atom/build ⇒ ~0.2 µs/atom ⇒ 10⁵ atoms ≈ 20 ms/build. Never the bottleneck above ~10³ atoms.

**Purification phase (compute-bound).** FLOPs/atom/SP2-iter ≈ 2·f·m²·n_b³. With m=200, n_b=15, f=0.3 ⇒ ~8×10⁷ FLOP; ×35 iters ⇒ ~2.8×10⁹ FLOP/atom ⇒ 10⁵ atoms ≈ 2.8×10¹⁴ FLOP ≈ **~1 s per purification on one GPU**.

**Consequences (order-of-magnitude targets):**

| System | Mode | Target (1 GPU unless noted) |
|---|---|---|
| 30-atom organic, PBE/DZP | R0 batched screening | ≥10⁴ single-points/hour |
| 64 H₂O (192 atoms) | XL-BOMD MD | ~10 ms/step ⇒ ~100 steps/s (DFTB+ ≈ 10³/s ⇒ within ~10×) |
| 1,000-atom slab, PBE | R1/R2 | SCF ground state in minutes; MD 2–5 steps/s |
| 10⁴ atoms gapped | R2, XL-BOMD | ~0.2–0.3 s/step ⇒ ~15–25 ps/day (0.5 fs) |
| 10⁵ atoms gapped | R2, XL-BOMD | ~2–3 s/step, ~50–70 GB — single-GPU headline |
| 10⁶ atoms gapped | R2 on 8–32 GPUs | ~3–10 s/step at ≥80% weak-scaling efficiency |
| 10⁵ Al @ Tₑ=3,000 K | R3 FOE/SQ | ≤5× the gapped R2 cost at same N (order grows with β) |

These are **analytical design targets with a written assumption ledger**, not promises; gates M-Y1Q4, M-Y2Q2, M-Y2Q4 (§10) exist precisely to confront them with measurements, and every deviation >2× triggers a documented model revision. This discipline — model first, measure against it — is itself a deliverable (it is how we avoid the multiplicative-speedup fallacy that sank the original GridFlow sketch).
---

## 8. Debugging & correctness protocol (the ladder)

Every physics result must be traceable down a ladder where each rung is independently trusted:

1. **Kernel rung.** Every CUDA/HIP kernel has an FP64 CPU oracle and a ULP/absolute-error budget in `tolerances.yaml`; tested on random + adversarial inputs (clustered eigenvalues, tiny gaps, huge dynamic range). compute-sanitizer (memcheck/racecheck) is a CI gate, not an occasional tool.
2. **Operator rung.** S, T, V_nl, v_H, v_xc each validated against closed forms (Gaussian charges, hydrogenic states) and against an external code with matched basis/pseudo (ABACUS/SIESTA same-NAO; PySCF for molecular integrals).
3. **Energy rung.** Total energies vs R1 dense reference on control systems; mixed-precision vs full-FP64 A/B nightly with ≤0.5 meV/atom budget; virial vs direct stress cross-check.
4. **Force rung.** Central 5-point finite differences on randomized displacements, every module that touches forces, every night. Egg-box scan published per release.
5. **Dynamics rung.** NVE drift budget (≤30 µHa/atom/ps), time-step convergence, XL-BOMD-vs-converged-SCF trajectory statistics (RDFs, VACFs).
6. **Physics rung.** ACWF/Δ-subset lattice constants & bulk moduli; S22/W4-11 subsets; slab work functions & adsorption energies vs curated literature references.

**Bisect-the-physics harness:** every stage (density, potential, H, P) can be dumped/injected in a documented HDF5 schema, so any module can be swapped against a reference implementation to localize a discrepancy in hours, not weeks. **Deterministic mode** (ordered reductions, fixed seeds) makes every bug reproducible bit-for-bit. **Python layer:** explicit `Status` returns and `assert`-based invariants; no `try`/`except` control flow anywhere in `api/python/` or `examples/` (enforced by a linter rule in CI).

---

## 9. Benchmarking protocol

**Rules of engagement (apply to every number we publish):**
- Fixed-accuracy comparisons only: report time-to-solution at a stated, verified accuracy (≤1 meV/atom vs each code's own converged reference for solids; ≤0.5 kcal/mol for molecular energetics), never raw walltime at unmatched settings.
- Same physics where possible: PBE (+D4 where relevant), same PseudoDojo ONCV pseudopotentials (accepted by SPARC/QE/ABACUS/GPAW), same k-sampling/smearing, documented basis/grid convergence for every code.
- 3 repeats, cold and warm reported separately; energy (kWh) via NVML/rocm-smi in every table; exact commits, inputs, logs, and container digests published (CC-BY, DOI per campaign).
- Competitor codes run at their best-known settings, with their developers invited to review our configurations before publication (adversarial-collaboration etiquette).

**9.1 Piece-by-piece matrix (all baselines open source; licenses as of mid-2026 — reverify at kickoff):**

| # | TIDES module | Baseline (license) | Payload | Metric | Excellence bar |
|---|---|---|---|---|---|
| 1 | NAO H/S build | ABACUS (LGPL-3), SIESTA (GPL-3), CP2K GPW (GPL-2) | 10³-atom Si; 512-H₂O | atoms/s @ matched sparsity | ≥3× ABACUS-GPU |
| 2 | ρ/V grid ops | GPAW (GPL-3), SPARC (GPL-3) | same | % of HBM roofline | ≥60% sustained |
| 3 | Poisson (all BCs) | BigDFT PSolver (GPL), cuFFT reference | 256³–1024³, free/slab/bulk | time @ 1e-10 | ≥ parity with cuFFT path |
| 4 | Dense eigensolver bridge | ELPA (LGPL-3), cuSOLVERMp | 10k–60k dense | time | tracked (external libs) |
| 5 | ChFSI | DFT-FE (LGPL-2.1), SPARC | 5k-atom Mo | time/SCF @ ≤1 meV/atom | ≤1.5× DFT-FE |
| 6 | SP2-submatrix | CP2K NOLSM/DBCSR (GPL-2), NTPoly (MIT) | 10⁴–10⁶ a-Si:H | time/purification @ ≤0.5 meV/atom | ≥2× DBCSR-GPU |
| 7 | FOE/SQ metals | SPARC-SQ (GPL-3), CheSS (GPL) | 10⁴ Al, Tₑ 1–10 kK | time/step | ≥ parity, GPU-native |
| 8 | Hybrids (ISDF+ACE) | CP2K ADMM, QE ACE (GPL) | 500-atom TiO₂ slab, HSE06 | time/SCF @ matched accuracy | ≤4× own PBE |
| 9 | MD engine | CP2K, GPAW; speed anchors DFTB+ (LGPL-3), tblite (LGPL) | 64/512/4096 H₂O NVE | steps/s + drift | see §7 targets |
| 10 | Small-molecule end-to-end | GPU4PySCF (Apache-2), PySCF (Apache-2); xQC if open at kickoff | W4-11 & S22 subsets | batched mol/h; cold latency | ≥10× throughput; ≤2× cold |
| 11 | Accuracy end-to-end | published all-electron references (ACWF; FHI-aims/WIEN2k data) | ACWF subset, S22, surface DB | meV/atom; kcal/mol | DZP→TZP convergence documented |
| 12 | Scaling | CONQUEST (MIT), CP2K | 10⁵→10⁶ a-Si, water | weak/strong efficiency | ≥80% weak to 64 GPUs |

**9.2 End-to-end campaigns** (Y2Q4 "pieces" paper; Y4Q4 "spectrum" paper; Y5Q3 final rigorous campaign): a fixed 12-system gauntlet spanning the spectrum — W4-11 subset (batched), aspirin MD, 64/512/4096-H₂O MD, benzene@graphene, CO/Pt(111) (HSE06), 1k-atom amorphous HfO₂, 10⁴ a-Si:H, 10⁵ polymer melt, 10⁵ Al @ 3000 K (R3), 10⁶ a-Si demo — each with cold/warm times, kWh, certified accuracy, and the same systems run on every competitor that can run them. Where a competitor cannot run a payload (size, BCs, feature), that is recorded as a capability result, not a speed win.

---

## 10. Five-year plan, milestones, and go/no-go gates

**Y1 — substrate + molecules.** Q1: repo, CI (datacenter + RTX runners), tile substrate at ≥90% grouped-GEMM parity; Ozaki reductions validated. Q2: NAO generator + two-center integrals vs PySCF/ABACUS at rung-2 tolerances. Q3: LDA/GGA SCF for molecules; analytic forces pass FD; D3/D4. Q4 **[Gate G1]**: alpha release; R0 batching demo ≥10⁴ single-points/h/GPU; piecewise report #1 (rows 1–4). *G1 criterion: perf model §7 within 2× on rows measured, else model revision + scope review.*
**Y2 — extended systems + linear scaling.** Q1: PBCs (Γ for large; k-mesh ≤2k atoms), stress. Q2 **[Gate G2]**: SP2-submatrix 10⁴ atoms on one GPU at ≤0.5 meV/atom vs R1 control — the make-or-break gate for the large end; fallback if missed: OMM/FOE substitution for R2, timeline +2 quarters. Q3: XL-BOMD stable (drift budget) on 64/512 H₂O. Q4: 8-GPU 10⁵ atoms; **QTT gate R-1** (continue iff QTT Poisson+ρ shows ≥2× speed or ≥4× memory vs FFT path at equal accuracy on 512-H₂O); benchmark paper #1.
**Y3 — metals, hybrids, surfaces.** Q1: FOE/SQ + Mermin (R3). Q2: ISDF+ACE hybrids. Q3: surface-chemistry validation suite (adsorption DB, CO/Pt(111) documented against literature including its known GGA failure). Q4: PAW-in-NAO go/no-go; v0.9.
**Y4 — extreme scale + product.** Q1–Q2: 10⁶-atom demo (8–32 GPUs). Q3: screening product mode; geometry/NEB hardening; optional Gaussian fast-path decision for absolute cold-latency parity at <50 atoms (stretch). Q4: **QTT gate R-2** (merge to default or archive with tech report); benchmark paper #2 (end-to-end spectrum).
**Y5 — hardening + community + final campaign.** Q1: API freeze, docs complete. Q2: external beta with ≥5 independent groups. Q3: final rigorous campaign (§9.2) + reproducibility archive with DOI. Q4: v1.0, flagship paper, governance handoff to a community technical steering committee.

---

## 11. Team of 10 — split, interfaces, cadence

| Scientist | Work package | Also on-call for |
|---|---|---|
| S1 | WP1 substrate & precision; chief performance engineer | architecture council |
| S2 | WP2 basis & integrals | pseudo/PAW liaison to S7 |
| S3 | WP3 grids/Poisson/XC | WP-R QTT (20%) |
| S4 | WP4 mid-range solvers | broker calibration |
| S5 | WP5 linear-scaling solvers | WP-R QTT (20%) |
| S6 | WP6 SCF/XL-BOMD/forces/MD | physics validation with S9 |
| S7 | WP7 hybrids/dispersion/PAW | surface-chemistry suite |
| S8 | WP8 parallel/HPC/portability | HIP quarterly gate |
| S9 | WP9 verification & benchmarking — the red team | release veto |
| S10 | WP10 API/docs/community/releases | user support rotation |

Interfaces are **contract tests**: WP boundaries (`TileMat`, grid arrays, HDF5 stage dumps) are frozen schemas with their own test suites, so teams integrate against contracts, not against each other's branches. Cadence: 2-week sprints; biweekly architecture council (S1/S4/S5/S9); quarterly landscape review (S9 presents competitor movement — ABACUS-GPU, DFT-FE/PAW-FE, SPARC, CP2K, GPU4PySCF/xQC — and proposes bar adjustments); every scientist spends ≥1 week/year on another WP (bus-factor insurance).

## 12. Risk register (top 10)

| Risk | L×I | Mitigation / trigger |
|---|---|---|
| Submatrix accuracy insufficient for full-DFT H (original NOLSM demos were tight-binding-quality) | M×H | Error-compensated truncation; G2 gate Y2Q2; fallback OMM/FOE for R2 |
| Mixed-precision instability near degeneracies/small gaps | M×H | f64e refinement sweeps; gap monitors auto-escalate precision; nightly A/B budget |
| NAO basis ceiling (BSSE, anions, polarizabilities) | M×M | TZP+diffuse sets; counterpoise tooling; PAW thrust decision Y3Q4 |
| XL-BOMD fragility at vanishing gaps | L×M | KSA kernel + fractional occupations (per Niklasson-line methods); SCF fallback per-step |
| k-point (complex-tile) path doubles solver work | M×M | Γ-supercell strategy for large N; complex tiles only in R0/R1 |
| Competitors move (ABACUS-GPU, xQC, DFT-FE/PAW-FE) | H×M | differentiators are orthogonal (XL-BOMD + Ozaki-MP + batching + broker); quarterly review re-aims bars |
| Hardware shift (INT8 removal → FP8-centric emulation) | H×L | Ozaki layer abstracts slice type; FP8/FP16 variants both implemented (2025 literature already covers this) |
| Key-person loss | M×H | contract tests, docs-with-derivations, cross-WP weeks, everything upstreamed |
| Scope creep (TDDFT, GW, ML asks) | H×M | change-control: new physics only via RFC + council + explicit descope of something else |
| Funding/hardware access gaps | M×H | consumer-GPU CI keeps development unblocked; cloud burst budget for gates only |

## 13. Governance — "democratic" made operational

Apache-2.0 license (maximal reuse, including commercial — access without gatekeeping; the team may choose LGPL-3 at kickoff if copyleft is preferred: decision D-1). DCO sign-off; public RFC process for any user-visible change; semantic versioning; monthly open community call from Y2; all benchmark data CC-BY with DOIs; container images per release; a standing commitment that the full test suite passes on one consumer RTX-class GPU — if science requires a supercomputer to *verify*, it is not democratic. Y5Q4: governance transfers to an elected technical steering committee.

## 14. Key references (paraphrased throughout; verify versions at kickoff)

1. GPU4PySCF: Li, Wu et al., GPU-accelerated PySCF, arXiv:2404.09452 (v1/v2). 2. xQC JIT quantum chemistry, arXiv:2507.09772 (2025). 3. DFT-FE 1.0, arXiv:2203.07820; 2023 ACM Gordon Bell Prize; PAW-FE, arXiv:2604.26037 (2026). 4. SPARC: SoftwareX 15, 100709 (2021); v2.0.0 Softw. Impacts (2024); GPU local/semilocal, J. Chem. Phys. 158, 204117 (2023); GPU hybrids, arXiv:2501.16572 (2025); SQ O(N) (million-atom capability), Suryanarayana et al. 5. NOLSM 100M-atom AIMD & submatrix method: Schade, Lass, Kühne, Plessl et al., arXiv:2004.10811 and follow-ups. 6. ABACUS GPU-NAO, arXiv:2409.09399 (2024). 7. XL-BOMD: Niklasson, PRL 100, 123004 (2008); density-matrix XL-BOMD, JCTC 16, 3628 (2020); Krylov-subspace integration, J. Chem. Phys. 152, 104103 (2020); graph-based shadow MD, J. Chem. Phys. 158, 074108 (2023); tensor-core quantum MD (Finkelstein et al.). 8. Ozaki-scheme FP64 emulation: Ozaki et al. (2012); Ootomo–Ozaki–Yokota, IJHPCA 38, 297 (2024); Uchino et al., IJHPCA 39 (2025); guaranteed-accuracy extensions, arXiv:2511.13778 (2025); FP8 variants, arXiv:2508.00441 (2025); DOE MFP64 whitepaper (2025). 9. QTT: Shinaoka et al., PRX 13, 021015 (2023); Lindsey, multiscale interpolative QTT (2023); tensorized orbitals, Jolly, Núñez-Fernández, Waintal, PRB 111 (2025). 10. ISDF/ACE: Lu–Ying JCP (2015); Lin Lin, ACE JCTC (2016); Hu et al. ISDF-hybrid line. 11. ELSI/ELPA solver-interface precedent: Yu et al., Comput. Phys. Commun. 12. CONQUEST 2M-atom O(N): Bowler–Miyazaki. 13. libxc: Lehtola et al. 14. PseudoDojo ONCV: van Setten et al. 15. ACWF verification: Bosoni et al., Nat. Rev. Phys. (2024). 16. DFTK a-posteriori error control: Cancès, Herbst et al.

*End of proposal v1.0. Decision list for kickoff: D-1 license (Apache-2.0 vs LGPL-3); D-2 project name; D-3 hardware baseline (which datacenter GPU + which consumer GPU define the CI contract); D-4 whether the Y4 Gaussian fast-path stretch is in scope from day one.*
