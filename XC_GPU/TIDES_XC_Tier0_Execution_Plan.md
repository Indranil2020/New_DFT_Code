# TIDES XC Tier-0 Execution Plan
## Top-15 functionals, fused GPU engine: concrete tasks, files, and gates (July 2026)

Companion to `TIDES_GPU_XC_Design.md` (the *why*) and `TIDES_Codebase_Audit_2026-07-10.md`
(the *current state*). This file is the *what to type*. Task IDs are `T-X<phase>.<n>`.

**Hard prerequisite from the audit:** Tier-0 XC fuses into the NAO product pipeline
(audit item P1.6). Phase X0/X1 tasks below include exactly the slices of that pipeline XC needs
(device-resident ρ/∇ρ and the vmat adjoint). Do not build XC against the GTO toy driver — it is
the validation oracle, not the integration target.

---

## 1. The Top-15 (what "hot set" means, precisely)

| # | Functional | Family | Composition (libxc IDs for the oracle) | Why in the set |
|---|---|---|---|---|
| 1 | LDA-PW92 | LDA | `LDA_X` + `LDA_C_PW` | launch default; already partially present |
| 2 | SVWN5 | LDA | `LDA_X` + `LDA_C_VWN` | B3LYP ingredient; legacy comparisons |
| 3 | PBE | GGA | `GGA_X_PBE` + `GGA_C_PBE` | the workhorse |
| 4 | PBEsol | GGA | `GGA_X_PBE_SOL` + `GGA_C_PBE_SOL` | solids/lattice constants (TIDES large-N regimes) — same code as PBE, two constants |
| 5 | revPBE | GGA | `GGA_X_PBE_R` + `GGA_C_PBE` | surfaces/adsorption |
| 6 | RPBE | GGA | `GGA_X_RPBE` + `GGA_C_PBE` | catalysis community default |
| 7 | BLYP | GGA | `GGA_X_B88` + `GGA_C_LYP` | molecular chemistry; B3LYP ingredient |
| 8 | PBE0 (local part) | hyb-GGA | PBE with a_x=0.25 scaling | EXX from WP7 (ISDF+ACE) |
| 9 | B3LYP (local part) | hyb-GGA | 0.08·Slater + 0.72·B88 + 0.19·VWN + 0.81·LYP | most-cited functional in history |
| 10 | HSE06 (SR-ωPBE local part) | RSH-GGA | `GGA_X_WPBEH`(ω=0.11) + `GGA_C_PBE` | the solid-state hybrid; proposal WP7 targets it explicitly |
| 11 | TPSS | mGGA | `MGGA_X_TPSS` + `MGGA_C_TPSS` | nonempirical mGGA baseline |
| 12 | r²SCAN | mGGA | `MGGA_X_R2SCAN` + `MGGA_C_R2SCAN` | default mGGA (numerically tame; grid-robust) |
| 13 | SCAN | mGGA | `MGGA_X_SCAN` + `MGGA_C_SCAN` | benchmark king; **FP64-only** (α-switching sensitivity) |
| 14 | ωB97X (local part) | RSH-GGA | `HYB_GGA_XC_WB97X` semilocal part | modern chemistry default; -V variant waits for VV10 (M6+) |
| 15 | M06-2X (local part) | hyb-mGGA | `HYB_MGGA_X_M06_2X` + `MGGA_C_M06_2X` | kinetics community; stress-test for register pressure |

Order of implementation: 1→3→4(free)→7→5,6(near-free)→9,8→12→11→10→13→14→15.
Everything not in this table = Tier 1 (libxc-CUDA catalog) or Tier 2 (CPU fallback).

---

## 2. Files and folders to create

```
tides/core/grid/xc/                          # NEW — the XC engine (replaces ad-hoc xc.cu/xc.hpp over time)
├── xc_engine.hpp                # T-X0.3  public contract: XcSpec, XcGridIn, XcGridOut, XcEval()
├── xc_engine.cu                 # T-X0.3  host dispatch: XcSpec → template launch (Tier 0) / tier1 / tier2
├── xc_arena.hpp                 # T-X0.1  device-resident buffer arena (SCF-scope lifetime, cudaMallocAsync pool)
├── functionals/                 # Tier 0: header-only, __device__, plain arithmetic (CPU-compilable too)
│   ├── common.cuh               # T-X1.1  rs/zeta/s² helpers, spin-scaling wrapper, cbrt-based pow replacements,
│   │                            #         threshold constants MIRRORING libxc dens/sigma/tau thresholds
│   ├── lda_slater.cuh           # T-X1.1  ε_x, v_ρ (+ f_xc later)
│   ├── lda_pw92.cuh             # T-X1.1  analytic derivatives (kills the FD hack, audit B2)
│   ├── lda_vwn5.cuh             # T-X2.3
│   ├── gga_pbe.cuh              # T-X1.1  templated on {κ, μ, β} → PBE / PBEsol / revPBE from one source
│   ├── gga_rpbe.cuh             # T-X2.1
│   ├── gga_b88.cuh              # T-X2.1
│   ├── gga_lyp.cuh              # T-X2.1
│   ├── gga_wpbeh.cuh            # T-X3.3  SR-ωPBE (HJS form) + device expint E1
│   ├── mgga_tpss.cuh            # T-X3.2
│   ├── mgga_r2scan.cuh          # T-X3.2
│   ├── mgga_scan.cuh            # T-X3.2  FP64-only enforced via static_assert on precision template
│   ├── mgga_m06.cuh             # T-X3.5
│   ├── gga_wb97x.cuh            # T-X3.5
│   └── compose.cuh              # T-X2.2  Weighted<F,c...> pack → B3LYP/PBE0/ωB97X local parts
├── kernels/
│   ├── xc_lda_kernel.cu         # T-X1.2  fused: load ρ → eval → write w·v_ρ → in-kernel E_xc reduction
│   ├── xc_gga_kernel.cu         # T-X1.2  + σ from ∇ρ in registers, write w·v_ρ and 2w·v_σ∇ρ
│   ├── xc_mgga_kernel.cu        # T-X3.1  + τ path, write w·v_τ
│   └── reduce.cuh               # T-X1.2  warp-shuffle + atomic (fast) AND ordered per-block (deterministic)
├── tier1/
│   ├── CMakeLists.txt           # T-X4.1  compiles pinned external/libxc maple2c sources with nvcc -x cu
│   ├── libxc_device_adapter.cu  # T-X4.1  XcGridIn(SoA) ↔ libxc interleaved layout; device-pointer calls
│   └── peephole.py              # T-X4.2  pow(x, p/q) → cbrt/sqrt chains on generated C (build step, not committed output)
├── tier2/
│   └── cpu_fallback.cpp         # T-X4.3  pinned-memory batched CPU libxc for exotics (async overlap)
└── tests/
    ├── rung0_oracle_sweep.cpp   # T-X0.4  pointwise vs CPU libxc across (ρ,σ,τ,ζ) lattice, per functional
    ├── rung1_adjoint_tests.cpp  # T-X1.4  Tr[V_xc·ΔP] vs FD of E_xc — catches dropped-term bugs structurally
    ├── rung1_molecules.cpp      # T-X1.6  He/Ne/H2O energies vs pinned PySCF refs (matched basis/grid)
    ├── ab_precision_tests.cpp   # T-X4.4  FP32-vs-FP64 nightly, budget from tolerances.yaml
    └── bench_roofline.cpp       # T-X1.5  Gpt/s per kernel vs BW roofline; regression-tracked

tides/verification/tolerances.yaml           # T-X0.4 add: xc_rung0_rel: 5e-14, xc_energy_abs_Ha: 1e-8,
                                             #   xc_ab_budget_meV_atom: 0.5, xc_adjoint: 1e-9
tides/external/libxc/                        # T-X4.1 pinned submodule (currently using system install — pin it)
```

Retirement path for existing files (audit B1/B2/B10): `xc.cu`'s `XCEvalPbeCuda` **deleted** at
T-X1.2; `XCEvalLdaCuda` re-implemented as a thin call into `xc_engine` at T-X1.6 (keep signature
for E3 tests until they migrate); `libxc_wrapper.hpp` survives as the Tier-2/oracle interface only.

---

## 3. Task list

### Phase X0 — Contract, residency, oracle (week 1–2)
| ID | Task | Depends | Acceptance |
|---|---|---|---|
| T-X0.1 | `XcArena`: device buffers with SCF-scope lifetime; stream-ordered alloc; **no `cudaMalloc`/`cudaDeviceSynchronize` in any per-iteration path** | — | compute-sanitizer clean; alloc count per SCF iter = 0 |
| T-X0.2 | Analytic PW92 derivatives in a shared header used by BOTH the atomgen CPU path and the device functor; delete the FD version | — | matches libxc `LDA_C_PW` v_ρ to ≤1e-14 rel on rung-0 lattice |
| T-X0.3 | `xc_engine.hpp/.cu` contract (XcSpec/XcGridIn/XcGridOut per Design §2.3) + host dispatch skeleton; static_asserts on alignment/padding | — | header compiles CPU-only and CUDA; contract doc in header comment |
| T-X0.4 | Rung-0 oracle harness: (ρ,σ,τ,ζ) sweep lattice, CPU libxc reference, per-functional CSV + pass/fail vs `tolerances.yaml` | X0.3 | harness runs for LDA_X in CI on both GPUs |

### Phase X1 — LDA + PBE, fused and integrated (week 3–6) ⇒ **Gate GX1**
| ID | Task | Depends | Acceptance |
|---|---|---|---|
| T-X1.1 | Functors: `lda_slater`, `lda_pw92`, `gga_pbe` (templated κ,μ,β), `common.cuh` (pow-free, libxc-matched thresholds) | X0.2 | rung-0: ≤5e-14 rel vs libxc for ε, v_ρ, v_σ |
| T-X1.2 | Fused kernels `xc_lda_kernel.cu`, `xc_gga_kernel.cu` + `reduce.cuh` (fast + deterministic); delete `XCEvalPbeCuda` | X1.1, X0.1 | E_xc matches CPU quadrature ≤1e-12 rel; deterministic mode bit-identical over 100 runs; 0 register spills (`-Xptxas -v`) |
| T-X1.3 | Analytic ∇ρ: rho_build produces ρ and ∇ρ on device via tile-batched GEMM (P·φφ, P·φ∇φ) — replaces FD `ComputeSigma`/`ReducedGradient` (audit C2 slice) | WP1 TileMat | ∇ρ vs analytic Gaussian model ≤1e-10; feeds XcGridIn directly (no reshuffle) |
| T-X1.4 | GGA adjoint in vmat_build: consume w·v_ρ and 2w·v_σ∇ρ (integration-by-parts form — the −2∇·(v_σ∇ρ) term lives HERE, nowhere else); rung-1 adjoint test | X1.2, X1.3 | Tr[V_xc·ΔP] vs FD(E_xc) ≤1e-9 on 100 random ΔP |
| T-X1.5 | Roofline bench: Gpt/s for LDA/PBE kernels on both CI GPUs, regression-tracked | X1.2 | ≥60% of measured HBM roofline (proposal WP3 bar) on datacenter+RTX |
| T-X1.6 | Integrate into the NAO product driver (audit P1.6): SCF iterates with ρ→XC→V_xc fully device-resident; PCIe per iteration = scalars only | X1.4, audit P1.6 | He/Ne/H2O LDA & PBE vs pinned PySCF refs (matched basis/grid) ≤1e-6 Ha initially, tightening to 1e-8; **profile shows XC phase ≤5% of SCF step** |

**Gate GX1 (the "20× generalized" gate):** zero H2D/D2H bytes in the SCF iteration trace
(Nsight Systems), XC ≤5% of step, energies green. If missed by >2×: written model revision.

### Phase X2 — Hot-set GGAs, spin, hybrids' local parts, batching (week 7–12)
| ID | Task | Depends | Acceptance |
|---|---|---|---|
| T-X2.1 | Functors: B88, LYP, RPBE (revPBE/PBEsol = parameter instantiations of X1.1) | X1.1 | rung-0 green each |
| T-X2.2 | `compose.cuh` Weighted-pack; spin-polarized paths via spin-scaling (exchange) + polarized correlation branches | X2.1 | rung-0 polarized lattice (ζ≠0) green; unpolarized results bit-identical to before |
| T-X2.3 | B3LYP/PBE0 local parts (+ VWN5); coefficients from one table shared with future EXX wiring | X2.2 | B3LYP local ε/v vs libxc `HYB_GGA_XC_B3LYP` semilocal part ≤1e-13 |
| T-X2.4 | R0 batched layout: concatenated grids, per-system E_xc segmented reduction, CUDA-graph capture of rho→XC→vmat | X1.6 | 10³ × 30-atom batch: one launch per sweep; ≥5× launch-overhead reduction vs loop |

### Phase X3 — mGGA + range-separated (week 13–18)
| ID | Task | Depends | Acceptance |
|---|---|---|---|
| T-X3.1 | τ plumbing: rho_build emits τ (½Σf|∇ψ|² via GEMM path or P-based equivalent); mGGA kernel + w·v_τ adjoint in vmat_build | X1.3 | τ vs analytic model ≤1e-10; adjoint test extended to v_τ |
| T-X3.2 | TPSS, r²SCAN, SCAN functors (SCAN: static_assert FP64) | X3.1 | rung-0 green; register report: 0 spills at `__launch_bounds__(128)` |
| T-X3.3 | SR-ωPBE (HJS) + device E₁ expint; HSE06 local part end-to-end | X2.3 | rung-0 vs `GGA_X_WPBEH` ≤1e-12; ω parameter plumbed through XcSpec |
| T-X3.4 | Occupancy/ILP tuning pass on mGGA (Nsight Compute); kernel split ONLY if spills measured | X3.2 | documented ncu report; ≥the X1.5 roofline fractions |
| T-X3.5 | ωB97X and M06-2X local parts (power-series enhancement factors — register stress test) | X3.2 | rung-0 green; spills = 0 or documented split |

### Phase X4 — Catalog, precision, derivatives (week 19–26)
| ID | Task | Depends | Acceptance |
|---|---|---|---|
| T-X4.1 | Tier-1: pin libxc submodule; CMake wrapper compiling maple2c+work sources via `nvcc -x cu`; SoA↔interleaved adapter; route `XcSpec` fallthrough | X0.3 | PBE via Tier-1 matches Tier-0 ≤1e-13; any libxc ID runnable device-resident |
| T-X4.2 | `peephole.py` pow→cbrt pass on generated sources (build artifact) | X4.1 | Tier-1 GGA kernel ≥2× faster than unpatched; results unchanged ≤1e-14 |
| T-X4.3 | Coverage matrix CI: rung-0 sweep over the full catalog nightly; auto-published table; Tier-2 CPU batch fallback for failures/exotics | X4.1 | ≥95% catalog green; exotics documented with route |
| T-X4.4 | FP32 pointwise mid-SCF path (template T=float) behind the A/B gate; per-family enable flags; hazard escalations (ζ→±1, SR-erfc, α-numerator) in FP64 | X1.6 | nightly A/B ≤0.5 meV/atom for enabled families; RTX speedup ≥2× documented on PBE/r²SCAN |
| T-X4.5 | Stress-tensor grid terms; f_xc (order 2) for hot set (TDDFT/Hessian prep) | X3.2 | f_xc rung-0 vs libxc maxorder=2 ≤1e-12 |

---

## 4. Standing rules for this workstream

1. **Every threshold lives in `tolerances.yaml`** — a test may not invent its own bar (audit §E).
2. **No new code path may bypass `xc_engine.hpp`** — one entry point, or the B1-style divergence
   returns.
3. **The GTO driver is an oracle, never a benchmark subject.** All published numbers come from the
   NAO driver.
4. **Delete as you land:** each phase retires its predecessor (X1.2 deletes `XCEvalPbeCuda`; X1.3
   deletes FD gradients). Dead wrong code is worse than no code.
5. **Gates are confrontations, not ceremonies:** GX1's Nsight trace and the §7 design-target table
   get updated with measured numbers; deviations >2× ⇒ written model revision (proposal §7
   discipline).
