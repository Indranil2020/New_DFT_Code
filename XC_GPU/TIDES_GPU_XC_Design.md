# TIDES GPU XC — The Design
## Single best architecture for XC evaluation in the fastest DFT code (July 2026)

**Status:** Design v1.0 — supersedes the "port all 700+ functionals via Maple→CUDA fork" strategy of
`TIDES_GPU_XC_Master_Plan.md` where the two disagree. Rationale for every override is given inline.

---

## 0. The reframe (read this first)

**"Convert libxc to CUDA" is the wrong problem statement for the fastest DFT code.** Here is the
quantitative reality that must drive the design:

1. **The XC functional math is cheap. The memory movement is not.**
   For a 200-atom molecule (~1.2M fine-grid points), a PBE evaluation moves per SCF iteration:
   - CPU-libxc path (current `XCEvalPbeCuda`): ρ+σ down (16 B/pt) + ε,v_ρ,v_σ up (24 B/pt)
     ≈ **48 MB across PCIe ≈ 4–6 ms** at PCIe 4.0 effective rates, plus launch/sync latency.
   - Device-resident fused path: same data touched once in HBM at ~1–3 TB/s ≈ **0.05–0.15 ms**.

   That ~30–50× ratio *is* your observed 20× LDA speedup, explained. It has nothing to do with how
   clever the functional kernel is — it is a data-locality result. The design below therefore
   optimizes **where bytes live and how many times they move**, and treats the functional math as
   a secondary (though real) concern.

2. **Once on-GPU, XC evaluation is <5–10% of the SCF step.** In GauXC and in Stocks & Barca
   (JCTC 21, 10263, 2025), the grid pipeline cost is dominated by basis collocation, density
   formation, and V_xc matrix formation (all GEMM-shaped) — not by the pointwise functional.
   For TIDES this means: the XC kernel must be *fused into* `rho_build`/`vmat_build`'s data flow,
   not shipped as a standalone library call, and the GEMM-shaped parts stay in WP1 tile territory.

3. **Functional usage is extremely skewed.** ~15 functionals (LDA/PW92, PBE, revPBE, RPBE, BLYP,
   B3LYP, PBE0, HSE06, TPSS, SCAN, r²SCAN, ωB97X-family, B97/M06 selections) cover >99% of
   published production DFT. Engineering all 700+ equally is a uniform investment against a
   power-law payoff. The right shape is **two tiers**: a small hot set engineered to the roofline,
   and the full catalog available on-GPU at "correct, unfused" speed.

**The single best design, in one sentence:**

> A three-tier XC engine — **Tier 0:** ~15 production functionals as header-only templated
> `__device__` functors fused directly into the TIDES grid pipeline (zero HBM round-trips for
> intermediates, zero PCIe traffic except one scalar); **Tier 1:** the full libxc catalog compiled
> *unmodified* with libxc's own CUDA backend (device-resident, unfused) behind the same interface;
> **Tier 2:** CPU libxc as the FP64 validation oracle and batch fallback for the ~24 non-closed-form
> exotics — with FP64 accumulation everywhere, dynamic FP32/FP64 pointwise precision as a
> *measured, gated* option, and a nightly A/B accuracy contract (≤0.5 meV/atom) that owns the
> precision decision empirically.

### 0.1 Explicit overrides of the Master Plan (Rule 7: surface conflicts, pick one)

| Master Plan says | This design says | Why |
|---|---|---|
| Fork libxc, modify `maple2c.py`, regenerate all 700+ functionals as CUDA | **Do not fork the generator.** The Maple-generated C sources ship *inside* the libxc repo (`src/maple2c/`, checked in); regeneration needs a proprietary Maple license you don't have. Libxc's own CUDA backend already compiles these sources with `nvcc -x cu` and `GPU_FUNCTION` macros. Vendoring + a thin CMake wrapper gets the whole catalog on GPU with ~zero divergence from upstream. | Removes the largest cost & bus-factor risk (Maple licenses, 700-file fork rebased against upstream forever). The master plan itself hypothesized this is how libxc CUDA works (§5.5) — it is; build on it instead of duplicating it. |
| All 700+ functionals are first-class targets | Top ~15 are first-class (fused, roofline-engineered); the rest are catalog-class (on-GPU, correct, unfused) | Power-law usage; engineering effort goes where FLOPs and users are. "Fastest DFT code" is measured on PBE/B3LYP/r²SCAN/HSE06, not on `gga_x_hjs_b88_v2`. |
| Standalone `__global__` XC kernels called between rho_build and vmat_build | XC evaluation **fused** with quadrature weighting and the vmat prologue; ε/v never materialized as standalone arrays in the hot path | Halves HBM traffic of the XC phase; the arrays the vmat kernel actually needs (w·v_ρ, 2w·v_σ∇ρ, w·v_τ) are computed in registers and written once. |
| Milestones organized around porting functional families | Milestones organized around **pipeline residency first, coverage second** (fix the round-trip TODAY with 2 functionals, then widen) | The 20× is in residency+fusion; family coverage adds availability, not speed. |

Everything else in the Master Plan (survey, history, links, risk table) stands and is referenced, not repeated.

### 0.2 What already exists in TIDES and what's wrong with it (audit of `core/grid/`)

| File | Finding | Severity |
|---|---|---|
| `xc.cu:291` `XCEvalPbeCuda` | **The gpu4pyscf anti-pattern, live in our code:** evaluates PBE on CPU via libxc, uploads ε to GPU *only to do the energy reduction*, downloads again. Two PCIe round-trips to save one trivial sum. | Fix first |
| `xc.cu:67–75` | PW92 correlation potential via **finite-difference** d(ε_c)/d(r_s). The analytic derivative is 5 lines and exact; FD costs 3× the evals and injects ~1e-11 noise into V_xc. | Fix first |
| `xc.cu:180` `XCEvalLdaCuda` | `cudaMalloc`/H2D/D2H/`cudaFree` **every call**, plus `cudaDeviceSynchronize`. Even the "GPU" LDA path is not device-resident. In an SCF loop this is allocator thrash + full-device stalls. | Fix first |
| `libxc_wrapper.hpp:185–189` | GGA V_xc drops the −2∇·(v_σ∇ρ) term at this level ("computed in the SCF loop"). The adjoint (integration-by-parts) form must own this term inside vmat_build; a deferred scalar-potential patch-up is how sign/factor bugs are born. | Design-level |
| `libxc_wrapper.hpp:128` `ComputeSigma` | ∇ρ from central finite differences of grid ρ. For an NAO code, ∇ρ must come **analytically** from the density build (Σ P_μν φ_μ∇φ_ν) — FD gradients poison forces and the egg-box budget (WP3 acceptance: <1e-4 Ha/Bohr). | Design-level |

These five findings are the concrete evidence for the reframe: the current bottleneck is
architecture, not functional coverage.

---

## 1. Mathematical–physics decomposition

**Why this is the best:** the decomposition below is exact (no approximation is introduced), it
isolates the 100%-parallel core from the three genuinely nonlocal pieces, and every reformulation
listed is a known-good identity that reduces device work without touching the physics.

### 1.1 What is embarrassingly parallel
Everything semilocal. For every family the evaluation at grid point *r* depends only on local
quantities at *r*:

| Family | Inputs per point | Outputs per point (order 1) |
|---|---|---|
| LDA | ρ (or ρ↑,ρ↓) | ε, v_ρ |
| GGA | ρ, σ=∇ρ·∇ρ | ε, v_ρ, v_σ |
| mGGA | ρ, σ, τ (± ∇²ρ) | ε, v_ρ, v_σ, v_τ (± v_l) |

All higher derivatives (f_xc, k_xc for TDDFT/Hessians) are equally pointwise. There is **no
intrinsically sequential arithmetic anywhere in the semilocal catalog** — perfect map over 10⁵–10⁸
points, no inter-thread communication except the single energy reduction Σ w_i ρ_i ε_i.

### 1.2 The three genuinely nonlocal pieces (separate algorithms, not XC-kernel work)
1. **Exact exchange** (hybrids/RSH): handled by ISDF+ACE per WP7. The XC engine's job is only the
   *semilocal remainder* with correct coefficients (e.g. B3LYP local part; HSE's short-range-screened
   PBE exchange, which is still pointwise).
2. **Nonlocal vdW correlation** (VV10/rVV10, vdW-DF): a double-grid sum, O(N log N) with the
   standard FFT factorization. Needed for ωB97X-V, B97M-V, SCAN+rVV10. Schedule as its own kernel
   (M5); do not entangle with the pointwise engine.
3. **D3/D4 dispersion:** atom-pairwise, already WP7, not grid work at all.

### 1.3 Pointwise-implicit exceptions (~24 functionals)
Becke–Roussel-type (BR89, TB09/TPSS-BJ, B00…) require solving x·exp(−2x/3)/(x−2)=y per point.
Still pointwise-parallel, but not closed-form. Resolution, in order of preference:
1. **Proynov–Gan–Kong analytic interpolation** (Chem. Phys. Lett. 455, 103, 2008): closed-form fit
   accurate to ~1e-9 across the full range → a plain `__device__` function, no iteration.
2. Guarded per-thread Newton (converges <8 iters; warp divergence is negligible because the
   iteration count is nearly uniform in practice).
3. Tier-2 CPU batch fallback (these functionals are <<1% of usage).

`gga_x_wpbeh` (HSE ingredient) needs the exponential integral E₁: implement the standard
series/continued-fraction split in FP64 on device (~30 lines), or use the
Henderson–Janesko–Scuseria closed forms for short-range PBE exchange. Known solutions; M3 item.

### 1.4 GPU-friendly reformulations (exact identities only)
- **Spin-scaling for exchange:** E_x[ρ↑,ρ↓] = ½(E_x[2ρ↑]+E_x[2ρ↓]) → every exchange functional is
  written once, unpolarized; the polarized case is two calls on scaled inputs. Halves the code and
  register pressure of the polarized path.
- **Reduced-variable form:** evaluate in (r_s, ζ, s² or x², τ̃) — avoids recomputing cube roots,
  and s² avoids a sqrt in most GGA enhancement factors.
- **One kernel computes ε and *all* requested derivatives together.** The Maple-generated code
  shares ~70% of subexpressions between ε and v; separate ε-kernels and v-kernels (as some codes
  do) roughly double the transcendental cost. Never split them.
- **pow() elimination:** rational exponents x^(1/3), x^(4/3), x^(7/6)… become
  `cbrt`/`sqrt`/`rsqrt`/multiply chains. Double `pow` is hundreds of cycles and polluted by
  argument-reduction branches; `cbrt` is tens. Generated libxc code is saturated with
  `pow(x, 0.1e1/0.3e1)` patterns → a **peephole source pass** (small clang-based or regex tool over
  the generated C, ~a day of work) is the single highest-value micro-optimization for the Tier-1
  catalog; Tier-0 functors are written pow-free by hand.
- **Branch-free thresholds:** density cutoffs (`rho < dens_threshold`) as `fmax`/select, not `if`,
  to avoid warp divergence at grid-cell boundaries — but replicate libxc's threshold *values*
  exactly, or the oracle comparison fails at the 1e-10 level for diffuse tails.

---

## 2. Algorithmic redesign: dispatch and composition

**Why this is the best:** compile-time functional resolution gives the compiler full inlining and
per-functional register allocation (the difference between 25% and 60% occupancy on mGGAs), while
launch-time dispatch keeps the SCF loop generic. Every alternative — per-point `switch`, function
pointers, virtual dispatch — defeats inlining on `nvcc` and is measurably slower; JIT (xQC-style)
adds a compiler dependency TIDES doesn't need at these kernel counts.

### 2.1 Keep functional-by-functional, but resolve at compile time
```
                       host: one switch per SCF iteration (not per point!)
   XcSpec {family, ids[], coeffs[], flags} ──► launch< PbeXC   >(views…)   Tier 0
                                          ├──► launch< R2ScanXC>(views…)   Tier 0
                                          └──► tier1_libxc_cuda(spec, views…)  catalog
```
- **Tier 0:** each functional (or standard X+C combo: PBE, BLYP, r²SCAN…) is a `struct` functor
  with a static `__device__ eval()`; the fused grid kernel is a template taking the functor as a
  parameter. ~15 instantiations ≈ seconds of compile time. This is ExchCXX's proven design — we
  adopt the pattern, fused into our own pipeline instead of called as an external library.
- **Composed functionals** (aliases like B3LYP-local = 0.08·Slater + 0.72·B88 + 0.81·LYP + 0.19·VWN):
  a compile-time pack `Weighted<F1,c1, F2,c2,…>` whose `eval` accumulates in registers — still one
  pass over memory, zero intermediate arrays.
- **"Batch-of-functionals" kernels** (evaluating many functionals per point in one launch) are the
  wrong tool for SCF — you run exactly one functional per calculation. They only pay off for
  functional-fitting/ML-XC workloads; TIDES keeps a hook (the Tier-1 interface accepts a list) but
  does not engineer for it.

### 2.2 Batch over *systems*, not functionals (the R0 differentiator)
The proposal's R0 mode (10³–10⁴ molecules/GPU) is where TIDES beats everyone on throughput. The XC
engine must be batched-native from day one:
- Grids of all systems concatenated into one SoA arena; per-point `system_id` implicit via
  per-system offsets.
- **One kernel launch** (grid-stride) covers all systems; per-system E_xc via segmented reduction
  (block-per-segment ownership, FP64 atomics into a per-system accumulator array).
- The whole SCF sweep (rho→XC→vmat) captured in a **CUDA graph** → launch overhead amortized to ~0
  for thousands of small grids. This is free once the kernels are resident and fused.

### 2.3 The interface contract (frozen schema, WP-style)
```cpp
// core/grid/xc_engine.hpp — the only XC entry point TIDES code may call.
struct XcSpec   { Family family; int nspin; std::vector<XcTerm> terms; PrecisionPolicy prec; };
struct XcGridIn {           // ALL device pointers; SoA; 256-B aligned; lifetime = SCF scope
  const double* rho;        // [nspin][np]
  const double* grad;       // [nspin][3][np]  (GGA+; analytic from P·φ∇φ, never FD)
  const double* tau;        // [nspin][np]     (mGGA)
  const double* w;          // [np] quadrature weights
  int64_t np;  int nsys;  const int64_t* sys_offsets;
};
struct XcGridOut {          // exactly what vmat_build consumes — nothing else materialized
  double* wv_rho;           // [nspin][np]      w·v_ρ
  double* wv_grad;          // [nspin][3][np]   2w·v_σ·∇ρ (+cross term, polarized)
  double* wv_tau;           // [nspin][np]      w·v_τ (mGGA)
  double* exc_per_system;   // [nsys] FP64 accumulators (device)
};
Status XcEval(const XcSpec&, const XcGridIn&, XcGridOut&, cudaStream_t);
```
Both tiers implement this signature; the caller cannot tell them apart except by speed. This is the
contract test surface for WP3↔WP1/WP6, and the bisect-the-physics dump/inject point.

---

## 3. Precision strategy

**Why this is the best:** it puts FP64 where error provably accumulates (reductions, thresholds,
RSH attenuation), allows FP32 where SCF self-correction provably absorbs it (early-iteration
pointwise math), and — critically — makes every precision downgrade a *measured, gated* decision
against the existing nightly ≤0.5 meV/atom A/B budget rather than a belief. Note: **Ozaki/tensor
cores are irrelevant here** — pointwise XC is transcendental ALU/SFU work, not GEMM; the tensor-core
story lives in rho_build/vmat_build (WP1), exactly as the proposal says.

### 3.1 The hardware asymmetry that decides the defaults
| | FP64 peak | HBM BW | FP64 balance point | Fused PBE (AI≈7 FLOP/B) is… |
|---|---|---|---|---|
| H100/GB200 class | 34+ TF | 3.3+ TB/s | ~10 FLOP/B | **memory-bound** → FP64 math costs nothing extra; FP32 helps only via halved *bytes* |
| RTX 4090/5090 class | ~1.3 TF (1/64 rate) | ~1 TB/s | ~1.3 FLOP/B | **FP64-compute-bound** → FP32 pointwise math is worth 2–5× |

So the "democratic" consumer-GPU constraint (proposal §2.1) is precisely the case where pointwise
precision matters. Defaults:

| Quantity | Datacenter | Consumer RTX | Never negotiable |
|---|---|---|---|
| ε, v pointwise eval (SCF iters until residual <1e-5) | FP64 | FP32 | — |
| Final-iteration + reported-energy pass, forces/stress pass | FP64 | FP64 | ✔ fixed |
| E_xc reduction, all grid quadrature sums | FP64 (emulated-safe) | FP64 | ✔ fixed |
| Grid weights, density thresholds, RSH erf/erfc(μr) factors | FP64 | FP64 | ✔ fixed |
| Storage of ρ,∇ρ,τ arrays | FP64 (FP32 option gated by A/B) | FP32 store + FP64 promote | — |

### 3.2 Error budget (why FP32 pointwise is safe mid-SCF)
- Target: ≤0.5 meV/atom = 1.8e-5 Ha/atom (existing WP9 budget). E_xc ≈ 1–10 Ha/atom ⇒ need ~1e-6
  relative in the *converged* energy.
- FP32 pointwise ε with FP64 accumulation: per-point relative error ~1e-7, quasi-random across
  ~6×10³ points/atom ⇒ energy error ~1e-7–1e-6 Ha/atom. Inside budget, but only claimed after the
  A/B harness confirms it per functional family.
- V_xc errors enter the *converged* energy at second order (variational); mid-SCF FP32 potentials
  perturb the iterate, not the answer — this is the TeraChem dynamic-precision result (Luehr,
  Ufimtsev & Martínez, JCTC 7, 949, 2011: SP early / DP late, final errors <1e-6 Ha), and Yasuda's
  original GPU-XC finding (2008).
- Known FP32 hazards, each individually escalated to FP64 regardless of mode: ζ→±1 spin factors
  ((1±ζ)^(4/3) cancellation), s→large tails in exponential enhancement factors, erfc(μr) underflow
  in RSH short-range exchange, τ near the von-Weizsäcker bound (SCAN's α variable is a ratio of
  small differences — compute α's numerator in FP64 always).
- mGGA iso-orbital indicators are the most precision-sensitive objects in the catalog; r²SCAN
  exists partly *because* SCAN's α switching is numerically vicious. Ship r²SCAN as the default
  mGGA; keep SCAN FP64-only.

### 3.3 Determinism
Deterministic mode (proposal §3.4): per-block FP64 partials written to a buffer, single ordered
tree-reduction pass; no atomics. Default fast mode: warp-shuffle + one FP64 `atomicAdd` per block
(bitwise-nondeterministic across runs, ~zero cost). Both modes exist from day one because the
debugging ladder (§8 of proposal) requires bit-reproducibility.

---

## 4. GPU data structures & layout

**Why this is the best:** pure SoA with power-of-two padding gives fully-coalesced 32-byte sector
loads for every access pattern the three kernel families have; device-resident arena allocation
kills the per-call `cudaMalloc` thrash found in the current code; and the layout is identical to
what rho_build already produces, so fusion needs no transposes. Textures/constant memory buy nothing
here (regular streaming access); `const __restrict__` gets the same `LDG.CI` path.

- **SoA everywhere.** Never libxc's interleaved `rho[2i],rho[2i+1]` polarized layout in our arrays
  (Tier-1 adapter re-strides at the kernel boundary — a register shuffle, not a memory pass).
- **Arena:** one `XcArena` per SCF scope holding all grid arrays, allocated once via the WP1 pool
  (`cudaMallocAsync`/pool), 256-byte aligned, padded to `np_pad = round_up(np, 512)` so vectorized
  `double2`/`float4` loads never peel.
- **Pinned staging** only at SCF setup (grids, weights up once) and teardown (nothing comes back
  but scalars and, on request, dump-stage snapshots for the bisect harness).
- **Kernel launch signature sketch (Tier 0, GGA family):**
```cuda
template <class F, class T>                     // F: functional functor, T: fp32/fp64 pointwise
__global__ void __launch_bounds__(256, 2)
xc_gga_fused(const T*  __restrict__ rho,        // [np_pad]
             const T*  __restrict__ gx,         // …gy, gz [np_pad] (SoA components)
             const T*  __restrict__ gy,
             const T*  __restrict__ gz,
             const double* __restrict__ w,      // FP64 weights, always
             double*   __restrict__ wv_rho,     // outputs: exactly vmat_build's diet
             double*   __restrict__ wvg_x,      //  2w·v_σ·∇ρ components
             double*   __restrict__ wvg_y,
             double*   __restrict__ wvg_z,
             double*   __restrict__ exc_acc,    // [nsys] FP64 device accumulators
             const int64_t* __restrict__ sys_of, int nsys, int64_t np);
```

---

## 5. Kernel design & launch configuration (worked example: r²SCAN, the expensive case)

**Why this is the best:** one-thread-per-point with grid-stride is optimal for a pointwise map —
no shared memory, no cooperative groups, no dynamic parallelism (all three would add sync cost for
zero reuse; there *is* no inter-point data reuse in XC). The only cross-thread operation is the
energy reduction, which warp shuffles handle at negligible cost. Register pressure, not occupancy,
is the binding constraint for mGGA — and compute-bound kernels with high ILP tolerate 25–33%
occupancy fine (Volta+ latency hiding needs far fewer warps than the occupancy calculator implies).

```
Kernel body (per thread, grid-stride loop over points):
  1. load ρ, g=(gx,gy,gz), τ           → 5 loads (coalesced, __ldg)
  2. σ = g·g;  promote to FP64 if T=fp32 and hazard flags fire
  3. F::eval(ρ, σ, τ) → ε, v_ρ, v_σ, v_τ   (all in registers; ~1–2 kFLOP for r²SCAN)
  4. write w·v_ρ, 2w·v_σ·g (3 comps), w·v_τ  → 5 stores
  5. e_local += w·ρ·ε                  (register)
  end loop
  6. warp-shuffle reduce e_local → block partial → one FP64 atomicAdd into exc_acc[sys]
```

- **Launch:** 256 threads/block, `blocks = min(32*SMs, ceil(np/256))`, grid-stride. One config for
  all families; no tuning per GPU needed (bandwidth-bound kernels are launch-insensitive above
  ~50% occupancy of the memory system).
- **Registers:** generated r²SCAN order-1 code runs ~160–220 regs in FP64. With
  `__launch_bounds__(256,2)` (≤128 regs) nvcc will spill; therefore for mGGA use
  `__launch_bounds__(128)` and accept 2 blocks/SM ≈ 33% occupancy — measured on comparable kernels
  (ExchCXX, GauXC) this loses <5% vs. unbounded while preventing spills. Verify with
  `-Xptxas -v` (spill count must be 0) and Nsight Compute; **kernel splitting (TeraChem-style,
  ε+v_ρ | v_σ+v_τ) only if spills appear** — it re-reads inputs, so it must be justified by
  measurement, not applied preemptively.
- **Shared memory:** zero (except the 32-entry block-reduction scratch). Don't spend it; it buys
  nothing for pointwise math and caps occupancy.
- **Expected rates** (design targets, assumption-ledger style — to be confronted with measurement
  at M1/M3 gates):

| Kernel (FP64, fused) | B/pt | FLOP/pt | H100 (BW-bound est.) | RTX 4090 est. |
|---|---|---|---|---|
| LDA (PW92) | ~32 | ~150 | ~10 Gpt/s | ~4 Gpt/s |
| PBE | ~104 | ~500 | ~3 Gpt/s | ~1.0 Gpt/s (FP64) / ~2.5 (mixed) |
| r²SCAN | ~136 | ~2000 | ~1.5 Gpt/s | compute-bound → mixed path matters |

  At even 1 Gpt/s, the 1.2M-point/200-atom system costs ~1 ms — i.e. XC-eval disappears from the
  profile, which is the actual goal.

---

## 6. Performance-engineering knobs

**Why this is the best:** every knob below attacks the measured bottleneck class (bytes, then
transcendentals); nothing speculative survives.

- **Libraries:** none in the pointwise kernel. cuBLAS/CUTLASS grouped GEMM own rho_build/vmat_build
  (WP1); CUB owns nothing here (hand warp-shuffle reduction is 20 lines and avoids a dependency in
  a header-only tier); cuFFT owns Poisson and the future VV10 factorization. cuSOLVER/tensor-core
  libraries: not applicable to XC.
- **Math:** never global `--use_fast_math` (breaks the accuracy contract and libxc-oracle
  agreement). Selectively: `fma()` everywhere (write formulas fma-shaped), `cbrt/rsqrt` instead of
  `pow` (see §1.4), `erfc` device intrinsic for RSH (FP64), `__expf/__logf` only inside the FP32
  mid-SCF path where the A/B gate has signed off.
- **Unrolling/prefetch:** grid-stride loop with `#pragma unroll 2` and independent `double2` loads
  gives enough ILP; hardware prefetching handles unit-stride streams — manual prefetch is noise here.
- **Compilation:** per-functional-family TUs (`xc_lda.cu`, `xc_gga.cu`, `xc_mgga.cu` instantiate
  their functor sets) → parallel builds, selective linking; `-rdc=false`; `-lineinfo` in CI perf
  builds; fatbin for {consumer, datacenter} SM archs per the CI contract (D-3).
- **CUDA graphs:** the whole R0 SCF sweep captured once per shape class (§2.2).

---

## 7. Testing & validation pipeline

**Why this is the best:** it reuses the ladder TIDES already mandates (proposal §8) instead of
inventing a parallel harness, it pins the oracle (CPU libxc FP64) that every tier shares, and its
tolerances are stated per rung so a failure localizes itself.

**Rung 0 — pointwise oracle sweep (per functional, per derivative, per precision path):**
- Input lattice: ρ ∈ log-space[1e-15, 1e4] (64 pts) × s ∈ [0, 10³] (log, 48 pts) ×
  τ ∈ {τ_vW·(1+10^k)} × ζ ∈ {0, ±0.3, ±0.9, ±(1−1e-12)} — plus libxc's own regression points.
- Tolerance: FP64 tiers ≤ 5e-14 relative away from thresholds; behavior *at* thresholds must match
  libxc's `dens_threshold`/`sigma_threshold` semantics exactly (documented per functional).
  FP32 path: ≤ 5e-6 relative pointwise, and the *integrated* budget below governs.
- Runs on both CI GPUs (RTX + datacenter runners already planned); compute-sanitizer memcheck +
  racecheck as a gate, per proposal.

**Rung 1 — operator/energy:** He, Ne, H₂O, benzene, Fe (spin-polarized) SCF vs PySCF/libxc
references ≤1e-8 Ha (extends the existing T3.5 observable); GGA/mGGA V_xc validated through the
**adjoint route**: FD of E_xc w.r.t. density-matrix perturbations vs. Tr[V_xc·ΔP] ≤1e-9 — this is
the test that catches the dropped-divergence-term class of bug in `libxc_wrapper.hpp` structurally.

**Rung 2 — forces/egg-box:** 5-point FD forces nightly (existing WP9 machinery); egg-box scan with
analytic ∇ρ (must beat the current FD-σ implementation — regression-tracked).

**Rung 3 — precision A/B:** nightly mixed-vs-FP64 on the 12-system gauntlet subset, ≤0.5 meV/atom
(existing budget). **The FP32 pointwise path ships disabled and is enabled per-functional-family
only by this gate.** Coverage matrix (functional × tier × precision × pass/fail) auto-generated
into docs each release — this is also our contribution back upstream (libxc has no published GPU
coverage matrix; ours becomes the community reference).

---

## 8. Build & portability

**Why this is the best:** it keeps upstream libxc bit-for-bit unforked (patch-set of ~zero), gives
CMake-native builds that upstream lacks, and reuses the proposal's existing HIP strategy instead of
adding a new portability layer.

```
external/libxc/                 # pinned submodule, UNMODIFIED upstream
core/grid/xc/
├── xc_engine.hpp               # the §2.3 contract (host API)
├── functionals/                # Tier 0: header-only __device__ functors
│   ├── lda_pw92.cuh  gga_pbe.cuh  gga_b88_lyp.cuh  mgga_r2scan.cuh  …
│   └── compose.cuh             # Weighted<F,c,…> pack
├── kernels/                    # fused family kernels (xc_lda.cu, xc_gga.cu, xc_mgga.cu)
├── tier1/
│   ├── CMakeLists.txt          # compiles selected external/libxc/src/{maple2c,work_*} with
│   │                           #   nvcc -x cu  + libxc's own GPU_FUNCTION macro path
│   │                           #   (upstream is autotools-only for CUDA; this wrapper is the
│   │                           #    only thing we maintain — ~100 lines of CMake)
│   └── peephole/               # pow→cbrt source pass applied at build time to generated C
│                               #   (build artifact, never committed → still zero fork)
└── tests/                      # rungs 0–1 harnesses
```
- Targets: `tides_xc` (Tier 0, header-only + family TUs), `tides_xc_catalog` (Tier 1, per-family
  object libs, selective link — avoids the 700-functional compile-time wall), CPU fallback builds
  from the same functor headers (they're plain arithmetic; `__device__` via macro = the existing
  team pattern).
- **HIP:** Tier 0 functors are portable by construction (arith + intrinsics behind the WP1 device
  macro layer); Tier-1 catalog via libxc's own HIP path. Quarterly MI300 gate as already scheduled.
- Static asserts on all layout contracts (alignment, padding, SoA strides) at the `XcGridIn`
  boundary; `-Werror` on narrowing.

---

## 9. Roadmap & milestones (replaces Master Plan §10; fits WP3 cadence)

**Why this is the best:** it front-loads the entire architectural win (residency + fusion — the
20×) into the first six weeks with just two functionals, then widens coverage on a proven skeleton.
Coverage-first roadmaps (the Master Plan's) spend months porting families before the pipeline
exists to make them fast.

| M | Weeks | Deliverable | Gate (quantitative) |
|---|---|---|---|
| **M0** | 1–2 | Honesty pass on current code: device-resident LDA path (arena, no per-call malloc/copies/global syncs); analytic PW92 derivative; delete CPU-libxc-on-GPU-reduction PBE path; profile baseline | GPU LDA ≥10× faster per SCF iter than current `XCEvalLdaCuda` incl. transfers; He/Ne 1e-8 Ha observable still green |
| **M1** | 3–6 | Tier-0 skeleton: `xc_engine` contract; fused GGA kernel; PW92 + PBE functors; analytic ∇ρ from rho_build; adjoint V_xc in vmat_build; rung-0/1 harness | Zero PCIe bytes per SCF iter except scalars; XC phase ≤5% of SCF step on 200-atom test; ≥60% of BW roofline (both CI GPUs); adjoint test ≤1e-9 |
| **M2** | 7–12 | Coverage-of-the-hot-set: BLYP, revPBE, RPBE, PBE0/B3LYP local parts; spin polarization (spin-scaling route); R0 batched layout + CUDA graphs | Rung-0 matrix green for all Tier-0 functionals; R0: ≥10⁴ LDA/PBE single-point XC-phases/hour equivalent on one GPU; graphs cut launch overhead ≥5× on 30-atom batch |
| **M3** | 13–18 | mGGA: τ plumbing through rho_build/vmat_build; TPSS, r²SCAN (SCAN FP64-only); HSE short-range exchange (device E₁/HJS); BR89 via Proynov fit | r²SCAN rung-0/1 green; register report: 0 spills; HSE06 local part validated vs CPU libxc ≤1e-13 pointwise |
| **M4** | 19–24 | Tier-1 catalog: CMake wrapper over unmodified libxc-CUDA + pow-peephole; SoA↔libxc-layout adapter; auto-generated coverage matrix; Tier-2 CPU batch fallback | ≥95% of libxc catalog passes rung-0 on GPU; the ~24 exotics documented with fallback route; matrix published in docs |
| **M5** | 25–30 | Precision program: FP32 mid-SCF path behind the A/B gate; deterministic-mode reductions; stress-term outputs; f_xc (order 2) for the hot set (TDDFT prep) | Nightly A/B ≤0.5 meV/atom with mixed mode ON for gated families; RTX-class speedup of mixed vs FP64 ≥2× on PBE/r²SCAN documented; deterministic mode bit-reproducible across 100 runs |
| **M6+** | later | VV10/rVV10 nonlocal kernel (enables ωB97X-V, B97M-V, SCAN+rVV10); order-3/4 derivatives on demand | separate design note; not in the critical path |

Failure protocol per gate: any miss >2× triggers a written model revision (same discipline as
proposal §7), never silent target erosion.

---

## 10. Scientific-accuracy pitfalls checklist (CPU-FP64 → GPU migration)

1. **Threshold mismatch**, not arithmetic, is the #1 source of GPU-vs-libxc discrepancies — pin
   `dens_threshold`/`sigma_threshold`/`tau_threshold` per functional to upstream values.
2. **FMA contraction** changes last-bit results vs. CPU; compare with tolerance bands, never
   bitwise, against the oracle (deterministic mode is for GPU-vs-GPU reproducibility).
3. **SCAN's α switching** amplifies τ noise — r²SCAN default, SCAN FP64-only (see §3.2).
4. **Spin limits** ζ→±1: (1−ζ) cancellation in FP32 — always-FP64 spin factors.
5. **RSH attenuation** erfc(μr)·small-ρ products underflow FP32 — always-FP64.
6. **Grid-weight sums** over 10⁸ points lose ~3 digits in naive FP32 accumulation — FP64
   accumulators are non-negotiable (already fixed in §3.1).
7. **FD σ vs analytic ∇ρ** biases forces (current wrapper) — analytic gradients from the density
   build, enforced at M1.
8. **Energy-only validation hides potential bugs** (the dropped divergence term would pass an
   E_xc test on a converged density) — the adjoint test (rung 1) exists precisely for this.

---

## 11. Relation to Master Plan document

`TIDES_GPU_XC_Master_Plan.md` remains the survey/literature record (its §§3–5, 8–9 are referenced
throughout). Its §6 (Maple fork), §10 (roadmap) are superseded by §§0.1, 2, 9 here. Risk table
(§11 there) updates: "Maple/C99-vs-nvcc incompatibility" and "compilation-time" risks collapse to
low (we compile upstream's own CUDA-tested sources); a new risk appears — "upstream libxc CUDA
regressions on untested functionals" — mitigated by our rung-0 coverage matrix in CI, which is also
the thing worth contributing back.

*Design v1.0 — July 2026. Decision needed at kickoff: none blocking; M0 can start immediately.*
