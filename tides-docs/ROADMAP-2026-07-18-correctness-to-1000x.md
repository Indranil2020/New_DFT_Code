# TIDES Roadmap 2026-07-18: Correctness → Parity → Throughput → Scale

**Status:** For team review. No code has been changed with this document; every claim below carries a file:line citation into the working tree as of 2026-07-18 (commit 7004681 + uncommitted working tree).
**Inputs:** `tides/bench/profiling_results/four_route_ground_report_2026-07-18.md` (four-route matrix), `tides/bench/optimization_ledger_2026-07-17.md` (OPT-1..16, BN-1..7), `tides/bench/audit_ledger_gga_screen_2026-07-17.md` (BUG-1..7), `tides-docs/CLINICAL-AUDIT-UPDATE-2026-07-15.md`, `TIDES_5yr_proposal.md`, plus a fresh line-level code audit (this document).
**Day-0 baseline:** `tides/bench/profiling_results/baseline_2026-07-18/` (fair-comparison ladder CH4→C10H22; all phase exit criteria are measured against these numbers).

---

## 1. Executive summary

### 1.1 Where we are

| Fact | Evidence |
|---|---|
| TIDES is 4–16× **slower** than PySCF on CH4/H2O across all four routes (PBE, grid_h=0.3 protocol) | four_route_ground_report_2026-07-18.md §"Relative performance" |
| Best case ever measured: C2H6 at 0.87× of gpu4pyscf (B3LYP PP, grid_h=0.5); typical small molecules 0.26–0.32× | optimization_bench_results.json |
| 2 of 4 routes are **physics-broken**: PP CPU (CH4 −4.6 Ha, H2O diverges), AE (CH4 −7.5 Ha vs PySCF) | four-route report §"Key route-specific findings" |
| SCF needs 12–13 iterations where gpu4pyscf needs 5–6 — and the day-0 baseline is far worse at size: 60–250 iterations beyond 4 heavy atoms, with 6 of 16 TIDES routes hitting max_iter unconverged despite baked-in level shifts | optimization_ledger_2026-07-17.md BN-1; baseline_2026-07-18/README.md |
| Day-0 fair ladder (B3LYP, h=0.5, warm): TIDES beats gpu4pyscf at ≤5 atoms (up to 2.1×) but falls to 0.27–0.43× at 26–32 atoms, driven almost entirely by iteration count; per-iteration build_H is ~9× faster than gpu4pyscf per cycle at C10H22, and xc_eval is ~100% of build_H | baseline_2026-07-18/README.md |
| GPU pipeline silently reverts the **whole run** to CPU beyond ~12 atoms (build_H 600 → 9300 ms, ~15×) | BUG-6; `nao_driver.hpp:1200–1216` |
| "MPI route" is OpenMP-only; `NaoDriver` never calls `MpiWorld` | four-route report finding 4 |
| Independent audit: ~50% of the proposal genuinely implemented; tile substrate "partially decorative"; batched R0 mode **not started** | CLINICAL-AUDIT-UPDATE-2026-07-15.md |

### 1.2 What "1000× vs PySCF" honestly means

The proposal (TIDES_5yr_proposal.md §1.2) never promises 1000× on a single molecule — it promises **≤2× of gpu4pyscf cold-start** at the small end and calls beating analytic-integral codes on a cold 10-atom single point "launch-latency-bound and scientifically uninteresting." The 1000× goal (optimization_ledger_2026-07-17.md line 3) is only reachable as a **product of three orthogonal factors**, each a real proposal pillar:

| Factor | Mechanism | Magnitude | Unlocked by |
|---|---|---|---|
| Large-N scaling | O(N) SP2/submatrix vs PySCF O(N³–N⁴) | 10²–10³× at 10³–10⁴ atoms | Phases 3 + 4d |
| Batched throughput | R0 many-molecule mode (proposal differentiator #3, not started) | ~10× molecules/hour/GPU | Phase 4a |
| MD amortization | XL-BOMD ~0 SCF solves/step (already verified: 0.001 solves/step, 17 µHa/atom/ps drift) | 10–50× on MD campaigns | works today; needs Phases 3/4 engine |

Multiply where a workload uses all three (large-system batched MD screening) and 10³–10⁴× is honest. On one small-molecule SCF the ceiling is parity-to-2× — and we are not there yet either.

### 1.3 Governing principles

1. **No performance work counts until Phase 0 is green.** Two of four routes currently produce wrong physics; any speedup measured on wrong energies is fiction.
2. **Every phase closes by re-running the four-route matrix** (AE/PP × CPU/GPU) and the fixed benchmark ladder against the day-0 baseline. A route regression blocks the phase close.
3. **Fix causes, not symptoms.** BUG-6's current "fix" (detect OOM → silently run everything on CPU) is the canonical anti-pattern this roadmap eliminates.

---

## 2. Root-cause dossier (the two physics bugs)

### 2.1 PP CPU route: missing frame rotation in the semi-on-site correction

`BuildAnalyticVextPP` (`tides/core/scf/nao_driver.hpp:515–749`) builds V_ext for pseudopotential runs as: sample V_loc^A on the Cartesian grid (:544–565) → project to the basis by GEMM (:568) → zero the on-site block and refill it analytically (:570–577, :697–741) → **"semi-on-site" correction** (:579–693): for basis pairs (i,j) both on atom B≠A with same l and same m, *replace* the grid-projected matrix element with an analytic radial×angular quadrature.

**The defect:** the angular part evaluates `Y_lm(θ, 0)` with `phi_factor = (m==0) ? 2π : π` (:640–687) — an expression that is only valid if the m quantization axis is the A→B bond axis. The basis functions' real spherical harmonics are quantized along the **global z-axis**, and no Wigner rotation is applied. Consequences:

- For any p/d function on a bond not parallel to z, the replaced element is wrong (magnitude and sign).
- The true matrix element is not even diagonal in (l,m) in the global frame — the correction *should* couple different-m (and different-l) pairs, which the code leaves at their grid values. The result is an inconsistent mixture of two frames.
- CH4 (bonds along ±(1,1,1)-type directions) lands ~4.6 Ha too negative; H2O oscillates and never converges. The **GPU PP path does not apply this correction** (`BuildVlocDevice`, `pp_build.cu:487–507` + on-site analytic only) and is the only physics-plausible route — exactly what this diagnosis predicts.

**Cheap diagnostic (hours):** env-gate the semi-on-site block off on CPU; CPU PP must then agree with GPU PP to ≤1e-8 Ha. If it does, the diagnosis is confirmed without writing any new physics.

**Fix options (Phase 0, P0.4):** (a) rotate each (l, {m}) block into the bond frame (Wigner d-matrices for real harmonics, l≤3 is closed-form), apply the 1-D correction there, rotate back — the standard SIESTA/OpenMX approach; or (b) drop the bond-frame shortcut and compute the correction by dense radial×Lebedev quadrature (exact, slower, only runs once at setup). Recommendation: (b) first as the oracle, then (a) as the fast path validated against (b).

### 2.2 AE route: unregularized −Z/r sampled across the nuclear singularity

`BuildAnalyticVext` (`nao_driver.hpp:405–507`) handles all-electron V_ext as: sample **bare −Z/r** on the Cartesian grid — the only guard is `r < r_smooth = 1e-6` Bohr (:438–450), which essentially never fires — then replace the *on-site* block analytically (:466–499). All **cross-atom** matrix elements ⟨φ_i^B|−Z_A/r|φ_j^C⟩ are integrated numerically on the h=0.3 Bohr grid straight across the singularity at atom A.

**Why this diverges slowly (or never):** a grid quadrature of a 1/r singularity does not converge as any positive power of h — the result depends on where grid points happen to land relative to the nucleus (this is the same mechanism as fixed BUG-5, where a near-coincident grid point produced −1.6M Ha; the r<1e-6 clamp fixed only the exact-coincidence catastrophe, not the quadrature error). The error grows with Z and with the number of neighbor basis pairs overlapping the core region: CH4 (Z=6, 4 C–H bonds, 8 H–H/C–H pair families) is off by −7.5 Ha while H2O is off by only −0.5 Ha. The day-0 baseline confirms the signature decisively: AE errors vs gpu4pyscf span **−26.7 to +45.8 Ha with geometry-dependent sign** (NH3 +45.8, C2H6 +18.0, C8H18 −26.7 — baseline_2026-07-18/README.md), which only a placement-dependent singular quadrature produces; a systematic physics offset would have one sign. The code comment claims "erf smoothing"; none is implemented.

**Fix (Phase 0, P0.3) — the standard erf split:**
`−Z/r = −Z·erf(r/r_c)/r  (smooth, grid-integrable, r_c ≈ 1.5–2h)  +  −Z·erfc(r/r_c)/r  (short-range)`.
The long-range term goes on the grid (bounded, converges as the grid's normal order). The short-range term is nonzero only within ~3r_c of nucleus A and is evaluated per atom-pair on the radial/angular quadrature (same machinery as 2.1's fix; on-site case reduces to the existing radial integral). This also retires BUG-5's clamp with real mathematics. **Interplay warning:** the current on-site analytic replacement was tuned alongside the broken cross-atom terms; land P0.1–P0.4 as one unit and re-validate, or fixed pieces will look "worse" against compensating errors.

### 2.3 Secondary numerics (armor, not primary causes)

- The radial integrals in `BuildAnalyticVext` (:485), `BuildAnalyticVextPP` (:605, :714), and the KB on-site refill (`dr = nao_rg[1]-nao_rg[0]`, :1546–1589) all assume uniform radial spacing. The NAO generator's grid **is** currently uniform (`nao_generator.hpp:163–166`: `r[i] = i·r_max/(n_r−1)`), so this is consistent today — but it is a landmine (any future log-grid basis silently breaks all three sites), and a uniform grid to r_max=40 Bohr under-resolves the near-nucleus region for AE second-row atoms. Transplant the per-interval trapezoid already used correctly at `nao_driver.hpp:271, :299, :343`; audit `two_center_builder.hpp:219` (`h = r[1]-r[0]`) the same way.
- PP V_loc interpolation from the pseudopotential file grid is done properly with `upper_bound` (handles non-uniform pp grids); keep.

---

## 3. Phase 0 — Correctness gate (BLOCKING; ~3–4 weeks)

**Objective:** all four routes physically correct. Exit gates are hard; nothing else in this roadmap is measurable before this closes.

| ID | Task | Files | Effort |
|---|---|---|---|
| P0.1 | Per-interval `dr` transplant at the 3 uniform-dr sites + `two_center_builder.hpp:219` audit; add a CI grep gate for the `r[1]-r[0]` pattern | `nao_driver.hpp:485, :605–:714, :1546–1589` | S |
| P0.2 | Radial-integral unit tests vs closed forms (Gaussian moments, hydrogenic ⟨1s\|−Z/r\|1s⟩=−Z², norms), on uniform **and** log grids, ≤1e-8 rel | new test under `tides/core/scf/tests/` | S |
| P0.3 | Genuine erf-split AE V_ext (long-range on grid, short-range analytic per atom-pair); retire the BUG-5 clamp; r_c convergence study (r_c = 1.5h, 2h, 3h) | `nao_driver.hpp:405–507` + GPU `VlocGridKernel` (`pp_build.cu:325–377`) for the AE case | M |
| P0.4 | PP semi-on-site fix: dense radial×Lebedev oracle first, then bond-frame Wigner rotation as fast path; GL angular order convergence test (4/8 → 16/32); the same oracle validates the KB on-site block | `nao_driver.hpp:579–693, :1546–1589` | M |
| P0.5 | Single-atom oracle: isolated C, N, O total energies vs `AtomicLDA::Solve` (`tides/core/basis/atomgen/atomic_lda.hpp`), same XC — isolates V_ext/core handling with zero cross-atom terms | new test | S |
| P0.6 | Per-term H dump harness: env-flag dump of S, T, V_ext, V_nl, and J[P], V_xc[P] at a **fixed** input P + per-term energies; comparison scripts: (a) analytic terms vs PySCF with the NAO radial functions least-squares fitted to ~10–14 Gaussians (µHa fidelity → element-wise match to fit error), (b) PP terms vs a dense-quadrature Python oracle | dump hook in `nao_driver.hpp`; runners + `tolerances.yaml` under `tides/verification/` | M |

**Diagnostics to run first (before writing fixes):** (1) semi-on-site off → CPU PP ≡ GPU PP check (§2.1); (2) P0.5 single-atom sweep — if isolated C is already wrong, the AE bug has an on-site component too; if clean, it is purely cross-atom (§2.2).

**Exit criteria (all hard):**
- Radial unit tests green on both grids; single-atom C/N/O ≤0.5 mHa vs AtomicLDA.
- Per-term S/T/V_ext vs PySCF-on-fitted-basis: max element ≤1e-5.
- CH4, H2O, NH3, C2H4: AE vs PySCF (matched basis) ≤1 mHa/atom; PP routes all converge; PP-vs-AE atomization-energy consistency ≤2 mHa; **CPU ≡ GPU per route ≤1e-8 Ha**.
- Four-route matrix re-run: 4/4 physically plausible and converged.
- P0.1–P0.4 land as **one validated unit** (compensating-error risk, §2.2).

**Speedup delivered: none.** Possibly slightly slower. Everything after this is real.

---

## 4. Phase 1 — SCF convergence parity (~2–3 weeks; after P0)

**Objective:** iteration parity with gpu4pyscf (6–8). On the small-molecule set this is ~2×; on the day-0 fair ladder it is far larger — TIDES needs 60–250 iterations beyond 4 heavy atoms and 6 of 16 routes never converge (baseline README §"iterations"), while per-iteration cost is already competitive. Expect part of the pathology to disappear with Phase 0 (converging onto wrong physics is hard); Phase 1 owns whatever remains. Context: the mixer is already commutator-DIIS (`scf_driver.hpp:234–257, :341–415`, depth 8, α=0.5) — the same family PySCF uses — so the gap is guess quality and DIIS details, not the algorithm class.

**On the VASP question (team note):** VASP's celebrated convergence machinery (Kerker-preconditioned Broyden/Pulay charge mixing) targets **charge sloshing in metals and large periodic cells** — long-wavelength instability of the Hartree kernel (~4π/q²). For isolated molecules that instability barely exists; PySCF converges CH4 in 5–6 iterations with plain commutator-DIIS plus a **real SAD guess**. So: adopt VASP-grade *discipline* now (good guess, preconditioned residual metric, automatic stabilizers), and schedule actual Kerker/Broyden for the periodic/metallic milestones (Phase 4d/R3 era) — where it will be mandatory. The in-tree `broyden_mixer.hpp` (complete Broyden+Kerker, currently **dead code with zero call sites**, dense n×n Jacobian) is quarantined until then and needs a limited-memory rewrite before large-N use.

| ID | Task | Files | Effort |
|---|---|---|---|
| P1.1 | **Real SAD guess** (the big lever): converged spherical-atom densities from `AtomicLDA::Solve`, occupations mapped onto per-atom NAO diagonal blocks by matching (n,l); cache per (element, basis). Replaces the uniform diagonal fill at `nao_driver.hpp:2802–2842` | `nao_driver.hpp`, `atomic_lda.hpp` | M |
| P1.2 | DIIS quality: S-metric error vector e = Xᵀ(FPS−SPF)X; start DIIS at iter 1–2 with a 2-step damping ramp (replacing the fixed 70/30 DM-Pulay damp); history reset on stagnation; keep coefficient sanity checks | `scf_driver.hpp:341–415, :943–1086` | M |
| P1.3 | Convergence criteria: add RMS(ΔP) < 1e-6 AND max‖FPS−SPF‖ < 1e-5 alongside \|ΔE\| (currently energy-only, `scf_driver.hpp:901` — corrupts both results and iteration-count benchmarking) | `scf_driver.hpp` | S |
| P1.4 | Automatic level shift: trigger on small HOMO–LUMO gap or growing DIIS error, 0.2–0.5 Ha decayed to 0 as the error falls; retires the manual C6H6 `level_shift=0.2, 250 iters` regime (`benchmark_fair_comparison.py:71–84` bakes this workaround into every benchmark ≥5 atoms) | `scf_driver.hpp:630–663` | S–M |
| P1.5 | Quarantine `broyden_mixer.hpp` behind a flag with a doc-comment pointing at the periodic/metallic milestone | — | S |

**Exit criteria:** iterations ≤ gpu4pyscf+1 on CH4/H2O/NH3/C2H4/C6H6/glycine with **zero manual settings** (no hand level shift anywhere in the bench scripts); converged energies unchanged vs Phase 0 references ≤1e-8 (the guess/mixer must not move the answer); C6H6 ≤15 iterations.

**Cumulative vs day-0:** small molecules ~2×; 20–50 atoms ~2×.

---

## 5. Phase 2 — Per-iteration + setup GPU completion (~5–8 weeks; after P0, parallel with P1)

**Objective:** kill the remaining CPU serialization and PCIe round-trips; hit the proposal's **≤2× gpu4pyscf cold-start** bar on small molecules. The per-iteration hot loop (ρ GEMM `rho_build.cu:528–604`, device-resident cuFFT Poisson `poisson_fft.cu:690–892`, XC kernel `xc/kernels/reduce.cuh:29–79`, screened vmat GEMMs incl. the batched GGA 4-in-1 `vmat_build.cu:700–793`) is already decent; what remains is everything around it.

Ordered tasks (quick wins first):

1. **P2.1 GpuArena fix [S; ~20 lines; do first].** `gpu_arena.hpp`: `Free()` stores `{ptr, 0}` (:69) but `Alloc()` requires `cached_size >= bytes` (:47) → the pool has **never reused a block**; every Alloc is a fresh `cudaMalloc` and up to 32 freed blocks sit as unusable dead VRAM (inflating pressure right before the BUG-6 check). Track sizes in a live-allocation map, or replace the internals with `cudaMallocAsync`/`cudaMemPool` behind the same interface.
2. **P2.2 Workspace + pinned staging [S].** Stop reallocating `tmp/Hp/Hp_work/C_evec/P_new/P_next/B/FP/SP/e` every iteration (`scf_driver.hpp:237–252, :424, :570, :668, :758, :945`); pinned buffers for all recurring transfers; pinned upload of φ/∇φ (currently pageable at `st_gpu.cu:69, :77`, `nao_driver.hpp:1219, :1238`).
3. **P2.3 st_gpu 3-block extraction [S].** `st_gpu.cu:106–150` computes the full [3n×3n] T_full GEMM, copies all 9n² back, and extracts 3 diagonal blocks in a host triple loop — 3× wasted GEMM plus wasted D2H. Do 3 block GEMMs, or extract on device and download 3n².
4. **P2.4 Device-resident SCF core [M].** Keep P, H, X, H′, V_H, V_xc, C on device across the iteration: AssembleH on GPU (now host, `nao_driver.hpp:2037`), H′=XᵀHX via cuBLAS (now host dgemms, `scf_driver.hpp:417–625`), density build on device. Eliminates the current per-iteration P(H2D)/V_H,V_xc,H(D2H)/H′(H2D)/evecs(D2H) PCIe round-trips; make the `exc` reduction async (blocking sync today at `nao_driver.hpp:2011–2013`).
5. **P2.5 Eigensolve unification [M].** UKS/k-point paths call CPU `dsygvd_` every iteration (`solvers/dense/batched_eig.hpp:52–95` via `nao_driver.hpp:2953–2957, :3404`), re-factorizing S each time; route them through the RKS pattern (Löwdin once + cuSOLVER standard solve, `scf_driver.hpp:702–724`).
6. **P2.6 V_nl to GPU + out of the hot path [M].** KB assembly is CPU-only: full-grid Y_lm fields per atom/channel (`nao_driver.hpp:1494–1528`) + an O(n²·n_β²·n_m) accumulation loop (:1593–1611), 0.5–1.3 s. It depends only on geometry/basis — compute ⟨β|φ⟩ once at setup on GPU; per-iteration cost becomes zero.
7. **P2.7 Setup pipeline [M].** φ/∇φ table evaluation on GPU (CPU OpenMP today, `nao_driver.hpp:922–980`); V_ext grid sampling on GPU for the CPU-route parity.
8. **P2.8 Streams & stragglers [S–M].** Wire the dead `xc_stream` (`nao_driver.hpp:1763–1765` — created, never used) to overlap XC with Poisson; cache cuFFT plans on the non-resident fallback (`poisson_fft.cu:237, :467–487` rebuilds per call); give mGGA vmat the screened-GEMM treatment (`vmat_build.cu:453–489` is always the naive one-block-per-pair kernel); delete or fix `WeightedVmatMixedKernel`'s unused 2 KB/thread local array (`pp_build.cu:64`).
9. **P2.9 CUDA-graph the iteration [M–L; last].** With 4–6 done, capture ρ→Poisson→XC→vmat→assemble→transform→eigensolve→DM as one graph (scaffolding exists: `tile::CudaGraphSCF`, `nao_driver.hpp:1774`). Known risk: cuSOLVER capture compatibility — fallback is graphing everything except the eigensolve, which still removes nearly all launch latency at small n.

**Exit criteria:** nsys trace of a CH4 SCF shows <5% wall in CPU inside the loop; zero per-iteration `cudaMalloc`; per-iteration D2H limited to convergence scalars; setup ≤1 s at 12 atoms (from 2–5 s); **cold CH4/H2O/NH3 ≤2× gpu4pyscf**; four-route matrix and all Phase-0 gates still green.

**Cumulative vs day-0:** small molecules ~5–10× (iteration parity × per-iter/setup/latency) — i.e., at or under the proposal's ≤2× bar; 20–50 atoms ~2–3× (still fallback-poisoned until Phase 3).

---

## 6. Phase 3 — Memory scaling: the real >12-atom fix (~6–10 weeks; after P2.1/P0)

**Objective:** replace "pre-flight VRAM estimate → silent full-CPU fallback" (`nao_driver.hpp:1200–1216`, BUG-6) with an architecture whose VRAM footprint is O(active data), so a 12 GB consumer card runs 100-atom systems **on the GPU** — the proposal's "democratic" commitment.

**Why it collapses today:** the device tables `d_phi` (n_basis × n_grid × 8 B) and `d_grad_phi` (3×) (`nao_driver.hpp:1218–1241`) are **dense**, though every NAO is strictly zero beyond its cutoff radius — the defining property of the basis (proposal §1) is ignored by the storage layer. n_grid grows with bounding-box volume and n_basis with atom count, so the tables grow ~O(N²) and 12 atoms already exhausts 12 GB at GGA (needs φ, ∇φ, 4-plane gradient + screened + orbital buffers — BN-7/BUG-6).

| ID | Task | Effort |
|---|---|---|
| P3.1 | **Grid-blocked sparse φ storage** (structural centerpiece): partition the grid into blocks (start 16³); per block store the list of basis functions whose r_cut sphere intersects it + φ/∇φ values for only those pairs. ρ build = per-block skinny GEMMs (grouped-GEMM machinery exists: `core/tile/gemm_grouped.cu`); vmat = per-block GEMM + scatter-add into H. Storage drops from O(n_basis·n_grid) to O(Σ_block n_active·n_pts) ≈ **O(N)** (each grid point sees only nearby atoms' functions). This is the standard SIESTA/ABACUS/GPAW design. Keep the dense path compiled behind `TIDES_DENSE_PHI=1` as the A/B oracle (mirror the `TIDES_GRID_SCREEN` pattern). New `grid_blocking.hpp`; rewires `rho_build_gpu.hpp`, `vmat_build_gpu.hpp`, `st_gpu.cu`, driver wiring in `nao_driver.hpp` | L |
| P3.2 | **Chunked streaming + loud degradation:** if the blocked working set still exceeds VRAM, stream block batches (double-buffered H2D on a copy stream). **Delete the silent CPU fallback**; policy = (i) chunked GPU, (ii) loud warning + spill, (iii) hard error with an actionable message. Full-CPU mode survives only as an explicit debug flag | M |
| P3.3 | 64-bit (or block-relative 32-bit) screening indices (`pp_build.cu:211`, `vmat_build.cu:177` cap at 2³¹ grid points); audit remaining O(n_grid·n_basis) temporaries with the P0.6 dump hooks | S |
| P3.4 | Sparsity-aware S/T assembly over overlapping pairs only — feeds the tile substrate's block-sparse layout (`core/tile/layout.hpp`) and is the on-ramp for Phase 4b | M |

**Exit criteria:** 50- and 100-atom molecules run fully on GPU on 12 GB; peak VRAM measured O(N) (published plot); no fallback anywhere on the ladder; blocked ≡ dense-oracle ≤1e-8 on the small set; small-molecule regression ≤10%.

**Cumulative vs day-0:** 20–50 atoms ≥15–30× (removing the 15× fallback penalty × blocked-GEMM wins); small molecules held.

---

## 7. Phase 4 — Throughput and scale-out (~1–2 quarters; sub-streams parallelizable)

- **P4a Batched R0 mode [L]** (proposal differentiator #3, currently not started): bucket molecules by (n_basis, n_grid) class; batched cuSOLVER syev; per-class CUDA-graph replay (P2.9 makes each molecule one graph — batching = concurrent graph launches); shared per-element basis/φ caches. **Gate G1: ≥10⁴ single-points/hour/GPU on a 30-atom organic set** (TIDES_5yr_proposal.md §10 Y1Q4). Risk: padding waste on heterogeneous sets → size bucketing, measure occupancy.
- **P4b Tile substrate into the SCF loop [L]** (after P3.4): H, S, P as CSR-of-tiles end-to-end per `tides-docs/architecture-2026-07-11.md` Phase 3 — today tiles are used only for the H@X product and the Ozaki energy trace ("partially decorative", audit). Exit: full SCF on tiles ≡ dense ≤1e-8; ≥90% grouped-GEMM parity held.
- **P4c True MPI [L; strictly last]:** `NaoDriver` never calls `MpiWorld` — wire RCB partitioning over P3.1's grid blocks (the natural decomposition unit), halo exchange for ρ/vmat, distributed Poisson. Exit: ≥70% weak scaling 2→8 GPUs at ~500 atoms. Delivers nothing below 100 atoms; do not let it jump the queue.
- **P4d Linear scaling at 10³–10⁴ atoms [L]:** extend SP2/submatrix from CPU + single-block GPU to multi-block on the tile substrate; O(N) verified beyond the current 50–400-atom range. **Gate G2: 10⁴ atoms on one GPU, ≤0.5 meV/atom vs R1 control** (proposal Y2Q2; budgeted fallback: OMM/FOE, +2 quarters).

## 8. Phase 5 — Assembling the honest 1000× (ongoing after P4)

Couple the verified XL-BOMD engine (0.001 solves/step, 17.19 µHa/atom/ps — the one differentiator already working) to the blocked/tile engine; batched-MD screening pipelines; 10⁵-atom single-GPU demo (~50–70 GB per proposal §2.2). Public claims follow the proposal's benchmarking protocol (§9: fixed accuracy, kWh, 3 repeats, competitor best-settings) — the 1000× row is reported as the factor product of §1.2, never as a single-SCF number.

---

## 9. Continuous verification protocol (all phases, starting now)

- **Reference ladder:** CH4, H2O, NH3, C2H4, C6H6, glycine, (H2O)₁₆, ~50-atom peptide, then P4d's a-Si supercells. Geometries + PySCF/gpu4pyscf reference energies frozen under `tides/verification/references/`; tolerances in `verification/tolerances.yaml`.
- **Four-route matrix** re-run at every phase close and weekly; any route regression blocks merge.
- **Per-commit perf tracking:** wall, per-iteration, iteration count, peak VRAM, setup time; CI fails on >10% regression on any rung. Day-0 numbers: `profiling_results/baseline_2026-07-18/`.
- **Monthly gpu4pyscf head-to-head** on identical hardware, archived under `benchmarks/report/`.
- **Phase-0 tests are permanent:** radial closed-form suite, single-atom oracle, per-term dump comparisons run on every commit, forever.

## 10. Dependency graph, risks, quick wins

```
P0 ──► P1 ──────────────┐
 │                      ▼
 └──► P2 (P2.1 first) ──► P3 ──► P4c, P4d ──► P5
          │                        ▲
          └──► P4a          P4b ───┘ (needs P3.4)
```

**Risk register:**
1. **Compensating errors in P0** — the on-site analytic pieces were tuned alongside broken cross-atom terms; fixing one piece may "worsen" totals. Mitigation: land P0.1–P0.4 as one unit, judged only by the P0.5/P0.6 oracles.
2. **P3.1 is the largest structural change** — mitigation: dense path retained as env-flagged oracle, A/B in CI.
3. **CUDA graphs × cuSOLVER capture limits** — mitigation: graph all but the eigensolve.
4. **G2 submatrix accuracy at full-DFT H** (proposal risk #1) — budgeted fallback OMM/FOE.
5. **MPI scope creep** — P4c is sequenced strictly after single-GPU maturity.

**Quick-win shortlist (first 2 weeks, alongside P0):** GpuArena size fix (~20 lines) · per-interval dr transplant · SCF workspace preallocation · pinned staging · st_gpu 3-block extraction · wire xc_stream · RMS(ΔP)/commutator convergence criteria.

**Cumulative position vs day-0 baseline:**

| After | CH4/H2O wall | 20–50 atoms | 1000× narrative |
|---|---|---|---|
| P0 | ~1× (but correct) | ~1× (correct) | credibility restored |
| P1 | ~2× | ~2× | — |
| P2 | ~5–10× → ≤2× gpu4pyscf | ~2–3× | per-molecule CUDA graph = batching substrate |
| P3 | held | ~15–30× | O(N) memory = large-N substrate |
| P4 | held | ~20–50× | G1 (≥10⁴ mol/h) + G2 (10⁴ atoms O(N)) |
| P5 | held | held | 10²–10³ (scaling) × ~10 (batch) × 10–50 (XL-BOMD) — a product, never a single SCF |
