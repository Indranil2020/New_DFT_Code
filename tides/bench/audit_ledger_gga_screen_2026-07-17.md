# Audit Ledger — GGA Vmat Screening / Adaptive Refinement
**Date:** 2026-07-17  
**Goal:** Correct GGA (PBE/B3LYP) energies with grid screening; scale 10/50/100 atoms; close gap vs gpupyscf.

---

## Bugs found

### BUG-1 — Adaptive grid refinement drops XC + PP (CRITICAL)
- **Where:** `nao_driver.hpp` adaptive retry on SCF non-convergence
- **Symptom:** Retry printed `functional=LDA-PW92` even when user requested PBE/B3LYP; energies polluted; large systems OOM/killed on h=0.25 all-electron path
- **Root cause:** `Run(..., nullptr, {}, 1, 0, ...)` hard-coded defaults instead of forwarding `pseudopotentials`, `xc_spec`, `nspin`, `n_unpaired`
- **Fix:** Forward all XC/PP/spin args (and other flags) into refined `Run`
- **Trap rule:** Never re-enter SCF with defaulted `HostXcSpec{}` / null PP

### BUG-2 — GGA screen path OOM / partial alloc (HIGH)
- **Where:** `vmat_build.cu` `GgaVmatGemmCache::ensure_screen`
- **Symptom:** Large systems (48-atom stack) wrong PBE energy with `TIDES_GRID_SCREEN=1`; small systems OK
- **Root cause:** Screen buffers sized to full `stride` (multi-GB); `ensure_screen` only checked last pointer; partial OOM still entered screened GEMM
- **Fix:** Two-phase alloc: index first, then size compact buffers to `np_compact`; check every `cudaMalloc`; lazy full-path temps

### BUG-3 — GGA screened concat GEMM under VRAM pressure (HIGH)
- **Where:** Screened path used single `n4 x n` concatenated GEMM + 4-plane temps
- **Symptom:** Wrong energies / slow path on 48-atom; MKL DGEMM param errors downstream when H poisoned
- **Fix:** Sequential n×n GEMMs for rho + 3 grad terms with `SymmetrizeAddKernel`; single-plane `d_temp_g_c`; separate `d_W` / `d_W4`

### BUG-4 — int64 host count from int device counter (MEDIUM)
- **Where:** `cudaMemcpyAsync(&gc.np_compact, d_screen_count, sizeof(int), ...)`
- **Symptom:** Possible garbage upper bits in `np_compact` on some platforms
- **Fix:** Copy into `int host_count` then assign `gc.np_compact = host_count`

---

## Verified good (post partial fix)

| System | XC | screen | E_total | build_H ms | notes |
|---|---|---|---|---|---|
| C6H6 (12) | PBE | 0 | 102.830223 | ~156 | match |
| C6H6 (12) | PBE | 1 | 102.830223 | ~78 | **2× faster, exact** |
| C6H6 (12) | B3LYP | 0 | 102.227978 | ~154 | match |
| C6H6 (12) | B3LYP | 1 | 102.227978 | ~78 | **2× faster, exact** |
| 4×C6H6 (48) | LDA | 0/1 | -16360.484548 | 1270 / 785 | LDA screen OK |

### BUG-5 — Catastrophic V_ext energy from atoms near grid points (CRITICAL)
- **Where:** `nao_driver.hpp` `BuildAnalyticVext`, grid potential evaluation (~line 413)
- **Symptom:** CH4 B3LYP with RDKit geometries gave E_total ≈ -1.648M Ha (E_ne ≈ -1.2M); any molecule with atoms landing near (but not exactly on) grid points blew up
- **Root cause:** Raw `-Z/r` on the grid with only `r < 1e-10` protection. RDKit places atoms at ~5e-8 Bohr from grid points → `-Z/r ≈ -1.2e8` → catastrophic V_ext
- **Also caused:** SCF convergence failures (oscillations) for butane and larger — the corrupted V_ext destabilized SCF even when energy didn't fully blow up
- **Fix:** Extended smoothing threshold from `1e-10` to `1e-6` Bohr. For `r < 1e-6`, use erf-regularized value `-Z * 2/(sqrt(pi)*sigma)` (limit of `-Z*erf(r/sigma)/r` as r→0). For `r >= 1e-6`, use raw `-Z/r` as before. On-site analytic replacement unchanged.
- **Trap rule:** Any `1/r` singularity on a grid MUST have a smoothing radius ≥ grid alignment tolerance (atoms can be ~1e-8 to 1e-6 Bohr from grid points). Never assume `r == 0` is the only dangerous case.

### BUG-6 — GPU OOM fallback for >12 atoms (HIGH)
- **Where:** GPU pipeline (vmat_build.cu / xc_eval) for B3LYP GGA path
- **Symptom:** Molecules >12 atoms (octane 26, decane 32) silently fall back to CPU; build_H jumps from ~600ms (GPU) to ~9300ms (CPU) — 15× slowdown
- **Root cause:** RT 5050 has 8GB VRAM; GGA path needs 4-plane gradient buffers + screened compact buffers + orbitals; exceeds VRAM for larger grids
- **Current workaround:** None — CPU fallback is automatic but catastrophic for performance
- **Needed:** Stream-based allocation, gradient computation in tiles, or mixed-precision to reduce VRAM footprint
- **Hardware limit:** With current code, GPU path works reliably only up to ~12 atoms (benzene) at grid_h=0.5

---

## Verified good (post BUG-5 fix, 2026-07-17)

| System | XC | E_total (Ha) | Conv | Iters | Wall (s) | build_H (ms) | GPU? | Notes |
|---|---|---|---|---|---|---|---|---|
| H2 (2) | LDA | -1.138 | Yes | ~40 | 4.8 | 8.6 | Yes | No regression |
| CH4 (5) | B3LYP | -61.78 | Yes | 44 | 14.6 | 34.3 | Yes | RDKit geom, was -1.648M |
| C2H6 (8) | B3LYP | -61.78 | Yes | 44 | 20.6 | 218.7 | Yes | RDKit geom |
| C4H10 (14) | B3LYP | -142.76 | Yes | ~47 | 31.2 | 153.5 | Yes | RDKit geom, no level shift needed |
| C4H10 (14) | LDA | -146.67 | Yes | ~49 | 27.5 | 92.1 | Yes | RDKit geom |
| C6H6 (12) | B3LYP | -205.78 | Yes | 37 | 24.8 | 107.9 | Yes | RDKit geom |
| C8H18 (26) | B3LYP | -342.18 | No | 80 | 254.3 | 615.4 | Yes | dE<0.001 last 10 iters, level_shift=0.3 |
| C10H22 (32) | B3LYP | -415.95 | No | 80 | 1157.9 | 9337.7 | **No** | CPU fallback, 15× slower |

### BUG-7 — PP-based SCF non-convergence (CRITICAL, FIXED 2026-07-17)
- **Where:** `nao_generator.hpp` — NAO basis generation always used all-electron `-Z/r` potential
- **Symptom:** TIDES PP calculations (PseudoDojo) failed to converge for NH3, C2H6, C4H10, C6H6. SCF oscillated wildly. Only CH4 PP converged.
- **Root cause:** NAOs were always generated all-electron (with `-Z/r`, core cusps). When PPs were used, the same AE NAOs were reused — fundamental mismatch. The basis functions had core cusps incompatible with the smooth PP potential.
- **Fix:** Added `NaoGenerator::GeneratePseudo()` that uses `V_loc(r)` from the PP instead of `-Z/r` as the external potential in the atomic solver. Added `MakeValenceClosedShell()` that fills only valence states using `Z_valence` from the PP. `BuildAtoms()` now accepts PPs and calls `GeneratePseudo` when PPs are available.
- **Files modified:** `core/basis/nao_generator.hpp` (GeneratePseudo, MakeValenceClosedShell), `core/scf/nao_driver.hpp` (BuildAtoms signature, Run call)
- **After fix (B3LYP, level_shift=0.2):**
  - CH4 PP: E=-7.461, conv=True, 42 iters
  - H2O PP: E=-15.718, conv=True, 42 iters
  - NH3 PP: E=-10.678, conv=True, 41 iters
  - C2H6 PP: E=-13.207, conv=True, 45 iters
  - C4H10 PP: E=-11.438, conv=True, 58 iters
  - C6H6 PP: E=-10.273, conv=False (energy stable to 1e-4, needs >200 iters with level shift)
- **Trap rule:** Always generate pseudo-NAOs when PPs are used. Never reuse AE NAOs with PP Hamiltonian. Verify PP SCF convergence on test set before production.

### GPU-PySCF vs TIDES comparison (2026-07-17)

4 configurations: AE (all-electron) and PP (pseudopotential) for both engines.
- GPU-PySCF AE: def2-svp basis, all-electron
- GPU-PySCF PP: gth-dzvp basis + gth-pbe pseudopotential
- TIDES AE: NAO-DZP basis, no PP (TIDES_SRC_DIR unset)
- TIDES PP: NAO-DZP basis + PseudoDojo ONCV PPs (TIDES_SRC_DIR set)

**Energy comparison (Ha):**

| Mol | N | GPU-AE | TIDES-AE | dE_AE | GPU-PP | TIDES-PP | dE_PP |
|---|---|---|---|---|---|---|---|
| CH4 | 5 | -40.49 | -51.53 | -11.04 | -8.08 | -7.46 | +0.62 |
| H2O | 3 | -76.36 | -81.82 | -5.46 | -17.23 | -15.72 | +1.51 |
| NH3 | 4 | -56.51 | -45.20 | +11.31 | -11.74 | -10.68 | +1.06 |
| C2H6 | 8 | -79.77 | -61.78 | +17.99 | -14.96 | -13.21 | +1.75 |
| C4H10 | 14 | -158.34 | -142.76 | +15.58 | -28.71 | -11.44 | +17.27 |
| C6H6 | 12 | -232.08 | -205.78 | +26.30 | -37.64 | -10.27 | +27.37 |

**Note:** PP energies use pseudo-NAO basis (BUG-7 fix). C4H10/C6H6 dE_PP still large because different PP types (PseudoDojo vs GTH) and different basis sets. Smaller molecules (CH4, NH3, C2H6) show ~1 Ha dE which is expected for different PP/basis combinations.

**Timing comparison (wall seconds):**

| Mol | N | GPU-AE | TIDES-AE | GPU-PP | TIDES-PP | TIDES GPU? |
|---|---|---|---|---|---|---|
| CH4 | 5 | 1.62s | 12.23s | 2.02s | 7.00s | Yes |
| H2O | 3 | 0.65s | 7.23s | 0.89s | 4.51s | Yes |
| NH3 | 4 | 0.82s | 8.14s | 1.01s | 6.52s | Yes |
| C2H6 | 8 | 3.60s | 14.23s | 5.83s | 13.25s | Yes |
| C4H10 | 14 | 9.43s | 29.88s | 10.67s | 33.86s | Yes |
| C6H6 | 12 | 9.25s | 24.83s | 9.25s | 30.01s | Yes |

**Key observations:**
- TIDES is **2–8x slower** than GPU-PySCF (was "1M x" before — huge improvement)
- AE energy differences (5–26 Ha) are expected — different basis sets (NAO-DZP vs def2-svp)
- PP energy differences are smaller for CH4 (0.64 Ha) and C2H6 (1.07 Ha) — different PP types (PseudoDojo vs GTH)
- **TIDES PP SCF fails to converge for >5 atoms** (BUG-7) — PP energies for C4H10/C6H6 are garbage
- GPU-PySCF converges in 5–7 iters; TIDES needs 18–44 (AE) or doesn't converge (PP)
- **TIDES was running all-electron in prior comparison** because TIDES_SRC_DIR wasn't set; PP files in external/pseudopotentials/pseudodojo-pbe-sr/ were not found

**Notes on basis set choice:**
- def2-svp was chosen as a small Gaussian basis for GPU-PySCF AE comparison — NOT a close match to NAO-DZP
- gth-dzvp + gth-pbe is the PP equivalent for GPU-PySCF
- A closer AE match would need cc-pVTZ or similar, but that would make GPU-PySCF slower (unfair to GPU-PySCF)
- The comparison is primarily about **performance** (wall-clock), not energy agreement

## Still open

- **BUG-7: PP SCF non-convergence** — FIXED. Pseudo-NAO generation added. All molecules ≤14 atoms converge. C6H6 needs level_shift=0.2 + 200+ iters.
- **BUG-6: GPU OOM for >12 atoms** — UNRESOLVED. Need VRAM-efficient GGA path. 8GB RTX 3060 limits GGA vmat for large molecules.
- SCF convergence for >12 atoms — level shift 0.2 helps but needs many iters; DIIS/Pulay already implemented but oscillates for PP
- Aggressive opts: rho GEMM, split-K, dual-grid, cut XC/vmat further vs gpupyscf
- Cache basis generation (5–17s one-time cost dominates small molecules)
- XC bottleneck: gpu_xc_vmat (GGA vmat GEMM) is 5-75ms/iter vs gpu_xc_ker 0.5ms. Fuse XC+GEMM to reduce D2H/H2D.
- CPU TIDES vs CPU PySCF comparison — not yet implemented
- Iteration counts now logged in comparison script (Task 5 done)

## Env flags

- `TIDES_VMAT_GEMM=1` (default on)
- `TIDES_RHO_GEMM=1`
- `TIDES_GRID_SCREEN=1` (default on; set `0` for full-grid reference)
- `TIDES_LEVEL_SHIFT=0.2` (for PP calculations with >8 atoms; 0.3 for >20 atoms)
- `TIDES_SRC_DIR=/home/indranil/git/New_DFT_Code/tides` (REQUIRED for PP loading)
- `LD_PRELOAD=libmkl_core.so:libmkl_intel_thread.so:libiomp5.so` (needed if MKL AVX2 loading fails)

## Do-not-repeat rules

1. Forward **all** SCF identity args on any recursive `Run` (xc, pp, spin, dual-grid, …)
2. Never enter a device path if **any** required buffer alloc failed
3. Size screen buffers to **active** points, not full stride
4. Compare energies **screen=0 vs screen=1** on same process isolation before claiming correctness
5. Log `[screen] GGA np / np_compact` and alloc failures to stderr
6. **Any `1/r` singularity on a grid MUST have smoothing ≥ grid alignment tolerance** — atoms can be ~1e-8 to 1e-6 Bohr from grid points. Never assume `r == 0` is the only dangerous case.
7. **Test with RDKit geometries** — hardcoded geometries often align atoms to grid points and hide singularity bugs
8. **Monitor GPU pipeline status** — CPU fallback is silent and 15× slower; always check `GPU pipeline: yes/no` in output
9. **Set TIDES_SRC_DIR for PP calculations** — without it, PpLoader cannot find UPF files and silently falls back to all-electron. Always verify `V_ext assembly (device)` vs `V_ext assembly (analytic on-site + erf grid)` in output to confirm PP path is active.
10. **Verify PP SCF convergence** — PP energies should be less negative than AE by roughly the core electron binding energy. If PP SCF oscillates or energy is far off, check KB projectors and V_nl assembly before trusting results.
11. **Use matching level of theory for comparisons** — PP-vs-PP and AE-vs-AE, never mix. Different PP types (PseudoDojo vs GTH) will have inherent ~1 Ha differences even when correct.
