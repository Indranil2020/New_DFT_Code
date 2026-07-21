# Phase 3 design: grid-blocked sparse-φ storage (the O(N) substrate)

**Status:** design, 2026-07-20 (Opus). Grounds roadmap §6 in the actual code.
**Goal:** replace dense φ/∇φ tables (O(n_basis·n_grid) ≈ O(N²)) with block-sparse
storage (O(N)), so a 12 GB card runs 50–100 atom molecules on the GPU and the
>12-atom silent-CPU-fallback (BUG-6) is deleted. This is the structural lever
that (a) fixes the "GPU falls apart on bigger systems" symptom and (b) is the
substrate on which large-N linear scaling (Phase 4d) and the honest 1000×
narrative are built.

## 1. What exists today (the waste, cited)

- `orbitals` : `std::vector<std::vector<double>>`, size `n_basis`, each
  `np_total` doubles — dense `n_basis × n_grid`
  (`nao_driver.hpp:1128–1144`). Evaluated by a full triple loop over **every**
  grid point per basis function, though `EvalNaoBasisFn` returns 0 beyond
  `r_cut`.
- `grad_orbitals_3d` : `3 × n_basis × n_grid` (`nao_driver.hpp:1158–1184`).
- Both uploaded whole to `d_phi` / `d_grad_phi` (`nao_driver.hpp:1218–1241`).
- The consumers are all the same shape — a contraction over grid points of the
  φ table against a per-point weight or against P:
  - **S/T**: `S = dv·Φᵀ Φ`, `T = ½dv·Σ_c ∂Φ_cᵀ ∂Φ_c` (`nao_driver.hpp:1236–1253`;
    GPU `BuildStFromGridGpu`).
  - **ρ build**: `ρ(g) = Σ_ij P_ij φ_i(g)φ_j(g)` via `temp = PΦ`, `ρ = Σ_i Φ_i⊙temp_i`
    (`vmat_build.hpp BuildRhoGemm/BuildRhoWithGrad`; GPU `rho_build.cu`).
  - **vmat**: `H_ij = Σ_g v(g) φ_i(g)φ_j(g) = Φ diag(v) Φᵀ`
    (`vmat_build.hpp BuildHmatGemm/BuildGgaHmatGemm`; GPU `pp_build.cu`,
    `vmat_build.cu`).
- **BUG-6** (`nao_driver.hpp:1200–1216`): a pre-flight `cudaMemGetInfo`
  estimate of `phi + grad_phi + 4n² + np` bytes; if it exceeds free VRAM the
  whole run reverts to CPU. The estimate is dominated by the dense
  `n_basis·np` and `3·n_basis·np` terms.

**Memory model.** Dense: `4·n_basis·n_grid·8 B`. For fixed grid density,
`n_grid ∝ box_volume ∝ N` and `n_basis ∝ N`, so dense ≈ **O(N²)**. Every NAO is
zero outside a sphere of radius `r_cut` (~5–7 bohr) → volume ~const,
independent of N. Blocked storage keeps only (block, active-function) pairs, so
`Σ_blocks n_active·n_block_pts ≈ n_grid · (avg functions overlapping a point) =
O(N)` (avg overlap is set by local atom density × r_cut³, not by total N).

## 2. Data structure

Partition the uniform grid (`UniformGrid3D`, flat index
`g = ix + n0·(iy + n1·iz)`) into cubic blocks of `BS³` points (start `BS = 8`;
tune 8/16). For block `b`:

```
struct GridBlock {
  int64_t g_offset;                 // flat index of the block's (0,0,0) corner
  int32_t bx, by, bz;               // block coords; extent BS (clipped at edges)
  std::vector<int32_t> active;      // basis-function indices with r_cut sphere
                                    //   intersecting this block's AABB
  // Compact, column(point)-major so a GEMM sees active fns as rows:
  //   phi_blk[a*n_pts + p]  for a in [0,active.size()), p in [0,n_pts)
  std::vector<double> phi_blk;      // active.size() × n_pts
  std::vector<double> grad_blk;     // 3 × active.size() × n_pts   (GGA only)
};
```

**Active-list build:** a function `i` centered at `R_atom(i)` with cutoff
`r_cut(i)` is active in block `b` iff `dist(R_atom(i), block_AABB) < r_cut(i)`.
Cheap: loop atoms × blocks with an AABB–sphere test; near-linear because each
atom touches O(1) blocks. This replaces the dense eval loop entirely — φ is only
evaluated at (block, active) pairs.

Keep a global inverse map `active_of_point` implicitly via the block structure;
no dense `n_grid×n_basis` table is ever materialized.

## 3. The three operations, blocked

All become **per-block skinny GEMM + scatter into the small global n×n matrix**
— exactly the grouped-GEMM shape the tile substrate (Phase 4b) wants.

1. **S/T** (once): per block, `S_loc = dv · phi_blk · phi_blkᵀ`
   (active×active), scatter-add `S[active[i], active[j]] += S_loc[i,j]`.
   T likewise from `grad_blk`. Only overlapping pairs are ever touched.
2. **ρ** (per iter): per block, `temp = P[active,active] · phi_blk`
   (active×n_pts), `ρ_blk[p] = Σ_a phi_blk[a,p]·temp[a,p]`, write ρ at the
   block's points. GGA: also `∇ρ_blk` from `grad_blk`.
3. **vmat** (per iter): per block, weight `phi_blk` by `v` at the block points,
   `H_loc = phi_wblk · phi_blkᵀ`, scatter-add into H. GGA: the 4-plane variant.

Scatter target (S, H, P sub-blocks) is the small `n×n` matrix — cheap. On GPU
the scatter is `atomicAdd` into H (n² is small) or a deterministic per-block
reduction for `TIDES_DETERMINISTIC`.

## 4. Increment plan (each lands with a dense-path A/B oracle)

Keep the dense path compiled behind `TIDES_DENSE_PHI=1` (mirror the existing
`TIDES_USE_GRID_ST` / `TIDES_GRID_SCREEN` env-gate pattern). CI A/B compares
blocked vs dense on the small set at ≤1e-10.

- **Inc 1 — blocking module + memory report [me].** New
  `tides/core/grid/grid_blocking.hpp`: `BuildGridBlocks(atoms, basis_map, grid,
  need_grad) → std::vector<GridBlock>` with compact φ/∇φ eval into blocks (CPU,
  OpenMP over blocks). Add `ReconstructDensePhi(blocks) → dense` for the oracle
  and a `[grid-block] mem dense=… blocked=… ratio=…` log line. **Exit:**
  reconstructed φ ≡ dense eval ≤1e-14 on CH4/H2O/C6H6; measured blocked φ memory
  ≤ dense, ratio grows with N (publish CH4→C10H22 curve).
- **Inc 2 — blocked S/T (CPU) [agent].** Per-block GEMM + scatter behind
  `TIDES_BLOCKED_ST`. **Exit:** S/T ≡ dense two-center path ≤1e-10 on the small
  set; no perf regression.
- **Inc 3 — blocked ρ + vmat (CPU) [agent].** **Exit:** SCF total energy ≡
  dense ≤1e-8 on CH4/H2O/NH3/C2H4 (PP).
- **Inc 4 — GPU port [me + agent].** Upload blocked φ once; per-block skinny
  GEMMs (grouped via `core/tile/gemm_grouped.cu`) + scatter kernels; ρ/vmat on
  device. **Exit:** blocked-GPU ≡ blocked-CPU ≤1e-8; peak VRAM O(N); 50- and
  100-atom molecules run fully on GPU on 12 GB.
- **Inc 5 — kill the fallback [me].** Delete the BUG-6 silent full-CPU revert
  (`nao_driver.hpp:1200–1216`); policy = (i) chunked block streaming
  (double-buffered H2D on a copy stream) when the resident set exceeds VRAM,
  (ii) loud warning + partial spill, (iii) hard error with an actionable
  message. Full-CPU only via explicit `TIDES_FORCE_CPU=1`.

## 5. Risks

- **Scatter races (GPU):** H/S are small; `atomicAdd` is fine for production,
  with a deterministic per-block path under `TIDES_DETERMINISTIC`.
- **Edge blocks / partial cutoff:** the AABB–sphere test is conservative (may
  mark a function active in a block where it is ~0); harmless (adds a few zero
  columns), never drops a nonzero. Verified by the Inc-1 reconstruction oracle.
- **Index width:** block offsets and point counts are `int64`; per-block local
  point index stays `int32` (BS³ ≤ 4096).
- **Regression on tiny molecules:** blocking overhead may slightly slow CH4
  (few blocks, low sparsity). Keep dense path as the small-N fast option; the
  broker picks blocked above a measured crossover (~15–20 atoms), dense below.

## 6. Why this is the 1000× lever (honest accounting)

Phase 3 does **not** speed up a single small-molecule SCF (roadmap §1.2 ceiling
is ≤2× there, owned by Phase 2). It delivers:
1. **Removal of the 15× CPU-fallback penalty** on 20–100 atom systems (the
   user's "GPU falls apart on bigger systems").
2. **O(N) memory** → the system-size regime where gpu4pyscf is O(N³–N⁴) and
   eventually cannot run at all; TIDES starts *winning* past ~30–50 atoms and
   the margin compounds with N. This is the "large-N scaling" factor in the
   roadmap's 1000× product table.
3. The **block-sparse layout** is the exact input the SP2/submatrix
   linear-scaling solver (Phase 4d) and the tile substrate (Phase 4b) consume —
   Phase 3 is their on-ramp.
