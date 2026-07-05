# WP1 — Tile substrate & precision (owner S1)
Purpose: the one layer everything runs on. Consumes: nothing. Provides: TileMat + precision
descriptors + f64e reductions to every other WP. Math: 20-math/20, 21. Model: grouped-GEMM bound;
target >=90% of cuBLASLt grouped throughput on realistic tile mixes. Phase A start; T1.8 Phase B.

### T1.1 — TileMat core (CPU FP64 reference + layout)
- Problem: define and implement the block-sparse CSR-of-tiles container with per-tile scales,
  symmetry flags, and serialization; FP64 CPU reference for every later GPU op.
- Start: create core/tile/layout.hpp + tests/; read 20-math/20 and 30-architecture/31.
- Requirements: tile edges {16,32,64} zero-padded; dense<->tile conversion; iterators; HDF5 dump.
- Observables: (1) round-trip dense<->tile exact in FP64 on 100 random patterns; (2) symmetry
  preserved when flagged; (3) serialization bitwise round-trip. Effort 4 pw. Depends –. Unblocks all.

### T1.2 — Grouped GEMM GPU path
- Problem: batched heterogeneous tile GEMM (BF16/FP16/TF32 store, FP32 accumulate) via CUTLASS/cuBLASLt.
- Start: core/tile/gemm_grouped.cu; CUTLASS grouped-GEMM examples; input T1.1.
- Requirements: stream-ordered; graphs-capturable; per-tile scale applied in epilogue.
- Observables: (1) matches CPU FP64 within rung-1 budget; (2) >=90% of raw cuBLASLt grouped GEMM on a
  tile-size mix sampled from real H patterns, measured on RTX (record A40/H100). Effort 6 pw.
  Depends T1.1. Unblocks T1.3–T1.6, T2.5, T3.2, T4.*, T5.*.

### T1.3 — Filtered tile SpGEMM
- Problem: C = A*B on block-sparse operands with on-the-fly tile-norm filtering and a logged error ledger.
- Start: core/tile/spgemm_filtered.cu; read 20-math/20 invariants.
- Requirements: eps_filter parameter; ledger returned; deterministic-mode support.
- Observables: (1) vs CPU FP64 SpGEMM <=1e-12 (FP32 path) / <=1e-6 rel (mixed); (2) ledger bound never
  exceeded on adversarial cases; (3) >=60% of dense grouped-GEMM FLOP rate at fill 0.3 (RTX).
  Effort 6 pw. Depends T1.2. Unblocks T5.*, T7.4.

### T1.4 — Ozaki f64e GEMM and reductions
- Problem: FP64-equivalent GEMM/dot/trace by error-free operand slicing on tensor cores (FP16 slices
  first; FP8 variant maintained). This is existential on RTX/A40 (see 00-project/01).
- Start: core/tile/ozaki.cu, reduce_f64e.cu; read 20-math/21 and the four cited papers.
- Requirements: slice count auto-chosen from operand dynamic range; deterministic option.
- Observables: (1) elementwise error <= 4x FP64-GEMM ulp bound on dynamic range 1e+/-9;
  (2) >=5x faster than native FP64 GEMM on RTX at n=4096; (3) trace_f64e matches FP64 to <=1e-13 rel.
  Effort 6 pw. Depends T1.2. Unblocks T5.6, T6.2, all energy paths.

### T1.5 — Deterministic mode
- Observables: bitwise-identical results across 100 repeated runs of the full gauntlet-10;
  documented overhead <=2x. Effort 2 pw. Depends T1.2.

### T1.6 — CUDA-graph capture of solver sweeps
- Observables: launch count reduced >=10x on an R0 SCF sweep; wall overhead <=5%. Effort 3 pw. Depends T1.2.

### T1.7 — Precision descriptors + error ledger API
- Problem: every op logs its precision and error bound; feeds the nightly mixed-vs-FP64 A/B (T9.3).
- Observables: ledger emitted for a full SCF; WP9 harness consumes it without modification.
  Effort 3 pw. Depends T1.4. Unblocks T9.3.

### T1.8 — HIP build of the substrate (Phase B end)
- Observables: full tile test suite green on ROCm target; quarterly gate thereafter. Effort 4 pw. Depends T1.2–T1.4.
