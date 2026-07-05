# Tile algebra — the universal substrate
## Definitions
TileMat: block-sparse matrix stored CSR-of-tiles; micro-tile edges in {16,32,64} (zero-padded);
per-tile scale factor (MX-style) carries dynamic range so payload can be BF16/FP16.
Complex tiles (k-points) are a storage variant, R0/R1 only.
## Operations (all reduce to grouped GEMM streams)
spgemm_filtered(A,B,eps_filter): tile-norm bound |C_ij| <= sum_k |A_ik||B_kj| drops products below
eps_filter with a LOGGED error bound; axpy, trace, norms via deterministic reductions.
## Invariants (contract tests)
Dense<->tile round trip exact in FP64 path; symmetry preserved when declared; filter error ledger
never exceeds declared eps; iteration order independence in deterministic mode.
