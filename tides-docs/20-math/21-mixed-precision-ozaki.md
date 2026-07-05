# Mixed precision + Ozaki-scheme FP64 emulation (f64e)
## Policy (see 30-architecture/33 for the op-by-op table)
Store BF16/FP16 + per-tile scale; accumulate FP32; CRITICAL reductions (total energies, traces,
Fermi search, mixer dot products, level shifts) in f64e = FP64-equivalent GEMM/dot built by
error-free slicing of operands onto tensor cores (Ozaki scheme; FP16-slice variant first, FP8 variant
maintained because INT8 is being phased out on new hardware).
## Why existential here
RTX/A40 native FP64 ~ 1/64 of FP32; f64e on tensor cores is faster than native FP64 on these cards
by construction, and on datacenter parts modern extensions even beat native DGEMM (~2.3x on GB200-class).
## Error model (must be written before kernels)
eps_store -> ||dH|| -> ||dP|| <= c * ||dH|| / gap -> |dE|/atom bound; nightly A/B (mixed vs FP64)
enforces <=0.5 meV/atom on the gauntlet; gap monitor escalates precision automatically.
## Read first
Ozaki et al. (2012); Ootomo–Ozaki–Yokota IJHPCA 38 (2024); Uchino et al. IJHPCA 39 (2025);
arXiv:2511.13778 (guaranteed-accuracy extensions); arXiv:2508.00441 (FP8 variant).
