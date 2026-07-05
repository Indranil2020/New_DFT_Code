# WP5 — Linear-scaling solvers R2/R3 (owner S5)
Purpose: SP2-submatrix purification (gapped) and FOE/SQ (metallic, Mermin). Math: 20-math/22, 23.
Hardware note: ALL validation tasks below fit RTX/A40 (10^4 atoms ~ 5–10 GB mixed precision);
only T5.9 (Phase C) needs the cluster. Model: ~2.8e9 FLOP/atom per purification (m=200, n_b=15).

### T5.1 — SP2 CPU FP64 reference (sparse, small)
- Observables: ||P^2-P||_F <=1e-10; tr(PS)=N_e <=1e-10 on 100–500-atom proxies. Effort 3 pw. Depends –.

### T5.2 — Submatrix construction (halo extraction + batching plan)
- Problem: per-atom principal submatrices from the sparsity graph; batch by size class.
- Observables: equals global SP2 on 500–2,000-atom proxies within declared block tolerance;
  runs on RTX 24 GB. Effort 5 pw. Depends T5.1, T1.1.

### T5.3 — GPU batched submatrix SP2, mixed precision  [gate GB1 evidence]
- Observables: 2,000-atom a-Si:H total energy within 0.5 meV/atom of R1 dense reference, on A40;
  time and VRAM recorded vs model. Effort 6 pw. Depends T5.2, T1.3, T1.4.

### T5.4 — Truncation policy + error compensation
- Observables: error-vs-radius curves published; compensation reduces energy error >=5x at fixed
  radius on the a-Si:H control. Effort 4 pw. Depends T5.3.

### T5.5 — FOE/Chebyshev density matrix (Mermin, metals)
- Observables: 500-atom Al at Te=3000 K free energy <=1 meV/atom vs dense; order-vs-beta curve
  matches 20-math/23 theory. Effort 6 pw. Depends T1.3, T5.1.

### T5.6 — Fermi-level search in f64e
- Observables: N_e error <=1e-10 across 10^2–10^4-atom cases; robust bracketing (no failure on the
  gauntlet). Effort 2 pw. Depends T1.4.

### T5.7 — Scale-out interface spec (with S8; document only)
- Observables: memory model table + message-pattern spec signed off by S8; no code. Effort 2 pw. Depends T5.3.

### T5.8 — 10^4-atom single-card run  [gate GB3 evidence]
- Observables: <=1 meV/atom vs size-extrapolated control; time and VRAM recorded on RTX and A40.
  Effort 5 pw. Depends T5.4.

### T5.9 — Distributed R2/R3 at 10^5–10^6 (Phase C, with WP8)
- Observables: weak scaling >=80% to 8 GPUs at 10^5 a-Si (GC1); 10^6-atom demo (GC2).
  Effort 6 pw. Depends T5.7, T8.6.
