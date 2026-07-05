# WP8 — Parallel, HPC, portability, packaging (owner S8)
Purpose: from single workstation GPU to Phase-C clusters without touching physics code.
Architecture: 30-architecture/34. Phases A(pkg/CI) -> B(single-node) -> C(multi-node).

### T8.1 — Single-node 2-GPU data model (NCCL)
- Observables: 2xRTX efficiency >=85% on a 4096-H2O proxy step. Effort 6 pw. Depends T1.2, T3.2.

### T8.2 — METIS tile partitioner
- Observables: imbalance <=10% on gauntlet graphs. Effort 3 pw. Depends T1.1.

### T8.3 — Halo exchange + overlap
- Observables: exposed communication <=15% of step time on the 2-GPU proxy. Effort 4 pw. Depends T8.1, T8.2.

### T8.4 — HDF5 stage-dump/restart (bisect-the-physics enabler)
- Observables: bitwise round-trip; schema versioned; injection demo swaps our Poisson stage for a
  reference dump and reproduces energies. Effort 4 pw. Depends T1.1. Unblocks the whole debug ladder.

### T8.5 — Packaging + CI runners
- Observables: CMake presets + Spack recipe + container; nightly CI on RTX, weekly on A40,
  opportunistic H100; one-command developer setup <=30 min. Effort 5 pw. Depends –.

### T8.6 — MPI + NVSHMEM multi-node (Phase C)  [gates GC1/GC2 with T5.9]
- Observables: weak >=80% to 8 GPUs at 10^5 a-Si; restart-safe 24 h run. Effort 6 pw. Depends T5.7, T8.3.

### T8.7 — HIP quarterly gate
- Observables: full verification suite green on ROCm each quarter after T1.8. Effort ~3 pw/yr. Depends T1.8.
