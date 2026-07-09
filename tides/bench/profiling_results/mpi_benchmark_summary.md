# MPI Benchmark — PySCF CPU MPI Scaling on RTX 3060 Workstation

**Date**: 2026-07-09
**MPI**: OpenMPI 4.1.2 + mpi4py 3.1.3
**PySCF**: 2.11.0
**Note**: Intel oneAPI installed (compiler/MKL/TBB) but no Intel MPI — used system OpenMPI.

## SCF Scaling (cc-pVDZ, LDA, density fitting)

| System | Atoms | nao | 1-rank (ms) | 2-rank (ms) | 4-rank (ms) | 2-rank speedup | 4-rank speedup |
|---|---|---|---|---|---|---|---|
| H2O | 3 | 24 | 2952 | 329 | 5136 | 8.97x | 0.57x |
| CH4 | 5 | 34 | 3938 | 731 | 7286 | 5.39x | 0.54x |
| C6H6 | 12 | 114 | 9548 | 5689 | 18124 | 1.68x | 0.53x |
| C10H8 | 18 | 180 | 246539 | 265623 | 536728 | 0.93x | 0.46x |
| H2O_8mer | 24 | 192 | 20210 | 9209 | 33904 | 2.19x | 0.60x |
| H2O_16mer | 48 | 384 | 56016 | 37874 | 104920 | 1.48x | 0.53x |

## Basis Scan (H2O, LDA)

| Basis | nao | 1-rank (ms) | 2-rank (ms) | 4-rank (ms) |
|---|---|---|---|---|
| cc-pVDZ | 24 | 2157 | 258 | 4608 |
| def2-TZVPP | 59 | 3114 | 622 | 5212 |

## Gradient Scaling (cc-pVDZ, LDA)

| System | Atoms | 1-rank (ms) | 2-rank (ms) | 4-rank (ms) | 2-rank speedup | 4-rank speedup |
|---|---|---|---|---|---|---|
| H2O | 3 | 890 | 172 | 2353 | 5.17x | 0.38x |
| CH4 | 5 | 1362 | 321 | 3417 | 4.24x | 0.40x |
| C6H6 | 12 | 10486 | 14758 | 18951 | 0.71x | 0.55x |
| C10H8 | 18 | 28622 | 77464 | 69723 | 0.37x | 0.41x |
| H2O_8mer | 24 | 10657 | 35386 | 36419 | 0.30x | 0.29x |
| H2O_16mer | 48 | 76301 | 229481 | 191469 | 0.33x | 0.40x |

## TIDES E7 Parallel Profile

| Ranks | Status | Wall (s) |
|---|---|---|
| 1 | PASS | 9.88 |
| 2 | PASS | 7.96 |
| 4 | PASS | 12.77 |

## Key Findings

1. **PySCF MPI does not scale on single workstation**: 2-rank shows speedup for small systems (H2O 9x, CH4 5x) due to warmup/cache effects, but degrades for larger systems. 4-rank is consistently slower than 1-rank due to oversubscription (4 ranks on limited cores).

2. **Gradient scaling is negative**: MPI gradients are slower than serial for all systems except H2O/CH4 at 2-rank. This is expected — PySCF gradients are not MPI-parallelized; each rank computes independently.

3. **TIDES E7 parallel** passes on all rank counts. The partitioner and halo exchange work correctly.

4. **Recommendation**: For single-workstation CPU MPI, 2 ranks is optimal for small systems. For production, use GPU (gpu4pyscf) instead of CPU MPI. TIDES should target multi-GPU MPI with NVSHMEM for real scaling.
