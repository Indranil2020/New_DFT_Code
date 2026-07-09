# MPI Benchmark — PySCF CPU MPI vs TIDES Parallel

**Date**: 2026-07-09 10:50:07
**PySCF**: 2.11.0 | **MPI**: True | **Ranks**: 4

## SCF Benchmark (cc-pVDZ, LDA, density fitting)

| System | Atoms | nao | SCF (ms) |
|---|---|---|---|
| H2O | 3 | 24 | 5136.2 |
| CH4 | 5 | 34 | 7285.9 |
| C6H6 | 12 | 114 | 18124.2 |
| C10H8 | 18 | 180 | 536728.3 |
| H2O_8mer | 24 | 192 | 33903.8 |
| H2O_16mer | 48 | 384 | 104920.1 |
| H2O | 3 | 24 | 4607.7 |

## Basis Scan (H2O, LDA)

| Basis | nao | SCF (ms) |
|---|---|---|
| cc-pVDZ | 24 | 5136.2 |
| cc-pVDZ | 24 | 4607.7 |
| def2-TZVPP | 59 | 5211.7 |

## Gradient Benchmark (cc-pVDZ, LDA)

| System | Atoms | Grad (ms) | max|g| |
|---|---|---|---|
| H2O | 3 | 2353.1 | 0.040151 |
| CH4 | 5 | 3416.5 | 0.023283 |
| C6H6 | 12 | 18951.0 | 0.024715 |
| C10H8 | 18 | 69723.0 | 1.570760 |
| H2O_8mer | 24 | 36418.7 | 0.037859 |
| H2O_16mer | 48 | 191468.8 | 0.037858 |

## TIDES E7 Parallel Profile

- **Status**: pass
- **Wall**: 12.77s
