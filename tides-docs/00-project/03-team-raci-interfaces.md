# Team of 10 — ownership, interfaces, cadence
| Owner | Engine (40-engines/) | Secondary duty |
|---|---|---|
| S1 | WP1 tile substrate & precision | chief performance engineer; architecture council |
| S2 | WP2 basis & integrals | pseudo/PAW liaison to S7 |
| S3 | WP3 grids/Poisson/XC | QTT research 20% (with S5) |
| S4 | WP4 mid-range solvers (R0/R1) | broker calibration |
| S5 | WP5 linear scaling (R2/R3) | QTT research 20% |
| S6 | WP6 SCF/XL-BOMD/forces/MD | physics validation with S9 |
| S7 | WP7 hybrids/dispersion/PAW | surface-chemistry suite |
| S8 | WP8 parallel/HPC/portability | packaging; HIP quarterly gate |
| S9 | WP9 verification & benchmarks (red team) | release veto |
| S10 | WP10 API/docs/community | user support rotation |

Interfaces are CONTRACT TESTS (30-architecture/31): TileMat, GridArray, HDF5 stage dumps are frozen,
versioned schemas with their own suites; teams integrate against contracts, never against branches.
Cadence: 2-week sprints keyed by task IDs; biweekly architecture council (S1,S4,S5,S9); quarterly
competitor-landscape review by S9; >=1 cross-WP week/person/year. RACI: task owner is R+A; S9 is
always C on observables; S1 is C on any GPU kernel.
