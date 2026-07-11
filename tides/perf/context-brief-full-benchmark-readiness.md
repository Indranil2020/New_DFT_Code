# RALPH Phase R — Context Brief: Full Proposal Benchmark Readiness

## Task
Make the TIDES codebase ready for the full benchmarking campaign described in the 5-year proposal and `60-benchmarks/60-protocol.md` / `61-piecewise-matrix.md`.

## Scope Reality
The full proposal benchmark matrix (rows 1–12, 10⁴–10⁶ atoms, GPU, CPU, NAO, pseudopotentials, hybrids, MD, scaling) is the Year 1–Year 5 end state. The current codebase is at **Phase A bootstrap** (GTO/STO-3G `MoleculeDriver`, NAO/pseudopotential code exists but is not assembled into a production pipeline, GPU kernels exist but are not resident in the SCF loop).

## Current State (evidence from 2026-07-10)
- **C++ tests**: 70/74 pass; 4 expected open defects (audit P0.2).
- **Python tests**: 30/30 pass.
- **P0–P3 audit remediation**: complete.
- **NAO**: `nao_driver.hpp` exists but supports only s–s overlap/kinetic; higher angular momentum is zero; no Python binding; not benchmarked.
- **Pseudopotentials**: `Pseudopotential` struct + UPF2 reader + validators pass tests; not used in SCF.
- **GPU**: CUDA kernels tested standalone; no GPU dispatch in `MoleculeDriver` or `NaoDriver` SCF loops.
- **CPU**: `MoleculeDriver` works end-to-end on CPU but is a GTO/STO-3G bootstrap with grid-based V_H/V_xc and large errors vs PySCF (H2O 13 Ha off).

## Acceptance Criteria for "Full Benchmark Readiness" (all must be true)
1. **NAO + pseudopotential CPU product pipeline** is the default SCF path; `MoleculeDriver` becomes a bootstrap-only test.
2. **Two-center / three-center integrals** support all angular momenta in the DZP/TZP basis and pass matched-basis tests vs PySCF/ABACUS to ≤1e-6 Ha.
3. **GPU residency**: `rho_build`, `vmat_build`, `Poisson`, `XC` and `N` build are on the device and are exercised from the SCF loop.
4. **Tile substrate** is integrated: `TileMat` / grouped GEMM / f64e reductions are used for the grid maps and the Hamiltonian build.
5. **Solver broker** dispatches R0–R3 based on system size/gap and passes regime-transition tests.
6. **Analytic forces** (HF + Pulay + grid + PP) pass 5-point FD to ≤1e-6 Ha/Bohr on gauntlet-10.
7. **XL-BOMD** is coupled to the real Hamiltonian and NVE drift ≤ 30 µHa/at/ps for 64-H2O 100 ps.
8. **Benchmark protocol compliance**: fixed-accuracy, same PseudoDojo ONCV files, kWh metering, 3 repeats, cold+warm, reproducible archive.
9. **Competitor farm** (ABACUS, GPU4PySCF, PySCF, SPARC, GPAW, DFTB+ containers) runs the 12-row piecewise matrix with TIDES.
10. **GA1 (M6) evidence**: molecular SCF + forces pass FD ladder.
11. **GA2 (M12) evidence**: alpha release with R0 batching ≥ 5 × 10³ single-points/hour on RTX.

## First Concrete Sub-Task (Phase A critical path)
Assemble the **NAO + ONCV pseudopotential CPU SCF pipeline** (`NaoDriver`) so it can replace the GTO bootstrap as the honest production path. This is the prerequisite for every downstream benchmark (rows 1–12).

## Open Questions / Need User Decision
- Confirm first sub-task: **NAO+PP CPU SCF pipeline** (not forces/GPU/scale-out first).
- Confirm tolerance target for matched-basis comparison: ≤1e-6 Ha/atom or looser for the first pass?
- Confirm scope of first gate: GA1 evidence (forces) or just a working NAO+PP SCF before adding forces?
