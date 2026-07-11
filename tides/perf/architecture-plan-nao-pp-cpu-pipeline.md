# RALPH Phase A — Architecture Plan: NAO + Pseudopotential CPU Product Pipeline

## Goal
Make `NaoDriver` the first honest, end-to-end CPU product pipeline for molecular DFT. It must be able to run H, small molecules, and light-element systems with ONCV pseudopotentials and DZP NAOs, producing converged energies and densities that can be compared against PySCF/ABACUS.

## Why this first
- Every downstream gate (GA1, GA2, GB1, GB2, GB3, piecewise rows) depends on a correct NAO+PP Hamiltonian.
- The current `MoleculeDriver` is a GTO/STO-3G bootstrap; the `NaoDriver` is only 25% complete (s–s only, no PP, no Python binding).
- We cannot honestly benchmark against ABACUS/SIESTA/SPARC until we use the same physics they use: NAO + ONCV pseudopotentials.

## Module boundaries (one concern per module)

| Module | File | Responsibility | Owner |
|---|---|---|---|
| NAO generation | `core/basis/nao_generator.hpp` | Generate DZP/TZP radial functions per element | WP2 (S2) |
| Two-center integrals | `core/basis/two_center_integrals.hpp` | Spline tables + Slater-Koster angular coupling for all (l_a, l_b) | WP2 (S2) |
| Three-center KB | `core/basis/three_center.cu` / `three_center_gpu.hpp` | KB nonlocal matrix assembly | WP2 (S2) |
| Pseudopotential data | `core/basis/pseudo/pseudopotential.hpp` | `Pseudopotential` struct, validators, KB channels | WP2 (S2) |
| UPF2 reader | `core/basis/pseudo/upf2_reader.hpp` | Parse UPF2 files into `Pseudopotential` | WP2 (S2) |
| Grid evaluation | `core/grid/vmat_build.hpp`, `core/grid/rho_build.hpp` | `BuildRho` / `BuildHmat` from orbitals | WP3 (S3) |
| Poisson | `core/grid/poisson.hpp` | `PoissonSolver::Solve` for V_H | WP3 (S3) |
| XC | `core/grid/xc.hpp` | `XCGridEvaluator::EvaluateLDA` / `PBE` | WP3 (S3) |
| SCF driver | `core/scf/scf_driver.hpp` | DIIS/Pulay/Broyden convergence | WP6 (S6) |
| NAO product SCF | `core/scf/nao_driver.hpp` | Assemble NAO + PP + grid + SCF | WP6 (S6) |
| Python binding | `api/python/tides/_native.cpp` | Expose `NaoDriver` to Python | WP10 (S10) |
| Benchmark script | `bench/pyscf_benchmark.py` | Matched-basis comparison vs PySCF | WP9 (S9) |

## Integration plan (sequential steps)

### Step A: Complete two-center S/T for all angular momenta
- Fix `NaoDriver::Run` to use `TwoCenterIntegralEval` with `RadialTable` built from actual NAO radial functions, not only s–s.
- Add `TwoCenterIntegralEval::BuildFromNAOs(fa, fb, l_a, l_b)` to tabulate h_L(R) for overlap and kinetic by direct 1D radial integration on the NAO grids.
- Replace the `// Higher angular momentum: deferred` block in `nao_driver.hpp` with full SK assembly.
- Validate: H₂O (DZP) overlap vs PySCF/ABACUS to ≤1e-6.

### Step B: Add ONCV local pseudopotential on the grid
- Read `Pseudopotential` for each element type via `Upf2Reader::Read`.
- In `NaoDriver`, build `v_local_grid` as sum over atom types of `v_local(|r-R_A|)` interpolated onto the UPF radial grid.
- Add `v_local_grid` to `v_ext_grid` (replacing `−Z/|r−R|` for PP atoms).
- Keep all-electron path for H (Z_valence = 1, no PP file, or use a tiny H PP).

### Step C: Add Kleinman-Bylander nonlocal potential as a matrix
- For each atom, compute `V_nl` matrix in the NAO basis: `V_nl_{ij} = Σ_l |β_l^i><β_l|δV_l|β_l><β_l^j| * E_kin` (standard KB form).
- Add `V_nl` to `H` in every SCF iteration.
- Validate: Si₂ (ONCV) energy vs ABACUS/PySCF with same PP to ≤1e-5 Ha.

### Step D: Wire Python binding and benchmark
- Expose `NaoDriver::Run` in `tides/_native.cpp`.
- Add `--driver=nao` mode to `bench/pyscf_benchmark.py`.
- Run matched-basis/matched-PP comparisons on H, H₂, He, H₂O, SiH₄.

### Step E: Force correctness (next RALPH cycle)
- Implement `NaoDriver` forces (HF + Pulay + PP + grid) and FD validation.

## Data contracts
- `Pseudopotential` is the canonical PP struct. `Upf2Reader` produces it; `NaoDriver` consumes it.
- `NaoBasis` is the canonical basis struct. `NaoGenerator` produces it; `TwoCenterIntegralEval` and `NaoDriver` consume it.
- `RadialTable` is the canonical two-center table. `TwoCenterIntegralEval` builds it; `NaoDriver` uses it.
- `SCFDriver::Run` contract: `build_H(P) -> H`, `energy_fn(P, eigenvalues) -> E_total`. Unchanged.

## Test plan (PHASE P)
1. `wp2_nao_generation` still passes.
2. `wp2_pseudo_validators_and_ghost_detector` still passes.
3. New test: `nao_driver_hydrogen_atom` — H atom all-electron DZP energy vs PySCF to ≤1e-4.
4. New test: `nao_driver_h2_molecule` — H₂ DZP energy vs PySCF to ≤1e-4.
5. New test: `nao_driver_silicon_dimer` — Si₂ with ONCV PP energy vs ABACUS to ≤1e-3.
6. New test: `two_center_integrals_all_angular` — all (l_a,l_b)≤2 vs direct 3D integration to ≤1e-7.

## Risks and mitigations
- **UPF2 files missing**: use `ncpp` files or convert from PseudoDojo. If unavailable, generate a synthetic H PP for testing and flag T2.3 as blocked.
- **KB nonlocal accuracy**: start with a local-only PP path for H/C/O; validate before adding KB.
- **Grid convergence**: use a 0.2–0.3 Bohr grid and document grid error.
- **Two-center SK correctness**: validate against direct 3D numerical integration for each (l_a,l_b) pair.

## First deliverable (Phase A exit)
A working `NaoDriver::Run` that can run H and Si₂ with a pseudopotential and produce a total energy, exposed in Python, with a new test that passes.
