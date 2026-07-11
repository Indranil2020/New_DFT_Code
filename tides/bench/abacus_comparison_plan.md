# ABACUS Benchmark Comparison Plan

## Objective
Compare TIDES NAO+pseudopotential SCF against ABACUS (Atomic-orbital Based Ab-initio Computation at UStc)
using matched NAO basis sets and pseudopotentials.

## Prerequisites
1. ABACUS installed (CPU version, with NAO support)
2. Matched ONCV pseudopotentials from PseudoDojo (same UPF2 files)
3. Matched DZP NAO basis sets (same radial cutoffs and confinement)
4. Same XC functional (LDA-PW92 or PBE)
5. Same grid spacing for real-space integration

## Benchmark Systems (from proposal §9.1)
| System | Atoms | Electrons | Basis | Purpose |
|---|---|---|---|---|
| H atom | 1 | 1 | DZP | Single-atom validation |
| H2 | 2 | 2 | DZP | Dimer binding energy |
| H2O | 3 | 10 | DZP | Molecular energy |
| Ne atom | 1 | 10 | DZP | All-electron vs PP |
| CH4 | 5 | 10 | DZP | Covalent bonding |
| Benzene | 12 | 42 | DZP | Larger molecule |

## Metrics
1. **Total energy** (Ha) — tolerance: 1e-3 Ha (0.03 eV) for matched basis/PP
2. **Energy components**: E_kin, E_ne, E_H, E_xc, E_ion
3. **SCF iterations** — convergence speed comparison
4. **Wall time** — per-SCF-iteration and total
5. **Forces** (Ha/Bohr) — FD5 validation against ABACF analytic

## Current Status
- TIDES NaoDriver: works for H, H2 (passes tests)
- TIDES pseudopotential: parser + KB projector assembly implemented but NOT validated
- ABACUS: not installed in current environment
- Pseudopotentials: UPF2 reader exists, no PP files in repo

## Blockers
1. ABACUS requires a separate build/install
2. Need PseudoDojo ONCV UPF2 files (not in repo)
3. TIDES NAO driver uses model Gaussian radials, not real confined-atom NAO
   (this would make the comparison unfair — model vs production basis)

## Next Steps
1. Install ABACUS (spack install abacus or build from source)
2. Download PseudoDojo ONCV pseudopotentials for H, C, O, Ne
3. Generate matched NAO basis (same confinement parameters)
4. Run both codes on benchmark systems
5. Compare energies, forces, timing

## Honest Assessment
This comparison cannot be done until:
- Real confined-atom NAO radial functions are used (not model Gaussians)
- Pseudopotentials are actually applied in the SCF loop
- ABACUS is installed

These are Phase B tasks per the proposal roadmap.
