# Pseudopotentials
## Decisions
ONCV norm-conserving (Hamann PRB 88, 085117 (2013)) from PseudoDojo v0.4/0.5 standard tables —
the same files SPARC/QE/ABACUS/GPAW accept, enabling apples-to-apples benchmarking. Formats: UPF2
and PSML readers. Nonlocal part in Kleinman–Bylander form; nonlinear core correction supported.
PAW is a Phase-C decision (input: T7.6 memo; PAW-FE 2026 results show large DOF reductions).
## Validators (mandatory)
Norm checks; projector completeness; ghost-state detector (log-derivative scan); checksum against
PseudoDojo published hashes.
## Observables of understanding
Free-atom eigenvalues for H, C, O, Si vs PseudoDojo reference data to <=1e-4 Ha; a written half-page
on why KB separable form can create ghosts and how the detector catches them.
