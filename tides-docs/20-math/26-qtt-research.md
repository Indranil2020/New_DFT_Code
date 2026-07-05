# QTT research thrust (flag-gated; never on the critical path)
## Idea
Quantics tensor trains: index a 2^n grid by binary digits of the coordinate; smooth-with-sharp-features
fields (electron density, Hartree potential) compress with low bond dimension => multiscale adaptivity
as DENSE regular linear algebra (GPU/tensor-core friendly), sidestepping irregular adaptive meshes.
Construction by tensor cross interpolation (TCI). Relevant 2023–2025 results: QTT PDE solvers,
multiscale interpolative constructions, and tensorized orbitals for chemistry (PRB 111, 2025).
## Scope here
ONLY rho(r) compression and the Hartree solve, as an alternative backend to 13-electrostatics.
## Gates
R-1 (M30): continue iff >=2x speed OR >=4x memory vs FFT path at equal accuracy on 64-H2O.
R-2 (M48): merge as default backend or archive with a public tech report. Owners: S3+S5 at 20%.
