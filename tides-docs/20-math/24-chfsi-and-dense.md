# Mid-range solvers: batched dense (R0) and ChFSI (R1)
R0: batched Jacobi eigensolver (cuSOLVER syevjBatched) over thousands of small H per GPU; one CUDA
graph per SCF sweep; structure-of-arrays over systems.
R1: Chebyshev-filtered subspace iteration — filter degree from spectral bounds; Rayleigh–Ritz in the
filtered subspace; locking of converged states; subspace REUSE across SCF steps and MD steps (the big
practical win). Dense fallback bridge: cuSOLVERMp / ELPA for validation and for pathological cases.
## Observables of understanding
Residual ||H x - e S x|| <= 1e-9 on gauntlet; ChFSI SCF iteration count within +2 of dense reference.
