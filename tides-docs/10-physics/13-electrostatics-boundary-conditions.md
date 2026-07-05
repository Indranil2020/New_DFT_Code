# Electrostatics and boundary conditions
## Decisions
Dual real-space grid (coarse: orbital ops; fine: density/potential, default h ~ 0.15 A).
Hartree: cuFFT for 3D-periodic; interpolating-scaling-function (ISF) kernels for free/wire/slab
(BigDFT-PSolver approach) => all BCs in one interface, no supercell vacuum tricks for the Poisson step.
Ion–ion: Ewald (periodic) / direct (finite). Neutralization and dipole corrections for slabs.
Egg-box control: fine-grid density + filtered projection; published egg-box scan per release.
## Observables of understanding
Analytic Gaussian-charge Hartree energy to <=1e-10 Ha under all four BCs; slab dipole correction
reproduces the textbook potential step for an asymmetric slab.
