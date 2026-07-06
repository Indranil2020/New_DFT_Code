# Electrostatics and boundary conditions
## Decisions
Dual real-space grid (coarse: orbital ops; fine: density/potential, default h ~ 0.15 A).
Hartree: cuFFT for 3D-periodic; interpolating-scaling-function (ISF) kernels for free/wire/slab
(BigDFT-PSolver approach) => all BCs in one interface, no supercell vacuum tricks for the Poisson step.
Ion-ion: Ewald (periodic) / direct (finite). Neutralization and dipole corrections for slabs.
ESP/prolate Ewald is a Phase-B/C candidate backend for periodic long-range electrostatics, especially
when FFT communication becomes limiting. Implementation reading:
- `tides-docs/10-physics/s41467-026-73232-8_reference.pdf` for fast Ewald summation with prolates.
- `tides-docs/10-physics/ankh-a-generalized-o(n)-interpolated-ewald-strategy-for-molecular-dynamics-simulations.pdf`
  as adjacent FFT-free/interpolated Ewald context and a scalability comparison point.
This backend is not default until energy, forces, stress, and neutralization match the FFT/Ewald path
at fixed accuracy and show a measured speed or communication win.
Egg-box control: fine-grid density + filtered projection; published egg-box scan per release.
## Observables of understanding
Analytic Gaussian-charge Hartree energy to <=1e-10 Ha under all four BCs; slab dipole correction
reproduces the textbook potential step for an asymmetric slab. ESP/prolate prototype, if enabled,
matches conventional periodic Ewald/FFT energies and forces on neutral and charged-compensated cells.
