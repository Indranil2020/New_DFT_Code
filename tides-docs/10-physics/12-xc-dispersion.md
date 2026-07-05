# Exchange–correlation and dispersion
## Decisions
libxc as the single XC source. Launch set: LDA-PW92, PBE, PBEsol; collinear spin. r2SCAN (mGGA)
is a Phase-B addition (needs tau on the grid — plan kernels accordingly). Dispersion: DFT-D3(BJ)
and D4 via the reference open libraries (simple-dftd3, dftd4), wired into E, F, stress from Phase A.
## Observables of understanding
He and Ne atom total energies vs PySCF (same functional, matched integration) <=1e-8 Ha;
D4 energies/forces vs reference implementation <=1e-10 on a 10-dimer set.
