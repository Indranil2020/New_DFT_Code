# XL-BOMD shadow dynamics — the default MD engine
## Idea
Auxiliary electronic degrees of freedom n(t) evolve harmonically around the ground state; nuclei move
on a SHADOW potential exactly consistent with the approximate electronic solution => time-reversible,
energy-conserving MD with ~ONE density-matrix solve per step and NO SCF loop. Kernel (approximate
inverse Jacobian) integrated by low-rank Krylov approximation (KSA); fractional-occupation version
handles small/unsteady gaps (graph-based shadow MD lineage, incl. tensor-core demonstrations).
## Decisions
Default MD mode from Phase B; per-step SCF fallback switch; thermostats: Langevin + Nose–Hoover chains;
ASPC extrapolation for warm starts in plain SCF mode.
## Read first
Niklasson PRL 100, 123004 (2008); JCTC 16, 3628 (2020) (density-matrix, fractional occ.);
JCP 152, 104103 (2020) (KSA); JCP 158, 074108 (2023) (graph-based shadow MD).
## Kill numbers (gate GB2)
NVE 64-H2O, 0.5 fs, 100 ps: drift <=30 uHa/atom/ps at ~1 solve/step on RTX; RDF statistically
indistinguishable from converged SCF-MD.
