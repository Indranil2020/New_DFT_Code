# Hybrid functionals (surface-chemistry pillar)
## Decisions
PBE0 and HSE06 via ISDF (interpolative separable density fitting; randomized point selection on the
fine grid) + ACE (adaptively compressed exchange) so exchange application costs like a semilocal term
per SCF iteration after construction. Short-range (HSE) screening implemented at tile level.
Target: <=4x own-PBE time/SCF at matched accuracy on a 500-atom slab (A40).
## What to read first
Lin, ACE, JCTC 12, 2242 (2016); Lu–Ying ISDF JCP (2015) lineage; CP2K-ADMM and QE-ACE docs as baselines.
## Observables of understanding
PBE0 for H2O and benzene vs PySCF <=0.1 mHa; a plot of exchange-energy error vs ISDF rank for benzene.
