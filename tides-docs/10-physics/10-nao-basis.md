# Numeric atom-centered orbitals (NAO)
## Purpose
Primary basis for all regimes: phi_nlm(r) = R_nl(|r-R_a|) Y_lm, with R_nl solutions of a confined
atomic Kohn–Sham problem on a logarithmic radial grid, strictly zero beyond r_cut (=> block sparsity).
## Decisions
- Default DZP (~13–15 fns/atom light elements, ~25 TM); TZP(+diffuse) for benchmarks/anions/surfaces.
- Charged molecules, anions, radicals, and slab spill-out require explicit diffuse-tail validation;
  diffuse/ghost/off-site functions are allowed only as documented basis recipes with convergence data.
- Confinement: smooth polynomial potential rising to infinity at r_cut; per-element r_cut recipes stored
  with the basis file; on-the-fly generation supported (deterministic given recipe hash).
- Basis file format: HDF5 (radial grids, splines, metadata, recipe hash); versioned.
## What to read first
Blum et al., FHI-aims, Comput. Phys. Commun. 180, 2175 (2009); Chen–Guo–He, J. Phys.: Condens. Matter
22, 445501 (2010) (systematically improvable optimized NAOs); SIESTA method papers.
## Observables of understanding (before coding)
Reproduce, with a scripted study: DZP->TZP monotone convergence of atomization energies on a
10-molecule set; show ghost-state detection on a deliberately over-diffuse basis; show charged UKS
anion/cation convergence with diffuse recipes and quantify density tail/spill-out error vs reference.
