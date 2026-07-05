# Benchmarking protocol — rules of engagement (every published number)
1 Fixed-accuracy only: time-to-solution at stated, verified accuracy (<=1 meV/atom solids vs each
  code's own converged reference; <=0.5 kcal/mol molecular). Never raw walltime at unmatched settings.
2 Same physics where possible: PBE(+D4), same PseudoDojo ONCV files, same k/smearing; documented
  basis/grid convergence for every code.
3 Three repeats; cold and warm reported separately; kWh via NVML/rocm-smi in every table.
4 Publish commits, inputs, logs, container digests (CC-BY, DOI per campaign).
5 Competitors at their best-known settings; their developers invited to review configs pre-publication.
6 Where a competitor cannot run a payload (size/BCs/feature), record a capability result, not a win.
