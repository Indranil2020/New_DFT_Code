# The six-rung correctness ladder (nothing skips a rung)
1 Kernel: every GPU kernel vs FP64 CPU oracle on random + adversarial inputs (clustered eigenvalues,
  tiny gaps, 1e+/-9 dynamic range); ULP/abs budgets from 51-tolerances.
2 Operator: S, T, V_nl, v_H, v_xc vs closed forms (Gaussian charges, hydrogenic states) and vs an
  external code with matched basis/pseudo (PySCF molecules; ABACUS/SIESTA same-NAO).
3 Energy: totals vs R1 dense on controls; mixed-vs-FP64 A/B nightly <=0.5 meV/atom; virial vs stress.
4 Force: central 5-point FD on randomized displacements, per-term isolation, nightly; egg-box scan
  published per release.
5 Dynamics: NVE drift budget <=30 uHa/atom/ps; timestep convergence; XL-BOMD vs converged-SCF
  trajectory statistics (RDF, VACF).
6 Physics: ACWF/Delta subset lattice constants & bulk moduli; S22/W4-11 subsets; slab work functions
  and adsorption energies vs curated references.
Bisect-the-physics: HDF5 stage dumps (31) let any stage be injected from a reference code to localize
a discrepancy in hours. Deterministic mode makes every bug bit-for-bit reproducible.
