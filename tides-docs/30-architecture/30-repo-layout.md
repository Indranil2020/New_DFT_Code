# Repository layout (monorepo `tides/`)
core/ (C++20, no Python deps)
  common/   status.hpp units.hpp config.hpp logging.hpp
  tile/     layout.hpp gemm_grouped.cu spgemm_filtered.cu ozaki.cu reduce_f64e.cu graphs.hpp tests/
  basis/    atomgen/ two_center.cu three_center.cu pseudo/ paw/(flag)
  grid/     dual_grid.hpp rho_build.cu vmat_build.cu poisson_fft.cu poisson_qtt/(flag) xc.cu
  ham/      assembly of S, H(rho), dH/dR streams
  solvers/  broker.cpp dense/ chfsi/ omm/ sp2_submatrix/ foe_sq/
  scf/      mixers, smearing, fermi search
  dynamics/ xlbomd/ md_driver/ optimizers/ neb/
  forces/   all analytic force+stress terms (one path, P-based)
  hybrids/  isdf/ ace/
  parallel/ partitioner, halos (NCCL; MPI/NVSHMEM Phase C), hdf5 io, restart
api/       python/(nanobind, ase_calculator.py) jax_bridge/ cli/
verification/  references/ tolerances.yaml runners/
benchmarks/    piecewise/ end2end/ energy_meter/ report/
ci/  docs/  examples/   (each example is also an integration test; Python: no try/except)
Rule: a directory = an owner (00-project/03); cross-directory changes need both owners' review.
