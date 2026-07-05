# Data contracts (frozen, versioned, contract-tested)
## TileMat (C++ and Python mirrors)
create(pattern, tile_sizes, dtype, scale_mode); spgemm_filtered(A,B,eps)->C+error_ledger;
axpy; trace_f64e; norm_f64e; to_dense (debug); symmetry flag; complex variant (R0/R1 only).
Stability: any signature change bumps schema version + migration test.
## GridArray
domain(BCs, spacings), halo spec, device residency; map_orbitals_to_grid / adjoint declared as a pair
with an adjointness contract test.
## HDF5 stage-dump schema (the bisect-the-physics enabler)
Stages: geometry | S | H0 | rho | vH | vxc | H | P | E_components | forces | stress.
Every stage dumpable and injectable; bitwise round-trip test; version attribute mandatory.
This is what lets any module be swapped against a reference code to localize a bug in hours.
## Status (error handling)
Typed status codes returned across the C API; C++ exceptions never cross the boundary;
Python raises nothing in library control flow — callers check Status (see 35-coding-standards).
