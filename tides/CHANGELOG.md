# Changelog

All notable changes to TIDES are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — WP10 (API, docs, community, releases)
- **T10.1**: Python `Status`/`Result` objects mirroring C++ `status.hpp`; `TidesCalculator` API facade with SCF, energy, forces, MD methods; nanobind binding stubs (`_native.cpp`).
- **T10.2**: ASE-compatible calculator (`TIDESCalculator`) supporting energy, forces, stress properties.
- **T10.3**: CLI with `run`, `tune`, `bench`, `verify` subcommands.
- **T10.4**: TOML input schema with 10 sections (system, basis, pseudo, xc, scf, solver, md, precision, grid, output); validator with precise error messages; auto-docs generator.
- **T10.5**: Theory manual stub with forces chapter derivation.
- **T10.6**: Five tutorials doubling as integration tests (single-point SCF, forces/optimization, XL-BOMD MD, TOML input, solver broker).
- **T10.7**: JAX bridge with `energy_and_forces` custom VJP and `gradcheck` vs FD.
- **T10.8**: `pyproject.toml`, `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`, `GOVERNANCE.md`, `LICENSE`.

### WP1–WP9 (prior work)
- WP1: Tile substrate (CSR-of-tiles, grouped GEMM, SpGEMM, Ozaki f64e, deterministic mode, CUDA graphs).
- WP2: NAO basis generation, two-center integrals, ONCV pseudopotential readers, derivative streams.
- WP3: Dual-grid, Poisson (FFT+ISF), rho builder, XC integration, grid forces.
- WP4: Batched dense eigensolver (R0), ChFSI, OMM, ELPA bridge, solver broker.
- WP5: SP2 CPU reference, submatrix construction, FOE/Chebyshev, Fermi-level search, truncation policy.
- WP6: SCF driver (Pulay/Kerker), energy assembly, analytic forces, XL-BOMD, optimizers (FIRE/L-BFGS), stress.
- WP7: D3/D4 dispersion, ISDF, ACE hybrids, PAW feasibility.
- WP8: METIS partitioner, halo exchange, HDF5 stage-dump, CI runners, packaging.
- WP9: Tolerances framework, reference data, FD force checks, regression dashboard.

## [0.1.0-alpha] — 2026-07-01

Initial alpha release. Molecular SCF + forces + XL-BOMD on the CPU reference backend.
