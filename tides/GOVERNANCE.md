# TIDES Governance

## Project Structure

TIDES is a monorepo with the following directory owners:

| Directory | Owner | Description |
|---|---|---|
| `core/` | Physics team | C++20/CUDA core: tile substrate, NAO, grids, solvers, SCF, dynamics |
| `api/` | API team (WP10) | Python bindings, ASE calculator, CLI, JAX bridge |
| `verification/` | Red team (WP9) | Tolerances, reference data, regression dashboard |
| `benchmarks/` | Performance team | Benchmark payloads, profiling |
| `ci/` | Release engineering | CI runners, packaging, spack environments |
| `docs/` | Documentation | Theory manual, tutorials, auto-docs |
| `examples/` | Education team | Tutorials doubling as integration tests |

## Decision Process

- **Technical decisions**: Propose in an ADR (Architecture Decision Record) in `docs/adr/`.
- **Cross-directory changes**: Require both directory owners' review.
- **Breaking changes**: Require a deprecation cycle (one minor release).

## Release Cadence

- **Alpha**: Monthly tags (`v0.1.0-alpha`, `v0.2.0-alpha`, ...).
- **Beta**: After ACWF subset passes (`v0.1.0-beta`).
- **Stable**: After full S22 + ACWF pass (`v1.0.0`).

## License

Apache-2.0. All contributions are licensed under the same terms.
