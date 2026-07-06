# Contributing to TIDES

Thank you for your interest in contributing to TIDES! This document covers the
essential standards and workflows.

## Development Setup

```bash
git clone https://github.com/tides-dft/tides.git
cd tides
pip install -e ".[dev,ase]"
```

## Coding Standards (enforced by CI)

### C++20
- No raw owning pointers; use `std::unique_ptr` / `std::span` / `mdspan`.
- Every CUDA kernel has an FP64 CPU oracle test with a ULP/abs budget in `tolerances.yaml`.
- `compute-sanitizer` (memcheck + racecheck) green is a merge gate.
- Deterministic mode must be supported by every reduction.

### Python
- **No `try`/`except` control flow** anywhere (linter rule ERR001 fails CI).
  Use explicit `Status`/`Result` objects and `assert`-based invariants.
  Errors are values, not jumps.
- Type hints mandatory.
- Examples double as integration tests.

### Commits
- DCO sign-off required (`git commit -s`).
- Task ID `T<wp>.<n>` in the commit message.
- Benchmark-affecting PRs attach a dashboard link.

## Pull Request Process

1. Open an issue describing the change (or link an existing one).
2. Create a branch named `T<wp>.<n>-<short-description>`.
3. Implement the change following the coding standards above.
4. Ensure all tests pass: `pytest` and `ctest`.
5. If adding a physics/math equation to code, include its derivation in the theory manual.
6. Open a PR with a description linking to the issue and task ID.

## Testing

```bash
# C++ tests
cd build && ctest --output-on-failure

# Python tests
pytest api/python/tests/ -v

# Tutorials as integration tests
pytest examples/ -v

# Verification ladder
python -m tides.cli verify
```

## Release Process

1. Update `CHANGELOG.md` with all changes since the last release.
2. Bump version in `pyproject.toml` and `CITATION.cff` (semver).
3. Create a signed git tag: `git tag -s v0.1.0-alpha`.
4. Build distribution: `python -m build`.
5. Publish to PyPI (for stable releases only).
6. Publish a GitHub Release with the changelog excerpt.

## License

Apache-2.0. All contributions are licensed under the same terms.
