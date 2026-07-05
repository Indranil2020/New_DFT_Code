# WP10 — API, docs, community, releases (owner S10)
Purpose: the project is usable, teachable, and governed. Standards: 30-architecture/35.

### T10.1 — nanobind bindings + Status objects
- Requirements: C++ exceptions never cross the boundary; Python returns Status/result objects;
  NO try/except control flow (linter ERR001); full type hints.
- Observables: bindings cover run/SCF/MD/forces; linter gate active in CI. Effort 5 pw. Depends T6.1.

### T10.2 — ASE calculator
- Observables: passes the ASE calculator interface test battery; used by all examples.
  Effort 4 pw. Depends T10.1.

### T10.3 — CLI: run / tune / bench / verify
- Observables: each subcommand documented; `tides verify` runs the ladder; `tides tune` writes the
  broker table. Effort 4 pw. Depends T10.1, T4.6.

### T10.4 — Input schema (TOML) + validator + auto-docs
- Observables: every key documented or the docs build fails; invalid input yields a precise Status
  message (no stack traces). Effort 4 pw. Depends –.

### T10.5 — Theory manual with derivations
- Observables: forces chapter complete by GA1; every merged physics module has its derivation section
  (enforced per 35-coding-standards). Effort 5 pw ongoing. Depends rolling.

### T10.6 — Five tutorials doubling as integration tests
- Observables: a new user reproduces the S22 benchmark from docs alone in <1 hour (tested on a
  volunteer). Effort 4 pw. Depends T10.2.

### T10.7 — JAX bridge (Phase B/C)
- Observables: energy_and_forces custom call with analytic-gradient VJP; gradcheck vs FD <=1e-6.
  Effort 4 pw. Depends T10.1, T6.3.

### T10.8 — Release engineering
- Observables: semver, CITATION.cff, changelog automation, signed tags; alpha (M12), v0.9 (M36),
  v1.0 (M60). Effort 3 pw. Depends T8.5.
