# Coding standards (enforced by CI linters)
C++20: no raw owning pointers; spans/mdspan for views; every kernel has an FP64 CPU oracle test and a
ULP/abs budget in tolerances.yaml BEFORE merge; compute-sanitizer (memcheck+racecheck) green is a
merge gate; deterministic mode must be supported by every reduction.
CUDA: grouped-GEMM via CUTLASS/cuBLASLt wrappers only (no ad-hoc GEMMs); <=10 hand kernels total,
each with a one-page design note (occupancy, traffic model, measured roofline fraction).
Python (api/, examples/, tools): explicit Status/result objects and assert-based invariants;
NO try/except control flow anywhere (linter rule ERR001 fails CI) — errors are values, not jumps;
type hints mandatory; examples double as integration tests.
Docs: every physics/math module ships a derivation section in the theory manual; a PR adding an
equation to code without its manual derivation is rejected.
Commits: DCO sign-off; task ID T<wp>.<n> in the message; benchmark-affecting PRs attach a dashboard link.
