# TIDES documentation pack — v2.0 (workstation-first revision)
One folder = one concern. Every engine (WP) file in `40-engines/` decomposes into small tasks,
each with: Problem, Starting point, Requirements, Observables (= Definition of Done), Effort, Depends/Unblocks.
A task is assignable to one researcher and verifiable without trusting the researcher.

Reading order for a new team member:
1. `00-project/00-vision-scope-claims.md` → `01-hardware-strategy.md` → `02-roadmap-phases-milestones.md`
2. Your WP file in `40-engines/`, then the physics/math files it links.
3. `30-architecture/31-data-contracts.md` and `35-coding-standards.md` before writing any code.

Folders:
- `00-project/`  vision, hardware strategy, roadmap, team, risks, governance, task management
- `10-physics/`  the physical model: basis, pseudos, XC, electrostatics, T_e, hybrids, forces
- `20-math/`     the numerics: tiles, mixed precision, purification, FOE/SQ, ChFSI, XL-BOMD, QTT, error control
- `30-architecture/` repo layout, data contracts, broker, precision policy, parallelism, coding standards
- `40-engines/`  WP1–WP10: one file per engine with the full task decomposition
- `50-verification/` test ladder, tolerances, reference data
- `60-benchmarks/`  protocol, piecewise competitor matrix, campaigns

Conventions: task IDs are T<wp>.<n>; effort in person-weeks (pw); "RTX" = 24 GB RTX-4090-class
(primary CI device), "A40" = 48 GB, "H100" = when available. Every performance observable names its device.
