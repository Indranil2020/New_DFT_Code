# How to run this project off these files
1. Each 40-engines/WPx file is the single source of truth for that engine's tasks.
2. To assign: copy one task block T<wp>.<n> into an issue; one researcher; the Observables list IS the
   Definition of Done — nothing is done until S9's harness shows it green.
3. A task must fit <=6 person-weeks; if not, the owner splits it and updates the WP file by PR.
4. A task may start when all "Depends" are green. Phase-A critical path:
   T1.1 -> T1.2 -> (T2.5, T3.2) -> T6.1 -> T6.3 -> GA1. Protect it.
5. Weekly per owner: (a) tasks green, (b) observable currently failing, (c) blocker.
6. Changing an Observable requires S9 approval, recorded in the WP file.
Task template:
### T<wp>.<n> — <title>
- Problem: <one self-contained paragraph>
- Start: <files to create; refs to read (10-physics/20-math links); inputs from tasks>
- Requirements: <interface + behavior spec>
- Observables (DoD): <numbered, measurable, device-named>
- Effort: <pw>. Depends: <IDs>. Unblocks: <IDs>.
