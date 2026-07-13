# TIDES Remediation Plan — Dependency-Ordered Roadmap
*Authored 2026-07-13. Grounds the 30-item audit in verified source state and sequences it into executable work packages. RALPH Phase R+A deliverable.*

---

## 0. Framing (read this first)

The 30 items are **not a bug list** — they are the proposal's 5-year / 10-scientist / ~450-person-month program restated as gaps. No single session "resolves all 30." This plan does three honest things:

1. **Reorders by dependency**, not by audit numbering. ~15 items are *void* until one root cause is fixed.
2. **Classifies each item by where it can be verified**: `CPU` (fully checkable in this environment), `GPU` (code can be written + CPU-fallback tested, but the performance/accuracy claim needs datacenter/RTX hardware), `CLUSTER` (needs multi-GPU/multi-node + hours–days of compute; cannot be validated here at all).
3. **States acceptance from the proposal's own gates** (§6/§7/§9), so "done" is falsifiable.

**Decision principle inherited from AGENT.md:** Accuracy > Speed. Evidence > Assumption. Correctness > Completion. An item is not "done" because code exists; it is done when its proposal gate is met and verified.

### Scope-completeness flag (fail-loud)
The pasted audit fully enumerates **B1–B15** and **M1, M5, M6, M9, M10, M11, M14, M15**. The "Entirely Missing" table is **truncated**: **M2, M3, M4, M7, M8, M12, M13 are not visible**. This plan covers the **23 verifiable items** and lists the 7 as `NEED-SOURCE`. Provide those rows to make the plan complete.

---

## 1. The critical-path insight

```
                    ┌──────────────────────────────────────────┐
                    │  P0  ACCURACY FOUNDATION (Becke grid)     │  ← ROOT
                    │  Fixes B1 B2 B3 B4 B5                     │
                    └───────────────┬──────────────────────────┘
        ┌───────────────┬───────────┼───────────────┬─────────────────┐
        ▼               ▼           ▼               ▼                 ▼
   P1 real PP      P2 diffuse   P3 XL-BOMD      P14 PAW          P15 hybrids
   (M11)           basis (M10)  shadow (B6/B7/  (M9)             (M15)
        │               │        M14)              │                 │
        └───────┬───────┴───────────┬──────────────┘                 │
                ▼                    ▼                                ▼
          P4 competitor        (accuracy-gated benchmarks) ──────────┘
          harness (B13)
```

Everything that reports a **number vs a reference** (B13 competitor rows, B14 10⁴-atom, B15 O(N), M14 XL-BOMD-vs-SCF, M15 hybrid accuracy, M11 PP ≤1e-4 Ha) is **meaningless until P0 lands**. This is why P0 is the sole root and why the earlier recommendation was to start there. The architectural-wiring items (GPU eigensolve, mixed precision, metering) are *parallel* — they change speed, not correctness, so they do not block P0 and P0 does not block them.

**Root cause of P0 (verified in source):** [molecule_driver.hpp:341](../tides/core/scf/molecule_driver.hpp#L341) sets `xc_in.grid_weight = dv` where `dv = grid_h³` on a **single uniform Cartesian grid** (default 0.15 Å). No Becke/Lebedev atom-centered grid exists anywhere in `tides/core/`. A uniform Cartesian grid cannot resolve the nuclear cusp ρ∝e^{−Zr}; hence H=0.08 Ha, H₂O=2.8 Ha. Both `molecule_driver` and `nao_driver` share this grid, so a single fix repairs all of B1–B5.

---

## 2. Work packages (dependency-ordered)

Each WP: **Items · Gate (proposal) · Verify-class · Files · Blockers · Definition of done.**

### P0 — Accuracy foundation: atom-centered integration grid  `CPU`
- **Items:** B1, B2, B3, B4, B5
- **Gate:** proposal §6 WP2/WP3 → total energy vs PySCF ≤1e-8 Ha is the *stretch*; the realistic first milestone is **≤1e-4 Ha (1 meV/atom-class)** for H, H₂, H₂O, then push. Ne (B5, currently 65.8 Ha error) is the d/coarse-grid stress test.
- **Work:** implement Becke fuzzy-cell partition + per-atom radial quadrature (Mura–Knowles or Euler–Maclaurin) + Lebedev angular grid; route XC energy/potential integration (and optionally grid-Hartree) through it. Replace uniform `grid_weight = dv`.
- **Files:** new `tides/core/grid/atomic_grid.hpp` (+ `becke_weights`, `lebedev`, `radial_quad`); wire into [molecule_driver.hpp](../tides/core/scf/molecule_driver.hpp) and [nao_driver.hpp](../tides/core/scf/nao_driver.hpp) grid setup; XC path via `grid/xc/xc_engine.hpp`.
- **Blockers:** none. This is the unblocker.
- **Done when:** H/H₂/H₂O tolerances in [nao_driver_tests.cpp](../tides/core/scf/tests/nao_driver_tests.cpp), [molecule_driver_tests.cpp](../tides/core/scf/tests/molecule_driver_tests.cpp), [nao_benchmark_tests.cpp](../tides/core/scf/tests/nao_benchmark_tests.cpp) are tightened to the *actually achieved* value and the achieved value is reported honestly (not a loosened bar). A Ne test is added (B5) with its real achieved error.

### P1 — Real ONCV pseudopotentials validated  `CPU`
- **Items:** M11
- **Gate:** §6 WP2 → real PseudoDojo ONCV (UPF2/PSML) loaded, KB nonlocal in SCF, energy ≤1e-4 Ha vs reference.
- **Work:** load a real PseudoDojo file (H, then a TM), assemble KB nonlocal projectors into H, replace `MakeTrivialPP()` (v_local=−Z/r) in [pp_scf_validation_tests.cpp](../tides/core/scf/tests/pp_scf_validation_tests.cpp) with a real PP.
- **Files:** [basis/pseudo/upf2_reader.hpp](../tides/core/basis/pseudo/upf2_reader.hpp), [pseudopotential.hpp](../tides/core/basis/pseudo/pseudopotential.hpp) (reader infra already exists), KB assembly in `ham/` + `nao_driver`.
- **Blockers:** **P0** (cannot certify PP accuracy on a broken integration grid). Needs a real PseudoDojo file checked into `verification/references/` or `external/`.
- **Done when:** one real ONCV element runs SCF and matches an external reference ≤1e-4 Ha, with nonlocal projectors actually contributing (assert `V_nl ≠ 0`).

### P2 — Diffuse basis augmentation  `CPU`
- **Items:** M10
- **Gate:** §3.1.1 → diffuse-augmented sets for anions/surfaces; §6 WP7 S22 MAD ≤0.35 kcal/mol.
- **Work:** add diffuse (small-exponent) radial functions to the DZP/TZP recipes in [nao_generator.hpp](../tides/core/basis/nao_generator.hpp); validate on an anion or S22 dimer.
- **Blockers:** **P0**.
- **Done when:** a diffuse-augmented basis measurably improves an anion/electron-affinity or S22 binding energy vs DZP, documented.

### P3 — XL-BOMD true shadow dynamics on real DFT  `CPU`
- **Items:** B6, B7, M14
- **Gate:** §6 WP6 → ~1 solve/step (no per-step SCF), NVE drift ≤30 µHa/atom/ps on 64-H₂O; §3.2 KSA kernel = real low-rank Jacobian inverse.
- **Work:** (a) make `density_fn` in [nao_driver.hpp:2656](../tides/core/scf/nao_driver.hpp#L2656) *not* run a full SCF each step — one build + one solve on the propagated shadow density; (b) replace the heuristic `kernel_order` scaling in [xlbomd.hpp](../tides/core/dynamics/xlbomd/xlbomd.hpp) with a genuine low-rank kernel/Krylov (KSA) update; (c) point the drift test at the real NAO engine instead of a model Hamiltonian.
- **Blockers:** **P0** (drift on a wrong potential is meaningless).
- **Done when:** a real-NAO NVE run shows ≤30 µHa/atom/ps drift with 1 solve/step, and the KSA kernel is a documented low-rank inverse, not magnitude scaling.

### P4 — Competitor benchmark harness executed  `CPU`(TIDES side) / external
- **Items:** B13
- **Gate:** §9.1 12-row matrix; each row is fixed-accuracy time-to-solution.
- **Work:** wire `competitor_farm.hpp` to actually emit TIDES-side rows (atoms/s, time/SCF, kWh) for the systems it parses; competitor columns require the competitor codes installed (ABACUS/CP2K/SIESTA/SPARC/GPAW) — out of scope for a code session but the harness + TIDES numbers are in scope.
- **Blockers:** **P0, P1, P3** (numbers must be at real accuracy). Competitor binaries.
- **Done when:** ≥1 real benchmark row with a TIDES number at a stated, verified accuracy is emitted to a report file.

### P5 — GPU eigensolve wired into SCF  `GPU`
- **Items:** B8 (partial), B9
- **Gate:** §6 WP4 → GPU-native compute path.
- **Work:** call `cusolver_batched.hpp` from [scf_driver.hpp](../tides/core/scf/scf_driver.hpp) under `TIDES_HAVE_CUDA`, keep `dsyev_` as CPU fallback.
- **Verify:** CPU fallback unit-testable here; **GPU speed/parity needs hardware** — cannot certify in this environment.
- **Blockers:** none (parallel to P0). GPU for validation.
- **Done when:** on a CUDA build, SCF diagonalization uses cuSOLVER and matches the LAPACK path bit-close; **flagged as hardware-unverified until run on a GPU.**

### P6 — Tile substrate as compute backbone  `GPU`
- **Items:** B8
- **Gate:** §6 WP1 "one true layer" → rho/vmat/eigen GEMMs flow through TileMat grouped GEMM, not stray CPU dgemm.
- **Work:** route [BuildRhoGemm/BuildHmatGemm](../tides/core/scf/nao_driver.hpp) through the tile GEMM path so the 967-GFLOPS kernel has a consumer.
- **Blockers:** P5 desirable first. GPU for validation.
- **Done when:** SCF GEMMs dispatch through TileMat; CPU parity held; **GPU throughput unverified here.**

### P7 — Tensor-core mixed precision in SCF  `GPU`
- **Items:** B11
- **Gate:** §3.2 → BF16/FP16 tensor-core GEMM + Ozaki f64e reductions; nightly A/B ≤0.5 meV/atom drift.
- **Work:** call the existing GPU Ozaki kernel from the SCF build_H path instead of CPU element-wise `QuantizeMatrix`.
- **Blockers:** P6. **GPU-gated** — tensor-core path cannot run on CPU.
- **Done when:** mixed-vs-FP64 A/B ≤0.5 meV/atom on a GPU build; **hardware-unverified here.**

### P8 — Real energy metering  `GPU`
- **Items:** B12
- **Gate:** §9 → measured kWh via NVML/rocm-smi in every report.
- **Work:** replace hardcoded `125.0*16*0.7` / `350.0*0.8` at [nao_driver.hpp:2153](../tides/core/scf/nao_driver.hpp#L2153) with `nvmlDeviceGetPowerUsage` (GPU) and RAPL (CPU).
- **Blockers:** GPU/NVML for real GPU numbers; RAPL path is `CPU`-testable.
- **Done when:** metering reads real counters, not TDP estimates; CPU-RAPL path verified here, GPU-NVML flagged unverified.

### P9 — Broker regime dispatch actually triggered  `CPU`(R2 small) / `CLUSTER`(R2/R3 large)
- **Items:** B10
- **Gate:** §4 → R0/R1/R2/R3 all reachable; crossovers measured.
- **Work:** add test systems with n_basis > 200 so R1/R2/R3 fire; run a small R2 (SP2) case on CPU.
- **Blockers:** P0. Large-N R2/R3 need GPU/cluster.
- **Done when:** a test drives each regime past the broker threshold; small R2 verified on CPU; large-N flagged.

### P10 — Multi-block GPU SP2  `GPU`
- **Items:** M1
- **Gate:** §6 WP5 → multi-block GPU SpGEMM purification.
- **Blockers:** P6. **GPU-gated.**

### P11 — METIS at runtime  `CPU`
- **Items:** M5
- **Work:** compile the now-present `external/metis/` submodule, define `TIDES_HAVE_METIS`, exercise the partitioner in [graph_partitioner.hpp](../tides/core/parallel/graph_partitioner.hpp).
- **Blockers:** build wiring only.
- **Done when:** a partitioning test runs real METIS (not the RCB fallback) and the `#ifdef` path is live.

### P12 — Distributed / multi-node execution  `CLUSTER`
- **Items:** M6
- **Verdict:** **cannot be validated in this environment.** Requires ≥2 nodes + NVSHMEM/NCCL fabric. Deliverable here = code + single-node CPU simulation only, explicitly labeled as such.

### P13 — 10⁴-atom run + O(N) scaling 10⁴→10⁶  `CLUSTER`
- **Items:** B14, B15
- **Verdict:** **cannot be validated here.** A real 10⁴-atom SP2 needs a GPU + tens of GB; 10⁶ needs 8–32 GPUs (proposal §2.2, ~0.5 TB). Deliverable here = correct-scaling code + honest extrapolation labeled as extrapolation, never as a measured 10⁶ result.

### P14 — PAW in NAO  `CPU`(physics) — multi-week
- **Items:** M9
- **Gate:** §6 WP7 (Y3 thrust). Currently a 154-line memo, zero code.
- **Blockers:** P0. Large standalone effort; schedule as its own multi-session track.

### P15 — Full 4-center exact-exchange hybrid SCF  `CPU`(physics) — multi-week
- **Items:** M15
- **Gate:** §6 WP7 → HSE06 on a real molecule/slab ≤4× own-PBE. Currently screening approximation / model system only.
- **Blockers:** P0. Depends on ISDF+ACE being real. Own multi-session track.

### P-NEED-SOURCE — unenumerated missing items
- **Items:** M2, M3, M4, M7, M8, M12, M13 — **not in the pasted audit.** Cannot plan without the rows.

---

## 3. Recommended execution order

| Phase | WPs | Verify | Rationale |
|---|---|---|---|
| **1 (foundation)** | P0 | CPU | Unblocks 15 downstream items. Start here. |
| **2 (real physics)** | P1, P2, P3 | CPU | The physics that P0 makes meaningful. Parallelizable. |
| **3 (measurement)** | P4, P9(small) | CPU | Real numbers once physics is real. |
| **4 (wiring, hardware-gated)** | P5, P6, P7, P8, P11 | GPU/CPU | Speed, not correctness. CPU fallbacks testable now; GPU claims deferred to hardware. |
| **5 (scale, cannot verify here)** | P10, P12, P13 | CLUSTER | Code only; validation explicitly deferred. |
| **6 (multi-week physics)** | P14, P15 | CPU | Standalone tracks, one at a time. |

## 4. What this environment can and cannot deliver (no ambiguity)
- **CAN fully verify:** P0, P1, P2, P3, P4(TIDES side), P9(small-N), P11, and the CPU fallbacks of P5–P8.
- **CAN write code, CANNOT verify the headline claim:** P5, P6, P7, P8, P10 (need a GPU), P12 (need a cluster).
- **CANNOT deliver a validated result at all:** P13 (10⁴/10⁶-atom), the multi-node half of P12.

Any report that marks a GPU/CLUSTER item "done" from a CPU session is a fail-loud violation and will not be written.
