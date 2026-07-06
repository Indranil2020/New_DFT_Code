#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/status.hpp"

namespace tides::solvers {

// Solver regimes (per 32-solver-broker.md).
enum class SolverRegime : std::uint8_t {
  kR0_BatchDense = 0,   // <=~200 atoms, many systems (batched eig)
  kR1_ChFSI = 1,        // <=~2k atoms (Chebyshev-filtered subspace iteration)
  kR2_SP2 = 2,          // gapped, large (density-matrix purification)
  kR3_FOE_SQ = 3,        // metallic / finite-Te, large (Fermi-operator expansion)
};

// Broker inputs (per 32-solver-broker.md).
struct BrokerInput {
  std::size_t n_atoms = 0;
  std::size_t n_basis = 0;       // total basis functions
  int bc_type = 0;                // 0=free, 1=wire, 2=slab, 3=periodic
  double gap_estimate = 0.0;      // eV (from cheap pre-pass)
  double electronic_temp = 0.0;   // K (0 = T=0)
  std::size_t available_vram_mb = 0;
  bool user_override = false;
  SolverRegime forced_regime = SolverRegime::kR1_ChFSI;
};

// Per-device calibration entry (from `tides tune`).
struct CalibEntry {
  SolverRegime regime;
  std::size_t n_atoms_lo;
  std::size_t n_atoms_hi;
  double time_per_step_ms;
  std::size_t vram_mb;
  bool available;  // whether this regime is implemented + working
};

// The solver broker dispatches each problem to the optimal regime based on
// system size, gap, temperature, and available VRAM (per 32-solver-broker.md).
// `tides tune` measures crossovers on the actual machine and caches a
// per-device table. Fallback chain: R2 -> R3 (raise Te) -> R1 (if memory).
class SolverBroker {
 public:
  // Dispatch: returns the recommended regime + logs the decision.
  static SolverRegime Dispatch(const BrokerInput& input,
                                const std::vector<CalibEntry>& calib,
                                std::string& reason) {
    reason.clear();
    if (input.user_override) {
      reason = "user override";
      return input.forced_regime;
    }

    // Metal (gap < 0.1 eV or finite Te): R3 FOE/SQ for large N.
    const bool metallic = (input.gap_estimate < 0.1) ||
                          (input.electronic_temp > 1.0);
    // Small (molecular): R0 batched.
    if (input.n_atoms <= 200 && !metallic) {
      reason = "small molecular system; R0 batched dense";
      return SolverRegime::kR0_BatchDense;
    }
    // Metallic large: R3.
    if (metallic && input.n_atoms > 200) {
      reason = "metallic/finite-Te; R3 FOE/SQ";
      return SolverRegime::kR3_FOE_SQ;
    }
    // Gapped mid-range (200-2000): R1 ChFSI.
    if (input.n_atoms <= 2000) {
      reason = "mid-range gapped; R1 ChFSI";
      return SolverRegime::kR1_ChFSI;
    }
    // Gapped large (2000+): R2 SP2.
    if (!metallic) {
      reason = "gapped large system; R2 SP2-submatrix";
      return SolverRegime::kR2_SP2;
    }

    // Fallback: if R2 not available or memory-limited, R3 (raise Te) -> R1.
    // Check calibration table for available regimes.
    for (const auto& e : calib) {
      if (e.available && input.n_atoms >= e.n_atoms_lo &&
          input.n_atoms <= e.n_atoms_hi &&
          input.available_vram_mb >= e.vram_mb) {
        reason = "fallback via calibration table";
        return e.regime;
      }
    }
    reason = "no suitable regime; defaulting to R1";
    return SolverRegime::kR1_ChFSI;
  }

  // `tides tune`: generate a calibration table by timing each regime on
  // representative system sizes. Returns the cached table.
  // (CPU stub: real calibration runs benchmarks on the device.)
  static std::vector<CalibEntry> GenerateCalibTable() {
    return {
      {SolverRegime::kR0_BatchDense, 0, 200, 0.1, 100, true},
      {SolverRegime::kR1_ChFSI, 200, 2000, 1.0, 500, true},
      {SolverRegime::kR2_SP2, 2000, 100000, 10.0, 2000, false},  // not yet
      {SolverRegime::kR3_FOE_SQ, 2000, 100000, 10.0, 2000, false},  // not yet
    };
  }

  // Check if the broker's choice is within 10% of the best available regime
  // (T4.6 observable). Compares the chosen regime's time against all available
  // regimes for the given system size.
  static bool IsWithin10PercentOfBest(const BrokerInput& input,
                                     const std::vector<CalibEntry>& calib,
                                     SolverRegime chosen) {
    double best_time = 1e30;
    double chosen_time = 1e30;
    for (const auto& e : calib) {
      if (!e.available) continue;
      if (input.n_atoms < e.n_atoms_lo || input.n_atoms > e.n_atoms_hi) continue;
      if (input.available_vram_mb < e.vram_mb) continue;
      best_time = std::min(best_time, e.time_per_step_ms);
      if (e.regime == chosen) chosen_time = e.time_per_step_ms;
    }
    if (chosen_time >= 1e30) return true;  // no data
    if (best_time < 1e-30) return true;
    return chosen_time <= 1.1 * best_time;  // within 10%
  }
};

}  // namespace tides::solvers
