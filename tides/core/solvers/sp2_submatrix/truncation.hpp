#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::solvers {

// Truncation policy + error compensation (T5.4).
//
// The submatrix method (T5.2) truncates the density matrix at a cutoff radius
// r_cut. This introduces error: P_truncated != P_exact. The error compensation
// tracks the dropped norm and corrects the trace/energy.
//
// Observable (T5.4): error-vs-radius curves published; compensation reduces
// energy error >= 5x at fixed radius on the a-Si:H control.
//
// Strategy: compute P at several r_cut values, measure the error vs the
// converged reference, and apply a trace correction:
//   P_corrected = P_truncated + (N_e - tr(P_truncated S)) / tr(S) * S
// This enforces tr(P_corrected S) = N_e exactly.

struct TruncationResult {
  std::vector<double> P_corrected;
  double trace_error_before = 0.0;
  double trace_error_after = 0.0;
  double energy_error_before = 0.0;  // vs reference (if provided)
  double energy_error_after = 0.0;
  double improvement_factor = 0.0;
};

class TruncationCompensation {
 public:
  // Apply trace correction to a truncated density matrix.
  //   n:       matrix dimension
  //   P_trunc: truncated density matrix (row-major)
  //   S:       overlap matrix
  //   n_e:     target electron count
  //   E_ref:   reference energy (for measuring improvement; 0 = skip)
  //   H:       Hamiltonian (for computing E = tr(P H); needed if E_ref != 0)
  static TruncationResult Apply(std::size_t n, const std::vector<double>& P_trunc,
                                const std::vector<double>& S, double n_e,
                                double E_ref = 0.0,
                                const std::vector<double>& H = {}) {
    TruncationResult res;
    res.P_corrected = P_trunc;

    // Trace before correction.
    double tr_before = SP2Purification::TraceS(n, P_trunc, S);
    res.trace_error_before = std::fabs(tr_before - n_e);

    // Energy before correction (if reference provided).
    if (E_ref != 0.0 && !H.empty()) {
      double E_before = SP2Purification::TraceAB(n, P_trunc, H);
      res.energy_error_before = std::fabs(E_before - E_ref);
    }

    // Trace correction: add delta * S / tr(S^2) to enforce tr(P S) = N_e.
    // P_corrected = P_trunc + (N_e - tr(P_trunc S)) / tr(S^2) * S
    const double tr_S2 = [&]() {
      double s = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          s += S[i * n + j] * S[j * n + i];
      return s;
    }();

    if (tr_S2 > 1e-30) {
      const double delta = (n_e - tr_before) / tr_S2;
      for (std::size_t i = 0; i < n * n; ++i)
        res.P_corrected[i] += delta * S[i];
    }

    // Trace after correction.
    double tr_after = SP2Purification::TraceS(n, res.P_corrected, S);
    res.trace_error_after = std::fabs(tr_after - n_e);

    // Energy after correction.
    if (E_ref != 0.0 && !H.empty()) {
      double E_after = SP2Purification::TraceAB(n, res.P_corrected, H);
      res.energy_error_after = std::fabs(E_after - E_ref);
      if (res.energy_error_before > 1e-30)
        res.improvement_factor = res.energy_error_before / res.energy_error_after;
    }

    return res;
  }

  // Build an error-vs-radius curve: compute P at several truncation radii,
  // measure the error vs a converged reference.
  struct RadiusPoint {
    int radius;
    double trace_error;
    double energy_error;
  };
  static std::vector<RadiusPoint> ErrorVsRadius(
      std::size_t n, const std::vector<double>& H,
      const std::vector<double>& S, double n_e,
      const std::vector<double>& P_ref,
      const std::vector<std::vector<std::vector<std::size_t>>>& neighbor_lists_by_radius,
      double lambda_min, double lambda_max) {
    std::vector<RadiusPoint> curve;
    for (std::size_t r = 0; r < neighbor_lists_by_radius.size(); ++r) {
      auto sub = SubmatrixBuilder::BuildAndPurify(
          n, H, S, n_e, 0.0, neighbor_lists_by_radius[r], lambda_min, lambda_max);
      double tr_err = std::fabs(SP2Purification::TraceS(n, sub.P, S) - n_e);
      double E_err = 0.0;
      // Energy error: |tr((P_sub - P_ref) H)|.
      std::vector<double> dP(n * n);
      for (std::size_t i = 0; i < n * n; ++i) dP[i] = sub.P[i] - P_ref[i];
      E_err = std::fabs(SP2Purification::TraceAB(n, dP, H));
      curve.push_back({static_cast<int>(r + 1), tr_err, E_err});
    }
    return curve;
  }
};

}  // namespace tides::solvers
