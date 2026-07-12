#pragma once

// ASPC (Always Stable Predictor-Corrector) density/DM extrapolation (§3.3.3).
//
// In Born-Oppenheimer MD, the initial density matrix for each SCF solve can
// be extrapolated from previous MD steps, reducing the number of SCF
// iterations needed. ASPC provides a stable order-k extrapolation:
//
//   P_pred(t) = Σ_{i=0}^{k} c_i * P(t - i*dt)
//
// where the coefficients c_i are determined by the order-k ASPC formula.
// For order 1: P_pred = 2*P(t-dt) - P(t-2dt)
// For order 2: P_pred = 3*P(t-dt) - 3*P(t-2dt) + P(t-3dt)
// For order 3: P_pred = 4*P(t-dt) - 6*P(t-2dt) + 4*P(t-3dt) - P(t-4dt)
//
// The general formula uses binomial coefficients with alternating signs:
//   c_i = (-1)^i * C(k+1, i+1)  for i = 0, ..., k
//
// Observable: solve reduction >= 3x vs cold starts (T6.6).

#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::dynamics {

class ASPCExtrapolator {
 public:
  explicit ASPCExtrapolator(int order = 3) : order_(order) {}

  // Add a density matrix to the history.
  void PushBack(const std::vector<double>& P) {
    history_.push_back(P);
    // Keep only the last (order_ + 1) entries.
    while (static_cast<int>(history_.size()) > order_ + 1)
      history_.erase(history_.begin());
  }

  // Predict the next density matrix via ASPC extrapolation.
  // Returns empty vector if not enough history.
  std::vector<double> Predict() const {
    if (history_.size() < 2) return {};

    int k = std::min(order_, static_cast<int>(history_.size()) - 1);
    std::size_t n = history_[0].size();
    std::vector<double> predicted(n, 0.0);

    // ASPC coefficients: c_i = (-1)^i * C(k+1, i+1)
    for (int i = 0; i <= k; ++i) {
      double coeff = BinomialSign(k + 1, i + 1);
      int idx = static_cast<int>(history_.size()) - 1 - i;
      if (idx < 0) break;
      for (std::size_t j = 0; j < n; ++j)
        predicted[j] += coeff * history_[idx][j];
    }

    return predicted;
  }

  // Check if enough history is available for extrapolation.
  bool Ready() const { return static_cast<int>(history_.size()) >= 2; }

  // Get the order.
  int Order() const { return order_; }

  // Clear history (e.g., when the system changes significantly).
  void Clear() { history_.clear(); }

  // Get the number of stored history points.
  std::size_t HistorySize() const { return history_.size(); }

 private:
  // Compute (-1)^i * C(n, k) — the ASPC coefficient.
  static double BinomialSign(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    double result = 1.0;
    for (int i = 0; i < k; ++i)
      result *= static_cast<double>(n - i) / static_cast<double>(i + 1);
    // Apply (-1)^(k-1) since ASPC uses c_i = (-1)^i * C(k+1, i+1)
    // and the sign alternates.
    if ((k - 1) % 2 != 0) result = -result;
    return result;
  }

  int order_;
  std::vector<std::vector<double>> history_;
};

}  // namespace tides::dynamics
