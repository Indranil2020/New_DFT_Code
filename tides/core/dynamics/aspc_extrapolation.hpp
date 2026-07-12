#pragma once

// E5: ASPC (Always Stable Predictor-Corrector) density/DM extrapolation
// in production MD.
//
// ASPC provides a predictor for the density matrix at the next MD step based
// on the history of previous densities. This reduces the number of SCF
// iterations needed (>=3x solve reduction target, per T6.6).
//
// The ASPC predictor of order K uses:
//   P_pred(t+dt) = sum_{k=0}^{K-1} c_k * P(t - k*dt)
// where the coefficients c_k are chosen to minimize the extrapolation error
// while maintaining stability.
//
// For the standard ASPC (Kolm fatja), the coefficients are:
//   K=1: c = [2, -1]
//   K=2: c = [3, -3, 1]
//   K=3: c = [4, -6, 4, -1]
//   K=4: c = [5, -10, 10, -5, 1]
//   General: c_k = (-1)^k * C(K, k) * (K+1)/(K+1-k)
//
// Observable (E5): ASPC predictor reduces SCF iterations by >=3x vs cold start.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::dynamics {

// E5: ASPC extrapolation for density matrix prediction in MD.
class ASPCExtrapolator {
 public:
  explicit ASPCExtrapolator(int order = 3) : order_(order) {}

  // Add a new density matrix to the history.
  void Push(const std::vector<double>& P) {
    history_.push_back(P);
    if (static_cast<int>(history_.size()) > order_ + 1)
      history_.erase(history_.begin());
  }

  // Compute the ASPC predictor for the next density matrix.
  // Returns empty if insufficient history.
  std::vector<double> Predict() const {
    if (history_.size() < 2) return {};

    const int K = std::min(static_cast<int>(history_.size()) - 1, order_);
    if (K < 1) return {};

    // ASPC coefficients (Kolafa fatja, JCP 2004):
    //   c_k = (-1)^k * C(K+1, k+1)
    // For K=1: c = [2, -1]
    // For K=2: c = [3, -3, 1]
    // For K=3: c = [4, -6, 4, -1]
    std::vector<double> coeffs(K + 1);
    for (int k = 0; k <= K; ++k) {
      coeffs[k] = binomial(K + 1, k + 1);
      if (k % 2 == 1) coeffs[k] = -coeffs[k];
    }

    // P_pred = sum_{k=0}^{K} coeffs[k] * history_[history_.size() - 1 - k]
    std::size_t np = history_[0].size();
    std::vector<double> pred(np, 0.0);
    for (int k = 0; k <= K; ++k) {
      std::size_t idx = history_.size() - 1 - static_cast<std::size_t>(k);
      for (std::size_t i = 0; i < np; ++i)
        pred[i] += coeffs[k] * history_[idx][i];
    }
    return pred;
  }

  // Check if enough history is available for prediction.
  bool HasHistory() const { return history_.size() >= 2; }

  // Get the current order.
  int order() const { return order_; }

  // Clear history (e.g., when switching from MD to optimization).
  void Clear() { history_.clear(); }

  // Get the number of stored history entries.
  std::size_t history_size() const { return history_.size(); }

 private:
  static double binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    double result = 1.0;
    for (int i = 0; i < k; ++i)
      result *= static_cast<double>(n - i) / static_cast<double>(i + 1);
    return result;
  }

  int order_;
  std::vector<std::vector<double>> history_;
};

}  // namespace tides::dynamics
