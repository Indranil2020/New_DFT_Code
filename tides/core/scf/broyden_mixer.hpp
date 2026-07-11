#pragma once

// Broyden mixer for SCF acceleration.
//
// The Broyden method approximates the inverse Jacobian of the SCF mapping
// G(P) = P_new using a rank-1 (or limited-memory rank-m) update. It's
// an alternative to DIIS/Pulay that can be more robust for metallic systems
// and systems with small gaps.
//
// Algorithm (good Broyden / Broyden's second method):
//   Given: P_in (input density), P_out (output density from SCF step)
//   Residual: R = P_out - P_in
//   Update: dP = P_in - P_in_prev, dR = R - R_prev
//   Inverse Jacobian update: J^{-1} += (dP - J^{-1} dR) dR^T / (dR^T dR)
//   Next mixing: P_next = P_in - alpha * J^{-1} * R
//
// For limited-memory variant, we store m past (dP, dR) pairs and
// apply the update via a two-loop recursion similar to L-BFGS.
//
// Reference: Srivastava & Weaire (1987), Johnson (1988), Kresse & Furthmüller (1996)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace tides::scf {

struct BroydenConfig {
  int max_history = 10;       // Number of past iterations to store
  double alpha = 0.4;         // Mixing parameter (0 < alpha <= 1)
  double tolerance = 1e-8;    // Convergence tolerance on residual norm
  int max_iterations = 100;   // Maximum SCF iterations
  bool use_kerker = true;     // Apply Kerker preconditioning
  double kerker_q0 = 0.5;     // Kerker cutoff parameter
};

class BroydenMixer {
 public:
  // Initialize the mixer with the problem size.
  void Init(std::size_t n) {
    n_ = n;
    P_prev_.clear();
    R_prev_.clear();
    dP_list_.clear();
    dR_list_.clear();
    rho_list_.clear();
    B_.clear();
    first_step_ = true;
  }

  // Mix the input and output densities.
  //   P_in:  input density (what was used to build H)
  //   P_out: output density (what came from diagonalizing H)
  //   Returns: mixed density for the next SCF step.
  std::vector<double> Mix(const std::vector<double>& P_in,
                           const std::vector<double>& P_out,
                           const BroydenConfig& cfg) {
    if (P_in.size() != n_ || P_out.size() != n_) return P_out;

    // Compute residual: R = P_out - P_in.
    std::vector<double> R(n_);
    for (std::size_t i = 0; i < n_; ++i) R[i] = P_out[i] - P_in[i];

    std::vector<double> P_next(n_);

    if (first_step_) {
      // First step: simple mixing.
      for (std::size_t i = 0; i < n_; ++i)
        P_next[i] = P_in[i] + cfg.alpha * (P_out[i] - P_in[i]);
      first_step_ = false;
    } else {
      // Compute dP and dR.
      std::vector<double> dP(n_), dR(n_);
      for (std::size_t i = 0; i < n_; ++i) {
        dP[i] = P_in[i] - P_prev_[i];
        dR[i] = R[i] - R_prev_[i];
      }

      // Store (dP, dR) pair.
      double dr_dot_dr = 0.0;
      for (std::size_t i = 0; i < n_; ++i) dr_dot_dr += dR[i] * dR[i];

      if (dr_dot_dr > 1e-30) {
        if (static_cast<int>(dP_list_.size()) >= cfg.max_history) {
          dP_list_.erase(dP_list_.begin());
          dR_list_.erase(dR_list_.begin());
          rho_list_.erase(rho_list_.begin());
        }
        dP_list_.push_back(dP);
        dR_list_.push_back(dR);
        rho_list_.push_back(1.0 / dr_dot_dr);
      }

      // Broyden's method (second/bad Broyden): direct rank-1 update of
      // the inverse Jacobian approximation B ≈ J^{-1}.
      //   B_{k+1} = B_k + (dP - B_k dR) dR^T / (dR^T dR)
      //   P_next = P_in + alpha * B_k * R
      //
      // For the first update, B_0 = alpha * I.
      // We store B as a dense n×n matrix (feasible for SCF density vectors
      // that are typically small: n_basis^2 or grid_size).

      // Initialize B on first Broyden step.
      if (B_.empty()) {
        B_.resize(n_ * n_, 0.0);
        for (std::size_t i = 0; i < n_; ++i)
          B_[i * n_ + i] = cfg.alpha;
      }

      // Update B: B += (dP - B*dR) * dR^T / (dR^T dR)
      std::vector<double> BdR(n_, 0.0);
      for (std::size_t i = 0; i < n_; ++i)
        for (std::size_t j = 0; j < n_; ++j)
          BdR[i] += B_[i * n_ + j] * dR[j];

      double dr_norm_sq = 0.0;
      for (std::size_t i = 0; i < n_; ++i) dr_norm_sq += dR[i] * dR[i];

      if (dr_norm_sq > 1e-30) {
        for (std::size_t i = 0; i < n_; ++i) {
          double coeff = (dP[i] - BdR[i]) / dr_norm_sq;
          for (std::size_t j = 0; j < n_; ++j)
            B_[i * n_ + j] += coeff * dR[j];
        }
      }

      // Compute P_next = P_in - B * R (Newton step to zero the residual).
      bool q_ok = true;
      for (std::size_t i = 0; i < n_; ++i) {
        double val = 0.0;
        for (std::size_t j = 0; j < n_; ++j)
          val += B_[i * n_ + j] * R[j];
        if (!std::isfinite(val)) { q_ok = false; break; }
        P_next[i] = P_in[i] - val;
      }

      if (!q_ok) {
        // Fall back to simple mixing.
        for (std::size_t i = 0; i < n_; ++i)
          P_next[i] = P_in[i] + cfg.alpha * (P_out[i] - P_in[i]);
        // Reset B.
        B_.assign(n_ * n_, 0.0);
        for (std::size_t i = 0; i < n_; ++i)
          B_[i * n_ + i] = cfg.alpha;
      }
    }

    // Store current state for next iteration.
    P_prev_ = P_in;
    R_prev_ = R;

    return P_next;
  }

  // Check convergence: residual norm < tolerance.
  double ResidualNorm(const std::vector<double>& P_in,
                       const std::vector<double>& P_out) const {
    double norm = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
      double d = P_out[i] - P_in[i];
      norm += d * d;
    }
    return std::sqrt(norm);
  }

  // Run a full SCF loop with Broyden mixing.
  //   P_init:  initial density
  //   scf_fn:  function that takes input density and returns output density
  //   cfg:     Broyden configuration
  // Returns: converged density and iteration count.
  struct Result {
    std::vector<double> P;
    int n_iterations = 0;
    double residual = 0.0;
    bool converged = false;
  };

  Result Run(const std::vector<double>& P_init,
              const std::function<std::vector<double>(const std::vector<double>&)>& scf_fn,
              const BroydenConfig& cfg) {
    Init(P_init.size());
    Result result;
    std::vector<double> P_in = P_init;

    for (int iter = 0; iter < cfg.max_iterations; ++iter) {
      auto P_out = scf_fn(P_in);
      double res = ResidualNorm(P_in, P_out);
      result.residual = res;
      result.n_iterations = iter + 1;

      if (res < cfg.tolerance) {
        result.P = P_out;
        result.converged = true;
        return result;
      }

      P_in = Mix(P_in, P_out, cfg);
    }

    result.P = P_in;
    return result;
  }

 private:
  std::size_t n_ = 0;
  bool first_step_ = true;
  std::vector<double> P_prev_;
  std::vector<double> R_prev_;
  std::vector<std::vector<double>> dP_list_;
  std::vector<std::vector<double>> dR_list_;
  std::vector<double> rho_list_;
  std::vector<double> B_;  // Dense inverse Jacobian approximation (n*n).
};

}  // namespace tides::scf
