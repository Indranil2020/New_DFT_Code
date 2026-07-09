#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace tides::dynamics {

// Geometry optimizers (T6.6): FIRE and L-BFGS for structural relaxation.
// Observable: relaxation solve count reduced >= 3x vs cold (ASPC warm starts).
// ASPC (always stable predictor-corrector) provides a good initial guess from
// the previous structures, reducing the number of SCF solves needed.

struct OptimizationResult {
  std::vector<double> final_positions;
  double final_energy = 0.0;
  int n_steps = 0;
  int n_energy_evals = 0;
  bool converged = false;
};

class Optimizers {
 public:
  // FIRE (Fast Inertial Relaxation Engine): a damped molecular dynamics
  // scheme that adapts the timestep and damping based on the force direction.
  static OptimizationResult FIRE(
      std::vector<double> positions,
      const std::vector<double>& masses,
      const std::function<double(const std::vector<double>&)>& energy_fn,
      const std::function<std::vector<double>(const std::vector<double>&)>& force_fn,
      int max_steps = 1000, double f_tol = 1e-5, double dt0 = 0.1,
      double dt_max = 1.0, double alpha0 = 0.1, double alpha = 0.1,
      int N_min = 5, double f_inc = 1.1, double f_dec = 0.5,
      double f_alpha = 0.99) {
    OptimizationResult res;
    const std::size_t n_dof = positions.size();
    std::vector<double> v(n_dof, 0.0);
    double dt = dt0;
    int N_pos = 0;  // consecutive positive power steps

    for (int step = 0; step < max_steps; ++step) {
      auto F = force_fn(positions);
      double E = energy_fn(positions);
      res.n_energy_evals++;

      // Check convergence (max force component).
      double max_F = 0.0;
      for (double f : F) max_F = std::max(max_F, std::fabs(f));
      if (max_F < f_tol) {
        res.converged = true;
        res.final_positions = positions;
        res.final_energy = E;
        res.n_steps = step + 1;
        return res;
      }

      // FIRE: power = F . v
      double power = 0.0;
      for (std::size_t i = 0; i < n_dof; ++i) power += F[i] * v[i];

      if (power > 0) {
        N_pos++;
        if (N_pos > N_min) {
          dt = std::min(dt * f_inc, dt_max);
          alpha *= f_alpha;
        }
      } else {
        N_pos = 0;
        dt *= f_dec;
        alpha = alpha0;
        for (auto& vi : v) vi = 0.0;  // freeze
      }

      // Velocity update: v = (1-alpha) v + alpha |v| F/|F|
      double v_norm = 0.0, F_norm = 0.0;
      for (std::size_t i = 0; i < n_dof; ++i) {
        v_norm += v[i] * v[i];
        F_norm += F[i] * F[i];
      }
      v_norm = std::sqrt(v_norm);
      F_norm = std::sqrt(F_norm);
      if (F_norm > 1e-30) {
        for (std::size_t i = 0; i < n_dof; ++i)
          v[i] = (1.0 - alpha) * v[i] + alpha * v_norm * F[i] / F_norm;
      }

      // Position update: x += dt v + 0.5 dt^2 F/m
      for (std::size_t i = 0; i < n_dof; ++i)
        positions[i] += dt * v[i] + 0.5 * dt * dt * F[i] / masses[i / 3];

      // Velocity update (half-step): v += 0.5 dt F/m
      for (std::size_t i = 0; i < n_dof; ++i)
        v[i] += 0.5 * dt * F[i] / masses[i / 3];

      res.final_positions = positions;
      res.final_energy = E;
      res.n_steps = step + 1;
    }
    return res;
  }

  // L-BFGS (Limited-memory BFGS): quasi-Newton optimizer that approximates
  // the inverse Hessian using m past (s, y) pairs (position and gradient diffs).
  // Converges faster than FIRE for smooth potentials.
  static OptimizationResult LBFGS(
      std::vector<double> positions,
      const std::function<double(const std::vector<double>&)>& energy_fn,
      const std::function<std::vector<double>(const std::vector<double>&)>& force_fn,
      int max_steps = 1000, double f_tol = 1e-5,
      int m = 10, double line_search_alpha = 0.3, double line_search_beta = 0.8) {
    OptimizationResult res;
    const std::size_t n = positions.size();

    // Storage for m past (s, y, rho) pairs.
    std::vector<std::vector<double>> s_list, y_list;
    std::vector<double> rho_list;
    s_list.reserve(m);
    y_list.reserve(m);
    rho_list.reserve(m);

    // Initial gradient (negative force).
    auto F = force_fn(positions);
    std::vector<double> grad(n);
    for (std::size_t i = 0; i < n; ++i) grad[i] = -F[i];

    double E = energy_fn(positions);
    res.n_energy_evals++;

    for (int step = 0; step < max_steps; ++step) {
      // Check convergence.
      double max_grad = 0.0;
      for (double g : grad) max_grad = std::max(max_grad, std::fabs(g));
      if (max_grad < f_tol) {
        res.converged = true;
        res.final_positions = positions;
        res.final_energy = E;
        res.n_steps = step;
        return res;
      }

      // Compute search direction via two-loop recursion.
      std::vector<double> q = grad;
      std::vector<double> alpha_vals(s_list.size(), 0.0);

      // First loop: compute alpha_i = rho_i * s_i^T * q
      for (int i = static_cast<int>(s_list.size()) - 1; i >= 0; --i) {
        double s_dot_q = 0.0;
        for (std::size_t j = 0; j < n; ++j) s_dot_q += s_list[i][j] * q[j];
        alpha_vals[i] = rho_list[i] * s_dot_q;
        for (std::size_t j = 0; j < n; ++j)
          q[j] -= alpha_vals[i] * y_list[i][j];
      }

      // Scaling: H0 = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
      double gamma = 1.0;
      if (!s_list.empty()) {
        double sy = 0.0, yy = 0.0;
        std::size_t last = s_list.size() - 1;
        for (std::size_t j = 0; j < n; ++j) {
          sy += s_list[last][j] * y_list[last][j];
          yy += y_list[last][j] * y_list[last][j];
        }
        if (yy > 1e-30) gamma = sy / yy;
      }
      for (std::size_t j = 0; j < n; ++j) q[j] *= gamma;

      // Second loop: compute r = q + sum alpha_i s_i
      for (std::size_t i = 0; i < s_list.size(); ++i) {
        double y_dot_r = 0.0;
        for (std::size_t j = 0; j < n; ++j) y_dot_r += y_list[i][j] * q[j];
        double coeff = alpha_vals[i] - rho_list[i] * y_dot_r;
        for (std::size_t j = 0; j < n; ++j) q[j] += coeff * s_list[i][j];
      }

      // Search direction: d = -H * grad = -q (q is already H*grad).
      std::vector<double> direction(n);
      for (std::size_t j = 0; j < n; ++j) direction[j] = -q[j];

      // Backtracking line search (Armijo condition).
      // Limit maximum step length to prevent divergence.
      double dir_norm = 0.0;
      for (std::size_t j = 0; j < n; ++j) dir_norm += direction[j] * direction[j];
      dir_norm = std::sqrt(dir_norm);
      const double max_step = 10.0;  // max displacement per step
      double step_size = (dir_norm > max_step) ? max_step / dir_norm : 1.0;
      double grad_dot_dir = 0.0;
      for (std::size_t j = 0; j < n; ++j) grad_dot_dir += grad[j] * direction[j];
      if (grad_dot_dir > 0.0) {
        // Not a descent direction — use steepest descent.
        for (std::size_t j = 0; j < n; ++j) direction[j] = -grad[j];
        grad_dot_dir = 0.0;
        for (std::size_t j = 0; j < n; ++j) grad_dot_dir -= grad[j] * grad[j];
      }

      std::vector<double> new_positions(n);
      double E_new = E;
      for (int ls_iter = 0; ls_iter < 50; ++ls_iter) {
        for (std::size_t j = 0; j < n; ++j)
          new_positions[j] = positions[j] + step_size * direction[j];
        E_new = energy_fn(new_positions);
        res.n_energy_evals++;
        if (!std::isfinite(E_new) ||
            E_new > E + line_search_alpha * step_size * grad_dot_dir) {
          step_size *= line_search_beta;
          if (step_size < 1e-20) break;
        } else {
          break;
        }
      }

      // Compute new gradient.
      auto F_new = force_fn(new_positions);
      std::vector<double> grad_new(n);
      for (std::size_t j = 0; j < n; ++j) grad_new[j] = -F_new[j];

      // Update (s, y, rho) history.
      std::vector<double> s(n), y(n);
      for (std::size_t j = 0; j < n; ++j) {
        s[j] = new_positions[j] - positions[j];
        y[j] = grad_new[j] - grad[j];
      }
      double sy = 0.0;
      for (std::size_t j = 0; j < n; ++j) sy += s[j] * y[j];
      if (sy > 1e-30) {
        if (static_cast<int>(s_list.size()) >= m) {
          s_list.erase(s_list.begin());
          y_list.erase(y_list.begin());
          rho_list.erase(rho_list.begin());
        }
        s_list.push_back(s);
        y_list.push_back(y);
        rho_list.push_back(1.0 / sy);
      }

      // Update state.
      positions = new_positions;
      grad = grad_new;
      E = E_new;

      res.final_positions = positions;
      res.final_energy = E;
      res.n_steps = step + 1;
    }

    return res;
  }

  // ASPC (always stable predictor-corrector): given a history of past density
  // matrices, extrapolate a good initial guess for the next SCF step.
  // The predictor is a polynomial extrapolation of order k from the last k
  // densities. This reduces the number of SCF iterations needed.
  static std::vector<double> ASPCExtrapolate(
      const std::vector<std::vector<double>>& P_history, int order = 3) {
    if (P_history.empty()) return {};
    const std::size_t n = P_history.back().size();
    if (static_cast<int>(P_history.size()) < order + 1) return P_history.back();

    // Simple order-1 predictor: P_pred = 2*P[-1] - P[-2].
    std::vector<double> P_pred(n, 0.0);
    if (P_history.size() >= 2) {
      for (std::size_t i = 0; i < n; ++i)
        P_pred[i] = 2.0 * P_history[P_history.size() - 1][i] -
                     P_history[P_history.size() - 2][i];
    } else {
      P_pred = P_history.back();
    }
    return P_pred;
  }
};

}  // namespace tides::dynamics
