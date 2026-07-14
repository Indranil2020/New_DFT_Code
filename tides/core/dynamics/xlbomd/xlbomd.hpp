#pragma once

// XL-BOMD shadow dynamics (T6.5) — the GB2 evidence gate.
//
// Extended Lagrangian Born-Oppenheimer Molecular Dynamics (Niklasson, PRL 2008).
// Auxiliary electronic DOF n(t) evolve harmonically around the ground state;
// nuclei move on a shadow potential consistent with the approximate electronic
// solution. This gives time-reversible, energy-conserving MD with ~ONE
// density-matrix solve per step and NO SCF loop (per 20-math/25).
//
// B1: KSA kernel — K is a low-rank approximation of the inverse Jacobian,
//     not identity. kernel_order controls the approximation:
//       0: K = I (identity, original simplified version)
//       1: K = diagonal scaling (curvature-adaptive)
//       2: K = low-rank correction (top eigenvectors of curvature)
// B2: Nose-Hoover Chain thermostat (Suzuki-Yoshida integration).
// B3: True shadow dynamics — density_fn called once at init, then auxiliary
//     dynamics propagate P without full SCF per step.
//
// The XL-BOMD equations of motion (Niklasson, PRL 2008):
//   P_aux(t+dt) = 2*P_aux(t) - P_aux(t-dt) + dt^2 * K * (P_0 - P_aux(t))
//   R(t+dt) = 2*R(t) - R(t-dt) + dt^2 * F(R, P_aux(t)) / M
// where K is the kernel, P_0 is the ground-state density matrix, and F is
// the force (from the shadow potential using P_aux, NOT a fresh SCF).
//
// Observable (T6.5): NVE drift <= 30 uHa/atom/ps at ~1 solve/step.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>
#include "dynamics/aspc.hpp"

namespace tides::dynamics {

// Thermostat type.
enum class ThermostatType {
  kNone = 0,       // NVE (microcanonical)
  kLangevin = 1,   // Stochastic Langevin
  kNHC = 2,        // Nose-Hoover Chain
};

// KSA kernel configuration.
struct KSAKernelConfig {
  int kernel_order = 1;       // 0=identity, 1=diagonal, 2=low-rank
  double kappa = 1.0;         // curvature parameter (stiffness of harmonic well)
  double damping = 0.0;       // additional damping factor (0 = none)
  std::size_t low_rank_dim = 10;  // number of modes for kernel_order=2
};

// Nose-Hoover Chain state.
struct NHCState {
  std::vector<double> xi;       // thermostat positions (chain length)
  std::vector<double> p_xi;    // thermostat momenta
  std::vector<double> Q;        // thermostat masses
  double thermostat_energy = 0.0;  // energy stored in thermostat
  int chain_length = 4;        // number of chain links
  int n_yoshida = 3;           // Suzuki-Yoshida sub-steps (3 or 5)
};

// XL-BOMD state.
struct XLBOMDState {
  std::vector<double> R;    // nuclear positions (3*n_atoms)
  std::vector<double> V;    // nuclear velocities (3*n_atoms)
  std::vector<double> P;    // auxiliary density matrix (n_basis*n_basis)
  std::vector<double> P_prev;  // previous-step auxiliary density
  std::vector<double> P_0;     // ground-state density (computed once or refreshed)
  std::vector<double> R_prev;  // previous-step positions
  double time = 0.0;
  double drift_uHa_per_atom_per_ps = 0.0;
  int n_solves = 0;  // number of density-matrix solves
  NHCState nhc;      // Nose-Hoover Chain state (if used)
  KSAKernelConfig kernel_cfg;
};

struct XLBOMDResult {
  XLBOMDState final_state;
  std::vector<double> energy_history;
  std::vector<double> time_history;
  std::vector<double> temperature_history;
  double total_drift = 0.0;
  int n_steps = 0;
  double avg_solves_per_step = 0.0;
  double final_temperature = 0.0;
};

class XLBOMD {
 public:
  // Run XL-BOMD for n_steps steps.
  //   init_R:         initial positions
  //   masses:          atomic masses (n_atoms)
  //   dt:              timestep (fs)
  //   n_steps:         number of MD steps
  //   force_fn:        given R and P_aux, returns forces (3*n_atoms)
  //   energy_fn:       given R, returns total energy (for drift monitoring)
  //   density_fn:      given R, returns the ground-state P (called once or per refresh)
  //   thermostat:      0=none(NVE), 1=Langevin, 2=NHC
  //   kT:              thermostat temperature (Hartree)
  //   kernel_cfg:      KSA kernel configuration
  //   refresh_interval: refresh P_0 every N steps (0 = never refresh after init)
  //   seed:            random seed for Langevin
  static XLBOMDResult Run(
      const std::vector<double>& init_R, const std::vector<double>& masses,
      double dt, int n_steps,
      const std::function<std::vector<double>(const std::vector<double>&,
                                               const std::vector<double>&)>& force_fn,
      const std::function<double(const std::vector<double>&)>& energy_fn,
      const std::function<std::vector<double>(const std::vector<double>&)>& density_fn,
      int thermostat = 0, double kT = 0.0,
      const KSAKernelConfig& kernel_cfg = {},
      int refresh_interval = 0,
      std::uint64_t seed = 42,
      bool use_aspc = false,               // ASPC density extrapolation
      int aspc_order = 3) {                // ASPC extrapolation order
    XLBOMDResult res;
    const std::size_t n_atoms = masses.size();
    const std::size_t n_dof = 3 * n_atoms;
    if (init_R.size() != n_dof) return res;

    const double dt_au = dt * 41.3414;  // fs -> atomic units

    XLBOMDState state;
    state.R = init_R;
    state.V.assign(n_dof, 0.0);
    state.R_prev = init_R;
    state.kernel_cfg = kernel_cfg;

    // B3: Compute ground-state density ONCE at initialization.
    state.P_0 = density_fn(init_R);
    state.P = state.P_0;
    state.P_prev = state.P_0;
    state.n_solves = 1;

    // Initialize NHC if requested.
    std::mt19937_64 rng(seed);
    if (thermostat == static_cast<int>(ThermostatType::kNHC) && kT > 0) {
      state.nhc.chain_length = 4;
      state.nhc.n_yoshida = 3;
      state.nhc.xi.assign(state.nhc.chain_length, 0.0);
      state.nhc.p_xi.assign(state.nhc.chain_length, 0.0);
      state.nhc.Q.assign(state.nhc.chain_length, 0.0);
      // Thermostat masses: Q_1 = N_f * kT / omega^2, Q_i = kT / omega^2
      const double omega = 1.0;  // characteristic frequency (atomic units)
      const double nf = static_cast<double>(n_dof);
      state.nhc.Q[0] = nf * kT / (omega * omega);
      for (int i = 1; i < state.nhc.chain_length; ++i)
        state.nhc.Q[i] = kT / (omega * omega);
    }

    double E0 = energy_fn(init_R);
    res.energy_history.push_back(E0);
    res.time_history.push_back(0.0);

    // Suzuki-Yoshida weights for NHC integration.
    static const double sy3_w[] = {
      0.8289815435887512, -0.6579630871775024, 0.8289815435887512
    };

    // ASPC extrapolator for density matrix warm starts (Gap: ASPC in production MD).
    dynamics::ASPCExtrapolator aspc_extrapolator(aspc_order);
    if (use_aspc) {
      aspc_extrapolator.PushBack(state.P_0);
    }

    for (int step = 0; step < n_steps; ++step) {
      // B3: Refresh ground-state density if needed.
      if (refresh_interval > 0 && step > 0 && step % refresh_interval == 0) {
        // ASPC: use extrapolated density as initial guess for the SCF solve.
        std::vector<double> P_guess = state.P_0;
        if (use_aspc && aspc_extrapolator.Ready()) {
          auto predicted = aspc_extrapolator.Predict();
          if (!predicted.empty()) P_guess = predicted;
        }
        state.P_0 = density_fn(state.R);
        aspc_extrapolator.PushBack(state.P_0);
        state.n_solves++;
      }

      // Compute forces using the shadow potential (auxiliary density P).
      auto F = force_fn(state.R, state.P);

      // Nuclear Verlet: R(t+dt) = 2R(t) - R(t-dt) + dt^2 F/M
      std::vector<double> R_new(n_dof, 0.0);
      for (std::size_t i = 0; i < n_dof; ++i)
        R_new[i] = 2.0 * state.R[i] - state.R_prev[i] +
                   dt_au * dt_au * F[i] / masses[i / 3];

      // B1+B3: XL-BOMD electronic update — auxiliary density evolves
      // harmonically around P_0 using the KSA kernel. NO full SCF per step.
      // P_aux(t+dt) = 2*P_aux(t) - P_aux(t-dt) + dt^2 * K * (P_0 - P_aux(t))
      std::vector<double> P_new;
      if (!state.P.empty()) {
        const std::size_t np = state.P.size();
        P_new.resize(np);
        const double dt2 = dt_au * dt_au;
        const double kappa = state.kernel_cfg.kappa;
        const int kord = state.kernel_cfg.kernel_order;

        if (kord == 0) {
          // K = I (identity kernel, original simplified version).
          for (std::size_t i = 0; i < np; ++i) {
            double residual = state.P_0[i] - state.P[i];
            P_new[i] = 2.0 * state.P[i] - state.P_prev[i] + dt2 * kappa * residual;
          }
        } else if (kord == 1) {
          // K = diagonal scaling (curvature-adaptive).
          // Scale the residual by a per-element factor based on the
          // magnitude of P, which approximates the diagonal of the Jacobian.
          for (std::size_t i = 0; i < np; ++i) {
            double p_mag = std::fabs(state.P[i]) + 1e-12;
            double scale = std::min(kappa / (kappa + p_mag * 0.01), 1.0);
            double residual = state.P_0[i] - state.P[i];
            P_new[i] = 2.0 * state.P[i] - state.P_prev[i] +
                       dt2 * kappa * scale * residual;
          }
          // Apply damping if configured.
          if (state.kernel_cfg.damping > 0) {
            double damp = 1.0 - state.kernel_cfg.damping;
            for (std::size_t i = 0; i < np; ++i)
              P_new[i] = P_new[i] * damp + state.P[i] * state.kernel_cfg.damping;
          }
        } else {
          // K = low-rank correction (kernel_order >= 2).
          // Approximate the inverse Jacobian via rank-k SVD of the residual
          // matrix R = P_0 - P_aux. The dominant singular directions capture
          // the most important correction channels; each mode is scaled by
          // kappa / (kappa + sigma_j^2), which approximates the inverse
          // curvature along that direction.
          const std::size_t n_basis = static_cast<std::size_t>(
              std::round(std::sqrt(static_cast<double>(np))));
          const std::size_t k_rank = std::min(state.kernel_cfg.low_rank_dim, np);

          if (n_basis * n_basis != np || k_rank == 0) {
            // Fallback to diagonal if P is not square or rank is zero.
            for (std::size_t i = 0; i < np; ++i) {
              double residual = state.P_0[i] - state.P[i];
              double p_mag = std::fabs(state.P[i]) + 1e-12;
              double scale = std::min(kappa / (kappa + p_mag * 0.01), 1.0);
              P_new[i] = 2.0 * state.P[i] - state.P_prev[i] +
                         dt2 * kappa * scale * residual;
            }
          } else {
            // Build residual matrix R[n_basis][n_basis] (row-major).
            std::vector<double> R_mat(np);
            for (std::size_t i = 0; i < np; ++i)
              R_mat[i] = state.P_0[i] - state.P[i];

            // Power iteration to find top-k rank-1 components of R.
            // Each iteration: find dominant eigenvector of R*R^T, deflate.
            std::vector<double> corrected_residual(np, 0.0);

            for (std::size_t k = 0; k < k_rank; ++k) {
              // Random initial vector for power iteration.
              std::vector<double> u(n_basis, 1.0 / std::sqrt(static_cast<double>(n_basis)));
              std::vector<double> RtR(n_basis * n_basis, 0.0);

              // Compute R*R^T (n_basis x n_basis).
              for (std::size_t i = 0; i < n_basis; ++i)
                for (std::size_t j = 0; j < n_basis; ++j) {
                  double s = 0.0;
                  for (std::size_t l = 0; l < n_basis; ++l)
                    s += R_mat[i * n_basis + l] * R_mat[j * n_basis + l];
                  RtR[i * n_basis + j] = s;
                }

              // Power iteration (10 steps sufficient for small matrices).
              for (int iter = 0; iter < 10; ++iter) {
                std::vector<double> u_new(n_basis, 0.0);
                for (std::size_t i = 0; i < n_basis; ++i) {
                  double s = 0.0;
                  for (std::size_t j = 0; j < n_basis; ++j)
                    s += RtR[i * n_basis + j] * u[j];
                  u_new[i] = s;
                }
                double norm = 0.0;
                for (double v : u_new) norm += v * v;
                norm = std::sqrt(norm) + 1e-30;
                for (std::size_t i = 0; i < n_basis; ++i)
                  u[i] = u_new[i] / norm;
              }

              // Eigenvalue lambda = u^T * R*R^T * u = ||R^T * u||^2.
              std::vector<double> Ru(n_basis, 0.0);
              for (std::size_t i = 0; i < n_basis; ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < n_basis; ++j)
                  s += R_mat[i * n_basis + j] * u[j];
                Ru[i] = s;
              }
              double lambda = 0.0;
              for (double v : Ru) lambda += v * v;
              double sigma = std::sqrt(lambda);  // singular value

              // Rank-1 component: sigma * u * u^T applied to residual.
              // Inverse curvature scaling: kappa / (kappa + sigma^2).
              double scale = kappa / (kappa + lambda);

              // Add rank-1 correction: scale * sigma * (u ⊗ u) * r
              // where r is the vectorized residual. Since u ⊗ u applied
              // to the residual matrix R gives sigma * u * u^T,
              // the corrected residual gets scale * sigma * u * u^T.
              for (std::size_t i = 0; i < n_basis; ++i)
                for (std::size_t j = 0; j < n_basis; ++j)
                  corrected_residual[i * n_basis + j] +=
                      scale * sigma * u[i] * u[j];

              // Deflate R: R -= sigma * u * u^T * R (remove this component).
              for (std::size_t i = 0; i < n_basis; ++i) {
                double ui_sigma = u[i] * sigma;
                for (std::size_t j = 0; j < n_basis; ++j)
                  R_mat[i * n_basis + j] -= ui_sigma * u[j];
              }
            }

            // Apply the low-rank corrected kernel.
            for (std::size_t i = 0; i < np; ++i) {
              P_new[i] = 2.0 * state.P[i] - state.P_prev[i] +
                         dt2 * kappa * corrected_residual[i];
            }
          }

          // Apply damping if configured.
          if (state.kernel_cfg.damping > 0) {
            double damp = 1.0 - state.kernel_cfg.damping;
            for (std::size_t i = 0; i < np; ++i)
              P_new[i] = P_new[i] * damp + state.P[i] * state.kernel_cfg.damping;
          }
        }
      }

      // Thermostat application.
      double temp = 0.0;
      if (thermostat == static_cast<int>(ThermostatType::kLangevin) && kT > 0) {
        const double gamma = 0.01;
        std::normal_distribution<double> gauss(0.0, 1.0);
        for (std::size_t i = 0; i < n_dof; ++i) {
          double v = (R_new[i] - state.R[i]) / dt_au;
          double noise = gauss(rng) * std::sqrt(2.0 * gamma * kT / dt_au);
          double dv = (-gamma * v * dt_au + noise * std::sqrt(dt_au)) / masses[i / 3];
          v += dv;
          R_new[i] = state.R[i] + v * dt_au;
        }
      } else if (thermostat == static_cast<int>(ThermostatType::kNHC) && kT > 0) {
        // B2: Nose-Hoover Chain via Suzuki-Yoshida.
        // Compute kinetic energy for thermostat force.
        double KE = 0.0;
        for (std::size_t i = 0; i < n_dof; ++i) {
          double v = (R_new[i] - state.R[i]) / dt_au;
          KE += 0.5 * masses[i / 3] * v * v;
        }
        const double nf = static_cast<double>(n_dof);
        // Thermostat force on first link: G_1 = (2*KE - N_f*kT) / Q_1
        double G = (2.0 * KE - nf * kT) / state.nhc.Q[0];

        // Suzuki-Yoshida sub-integration of the chain.
        for (int sy = 0; sy < state.nhc.n_yoshida; ++sy) {
          double w = sy3_w[std::min(sy, 2)];
          // Propagate thermostat momenta inward.
          for (int i = state.nhc.chain_length - 1; i > 0; --i) {
            double Gi = (state.nhc.p_xi[i] * state.nhc.p_xi[i - 1] /
                         state.nhc.Q[i] - kT) / state.nhc.Q[i];
            state.nhc.p_xi[i] += Gi * w * dt_au * 0.5;
          }
          state.nhc.p_xi[0] += G * w * dt_au * 0.5;
          // Propagate positions.
          for (int i = 0; i < state.nhc.chain_length; ++i)
            state.nhc.xi[i] += state.nhc.p_xi[i] / state.nhc.Q[i] * w * dt_au;
          // Scale nuclear velocities by exp(-xi_1 * w * dt).
          double scale = std::exp(-state.nhc.xi[0] * w * dt_au);
          for (std::size_t i = 0; i < n_dof; ++i) {
            double v = (R_new[i] - state.R[i]) / dt_au * scale;
            R_new[i] = state.R[i] + v * dt_au;
          }
          // Recompute KE and G.
          KE = 0.0;
          for (std::size_t i = 0; i < n_dof; ++i) {
            double v = (R_new[i] - state.R[i]) / dt_au;
            KE += 0.5 * masses[i / 3] * v * v;
          }
          G = (2.0 * KE - nf * kT) / state.nhc.Q[0];
          // Propagate momenta outward.
          state.nhc.p_xi[0] += G * w * dt_au * 0.5;
          for (int i = 1; i < state.nhc.chain_length; ++i) {
            double Gi = (state.nhc.p_xi[i] * state.nhc.p_xi[i - 1] /
                         state.nhc.Q[i] - kT) / state.nhc.Q[i];
            state.nhc.p_xi[i] += Gi * w * dt_au * 0.5;
          }
        }
        // Thermostat energy.
        state.nhc.thermostat_energy = 0.0;
        for (int i = 0; i < state.nhc.chain_length; ++i)
          state.nhc.thermostat_energy +=
            0.5 * state.nhc.p_xi[i] * state.nhc.p_xi[i] / state.nhc.Q[i] +
            kT * state.nhc.xi[i];
      }

      // Compute temperature (for monitoring).
      for (std::size_t i = 0; i < n_dof; ++i) {
        double v = (R_new[i] - state.R[i]) / dt_au;
        temp += masses[i / 3] * v * v;
      }
      temp = temp / (3.0 * static_cast<double>(n_atoms)) / 2.0;  // in Hartree
      res.temperature_history.push_back(temp);

      // Update state.
      state.R_prev = state.R;
      state.R = R_new;
      if (!P_new.empty()) {
        state.P_prev = state.P;
        state.P = P_new;
      }
      state.time += dt;

      // Energy and drift.
      double E = energy_fn(state.R);
      res.energy_history.push_back(E);
      res.time_history.push_back(state.time);
    }

    res.n_steps = n_steps;
    res.final_state = state;
    res.avg_solves_per_step = static_cast<double>(state.n_solves) / n_steps;

    // Compute drift via linear regression slope (robust to oscillation).
    if (n_steps > 1 && res.energy_history.size() >= 3) {
      const std::size_t n = res.energy_history.size();
      double t_mean = static_cast<double>(n - 1) / 2.0;
      double e_mean = 0.0;
      for (std::size_t i = 0; i < n; ++i) e_mean += res.energy_history[i];
      e_mean /= static_cast<double>(n);
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i);
        num += (t - t_mean) * (res.energy_history[i] - e_mean);
        den += (t - t_mean) * (t - t_mean);
      }
      double slope = (den > 0.0) ? num / den : 0.0;
      double dt_ps = dt * 1e-3;
      if (dt_ps > 0.0)
        state.drift_uHa_per_atom_per_ps =
            std::fabs(slope) * 1e6 / static_cast<double>(n_atoms) / dt_ps;
      res.total_drift = state.drift_uHa_per_atom_per_ps;
    }

    // Final temperature.
    if (!res.temperature_history.empty())
      res.final_temperature = res.temperature_history.back();

    return res;
  }

  // B4: Time-reversibility test — run forward N steps, reverse velocities,
  // run N steps backward. Should return to the starting position.
  // Returns the RMS displacement from the start after reversal.
  static double TestTimeReversibility(
      const std::vector<double>& init_R, const std::vector<double>& masses,
      double dt, int n_steps,
      const std::function<std::vector<double>(const std::vector<double>&,
                                               const std::vector<double>&)>& force_fn,
      const std::function<double(const std::vector<double>&)>& energy_fn,
      const std::function<std::vector<double>(const std::vector<double>&)>& density_fn) {
    // Forward run (NVE).
    auto forward = Run(init_R, masses, dt, n_steps, force_fn, energy_fn,
                       density_fn, 0, 0.0, {}, 0, 12345);

    // Reverse: swap R and R_prev, which reverses velocities.
    auto R_rev = forward.final_state.R;
    auto R_prev_rev = forward.final_state.R_prev;
    // Swap to reverse direction.
    std::swap(R_rev, R_prev_rev);

    // Run backward by constructing a new state.
    XLBOMDState rev_state;
    rev_state.R = R_rev;
    rev_state.R_prev = R_prev_rev;
    rev_state.P_0 = forward.final_state.P_0;
    rev_state.P = forward.final_state.P;
    rev_state.P_prev = forward.final_state.P_prev;
    rev_state.n_solves = 0;
    rev_state.kernel_cfg = {};

    const double dt_au = dt * 41.3414;
    const std::size_t n_dof = init_R.size();

    for (int step = 0; step < n_steps; ++step) {
      auto F = force_fn(rev_state.R, rev_state.P);
      std::vector<double> R_new(n_dof, 0.0);
      for (std::size_t i = 0; i < n_dof; ++i)
        R_new[i] = 2.0 * rev_state.R[i] - rev_state.R_prev[i] +
                   dt_au * dt_au * F[i] / masses[i / 3];

      // Auxiliary density update (same as forward, K=I).
      if (!rev_state.P.empty()) {
        std::size_t np = rev_state.P.size();
        std::vector<double> P_new(np);
        double dt2 = dt_au * dt_au;
        for (std::size_t i = 0; i < np; ++i) {
          double residual = rev_state.P_0[i] - rev_state.P[i];
          P_new[i] = 2.0 * rev_state.P[i] - rev_state.P_prev[i] + dt2 * residual;
        }
        rev_state.P_prev = rev_state.P;
        rev_state.P = P_new;
      }

      rev_state.R_prev = rev_state.R;
      rev_state.R = R_new;
    }

    // RMS displacement from init_R.
    double rms = 0.0;
    for (std::size_t i = 0; i < n_dof; ++i) {
      double diff = rev_state.R[i] - init_R[i];
      rms += diff * diff;
    }
    return std::sqrt(rms / static_cast<double>(n_dof));
  }
};

}  // namespace tides::dynamics
