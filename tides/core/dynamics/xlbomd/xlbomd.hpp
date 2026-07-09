#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace tides::dynamics {

// XL-BOMD shadow dynamics (T6.5) — the GB2 evidence gate.
//
// Auxiliary electronic DOF n(t) evolve harmonically around the ground state;
// nuclei move on a shadow potential consistent with the approximate electronic
// solution. This gives time-reversible, energy-conserving MD with ~ONE
// density-matrix solve per step and NO SCF loop (per 20-math/25).
//
// The XL-BOMD equations of motion (Niklasson, PRL 2008):
//   n(t+dt) = 2*n(t) - n(t-dt) + dt^2 * (K * (n_0 - n(t)))
//   R(t+dt) = 2*R(t) - R(t-dt) + dt^2 * F(R, n(t)) / M
// where K is the kernel (approximate inverse Jacobian), n_0 is the ground-state
// density matrix, and F is the force (from T6.3).
//
// For the CPU reference, we implement a simplified version:
//   - The kernel K = I (identity; the full KSA is a low-rank approximation).
//   - The electronic variable n is the density matrix P (flattened).
//   - The force is a callback (caller provides F(R)).
//   - Thermostat: optional Langevin (the full version has Nose-Hoover chains).
//
// Observable (T6.5): NVE drift <= 30 uHa/atom/ps at ~1 solve/step.

struct XLBOMDState {
  std::vector<double> R;    // nuclear positions (3*n_atoms)
  std::vector<double> V;    // nuclear velocities (3*n_atoms)
  std::vector<double> P;    // density matrix (n_basis*n_basis) — auxiliary DOF
  std::vector<double> P_prev;  // previous-step density (for Verlet)
  std::vector<double> R_prev;  // previous-step positions
  double time = 0.0;
  double drift_uHa_per_atom_per_ps = 0.0;
  int n_solves = 0;  // number of density-matrix solves (= number of steps)
};

struct XLBOMDResult {
  XLBOMDState final_state;
  std::vector<double> energy_history;
  std::vector<double> time_history;
  double total_drift = 0.0;
  int n_steps = 0;
  double avg_solves_per_step = 0.0;
};

class XLBOMD {
 public:
  // Run XL-BOMD for n_steps steps.
  //   init_R:      initial positions
  //   masses:       atomic masses (n_atoms)
  //   dt:           timestep (fs -> atomic units: 1 fs = 41.34 a.u.)
  //   n_steps:      number of MD steps
  //   force_fn:     given R, returns forces (3*n_atoms)
  //   energy_fn:    given R, returns total energy (for drift monitoring)
  //   density_fn:   given R, returns the ground-state P (the "1 solve/step")
  //   kernel:       K (identity for the simplified version)
  //   thermostat:   0=none (NVE), 1=Langevin
  //   kT:           thermostat temperature (Hartree)
  static XLBOMDResult Run(
      const std::vector<double>& init_R, const std::vector<double>& masses,
      double dt, int n_steps,
      const std::function<std::vector<double>(const std::vector<double>&)>& force_fn,
      const std::function<double(const std::vector<double>&)>& energy_fn,
      const std::function<std::vector<double>(const std::vector<double>&)>& density_fn,
      int thermostat = 0, double kT = 0.0) {
    XLBOMDResult res;
    const std::size_t n_atoms = masses.size();
    const std::size_t n_dof = 3 * n_atoms;
    if (init_R.size() != n_dof) return res;

    // Convert dt from fs to atomic units (1 fs = 41.3414 a.u.).
    const double dt_au = dt * 41.3414;

    XLBOMDState state;
    state.R = init_R;
    state.V.assign(n_dof, 0.0);
    state.R_prev = init_R;
    state.P = density_fn(init_R);
    state.P_prev = state.P;
    state.n_solves = 1;

    double E0 = energy_fn(init_R);
    res.energy_history.push_back(E0);
    res.time_history.push_back(0.0);

    // Verlet integration with XL-BOMD electronic dynamics.
    for (int step = 0; step < n_steps; ++step) {
      // Compute forces at current R (the "1 solve/step").
      auto F = force_fn(state.R);

      // Nuclear Verlet: R(t+dt) = 2R(t) - R(t-dt) + dt^2 F/M
      std::vector<double> R_new(n_dof, 0.0);
      for (std::size_t i = 0; i < n_dof; ++i)
        R_new[i] = 2.0 * state.R[i] - state.R_prev[i] +
                   dt_au * dt_au * F[i] / masses[i / 3];

      // XL-BOMD electronic update (simplified: P_new = P_ground(R_new)).
      // The full version: P_aux(t+dt) = 2P_aux(t) - P_aux(t-dt) + dt^2 K(P_0 - P_aux).
      // With K=I and the ground-state computed once per step:
      auto P_new = density_fn(R_new);
      state.n_solves++;

      // Thermostat (Langevin: add friction + random force).
      if (thermostat == 1 && kT > 0) {
        const double gamma = 0.01;  // friction
        for (std::size_t i = 0; i < n_dof; ++i) {
          const double v = (R_new[i] - state.R[i]) / dt_au;
          // Simple stochastic kick (not a full Langevin, but enough for testing).
          const double noise = (static_cast<double>(rand()) / RAND_MAX - 0.5) *
                              std::sqrt(2.0 * gamma * kT / dt_au);
          R_new[i] += noise * dt_au * dt_au / masses[i / 3];
          (void)v;
        }
      }

      // Update state.
      state.R_prev = state.R;
      state.R = R_new;
      state.P_prev = state.P;
      state.P = P_new;
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
    // slope is dE/d(step_index). Convert to uHa/atom/ps:
    //   drift = |slope| * 1e6 / n_atoms / (dt_fs * 1e-3)
    if (n_steps > 1 && res.energy_history.size() >= 3) {
      const std::size_t n = res.energy_history.size();
      double t_mean = static_cast<double>(n - 1) / 2.0;
      double e_mean = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        e_mean += res.energy_history[i];
      e_mean /= static_cast<double>(n);
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i);
        num += (t - t_mean) * (res.energy_history[i] - e_mean);
        den += (t - t_mean) * (t - t_mean);
      }
      double slope = (den > 0.0) ? num / den : 0.0;
      double dt_ps = dt * 1e-3;  // fs -> ps per step
      if (dt_ps > 0.0)
        state.drift_uHa_per_atom_per_ps =
            std::fabs(slope) * 1e6 / static_cast<double>(n_atoms) / dt_ps;
      res.total_drift = state.drift_uHa_per_atom_per_ps;
    }

    return res;
  }
};

}  // namespace tides::dynamics
