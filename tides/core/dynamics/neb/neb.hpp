#pragma once

// T6.7: NEB (Nudged Elastic Band) with climbing image.
//
// NEB finds the minimum energy path (MEP) between two configurations
// (reactant and product) by connecting them with a chain of images
// connected by spring forces. The "nudging" projects out the perpendicular
// component of the true force and the parallel component of the spring force,
// preventing corner-cutting and sliding.
//
// The climbing image (CI) variant identifies the highest-energy image and
// inverts the force along the tangent, driving it to the saddle point.
//
// Reference: Henkelman & Jonsson, J. Chem. Phys. 113, 9901 (2000).
//
// Observable: the saddle point energy is found within tolerance, and the
// path converges to the MEP.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "common/status.hpp"

namespace tides::dynamics {

struct NEBResult {
  std::vector<std::vector<double>> images;    // final image positions
  std::vector<double> energies;               // energy at each image
  int n_steps = 0;
  int n_energy_evals = 0;
  bool converged = false;
  double max_force = 0.0;                     // max force on any image
  std::size_t climbing_image = 0;             // index of CI image
};

class NEB {
 public:
  // Run NEB with climbing image.
  //
  // @param initial_images  Vector of image positions (each is 3*n_atoms flat).
  //                        images[0] = reactant, images[-1] = product.
  //                        Intermediate images are linearly interpolated if
  //                        only endpoints are provided.
  // @param energy_fn       Energy function: takes flat positions, returns E.
  // @param force_fn        Force function: takes flat positions, returns flat forces.
  // @param k_spring        Spring constant (default 0.1 eV/Ang^2 ~ 0.0037 Ha/Bohr^2).
  // @param max_steps       Maximum optimization steps.
  // @param f_tol           Force convergence tolerance (Ha/Bohr).
  // @param dt              Time step for the damped dynamics optimizer.
  // @param climb           Whether to enable climbing image (default true).
  // @param climb_start     Step after which to enable CI (allow path to settle).
  static NEBResult Run(
      const std::vector<std::vector<double>>& initial_images,
      const std::function<double(const std::vector<double>&)>& energy_fn,
      const std::function<std::vector<double>(const std::vector<double>&)>& force_fn,
      double k_spring = 0.1, int max_steps = 500, double f_tol = 1e-4,
      double dt = 0.1, bool climb = true, int climb_start = 20) {
    NEBResult res;
    const std::size_t n_images = initial_images.size();
    if (n_images < 3) return res;  // need at least reactant, 1 middle, product

    const std::size_t n_dof = initial_images[0].size();

    // Initialize images.
    std::vector<std::vector<double>> images = initial_images;
    // If only endpoints provided, linearly interpolate.
    if (n_images == 2) {
      std::vector<std::vector<double>> interp(7);
      interp[0] = initial_images[0];
      interp[6] = initial_images[1];
      for (std::size_t i = 1; i < 6; ++i) {
        double t = static_cast<double>(i) / 6.0;
        interp[i].resize(n_dof);
        for (std::size_t j = 0; j < n_dof; ++j)
          interp[i][j] = (1.0 - t) * initial_images[0][j] +
                         t * initial_images[1][j];
      }
      images = interp;
    }

    const std::size_t n_img = images.size();
    std::vector<double> energies(n_img, 0.0);
    std::vector<std::vector<double>> velocities(n_img,
        std::vector<double>(n_dof, 0.0));

    // Compute initial energies.
    for (std::size_t i = 0; i < n_img; ++i) {
      energies[i] = energy_fn(images[i]);
      res.n_energy_evals++;
    }

    for (int step = 0; step < max_steps; ++step) {
      res.n_steps = step + 1;

      // Compute tangents using the energy-weighted scheme.
      std::vector<std::vector<double>> tangents(n_img,
          std::vector<double>(n_dof, 0.0));
      for (std::size_t i = 1; i < n_img - 1; ++i) {
        // Tangent direction: weighted by energies of neighbors.
        std::vector<double> tau_plus(n_dof), tau_minus(n_dof);
        for (std::size_t j = 0; j < n_dof; ++j) {
          tau_plus[j] = images[i + 1][j] - images[i][j];
          tau_minus[j] = images[i][j] - images[i - 1][j];
        }
        // Normalize.
        double norm_plus = 0.0, norm_minus = 0.0;
        for (std::size_t j = 0; j < n_dof; ++j) {
          norm_plus += tau_plus[j] * tau_plus[j];
          norm_minus += tau_minus[j] * tau_minus[j];
        }
        norm_plus = std::sqrt(norm_plus);
        norm_minus = std::sqrt(norm_minus);
        if (norm_plus < 1e-30) norm_plus = 1.0;
        if (norm_minus < 1e-30) norm_minus = 1.0;

        // Energy-weighted tangent.
        double dE_plus = energies[i + 1] - energies[i];
        double dE_minus = energies[i] - energies[i - 1];
        std::vector<double> tau(n_dof, 0.0);
        if (dE_plus > 0 && dE_minus > 0) {
          // Use tau_plus.
          for (std::size_t j = 0; j < n_dof; ++j)
            tau[j] = tau_plus[j] / norm_plus;
        } else if (dE_plus < 0 && dE_minus < 0) {
          // Use tau_minus.
          for (std::size_t j = 0; j < n_dof; ++j)
            tau[j] = tau_minus[j] / norm_minus;
        } else {
          // Weighted average.
          double w_plus = std::fabs(dE_plus);
          double w_minus = std::fabs(dE_minus);
          for (std::size_t j = 0; j < n_dof; ++j)
            tau[j] = (w_plus * tau_plus[j] / norm_plus +
                      w_minus * tau_minus[j] / norm_minus);
        }
        // Normalize tau.
        double tau_norm = 0.0;
        for (std::size_t j = 0; j < n_dof; ++j) tau_norm += tau[j] * tau[j];
        tau_norm = std::sqrt(tau_norm);
        if (tau_norm > 1e-30)
          for (std::size_t j = 0; j < n_dof; ++j) tangents[i][j] = tau[j] / tau_norm;
      }

      // Find climbing image (highest energy interior image).
      std::size_t ci_idx = 1;
      double max_E = energies[1];
      for (std::size_t i = 2; i < n_img - 1; ++i) {
        if (energies[i] > max_E) {
          max_E = energies[i];
          ci_idx = i;
        }
      }
      res.climbing_image = ci_idx;

      // Compute forces on each image.
      std::vector<std::vector<double>> forces(n_img,
          std::vector<double>(n_dof, 0.0));
      res.max_force = 0.0;

      for (std::size_t i = 1; i < n_img - 1; ++i) {
        auto F_true = force_fn(images[i]);
        res.n_energy_evals++;  // force eval implies energy eval

        // Project out perpendicular component: F_perp = F - (F.tau) * tau
        double F_dot_tau = 0.0;
        for (std::size_t j = 0; j < n_dof; ++j)
          F_dot_tau += F_true[j] * tangents[i][j];
        std::vector<double> F_perp(n_dof);
        for (std::size_t j = 0; j < n_dof; ++j)
          F_perp[j] = F_true[j] - F_dot_tau * tangents[i][j];

        // Spring force: F_spring = k * (R_{i+1} - 2*R_i + R_{i-1}).
        // Only parallel component is kept.
        std::vector<double> F_spring(n_dof, 0.0);
        for (std::size_t j = 0; j < n_dof; ++j)
          F_spring[j] = k_spring * (images[i + 1][j] - 2.0 * images[i][j] +
                                     images[i - 1][j]);
        double Fs_dot_tau = 0.0;
        for (std::size_t j = 0; j < n_dof; ++j)
          Fs_dot_tau += F_spring[j] * tangents[i][j];
        std::vector<double> F_spring_parallel(n_dof);
        for (std::size_t j = 0; j < n_dof; ++j)
          F_spring_parallel[j] = Fs_dot_tau * tangents[i][j];

        // Total nudged force.
        for (std::size_t j = 0; j < n_dof; ++j)
          forces[i][j] = F_perp[j] + F_spring_parallel[j];

        // Climbing image: invert the parallel component of the true force.
        if (climb && step >= climb_start && i == ci_idx) {
          // F_CI = F_perp - F_parallel = F_true - 2 * (F_true . tau) * tau
          for (std::size_t j = 0; j < n_dof; ++j)
            forces[i][j] = F_true[j] - 2.0 * F_dot_tau * tangents[i][j];
        }

        // Track max force.
        for (std::size_t j = 0; j < n_dof; ++j)
          res.max_force = std::max(res.max_force, std::fabs(forces[i][j]));
      }

      // Check convergence.
      if (res.max_force < f_tol) {
        res.converged = true;
        break;
      }

      // Damped dynamics update (simple velocity Verlet with damping).
      const double damping = 0.9;
      for (std::size_t i = 1; i < n_img - 1; ++i) {
        for (std::size_t j = 0; j < n_dof; ++j) {
          velocities[i][j] = damping * velocities[i][j] + dt * forces[i][j];
          images[i][j] += dt * velocities[i][j];
        }
        // Recompute energy.
        energies[i] = energy_fn(images[i]);
        res.n_energy_evals++;
      }
    }

    // Finalize.
    res.images = images;
    res.energies = energies;
    return res;
  }
};

}  // namespace tides::dynamics
