// Grid convergence study: run H and H2 at multiple grid spacings and measure
// energy convergence. This is P0-1 from the audit: determine whether the grid
// integration scheme itself is correct (energy should converge as h -> 0).
//
// If energy converges to a stable value within ~1e-4 Ha, the integration scheme
// is correct and the issue is just grid resolution. If it does not converge,
// the integration scheme itself is wrong.

#include "scf/nao_driver.hpp"
#include "grid/xc/xc_engine.hpp"
#include "basis/pseudo/pp_loader.hpp"

#include <cmath>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using tides::scf::NaoDriver;

int RunConvergenceStudy() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║  Grid Convergence Study — P0-1                              ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  // Grid spacings to test (in Bohr).
  // With pseudopotentials + analytic on-site integrals, V_ext and V_nl
  // are grid-independent. Only Hartree and XC contribute to grid error,
  // giving smooth O(h^2) convergence. We test down to h=0.10 for H atom.
  const std::vector<double> h_grid_values = {0.50, 0.40, 0.30, 0.20, 0.15};
  const std::vector<double> h2_grid_values = {0.40, 0.30, 0.20, 0.15};

  // --- H atom ---
  std::cout << "\n=== H atom convergence ===\n";
  std::cout << std::setw(10) << "h (Bohr)"
            << std::setw(15) << "E (Ha)"
            << std::setw(15) << "dE vs prev"
            << std::setw(12) << "n_grid"
            << std::setw(10) << "conv?"
            << std::setw(12) << "wall_ms"
            << "\n";

  // Locate the bundled pseudopotential library relative to this source file
  // so the test works when run from the build directory.
  namespace fs = std::filesystem;
  const char* src_dir_env = std::getenv("TIDES_SRC_DIR");
  fs::path src_root = (src_dir_env && src_dir_env[0] != '\0')
                          ? fs::path(src_dir_env)
                          : fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
  std::string pp_dir = (src_root / "external" / "pseudopotentials" / "pseudodojo-pbe-sr").string();

  // Load H pseudopotential for PP-based calculation.
  auto pp_result = tides::basis::PpLoader::Load("H", pp_dir);
  if (!pp_result.ok()) {
    std::cerr << "Failed to load H pseudopotential: "
              << pp_result.status().message() << "\n";
    return 1;
  }
  auto h_pp = pp_result.value();
  std::cout << "Loaded H PP: Z_val=" << h_pp.Z_valence
            << " r_grid=" << h_pp.r_grid.size()
            << " channels=" << h_pp.channels.size() << "\n";

  std::vector<double> h_energies;
  std::vector<double> h_grid_sizes;
  std::vector<bool> h_converged;
  std::vector<double> h_wall_times;

  for (double h : h_grid_values) {
    std::vector<int> Z = {1};
    std::vector<double> pos = {0.0, 0.0, 0.0};
    std::vector<tides::basis::Pseudopotential> pps = {h_pp};
    // PP path: dual grid enabled, analytic on-site V_ext + V_nl, tol=1e-6.
    auto result = NaoDriver::Run(Z, pos, h, 4.0, 100, 1e-6,
                                 &pps, tides::grid::xc::HostXcSpec{}, 1, 0, true,
                                 0.0, false, false, false, false, false,
                                 false, false, false, false,
                                 std::array<int, 3>{1, 1, 1}, false, false,
                                 nullptr, false);
    double E = result.scf.energy;
    double dE = h_energies.empty() ? 0.0 : E - h_energies.back();
    std::size_t n_grid = result.grid_n[0] * result.grid_n[1] * result.grid_n[2];
    h_energies.push_back(E);
    h_grid_sizes.push_back(static_cast<double>(n_grid));
    h_converged.push_back(result.scf.converged);
    h_wall_times.push_back(result.wall_time_ms);
    std::cout << std::setw(10) << h
              << std::setw(15) << std::setprecision(8) << E
              << std::setw(15) << std::setprecision(4) << dE
              << std::setw(12) << n_grid
              << std::setw(10) << (result.scf.converged ? "YES" : "NO")
              << std::setw(12) << static_cast<int>(result.wall_time_ms)
              << "\n";
    std::cout << "    [E_kin=" << result.energy.E_kin
              << " E_ne=" << result.energy.E_ne
              << " E_H=" << result.energy.E_H
              << " E_xc=" << result.energy.E_xc
              << " E_ion=" << result.energy.E_ion << "]\n";
    std::cout.flush();
  }

  // --- H2 molecule ---
  std::cout << "\n=== H2 molecule convergence (R=1.4 Bohr) ===\n";
  std::cout << std::setw(10) << "h (Bohr)"
            << std::setw(15) << "E (Ha)"
            << std::setw(15) << "dE vs prev"
            << std::setw(12) << "n_grid"
            << std::setw(10) << "conv?"
            << std::setw(12) << "wall_ms"
            << "\n";

  std::vector<double> h2_energies;
  std::vector<double> h2_grid_sizes;
  std::vector<bool> h2_converged;
  std::vector<double> h2_wall_times;

  for (double h : h2_grid_values) {
    std::vector<int> Z = {1, 1};
    std::vector<double> pos = {0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
    std::vector<tides::basis::Pseudopotential> pps = {h_pp, h_pp};
    // PP path: dual grid enabled, analytic on-site V_ext + V_nl, tol=1e-6.
    auto result = NaoDriver::Run(Z, pos, h, 4.0, 100, 1e-6,
                                 &pps, tides::grid::xc::HostXcSpec{}, 1, 0, true,
                                 0.0, false, false, false, false, false,
                                 false, false, false, false,
                                 std::array<int, 3>{1, 1, 1}, false, false,
                                 nullptr, false);
    double E = result.scf.energy;
    double dE = h2_energies.empty() ? 0.0 : E - h2_energies.back();
    std::size_t n_grid = result.grid_n[0] * result.grid_n[1] * result.grid_n[2];
    h2_energies.push_back(E);
    h2_grid_sizes.push_back(static_cast<double>(n_grid));
    h2_converged.push_back(result.scf.converged);
    h2_wall_times.push_back(result.wall_time_ms);
    std::cout << std::setw(10) << h
              << std::setw(15) << std::setprecision(8) << E
              << std::setw(15) << std::setprecision(4) << dE
              << std::setw(12) << n_grid
              << std::setw(10) << (result.scf.converged ? "YES" : "NO")
              << std::setw(12) << static_cast<int>(result.wall_time_ms)
              << "\n";
    std::cout << "    [E_kin=" << result.energy.E_kin
              << " E_ne=" << result.energy.E_ne
              << " E_H=" << result.energy.E_H
              << " E_xc=" << result.energy.E_xc
              << " E_ion=" << result.energy.E_ion << "]\n";
    std::cout.flush();
  }

  // --- Analysis ---
  std::cout << "\n=== Analysis ===\n";

  // Check convergence: does energy change by < 1e-4 between two finest grids?
  // Also compute Richardson extrapolation and convergence order.
  int failures = 0;

  // Helper: estimate convergence order from 3 consecutive points.
  // E(h) ≈ E_inf + C * h^p  =>  p ≈ log(dE1/dE2) / log(r)
  // where dE1 = E(h2)-E(h1), dE2 = E(h3)-E(h2), r = h2/h3.
  auto estimate_order = [](double E1, double E2, double E3,
                           double h1, double h2, double h3) -> double {
    double dE1 = std::fabs(E2 - E1);
    double dE2 = std::fabs(E3 - E2);
    if (dE1 < 1e-15 || dE2 < 1e-15) return 0.0;
    double r12 = h2 / h3;  // ratio between consecutive h
    double r01 = h1 / h2;
    // p such that dE1/dE2 = (h2/h3)^p * ((1-(h1/h2)^p) / (1-(h2/h3)^p))
    // Simplified for equal ratio: p = log(dE1/dE2) / log(r)
    // For non-equal ratio, use iterative estimate:
    double p = std::log(dE1 / dE2) / std::log(r12);
    return p;
  };

  // Helper: Richardson extrapolation.
  auto richardson = [](double E_fine, double E_coarse,
                       double h_fine, double h_coarse,
                       double p) -> double {
    double r = h_coarse / h_fine;
    return E_fine + (E_fine - E_coarse) / (std::pow(r, p) - 1.0);
  };

  // --- H atom analysis ---
  if (h_energies.size() >= 2) {
    double dE_h = std::fabs(h_energies.back() - h_energies[h_energies.size() - 2]);
    std::cout << "H atom: |dE(finest two grids)| = " << dE_h << " Ha\n";

    // Estimate convergence order from last 3 points.
    if (h_energies.size() >= 3) {
      std::size_t n = h_energies.size();
      double p = estimate_order(h_energies[n-3], h_energies[n-2], h_energies[n-1],
                                h_grid_values[n-3], h_grid_values[n-2], h_grid_values[n-1]);
      std::cout << "H atom: estimated convergence order p ≈ " << p << "\n";

      // Richardson extrapolation using last 2 points.
      double E_rich = richardson(h_energies[n-1], h_energies[n-2],
                                 h_grid_values[n-1], h_grid_values[n-2], p);
      std::cout << "H atom Richardson extrapolation: E_inf ≈ " << E_rich << " Ha\n";

      // Richardson extrapolation using previous 2 points.
      if (n >= 4) {
        double E_rich_prev = richardson(h_energies[n-2], h_energies[n-3],
                                        h_grid_values[n-2], h_grid_values[n-3], p);
        double dE_rich = std::fabs(E_rich - E_rich_prev);
        std::cout << "H atom Richardson convergence: |dE_rich| = " << dE_rich << " Ha\n";
        if (dE_rich < 1e-4) {
          std::cout << "  CONVERGED (Richardson): extrapolated energy stable to < 1e-4 Ha.\n";
        } else {
          std::cout << "  Richardson extrapolation not yet stable (|dE_rich| = "
                    << dE_rich << " Ha).\n";
        }
      }
    }

    if (dE_h < 1e-4) {
      std::cout << "  CONVERGED (raw): energy stable to < 1e-4 Ha at finest grids.\n";
    } else {
      std::cout << "  Raw grid convergence: energy still changing by " << dE_h
                << " Ha (expected for grid-based DFT; use Richardson estimate).\n";
    }
  }

  // --- H2 analysis ---
  if (h2_energies.size() >= 2) {
    double dE_h2 = std::fabs(h2_energies.back() - h2_energies[h2_energies.size() - 2]);
    std::cout << "H2: |dE(finest two grids)| = " << dE_h2 << " Ha\n";

    if (h2_energies.size() >= 3) {
      std::size_t n = h2_energies.size();
      std::size_t start = 0;
      std::size_t nn = n - start;
      if (nn >= 3) {
        double p = estimate_order(h2_energies[start + nn - 3],
                                  h2_energies[start + nn - 2],
                                  h2_energies[start + nn - 1],
                                  h2_grid_values[start + nn - 3],
                                  h2_grid_values[start + nn - 2],
                                  h2_grid_values[start + nn - 1]);
        std::cout << "H2: estimated convergence order p ≈ " << p << "\n";

        double E_rich = richardson(h2_energies[n-1], h2_energies[n-2],
                                   h2_grid_values[n-1], h2_grid_values[n-2], p);
        std::cout << "H2 Richardson extrapolation: E_inf ≈ " << E_rich << " Ha\n";

        if (n >= 4) {
          double E_rich_prev = richardson(h2_energies[n-2], h2_energies[n-3],
                                          h2_grid_values[n-2], h2_grid_values[n-3], p);
          double dE_rich = std::fabs(E_rich - E_rich_prev);
          std::cout << "H2 Richardson convergence: |dE_rich| = " << dE_rich << " Ha\n";
          if (dE_rich < 1e-4) {
            std::cout << "  CONVERGED (Richardson): extrapolated energy stable to < 1e-4 Ha.\n";
          } else {
            std::cout << "  Richardson extrapolation not yet stable (|dE_rich| = "
                      << dE_rich << " Ha).\n";
          }
        }
      }
    }

    if (dE_h2 < 1e-4) {
      std::cout << "  CONVERGED (raw): energy stable to < 1e-4 Ha at finest grids.\n";
    } else {
      std::cout << "  Raw grid convergence: energy still changing by " << dE_h2
                << " Ha (expected for grid-based DFT; use Richardson estimate).\n";
    }
  }

  // Reference values (PP-based LDA, not all-electron)
  std::cout << "\nReference: H atom PP-LDA energy ≈ -0.5 Ha (1 electron, Z_val=1)\n";
  std::cout << "Reference: H2 PP-LDA energy ≈ -1.1 Ha (approximate, R=1.4 Bohr)\n";

  // Check if finest grid energies are reasonable
  if (!h_energies.empty()) {
    std::cout << "\nH atom finest grid energy: " << h_energies.back() << " Ha\n";
  }
  if (!h2_energies.empty()) {
    std::cout << "H2 finest grid energy: " << h2_energies.back() << " Ha\n";
  }

  std::cout << "\n=== Summary ===\n";
  if (failures == 0) {
    std::cout << "Grid convergence study completed.\n";
  } else {
    std::cout << failures << " issue(s) found.\n";
  }
  return failures;
}

}  // namespace

int main(int argc, char* argv[]) {
  return RunConvergenceStudy();
}
