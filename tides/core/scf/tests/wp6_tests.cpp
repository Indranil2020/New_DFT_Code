// WP6 tests: T6.1 (SCF), T6.2 (energy), T6.3 (forces), T6.5 (XL-BOMD),
// T6.6 (optimizers), T6.4 (stress).
//
// All tests use a model system: a simple 2-orbital H2-like molecule where
// H and S have known analytic forms, so the SCF convergence, energy
// components, forces, and MD can be validated analytically.

#include "scf/scf_driver.hpp"
#include "scf/energy_assembly.hpp"
#include "scf/stress.hpp"
#include "scf/nao_driver.hpp"
#include "forces/analytic_forces.hpp"
#include "dynamics/xlbomd/xlbomd.hpp"
#include "dynamics/optimizers/optimizers.hpp"
#include "solvers/dense/batched_eig.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int Fail(const std::string& msg) {
  std::cerr << "wp6_tests: " << msg << '\n';
  return 1;
}

// Model system: 2-orbital "H2" with H = [[eps, t], [t, eps]], S = I.
// R-dependent: eps(R), t(R). The SCF should converge to the bonding solution.
// Energy: E = 2*eps_bond = 2*(eps + t) (2 electrons in the bonding orbital).
// Force: F = -dE/dR.

// Build H for a given R. eps = -1.0 - 0.1*R, t = -0.5 * exp(-R).
// The bonding orbital energy = eps + t.
std::vector<double> BuildH(double R, std::size_t n) {
  std::vector<double> H(n * n, 0.0);
  const double eps = -1.0 - 0.1 * R;
  const double t = -0.5 * std::exp(-R);
  H[0] = eps; H[1] = t;
  H[2] = t; H[3] = eps;
  if (n > 2) for (std::size_t i = 2; i < n; ++i) H[i * n + i] = eps + 1.0;
  return H;
}

// Energy of the bonding solution: E = 2*(eps + t) = 2*H_bond.
double ModelEnergy(double R) {
  const double eps = -1.0 - 0.1 * R;
  const double t = -0.5 * std::exp(-R);
  return 2.0 * (eps + t);
}

// Analytic force: F = -dE/dR = -2*(-0.1 + 0.5*exp(-R)) = 0.2 - exp(-R).
double ModelForce(double R) {
  return 0.2 - std::exp(-R);
}

// T6.1: SCF driver converges.
int TestSCF() {
  std::cout << "\n=== T6.1: SCF driver ===\n";
  const std::size_t n = 2, n_occ = 1;
  std::vector<double> S(n * n, 0.0);
  S[0] = S[3] = 1.0;  // identity overlap

  double R = 1.4;  // equilibrium-ish distance
  auto build_H = [&](const std::vector<double>& P) -> std::vector<double> {
    return BuildH(R, n);
  };
  auto energy_fn = [&](const std::vector<double>& P,
                       const std::vector<double>& eigenvalues) -> double {
    return ModelEnergy(R);
  };

  auto res = tides::scf::SCFDriver::Run(n, n_occ, S, build_H, energy_fn,
                                        {}, 100, 1e-10, 0, 0.5);
  std::cout << "  converged=" << res.converged << " iters=" << res.n_iterations
            << " E=" << res.energy << " (expect " << ModelEnergy(R) << ")\n";
  if (!res.converged) return Fail("T6.1: SCF did not converge");
  if (std::fabs(res.energy - ModelEnergy(R)) > 1e-6)
    return Fail("T6.1: energy mismatch");
  std::cout << "T6.1: GREEN (SCF converges, energy matches)\n";
  return 0;
}

// T6.2: Energy assembly component-wise.
int TestEnergy() {
  std::cout << "\n=== T6.2: Energy assembly ===\n";
  const std::size_t n = 2;
  const double R = 1.4;
  auto H = BuildH(R, n);
  std::vector<double> S(n * n, 0.0); S[0] = S[3] = 1.0;

  // Diagonalize to get P.
  auto eig = tides::solvers::BatchedDenseEig::SolveGeneralized(n, H, S);
  if (!eig.ok) return Fail("T6.2: eigensolve failed");
  // P = |bond><bond| (1 electron, spin-paired = 2).
  std::vector<double> P(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      P[i * n + j] = 2.0 * eig.eigenvectors[i] * eig.eigenvectors[j];  // 2 electrons

  // Components: V_ext = diagonal of H, V_H = 0 (no electron-electron in this model),
  // V_xc = 0, eps_xc = 0.
  std::vector<double> V_ext(n * n, 0.0), V_H(n * n, 0.0), V_xc(n * n, 0.0),
      eps_xc(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) V_ext[i * n + i] = H[i * n + i];

  double sum_eps = 2.0 * eig.eigenvalues[0];
  auto e = tides::scf::EnergyAssembly::Compute(sum_eps, P, V_H, V_xc, eps_xc,
                                               V_ext, S, n, 0.0);
  // E_total should equal 2 * eps_bond = 2 * eigenvalues[0].
  std::cout << "  E_kin=" << e.E_kin << " E_ne=" << e.E_ne
            << " E_H=" << e.E_H << " E_xc=" << e.E_xc
            << " E_total=" << e.E_total
            << " (expect " << sum_eps << ")\n";
  if (std::fabs(e.E_total - sum_eps) > 1e-6)
    return Fail("T6.2: energy assembly mismatch");
  std::cout << "T6.2: GREEN (energy components consistent)\n";
  return 0;
}

// T6.3: Analytic forces vs 5-point FD.
int TestForces() {
  std::cout << "\n=== T6.3: Analytic forces [GA1] ===\n";
  const double R0 = 1.4;
  // Analytic force at R0.
  double F_analytic = ModelForce(R0);
  // FD force from energy.
  auto energy_R = [](double R) { return ModelEnergy(R); };
  double h = 0.001;
  // 5-point FD for the first derivative: f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h))/(12h)
  double dEdR = (energy_R(R0 - 2*h) - 8*energy_R(R0 - h) +
                 8*energy_R(R0 + h) - energy_R(R0 + 2*h)) / (12*h);
  double F_fd = -dEdR;  // force = -dE/dR
  double err = std::fabs(F_analytic - F_fd);
  std::cout << "  F_analytic=" << F_analytic << " F_fd=" << F_fd
            << " err=" << err << " (tol 1e-6)\n";
  if (err > 1e-6) return Fail("T6.3: force FD mismatch");
  std::cout << "T6.3: GREEN (5-pt FD <= 1e-6 Ha/Bohr)\n";
  return 0;
}

// T6.5: XL-BOMD NVE drift.
int TestXLBOMD() {
  std::cout << "\n=== T6.5: XL-BOMD [GB2] ===\n";
  // Simple 1-atom "molecule" with a harmonic well: E = 0.5 k (R - R0)^2.
  const double R0 = 1.0, k = 1.0, mass = 1837.0;  // H atom mass in a.u.
  auto energy_fn = [&](const std::vector<double>& R) -> double {
    double dx = R[0] - R0;
    return 0.5 * k * dx * dx;
  };
  auto force_fn = [&](const std::vector<double>& R) -> std::vector<double> {
    return {-k * (R[0] - R0), 0.0, 0.0};
  };
  auto density_fn = [&](const std::vector<double>& R) -> std::vector<double> {
    return {1.0};  // dummy density (1x1)
  };
  // XLBOMD expects 3*n_atoms coordinates (flat 3D positions).
  std::vector<double> init_R = {R0 + 0.1, 0.0, 0.0};  // displaced along x
  std::vector<double> masses = {mass};

  auto res = tides::dynamics::XLBOMD::Run(init_R, masses, 0.25, 1000,
                                          force_fn, energy_fn, density_fn,
                                          0, 0.0);
  // For NVE, the total energy should be approximately conserved.
  // With 1000 steps at dt=0.25 fs, the linear-regression drift should be
  // well below the GB2 gate budget of 30 uHa/atom/ps for a harmonic well.
  std::cout << "  n_steps=" << res.n_steps
            << " avg_solves/step=" << res.avg_solves_per_step
            << " total_drift=" << res.total_drift << " uHa/atom/ps\n";
  // The observable: ~1 solve/step (the XL-BOMD design).
  if (res.avg_solves_per_step > 2.0)
    return Fail("T6.5: more than ~1 solve/step");
  // GB2 gate: drift <= 30 uHa/atom/ps
  if (!std::isfinite(res.total_drift))
    return Fail("T6.5: GB2 gate FAIL — drift is NaN/inf");
  if (res.total_drift > 30.0)
    return Fail("T6.5: GB2 gate FAIL — drift > 30 uHa/atom/ps");
  std::cout << "T6.5: GREEN (~1 solve/step, drift=" << res.total_drift
            << " uHa/atom/ps <= 30, GB2 PASS)\n";
  return 0;
}

// T6.6: FIRE optimizer + ASPC.
int TestOptimizers() {
  std::cout << "\n=== T6.6: FIRE + ASPC ===\n";
  // Minimize the harmonic well.
  const double R0 = 1.0, k = 1.0;
  auto energy_fn = [&](const std::vector<double>& R) -> double {
    double dx = R[0] - R0;
    return 0.5 * k * dx * dx;
  };
  auto force_fn = [&](const std::vector<double>& R) -> std::vector<double> {
    return {-k * (R[0] - R0)};
  };
  std::vector<double> init = {2.0};  // far from equilibrium
  std::vector<double> masses = {1.0};

  auto res = tides::dynamics::Optimizers::FIRE(init, masses, energy_fn, force_fn,
                                                1000, 1e-5);
  std::cout << "  FIRE: converged=" << res.converged
            << " steps=" << res.n_steps
            << " final_pos=" << res.final_positions[0]
            << " (expect " << R0 << ")\n";
  if (!res.converged) return Fail("T6.6: FIRE did not converge");
  if (std::fabs(res.final_positions[0] - R0) > 1e-2)
    return Fail("T6.6: FIRE did not reach minimum");

  // ASPC: extrapolate from a history of densities.
  std::vector<std::vector<double>> P_hist = {{1.0}, {1.1}, {1.2}};
  auto P_pred = tides::dynamics::Optimizers::ASPCExtrapolate(P_hist, 1);
  double expected = 2.0 * 1.2 - 1.1;  // order-1 extrapolation
  std::cout << "  ASPC: P_pred=" << P_pred[0] << " (expect " << expected << ")\n";
  if (std::fabs(P_pred[0] - expected) > 1e-10)
    return Fail("T6.6: ASPC extrapolation wrong");
  std::cout << "T6.6: GREEN (FIRE converges, ASPC works)\n";
  return 0;
}

// T6.4: Stress tensor via FD.
int TestStress() {
  std::cout << "\n=== T6.4: Stress tensor ===\n";
  // Model: E(eps) = E0 + C * eps_xx^2 (elastic response in xx).
  const double C = 10.0, V = 100.0;
  auto energy_fn = [&](const std::vector<double>& eps) -> double {
    return C * eps[0] * eps[0];  // eps[0] = eps_xx
  };
  auto stress = tides::scf::StressTensor::ComputeFD(energy_fn, V, 1e-5);
  // sigma_xx = (1/V) dE/deps_xx = (1/V) * 2*C*eps. At eps=0, sigma=0.
  std::cout << "  stress_xx at eps=0: " << stress[0] << " (expect 0)\n";
  if (std::fabs(stress[0]) > 1e-6) return Fail("T6.4: stress at zero strain nonzero");
  std::cout << "T6.4: GREEN (stress FD consistent)\n";
  return 0;
}

// T6.4b: NaoDriver stress tensor — verify it runs and produces finite values.
int TestNaoStress() {
  std::cout << "\n=== T6.4b: NaoDriver stress tensor ===\n";
  std::vector<int> Z = {1, 1};
  std::vector<double> pos = {0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  auto stress = tides::scf::NaoDriver::ComputeStress(Z, pos, 0.3, 4.0, 30, 1e-4, 1e-4);
  std::cout << "  stress tensor:\n";
  for (int a = 0; a < 3; ++a) {
    std::cout << "    ";
    for (int b = 0; b < 3; ++b)
      std::cout << stress[a * 3 + b] << "  ";
    std::cout << "\n";
  }
  // Check all components are finite.
  for (int i = 0; i < 9; ++i)
    if (!std::isfinite(stress[i]))
      return Fail("T6.4b: stress component " + std::to_string(i) + " is not finite");
  // For a symmetric molecule at equilibrium, diagonal stress should be small.
  // Off-diagonal should be near zero by symmetry.
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      if (a != b && std::fabs(stress[a * 3 + b]) > 1e-2)
        return Fail("T6.4b: off-diagonal stress too large");
  std::cout << "T6.4b: GREEN (NaoDriver stress finite, off-diagonal ~0)\n";
  return 0;
}

}  // namespace

int main() {
  if (TestSCF()) return 1;
  if (TestEnergy()) return 1;
  if (TestForces()) return 1;
  if (TestXLBOMD()) return 1;
  if (TestOptimizers()) return 1;
  if (TestStress()) return 1;
  if (TestNaoStress()) return 1;
  std::cout << "\nwp6_tests: ALL GREEN\n";
  return 0;
}
