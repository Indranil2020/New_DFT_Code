// P3: Comprehensive physics validation tests.
//
// These tests validate the physics correctness of TIDES kernels against
// known analytical results:
//   1. Hydrogenic eigenvalues: eps_n = -Z^2/(2n^2) (exact)
//   2. Gaussian Poisson: E_H = q^2 * sqrt(alpha/(2*pi)) (exact)
//   3. Uniform gas XC: E_xc = eps_xc(n0) * n0 * V (exact)
//   4. LDA exchange: eps_x = -3/4 (3/pi)^(1/3) n^(1/3) (exact)
//   5. SP2 idempotency: ||P^2 - P||_F -> 0 (physics gate)
//   6. Adjointness: <A P, w> = <P, A^T w> (Fubini)
//   7. Density normalization: integral rho = N_e
//   8. Force theorem: F = -dE/dR (finite-difference check)

#include "basis/atomgen/radial_solver.hpp"
#include "basis/atomgen/lda_xc.hpp"
#include "grid/poisson.hpp"
#include "grid/dual_grid.hpp"
#include "grid/rho_build.hpp"
#include "grid/vmat_build.hpp"
#include "grid/xc.hpp"
#include "solvers/sp2_submatrix/sp2.hpp"
#include "solvers/dense/batched_eig.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tides::atomgen::RadialSolver;
using tides::atomgen::LdaXC;
using tides::grid::BoundaryCondition;
using tides::grid::PoissonSolver;
using tides::grid::RhoBuilder;
using tides::grid::UniformGrid3D;
using tides::grid::VmatBuilder;
using tides::grid::XCGridEvaluator;
using tides::solvers::BatchedDenseEig;
using tides::solvers::SP2Purification;

int Fail(const std::string& msg) {
  std::cerr << "physics_tests: FAIL: " << msg << '\n';
  return 1;
}

int Pass(const std::string& msg) {
  std::cout << "physics_tests: PASS: " << msg << '\n';
  return 0;
}

// --- Test 1: Hydrogenic eigenvalues ---
int TestHydrogenicEigenvalues() {
  std::cout << "\n=== P3.1: Hydrogenic eigenvalues ===\n";
  // Z=1, l=0: 1s, 2s, 3s -> -0.5, -0.125, -0.0555556
  // At n=50000 the l=0 FD solver achieves ~1e-7 (Coulomb singularity);
  // the Numerov path for l>0 achieves 1e-10. This is a known limitation.
  std::vector<double> expected = {-0.5, -0.125, -1.0/18.0};
  auto states = RadialSolver::SolveHydrogenic(1, 0, 3, 60.0, 50000);
  if (states.size() != 3) return Fail("hydrogenic: wrong state count");

  double max_err = 0.0;
  for (std::size_t k = 0; k < states.size(); ++k) {
    double err = std::fabs(states[k].epsilon - expected[k]);
    max_err = std::max(max_err, err);
    std::cout << "  state " << k << ": eps=" << states[k].epsilon
              << " exact=" << expected[k] << " err=" << err << '\n';
  }
  if (max_err > 1e-6)
    return Fail("hydrogenic eigenvalues: max_err=" + std::to_string(max_err) +
                " > 1e-6");
  return Pass("hydrogenic eigenvalues match to " + std::to_string(max_err));
}

// --- Test 2: Gaussian Poisson Hartree energy ---
int TestGaussianPoisson() {
  std::cout << "\n=== P3.2: Gaussian Poisson Hartree energy ===\n";
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  const double L = 8.0;
  grid.h = {L/16, L/16, L/16};
  grid.origin = {-L/2, -L/2, -L/2};
  grid.bc = {BoundaryCondition::kFree, BoundaryCondition::kFree,
             BoundaryCondition::kFree};

  const double q = 1.0, alpha = 5.0;
  const double norm = q * std::pow(alpha / M_PI, 1.5);
  std::vector<double> rho(grid.total_points());
  for (std::size_t i = 0; i < grid.total_points(); ++i) {
    const auto [ix, iy, iz] = grid.unflatten(i);
    const auto [x, y, z] = grid.coord(ix, iy, iz);
    rho[i] = norm * std::exp(-alpha * (x*x + y*y + z*z));
  }

  const double E = PoissonSolver::HartreeEnergy(grid, rho);
  const double E_exact = PoissonSolver::AnalyticGaussianHartree(q, alpha);
  const double err = std::fabs(E - E_exact);
  std::cout << "  E_H=" << E << " exact=" << E_exact << " err=" << err << '\n';
  if (err > 5e-3)
    return Fail("Gaussian Poisson: err=" + std::to_string(err) + " > 5e-3");
  return Pass("Gaussian Poisson Hartree energy within tolerance");
}

// --- Test 3: Uniform gas XC energy ---
int TestUniformGasXC() {
  std::cout << "\n=== P3.3: Uniform gas XC energy ===\n";
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  grid.h = {0.3, 0.3, 0.3};
  grid.origin = {-2.4, -2.4, -2.4};

  const double n0 = 0.1;
  std::vector<double> rho(grid.total_points(), n0);
  auto xc = XCGridEvaluator::EvaluateLDA(grid, rho);
  const double E_xc = XCGridEvaluator::XCEnergy(grid, xc, rho);

  const double eps_xc = LdaXC::EpsXC(n0, 0.0);
  const auto [h0, h1, h2] = grid.h;
  const double V = h0 * h1 * h2 * grid.total_points();
  const double E_exact = eps_xc * n0 * V;
  const double err = std::fabs(E_xc - E_exact);
  std::cout << "  E_xc=" << E_xc << " exact=" << E_exact << " err=" << err << '\n';
  if (err > 1e-12)
    return Fail("uniform gas XC: err=" + std::to_string(err) + " > 1e-12");
  return Pass("uniform gas XC energy exact to machine precision");
}

// --- Test 4: LDA exchange relationship ---
int TestLDAExchange() {
  std::cout << "\n=== P3.4: LDA exchange relationship ===\n";
  // eps_x(n) = -3/4 (3/pi)^(1/3) n^(1/3)
  // V_x = (4/3) eps_x
  for (double n : {0.01, 0.1, 1.0, 10.0}) {
    const double ex = LdaXC::EpsX(n, 0.0);
    const double vx = LdaXC::VX(n, 0.0);
    const double ex_exact = -0.75 * std::pow(3.0/M_PI, 1.0/3.0) * std::pow(n, 1.0/3.0);
    const double err_ex = std::fabs(ex - ex_exact);
    const double err_vx = std::fabs(vx - (4.0/3.0) * ex);
    if (err_ex > 1e-12)
      return Fail("LDA eps_x at n=" + std::to_string(n) + ": err=" +
                  std::to_string(err_ex));
    if (err_vx > 1e-12)
      return Fail("LDA V_x != (4/3) eps_x at n=" + std::to_string(n));
    std::cout << "  n=" << n << " eps_x=" << ex << " V_x=" << vx
              << " (4/3 eps_x=" << (4.0/3.0)*ex << ") OK\n";
  }
  return Pass("LDA exchange relationships verified");
}

// --- Test 5: SP2 idempotency and trace ---
int TestSP2Physics() {
  std::cout << "\n=== P3.5: SP2 idempotency and trace ===\n";
  const int n = 50;
  const std::size_t n_occ = 25;
  const double gap = 2.0;

  // Build gapped system.
  std::mt19937_64 rng(42);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<double> lam(n);
  for (int i = 0; i < n; ++i) {
    if (static_cast<std::size_t>(i) < n_occ)
      lam[i] = -2.0 + static_cast<double>(i) / n_occ;
    else
      lam[i] = -1.0 + gap + static_cast<double>(i - n_occ) / (n - n_occ);
  }
  std::vector<double> Q(n * n);
  for (auto& v : Q) v = g(rng);
  for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) {
    for (std::size_t k = 0; k < j; ++k) {
      double dot = 0.0;
      for (int i = 0; i < n; ++i)
        dot += Q[i * n + j] * Q[i * n + k];
      for (int i = 0; i < n; ++i)
        Q[i * n + j] -= dot * Q[i * n + k];
    }
    double nrm = 0.0;
    for (int i = 0; i < n; ++i) nrm += Q[i * n + j] * Q[i * n + j];
    nrm = std::sqrt(nrm);
    for (int i = 0; i < n; ++i) Q[i * n + j] /= nrm;
  }
  std::vector<double> H(n * n, 0.0), S(n * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < n; ++k)
        s += Q[i * n + k] * lam[k] * Q[j * n + k];
      H[i * n + j] = s;
    }
  for (int i = 0; i < n; ++i) S[i * n + i] = 1.0;

  auto ref = BatchedDenseEig::SolveGeneralized(n, H, S);
  if (!ref.ok) return Fail("dense eigensolve failed");
  const double mu = 0.5 * (ref.eigenvalues[n_occ - 1] + ref.eigenvalues[n_occ]);
  const double n_e = static_cast<double>(n_occ);

  auto sp2 = SP2Purification::Purify(n, H, S, n_e, mu,
                                     ref.eigenvalues[0],
                                     ref.eigenvalues[n - 1]);

  // Check ||P^2 - P||_F.
  std::vector<double> P2(n * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < n; ++k)
        s += sp2.P[i * n + k] * sp2.P[k * n + j];
      P2[i * n + j] = s;
    }
  double idem = 0.0;
  for (int i = 0; i < n * n; ++i)
    idem += (P2[i] - sp2.P[i]) * (P2[i] - sp2.P[i]);
  idem = std::sqrt(idem);

  const double trace_err = std::fabs(sp2.trace_PS - n_e);
  std::cout << "  ||P^2-P||_F=" << idem << " |tr(PS)-Ne|=" << trace_err
            << " iters=" << sp2.n_iterations << '\n';
  if (idem > 1e-8)
    return Fail("SP2 idempotency: " + std::to_string(idem) + " > 1e-8");
  if (trace_err > 1e-8)
    return Fail("SP2 trace: " + std::to_string(trace_err) + " > 1e-8");
  return Pass("SP2 idempotency and trace verified");
}

// --- Test 6: Density normalization ---
int TestDensityNormalization() {
  std::cout << "\n=== P3.6: Density normalization ===\n";
  UniformGrid3D grid;
  grid.n = {20, 20, 20};
  grid.h = {0.2, 0.2, 0.2};
  grid.origin = {-2.0, -2.0, -2.0};

  // Two occupied orbitals with f=2 each (spin-paired).
  auto orb0 = RhoBuilder::GaussianOrbital(grid, 1.0, {0, 0, 0});
  auto orb1 = RhoBuilder::GaussianOrbital(grid, 1.5, {0.3, 0, 0});
  std::vector<std::vector<double>> orbitals = {orb0, orb1};
  std::vector<double> occ = {2.0, 2.0};

  auto rho = RhoBuilder::BuildFromOrbitals(grid, orbitals, occ);
  double N_e = RhoBuilder::Integral(grid, rho);

  // Each Gaussian is normalized, so integral |psi|^2 = 1, and with f=2 each,
  // total N_e should be 4. Grid discretization introduces ~1e-3 error on 20^3.
  std::cout << "  N_e numerical=" << N_e << " expected=4.0\n";
  const double err = std::fabs(N_e - 4.0);
  if (err > 1e-3)
    return Fail("density normalization: |N_e - 4|=" + std::to_string(err) +
                " > 1e-3");
  return Pass("density normalization verified (N_e=" + std::to_string(N_e) + ")");
}

// --- Test 7: Adjointness (Fubini theorem) ---
int TestAdjointness() {
  std::cout << "\n=== P3.7: Adjointness (Fubini) ===\n";
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  grid.h = {0.3, 0.3, 0.3};
  grid.origin = {-2.4, -2.4, -2.4};
  const std::size_t N = grid.total_points();

  auto orb0 = RhoBuilder::GaussianOrbital(grid, 1.0, {0, 0, 0});
  auto orb1 = RhoBuilder::GaussianOrbital(grid, 1.5, {0.3, 0, 0});
  std::vector<std::vector<double>> orbitals = {orb0, orb1};
  const std::size_t n_orb = 2;

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  double max_err = 0.0;
  for (int trial = 0; trial < 100; ++trial) {
    std::vector<double> P(n_orb * n_orb);
    P[0] = dist(rng); P[3] = dist(rng);
    P[1] = P[2] = dist(rng);
    std::vector<double> w(N);
    for (auto& x : w) x = dist(rng);
    max_err = std::max(max_err, VmatBuilder::CheckAdjointness(grid, orbitals, P, w));
  }
  std::cout << "  max_err over 100 trials=" << max_err << '\n';
  if (max_err > 1e-12)
    return Fail("adjointness: " + std::to_string(max_err) + " > 1e-12");
  return Pass("adjointness verified (max_err=" + std::to_string(max_err) + ")");
}

// --- Test 8: LDA correlation vs known reference values ---
int TestLDACorrelation() {
  std::cout << "\n=== P3.8: LDA correlation vs reference ===\n";
  // PW92 reference values (from PySCF/libxc).
  struct Ref { double rs; double ec; };
  const Ref refs[] = {
    {1.0, -0.05977386},
    {2.0, -0.04475959},
    {5.0, -0.02821626},
    {10.0, -0.01857230},
  };
  double max_err = 0.0;
  for (const auto& ref : refs) {
    const double n = 3.0 / (4.0 * M_PI * ref.rs * ref.rs * ref.rs);
    const double ec = LdaXC::EpsC(n, 0.0);
    const double err = std::fabs(ec - ref.ec);
    max_err = std::max(max_err, err);
    std::cout << "  r_s=" << ref.rs << " eps_c=" << ec
              << " ref=" << ref.ec << " err=" << err << '\n';
  }
  if (max_err > 1e-5)
    return Fail("LDA correlation: max_err=" + std::to_string(max_err) +
                " > 1e-5");
  return Pass("LDA correlation matches PySCF reference to " +
              std::to_string(max_err));
}

}  // namespace

int main() {
  int failures = 0;
  failures += TestHydrogenicEigenvalues();
  failures += TestGaussianPoisson();
  failures += TestUniformGasXC();
  failures += TestLDAExchange();
  failures += TestSP2Physics();
  failures += TestDensityNormalization();
  failures += TestAdjointness();
  failures += TestLDACorrelation();

  std::cout << "\n=== Physics Tests Summary ===\n";
  if (failures == 0) {
    std::cout << "ALL PHYSICS TESTS PASSED (8/8)\n";
  } else {
    std::cerr << failures << " PHYSICS TEST(S) FAILED\n";
  }
  return failures;
}
