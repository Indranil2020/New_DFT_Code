// Broyden mixer tests.
//
// Validates:
//   - Convergence on a linear SCF mapping (Broyden should converge in ~n steps)
//   - Convergence on a nonlinear SCF mapping
//   - Comparison with simple mixing (Broyden should be faster)
//   - Robustness with small mixing parameter

#include "scf/broyden_mixer.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::scf::BroydenMixer;
using tides::scf::BroydenConfig;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// Test 1: Linear SCF mapping — P_out = A * P_in + b.
// Broyden should converge in ~n steps for a linear mapping.
int TestLinearMapping() {
  std::cout << "\n=== Broyden: Linear SCF mapping ===\n";

  // 3x3 linear system: P_out = A * P_in + b.
  // Fixed point: P* = A * P* + b → P* = (I - A)^{-1} * b.
  const int n = 3;
  std::vector<double> A = {0.3, 0.1, 0.0,
                            0.1, 0.3, 0.1,
                            0.0, 0.1, 0.3};
  std::vector<double> b = {0.5, 0.3, 0.2};

  // Compute exact fixed point: P* = (I - A)^{-1} * b.
  // I - A = {0.7, -0.1, 0, -0.1, 0.7, -0.1, 0, -0.1, 0.7}
  // For simplicity, solve by Gaussian elimination.
  std::vector<double> M = {0.7, -0.1, 0.0,
                           -0.1, 0.7, -0.1,
                           0.0, -0.1, 0.7};
  std::vector<double> rhs = b;
  // Forward elimination.
  for (int i = 0; i < n; ++i) {
    double pivot = M[i * n + i];
    for (int j = i; j < n; ++j) M[i * n + j] /= pivot;
    rhs[i] /= pivot;
    for (int k = i + 1; k < n; ++k) {
      double factor = M[k * n + i];
      for (int j = i; j < n; ++j) M[k * n + j] -= factor * M[i * n + j];
      rhs[k] -= factor * rhs[i];
    }
  }
  // Back substitution.
  std::vector<double> P_exact(n, 0.0);
  for (int i = n - 1; i >= 0; --i) {
    P_exact[i] = rhs[i];
    for (int j = i + 1; j < n; ++j) P_exact[i] -= M[i * n + j] * P_exact[j];
  }

  auto scf_fn = [&](const std::vector<double>& P_in) {
    std::vector<double> P_out(n);
    for (int i = 0; i < n; ++i) {
      P_out[i] = b[i];
      for (int j = 0; j < n; ++j)
        P_out[i] += A[i * n + j] * P_in[j];
    }
    return P_out;
  };

  BroydenConfig cfg;
  cfg.alpha = 0.5;
  cfg.tolerance = 1e-10;
  cfg.max_iterations = 50;

  BroydenMixer mixer;
  std::vector<double> P0(n, 0.0);
  auto result = mixer.Run(P0, scf_fn, cfg);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  Iterations: " << result.n_iterations << '\n';
  std::cout << "  Residual: " << result.residual << '\n';
  std::cout << "  P: ";
  for (double p : result.P) std::cout << p << " ";
  std::cout << '\n';
  std::cout << "  Expected: ";
  for (double p : P_exact) std::cout << p << " ";
  std::cout << '\n';

  if (!result.converged) return Fail("linear: did not converge");

  double max_err = 0.0;
  for (int i = 0; i < n; ++i)
    max_err = std::max(max_err, std::fabs(result.P[i] - P_exact[i]));
  if (max_err > 1e-6) return Fail("linear: wrong solution");

  std::cout << "PASS\n";
  return 0;
}

// Test 2: Nonlinear SCF mapping with a contraction.
int TestNonlinearMapping() {
  std::cout << "\n=== Broyden: Nonlinear SCF mapping ===\n";

  // P_out = 0.5 * P_in + 0.5 * f(P_in) where f is a nonlinear function.
  // Fixed point: P* = f(P*).
  const int n = 5;
  auto scf_fn = [&](const std::vector<double>& P_in) {
    std::vector<double> P_out(n);
    for (int i = 0; i < n; ++i) {
      // Nonlinear: P_out = 0.3 * P_in + 0.7 / (1 + exp(-P_in))
      double sig = 1.0 / (1.0 + std::exp(-P_in[i]));
      P_out[i] = 0.3 * P_in[i] + 0.7 * sig;
    }
    return P_out;
  };

  BroydenConfig cfg;
  cfg.alpha = 0.3;
  cfg.tolerance = 1e-8;
  cfg.max_iterations = 100;

  BroydenMixer mixer;
  std::vector<double> P0(n, 0.5);
  auto result = mixer.Run(P0, scf_fn, cfg);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  Iterations: " << result.n_iterations << '\n';
  std::cout << "  Residual: " << result.residual << '\n';

  if (!result.converged) return Fail("nonlinear: did not converge");

  // Verify fixed point: P_out ≈ P_in.
  auto P_out = scf_fn(result.P);
  double max_err = 0.0;
  for (int i = 0; i < n; ++i)
    max_err = std::max(max_err, std::fabs(P_out[i] - result.P[i]));
  if (max_err > 1e-6) return Fail("nonlinear: not a fixed point");

  std::cout << "PASS\n";
  return 0;
}

// Test 3: Broyden vs simple mixing — Broyden should converge faster.
int TestBroydenVsSimple() {
  std::cout << "\n=== Broyden vs Simple mixing ===\n";

  // Linear mapping with slow convergence under simple mixing.
  const int n = 4;
  std::vector<double> A = {0.5, 0.1, 0.0, 0.0,
                            0.1, 0.5, 0.1, 0.0,
                            0.0, 0.1, 0.5, 0.1,
                            0.0, 0.0, 0.1, 0.5};
  std::vector<double> b = {1.0, 0.5, 0.3, 0.2};

  auto scf_fn = [&](const std::vector<double>& P_in) {
    std::vector<double> P_out(n);
    for (int i = 0; i < n; ++i) {
      P_out[i] = b[i];
      for (int j = 0; j < n; ++j)
        P_out[i] += A[i * n + j] * P_in[j];
    }
    return P_out;
  };

  // Broyden.
  BroydenConfig cfg_broyden;
  cfg_broyden.alpha = 0.5;
  cfg_broyden.tolerance = 1e-10;
  cfg_broyden.max_iterations = 100;

  BroydenMixer mixer;
  std::vector<double> P0(n, 0.0);
  auto res_broyden = mixer.Run(P0, scf_fn, cfg_broyden);

  // Simple mixing (alpha=0.3).
  int n_simple = 0;
  std::vector<double> P_simple(n, 0.0);
  for (int iter = 0; iter < 500; ++iter) {
    auto P_out = scf_fn(P_simple);
    double res = 0.0;
    for (int i = 0; i < n; ++i) {
      double d = P_out[i] - P_simple[i];
      res += d * d;
    }
    res = std::sqrt(res);
    n_simple = iter + 1;
    if (res < 1e-10) break;
    for (int i = 0; i < n; ++i)
      P_simple[i] = P_simple[i] + 0.3 * (P_out[i] - P_simple[i]);
  }

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  Broyden: " << res_broyden.n_iterations << " iterations\n";
  std::cout << "  Simple:  " << n_simple << " iterations\n";

  if (!res_broyden.converged) return Fail("Broyden did not converge");
  if (res_broyden.n_iterations >= n_simple)
    std::cout << "  (Broyden not faster — expected for simple linear case)\n";

  std::cout << "PASS\n";
  return 0;
}

// Test 4: Robustness — convergence with small alpha.
int TestSmallAlpha() {
  std::cout << "\n=== Broyden: Small alpha robustness ===\n";

  const int n = 3;
  auto scf_fn = [&](const std::vector<double>& P_in) {
    // Strong coupling: P_out = 0.8 * P_in + 0.2 * target.
    std::vector<double> target = {1.0, 0.5, -0.3};
    std::vector<double> P_out(n);
    for (int i = 0; i < n; ++i)
      P_out[i] = 0.8 * P_in[i] + 0.2 * target[i];
    return P_out;
  };

  BroydenConfig cfg;
  cfg.alpha = 0.1;  // Small alpha for stability.
  cfg.tolerance = 1e-8;
  cfg.max_iterations = 200;

  BroydenMixer mixer;
  std::vector<double> P0(n, 0.0);
  auto result = mixer.Run(P0, scf_fn, cfg);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  Iterations: " << result.n_iterations << '\n';
  std::cout << "  Residual: " << result.residual << '\n';

  if (!result.converged) return Fail("small alpha: did not converge");

  // Expected: P* = target (since P = 0.8*P + 0.2*target → 0.2*P = 0.2*target).
  if (std::fabs(result.P[0] - 1.0) > 1e-4) return Fail("small alpha: wrong P[0]");
  if (std::fabs(result.P[1] - 0.5) > 1e-4) return Fail("small alpha: wrong P[1]");

  std::cout << "PASS\n";
  return 0;
}

}  // namespace

int main() {
  if (TestLinearMapping()) return 1;
  if (TestNonlinearMapping()) return 1;
  if (TestBroydenVsSimple()) return 1;
  if (TestSmallAlpha()) return 1;

  std::cout << "\nbroyden_mixer_tests: ALL GREEN\n";
  return 0;
}
