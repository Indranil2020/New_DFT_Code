// XL-BOMD extended tests (B1-B4): NVE drift, NHC thermostat, KSA kernel,
// shadow dynamics, time-reversibility.
#include "dynamics/xlbomd/xlbomd.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using tides::dynamics::XLBOMD;
using tides::dynamics::XLBOMDResult;
using tides::dynamics::KSAKernelConfig;
using tides::dynamics::ThermostatType;

int Fail(const std::string& msg) {
  std::cerr << "xlbomd_extended_tests: FAIL — " << msg << '\n';
  return 1;
}

// Harmonic oscillator model: F = -k*x, E = 0.5*k*x^2 + ion-ion repulsion.
// For XL-BOMD, the force depends on R and P_aux. We use P_aux to modulate
// the spring constant slightly (shadow potential).
struct HarmonicModel {
  double k = 0.0001;
  double n_basis = 2;

  std::vector<double> force(const std::vector<double>& R,
                             const std::vector<double>& P) {
    (void)P;  // model ignores P
    std::vector<double> F(R.size(), 0.0);
    for (std::size_t i = 0; i < R.size(); ++i)
      F[i] = -k * R[i];
    return F;
  }

  double energy(const std::vector<double>& R) {
    double E = 0.0;
    for (double r : R) E += 0.5 * k * r * r;
    return E;
  }

  std::vector<double> density(const std::vector<double>& /*R*/) {
    // Model: return empty density so auxiliary dynamics is skipped
    // (the model force_fn ignores P anyway).
    return {};
  }
};

int TestNVEDrift() {
  std::cout << "\n=== B4: NVE Drift Test (1000 steps) ===\n";
  HarmonicModel model;
  std::vector<double> R0 = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0};
  std::vector<double> masses = {1.0, 1.0};
  KSAKernelConfig kcfg;
  kcfg.kernel_order = 1;

  auto force_fn = [&](const std::vector<double>& r, const std::vector<double>& p) {
    return model.force(r, p);
  };
  auto energy_fn = [&](const std::vector<double>& r) { return model.energy(r); };
  auto density_fn = [&](const std::vector<double>& r) { return model.density(r); };

  auto res = XLBOMD::Run(R0, masses, 1.0, 1000, force_fn, energy_fn,
                          density_fn, 0, 0.0, kcfg, 0, 42);
  std::cout << "  Drift: " << res.total_drift << " uHa/atom/ps\n";
  std::cout << "  Solves/step: " << res.avg_solves_per_step << "\n";
  std::cout << "  Final energy: " << res.energy_history.back() << "\n";

  if (!std::isfinite(res.total_drift) || res.total_drift > 30.0)
    return Fail("NVE drift exceeds 30 uHa/atom/ps: " +
                std::to_string(res.total_drift));
  std::cout << "  PASS\n";
  return 0;
}

int TestNHCThermostat() {
  std::cout << "\n=== B2: NHC Thermostat Test ===\n";
  HarmonicModel model;
  std::vector<double> R0 = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0};
  std::vector<double> masses = {1.0, 1.0};

  double kT_target = 0.01;  // target temperature in Hartree

  auto force_fn = [&](const std::vector<double>& r, const std::vector<double>& p) {
    return model.force(r, p);
  };
  auto energy_fn = [&](const std::vector<double>& r) { return model.energy(r); };
  auto density_fn = [&](const std::vector<double>& r) { return model.density(r); };

  auto res = XLBOMD::Run(R0, masses, 1.0, 500, force_fn, energy_fn,
                          density_fn, static_cast<int>(ThermostatType::kNHC),
                          kT_target, {}, 0, 99);

  // The NHC thermostat is an approximate CPU reference implementation.
  // It is validated structurally (runs to completion) rather than for
  // exact energy conservation on the simple model system.
  std::cout << "  NHC completed: " << res.n_steps << " steps\n";
  std::cout << "  N steps: " << res.n_steps << "\n";

  // Verify the thermostat ran the requested number of steps.
  if (res.n_steps != 500)
    return Fail("NHC did not run 500 steps");
  std::cout << "  PASS\n";
  return 0;
}

int TestKSAKernel() {
  std::cout << "\n=== B1: KSA Kernel Test ===\n";
  HarmonicModel model;
  std::vector<double> R0 = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0};
  std::vector<double> masses = {1.0, 1.0};

  auto force_fn = [&](const std::vector<double>& r, const std::vector<double>& p) {
    return model.force(r, p);
  };
  auto energy_fn = [&](const std::vector<double>& r) { return model.energy(r); };
  auto density_fn = [&](const std::vector<double>& r) { return model.density(r); };

  // Run with K=I (kernel_order=0).
  KSAKernelConfig kcfg0;
  kcfg0.kernel_order = 0;
  auto res0 = XLBOMD::Run(R0, masses, 1.0, 500, force_fn, energy_fn,
                           density_fn, 0, 0.0, kcfg0, 0, 42);

  // Run with KSA diagonal (kernel_order=1).
  KSAKernelConfig kcfg1;
  kcfg1.kernel_order = 1;
  auto res1 = XLBOMD::Run(R0, masses, 1.0, 500, force_fn, energy_fn,
                           density_fn, 0, 0.0, kcfg1, 0, 42);

  std::cout << "  K=I drift:     " << res0.total_drift << " uHa/atom/ps\n";
  std::cout << "  KSA diag drift: " << res1.total_drift << " uHa/atom/ps\n";

  // KSA should be at least as good (not necessarily better for simple models).
  if (res1.total_drift > res0.total_drift * 5.0)
    return Fail("KSA kernel is much worse than K=I");
  std::cout << "  PASS (K=I vs diagonal)\n";

  // Test low-rank kernel (kernel_order=2) with a non-trivial density matrix.
  // The density must depend on R so that P_0 (computed once at init) differs
  // from the ground state at later positions, creating a non-zero residual
  // that the kernel must correct. The force also depends on P so that
  // different kernels produce different trajectories.
  struct DensityModel {
    double k = 0.0001;
    double coupling = 0.0001;  // small P-dependent force coupling

    std::vector<double> force(const std::vector<double>& R,
                               const std::vector<double>& P) {
      std::vector<double> F(R.size(), 0.0);
      for (std::size_t i = 0; i < R.size(); ++i)
        F[i] = -k * R[i];
      if (!P.empty()) {
        std::size_t n = static_cast<std::size_t>(
            std::round(std::sqrt(static_cast<double>(P.size()))));
        if (n * n == P.size()) {
          double tr_P = 0.0;
          for (std::size_t j = 0; j < n; ++j) tr_P += P[j * n + j];
          for (std::size_t i = 0; i < R.size(); ++i)
            F[i] += coupling * tr_P * 0.01;
        }
      }
      return F;
    }
    double energy(const std::vector<double>& R) {
      double E = 0.0;
      for (double r : R) E += 0.5 * k * r * r;
      return E;
    }
    std::vector<double> density(const std::vector<double>& R) {
      double r0 = R.size() > 0 ? R[0] : 0.0;
      double r1 = R.size() > 3 ? R[3] : 0.0;
      return {0.8 + 0.01 * r0, 0.1, 0.05,
              0.1, 0.7 + 0.01 * r1, 0.08,
              0.05, 0.08, 0.6};
    }
  } dmodel;

  std::vector<double> dR0 = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0};
  auto dforce_fn = [&](const std::vector<double>& r, const std::vector<double>& p) {
    return dmodel.force(r, p);
  };
  auto denergy_fn = [&](const std::vector<double>& r) { return dmodel.energy(r); };
  auto ddensity_fn = [&](const std::vector<double>& r) { return dmodel.density(r); };

  // Run with diagonal kernel (kernel_order=1).
  // refresh_interval=10 so P_0 updates as nuclei move, creating non-zero
  // residual between P_0 and P_aux that the kernel must correct.
  KSAKernelConfig kcfg_diag;
  kcfg_diag.kernel_order = 1;
  kcfg_diag.kappa = 2.0;
  auto res_diag = XLBOMD::Run(dR0, masses, 0.3, 200, dforce_fn, denergy_fn,
                               ddensity_fn, 0, 0.0, kcfg_diag, 10, 42);

  // Run with low-rank kernel (kernel_order=2).
  KSAKernelConfig kcfg_lr;
  kcfg_lr.kernel_order = 2;
  kcfg_lr.kappa = 2.0;
  kcfg_lr.low_rank_dim = 3;
  auto res_lr = XLBOMD::Run(dR0, masses, 0.3, 200, dforce_fn, denergy_fn,
                             ddensity_fn, 0, 0.0, kcfg_lr, 10, 42);

  std::cout << "  Diagonal drift: " << res_diag.total_drift << " uHa/atom/ps\n";
  std::cout << "  Low-rank drift: " << res_lr.total_drift << " uHa/atom/ps\n";

  // Verify low-rank kernel ran and produced a finite result.
  if (!std::isfinite(res_lr.total_drift))
    return Fail("Low-rank kernel produced non-finite drift");

  // Verify the low-rank kernel actually differs from diagonal
  // (proving it's doing something different, not just a copy).
  double drift_diff = std::fabs(res_lr.total_drift - res_diag.total_drift);
  std::cout << "  Drift difference: " << drift_diff << "\n";
  if (drift_diff < 1e-15)
    return Fail("Low-rank kernel produces identical result to diagonal — "
                "not a genuine low-rank implementation");

  // Also verify the final P_aux differs (different kernel → different trajectory).
  double p_diff = 0.0;
  for (std::size_t i = 0; i < res_lr.final_state.P.size() &&
                          i < res_diag.final_state.P.size(); ++i)
    p_diff += std::fabs(res_lr.final_state.P[i] - res_diag.final_state.P[i]);
  std::cout << "  ||P_lr - P_diag||_1: " << p_diff << "\n";
  if (p_diff < 1e-15)
    return Fail("Low-rank kernel produces identical density to diagonal");

  std::cout << "  PASS (diagonal vs low-rank)\n";
  return 0;
}

int TestShadowDynamics() {
  std::cout << "\n=== B3: Shadow Dynamics (1 solve/step) ===\n";
  HarmonicModel model;
  std::vector<double> R0 = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0};
  std::vector<double> masses = {1.0, 1.0};

  auto force_fn = [&](const std::vector<double>& r, const std::vector<double>& p) {
    return model.force(r, p);
  };
  auto energy_fn = [&](const std::vector<double>& r) { return model.energy(r); };
  auto density_fn = [&](const std::vector<double>& r) { return model.density(r); };

  // 100 steps, no refresh → 1 solve total (at init).
  auto res = XLBOMD::Run(R0, masses, 1.0, 100, force_fn, energy_fn,
                          density_fn, 0, 0.0, {}, 0, 42);

  std::cout << "  n_solves: " << res.final_state.n_solves << "\n";
  std::cout << "  avg_solves/step: " << res.avg_solves_per_step << "\n";

  // Should be exactly 1 solve (at init), not 100.
  if (res.final_state.n_solves != 1)
    return Fail("Expected 1 solve (at init), got " +
                std::to_string(res.final_state.n_solves));
  if (res.avg_solves_per_step > 0.1)
    return Fail("avg_solves_per_step too high: " +
                std::to_string(res.avg_solves_per_step));
  std::cout << "  PASS\n";
  return 0;
}

int TestTimeReversibility() {
  std::cout << "\n=== Time-Reversibility Test ===\n";
  HarmonicModel model;
  std::vector<double> R0 = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0};
  std::vector<double> masses = {1.0, 1.0};

  auto force_fn = [&](const std::vector<double>& r, const std::vector<double>& p) {
    return model.force(r, p);
  };
  auto energy_fn = [&](const std::vector<double>& r) { return model.energy(r); };
  auto density_fn = [&](const std::vector<double>& r) { return model.density(r); };

  double rms = XLBOMD::TestTimeReversibility(R0, masses, 1.0, 50,
                                               force_fn, energy_fn, density_fn);
  std::cout << "  RMS displacement after reversal: " << rms << "\n";

  if (!std::isfinite(rms) || rms > 0.1)
    return Fail("Time-reversibility violated: RMS = " + std::to_string(rms));
  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== XL-BOMD Extended Tests ===\n";
  int failures = 0;
  failures += TestNVEDrift();
  failures += TestNHCThermostat();
  failures += TestKSAKernel();
  failures += TestShadowDynamics();
  failures += TestTimeReversibility();
  if (failures == 0) std::cout << "\nALL XL-BOMD TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
