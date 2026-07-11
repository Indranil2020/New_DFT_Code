#include "grid/xc/functionals/gga_pbe.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifndef TIDES_XC_ADJOINT
#error "TIDES_XC_ADJOINT must be configured from verification/tolerances.yaml"
#endif

namespace {

using tides::grid::xc::GgaPbeStandard;

constexpr std::size_t kBasis = 3;
constexpr std::size_t kPoints = 23;
constexpr double kAdjointTolerance = TIDES_XC_ADJOINT;

struct GaussianGrid {
  std::array<double, kBasis * kPoints> phi{};
  std::array<double, 3 * kBasis * kPoints> grad_phi{};
  std::array<double, kPoints> weights{};
};

std::size_t PhiIndex(std::size_t basis, std::size_t point) {
  return basis * kPoints + point;
}

std::size_t GradIndex(std::size_t component, std::size_t basis,
                      std::size_t point) {
  return (component * kBasis + basis) * kPoints + point;
}

GaussianGrid MakeGaussianGrid() {
  constexpr std::array<std::array<double, 3>, kBasis> centers = {{
      {{-0.45, 0.10, -0.20}},
      {{0.30, -0.25, 0.15}},
      {{0.10, 0.35, 0.30}},
  }};
  constexpr std::array<double, kBasis> alpha = {{0.70, 1.10, 0.85}};
  GaussianGrid grid;
  for (std::size_t point = 0; point < kPoints; ++point) {
    const double t = -1.1 + 2.2 * static_cast<double>(point) /
        static_cast<double>(kPoints - 1);
    const std::array<double, 3> coordinate = {
        t, 0.35 * std::sin(1.7 * t), 0.25 * std::cos(0.9 * t)};
    grid.weights[point] = 0.08 + 0.002 * static_cast<double>(point);
    for (std::size_t basis = 0; basis < kBasis; ++basis) {
      double radius_squared = 0.0;
      std::array<double, 3> delta{};
      for (std::size_t component = 0; component < 3; ++component) {
        delta[component] = coordinate[component] - centers[basis][component];
        radius_squared += delta[component] * delta[component];
      }
      const double phi = std::exp(-alpha[basis] * radius_squared);
      grid.phi[PhiIndex(basis, point)] = phi;
      for (std::size_t component = 0; component < 3; ++component) {
        grid.grad_phi[GradIndex(component, basis, point)] =
            -2.0 * alpha[basis] * delta[component] * phi;
      }
    }
  }
  return grid;
}

struct RhoGradient {
  std::array<double, kPoints> rho{};
  std::array<double, 3 * kPoints> gradient{};
};

RhoGradient BuildRhoGradient(const std::array<double, kBasis * kBasis>& density,
                             const GaussianGrid& grid) {
  RhoGradient output;
  for (std::size_t point = 0; point < kPoints; ++point) {
    for (std::size_t mu = 0; mu < kBasis; ++mu) {
      const double phi_mu = grid.phi[PhiIndex(mu, point)];
      for (std::size_t nu = 0; nu < kBasis; ++nu) {
        const double coefficient = density[mu * kBasis + nu];
        const double phi_nu = grid.phi[PhiIndex(nu, point)];
        output.rho[point] += coefficient * phi_mu * phi_nu;
        for (std::size_t component = 0; component < 3; ++component) {
          output.gradient[component * kPoints + point] += coefficient *
              (grid.grad_phi[GradIndex(component, mu, point)] * phi_nu +
               phi_mu * grid.grad_phi[GradIndex(component, nu, point)]);
        }
      }
    }
  }
  return output;
}

double PbeEnergy(const std::array<double, kBasis * kBasis>& density,
                 const GaussianGrid& grid) {
  const RhoGradient rho_gradient = BuildRhoGradient(density, grid);
  double energy = 0.0;
  for (std::size_t point = 0; point < kPoints; ++point) {
    const double gx = rho_gradient.gradient[point];
    const double gy = rho_gradient.gradient[kPoints + point];
    const double gz = rho_gradient.gradient[2 * kPoints + point];
    const auto evaluation =
        GgaPbeStandard::Eval(rho_gradient.rho[point], gx * gx + gy * gy + gz * gz);
    energy += grid.weights[point] * rho_gradient.rho[point] * evaluation.eps;
  }
  return energy;
}

std::array<double, kBasis * kBasis> BuildGgaAdjoint(
    const std::array<double, kBasis * kBasis>& density,
    const GaussianGrid& grid) {
  const RhoGradient rho_gradient = BuildRhoGradient(density, grid);
  std::array<double, kBasis * kBasis> vmat{};
  for (std::size_t point = 0; point < kPoints; ++point) {
    const double gx = rho_gradient.gradient[point];
    const double gy = rho_gradient.gradient[kPoints + point];
    const double gz = rho_gradient.gradient[2 * kPoints + point];
    const auto evaluation =
        GgaPbeStandard::Eval(rho_gradient.rho[point], gx * gx + gy * gy + gz * gz);
    const double wv_rho = grid.weights[point] * evaluation.vrho;
    const std::array<double, 3> wv_gradient = {
        2.0 * grid.weights[point] * evaluation.vsigma * gx,
        2.0 * grid.weights[point] * evaluation.vsigma * gy,
        2.0 * grid.weights[point] * evaluation.vsigma * gz,
    };
    for (std::size_t mu = 0; mu < kBasis; ++mu) {
      const double phi_mu = grid.phi[PhiIndex(mu, point)];
      for (std::size_t nu = 0; nu < kBasis; ++nu) {
        const double phi_nu = grid.phi[PhiIndex(nu, point)];
        double value = wv_rho * phi_mu * phi_nu;
        for (std::size_t component = 0; component < 3; ++component) {
          value += wv_gradient[component] *
              (grid.grad_phi[GradIndex(component, mu, point)] * phi_nu +
               phi_mu * grid.grad_phi[GradIndex(component, nu, point)]);
        }
        vmat[mu * kBasis + nu] += value;
      }
    }
  }
  return vmat;
}

double TraceProduct(const std::array<double, kBasis * kBasis>& lhs,
                    const std::array<double, kBasis * kBasis>& rhs) {
  double result = 0.0;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    result += lhs[index] * rhs[index];
  }
  return result;
}

int Fail(const char* message) {
  std::cerr << "xc_rung1_adjoint: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  const GaussianGrid grid = MakeGaussianGrid();
  const std::array<double, kBasis * kBasis> density = {{
      1.20, 0.12, -0.04,
      0.12, 0.95, 0.08,
      -0.04, 0.08, 0.75,
  }};
  const auto vmat = BuildGgaAdjoint(density, grid);
  std::mt19937_64 random(0x58435f41444a4f49ULL);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  constexpr double kStep = 1.0e-5;
  double max_error = 0.0;
  for (int trial = 0; trial < 100; ++trial) {
    std::array<double, kBasis * kBasis> delta{};
    for (std::size_t mu = 0; mu < kBasis; ++mu) {
      for (std::size_t nu = mu; nu < kBasis; ++nu) {
        const double value = distribution(random);
        delta[mu * kBasis + nu] = value;
        delta[nu * kBasis + mu] = value;
      }
    }
    const double norm = std::sqrt(TraceProduct(delta, delta));
    for (double& value : delta) value /= norm;
    auto plus = density;
    auto minus = density;
    for (std::size_t index = 0; index < density.size(); ++index) {
      plus[index] += kStep * delta[index];
      minus[index] -= kStep * delta[index];
    }
    const double finite_difference =
        (PbeEnergy(plus, grid) - PbeEnergy(minus, grid)) / (2.0 * kStep);
    const double adjoint = TraceProduct(vmat, delta);
    max_error = std::max(max_error, std::abs(finite_difference - adjoint));
  }
  std::cout << std::setprecision(17)
            << "xc_rung1_adjoint: trials=100 max_abs_error=" << max_error
            << " step=" << kStep << " tolerance=" << kAdjointTolerance << '\n';
  if (max_error > kAdjointTolerance) {
    return Fail("GGA adjoint does not match the PBE energy finite difference");
  }
  return 0;
}
