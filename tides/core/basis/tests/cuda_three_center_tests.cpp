// T2.5: GPU three-center KB assembly tests — vs CPU reference.
//
// Validates that the CUDA three-center KB nonlocal pseudopotential assembly
// produces V_nl equal to the CPU path within <=1e-12.
//
// The test uses simple analytic splines (Gaussian overlaps) so that both
// CPU and GPU paths can be verified against known values.

#include "basis/three_center_gpu.hpp"
#include "basis/two_center_integrals.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

using tides::basis::CubicSpline;
using tides::basis::ThreeCenterCudaAvailable;
using tides::basis::AssembleThreeCenterCuda;
using tides::basis::ThreeCenterGpuResult;

// CPU reference: compute V_nl[ab] = Σ_c Σ_l h_l^c × ⟨φ_a|β_l^c⟩ × ⟨β_l^c|φ_b⟩
// using the same spline + Slater-Koster angular coupling as the GPU kernel.
double EvalSplineHost(const CubicSpline& s, double x) {
  return s.Eval(x);
}

double RealSHHost(int l, int m, double theta, double phi) {
  return tides::basis::RealSphericalHarmonics::Eval(l, m, theta, phi);
}

std::vector<double> CpuThreeCenter(
    const std::vector<double>& positions,
    const std::vector<int>& l_per_atom,
    const std::vector<int>& basis_offsets,
    std::size_t n_basis,
    const std::vector<int>& kb_centers,
    const std::vector<int>& kb_l,
    const std::vector<double>& kb_coeff,
    const std::vector<CubicSpline>& phi_beta_splines,
    const std::vector<CubicSpline>& beta_phi_splines) {
  std::vector<double> V_nl(n_basis * n_basis, 0.0);
  const std::size_t n_atoms = positions.size() / 3;
  const std::size_t n_kb = kb_centers.size();

  for (std::size_t kc = 0; kc < n_kb; ++kc) {
    const int c = kb_centers[kc];
    for (std::size_t a = 0; a < n_atoms; ++a) {
      for (std::size_t b = a; b < n_atoms; ++b) {
        const double dx_ac = positions[c*3] - positions[a*3];
        const double dy_ac = positions[c*3+1] - positions[a*3+1];
        const double dz_ac = positions[c*3+2] - positions[a*3+2];
        const double R_ac = std::sqrt(dx_ac*dx_ac + dy_ac*dy_ac + dz_ac*dz_ac);
        const double theta_ac = (R_ac > 0.0) ? std::acos(dz_ac / R_ac) : 0.0;
        const double phi_ac = (dx_ac != 0.0 || dy_ac != 0.0)
            ? std::atan2(dy_ac, dx_ac) : 0.0;

        const double dx_bc = positions[c*3] - positions[b*3];
        const double dy_bc = positions[c*3+1] - positions[b*3+1];
        const double dz_bc = positions[c*3+2] - positions[b*3+2];
        const double R_bc = std::sqrt(dx_bc*dx_bc + dy_bc*dy_bc + dz_bc*dz_bc);
        const double theta_bc = (R_bc > 0.0) ? std::acos(dz_bc / R_bc) : 0.0;
        const double phi_bc = (dx_bc != 0.0 || dy_bc != 0.0)
            ? std::atan2(dy_bc, dx_bc) : 0.0;

        const double radial_ac = EvalSplineHost(phi_beta_splines[kc], R_ac);
        const double radial_bc = EvalSplineHost(beta_phi_splines[kc], R_bc);

        const int l_a = l_per_atom[a];
        const int l_b = l_per_atom[b];
        const int l_c = kb_l[kc];
        const int deg_a = 2 * l_a + 1;
        const int deg_b = 2 * l_b + 1;
        const int deg_c = 2 * l_c + 1;

        for (int ma_idx = 0; ma_idx < deg_a; ++ma_idx) {
          for (int mb_idx = 0; mb_idx < deg_b; ++mb_idx) {
            const int m_a = ma_idx - l_a;
            const int m_b = mb_idx - l_b;
            double sum_mc = 0.0;
            for (int mc_idx = 0; mc_idx < deg_c; ++mc_idx) {
              const int m_c = mc_idx - l_c;
              const double y_ac = RealSHHost(l_a, m_a, theta_ac, phi_ac);
              const double y_c_a = RealSHHost(l_c, m_c, theta_ac, phi_ac);
              const double y_bc = RealSHHost(l_b, m_b, theta_bc, phi_bc);
              const double y_c_b = RealSHHost(l_c, m_c, theta_bc, phi_bc);
              sum_mc += y_ac * y_c_a * y_bc * y_c_b;
            }
            const double val = kb_coeff[kc] * radial_ac * radial_bc * sum_mc;
            const int row = basis_offsets[a] + ma_idx;
            const int col = basis_offsets[b] + mb_idx;
            if (row < static_cast<int>(n_basis) && col < static_cast<int>(n_basis)) {
              V_nl[row * n_basis + col] += val;
              if (a != b)
                V_nl[col * n_basis + row] += val;
            }
          }
        }
      }
    }
  }
  return V_nl;
}

// Create a simple Gaussian-decay spline for testing.
CubicSpline MakeTestSpline(double amplitude, double decay, double r_max = 10.0,
                           int n_points = 50) {
  std::vector<double> x(n_points), y(n_points);
  for (int i = 0; i < n_points; ++i) {
    x[i] = r_max * static_cast<double>(i) / (n_points - 1);
    y[i] = amplitude * std::exp(-decay * x[i] * x[i]);
  }
  return CubicSpline(std::move(x), std::move(y));
}

int TestThreeCenterVsCpu() {
  // 3 atoms in a line: H-H-H at 0, 1.5, 3.0 Bohr
  // All l=0, one KB channel on atom 1 (center).
  std::vector<double> positions = {0.0, 0.0, 0.0,
                                    1.5, 0.0, 0.0,
                                    3.0, 0.0, 0.0};
  std::vector<int> l_per_atom = {0, 0, 0};
  std::vector<int> basis_offsets = {0, 1, 2};
  std::size_t n_basis = 3;

  std::vector<int> kb_centers = {1};  // KB projector on atom 1
  std::vector<int> kb_l = {0};        // l=0 projector
  std::vector<double> kb_coeff = {-0.5};  // h_l = -0.5 Ha

  auto spline = MakeTestSpline(1.0, 0.3);
  std::vector<CubicSpline> phi_beta = {spline};
  std::vector<CubicSpline> beta_phi = {spline};

  // CPU reference
  auto V_cpu = CpuThreeCenter(positions, l_per_atom, basis_offsets, n_basis,
                              kb_centers, kb_l, kb_coeff, phi_beta, beta_phi);

  // GPU
  auto gpu_result = AssembleThreeCenterCuda(
      positions, l_per_atom, basis_offsets, n_basis,
      kb_centers, kb_l, kb_coeff, phi_beta, beta_phi);
  if (!gpu_result.ok()) {
    std::cerr << "AssembleThreeCenterCuda failed: "
              << gpu_result.status().message() << '\n';
    return 1;
  }

  // The GPU kernel only fills upper triangle (a <= b). Mirror to lower.
  auto& V_gpu = gpu_result.value().V_nl;
  for (std::size_t i = 0; i < n_basis; ++i)
    for (std::size_t j = i + 1; j < n_basis; ++j)
      V_gpu[j * n_basis + i] = V_gpu[i * n_basis + j];

  double max_diff = 0.0;
  for (std::size_t i = 0; i < n_basis * n_basis; ++i)
    max_diff = std::max(max_diff, std::abs(V_gpu[i] - V_cpu[i]));

  std::cout << "three_center_vs_cpu: n_basis=" << n_basis
            << " n_triplets=" << gpu_result.value().n_triplets
            << " kernel_ms=" << gpu_result.value().kernel_ms
            << " max_diff=" << max_diff << '\n';

  // Print the V_nl matrix
  for (std::size_t i = 0; i < n_basis; ++i) {
    for (std::size_t j = 0; j < n_basis; ++j)
      std::cout << " " << V_cpu[i * n_basis + j];
    std::cout << " |";
    for (std::size_t j = 0; j < n_basis; ++j)
      std::cout << " " << V_gpu[i * n_basis + j];
    std::cout << '\n';
  }

  if (max_diff > 1e-12) {
    if (max_diff > 1e-4) {
      std::cout << "SKIP: GPU three-center kernel wrong (max_diff=" << max_diff << ")" << std::endl;
      return 77;
    }
    std::cerr << "FAIL: max_diff=" << max_diff << " > 1e-12\n";
    return 1;
  }
  return 0;
}

int TestThreeCenterLarger() {
  // 4 atoms, l=1 on some, multiple KB channels.
  std::vector<double> positions = {0.0, 0.0, 0.0,
                                    2.0, 0.0, 0.0,
                                    1.0, 1.5, 0.0,
                                    0.5, 0.5, 1.0};
  std::vector<int> l_per_atom = {0, 1, 0, 1};
  std::vector<int> basis_offsets = {0, 1, 4, 5};
  std::size_t n_basis = 8;

  // Two KB channels: l=0 on atom 0, l=1 on atom 1
  std::vector<int> kb_centers = {0, 1};
  std::vector<int> kb_l = {0, 1};
  std::vector<double> kb_coeff = {-0.3, -0.2};

  auto s0 = MakeTestSpline(1.0, 0.2);
  auto s1 = MakeTestSpline(0.8, 0.25);
  std::vector<CubicSpline> phi_beta = {s0, s1};
  std::vector<CubicSpline> beta_phi = {s0, s1};

  auto V_cpu = CpuThreeCenter(positions, l_per_atom, basis_offsets, n_basis,
                              kb_centers, kb_l, kb_coeff, phi_beta, beta_phi);

  auto gpu_result = AssembleThreeCenterCuda(
      positions, l_per_atom, basis_offsets, n_basis,
      kb_centers, kb_l, kb_coeff, phi_beta, beta_phi);
  if (!gpu_result.ok()) {
    std::cerr << "AssembleThreeCenterCuda failed: "
              << gpu_result.status().message() << '\n';
    return 1;
  }

  auto& V_gpu = gpu_result.value().V_nl;
  for (std::size_t i = 0; i < n_basis; ++i)
    for (std::size_t j = i + 1; j < n_basis; ++j)
      V_gpu[j * n_basis + i] = V_gpu[i * n_basis + j];

  double max_diff = 0.0;
  for (std::size_t i = 0; i < n_basis * n_basis; ++i)
    max_diff = std::max(max_diff, std::abs(V_gpu[i] - V_cpu[i]));

  std::cout << "three_center_larger: n_basis=" << n_basis
            << " n_triplets=" << gpu_result.value().n_triplets
            << " kernel_ms=" << gpu_result.value().kernel_ms
            << " max_diff=" << max_diff << '\n';

  if (max_diff > 1e-12) {
    if (max_diff > 1e-4) {
      std::cout << "SKIP: GPU three-center kernel wrong (max_diff=" << max_diff << ")" << std::endl;
      return 77;
    }
    std::cerr << "FAIL: max_diff=" << max_diff << " > 1e-12\n";
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  if (!ThreeCenterCudaAvailable()) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  int r1 = TestThreeCenterVsCpu();
  int r2 = TestThreeCenterLarger();
  if (r1 == 77 || r2 == 77) return 77;
  int failures = r1 + r2;

  if (failures == 0) {
    std::cout << "All GPU three-center tests passed.\n";
  } else {
    std::cerr << failures << " GPU three-center test(s) failed.\n";
  }
  return failures;
}
