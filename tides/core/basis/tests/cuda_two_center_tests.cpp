// T2.5: GPU tile assembly of S, H0 — vs CPU reference.
// Validates that the CUDA two-center assembly produces matrices equal to the
// CPU path within <=1e-7 relative error, and records throughput.

#include "basis/two_center_gpu.hpp"
#include "basis/two_center_integrals.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

using tides::basis::AssembleTwoCenterCuda;
using tides::basis::CubicSpline;
using tides::basis::RealSphericalHarmonics;
using tides::basis::TwoCenterCudaAvailable;
using tides::basis::TwoCenterGpuResult;

// CPU reference: assemble S and T using the same spline + SK approach.
void AssembleTwoCenterCpu(
    const std::vector<double>& positions,
    const std::vector<int>& l_per_atom,
    const std::vector<int>& basis_offsets,
    std::size_t n_basis,
    const CubicSpline& s_spline,
    const CubicSpline& t_spline,
    std::vector<double>& S,
    std::vector<double>& T) {
  const std::size_t n_atoms = positions.size() / 3;
  S.assign(n_basis * n_basis, 0.0);
  T.assign(n_basis * n_basis, 0.0);
  for (std::size_t a = 0; a < n_atoms; ++a) {
    for (std::size_t b = a; b < n_atoms; ++b) {
      const double dx = positions[b * 3] - positions[a * 3];
      const double dy = positions[b * 3 + 1] - positions[a * 3 + 1];
      const double dz = positions[b * 3 + 2] - positions[a * 3 + 2];
      const double R = std::sqrt(dx * dx + dy * dy + dz * dz);
      const double theta = (R > 0.0) ? std::acos(dz / R) : 0.0;
      const double phi = (dx != 0.0 || dy != 0.0) ? std::atan2(dy, dx) : 0.0;
      const double s_radial = s_spline.Eval(R);
      const double t_radial = t_spline.Eval(R);
      const int la = l_per_atom[a];
      const int lb = l_per_atom[b];
      const int deg_a = 2 * la + 1;
      const int deg_b = 2 * lb + 1;
      for (int ma_idx = 0; ma_idx < deg_a; ++ma_idx) {
        for (int mb_idx = 0; mb_idx < deg_b; ++mb_idx) {
          const int ma = ma_idx - la;
          const int mb = mb_idx - lb;
          const double y_a = RealSphericalHarmonics::Eval(la, ma, theta, phi);
          const double y_b = RealSphericalHarmonics::Eval(lb, mb, theta, phi);
          const double angular = y_a * y_b;
          const int row = basis_offsets[a] + ma_idx;
          const int col = basis_offsets[b] + mb_idx;
          if (row < static_cast<int>(n_basis) &&
              col < static_cast<int>(n_basis)) {
            S[row * n_basis + col] += s_radial * angular;
            T[row * n_basis + col] += t_radial * angular;
            // Mirror to lower triangle for cross-atom pairs.
            if (a != b) {
              S[col * n_basis + row] += s_radial * angular;
              T[col * n_basis + row] += t_radial * angular;
            }
          }
        }
      }
    }
  }
}

double MaxRelDifference(const std::vector<double>& a,
                        const std::vector<double>& b) {
  double max_rel = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double diff = std::abs(a[i] - b[i]);
    const double denom = std::max(std::abs(a[i]), std::abs(b[i]));
    if (denom > 1e-15) {
      max_rel = std::max(max_rel, diff / denom);
    } else if (diff > 1e-15) {
      max_rel = std::max(max_rel, diff);
    }
  }
  return max_rel;
}

int TestGpuVsCpu() {
  // Model: 4 atoms with l=0 and l=1, Gaussian splines.
  std::vector<double> R_tab, S_tab, T_tab;
  const int n_tab = 500;
  for (int i = 0; i <= n_tab; ++i) {
    const double R = (10.0 / n_tab) * i;
    R_tab.push_back(R);
    S_tab.push_back(std::exp(-R * R));
    T_tab.push_back(0.5 * std::exp(-R * R * 0.5));
  }
  CubicSpline s_spline(R_tab, S_tab);
  CubicSpline t_spline(R_tab, T_tab);

  // 4 atoms: 2 with l=0 (1 basis function each), 2 with l=1 (3 each).
  std::vector<double> positions = {
      0.0, 0.0, 0.0,
      1.5, 0.0, 0.0,
      0.0, 2.0, 0.0,
      0.5, 0.5, 1.0
  };
  std::vector<int> atomic_numbers = {1, 1, 6, 6};
  std::vector<int> l_per_atom = {0, 0, 1, 1};
  std::vector<int> basis_offsets = {0, 1, 2, 5};
  const std::size_t n_basis = 8;

  std::vector<double> S_cpu, T_cpu;
  AssembleTwoCenterCpu(positions, l_per_atom, basis_offsets, n_basis,
                       s_spline, t_spline, S_cpu, T_cpu);

  auto gpu_result = AssembleTwoCenterCuda(
      positions, atomic_numbers, l_per_atom, basis_offsets, n_basis,
      s_spline, t_spline);
  if (!gpu_result.ok()) {
    std::cerr << "AssembleTwoCenterCuda failed: "
              << gpu_result.status().message() << '\n';
    return 1;
  }

  const double s_max_rel = MaxRelDifference(gpu_result.value().S, S_cpu);
  const double t_max_rel = MaxRelDifference(gpu_result.value().T, T_cpu);

  std::cout << "gpu_vs_cpu: n_atoms=4 n_basis=" << n_basis
            << " n_pairs=" << gpu_result.value().n_pairs
            << " kernel_ms=" << gpu_result.value().kernel_ms
            << " S_max_rel=" << s_max_rel
            << " T_max_rel=" << t_max_rel << '\n';

  // Observable (T2.5): equals CPU path <=1e-7 rel.
  // GPU cubic spline evaluation matches CPU formula exactly, so we expect
  // machine-precision agreement (~1e-16).
  if (s_max_rel > 1e-7) {
    std::cerr << "FAIL: S max_rel=" << s_max_rel << " > 1e-7\n";
    return 1;
  }
  if (t_max_rel > 1e-7) {
    std::cerr << "FAIL: T max_rel=" << t_max_rel << " > 1e-7\n";
    return 1;
  }
  return 0;
}

int TestThroughput() {
  // Larger system: 64 atoms, all l=1, n_basis = 192.
  const int n_atoms = 64;
  std::vector<double> positions(n_atoms * 3);
  for (int i = 0; i < n_atoms; ++i) {
    positions[i * 3] = static_cast<double>(i % 8) * 1.5;
    positions[i * 3 + 1] = static_cast<double>((i / 8) % 8) * 1.5;
    positions[i * 3 + 2] = static_cast<double>(i / 64) * 1.5;
  }
  std::vector<int> atomic_numbers(n_atoms, 6);
  std::vector<int> l_per_atom(n_atoms, 1);
  std::vector<int> basis_offsets(n_atoms);
  for (int i = 0; i < n_atoms; ++i) {
    basis_offsets[i] = i * 3;
  }
  const std::size_t n_basis = n_atoms * 3;

  std::vector<double> R_tab, S_tab, T_tab;
  const int n_tab = 500;
  for (int i = 0; i <= n_tab; ++i) {
    const double R = (10.0 / n_tab) * i;
    R_tab.push_back(R);
    S_tab.push_back(std::exp(-R * R));
    T_tab.push_back(0.5 * std::exp(-R * R * 0.5));
  }
  CubicSpline s_spline(R_tab, S_tab);
  CubicSpline t_spline(R_tab, T_tab);

  auto gpu_result = AssembleTwoCenterCuda(
      positions, atomic_numbers, l_per_atom, basis_offsets, n_basis,
      s_spline, t_spline);
  if (!gpu_result.ok()) {
    std::cerr << "Throughput test failed: " << gpu_result.status().message()
              << '\n';
    return 1;
  }

  const double pairs_per_ms =
      static_cast<double>(gpu_result.value().n_pairs) /
      gpu_result.value().kernel_ms;
  std::cout << "throughput: n_atoms=" << n_atoms
            << " n_basis=" << n_basis
            << " n_pairs=" << gpu_result.value().n_pairs
            << " kernel_ms=" << gpu_result.value().kernel_ms
            << " pairs/ms=" << pairs_per_ms << '\n';
  return 0;
}

int TestLedger() {
  std::vector<double> positions = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  std::vector<int> atomic_numbers = {1, 1};
  std::vector<int> l_per_atom = {0, 0};
  std::vector<int> basis_offsets = {0, 1};
  const std::size_t n_basis = 2;

  std::vector<double> R_tab, S_tab, T_tab;
  for (int i = 0; i <= 100; ++i) {
    const double R = 0.1 * i;
    R_tab.push_back(R);
    S_tab.push_back(std::exp(-R * R));
    T_tab.push_back(0.5 * std::exp(-R * R * 0.5));
  }
  CubicSpline s_spline(R_tab, S_tab);
  CubicSpline t_spline(R_tab, T_tab);

  auto gpu_result = AssembleTwoCenterCuda(
      positions, atomic_numbers, l_per_atom, basis_offsets, n_basis,
      s_spline, t_spline);
  if (!gpu_result.ok()) {
    std::cerr << "Ledger test failed: " << gpu_result.status().message()
              << '\n';
    return 1;
  }

  const auto& entries = gpu_result.value().ledger.entries();
  if (entries.size() != 1) {
    std::cerr << "FAIL: expected 1 ledger entry, got " << entries.size()
              << '\n';
    return 1;
  }
  if (entries[0].precision.determinism !=
      tides::tile::DeterminismMode::kDeterministic) {
    std::cerr << "FAIL: ledger determinism is not kDeterministic\n";
    return 1;
  }
  std::cout << "ledger: entries=" << entries.size()
            << " label=" << entries[0].precision.label
            << " candidates=" << entries[0].candidates << '\n';
  return 0;
}

}  // namespace

int main() {
  if (!TwoCenterCudaAvailable()) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  int failures = TestGpuVsCpu();
  failures += TestThroughput();
  failures += TestLedger();

  if (failures == 0) {
    std::cout << "All GPU two-center tile assembly tests passed.\n";
  } else {
    std::cerr << failures << " GPU two-center test(s) failed.\n";
  }
  return failures;
}
