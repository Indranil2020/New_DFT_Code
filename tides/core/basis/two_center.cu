// T2.5: GPU tile assembly of S, H0 via batched spline evaluation + Slater-Koster
// angular coupling.
//
// The kernel takes a list of atom pairs, their spline tables (tabulated radial
// integrals vs R), and assembles the overlap (S) and kinetic (T) matrix blocks
// on GPU. The angular coupling uses real spherical harmonics evaluated at the
// interatomic axis direction.
//
// Observable (T2.5): equals CPU path <=1e-7 rel; throughput recorded.

#include "basis/two_center_integrals.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/status.hpp"
#include "tile/precision.hpp"

namespace tides::basis {
namespace {

constexpr int kBlockDim = 16;

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) {
    return Status::Ok();
  }
  return Status::IoError(std::string(context) + ": " +
                         cudaGetErrorString(error));
}

// Device-side cubic spline evaluation: matches the CPU CubicSpline::Eval
// formula exactly (natural cubic spline with second derivatives d2y).
// The spline table is stored as (x, y, d2y) triples in flat device arrays.
__device__ double EvalSplineDevice(const double* x_tab,
                                   const double* y_tab,
                                   const double* d2y_tab,
                                   int n_tab, double xq) {
  if (n_tab == 0) return 0.0;
  if (n_tab == 1) return y_tab[0];
  if (xq <= x_tab[0]) {
    const double dx = x_tab[1] - x_tab[0];
    if (dx == 0.0) return y_tab[0];
    const double slope = (y_tab[1] - y_tab[0]) / dx;
    return y_tab[0] + slope * (xq - x_tab[0]);
  }
  if (xq >= x_tab[n_tab - 1]) {
    const double dx = x_tab[n_tab - 1] - x_tab[n_tab - 2];
    if (dx == 0.0) return y_tab[n_tab - 1];
    const double slope = (y_tab[n_tab - 1] - y_tab[n_tab - 2]) / dx;
    return y_tab[n_tab - 1] + slope * (xq - x_tab[n_tab - 1]);
  }
  // Binary search for bracketing interval.
  int lo = 0, hi = n_tab - 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (x_tab[mid] <= xq) lo = mid; else hi = mid;
  }
  const double h = x_tab[lo + 1] - x_tab[lo];
  if (h == 0.0) return y_tab[lo];
  const double a = (x_tab[lo + 1] - xq) / h;
  const double b = (xq - x_tab[lo]) / h;
  return a * y_tab[lo] + b * y_tab[lo + 1] +
         ((a * a * a - a) * d2y_tab[lo] +
          (b * b * b - b) * d2y_tab[lo + 1]) * (h * h) / 6.0;
}

// Device-side associated Legendre P_l^m(x) without Condon-Shortley phase.
__device__ double AssociatedLegendreDevice(int l, int m, double x) {
  if (m > l) return 0.0;
  double pmm = 1.0;
  if (m > 0) {
    const double somx2 = sqrt((1.0 - x) * (1.0 + x));
    double fact = 1.0;
    for (int i = 1; i <= m; ++i) {
      pmm *= fact * somx2;
      fact += 2.0;
    }
  }
  if (l == m) return pmm;
  double pmmp1 = x * (2.0 * m + 1.0) * pmm;
  if (l == m + 1) return pmmp1;
  double pll = 0.0;
  for (int ll = m + 2; ll <= l; ++ll) {
    pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
}

// Device-side normalization for real spherical harmonics.
__device__ double RealSHNormDevice(int l, int m) {
  double denom = 1.0;
  for (int i = l - m + 1; i <= l + m; ++i) denom *= static_cast<double>(i);
  double n = sqrt((2.0 * l + 1.0) / (4.0 * M_PI) / denom);
  if (m != 0) n *= sqrt(2.0);
  return n;
}

// Device-side real spherical harmonic evaluation.
__device__ double RealSHDevice(int l, int m, double theta, double phi) {
  const double x = cos(theta);
  const int am = abs(m);
  const double plm = AssociatedLegendreDevice(l, am, x);
  const double n = RealSHNormDevice(l, am);
  if (m > 0) return n * plm * cos(static_cast<double>(m) * phi);
  if (m < 0) return n * plm * sin(static_cast<double>(am) * phi);
  return n * plm;
}

struct PairInput {
  double r_a[3];
  double r_b[3];
  int l_a;
  int l_b;
  int basis_offset_a;  // starting index in the output matrix for atom A
  int basis_offset_b;  // starting index in the output matrix for atom B
};

// Kernel: for each atom pair, evaluate the radial spline at R = |r_a - r_b|,
// apply the Slater-Koster angular coupling, and write the (2l_a+1) x (2l_b+1)
// block into the S and T matrices.
__global__ void AssembleTwoCenterKernel(
    const PairInput* pairs, int n_pairs,
    const double* s_x, const double* s_y, const double* s_d2y, int s_n,
    const double* t_x, const double* t_y, const double* t_d2y, int t_n,
    double* S_out, double* T_out, int n_basis) {
  const int pair_idx = static_cast<int>(blockIdx.x);
  if (pair_idx >= n_pairs) return;

  const PairInput p = pairs[pair_idx];
  const double dx = p.r_b[0] - p.r_a[0];
  const double dy = p.r_b[1] - p.r_a[1];
  const double dz = p.r_b[2] - p.r_a[2];
  const double R = sqrt(dx * dx + dy * dy + dz * dz);
  const double theta = (R > 0.0) ? acos(dz / R) : 0.0;
  const double phi = (dx != 0.0 || dy != 0.0) ? atan2(dy, dx) : 0.0;

  const double s_radial = EvalSplineDevice(s_x, s_y, s_d2y, s_n, R);
  const double t_radial = EvalSplineDevice(t_x, t_y, t_d2y, t_n, R);

  const int deg_a = 2 * p.l_a + 1;
  const int deg_b = 2 * p.l_b + 1;

  // Each thread handles one (m_a, m_b) element of the block.
  const int tid = static_cast<int>(threadIdx.x);
  for (int idx = tid; idx < deg_a * deg_b; idx += blockDim.x) {
    const int m_a_idx = idx / deg_b;
    const int m_b_idx = idx % deg_b;
    const int m_a = m_a_idx - p.l_a;
    const int m_b = m_b_idx - p.l_b;

    const double y_a = RealSHDevice(p.l_a, m_a, theta, phi);
    const double y_b = RealSHDevice(p.l_b, m_b, theta, phi);
    const double angular = y_a * y_b;

    const int row = p.basis_offset_a + m_a_idx;
    const int col = p.basis_offset_b + m_b_idx;
    if (row < n_basis && col < n_basis) {
      const int out_idx = row * n_basis + col;
      atomicAdd(&S_out[out_idx], s_radial * angular);
      atomicAdd(&T_out[out_idx], t_radial * angular);
    }
  }
}

template <typename T>
Status CopyToDevice(const std::vector<T>& host, T** device) {
  if (host.empty()) {
    *device = nullptr;
    return Status::Ok();
  }
  const std::size_t bytes = host.size() * sizeof(T);
  cudaError_t error = cudaMalloc(reinterpret_cast<void**>(device), bytes);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaMalloc");
  }
  error = cudaMemcpy(*device, host.data(), bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cudaFree(*device);
    *device = nullptr;
    return CudaStatus(error, "cudaMemcpy H2D");
  }
  return Status::Ok();
}

template <typename T>
void FreeDevice(T* ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

}  // namespace

struct TwoCenterGpuResult {
  std::vector<double> S;
  std::vector<double> T;
  double kernel_ms = 0.0;
  std::size_t n_pairs = 0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool TwoCenterCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

[[nodiscard]] Result<TwoCenterGpuResult> AssembleTwoCenterCuda(
    const std::vector<double>& positions,
    const std::vector<int>& atomic_numbers,
    const std::vector<int>& l_per_atom,
    const std::vector<int>& basis_offsets,
    std::size_t n_basis,
    const CubicSpline& s_spline,
    const CubicSpline& t_spline) {
  const std::size_t n_atoms = positions.size() / 3;
  if (n_atoms == 0 || n_basis == 0) {
    TwoCenterGpuResult empty;
    empty.S.assign(n_basis * n_basis, 0.0);
    empty.T.assign(n_basis * n_basis, 0.0);
    return empty;
  }
  if (l_per_atom.size() != n_atoms) {
    return Status::InvalidArgument("l_per_atom size mismatch");
  }
  if (basis_offsets.size() != n_atoms) {
    return Status::InvalidArgument("basis_offsets size mismatch");
  }
  if (!TwoCenterCudaAvailable()) {
    return Status::IoError("CUDA runtime not available for two-center assembly");
  }

  // Build pair list (all unique pairs including self-pairs for on-site).
  std::vector<PairInput> pairs;
  pairs.reserve(n_atoms * (n_atoms + 1) / 2);
  for (std::size_t a = 0; a < n_atoms; ++a) {
    for (std::size_t b = a; b < n_atoms; ++b) {
      PairInput p;
      p.r_a[0] = positions[a * 3];
      p.r_a[1] = positions[a * 3 + 1];
      p.r_a[2] = positions[a * 3 + 2];
      p.r_b[0] = positions[b * 3];
      p.r_b[1] = positions[b * 3 + 1];
      p.r_b[2] = positions[b * 3 + 2];
      p.l_a = l_per_atom[a];
      p.l_b = l_per_atom[b];
      p.basis_offset_a = basis_offsets[a];
      p.basis_offset_b = basis_offsets[b];
      pairs.push_back(p);
    }
  }

  // Prepare spline table data (flatten x, y, d2y arrays).
  const auto& s_x = s_spline.x();
  const auto& s_y = s_spline.y();
  const auto& s_d2y = s_spline.d2y();
  const auto& t_x = t_spline.x();
  const auto& t_y = t_spline.y();
  const auto& t_d2y = t_spline.d2y();
  const int s_n = static_cast<int>(s_x.size());
  const int t_n = static_cast<int>(t_x.size());

  // Allocate device memory.
  PairInput* d_pairs = nullptr;
  double* d_sx = nullptr;
  double* d_sy = nullptr;
  double* d_sd2y = nullptr;
  double* d_tx = nullptr;
  double* d_ty = nullptr;
  double* d_td2y = nullptr;
  double* d_S = nullptr;
  double* d_T = nullptr;

  auto cleanup = [&]() {
    FreeDevice(d_pairs);
    FreeDevice(d_sx);
    FreeDevice(d_sy);
    FreeDevice(d_sd2y);
    FreeDevice(d_tx);
    FreeDevice(d_ty);
    FreeDevice(d_td2y);
    FreeDevice(d_S);
    FreeDevice(d_T);
  };

  auto st = CopyToDevice(pairs, &d_pairs);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(s_x, &d_sx);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(s_y, &d_sy);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(s_d2y, &d_sd2y);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(t_x, &d_tx);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(t_y, &d_ty);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(t_d2y, &d_td2y);
  if (!st.ok()) { cleanup(); return st; }

  const std::size_t mat_bytes = n_basis * n_basis * sizeof(double);
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_S), mat_bytes);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc S"); }
  cudaMemset(d_S, 0, mat_bytes);
  err = cudaMalloc(reinterpret_cast<void**>(&d_T), mat_bytes);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc T"); }
  cudaMemset(d_T, 0, mat_bytes);

  // Launch kernel.
  const int n_pairs_int = static_cast<int>(pairs.size());
  const int threads_per_block = 128;
  auto kernel_start = std::chrono::steady_clock::now();
  AssembleTwoCenterKernel<<<n_pairs_int, threads_per_block>>>(
      d_pairs, n_pairs_int,
      d_sx, d_sy, d_sd2y, s_n,
      d_tx, d_ty, d_td2y, t_n,
      d_S, d_T, static_cast<int>(n_basis));
  err = cudaDeviceSynchronize();
  auto kernel_end = std::chrono::steady_clock::now();
  if (err != cudaSuccess) {
    cleanup();
    return CudaStatus(err, "AssembleTwoCenterKernel");
  }

  TwoCenterGpuResult result;
  result.S.resize(n_basis * n_basis);
  result.T.resize(n_basis * n_basis);
  result.n_pairs = pairs.size();
  result.kernel_ms =
      std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

  err = cudaMemcpy(result.S.data(), d_S, mat_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy S D2H"); }
  err = cudaMemcpy(result.T.data(), d_T, mat_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy T D2H"); }

  // Symmetrize S and T matrices (GPU atomic adds from different thread orders).
  for (std::size_t i = 0; i < n_basis; ++i) {
    for (std::size_t j = i + 1; j < n_basis; ++j) {
      std::size_t ij = i * n_basis + j;
      std::size_t ji = j * n_basis + i;
      double s_avg = 0.5 * (result.S[ij] + result.S[ji]);
      double t_avg = 0.5 * (result.T[ij] + result.T[ji]);
      result.S[ij] = s_avg; result.S[ji] = s_avg;
      result.T[ij] = t_avg; result.T[ji] = t_avg;
    }
  }

  cleanup();

  tides::tile::PrecisionDescriptor desc;
  desc.storage = tides::tile::NumericFormat::kFloat64;
  desc.compute = tides::tile::NumericFormat::kFloat64;
  desc.reduction = tides::tile::NumericFormat::kFloat64;
  desc.determinism = tides::tile::DeterminismMode::kDeterministic;
  desc.label = "cuda-two-center-tile-assembly";
  const std::uint64_t candidates =
      static_cast<std::uint64_t>(pairs.size()) * n_basis * n_basis;
  result.ledger.Add(tides::tile::OperationLedgerEntry{
      tides::tile::OperationKind::kGemm,
      desc,
      tides::tile::ErrorBudget{tides::tile::ErrorMetric::kAbsolute, 0.0,
          "GPU spline + SK assembly vs CPU reference"},
      0.0, candidates, candidates, 0,
      "CUDA two-center tile assembly (spline + Slater-Koster)"});
  return result;
}

}  // namespace tides::basis
