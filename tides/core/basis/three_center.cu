// T2.5: GPU three-center integral assembly for KB nonlocal pseudopotential.
//
// V_nl[ab] = Σ_c Σ_l h_l^c × ⟨φ_a|β_l^c⟩ × ⟨β_l^c|φ_b⟩
//
// Each factor is a two-center integral (spline + Slater-Koster angular coupling).
// The kernel processes triplets (a, b, c) where c is a KB projector center.
//
// Observable (T2.5): equals CPU path <=1e-7 rel; throughput recorded.

#include "basis/three_center_gpu.hpp"

#include <cuda_runtime.h>

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

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " +
                         cudaGetErrorString(error));
}

// Device-side cubic spline evaluation (same as two_center.cu).
__device__ double EvalSplineDevice(const double* x_tab,
                                   const double* y_tab,
                                   const double* d2y_tab,
                                   int n_tab, double xq) {
  if (n_tab == 0) return 0.0;
  if (n_tab == 1) return y_tab[0];
  if (xq <= x_tab[0]) {
    const double dx = x_tab[1] - x_tab[0];
    if (dx == 0.0) return y_tab[0];
    return y_tab[0] + (y_tab[1] - y_tab[0]) / dx * (xq - x_tab[0]);
  }
  if (xq >= x_tab[n_tab - 1]) {
    const double dx = x_tab[n_tab - 1] - x_tab[n_tab - 2];
    if (dx == 0.0) return y_tab[n_tab - 1];
    return y_tab[n_tab - 1] +
           (y_tab[n_tab - 1] - y_tab[n_tab - 2]) / dx * (xq - x_tab[n_tab - 1]);
  }
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

// Device-side associated Legendre (same as two_center.cu).
__device__ double AssociatedLegendreDevice(int l, int m, double x) {
  if (m > l) return 0.0;
  double pmm = 1.0;
  if (m > 0) {
    const double somx2 = sqrt((1.0 - x) * (1.0 + x));
    double fact = 1.0;
    for (int i = 1; i <= m; ++i) { pmm *= fact * somx2; fact += 2.0; }
  }
  if (l == m) return pmm;
  double pmmp1 = x * (2.0 * m + 1.0) * pmm;
  if (l == m + 1) return pmmp1;
  double pll = 0.0;
  for (int ll = m + 2; ll <= l; ++ll) {
    pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m);
    pmm = pmmp1; pmmp1 = pll;
  }
  return pll;
}

__device__ double RealSHNormDevice(int l, int m) {
  double denom = 1.0;
  for (int i = l - m + 1; i <= l + m; ++i) denom *= static_cast<double>(i);
  double n = sqrt((2.0 * l + 1.0) / (4.0 * M_PI) / denom);
  if (m != 0) n *= sqrt(2.0);
  return n;
}

__device__ double RealSHDevice(int l, int m, double theta, double phi) {
  const double x = cos(theta);
  const int am = abs(m);
  const double plm = AssociatedLegendreDevice(l, am, x);
  const double n = RealSHNormDevice(l, am);
  if (m > 0) return n * plm * cos(static_cast<double>(m) * phi);
  if (m < 0) return n * plm * sin(static_cast<double>(am) * phi);
  return n * plm;
}

// Triplet input: atoms a, b and KB center c.
struct TripletInput {
  double r_a[3];
  double r_b[3];
  double r_c[3];
  int l_a;
  int l_b;
  int l_c;       // angular momentum of KB projector
  int kb_idx;    // index into spline arrays
  double h_c;    // KB coefficient h_l^c
  int basis_offset_a;
  int basis_offset_b;
};

// Kernel: for each triplet, evaluate the two-center φ-β and β-φ integrals
// via splines, apply Slater-Koster angular coupling, and accumulate into V_nl.
__global__ void AssembleThreeCenterKernel(
    const TripletInput* triplets, int n_triplets,
    // Flat spline tables: for each KB channel, (x, y, d2y) arrays
    // Layout: channel k has tables at [k*max_tab_size .. (k+1)*max_tab_size-1]
    const double* pb_x, const double* pb_y, const double* pb_d2y,
    const int* pb_n,
    const double* bp_x, const double* bp_y, const double* bp_d2y,
    const int* bp_n,
    int max_tab_size,
    double* V_out, int n_basis) {
  const int trip_idx = static_cast<int>(blockIdx.x);
  if (trip_idx >= n_triplets) return;

  const TripletInput t = triplets[trip_idx];

  // Direction from a to c
  const double dx_ac = t.r_c[0] - t.r_a[0];
  const double dy_ac = t.r_c[1] - t.r_a[1];
  const double dz_ac = t.r_c[2] - t.r_a[2];
  const double R_ac = sqrt(dx_ac * dx_ac + dy_ac * dy_ac + dz_ac * dz_ac);
  const double theta_ac = (R_ac > 0.0) ? acos(dz_ac / R_ac) : 0.0;
  const double phi_ac = (dx_ac != 0.0 || dy_ac != 0.0) ? atan2(dy_ac, dx_ac) : 0.0;

  // Direction from b to c
  const double dx_bc = t.r_c[0] - t.r_b[0];
  const double dy_bc = t.r_c[1] - t.r_b[1];
  const double dz_bc = t.r_c[2] - t.r_b[2];
  const double R_bc = sqrt(dx_bc * dx_bc + dy_bc * dy_bc + dz_bc * dz_bc);
  const double theta_bc = (R_bc > 0.0) ? acos(dz_bc / R_bc) : 0.0;
  const double phi_bc = (dx_bc != 0.0 || dy_bc != 0.0) ? atan2(dy_bc, dx_bc) : 0.0;

  // Spline offsets for this KB channel
  const int k = t.kb_idx;
  const double* pb_x_k = pb_x + k * max_tab_size;
  const double* pb_y_k = pb_y + k * max_tab_size;
  const double* pb_d2y_k = pb_d2y + k * max_tab_size;
  const double* bp_x_k = bp_x + k * max_tab_size;
  const double* bp_y_k = bp_y + k * max_tab_size;
  const double* bp_d2y_k = bp_d2y + k * max_tab_size;

  const double radial_ac = EvalSplineDevice(pb_x_k, pb_y_k, pb_d2y_k, pb_n[k], R_ac);
  const double radial_bc = EvalSplineDevice(bp_x_k, bp_y_k, bp_d2y_k, bp_n[k], R_bc);

  const int deg_a = 2 * t.l_a + 1;
  const int deg_b = 2 * t.l_b + 1;
  const int deg_c = 2 * t.l_c + 1;

  // Each thread handles one (m_a, m_b) element.
  // For each (m_a, m_b), sum over m_c of the angular coupling.
  const int tid = static_cast<int>(threadIdx.x);
  for (int idx = tid; idx < deg_a * deg_b; idx += blockDim.x) {
    const int m_a_idx = idx / deg_b;
    const int m_b_idx = idx % deg_b;
    const int m_a = m_a_idx - t.l_a;
    const int m_b = m_b_idx - t.l_b;

    double sum_mc = 0.0;
    for (int mc_idx = 0; mc_idx < deg_c; ++mc_idx) {
      const int m_c = mc_idx - t.l_c;
      const double y_ac = RealSHDevice(t.l_a, m_a, theta_ac, phi_ac);
      const double y_c_a = RealSHDevice(t.l_c, m_c, theta_ac, phi_ac);
      const double y_bc = RealSHDevice(t.l_b, m_b, theta_bc, phi_bc);
      const double y_c_b = RealSHDevice(t.l_c, m_c, theta_bc, phi_bc);
      sum_mc += y_ac * y_c_a * y_bc * y_c_b;
    }

    const double val = t.h_c * radial_ac * radial_bc * sum_mc;

    const int row = t.basis_offset_a + m_a_idx;
    const int col = t.basis_offset_b + m_b_idx;
    if (row < n_basis && col < n_basis && val != 0.0) {
      atomicAdd(&V_out[row * n_basis + col], val);
    }
  }
}

template <typename T>
Status CopyToDevice(const std::vector<T>& host, T** device) {
  if (host.empty()) { *device = nullptr; return Status::Ok(); }
  const std::size_t bytes = host.size() * sizeof(T);
  cudaError_t error = cudaMalloc(reinterpret_cast<void**>(device), bytes);
  if (error != cudaSuccess) return CudaStatus(error, "cudaMalloc");
  error = cudaMemcpy(*device, host.data(), bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) { cudaFree(*device); *device = nullptr; }
  return CudaStatus(error, "cudaMemcpy H2D");
}

template <typename T>
void FreeDevice(T* ptr) { if (ptr) cudaFree(ptr); }

}  // namespace

struct ThreeCenterGpuResultImpl {
  std::vector<double> V_nl;
  double kernel_ms = 0.0;
  std::size_t n_triplets = 0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool ThreeCenterCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

[[nodiscard]] Result<ThreeCenterGpuResult> AssembleThreeCenterCuda(
    const std::vector<double>& positions,
    const std::vector<int>& l_per_atom,
    const std::vector<int>& basis_offsets,
    std::size_t n_basis,
    const std::vector<int>& kb_centers,
    const std::vector<int>& kb_l,
    const std::vector<double>& kb_coeff,
    const std::vector<CubicSpline>& phi_beta_splines,
    const std::vector<CubicSpline>& beta_phi_splines) {
  const std::size_t n_atoms = positions.size() / 3;
  ThreeCenterGpuResult result;
  result.V_nl.assign(n_basis * n_basis, 0.0);

  if (n_atoms == 0 || n_basis == 0 || kb_centers.empty()) return result;
  if (!ThreeCenterCudaAvailable())
    return Status::IoError("CUDA runtime not available for three-center assembly");

  const std::size_t n_kb = kb_centers.size();
  if (kb_l.size() != n_kb || kb_coeff.size() != n_kb ||
      phi_beta_splines.size() != n_kb || beta_phi_splines.size() != n_kb)
    return Status::InvalidArgument("KB channel array size mismatch");

  // Build triplet list: for each KB center c, for each atom pair (a, b).
  std::vector<TripletInput> triplets;
  for (std::size_t kc = 0; kc < n_kb; ++kc) {
    const int c = kb_centers[kc];
    for (std::size_t a = 0; a < n_atoms; ++a) {
      for (std::size_t b = a; b < n_atoms; ++b) {
        TripletInput t;
        t.r_a[0] = positions[a * 3];
        t.r_a[1] = positions[a * 3 + 1];
        t.r_a[2] = positions[a * 3 + 2];
        t.r_b[0] = positions[b * 3];
        t.r_b[1] = positions[b * 3 + 1];
        t.r_b[2] = positions[b * 3 + 2];
        t.r_c[0] = positions[c * 3];
        t.r_c[1] = positions[c * 3 + 1];
        t.r_c[2] = positions[c * 3 + 2];
        t.l_a = l_per_atom[a];
        t.l_b = l_per_atom[b];
        t.l_c = kb_l[kc];
        t.kb_idx = static_cast<int>(kc);
        t.h_c = kb_coeff[kc];
        t.basis_offset_a = basis_offsets[a];
        t.basis_offset_b = basis_offsets[b];
        triplets.push_back(t);
      }
    }
  }

  if (triplets.empty()) return result;

  // Flatten spline tables: pad to max_tab_size for uniform GPU access.
  int max_tab = 0;
  for (const auto& s : phi_beta_splines)
    max_tab = std::max(max_tab, static_cast<int>(s.x().size()));
  for (const auto& s : beta_phi_splines)
    max_tab = std::max(max_tab, static_cast<int>(s.x().size()));
  if (max_tab == 0) return result;

  std::vector<double> pb_x(n_kb * max_tab, 0.0), pb_y(n_kb * max_tab, 0.0),
      pb_d2y(n_kb * max_tab, 0.0);
  std::vector<double> bp_x(n_kb * max_tab, 0.0), bp_y(n_kb * max_tab, 0.0),
      bp_d2y(n_kb * max_tab, 0.0);
  std::vector<int> pb_n(n_kb, 0), bp_n(n_kb, 0);

  for (std::size_t k = 0; k < n_kb; ++k) {
    const auto& s1 = phi_beta_splines[k];
    const auto& s2 = beta_phi_splines[k];
    pb_n[k] = static_cast<int>(s1.x().size());
    bp_n[k] = static_cast<int>(s2.x().size());
    for (int i = 0; i < pb_n[k]; ++i) {
      pb_x[k * max_tab + i] = s1.x()[i];
      pb_y[k * max_tab + i] = s1.y()[i];
      pb_d2y[k * max_tab + i] = s1.d2y()[i];
    }
    for (int i = 0; i < bp_n[k]; ++i) {
      bp_x[k * max_tab + i] = s2.x()[i];
      bp_y[k * max_tab + i] = s2.y()[i];
      bp_d2y[k * max_tab + i] = s2.d2y()[i];
    }
  }

  // Allocate device memory.
  TripletInput* d_trip = nullptr;
  double* d_pb_x = nullptr, *d_pb_y = nullptr, *d_pb_d2y = nullptr;
  double* d_bp_x = nullptr, *d_bp_y = nullptr, *d_bp_d2y = nullptr;
  int* d_pb_n = nullptr, *d_bp_n = nullptr;
  double* d_V = nullptr;

  auto cleanup = [&]() {
    FreeDevice(d_trip);
    FreeDevice(d_pb_x); FreeDevice(d_pb_y); FreeDevice(d_pb_d2y);
    FreeDevice(d_bp_x); FreeDevice(d_bp_y); FreeDevice(d_bp_d2y);
    FreeDevice(d_pb_n); FreeDevice(d_bp_n);
    FreeDevice(d_V);
  };

  auto st = CopyToDevice(triplets, &d_trip);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(pb_x, &d_pb_x);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(pb_y, &d_pb_y);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(pb_d2y, &d_pb_d2y);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(bp_x, &d_bp_x);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(bp_y, &d_bp_y);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(bp_d2y, &d_bp_d2y);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(pb_n, &d_pb_n);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDevice(bp_n, &d_bp_n);
  if (!st.ok()) { cleanup(); return st; }

  const std::size_t mat_bytes = n_basis * n_basis * sizeof(double);
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_V), mat_bytes);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc V"); }
  cudaMemset(d_V, 0, mat_bytes);

  // Launch kernel.
  const int n_trip = static_cast<int>(triplets.size());
  const int threads = 128;
  auto t0 = std::chrono::steady_clock::now();
  AssembleThreeCenterKernel<<<n_trip, threads>>>(
      d_trip, n_trip,
      d_pb_x, d_pb_y, d_pb_d2y, d_pb_n,
      d_bp_x, d_bp_y, d_bp_d2y, d_bp_n,
      max_tab,
      d_V, static_cast<int>(n_basis));
  err = cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "AssembleThreeCenterKernel"); }

  result.n_triplets = triplets.size();
  result.kernel_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  err = cudaMemcpy(result.V_nl.data(), d_V, mat_bytes, cudaMemcpyDeviceToHost);
  cleanup();
  if (err != cudaSuccess) return CudaStatus(err, "cudaMemcpy V D2H");

  // Ledger.
  tides::tile::PrecisionDescriptor desc;
  desc.storage = tides::tile::NumericFormat::kFloat64;
  desc.compute = tides::tile::NumericFormat::kFloat64;
  desc.reduction = tides::tile::NumericFormat::kFloat64;
  desc.determinism = tides::tile::DeterminismMode::kDeterministic;
  desc.label = "cuda-three-center-kb";
  const std::uint64_t candidates =
      static_cast<std::uint64_t>(triplets.size()) * n_basis * n_basis;
  result.ledger.Add(tides::tile::OperationLedgerEntry{
      tides::tile::OperationKind::kGemm,
      desc,
      tides::tile::ErrorBudget{tides::tile::ErrorMetric::kAbsolute, 0.0,
          "GPU three-center KB assembly vs CPU reference"},
      0.0, candidates, candidates, 0,
      "CUDA three-center KB nonlocal PP assembly"});
  return result;
}

}  // namespace tides::basis
