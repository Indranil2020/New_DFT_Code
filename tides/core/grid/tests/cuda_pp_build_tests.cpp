// PP-GPU Phase A tests: device v_loc grid build and weighted v->H adjoint
// vs the CPU references.
//
// Acceptance:
//   - BuildVlocDevice vs scf::BuildVlocGridReference: <= 1e-13 abs, on
//     synthetic PPs exercising every interpolation branch and on real
//     PseudoDojo ONCV UPFs (H, C, Si) when the library is available.
//   - BuildWeightedVmatDevice vs VmatBuilder::BuildHmat: <= 1e-12 abs,
//     including a padded point_stride and the pre-weighted (scale=1) form.

#include "grid/pp_build_gpu.hpp"
#include "grid/gpu_arena.hpp"
#include "grid/vmat_build.hpp"
#include "grid/rho_build.hpp"
#include "grid/dual_grid.hpp"
#include "scf/pp_reference.hpp"
#include "basis/pseudo/pp_loader.hpp"
#include "basis/pseudo/pseudopotential.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

namespace {

using tides::basis::PpLoader;
using tides::basis::Pseudopotential;
using tides::grid::BuildVlocDevice;
using tides::grid::BuildWeightedVmatDevice;
using tides::grid::FreePpVlocTables;
using tides::grid::GpuArena;
using tides::grid::PpBuildCudaAvailable;
using tides::grid::PpVlocTablesDevice;
using tides::grid::RhoBuilder;
using tides::grid::UniformGrid3D;
using tides::grid::UploadPpVlocTables;
using tides::grid::VlocDeviceIn;
using tides::grid::VmatBuilder;
using tides::grid::WeightedVmatDeviceIn;
using tides::scf::BuildVlocGridReference;
using tides::scf::FlattenVlocTables;

double MaxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

// Runs the device v_loc build for the given geometry and compares against
// the CPU reference. Returns the max abs difference, or -1 on device error.
double RunVlocParity(const UniformGrid3D& grid,
                     const std::vector<std::array<double, 3>>& positions,
                     const std::vector<int>& charges,
                     const std::vector<Pseudopotential>* pps,
                     const char* label) {
  const auto v_cpu = BuildVlocGridReference(grid, positions, charges, pps);

  GpuArena& arena = GpuArena::Instance();
  cudaStream_t stream = arena.Stream();
  auto tables_host = FlattenVlocTables(positions, charges, pps);
  PpVlocTablesDevice tables;
  auto st = UploadPpVlocTables(tables_host, &tables, stream);
  if (!st.ok()) {
    std::cerr << label << ": UploadPpVlocTables failed: " << st.message()
              << '\n';
    return -1.0;
  }

  const std::size_t np = grid.total_points();
  double* d_v = static_cast<double*>(arena.Alloc(np * sizeof(double)));
  VlocDeviceIn in;
  in.tables = &tables;
  in.n0 = static_cast<std::int64_t>(grid.n[0]);
  in.n1 = static_cast<std::int64_t>(grid.n[1]);
  in.n2 = static_cast<std::int64_t>(grid.n[2]);
  in.h0 = grid.h[0]; in.h1 = grid.h[1]; in.h2 = grid.h[2];
  in.ox = grid.origin[0]; in.oy = grid.origin[1]; in.oz = grid.origin[2];
  st = BuildVlocDevice(in, d_v, stream);
  std::vector<double> v_gpu(np, 0.0);
  if (st.ok()) {
    cudaMemcpyAsync(v_gpu.data(), d_v, np * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
  arena.Free(d_v);
  FreePpVlocTables(&tables);
  if (!st.ok()) {
    std::cerr << label << ": BuildVlocDevice failed: " << st.message() << '\n';
    return -1.0;
  }

  const double diff = MaxAbsDiff(v_cpu, v_gpu);
  std::cout << label << ": np=" << np << " n_atoms=" << positions.size()
            << " max_diff=" << diff << '\n';
  return diff;
}

// Synthetic PP whose radial grid is engineered so grid points land exactly
// on r=0 (v[0] branch), below r_grid.front() (zero branch), on interior
// intervals (interpolation), exactly on r_grid.back() (endpoint branch),
// and beyond the mesh (zero branch).
int TestVlocSynthetic() {
  UniformGrid3D grid;
  grid.n = {20, 20, 20};
  grid.h = {0.25, 0.25, 0.25};
  grid.origin = {-2.5, -2.5, -2.5};

  Pseudopotential pp;
  pp.element = "Xa";
  pp.Z_valence = 4;
  // front = 0.3 > h so the (2.5, 2.5, 2.5)-adjacent points at r=0.25 hit the
  // below-front branch; back = 2.0 = 8h so on-axis points hit it exactly.
  pp.r_grid = {0.3, 0.5, 0.8, 1.2, 1.7, 2.0};
  pp.v_local = {-9.0, -6.5, -4.0, -2.4, -1.5, -1.2};

  Pseudopotential pp_other = pp;
  pp_other.element = "Xb";
  pp_other.v_local = {-3.0, -2.5, -2.0, -1.4, -0.9, -0.6};

  // Atom 0: on a grid point (r < 1e-10 branch). Atom 1: same species as
  // atom 0 (dedup path). Atom 2: distinct species. Atom 3: all-electron
  // fallback (index beyond pps size).
  std::vector<std::array<double, 3>> positions = {
      {0.0, 0.0, 0.0}, {1.0, 0.25, -0.5}, {-0.75, 0.5, 0.25},
      {0.5, -1.0, 0.75}};
  std::vector<int> charges = {4, 4, 6, 3};
  std::vector<Pseudopotential> pps = {pp, pp, pp_other};

  const double diff =
      RunVlocParity(grid, positions, charges, &pps, "vloc_synthetic");
  if (diff < 0.0 || diff > 1e-13) {
    std::cerr << "FAIL: vloc_synthetic max_diff=" << diff << " > 1e-13\n";
    return 1;
  }
  return 0;
}

int TestVlocRealPP() {
  std::string err;
  auto pps = PpLoader::LoadMany({"C", "H", "Si"}, "", &err);
  if (pps.empty()) {
    std::cout << "vloc_real_pp: SKIPPED (PP library unavailable: " << err
              << ")\n";
    return 0;
  }

  UniformGrid3D grid;
  grid.n = {32, 32, 32};
  grid.h = {0.35, 0.35, 0.35};
  grid.origin = {-5.425, -5.425, -5.425};

  // CH-Si toy geometry (Bohr); C on a grid point to hit the r=0 branch.
  std::vector<std::array<double, 3>> positions = {
      {-0.175, -0.175, -0.175}, {1.9, 0.4, 0.1}, {-1.2, 1.7, -0.6}};
  std::vector<int> charges = {6, 1, 14};

  const double diff =
      RunVlocParity(grid, positions, charges, &pps, "vloc_real_pp");
  if (diff < 0.0 || diff > 1e-13) {
    std::cerr << "FAIL: vloc_real_pp max_diff=" << diff << " > 1e-13\n";
    return 1;
  }
  return 0;
}

int TestWeightedVmat() {
  UniformGrid3D grid;
  grid.n = {18, 18, 18};
  grid.h = {0.3, 0.3, 0.3};
  grid.origin = {-2.7, -2.7, -2.7};
  const std::size_t np = grid.total_points();
  const double dv = grid.h[0] * grid.h[1] * grid.h[2];

  std::vector<std::vector<double>> orbitals = {
      RhoBuilder::GaussianOrbital(grid, 1.0, {0, 0, 0}),
      RhoBuilder::GaussianOrbital(grid, 1.4, {0.4, 0, 0}),
      RhoBuilder::GaussianOrbital(grid, 0.9, {0, 0.5, -0.2}),
      RhoBuilder::GaussianOrbital(grid, 1.8, {-0.3, 0.2, 0.4})};
  const std::size_t n_orb = orbitals.size();

  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> dist(-1.5, 1.5);
  std::vector<double> v(np);
  for (auto& x : v) x = dist(rng);

  const auto H_cpu = VmatBuilder::BuildHmat(grid, orbitals, v);

  // Device layout with a padded stride to exercise point_stride handling.
  const std::size_t stride = np + 13;
  std::vector<double> phi_flat(n_orb * stride, 0.0);
  for (std::size_t k = 0; k < n_orb; ++k)
    for (std::size_t g = 0; g < np; ++g)
      phi_flat[k * stride + g] = orbitals[k][g];

  GpuArena& arena = GpuArena::Instance();
  cudaStream_t stream = arena.Stream();
  double* d_phi = static_cast<double*>(
      arena.Alloc(phi_flat.size() * sizeof(double)));
  double* d_wv = static_cast<double*>(arena.Alloc(np * sizeof(double)));
  double* d_vmat = static_cast<double*>(
      arena.Alloc(n_orb * n_orb * sizeof(double)));
  cudaMemcpyAsync(d_phi, phi_flat.data(), phi_flat.size() * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_wv, v.data(), np * sizeof(double),
                  cudaMemcpyHostToDevice, stream);

  WeightedVmatDeviceIn in;
  in.phi = d_phi;
  in.wv = d_wv;
  in.nbasis = static_cast<std::int64_t>(n_orb);
  in.np = static_cast<std::int64_t>(np);
  in.point_stride = static_cast<std::int64_t>(stride);
  in.scale = dv;  // bare potential: quadrature weight applied via scale
  auto st = BuildWeightedVmatDevice(in, d_vmat, stream);
  std::vector<double> H_gpu(n_orb * n_orb, 0.0);
  if (st.ok()) {
    cudaMemcpyAsync(H_gpu.data(), d_vmat, H_gpu.size() * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }

  int failures = 0;
  if (!st.ok()) {
    std::cerr << "BuildWeightedVmatDevice failed: " << st.message() << '\n';
    ++failures;
  } else {
    const double diff = MaxAbsDiff(H_cpu, H_gpu);
    std::cout << "weighted_vmat_scaled: n_orb=" << n_orb << " np=" << np
              << " stride=" << stride << " max_diff=" << diff << '\n';
    if (diff > 1e-12) {
      std::cerr << "FAIL: weighted_vmat_scaled max_diff=" << diff
                << " > 1e-12\n";
      ++failures;
    }
  }

  // Pre-weighted form: wv = dv * v with scale = 1 must give the same matrix.
  if (failures == 0) {
    std::vector<double> wv(np);
    for (std::size_t g = 0; g < np; ++g) wv[g] = dv * v[g];
    cudaMemcpyAsync(d_wv, wv.data(), np * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    in.scale = 1.0;
    st = BuildWeightedVmatDevice(in, d_vmat, stream);
    if (st.ok()) {
      cudaMemcpyAsync(H_gpu.data(), d_vmat, H_gpu.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      const double diff = MaxAbsDiff(H_cpu, H_gpu);
      std::cout << "weighted_vmat_preweighted: max_diff=" << diff << '\n';
      if (diff > 1e-12) {
        std::cerr << "FAIL: weighted_vmat_preweighted max_diff=" << diff
                  << " > 1e-12\n";
        ++failures;
      }
    } else {
      std::cerr << "BuildWeightedVmatDevice (preweighted) failed: "
                << st.message() << '\n';
      ++failures;
    }
  }

  arena.Free(d_phi);
  arena.Free(d_wv);
  arena.Free(d_vmat);
  return failures;
}

}  // namespace

int main() {
  if (!PpBuildCudaAvailable()) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  int failures = 0;
  failures += TestVlocSynthetic();
  failures += TestVlocRealPP();
  failures += TestWeightedVmat();

  if (failures == 0) {
    std::cout << "All GPU pp_build tests passed.\n";
  } else {
    std::cerr << failures << " GPU pp_build test(s) failed.\n";
  }
  return failures;
}
