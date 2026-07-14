#pragma once

// PP-GPU Phase A: device-resident pseudopotential local-potential build and
// weighted grid->basis back-projection.
//
// BuildVlocDevice evaluates v_ext(r) = sum_a v_loc,a(|r - R_a|) on the uniform
// grid, replicating the CPU Step-7 loop of NaoDriver::Run (linear
// interpolation on the PP radial grid with identical boundary branches),
// including the all-electron -Z/r fallback for atoms without a
// pseudopotential.
//
// BuildWeightedVmatDevice is the LDA-shaped adjoint
//   V_mn = scale * sum_g wv(g) phi_m(g) phi_n(g).
// When wv already carries quadrature weights (e.g. XcEval's wv_rho) pass
// scale = 1; when wv is a bare potential v(r) pass scale = dv. Writes
// [nbasis][nbasis] row-major on device without allocating or synchronizing.
//
// Everything here takes flat POD arrays so the grid layer stays independent
// of the basis/pseudo headers; flattening from Pseudopotential structs lives
// in scf/pp_reference.hpp.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "common/status.hpp"

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

namespace tides::grid {

// Env toggle for the device PP paths: TIDES_PP_DEVICE=0 forces the CPU
// reference implementations (A/B parity runs, CI on non-GPU machines).
inline bool PpDeviceEnabled() {
  const char* env = std::getenv("TIDES_PP_DEVICE");
  return !(env && env[0] == '0');
}

[[nodiscard]] bool PpBuildCudaAvailable();

// Flattened v_loc tables. Species k owns rows
// [species_offset[k], species_offset[k+1]) of r_tab/v_tab; atoms with
// atom_species[a] < 0 use the all-electron -Z/r fallback with
// atom_charge[a] = Z. A species with an empty row range contributes zero
// (mirrors the CPU loop's empty-v_local handling).
struct PpVlocTablesHost {
  std::vector<double> atom_pos;     // [3*n_atoms]
  std::vector<double> atom_charge;  // [n_atoms]
  std::vector<int> atom_species;    // [n_atoms]; -1 = all-electron
  std::vector<int> species_offset;  // [n_species+1]
  std::vector<double> r_tab;        // concatenated radial grids
  std::vector<double> v_tab;        // concatenated v_local tables
};

// Device mirror of PpVlocTablesHost. Buffers are owned by the GpuArena pool.
struct PpVlocTablesDevice {
  double* atom_pos = nullptr;
  double* atom_charge = nullptr;
  int* atom_species = nullptr;
  int* species_offset = nullptr;
  double* r_tab = nullptr;
  double* v_tab = nullptr;
  int n_atoms = 0;
  int n_species = 0;
};

// Uploads the flattened tables (async on stream, synchronized before return
// of the first kernel that consumes them via stream ordering). On failure the
// partially allocated buffers are released and *device is reset.
[[nodiscard]] Status UploadPpVlocTables(const PpVlocTablesHost& host,
                                        PpVlocTablesDevice* device,
                                        cudaStream_t stream);

// Returns the device buffers to the GpuArena pool and resets *device.
void FreePpVlocTables(PpVlocTablesDevice* device);

struct VlocDeviceIn {
  const PpVlocTablesDevice* tables = nullptr;
  std::int64_t n0 = 0, n1 = 0, n2 = 0;  // grid points per axis
  double h0 = 0.0, h1 = 0.0, h2 = 0.0;  // spacing (Bohr)
  double ox = 0.0, oy = 0.0, oz = 0.0;  // origin (Bohr)
};

// v_out[g] = sum_a v_loc,a(|r_g - R_a|) for all np = n0*n1*n2 grid points
// (flat index g = ix + n0*(iy + n1*iz), matching UniformGrid3D::flatten).
[[nodiscard]] Status BuildVlocDevice(const VlocDeviceIn& input, double* v_out,
                                     cudaStream_t stream);

struct WeightedVmatDeviceIn {
  const double* phi = nullptr;  // [nbasis][point_stride] device
  const double* wv = nullptr;   // [np] device
  std::int64_t nbasis = 0;
  std::int64_t np = 0;
  std::int64_t point_stride = 0;
  double scale = 1.0;
};

[[nodiscard]] Status BuildWeightedVmatDevice(const WeightedVmatDeviceIn& input,
                                             double* vmat, cudaStream_t stream);

}  // namespace tides::grid
