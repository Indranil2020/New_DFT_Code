// PP-GPU Phase A: v_loc grid build + weighted v->H adjoint on device.
//
// Parity contract (grid/tests/cuda_pp_build_tests.cpp):
//   - BuildVlocDevice matches the CPU reference (scf/pp_reference.hpp,
//     extracted from the NaoDriver Step-7 loop) to <= 1e-13 abs.
//   - BuildWeightedVmatDevice matches VmatBuilder::BuildHmat to <= 1e-12 abs.

#include "grid/pp_build_gpu.hpp"
#include "grid/gpu_arena.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "common/status.hpp"

namespace tides::grid {
namespace {

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " +
                         cudaGetErrorString(error));
}

// One thread per grid point; loops over atoms. The interpolation branches
// mirror the std::upper_bound-based CPU loop exactly:
//   r <  1e-10                 -> v_tab[first]
//   r <= r_tab[last]:
//     r <  r_tab[first]        -> no contribution
//     r == r_tab[last]         -> v_tab[last]
//     else                     -> linear interpolation on [j, j+1]
//   r >  r_tab[last]           -> no contribution
__global__ void VlocGridKernel(
    const double* atom_pos, const double* atom_charge,
    const int* atom_species, const int* species_offset,
    const double* r_tab, const double* v_tab,
    double* v_out, int n_atoms, std::int64_t np,
    std::int64_t n0, std::int64_t n1,
    double h0, double h1, double h2,
    double ox, double oy, double oz) {
  const std::int64_t g =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= np) return;
  const std::int64_t ix = g % n0;
  const std::int64_t iy = (g / n0) % n1;
  const std::int64_t iz = g / (n0 * n1);
  const double x = ox + h0 * static_cast<double>(ix);
  const double y = oy + h1 * static_cast<double>(iy);
  const double z = oz + h2 * static_cast<double>(iz);

  double v = 0.0;
  for (int a = 0; a < n_atoms; ++a) {
    const double dx = x - atom_pos[3 * a];
    const double dy = y - atom_pos[3 * a + 1];
    const double dz = z - atom_pos[3 * a + 2];
    const double r = sqrt(dx * dx + dy * dy + dz * dz);
    const int sp = atom_species[a];
    if (sp >= 0) {
      const int lo = species_offset[sp];
      const int hi = species_offset[sp + 1];
      if (hi <= lo) continue;  // empty table: zero contribution
      if (r < 1e-10) {
        v += v_tab[lo];
      } else if (r <= r_tab[hi - 1]) {
        // upper_bound over r_tab[lo, hi): first index with r_tab > r.
        int l = lo, h = hi;
        while (l < h) {
          const int mid = (l + h) / 2;
          if (r_tab[mid] <= r) l = mid + 1; else h = mid;
        }
        if (l > lo && l < hi) {
          const int j = l - 1;
          const double t = (r - r_tab[j]) / (r_tab[j + 1] - r_tab[j]);
          v += (1.0 - t) * v_tab[j] + t * v_tab[j + 1];
        } else if (l == hi) {
          v += v_tab[hi - 1];
        }
        // l == lo (r < r_tab[lo]): no contribution, matching the CPU loop.
      }
    } else {
      if (r > 1e-10) v -= atom_charge[a] / r;
    }
  }
  v_out[g] = v;
}

// Block (mu, nu) reduces over grid points; wv semantics documented in the
// header (pass scale = dv for bare potentials, scale = 1 for pre-weighted).
__global__ void WeightedVmatKernel(
    const double* phi, const double* wv, double* vmat,
    std::int64_t nbasis, std::int64_t np, std::int64_t point_stride,
    double scale) {
  const std::int64_t mu = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t nu = static_cast<std::int64_t>(blockIdx.y);
  if (mu >= nbasis || nu >= nbasis || nu < mu) return;
  double partial = 0.0;
  for (std::int64_t point = threadIdx.x; point < np;
       point += static_cast<std::int64_t>(blockDim.x)) {
    partial += wv[point] * phi[mu * point_stride + point] *
               phi[nu * point_stride + point];
  }
  __shared__ double reduction[256];
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (int offset = 128; offset > 0; offset /= 2) {
    if (threadIdx.x < offset)
      reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double value = scale * reduction[0];
    vmat[mu * nbasis + nu] = value;
    vmat[nu * nbasis + mu] = value;
  }
}

template <typename T>
Status UploadArray(GpuArena& arena, const std::vector<T>& host, T** device,
                   cudaStream_t stream) {
  *device = nullptr;
  if (host.empty()) return Status::Ok();
  const std::size_t bytes = host.size() * sizeof(T);
  *device = static_cast<T*>(arena.Alloc(bytes));
  if (*device == nullptr)
    return Status::IoError("GpuArena.Alloc failed for PP table upload");
  return CudaStatus(cudaMemcpyAsync(*device, host.data(), bytes,
                                    cudaMemcpyHostToDevice, stream),
                    "PP table H2D");
}

}  // namespace

[[nodiscard]] bool PpBuildCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

Status UploadPpVlocTables(const PpVlocTablesHost& host,
                          PpVlocTablesDevice* device, cudaStream_t stream) {
  if (device == nullptr)
    return Status::InvalidArgument("PP table upload requires a device struct");
  *device = PpVlocTablesDevice{};
  const std::size_t n_atoms = host.atom_charge.size();
  if (host.atom_pos.size() != 3 * n_atoms ||
      host.atom_species.size() != n_atoms) {
    return Status::InvalidArgument("PP table atom arrays are inconsistent");
  }
  if (host.species_offset.empty() ||
      host.v_tab.size() != host.r_tab.size() ||
      static_cast<std::size_t>(host.species_offset.back()) !=
          host.r_tab.size()) {
    return Status::InvalidArgument("PP table species offsets are inconsistent");
  }
  for (std::size_t k = 1; k < host.species_offset.size(); ++k) {
    if (host.species_offset[k] < host.species_offset[k - 1])
      return Status::InvalidArgument("PP table species offsets not sorted");
  }
  const int n_species = static_cast<int>(host.species_offset.size()) - 1;
  for (int sp : host.atom_species) {
    if (sp >= n_species)
      return Status::InvalidArgument("PP table species index out of range");
  }

  GpuArena& arena = GpuArena::Instance();
  device->n_atoms = static_cast<int>(n_atoms);
  device->n_species = n_species;

  auto st = UploadArray(arena, host.atom_pos, &device->atom_pos, stream);
  if (st.ok()) st = UploadArray(arena, host.atom_charge, &device->atom_charge, stream);
  if (st.ok()) st = UploadArray(arena, host.atom_species, &device->atom_species, stream);
  if (st.ok()) st = UploadArray(arena, host.species_offset, &device->species_offset, stream);
  if (st.ok()) st = UploadArray(arena, host.r_tab, &device->r_tab, stream);
  if (st.ok()) st = UploadArray(arena, host.v_tab, &device->v_tab, stream);
  if (!st.ok()) FreePpVlocTables(device);
  return st;
}

void FreePpVlocTables(PpVlocTablesDevice* device) {
  if (device == nullptr) return;
  GpuArena& arena = GpuArena::Instance();
  if (device->atom_pos) arena.Free(device->atom_pos);
  if (device->atom_charge) arena.Free(device->atom_charge);
  if (device->atom_species) arena.Free(device->atom_species);
  if (device->species_offset) arena.Free(device->species_offset);
  if (device->r_tab) arena.Free(device->r_tab);
  if (device->v_tab) arena.Free(device->v_tab);
  *device = PpVlocTablesDevice{};
}

Status BuildVlocDevice(const VlocDeviceIn& input, double* v_out,
                       cudaStream_t stream) {
  if (input.tables == nullptr || v_out == nullptr)
    return Status::InvalidArgument("v_loc device build requires tables and output");
  if (input.n0 <= 0 || input.n1 <= 0 || input.n2 <= 0)
    return Status::InvalidArgument("v_loc device build requires a non-empty grid");
  const std::int64_t np = input.n0 * input.n1 * input.n2;
  const PpVlocTablesDevice& t = *input.tables;
  if (t.n_atoms > 0 &&
      (t.atom_pos == nullptr || t.atom_charge == nullptr ||
       t.atom_species == nullptr)) {
    return Status::InvalidArgument("v_loc device build requires atom tables");
  }
  const int threads = 256;
  const int blocks = static_cast<int>((np + threads - 1) / threads);
  VlocGridKernel<<<blocks, threads, 0, stream>>>(
      t.atom_pos, t.atom_charge, t.atom_species, t.species_offset,
      t.r_tab, t.v_tab, v_out, t.n_atoms, np, input.n0, input.n1,
      input.h0, input.h1, input.h2, input.ox, input.oy, input.oz);
  return CudaStatus(cudaGetLastError(), "VlocGridKernel launch");
}

Status BuildWeightedVmatDevice(const WeightedVmatDeviceIn& input, double* vmat,
                               cudaStream_t stream) {
  if (input.nbasis <= 0 || input.np < 0 || input.point_stride < input.np) {
    return Status::InvalidArgument(
        "weighted vmat device build requires nbasis > 0, np >= 0, and "
        "point_stride >= np");
  }
  if (input.phi == nullptr || input.wv == nullptr || vmat == nullptr) {
    return Status::InvalidArgument(
        "weighted vmat device build requires non-null device pointers");
  }
  if (input.np == 0) {
    return CudaStatus(cudaMemsetAsync(
                          vmat, 0,
                          static_cast<std::size_t>(input.nbasis) *
                              input.nbasis * sizeof(double),
                          stream),
                      "cudaMemsetAsync weighted vmat");
  }
  dim3 grid_dim(static_cast<unsigned int>(input.nbasis),
                static_cast<unsigned int>(input.nbasis));
  WeightedVmatKernel<<<grid_dim, 256, 0, stream>>>(
      input.phi, input.wv, vmat, input.nbasis, input.np, input.point_stride,
      input.scale);
  return CudaStatus(cudaGetLastError(), "WeightedVmatKernel launch");
}

}  // namespace tides::grid
