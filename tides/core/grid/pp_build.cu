// PP-GPU Phase A: v_loc grid build + weighted v->H adjoint on device.
//
// Parity contract (grid/tests/cuda_pp_build_tests.cpp):
//   - BuildVlocDevice matches the CPU reference (scf/pp_reference.hpp,
//     extracted from the NaoDriver Step-7 loop) to <= 1e-13 abs.
//   - BuildWeightedVmatDevice matches VmatBuilder::BuildHmat to <= 1e-12 abs.

#include "grid/pp_build_gpu.hpp"
#include "grid/gpu_arena.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdint>
#include <cstdlib>
#include <string>

#include "common/status.hpp"

namespace tides::grid {
namespace {

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " +
                         cudaGetErrorString(error));
}

[[nodiscard]] Status CublasStatus(cublasStatus_t s, const char* ctx) {
  if (s == CUBLAS_STATUS_SUCCESS) return Status::Ok();
  return Status::IoError(std::string(ctx) + ": cuBLAS error " +
                         std::to_string(static_cast<int>(s)));
}

// Scale each column g of phi by wv[g]:  temp[mu, g] = wv[g] * phi[mu, g]
// 2D grid: (g, mu) for coalesced memory access along g.
__global__ void ScaleColumnsKernel(
    double* out, const double* phi, const double* wv,
    std::int64_t nbasis, std::int64_t np, std::int64_t stride) {
  const std::int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const std::int64_t mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (g >= np || mu >= nbasis) return;
  out[mu * stride + g] = wv[g] * phi[mu * stride + g];
}

// Custom mixed-precision weighted vmat kernel:
// V[mu,nu] = scale * sum_g wv[g] * phi[mu,g] * phi[nu,g]
// Uses FP32 for multiply, FP64 for accumulation.
// Each block computes a BM×BN tile of V. Iterates over K in BK-sized chunks.
// phi is row-major [nbasis][stride].
template <int BM, int BN, int BK>
__global__ void __launch_bounds__(BM* BN)
WeightedVmatMixedKernel(
    double* __restrict__ V,
    const double* __restrict__ phi,
    const double* __restrict__ wv,
    double scale,
    std::int64_t nbasis, std::int64_t np, std::int64_t stride) {
  const int mu0 = blockIdx.y * BM;
  const int nu0 = blockIdx.x * BN;
  const int tid = threadIdx.y * BN + threadIdx.x;

  // Accumulator in FP64.
  double acc[BM][BN];  // Not actually BM*BN per thread — each thread handles 1 element.
  // Actually: each thread handles exactly one (mu, nu) element.
  double my_acc = 0.0;

  const int mu = mu0 + threadIdx.y;
  const int nu = nu0 + threadIdx.x;

  // Shared memory tiles for phi rows.
  __shared__ float sphi_mu[BM][BK];
  __shared__ float sphi_nu[BN][BK];
  __shared__ float swv[BK];

  for (std::int64_t g0 = 0; g0 < np; g0 += BK) {
    const int g_cur = min((int)BK, (int)(np - g0));

    // Load phi tiles into shared memory (FP32).
    // Thread (ty, tx) loads sphi_mu[ty][tx] and sphi_nu[tx][ty] etc.
    // Use linear thread mapping for coalesced loads.
    {
      const int load_tid = threadIdx.y * BN + threadIdx.x;
      const int load_total = BM * BN;

      // Load wv chunk.
      for (int i = load_tid; i < BK && i < g_cur; i += load_total) {
        swv[i] = static_cast<float>(wv[g0 + i]);
      }

      // Load phi rows for mu indices.
      for (int i = load_tid; i < BM * BK && i < BM * g_cur; i += load_total) {
        int row = i / BK;
        int col = i % BK;
        if (mu0 + row < nbasis) {
          sphi_mu[row][col] = static_cast<float>(phi[(mu0 + row) * stride + g0 + col]);
        }
      }

      // Load phi rows for nu indices.
      for (int i = load_tid; i < BN * BK && i < BN * g_cur; i += load_total) {
        int row = i / BK;
        int col = i % BK;
        if (nu0 + row < nbasis) {
          sphi_nu[row][col] = static_cast<float>(phi[(nu0 + row) * stride + g0 + col]);
        }
      }
    }
    __syncthreads();

    // Compute partial products: FP32 multiply, FP64 accumulate.
    if (mu < nbasis && nu < nbasis) {
      for (int g = 0; g < g_cur; ++g) {
        float w = swv[g];
        float a = sphi_mu[threadIdx.y][g];
        float b = sphi_nu[threadIdx.x][g];
        my_acc += static_cast<double>(w * a * b);
      }
    }
    __syncthreads();
  }

  if (mu < nbasis && nu < nbasis) {
    V[mu * nbasis + nu] = scale * my_acc;
  }
}

// FP32 version: scale + cast to float in one pass.
__global__ void ScaleColumnsKernelF32(
    float* out, const double* phi, const double* wv,
    std::int64_t nbasis, std::int64_t np, std::int64_t stride) {
  const std::int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const std::int64_t mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (g >= np || mu >= nbasis) return;
  out[mu * stride + g] = static_cast<float>(wv[g] * phi[mu * stride + g]);
}

// Cast double array to float.
__global__ void CastToF32Kernel(
    float* out, const double* in, std::int64_t n) {
  std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = static_cast<float>(in[i]);
}

// Cast float array to double.
__global__ void CastToF64Kernel(
    double* out, const float* in, std::int64_t n) {
  std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = static_cast<double>(in[i]);
}

// Accumulate FP32 matrix into FP64: V_f64[i,j] += (double)W_f32[i,j]
__global__ void AccumulateF32ToF64Kernel(
    double* V, const float* W, std::int64_t n) {
  std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * n) return;
  V[idx] += static_cast<double>(W[idx]);
}

// 2D slice cast: out[mu*stride + g] = (float)in[mu*stride + g] for g in [0,k_cur)
__global__ void CastSliceToF32Kernel(
    float* out, const double* in,
    std::int64_t nbasis, std::int64_t k_cur, std::int64_t stride) {
  const std::int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const std::int64_t mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (g >= k_cur || mu >= nbasis) return;
  out[mu * stride + g] = static_cast<float>(in[mu * stride + g]);
}

// Symmetrize-accumulate FP32 into FP64: V_f64 += W_f32 + W_f32^T
__global__ void SymmetrizeAccumulateF32ToF64Kernel(
    double* V, const float* W, std::int64_t n) {
  std::int64_t mu = blockIdx.x * blockDim.x + threadIdx.x;
  if (mu >= n) return;
  for (std::int64_t nu = mu; nu < n; ++nu) {
    const double val = static_cast<double>(W[mu * n + nu] + W[nu * n + mu]);
    V[mu * n + nu] += val;
    if (nu != mu) V[nu * n + mu] += val;
  }
}

// Grid screening: count grid points where any basis function is non-zero.
// NAO basis is strictly zero beyond r_cut, so this is exact.
__global__ void CountActivePointsKernel(
    const double* phi, std::int64_t nbasis, std::int64_t np,
    std::int64_t stride, int* count) {
  std::int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= np) return;
  bool active = false;
  for (std::int64_t mu = 0; mu < nbasis; ++mu) {
    if (phi[mu * stride + g] != 0.0) { active = true; break; }
  }
  if (active) atomicAdd(count, 1);
}

// Build compaction index: for each active grid point, write its index.
// Uses atomic counter to assign compacted positions.
__global__ void BuildCompactionIndexKernel(
    const double* phi, std::int64_t nbasis, std::int64_t np,
    std::int64_t stride, int* index, int* compact_count) {
  std::int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= np) return;
  bool active = false;
  for (std::int64_t mu = 0; mu < nbasis; ++mu) {
    if (phi[mu * stride + g] != 0.0) { active = true; break; }
  }
  if (active) {
    int pos = atomicAdd(compact_count, 1);
    index[pos] = static_cast<int>(g);
  }
}

// Compact phi columns: out[mu * np_compact + pos] = phi[mu * stride + index[pos]]
__global__ void CompactPhiKernel(
    double* out, const double* phi, const int* index,
    std::int64_t nbasis, std::int64_t np_compact, std::int64_t stride) {
  std::int64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  std::int64_t mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (pos >= np_compact || mu >= nbasis) return;
  out[mu * np_compact + pos] = phi[mu * stride + index[pos]];
}

// Compact wv: out[pos] = wv[index[pos]]
__global__ void CompactWvKernel(
    double* out, const double* wv, const int* index, std::int64_t np_compact) {
  std::int64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= np_compact) return;
  out[pos] = wv[index[pos]];
}

// cuBLAS handle + temp buffer for GEMM-based vmat.
struct VmatGemmCache {
  cublasHandle_t handle = nullptr;
  double* d_temp = nullptr;   // [nbasis * stride] scaled phi (FP64)
  float* d_temp_f32 = nullptr;   // [nbasis * stride] scaled phi (FP32)
  float* d_phi_f32 = nullptr;    // [nbasis * stride] phi cast to FP32)
  float* d_vmat_f32 = nullptr;   // [nbasis * nbasis] result (FP32)
  // Grid screening buffers.
  int* d_compact_index = nullptr;     // [np_compact] indices of active grid points
  double* d_phi_compact = nullptr;    // [nbasis * np_compact] compacted phi
  double* d_wv_compact = nullptr;     // [np_compact] compacted wv
  double* d_temp_compact = nullptr;   // [nbasis * np_compact] compacted scaled phi
  int* d_screen_count = nullptr;      // device counter for screening
  std::int64_t np_compact = 0;        // number of active grid points
  bool screen_initialized = false;    // whether compaction index has been built
  std::int64_t cached_nbasis = 0, cached_stride = 0;

  ~VmatGemmCache() {
    if (handle) cublasDestroy(handle);
    if (d_temp) cudaFree(d_temp);
    if (d_temp_f32) cudaFree(d_temp_f32);
    if (d_phi_f32) cudaFree(d_phi_f32);
    if (d_vmat_f32) cudaFree(d_vmat_f32);
    if (d_compact_index) cudaFree(d_compact_index);
    if (d_phi_compact) cudaFree(d_phi_compact);
    if (d_wv_compact) cudaFree(d_wv_compact);
    if (d_temp_compact) cudaFree(d_temp_compact);
    if (d_screen_count) cudaFree(d_screen_count);
  }

  bool ensure(std::int64_t nbasis, std::int64_t stride, cudaStream_t stream) {
    if (!handle) {
      if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
      cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    }
    cublasSetStream(handle, stream);
    if (cached_nbasis != nbasis || cached_stride != stride) {
      if (d_temp) cudaFree(d_temp);
      if (d_temp_f32) cudaFree(d_temp_f32);
      if (d_phi_f32) cudaFree(d_phi_f32);
      if (d_vmat_f32) cudaFree(d_vmat_f32);
      if (d_compact_index) cudaFree(d_compact_index);
      if (d_phi_compact) cudaFree(d_phi_compact);
      if (d_wv_compact) cudaFree(d_wv_compact);
      if (d_temp_compact) cudaFree(d_temp_compact);
      if (d_screen_count) cudaFree(d_screen_count);
      d_compact_index = nullptr;
      d_phi_compact = nullptr;
      d_wv_compact = nullptr;
      d_temp_compact = nullptr;
      d_screen_count = nullptr;
      cudaMalloc(&d_temp, nbasis * stride * sizeof(double));
      cudaMalloc(&d_temp_f32, nbasis * stride * sizeof(float));
      cudaMalloc(&d_phi_f32, nbasis * stride * sizeof(float));
      cudaMalloc(&d_vmat_f32, nbasis * nbasis * sizeof(float));
      cached_nbasis = nbasis;
      cached_stride = stride;
      screen_initialized = false;
      np_compact = 0;
    }
    return d_temp != nullptr;
  }

  bool ensure_screen(std::int64_t nbasis, std::int64_t stride) {
    if (d_screen_count) return true;
    cudaMalloc(&d_compact_index, stride * sizeof(int));
    cudaMalloc(&d_phi_compact, nbasis * stride * sizeof(double));
    cudaMalloc(&d_wv_compact, stride * sizeof(double));
    cudaMalloc(&d_temp_compact, nbasis * stride * sizeof(double));
    cudaMalloc(&d_screen_count, sizeof(int));
    return d_screen_count != nullptr;
  }

  void ResetScreen() {
    screen_initialized = false;
    np_compact = 0;
  }
};

VmatGemmCache& vmat_gemm_cache() {
  static VmatGemmCache c;
  return c;
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

void ResetVmatScreenCache() {
  vmat_gemm_cache().ResetScreen();
}

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

  // Mixed-precision custom kernel: FP32 multiply, FP64 accumulate.
  // Uses shared memory tiling for high throughput on consumer GPUs.
  const bool use_mixed = [] {
    const char* e = std::getenv("TIDES_VMAT_MIXED");
    return e != nullptr && e[0] == '1';
  }();

  if (use_mixed && input.nbasis >= 4) {
    constexpr int BM = 16, BN = 16, BK = 64;
    dim3 block(BN, BM);
    dim3 grid(
        (static_cast<unsigned int>(input.nbasis) + BN - 1) / BN,
        (static_cast<unsigned int>(input.nbasis) + BM - 1) / BM);
    WeightedVmatMixedKernel<BM, BN, BK><<<grid, block, 0, stream>>>(
        vmat, input.phi, input.wv, input.scale,
        input.nbasis, input.np, input.point_stride);
    auto err = cudaGetLastError();
    if (err == cudaSuccess) return Status::Ok();
  }

  // GEMM path: V = scale * (diag(wv) * Phi) * Phi^T
  // phi is [nbasis][stride] row-major = [stride][nbasis] col-major.
  // V[mu,nu] = scale * sum_g temp[mu,g] * phi[nu,g]
  // col-major: V = Phi^T * temp  (CUBLAS_OP_T on phi, CUBLAS_OP_N on temp)
  const bool use_gemm = [] {
    const char* e = std::getenv("TIDES_VMAT_GEMM");
    return (e == nullptr || e[0] != '0');  // enabled by default
  }();

  if (use_gemm && input.nbasis >= 4) {
    auto& gc = vmat_gemm_cache();
    if (gc.ensure(input.nbasis, input.point_stride, stream)) {
      const int n = static_cast<int>(input.nbasis);
      const int k = static_cast<int>(input.np);
      const int lda = static_cast<int>(input.point_stride);

      // Check for split-K FP32 path (enabled by default for large K).
      const bool use_splitk = [] {
        const char* e = std::getenv("TIDES_VMAT_SPLITK");
        if (e == nullptr) return false;  // disabled by default (precision insufficient)
        return e[0] == '1';
      }();

      if (use_splitk && k >= 4096) {
        // Split-K FP32 GEMM with FP64 accumulation:
        // 1. Cast phi to FP32 (full, once)
        // 2. For each K-chunk: cast temp chunk to FP32, FP32 GEMM, accumulate to FP64
        // This uses tensor cores (12.7 TFLOPS) instead of FP64 (0.2 TFLOPS).
        constexpr int K_CHUNK = 1024;
        const int n_chunks = (k + K_CHUNK - 1) / K_CHUNK;

        // Cast full phi to FP32.
        {
          dim3 block(256);
          dim3 grid(static_cast<unsigned int>((input.nbasis * input.point_stride + 255) / 256));
          CastToF32Kernel<<<grid, block, 0, stream>>>(
              gc.d_phi_f32, input.phi, input.nbasis * input.point_stride);
        }

        // Zero FP64 accumulator.
        cudaMemsetAsync(vmat, 0, n * n * sizeof(double), stream);

        const float alpha_f = static_cast<float>(input.scale);
        const float beta_f = 0.0f;

        for (int c = 0; c < n_chunks; ++c) {
          const int k_offset = c * K_CHUNK;
          const int k_cur = (k_offset + K_CHUNK <= k) ? K_CHUNK : (k - k_offset);

          // Scale phi columns for this chunk into d_temp (FP64).
          {
            dim3 block(32, 4);
            dim3 grid((static_cast<unsigned int>(k_cur) + block.x - 1) / block.x,
                      (static_cast<unsigned int>(input.nbasis) + block.y - 1) / block.y);
            ScaleColumnsKernel<<<grid, block, 0, stream>>>(
                gc.d_temp, input.phi + k_offset, input.wv + k_offset,
                input.nbasis, k_cur, input.point_stride);
          }

          // Cast scaled temp chunk to FP32.
          {
            dim3 block(32, 4);
            dim3 grid((static_cast<unsigned int>(k_cur) + block.x - 1) / block.x,
                      (static_cast<unsigned int>(input.nbasis) + block.y - 1) / block.y);
            CastSliceToF32Kernel<<<grid, block, 0, stream>>>(
                gc.d_temp_f32, gc.d_temp, input.nbasis, k_cur, input.point_stride);
          }

          // FP32 GEMM: V_chunk[n x n] = Phi_f32^T * temp_f32
          // Phi_f32 is [n x stride] row-major = [stride x n] col-major
          // temp_f32 is [n x stride] row-major = [stride x n] col-major
          // V = Phi^T * temp: OP_T on phi, OP_N on temp
          cublasStatus_t cs = cublasGemmEx(
              gc.handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              n, n, k_cur,
              &alpha_f,
              gc.d_phi_f32 + k_offset, CUDA_R_32F, lda,
              gc.d_temp_f32, CUDA_R_32F, lda,
              &beta_f,
              gc.d_vmat_f32, CUDA_R_32F, n,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
          if (cs != CUBLAS_STATUS_SUCCESS) return CublasStatus(cs, "split-K FP32 GEMM failed");

          // Accumulate FP32 result into FP64.
          {
            dim3 block(256);
            dim3 grid(static_cast<unsigned int>((n * n + 255) / 256));
            AccumulateF32ToF64Kernel<<<grid, block, 0, stream>>>(
                vmat, gc.d_vmat_f32, n);
          }
        }
        return Status::Ok();
      }

      // Screened FP64 GEMM path: compact grid to only active points, then GEMM.
      // NAO basis is strictly zero beyond r_cut, so screening is exact.
      const bool use_screen = [] {
        const char* e = std::getenv("TIDES_GRID_SCREEN");
        return (e == nullptr || e[0] != '0');  // enabled by default
      }();

      if (use_screen && !use_splitk) {
        if (!gc.ensure_screen(input.nbasis, input.point_stride)) {
          // Screening alloc failed, fall through to full GEMM.
        } else
        // Build compaction index once (phi doesn't change across SCF iterations).
        if (!gc.screen_initialized) {
          cudaMemsetAsync(gc.d_screen_count, 0, sizeof(int), stream);
          {
            int threads = 256;
            int blocks = (static_cast<int>(input.np) + threads - 1) / threads;
            BuildCompactionIndexKernel<<<blocks, threads, 0, stream>>>(
                input.phi, input.nbasis, input.np, input.point_stride,
                gc.d_compact_index, gc.d_screen_count);
          }
          cudaMemcpyAsync(&gc.np_compact, gc.d_screen_count,
                          sizeof(int), cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          gc.screen_initialized = true;

          fprintf(stderr, "[screen] LDA np=%ld np_compact=%ld (%.1f%% active)\n",
                  (long)input.np, (long)gc.np_compact,
                  100.0 * gc.np_compact / input.np);

          // Precompute compacted phi.
          if (gc.np_compact > 0) {
            dim3 block(32, 4);
            dim3 grid((static_cast<unsigned int>(gc.np_compact) + block.x - 1) / block.x,
                      (static_cast<unsigned int>(input.nbasis) + block.y - 1) / block.y);
            CompactPhiKernel<<<grid, block, 0, stream>>>(
                gc.d_phi_compact, input.phi, gc.d_compact_index,
                input.nbasis, gc.np_compact, input.point_stride);
          }
        }

        if (gc.np_compact > 0 && gc.np_compact < input.np) {
          const int n = static_cast<int>(input.nbasis);
          const int k_compact = static_cast<int>(gc.np_compact);

          // Compact wv for this iteration.
          {
            int threads = 256;
            int blocks = (k_compact + threads - 1) / threads;
            CompactWvKernel<<<blocks, threads, 0, stream>>>(
                gc.d_wv_compact, input.wv, gc.d_compact_index, gc.np_compact);
          }

          // Scale compacted phi by compacted wv.
          {
            dim3 block(32, 4);
            dim3 grid((static_cast<unsigned int>(gc.np_compact) + block.x - 1) / block.x,
                      (static_cast<unsigned int>(input.nbasis) + block.y - 1) / block.y);
            ScaleColumnsKernel<<<grid, block, 0, stream>>>(
                gc.d_temp_compact, gc.d_phi_compact, gc.d_wv_compact,
                input.nbasis, gc.np_compact, gc.np_compact);
          }

          // GEMM: V = scale * Phi_compact^T * temp_compact
          const double alpha = input.scale;
          const double beta = 0.0;
          cublasStatus_t cs = cublasGemmEx(
              gc.handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              n, n, k_compact,
              &alpha,
              gc.d_phi_compact, CUDA_R_64F, static_cast<int>(gc.np_compact),
              gc.d_temp_compact, CUDA_R_64F, static_cast<int>(gc.np_compact),
              &beta,
              vmat, CUDA_R_64F, n,
              CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
          if (cs == CUBLAS_STATUS_SUCCESS)
            return Status::Ok();
        }
        // If screening didn't help (np_compact ≈ np), fall through to full GEMM.
      }

      // Full FP64 GEMM path.
      // Step 1: temp = wv * phi  (scale each column)
      {
        dim3 block(32, 4);
        dim3 grid((static_cast<unsigned int>(input.np) + block.x - 1) / block.x,
                  (static_cast<unsigned int>(input.nbasis) + block.y - 1) / block.y);
        ScaleColumnsKernel<<<grid, block, 0, stream>>>(
            gc.d_temp, input.phi, input.wv,
            input.nbasis, input.np, input.point_stride);
      }
      // Step 2: V = scale * Phi^T * temp  (col-major GEMM)
      const double alpha = input.scale;
      const double beta = 0.0;
      cublasStatus_t cs = cublasGemmEx(
          gc.handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          n, n, k,
          &alpha,
          input.phi, CUDA_R_64F, lda,
          gc.d_temp, CUDA_R_64F, lda,
          &beta,
          vmat, CUDA_R_64F, n,
          CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
      if (cs == CUBLAS_STATUS_SUCCESS)
        return Status::Ok();
    }
  }

  // Fallback: original per-(mu,nu) reduction kernel.
  dim3 grid_dim(static_cast<unsigned int>(input.nbasis),
                static_cast<unsigned int>(input.nbasis));
  WeightedVmatKernel<<<grid_dim, 256, 0, stream>>>(
      input.phi, input.wv, vmat, input.nbasis, input.np, input.point_stride,
      input.scale);
  return CudaStatus(cudaGetLastError(), "WeightedVmatKernel launch");
}

}  // namespace tides::grid
