#include "grid/xc/tier1/rsh_wb97x.hpp"
#include "grid/xc/tier1/util.h"
#include "grid/xc/tier1/special_functions.cuh"
#include "grid/xc/functionals/common.cuh"
#include "grid/xc/xc_engine.hpp"
#include "common/status.hpp"

#include <algorithm>
#include <cuda_runtime.h>

namespace tides::grid::xc::tier1 {

namespace gga_xc_wb97 {
struct gga_xc_wb97_params {
  double c_x[5];
  double c_ss[5];
  double c_ab[5];
};
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "/home/niel/src/libxc-7.0.0/src/maple2c/gga_exc/hyb_gga_xc_wb97.c"
}  // namespace gga_xc_wb97

namespace {

__device__ GgaEvaluation Wb97xEval(double rho, double sigma) {
  using tides::grid::xc::GgaEvaluation;

  gga_xc_wb97::gga_xc_wb97_params params = {
      {8.42294e-01, 7.26479e-01, 1.04760e+00, -5.70635e+00, 1.32794e+01},
      {1.00000e+00, -4.33879e+00, 1.82308e+01, -3.17430e+01, 1.72901e+01},
      {1.00000e+00, 2.37031e+00, -1.13995e+01, 6.58405e+00, -3.78132e+00}};

  xc_func_info_type info = {};
  info.flags = XC_FLAGS_3D | XC_FLAGS_HYB_CAM | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
  info.dens_threshold = 1e-14;

  xc_func_type func = {};
  func.info = &info;
  func.nspin = 1;
  func.dens_threshold = 1e-14;
  func.zeta_threshold = DBL_EPSILON;
  func.sigma_threshold = 1e-20;
  func.cam_omega = 0.3;
  func.cam_alpha = 1.0;
  func.cam_beta = -0.842294;
  func.dim = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  func.params = &params;

  double eval_rho = rho;
  double eval_sigma = sigma;
  if (eval_rho < func.dens_threshold) {
    eval_rho = func.dens_threshold;
  }
  const double sigma_floor = func.sigma_threshold * func.sigma_threshold;
  if (eval_sigma < sigma_floor) {
    eval_sigma = sigma_floor;
  }

  double zk = 0.0, vrho = 0.0, vsigma = 0.0;
  xc_gga_out_params out = {};
  out.zk = &zk;
  out.vrho = &vrho;
  out.vsigma = &vsigma;

  gga_xc_wb97::func_vxc_unpol(&func, 0, &eval_rho, &eval_sigma, &out);

  GgaEvaluation result = {};
  result.eps = zk;
  result.vrho = vrho;
  result.vsigma = vsigma;
  return result;
}

constexpr int kThreads = 128;

__device__ GgaPolEvaluation Wb97xPolEval(const double* rho, const double* sigma) {
  using tides::grid::xc::GgaPolEvaluation;

  gga_xc_wb97::gga_xc_wb97_params params = {
      {8.42294e-01, 7.26479e-01, 1.04760e+00, -5.70635e+00, 1.32794e+01},
      {1.00000e+00, -4.33879e+00, 1.82308e+01, -3.17430e+01, 1.72901e+01},
      {1.00000e+00, 2.37031e+00, -1.13995e+01, 6.58405e+00, -3.78132e+00}};

  xc_func_info_type info = {};
  info.flags = XC_FLAGS_3D | XC_FLAGS_HYB_CAM | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
  info.dens_threshold = 1e-14;

  xc_func_type func = {};
  func.info = &info;
  func.nspin = 2;
  func.dens_threshold = 1e-14;
  func.zeta_threshold = DBL_EPSILON;
  func.sigma_threshold = 1e-20;
  func.cam_omega = 0.3;
  func.cam_alpha = 1.0;
  func.cam_beta = -0.842294;
  func.dim = {2, 3, 1, 1, 1, 2, 3, 1, 1};
  func.params = &params;

  double my_rho[2] = {};
  double my_sigma[3] = {};
  my_rho[0] = rho[0] < func.dens_threshold ? func.dens_threshold : rho[0];
  my_rho[1] = rho[1] < func.dens_threshold ? func.dens_threshold : rho[1];
  const double sigma_floor = func.sigma_threshold * func.sigma_threshold;
  my_sigma[0] = sigma[0] < sigma_floor ? sigma_floor : sigma[0];
  my_sigma[2] = sigma[2] < sigma_floor ? sigma_floor : sigma[2];
  my_sigma[1] = sigma[1];
  const double s_ave = 0.5 * (my_sigma[0] + my_sigma[2]);
  if (my_sigma[1] < -s_ave) my_sigma[1] = -s_ave;
  if (my_sigma[1] > +s_ave) my_sigma[1] = +s_ave;

  double zk = 0.0, vrho[2] = {}, vsigma[3] = {};
  xc_gga_out_params out = {};
  out.zk = &zk;
  out.vrho = vrho;
  out.vsigma = vsigma;

  gga_xc_wb97::func_vxc_pol(&func, 0, my_rho, my_sigma, &out);

  GgaPolEvaluation result = {};
  result.eps = zk;
  result.vrho[0] = vrho[0];
  result.vrho[1] = vrho[1];
  result.vsigma[0] = vsigma[0];
  result.vsigma[1] = vsigma[1];
  result.vsigma[2] = vsigma[2];
  return result;
}

__global__ __launch_bounds__(128)
void Wb97xKernel(const double* __restrict__ rho,
                            const double* __restrict__ grad,
                            const double* __restrict__ weights,
                            double* __restrict__ wv_rho,
                            double* __restrict__ wv_grad,
                            double* __restrict__ exc,
                            std::int64_t np, std::int64_t point_stride,
                            bool fast_reduction) {
  double local_energy = 0.0;
  for (std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < np; point += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const double density = rho[point];
    const double weight = weights[point];
    const double gx = grad[point];
    const double gy = grad[point_stride + point];
    const double gz = grad[2 * point_stride + point];
    const double sigma = gx * gx + gy * gy + gz * gz;
    GgaEvaluation evaluation = Wb97xEval(density, sigma);
    const double weighted_gradient = 2.0 * weight * evaluation.vsigma;
    wv_grad[point] = weighted_gradient * gx;
    wv_grad[point_stride + point] = weighted_gradient * gy;
    wv_grad[2 * point_stride + point] = weighted_gradient * gz;
    wv_rho[point] = weight * evaluation.vrho;
    if (fast_reduction) local_energy += weight * density * evaluation.eps;
  }
  if (!fast_reduction) return;
  for (int offset = 16; offset > 0; offset /= 2) {
    local_energy += __shfl_down_sync(0xffffffff, local_energy, offset);
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(exc, local_energy);
}

__global__ __launch_bounds__(128)
void Wb97xPolKernelScalar(const double* __restrict__ rho,
                               const double* __restrict__ grad,
                               const double* __restrict__ weights,
                               double* __restrict__ wv_rho,
                               double* __restrict__ exc,
                               std::int64_t np, std::int64_t point_stride,
                               bool fast_reduction) {
  double local_energy = 0.0;
  for (std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < np; point += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const double rho_up = rho[point];
    const double rho_down = rho[point_stride + point];
    const double weight = weights[point];
    const double gx_up = grad[point];
    const double gy_up = grad[point_stride + point];
    const double gz_up = grad[2 * point_stride + point];
    const double gx_down = grad[3 * point_stride + point];
    const double gy_down = grad[4 * point_stride + point];
    const double gz_down = grad[5 * point_stride + point];
    const double rho_in[2] = {rho_up, rho_down};
    const double sigma_in[3] = {
        gx_up * gx_up + gy_up * gy_up + gz_up * gz_up,
        gx_up * gx_down + gy_up * gy_down + gz_up * gz_down,
        gx_down * gx_down + gy_down * gy_down + gz_down * gz_down};
    GgaPolEvaluation evaluation = Wb97xPolEval(rho_in, sigma_in);
    wv_rho[point] = weight * evaluation.vrho[0];
    wv_rho[point_stride + point] = weight * evaluation.vrho[1];
    if (fast_reduction) local_energy += weight * (rho_up + rho_down) * evaluation.eps;
  }
  if (!fast_reduction) return;
  for (int offset = 16; offset > 0; offset /= 2) {
    local_energy += __shfl_down_sync(0xffffffff, local_energy, offset);
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(exc, local_energy);
}

__global__ __launch_bounds__(128)
void Wb97xPolKernelGrad(const double* __restrict__ rho,
                               const double* __restrict__ grad,
                               const double* __restrict__ weights,
                               double* __restrict__ wv_grad,
                               std::int64_t np, std::int64_t point_stride) {
  for (std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < np; point += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const double weight = weights[point];
    const double gx_up = grad[point];
    const double gy_up = grad[point_stride + point];
    const double gz_up = grad[2 * point_stride + point];
    const double gx_down = grad[3 * point_stride + point];
    const double gy_down = grad[4 * point_stride + point];
    const double gz_down = grad[5 * point_stride + point];
    const double rho_up = rho[point];
    const double rho_down = rho[point_stride + point];
    const double rho_in[2] = {rho_up, rho_down};
    const double sigma_in[3] = {
        gx_up * gx_up + gy_up * gy_up + gz_up * gz_up,
        gx_up * gx_down + gy_up * gy_down + gz_up * gz_down,
        gx_down * gx_down + gy_down * gy_down + gz_down * gz_down};
    GgaPolEvaluation evaluation = Wb97xPolEval(rho_in, sigma_in);
    const double wv0 = 2.0 * weight * evaluation.vsigma[0];
    const double wv1 = weight * evaluation.vsigma[1];
    const double wv2 = 2.0 * weight * evaluation.vsigma[2];
    wv_grad[point] = wv0 * gx_up + wv1 * gx_down;
    wv_grad[point_stride + point] = wv0 * gy_up + wv1 * gy_down;
    wv_grad[2 * point_stride + point] = wv0 * gz_up + wv1 * gz_down;
    wv_grad[3 * point_stride + point] = wv1 * gx_up + wv2 * gx_down;
    wv_grad[4 * point_stride + point] = wv1 * gy_up + wv2 * gy_down;
    wv_grad[5 * point_stride + point] = wv1 * gz_up + wv2 * gz_down;
  }
}

__global__ void Wb97xPolDeterministicEnergyKernel(const double* __restrict__ rho,
                                                  const double* __restrict__ grad,
                                                  const double* __restrict__ weights,
                                                  double* __restrict__ exc,
                                                  std::int64_t np,
                                                  std::int64_t point_stride) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double energy = 0.0;
  for (std::int64_t point = 0; point < np; ++point) {
    const double rho_up = rho[point];
    const double rho_down = rho[point_stride + point];
    const double gx_up = grad[point];
    const double gy_up = grad[point_stride + point];
    const double gz_up = grad[2 * point_stride + point];
    const double gx_down = grad[3 * point_stride + point];
    const double gy_down = grad[4 * point_stride + point];
    const double gz_down = grad[5 * point_stride + point];
    const double rho_in[2] = {rho_up, rho_down};
    const double sigma_in[3] = {
        gx_up * gx_up + gy_up * gy_up + gz_up * gz_up,
        gx_up * gx_down + gy_up * gy_down + gz_up * gz_down,
        gx_down * gx_down + gy_down * gy_down + gz_down * gz_down};
    GgaPolEvaluation evaluation = Wb97xPolEval(rho_in, sigma_in);
    energy += weights[point] * (rho_up + rho_down) * evaluation.eps;
  }
  exc[0] = energy;
}

__global__ void Wb97xDeterministicEnergyKernel(const double* __restrict__ rho,
                                               const double* __restrict__ grad,
                                               const double* __restrict__ weights,
                                               double* __restrict__ exc,
                                               std::int64_t np,
                                               std::int64_t point_stride) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double energy = 0.0;
  for (std::int64_t point = 0; point < np; ++point) {
    const double density = rho[point];
    const double gx = grad[point];
    const double gy = grad[point_stride + point];
    const double gz = grad[2 * point_stride + point];
    const double sigma = gx * gx + gy * gy + gz * gz;
    GgaEvaluation evaluation = Wb97xEval(density, sigma);
    energy += weights[point] * density * evaluation.eps;
  }
  exc[0] = energy;
}

}  // namespace

Status LaunchRshWb97x(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin) {
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  if (nspin == 2) {
    Wb97xPolKernelScalar<<<blocks, kThreads, 0, stream>>>(
        input.rho, input.grad, input.w, output.wv_rho,
        output.exc_per_system, input.np, input.point_stride, !deterministic);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("Wb97xPolKernelScalar launch");
    Wb97xPolKernelGrad<<<blocks, kThreads, 0, stream>>>(
        input.rho, input.grad, input.w, output.wv_grad,
        input.np, input.point_stride);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("Wb97xPolKernelGrad launch");
    if (!deterministic) return Status::Ok();
    Wb97xPolDeterministicEnergyKernel<<<1, 1, 0, stream>>>(
        input.rho, input.grad, input.w, output.exc_per_system, input.np,
        input.point_stride);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("Wb97xPolDeterministicEnergyKernel launch");
    return Status::Ok();
  }
  Wb97xKernel<<<blocks, kThreads, 0, stream>>>(
      input.rho, input.grad, input.w, output.wv_rho, output.wv_grad,
      output.exc_per_system, input.np, input.point_stride, !deterministic);
  if (cudaGetLastError() != cudaSuccess) return Status::IoError("Wb97xKernel launch");
  if (!deterministic) return Status::Ok();
  Wb97xDeterministicEnergyKernel<<<1, 1, 0, stream>>>(
      input.rho, input.grad, input.w, output.exc_per_system, input.np,
      input.point_stride);
  if (cudaGetLastError() != cudaSuccess) return Status::IoError("Wb97xDeterministicEnergyKernel launch");
  return Status::Ok();
}

}  // namespace tides::grid::xc::tier1
