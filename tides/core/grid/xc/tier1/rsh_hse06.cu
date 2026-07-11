#include "grid/xc/tier1/rsh_hse06.hpp"
#include "grid/xc/tier1/util.h"
#include "grid/xc/tier1/special_functions.cuh"
#include "grid/xc/functionals/common.cuh"
#include "grid/xc/xc_engine.hpp"
#include "common/status.hpp"

#include <algorithm>
#include <cuda_runtime.h>

namespace tides::grid::xc::tier1 {

namespace gga_x_wpbeh {
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "gga_exc/gga_x_wpbeh.c"
}  // namespace gga_x_wpbeh

namespace gga_c_pbe {
struct gga_c_pbe_params {
  double beta, gamma, BB;
};
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "gga_exc/gga_c_pbe.c"
}  // namespace gga_c_pbe

namespace {

__device__ GgaEvaluation Hse06Eval(double rho, double sigma) {
  using tides::grid::xc::GgaEvaluation;

  xc_func_info_type x_info = {};
  x_info.flags = XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
  x_info.dens_threshold = 1e-14;

  xc_func_type x_func_0 = {};
  x_func_0.info = &x_info;
  x_func_0.nspin = 1;
  x_func_0.dens_threshold = 1e-14;
  x_func_0.zeta_threshold = DBL_EPSILON;
  x_func_0.sigma_threshold = 1e-20;
  x_func_0.dim = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  x_func_0.cam_omega = 0.0;

  xc_func_type x_func_pbe = {};
  x_func_pbe.info = &x_info;
  x_func_pbe.nspin = 1;
  x_func_pbe.dens_threshold = 1e-14;
  x_func_pbe.zeta_threshold = DBL_EPSILON;
  x_func_pbe.sigma_threshold = 1e-20;
  x_func_pbe.dim = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  x_func_pbe.cam_omega = 0.11;

  gga_c_pbe::gga_c_pbe_params c_params = {
      0.06672455060314922, 0.031090690869654895034, 1.0};

  xc_func_info_type c_info = {};
  c_info.flags = XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
  c_info.dens_threshold = 1e-12;

  xc_func_type c_func = {};
  c_func.info = &c_info;
  c_func.nspin = 1;
  c_func.dens_threshold = 1e-12;
  c_func.zeta_threshold = DBL_EPSILON;
  c_func.sigma_threshold = 1e-20;
  c_func.dim = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  c_func.params = &c_params;

  auto clamp = [](double rho_in, double sigma_in, const xc_func_type& func,
                  double& rho_out, double& sigma_out) {
    rho_out = rho_in;
    sigma_out = sigma_in;
    if (rho_out < func.dens_threshold) rho_out = func.dens_threshold;
    const double sigma_floor = func.sigma_threshold * func.sigma_threshold;
    if (sigma_out < sigma_floor) sigma_out = sigma_floor;
  };

  double eval_rho = 0.0, eval_sigma = 0.0;

  double x0_zk = 0.0, x0_vrho = 0.0, x0_vsigma = 0.0;
  xc_gga_out_params x0_out = {};
  x0_out.zk = &x0_zk;
  x0_out.vrho = &x0_vrho;
  x0_out.vsigma = &x0_vsigma;
  clamp(rho, sigma, x_func_0, eval_rho, eval_sigma);
  gga_x_wpbeh::func_vxc_unpol(&x_func_0, 0, &eval_rho, &eval_sigma, &x0_out);

  double x1_zk = 0.0, x1_vrho = 0.0, x1_vsigma = 0.0;
  xc_gga_out_params x1_out = {};
  x1_out.zk = &x1_zk;
  x1_out.vrho = &x1_vrho;
  x1_out.vsigma = &x1_vsigma;
  clamp(rho, sigma, x_func_pbe, eval_rho, eval_sigma);
  gga_x_wpbeh::func_vxc_unpol(&x_func_pbe, 0, &eval_rho, &eval_sigma, &x1_out);

  double c_zk = 0.0, c_vrho = 0.0, c_vsigma = 0.0;
  xc_gga_out_params c_out = {};
  c_out.zk = &c_zk;
  c_out.vrho = &c_vrho;
  c_out.vsigma = &c_vsigma;
  clamp(rho, sigma, c_func, eval_rho, eval_sigma);
  gga_c_pbe::func_vxc_unpol(&c_func, 0, &eval_rho, &eval_sigma, &c_out);

  constexpr double beta = 0.25;
  GgaEvaluation result = {};
  result.eps = x0_zk - beta * x1_zk + c_zk;
  result.vrho = x0_vrho - beta * x1_vrho + c_vrho;
  result.vsigma = x0_vsigma - beta * x1_vsigma + c_vsigma;
  return result;
}

constexpr int kThreads = 128;

__device__ GgaPolEvaluation Hse06PolEval(const double* rho, const double* sigma) {
  using tides::grid::xc::GgaPolEvaluation;

  xc_func_info_type x_info = {};
  x_info.flags = XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
  x_info.dens_threshold = 1e-14;

  xc_func_type x_func_0 = {};
  x_func_0.info = &x_info;
  x_func_0.nspin = 2;
  x_func_0.dens_threshold = 1e-14;
  x_func_0.zeta_threshold = DBL_EPSILON;
  x_func_0.sigma_threshold = 1e-20;
  x_func_0.dim = {2, 3, 1, 1, 1, 2, 3, 1, 1};
  x_func_0.cam_omega = 0.0;

  xc_func_type x_func_pbe = {};
  x_func_pbe.info = &x_info;
  x_func_pbe.nspin = 2;
  x_func_pbe.dens_threshold = 1e-14;
  x_func_pbe.zeta_threshold = DBL_EPSILON;
  x_func_pbe.sigma_threshold = 1e-20;
  x_func_pbe.dim = {2, 3, 1, 1, 1, 2, 3, 1, 1};
  x_func_pbe.cam_omega = 0.11;

  gga_c_pbe::gga_c_pbe_params c_params = {
      0.06672455060314922, 0.031090690869654895034, 1.0};

  xc_func_info_type c_info = {};
  c_info.flags = XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
  c_info.dens_threshold = 1e-12;

  xc_func_type c_func = {};
  c_func.info = &c_info;
  c_func.nspin = 2;
  c_func.dens_threshold = 1e-12;
  c_func.zeta_threshold = DBL_EPSILON;
  c_func.sigma_threshold = 1e-20;
  c_func.dim = {2, 3, 1, 1, 1, 2, 3, 1, 1};
  c_func.params = &c_params;

  auto clamp = [](const double rho_in[2], const double sigma_in[3],
                  const xc_func_type& func, double rho_out[2],
                  double sigma_out[3]) {
    rho_out[0] = rho_in[0] < func.dens_threshold ? func.dens_threshold : rho_in[0];
    rho_out[1] = rho_in[1] < func.dens_threshold ? func.dens_threshold : rho_in[1];
    const double sigma_floor = func.sigma_threshold * func.sigma_threshold;
    sigma_out[0] = sigma_in[0] < sigma_floor ? sigma_floor : sigma_in[0];
    sigma_out[2] = sigma_in[2] < sigma_floor ? sigma_floor : sigma_in[2];
    sigma_out[1] = sigma_in[1];
    const double s_ave = 0.5 * (sigma_out[0] + sigma_out[2]);
    if (sigma_out[1] < -s_ave) sigma_out[1] = -s_ave;
    if (sigma_out[1] > +s_ave) sigma_out[1] = +s_ave;
  };

  double my_rho[2] = {};
  double my_sigma[3] = {};

  double x0_zk = 0.0, x0_vrho[2] = {}, x0_vsigma[3] = {};
  xc_gga_out_params x0_out = {};
  x0_out.zk = &x0_zk;
  x0_out.vrho = x0_vrho;
  x0_out.vsigma = x0_vsigma;
  clamp(rho, sigma, x_func_0, my_rho, my_sigma);
  gga_x_wpbeh::func_vxc_pol(&x_func_0, 0, my_rho, my_sigma, &x0_out);

  double x1_zk = 0.0, x1_vrho[2] = {}, x1_vsigma[3] = {};
  xc_gga_out_params x1_out = {};
  x1_out.zk = &x1_zk;
  x1_out.vrho = x1_vrho;
  x1_out.vsigma = x1_vsigma;
  clamp(rho, sigma, x_func_pbe, my_rho, my_sigma);
  gga_x_wpbeh::func_vxc_pol(&x_func_pbe, 0, my_rho, my_sigma, &x1_out);

  double c_zk = 0.0, c_vrho[2] = {}, c_vsigma[3] = {};
  xc_gga_out_params c_out = {};
  c_out.zk = &c_zk;
  c_out.vrho = c_vrho;
  c_out.vsigma = c_vsigma;
  clamp(rho, sigma, c_func, my_rho, my_sigma);
  gga_c_pbe::func_vxc_pol(&c_func, 0, my_rho, my_sigma, &c_out);

  constexpr double beta = 0.25;
  GgaPolEvaluation result = {};
  result.eps = x0_zk - beta * x1_zk + c_zk;
  result.vrho[0] = x0_vrho[0] - beta * x1_vrho[0] + c_vrho[0];
  result.vrho[1] = x0_vrho[1] - beta * x1_vrho[1] + c_vrho[1];
  result.vsigma[0] = x0_vsigma[0] - beta * x1_vsigma[0] + c_vsigma[0];
  result.vsigma[1] = x0_vsigma[1] - beta * x1_vsigma[1] + c_vsigma[1];
  result.vsigma[2] = x0_vsigma[2] - beta * x1_vsigma[2] + c_vsigma[2];
  return result;
}

__global__ __launch_bounds__(128)
void Hse06Kernel(const double* __restrict__ rho,
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
    GgaEvaluation evaluation = Hse06Eval(density, sigma);
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
void Hse06PolKernelScalar(const double* __restrict__ rho,
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
    GgaPolEvaluation evaluation = Hse06PolEval(rho_in, sigma_in);
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
void Hse06PolKernelGrad(const double* __restrict__ rho,
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
    GgaPolEvaluation evaluation = Hse06PolEval(rho_in, sigma_in);
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

__global__ void Hse06PolDeterministicEnergyKernel(const double* __restrict__ rho,
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
    GgaPolEvaluation evaluation = Hse06PolEval(rho_in, sigma_in);
    energy += weights[point] * (rho_up + rho_down) * evaluation.eps;
  }
  exc[0] = energy;
}

__global__ void Hse06DeterministicEnergyKernel(const double* __restrict__ rho,
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
    GgaEvaluation evaluation = Hse06Eval(density, sigma);
    energy += weights[point] * density * evaluation.eps;
  }
  exc[0] = energy;
}

}  // namespace

Status LaunchRshHse06(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin) {
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  if (nspin == 2) {
    Hse06PolKernelScalar<<<blocks, kThreads, 0, stream>>>(
        input.rho, input.grad, input.w, output.wv_rho,
        output.exc_per_system, input.np, input.point_stride, !deterministic);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("Hse06PolKernelScalar launch");
    Hse06PolKernelGrad<<<blocks, kThreads, 0, stream>>>(
        input.rho, input.grad, input.w, output.wv_grad,
        input.np, input.point_stride);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("Hse06PolKernelGrad launch");
    if (!deterministic) return Status::Ok();
    Hse06PolDeterministicEnergyKernel<<<1, 1, 0, stream>>>(
        input.rho, input.grad, input.w, output.exc_per_system, input.np,
        input.point_stride);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("Hse06PolDeterministicEnergyKernel launch");
    return Status::Ok();
  }
  Hse06Kernel<<<blocks, kThreads, 0, stream>>>(
      input.rho, input.grad, input.w, output.wv_rho, output.wv_grad,
      output.exc_per_system, input.np, input.point_stride, !deterministic);
  if (cudaGetLastError() != cudaSuccess) return Status::IoError("Hse06Kernel launch");
  if (!deterministic) return Status::Ok();
  Hse06DeterministicEnergyKernel<<<1, 1, 0, stream>>>(
      input.rho, input.grad, input.w, output.exc_per_system, input.np,
      input.point_stride);
  if (cudaGetLastError() != cudaSuccess) return Status::IoError("Hse06DeterministicEnergyKernel launch");
  return Status::Ok();
}

}  // namespace tides::grid::xc::tier1
