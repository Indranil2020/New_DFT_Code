#include "grid/xc/tier1/mgga_m06_2x.hpp"
#include "grid/xc/tier1/util.h"
#include "grid/xc/tier1/mgga_pol_kernel.cuh"
#include "grid/xc/functionals/common.cuh"
#include "grid/xc/xc_engine.hpp"
#include "common/status.hpp"

#include <algorithm>
#include <cuda_runtime.h>

namespace tides::grid::xc::tier1 {

namespace mgga_x_m06_2x {
struct mgga_x_m05_params {
  const double a[12];
  double csi_HF;
  double cx;
};
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "mgga_exc/hyb_mgga_x_m05.c"
}  // namespace mgga_x_m06_2x

namespace mgga_c_m06_2x {
struct mgga_c_m06l_params {
  double gamma_ss;
  double gamma_ab;
  double alpha_ss;
  double alpha_ab;
  const double css[5];
  const double cab[5];
  const double dss[6];
  const double dab[6];
  double Fermi_D_cnst;
};
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "mgga_exc/mgga_c_m06l.c"
}  // namespace mgga_c_m06_2x

namespace {

__device__ MggaEvaluation MggaM06_2xEval(double rho, double sigma, double tau) {
  using tides::grid::xc::MggaEvaluation;

  mgga_x_m06_2x::mgga_x_m05_params x_params = {
      {4.600000e-01, -2.206052e-01, -9.431788e-02, 2.164494e+00,
       -2.556466e+00, -1.422133e+01, 1.555044e+01, 3.598078e+01,
       -2.722754e+01, -3.924093e+01, 1.522808e+01, 1.522227e+01},
      1.0,
      0.54};

  mgga_c_m06_2x::mgga_c_m06l_params c_params = {
      0.06, 0.0031, 0.00515088, 0.00304966,
      {3.097855e-01, -5.528642e+00, 1.347420e+01, -3.213623e+01, 2.846742e+01},
      {8.833596e-01, 3.357972e+01, -7.043548e+01, 4.978271e+01, -1.852891e+01},
      {6.902145e-01, 9.847204e-02, 2.214797e-01, -1.968264e-03, -6.775479e-03, 0.0},
      {1.166404e-01, -9.120847e-02, -6.726189e-02, 6.720580e-05, 8.448011e-04, 0.0},
      1e-10};

  xc_func_info_type x_info = {};
  x_info.flags = XC_FLAGS_3D | XC_FLAGS_NEEDS_TAU | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
  x_info.dens_threshold = 1e-15;

  xc_func_info_type c_info = {};
  c_info.flags = XC_FLAGS_3D | XC_FLAGS_NEEDS_TAU | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
  c_info.dens_threshold = 1e-12;

  xc_func_type x_func = {};
  x_func.info = &x_info;
  x_func.nspin = 1;
  x_func.dens_threshold = 1e-15;
  x_func.zeta_threshold = DBL_EPSILON;
  x_func.sigma_threshold = 1e-20;
  x_func.tau_threshold = 1e-20;
  x_func.dim = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  x_func.params = &x_params;

  xc_func_type c_func = {};
  c_func.info = &c_info;
  c_func.nspin = 1;
  c_func.dens_threshold = 1e-12;
  c_func.zeta_threshold = DBL_EPSILON;
  c_func.sigma_threshold = 1e-20;
  c_func.tau_threshold = 1e-20;
  c_func.dim = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  c_func.params = &c_params;

  double lapl = 0.0;
  double x_zk = 0.0, x_vrho = 0.0, x_vsigma = 0.0, x_vlapl = 0.0, x_vtau = 0.0;
  double c_zk = 0.0, c_vrho = 0.0, c_vsigma = 0.0, c_vlapl = 0.0, c_vtau = 0.0;

  xc_mgga_out_params x_out = {};
  x_out.zk = &x_zk;
  x_out.vrho = &x_vrho;
  x_out.vsigma = &x_vsigma;
  x_out.vlapl = &x_vlapl;
  x_out.vtau = &x_vtau;

  xc_mgga_out_params c_out = {};
  c_out.zk = &c_zk;
  c_out.vrho = &c_vrho;
  c_out.vsigma = &c_vsigma;
  c_out.vlapl = &c_vlapl;
  c_out.vtau = &c_vtau;

  double eval_rho = rho;
  double eval_tau = tau;
  double eval_sigma = sigma;
  if (eval_rho < x_func.dens_threshold) {
    eval_rho = x_func.dens_threshold;
  }
  if (eval_tau < x_func.tau_threshold) {
    eval_tau = x_func.tau_threshold;
  }
  const double sigma_floor = x_func.sigma_threshold * x_func.sigma_threshold;
  if (eval_sigma < sigma_floor) {
    eval_sigma = sigma_floor;
  }
  const double fhc_sigma = 8.0 * eval_rho * eval_tau;
  if (eval_sigma > fhc_sigma) {
    eval_sigma = fhc_sigma;
  }

  mgga_x_m06_2x::func_vxc_unpol(&x_func, 0, &eval_rho, &eval_sigma, &lapl, &eval_tau, &x_out);
  mgga_c_m06_2x::func_vxc_unpol(&c_func, 0, &eval_rho, &eval_sigma, &lapl, &eval_tau, &c_out);

  MggaEvaluation result = {};
  result.eps = x_zk + c_zk;
  result.vrho = x_vrho + c_vrho;
  result.vsigma = x_vsigma + c_vsigma;
  result.vtau = x_vtau + c_vtau;
  return result;
}

struct MggaM06_2xPolFunctor {
  __device__ MggaPolEvaluation operator()(const double* rho, const double* sigma,
                                          const double* lapl, const double* tau) const {
    using tides::grid::xc::MggaPolEvaluation;

    mgga_x_m06_2x::mgga_x_m05_params x_params = {
      {4.600000e-01, -2.206052e-01, -9.431788e-02, 2.164494e+00,
       -2.556466e+00, -1.422133e+01, 1.555044e+01, 3.598078e+01,
       -2.722754e+01, -3.924093e+01, 1.522808e+01, 1.522227e+01},
      1.0,
      0.54};

    mgga_c_m06_2x::mgga_c_m06l_params c_params = {
      0.06, 0.0031, 0.00515088, 0.00304966,
      {3.097855e-01, -5.528642e+00, 1.347420e+01, -3.213623e+01, 2.846742e+01},
      {8.833596e-01, 3.357972e+01, -7.043548e+01, 4.978271e+01, -1.852891e+01},
      {6.902145e-01, 9.847204e-02, 2.214797e-01, -1.968264e-03, -6.775479e-03, 0.0},
      {1.166404e-01, -9.120847e-02, -6.726189e-02, 6.720580e-05, 8.448011e-04, 0.0},
      1e-10};

    xc_func_info_type x_info = {};
    x_info.flags = XC_FLAGS_3D | XC_FLAGS_NEEDS_TAU | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
    x_info.dens_threshold = 1e-15;

    xc_func_info_type c_info = {};
    c_info.flags = XC_FLAGS_3D | XC_FLAGS_NEEDS_TAU | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
    c_info.dens_threshold = 1e-12;

    xc_func_type x_func = {};
    x_func.info = &x_info;
    x_func.nspin = 2;
    x_func.dens_threshold = 1e-15;
    x_func.zeta_threshold = DBL_EPSILON;
    x_func.sigma_threshold = 1e-20;
    x_func.tau_threshold = 1e-20;
    x_func.dim = {2, 3, 2, 2, 1, 2, 3, 2, 2};
    x_func.params = &x_params;

    xc_func_type c_func = {};
    c_func.info = &c_info;
    c_func.nspin = 2;
    c_func.dens_threshold = 1e-12;
    c_func.zeta_threshold = DBL_EPSILON;
    c_func.sigma_threshold = 1e-20;
    c_func.tau_threshold = 1e-20;
    c_func.dim = {2, 3, 2, 2, 1, 2, 3, 2, 2};
    c_func.params = &c_params;

    auto clamp = [](const double rho_in[2], const double sigma_in[3],
                    const double tau_in[2], const xc_func_type& func,
                    double rho_out[2], double sigma_out[3], double tau_out[2]) {
      rho_out[0] = rho_in[0] < func.dens_threshold ? func.dens_threshold : rho_in[0];
      rho_out[1] = rho_in[1] < func.dens_threshold ? func.dens_threshold : rho_in[1];
      tau_out[0] = tau_in[0] < func.tau_threshold ? func.tau_threshold : tau_in[0];
      tau_out[1] = tau_in[1] < func.tau_threshold ? func.tau_threshold : tau_in[1];
      const double sigma_floor = func.sigma_threshold * func.sigma_threshold;
      sigma_out[0] = sigma_in[0] < sigma_floor ? sigma_floor : sigma_in[0];
      sigma_out[2] = sigma_in[2] < sigma_floor ? sigma_floor : sigma_in[2];
      sigma_out[1] = sigma_in[1];
      const double s_ave = 0.5 * (sigma_out[0] + sigma_out[2]);
      if (sigma_out[1] < -s_ave) sigma_out[1] = -s_ave;
      if (sigma_out[1] > +s_ave) sigma_out[1] = +s_ave;
      const double fhc0 = 8.0 * rho_out[0] * tau_out[0];
      const double fhc2 = 8.0 * rho_out[1] * tau_out[1];
      if (sigma_out[0] > fhc0) sigma_out[0] = fhc0;
      if (sigma_out[2] > fhc2) sigma_out[2] = fhc2;
    };

    double my_rho[2] = {};
    double my_sigma[3] = {};
    double my_tau[2] = {};

    double x_zk = 0.0, x_vrho[2] = {}, x_vsigma[3] = {}, x_vlapl[2] = {}, x_vtau[2] = {};
    xc_mgga_out_params x_out = {};
    x_out.zk = &x_zk;
    x_out.vrho = x_vrho;
    x_out.vsigma = x_vsigma;
    x_out.vlapl = x_vlapl;
    x_out.vtau = x_vtau;
    clamp(rho, sigma, tau, x_func, my_rho, my_sigma, my_tau);
    mgga_x_m06_2x::func_vxc_pol(&x_func, 0, my_rho, my_sigma, lapl, my_tau, &x_out);

    double c_zk = 0.0, c_vrho[2] = {}, c_vsigma[3] = {}, c_vlapl[2] = {}, c_vtau[2] = {};
    xc_mgga_out_params c_out = {};
    c_out.zk = &c_zk;
    c_out.vrho = c_vrho;
    c_out.vsigma = c_vsigma;
    c_out.vlapl = c_vlapl;
    c_out.vtau = c_vtau;
    clamp(rho, sigma, tau, c_func, my_rho, my_sigma, my_tau);
    mgga_c_m06_2x::func_vxc_pol(&c_func, 0, my_rho, my_sigma, lapl, my_tau, &c_out);

    MggaPolEvaluation result = {};
    result.eps = x_zk + c_zk;
    result.vrho[0] = x_vrho[0] + c_vrho[0];
    result.vrho[1] = x_vrho[1] + c_vrho[1];
    result.vsigma[0] = x_vsigma[0] + c_vsigma[0];
    result.vsigma[1] = x_vsigma[1] + c_vsigma[1];
    result.vsigma[2] = x_vsigma[2] + c_vsigma[2];
    result.vtau[0] = x_vtau[0] + c_vtau[0];
    result.vtau[1] = x_vtau[1] + c_vtau[1];
    return result;
  }
};

constexpr int kThreads = 128;

__global__ __launch_bounds__(128)
void MggaM06_2xKernel(const double* __restrict__ rho,
                                 const double* __restrict__ grad,
                                 const double* __restrict__ tau,
                                 const double* __restrict__ weights,
                                 double* __restrict__ wv_rho,
                                 double* __restrict__ wv_grad,
                                 double* __restrict__ wv_tau,
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
    MggaEvaluation evaluation = MggaM06_2xEval(density, sigma, tau[point]);
    const double weighted_gradient = 2.0 * weight * evaluation.vsigma;
    wv_grad[point] = weighted_gradient * gx;
    wv_grad[point_stride + point] = weighted_gradient * gy;
    wv_grad[2 * point_stride + point] = weighted_gradient * gz;
    wv_rho[point] = weight * evaluation.vrho;
    wv_tau[point] = weight * evaluation.vtau;
    if (fast_reduction) local_energy += weight * density * evaluation.eps;
  }
  if (!fast_reduction) return;
  for (int offset = 16; offset > 0; offset /= 2) {
    local_energy += __shfl_down_sync(0xffffffff, local_energy, offset);
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(exc, local_energy);
}

__global__ void MggaM06_2xDeterministicEnergyKernel(const double* __restrict__ rho,
                                                    const double* __restrict__ grad,
                                                    const double* __restrict__ tau,
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
    MggaEvaluation evaluation = MggaM06_2xEval(density, sigma, tau[point]);
    energy += weights[point] * density * evaluation.eps;
  }
  exc[0] = energy;
}

}  // namespace

Status LaunchMggaM06_2x(const XcGridIn& input, XcGridOut& output,
                        cudaStream_t stream, bool deterministic, int nspin) {
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  if (nspin == 2) {
    MggaPolKernelScalar<MggaM06_2xPolFunctor><<<blocks, kThreads, 0, stream>>>(
        input.rho, input.grad, input.tau, input.w, output.wv_rho,
        output.wv_tau, output.exc_per_system, input.np,
        input.point_stride, !deterministic);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("MggaM06_2xPolKernelScalar launch");
    MggaPolKernelGrad<MggaM06_2xPolFunctor><<<blocks, kThreads, 0, stream>>>(
        input.rho, input.grad, input.tau, input.w, output.wv_grad,
        input.np, input.point_stride);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("MggaM06_2xPolKernelGrad launch");
    if (!deterministic) return Status::Ok();
    MggaPolDeterministicEnergyKernel<MggaM06_2xPolFunctor><<<1, 1, 0, stream>>>(
        input.rho, input.grad, input.tau, input.w, output.exc_per_system,
        input.np, input.point_stride);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("MggaM06_2xPolDeterministicEnergyKernel launch");
    return Status::Ok();
  }
  MggaM06_2xKernel<<<blocks, kThreads, 0, stream>>>(
      input.rho, input.grad, input.tau, input.w, output.wv_rho,
      output.wv_grad, output.wv_tau, output.exc_per_system, input.np,
      input.point_stride, !deterministic);
  if (cudaGetLastError() != cudaSuccess) return Status::IoError("MggaM06_2xKernel launch");
  if (!deterministic) return Status::Ok();
  MggaM06_2xDeterministicEnergyKernel<<<1, 1, 0, stream>>>(
      input.rho, input.grad, input.tau, input.w, output.exc_per_system,
      input.np, input.point_stride);
  if (cudaGetLastError() != cudaSuccess) return Status::IoError("MggaM06_2xDeterministicEnergyKernel launch");
  return Status::Ok();
}

}  // namespace tides::grid::xc::tier1
