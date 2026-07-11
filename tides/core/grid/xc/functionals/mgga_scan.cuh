#pragma once

// T-X4.4: Header-only mGGA SCAN device functor.

#include "grid/xc/functionals/common.cuh"
#include "grid/xc/tier1/util.h"

namespace tides::grid::xc {

namespace mgga_x_scan {
struct mgga_x_scan_params {
  double c1, c2, d, k1;
};
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "mgga_exc/mgga_x_scan.c"
}  // namespace mgga_x_scan

namespace mgga_c_scan {
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "mgga_exc/mgga_c_scan.c"
}  // namespace mgga_c_scan

struct MggaScanFunctor {
  static __device__ MggaEvaluation Eval(double rho, double sigma, double tau) {
    mgga_x_scan::mgga_x_scan_params x_params = {0.667, 0.8, 1.24, 0.065};

    xc_func_info_type x_info = {};
    x_info.flags = XC_FLAGS_3D | XC_FLAGS_NEEDS_TAU | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
    x_info.dens_threshold = 1e-15;

    xc_func_info_type c_info = {};
    c_info.flags = XC_FLAGS_3D | XC_FLAGS_NEEDS_TAU | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
    c_info.dens_threshold = 1e-15;

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
    c_func.dens_threshold = 1e-15;
    c_func.zeta_threshold = DBL_EPSILON;
    c_func.sigma_threshold = 1e-20;
    c_func.tau_threshold = 1e-20;
    c_func.dim = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    c_func.params = nullptr;

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

    mgga_x_scan::func_vxc_unpol(&x_func, 0, &eval_rho, &eval_sigma, &lapl, &eval_tau, &x_out);
    mgga_c_scan::func_vxc_unpol(&c_func, 0, &eval_rho, &eval_sigma, &lapl, &eval_tau, &c_out);

    MggaEvaluation result = {};
    result.eps = x_zk + c_zk;
    result.vrho = x_vrho + c_vrho;
    result.vsigma = x_vsigma + c_vsigma;
    result.vtau = x_vtau + c_vtau;
    return result;
  }
};

struct MggaScanPolFunctor {
  __device__ MggaPolEvaluation operator()(const double* rho, const double* sigma,
                                          const double* lapl, const double* tau) const {
    mgga_x_scan::mgga_x_scan_params x_params = {0.667, 0.8, 1.24, 0.065};

    xc_func_info_type x_info = {};
    x_info.flags = XC_FLAGS_3D | XC_FLAGS_NEEDS_TAU | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
    x_info.dens_threshold = 1e-15;

    xc_func_info_type c_info = {};
    c_info.flags = XC_FLAGS_3D | XC_FLAGS_NEEDS_TAU | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
    c_info.dens_threshold = 1e-15;

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
    c_func.dens_threshold = 1e-15;
    c_func.zeta_threshold = DBL_EPSILON;
    c_func.sigma_threshold = 1e-20;
    c_func.tau_threshold = 1e-20;
    c_func.dim = {2, 3, 2, 2, 1, 2, 3, 2, 2};
    c_func.params = nullptr;

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
    mgga_x_scan::func_vxc_pol(&x_func, 0, my_rho, my_sigma, lapl, my_tau, &x_out);

    double c_zk = 0.0, c_vrho[2] = {}, c_vsigma[3] = {}, c_vlapl[2] = {}, c_vtau[2] = {};
    xc_mgga_out_params c_out = {};
    c_out.zk = &c_zk;
    c_out.vrho = c_vrho;
    c_out.vsigma = c_vsigma;
    c_out.vlapl = c_vlapl;
    c_out.vtau = c_vtau;
    clamp(rho, sigma, tau, c_func, my_rho, my_sigma, my_tau);
    mgga_c_scan::func_vxc_pol(&c_func, 0, my_rho, my_sigma, lapl, my_tau, &c_out);

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

}  // namespace tides::grid::xc
