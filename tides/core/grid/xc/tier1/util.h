#pragma once

// Minimal libxc-compatible shim for including the generated maple2c device
// functional files.  This is only used inside the tier1 device adapter.

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <cassert>

#ifdef __CUDACC__
#define GPU_FUNCTION __host__ __device__
#define GPU_DEVICE_FUNCTION __device__
#else
#define GPU_FUNCTION
#define GPU_DEVICE_FUNCTION
#endif

// libxc flags
#define XC_FLAGS_HAVE_EXC         (1 <<  0)
#define XC_FLAGS_HAVE_VXC         (1 <<  1)
#define XC_FLAGS_HAVE_FXC         (1 <<  2)
#define XC_FLAGS_HAVE_KXC         (1 <<  3)
#define XC_FLAGS_HAVE_LXC         (1 <<  4)
#define XC_FLAGS_1D               (1 <<  5)
#define XC_FLAGS_2D               (1 <<  6)
#define XC_FLAGS_3D               (1 <<  7)
#define XC_FLAGS_HYB_CAM          (1 <<  8)
#define XC_FLAGS_HYB_CAMY         (1 <<  9)
#define XC_FLAGS_VV10             (1 << 10)
#define XC_FLAGS_HYB_LC           (1 << 11)
#define XC_FLAGS_HYB_LCY          (1 << 12)
#define XC_FLAGS_STABLE           (1 << 13)
#define XC_FLAGS_DEVELOPMENT      (1 << 14)
#define XC_FLAGS_NEEDS_LAPLACIAN  (1 << 15)
#define XC_FLAGS_NEEDS_TAU        (1 << 16)
#define XC_FLAGS_ENFORCE_FHC      (1 << 17)

#define XC_FLAGS_HAVE_ALL (XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | \
                           XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC | \
                           XC_FLAGS_HAVE_LXC)

// derivative compilation switches (define before including maple2c files)
#ifdef XC_DONT_COMPILE_VXC
#define XC_FLAGS_I_HAVE_VXC 0
#else
#define XC_FLAGS_I_HAVE_VXC XC_FLAGS_HAVE_VXC
#endif

#ifdef XC_DONT_COMPILE_FXC
#define XC_FLAGS_I_HAVE_FXC 0
#else
#define XC_FLAGS_I_HAVE_FXC XC_FLAGS_HAVE_FXC
#endif

#ifdef XC_DONT_COMPILE_KXC
#define XC_FLAGS_I_HAVE_KXC 0
#else
#define XC_FLAGS_I_HAVE_KXC XC_FLAGS_HAVE_KXC
#endif

#ifdef XC_DONT_COMPILE_LXC
#define XC_FLAGS_I_HAVE_LXC 0
#else
#define XC_FLAGS_I_HAVE_LXC XC_FLAGS_HAVE_LXC
#endif

#define XC_FLAGS_I_HAVE_EXC XC_FLAGS_HAVE_EXC
#define XC_FLAGS_I_HAVE_ALL (XC_FLAGS_HAVE_EXC   | XC_FLAGS_I_HAVE_VXC | \
                             XC_FLAGS_I_HAVE_FXC | XC_FLAGS_I_HAVE_KXC | \
                             XC_FLAGS_I_HAVE_LXC)

// Mathematical constants
#ifndef M_E
#define M_E 2.7182818284590452354
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

#define M_SQRTPI 1.772453850905516027298167483341145182798L
#define M_CBRTPI 1.464591887561523263020142527263790391739L
#define M_SQRT3 1.732050807568877293527446341505872366943L
#define M_CBRT2 1.259921049894873164767210607278228350570L
#define M_CBRT3 1.442249570307408382321638310780109588392L
#define M_CBRT4 1.587401051968199474751705639272308260391L
#define M_CBRT5 1.709975946676696989353108872543860109868L
#define M_CBRT6 1.817120592832139658891211756327260502428L
#define M_CBRT7 1.912931182772389101199116839548760282862L
#define M_CBRT9 2.080083823051904114530056824357885386338L

// Power macros
#define POW_2(x) ((x)*(x))
#define POW_3(x) ((x)*(x)*(x))

#define POW_1_2(x) sqrt(x)
#define POW_1_4(x) sqrt(sqrt(x))
#define POW_3_2(x) ((x)*sqrt(x))

#define CBRT(x) cbrt(x)
#define POW_1_3(x) cbrt(x)
#define POW_2_3(x) (cbrt(x)*cbrt(x))
#define POW_4_3(x) ((x)*cbrt(x))
#define POW_5_3(x) ((x)*cbrt(x)*cbrt(x))
#define POW_7_3(x) ((x)*(x)*cbrt(x))

// min/max helpers
#ifndef m_min
#define m_min(x,y) (((x)<(y)) ? (x) : (y))
#endif
#ifndef m_max
#define m_max(x,y) (((x)<(y)) ? (y) : (x))
#endif

#define LOG_DBL_MIN (log(DBL_MIN))
#define LOG_DBL_MAX (log(DBL_MAX))
#define SQRT_DBL_EPSILON (sqrt(DBL_EPSILON))

#define Heaviside(x) (((x) >= 0) ? 1.0 : 0.0)

#define my_piecewise3(c, x1, x2) ((c) ? (x1) : (x2))
#define my_piecewise5(c1, x1, c2, x2, x3) ((c1) ? (x1) : ((c2) ? (x2) : (x3)))

// Fractional spin constants
#define RS_FACTOR 0.6203504908994000166680068120477781673508
#define X_FACTOR_C 0.9305257363491000250020102180716672510262
#define K_FACTOR_C 4.557799872345597137288163759599305358515
#define MU_GE 0.1234567901234567901234567901234567901235
#define MU_PBE 0.2195149727645171
#define X2S 0.1282782438530421943003109254455883701296
#define FZETAFACTOR 0.5198420997897463295344212145564567011405

#define RS(x) (RS_FACTOR/CBRT(x))
#define FZETA(x) ((pow(1.0 + (x), 4.0/3.0) + pow(1.0 - (x), 4.0/3.0) - 2.0)/FZETAFACTOR)
#define DFZETA(x) ((CBRT(1.0 + (x)) - CBRT(1.0 - (x)))*(4.0/3.0)/FZETAFACTOR)

// output structures

typedef struct {
  /* order 0 */
  double *zk;
  /* order 1 */
  double *vrho;
  /* order 2 */
  double *v2rho2;
  /* order 3 */
  double *v3rho3;
  /* order 4 */
  double *v4rho4;
} xc_lda_out_params;

typedef struct {
  double *zk;
  double *vrho, *vsigma;
  double *v2rho2, *v2rhosigma, *v2sigma2;
  double *v3rho3, *v3rho2sigma, *v3rhosigma2, *v3sigma3;
  double *v4rho4, *v4rho3sigma, *v4rho2sigma2, *v4rhosigma3, *v4sigma4;
} xc_gga_out_params;

typedef struct {
  double *zk;
  double *vrho, *vsigma, *vlapl, *vtau;
  double *v2rho2, *v2rhosigma, *v2rholapl, *v2rhotau, *v2sigma2;
  double *v2sigmalapl, *v2sigmatau, *v2lapl2, *v2lapltau, *v2tau2;
  double *v3rho3, *v3rho2sigma, *v3rho2lapl, *v3rho2tau, *v3rhosigma2;
  double *v3rhosigmalapl, *v3rhosigmatau, *v3rholapl2, *v3rholapltau;
  double *v3rhotau2, *v3sigma3, *v3sigma2lapl, *v3sigma2tau;
  double *v3sigmalapl2, *v3sigmalapltau, *v3sigmatau2, *v3lapl3;
  double *v3lapl2tau, *v3lapltau2, *v3tau3;
  double *v4rho4, *v4rho3sigma, *v4rho3lapl, *v4rho3tau, *v4rho2sigma2;
  double *v4rho2sigmalapl, *v4rho2sigmatau, *v4rho2lapl2, *v4rho2lapltau;
  double *v4rho2tau2, *v4rhosigma3, *v4rhosigma2lapl, *v4rhosigma2tau;
  double *v4rhosigmalapl2, *v4rhosigmalapltau, *v4rhosigmatau2;
  double *v4rholapl3, *v4rholapl2tau, *v4rholapltau2, *v4rhotau3;
  double *v4sigma4, *v4sigma3lapl, *v4sigma3tau, *v4sigma2lapl2;
  double *v4sigma2lapltau, *v4sigma2tau2, *v4sigmalapl3, *v4sigmalapl2tau;
  double *v4sigmalapltau2, *v4sigmatau3, *v4lapl4, *v4lapl3tau;
  double *v4lapl2tau2, *v4lapltau3, *v4tau4;
} xc_mgga_out_params;

typedef struct {
  int rho, sigma, lapl, tau;
  int zk, vrho, vsigma, vlapl, vtau;
} xc_dimensions;

struct xc_func_type;

typedef struct {
  const char *ref, *doi, *bibtex, *key;
} func_reference_type;

typedef struct {
  int n;
  const char **names;
  const char **descriptions;
  const double *values;
  void (*set)(struct xc_func_type *p, const double *ext_params);
} func_params_type;

typedef struct {
  int number;
  int kind;
  const char *name;
  int family;
  func_reference_type *refs[5];
  int flags;
  double dens_threshold;
  func_params_type ext_params;
  void (*init)(struct xc_func_type *p);
  void (*end)(struct xc_func_type *p);
  const void *lda;
  const void *gga;
  const void *mgga;
} xc_func_info_type;

struct xc_func_type {
  xc_func_info_type *info;
  int nspin;
  int n_func_aux;
  struct xc_func_type **func_aux;
  double *mix_coef;
  double cam_omega, cam_alpha, cam_beta;
  double nlc_b;
  double nlc_C;
  xc_dimensions dim;
  double *ext_params;
  void *params;
  double dens_threshold;
  double zeta_threshold;
  double sigma_threshold;
  double tau_threshold;
};

// Special functions used by some generated functionals
// Definitions are provided in special_functions.cuh for the adapters that need them.
#define xc_E1_scaled(x) xc_expint_e1_impl(x, 1)

GPU_FUNCTION static inline double xc_cheb_eval(const double x, const double *cs, const int N) {
  double twox, b0 = 0.0, b1 = 0.0, b2 = 0.0;
  twox = 2.0 * x;
  for (int i = N - 1; i >= 0; --i) {
    b2 = b1;
    b1 = b0;
    b0 = twox * b1 - b2 + cs[i];
  }
  return 0.5 * (b0 - b2);
}

#define xc_E1_scaled(x) xc_expint_e1_impl(x, 1)
