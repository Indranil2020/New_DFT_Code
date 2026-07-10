#pragma once

// Tier-0 scalar helpers intentionally have no CUDA runtime dependency.  The
// same arithmetic is compiled in the CPU reference and in CUDA kernels.

#include <cmath>

#if defined(__CUDACC__)
#define TIDES_XC_HOST_DEVICE __host__ __device__
#else
#define TIDES_XC_HOST_DEVICE
#endif

namespace tides::grid::xc::detail {

inline constexpr double kPi = 3.141592653589793238462643383279502884;
// libxc LDA_C_PW and LDA_X both advertise this threshold, but LDA_X applies
// it to rho/2 in its generated unpolarized spin-scaling expression.
inline constexpr double kLdaPw92DensityThreshold = 1.0e-15;

TIDES_XC_HOST_DEVICE inline double Cbrt(double value) { return cbrt(value); }
TIDES_XC_HOST_DEVICE inline double Sqrt(double value) { return sqrt(value); }
TIDES_XC_HOST_DEVICE inline double Log(double value) { return log(value); }
TIDES_XC_HOST_DEVICE inline double Log1p(double value) { return log1p(value); }
TIDES_XC_HOST_DEVICE inline double Exp(double value) { return exp(value); }
TIDES_XC_HOST_DEVICE inline double Expm1(double value) { return expm1(value); }

}  // namespace tides::grid::xc::detail
