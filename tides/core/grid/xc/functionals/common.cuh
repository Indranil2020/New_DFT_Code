#pragma once

// Tier-0 scalar helpers and autodiff.  No CUDA runtime dependency; the same
// arithmetic compiles for CPU reference tests and for __device__ kernels.

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define TIDES_XC_HOST_DEVICE __host__ __device__
#else
#define TIDES_XC_HOST_DEVICE
#endif

namespace tides::grid::xc {

enum class Family : std::uint8_t { kLda, kGga, kMgga, kRsh };

struct LdaEvaluation {
  double eps = 0.0;
  double vrho = 0.0;
};

// Spin-polarized LDA: potential for each spin.
struct LdaPolEvaluation {
  double eps = 0.0;
  double vrho[2] = {0.0, 0.0};
};

struct GgaEvaluation {
  double eps = 0.0;
  double vrho = 0.0;
  double vsigma = 0.0;
};

// Spin-polarized GGA: potentials for rho (2 spins) and sigma (3 components).
struct GgaPolEvaluation {
  double eps = 0.0;
  double vrho[2] = {0.0, 0.0};
  double vsigma[3] = {0.0, 0.0, 0.0};
};

struct MggaEvaluation {
  double eps = 0.0;
  double vrho = 0.0;
  double vsigma = 0.0;
  double vtau = 0.0;
};

// Spin-polarized mGGA: potentials for rho, sigma, and tau (2 spins each).
struct MggaPolEvaluation {
  double eps = 0.0;
  double vrho[2] = {0.0, 0.0};
  double vsigma[3] = {0.0, 0.0, 0.0};
  double vtau[2] = {0.0, 0.0};
};

namespace detail {

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
TIDES_XC_HOST_DEVICE inline double Atan(double value) { return atan(value); }
TIDES_XC_HOST_DEVICE inline double Asinh(double value) { return asinh(value); }
TIDES_XC_HOST_DEVICE inline double Erf(double value) { return erf(value); }
TIDES_XC_HOST_DEVICE inline double Erfc(double value) { return erfc(value); }

// Forward-mode dual number for one independent variable.  Used for the
// LDA correlation functionals where the energy density is expressed as a
// function of rs or rho.
struct Dual {
  double value = 0.0;
  double derivative = 0.0;

  TIDES_XC_HOST_DEVICE Dual(double value_in = 0.0, double derivative_in = 0.0)
      : value(value_in), derivative(derivative_in) {}
};

TIDES_XC_HOST_DEVICE inline Dual operator+(Dual lhs, Dual rhs) {
  return {lhs.value + rhs.value, lhs.derivative + rhs.derivative};
}
TIDES_XC_HOST_DEVICE inline Dual operator-(Dual value) {
  return {-value.value, -value.derivative};
}
TIDES_XC_HOST_DEVICE inline Dual operator-(Dual lhs, Dual rhs) {
  return {lhs.value - rhs.value, lhs.derivative - rhs.derivative};
}
TIDES_XC_HOST_DEVICE inline Dual operator*(Dual lhs, Dual rhs) {
  return {lhs.value * rhs.value,
          lhs.derivative * rhs.value + lhs.value * rhs.derivative};
}
TIDES_XC_HOST_DEVICE inline Dual operator/(Dual lhs, Dual rhs) {
  const double inverse = 1.0 / rhs.value;
  const double quotient = lhs.value * inverse;
  return {quotient, (lhs.derivative - quotient * rhs.derivative) * inverse};
}
TIDES_XC_HOST_DEVICE inline Dual Cbrt(Dual value) {
  const double root = detail::Cbrt(value.value);
  return {root, value.derivative / (3.0 * root * root)};
}
TIDES_XC_HOST_DEVICE inline Dual Sqrt(Dual value) {
  const double root = detail::Sqrt(value.value);
  return {root, value.derivative / (2.0 * root)};
}
TIDES_XC_HOST_DEVICE inline Dual Log(Dual value) {
  return {detail::Log(value.value), value.derivative / value.value};
}
TIDES_XC_HOST_DEVICE inline Dual Log1p(Dual value) {
  return {detail::Log1p(value.value), value.derivative / (1.0 + value.value)};
}
TIDES_XC_HOST_DEVICE inline Dual Exp(Dual value) {
  const double exponential = detail::Exp(value.value);
  return {exponential, exponential * value.derivative};
}
TIDES_XC_HOST_DEVICE inline Dual Expm1(Dual value) {
  const double exponential = detail::Exp(value.value);
  return {detail::Expm1(value.value), exponential * value.derivative};
}
TIDES_XC_HOST_DEVICE inline Dual Atan(Dual value) {
  return {detail::Atan(value.value), value.derivative / (1.0 + value.value * value.value)};
}
TIDES_XC_HOST_DEVICE inline Dual Asinh(Dual value) {
  return detail::Log(value + detail::Sqrt(value * value + 1.0));
}

// Forward-mode dual number for (rho, sigma).  Used for GGA functionals.
struct DualRhoSigma {
  double value = 0.0;
  double d_rho = 0.0;
  double d_sigma = 0.0;

  TIDES_XC_HOST_DEVICE DualRhoSigma(double value_in = 0.0, double d_rho_in = 0.0,
                                    double d_sigma_in = 0.0)
      : value(value_in), d_rho(d_rho_in), d_sigma(d_sigma_in) {}
};

TIDES_XC_HOST_DEVICE inline DualRhoSigma operator+(DualRhoSigma lhs,
                                                   DualRhoSigma rhs) {
  return {lhs.value + rhs.value, lhs.d_rho + rhs.d_rho,
          lhs.d_sigma + rhs.d_sigma};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma operator-(DualRhoSigma value) {
  return {-value.value, -value.d_rho, -value.d_sigma};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma operator-(DualRhoSigma lhs,
                                                   DualRhoSigma rhs) {
  return {lhs.value - rhs.value, lhs.d_rho - rhs.d_rho,
          lhs.d_sigma - rhs.d_sigma};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma operator*(DualRhoSigma lhs,
                                                   DualRhoSigma rhs) {
  return {lhs.value * rhs.value,
          lhs.d_rho * rhs.value + lhs.value * rhs.d_rho,
          lhs.d_sigma * rhs.value + lhs.value * rhs.d_sigma};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma operator/(DualRhoSigma lhs,
                                                   DualRhoSigma rhs) {
  const double inverse = 1.0 / rhs.value;
  const double quotient = lhs.value * inverse;
  return {quotient,
          (lhs.d_rho - quotient * rhs.d_rho) * inverse,
          (lhs.d_sigma - quotient * rhs.d_sigma) * inverse};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma Cbrt(DualRhoSigma value) {
  const double root = detail::Cbrt(value.value);
  const double factor = 1.0 / (3.0 * root * root);
  return {root, factor * value.d_rho, factor * value.d_sigma};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma Sqrt(DualRhoSigma value) {
  const double root = detail::Sqrt(value.value);
  const double factor = 0.5 / root;
  return {root, factor * value.d_rho, factor * value.d_sigma};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma Log(DualRhoSigma value) {
  const double factor = 1.0 / value.value;
  return {detail::Log(value.value), factor * value.d_rho,
          factor * value.d_sigma};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma Log1p(DualRhoSigma value) {
  const double factor = 1.0 / (1.0 + value.value);
  return {detail::Log1p(value.value), factor * value.d_rho,
          factor * value.d_sigma};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma Exp(DualRhoSigma value) {
  const double exponential = detail::Exp(value.value);
  return {exponential, exponential * value.d_rho,
          exponential * value.d_sigma};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma Expm1(DualRhoSigma value) {
  const double exponential = detail::Exp(value.value);
  return {detail::Expm1(value.value), exponential * value.d_rho,
          exponential * value.d_sigma};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma Atan(DualRhoSigma value) {
  const double factor = 1.0 / (1.0 + value.value * value.value);
  return {detail::Atan(value.value), factor * value.d_rho,
          factor * value.d_sigma};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma Asinh(DualRhoSigma value) {
  return detail::Log(value + detail::Sqrt(value * value + 1.0));
}

// Convert an unpolarized scalar density to a DualRhoSigma variable for the
// autodiff machinery.  sigma is carried as a constant value with zero derivative.
TIDES_XC_HOST_DEVICE inline DualRhoSigma MakeRhoVariable(double rho) {
  return {rho, 1.0, 0.0};
}
TIDES_XC_HOST_DEVICE inline DualRhoSigma MakeSigmaVariable(double sigma) {
  return {sigma, 0.0, 1.0};
}

}  // namespace detail
}  // namespace tides::grid::xc
