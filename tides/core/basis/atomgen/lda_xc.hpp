#pragma once

// Local (spin) density approximation exchange-correlation.
//
// Exchange: Slater X-alpha (alpha = 2/3, the KS exchange). Spin-scaled by the
// standard formula. For a spin-unpolarized system (zeta = 0) this reduces to
// the simple Slater form.
//
// Correlation: Vosko-Wilk-Nusair (VWN5, RPA) parametrization of the uniform
// electron gas correlation energy per particle. This is the default "LDA" in
// most quantum-chemistry codes (PySCF 'LDA,VWN'); matching it lets us cross-
// validate the atomic LDA total energy against PySCF directly.
//
// All densities are total electron density n = n_up + n_dn (per volume).
// Energies per particle eps_xc(n, zeta); potentials V_xc(n, zeta).
// Atomic units (Hartree).

#include <cmath>
#include <cstddef>

#include "grid/xc/functionals/lda_pw92.cuh"

namespace tides::atomgen {

class LdaXC {
 public:
  // Total XC energy density eps_xc(n, zeta) [Hartree per electron].
  // n = total density, zeta = (n_up - n_dn)/n in [-1,1].
  static double EpsXC(double n, double zeta) {
    if (n <= 0.0) return 0.0;
    return EpsX(n, zeta) + EpsC(n, zeta);
  }

  // XC potential V_xc = d(n eps_xc)/dn (total derivative wrt n, at fixed zeta).
  static double VXC(double n, double zeta) {
    if (n <= 0.0) return 0.0;
    return VX(n, zeta) + VC(n, zeta);
  }

  // --- Exchange (Slater X-alpha, alpha=2/3) ---
  // eps_x(n, zeta) = -3/4 (3/pi)^(1/3) * f(zeta) * n^(1/3)
  // with f(zeta) = ((1+zeta)^(4/3) + (1-zeta)^(4/3))/2.
  static double EpsX(double n, double zeta) {
    if (n <= 0.0) return 0.0;
    const double fz = 0.5 * (std::pow(1.0 + zeta, 4.0/3.0) +
                             std::pow(1.0 - zeta, 4.0/3.0));
    return -0.75 * std::pow(3.0 / M_PI, 1.0/3.0) * fz * std::pow(n, 1.0/3.0);
  }

  // V_x = d(n eps_x)/dn = (4/3) eps_x (for the X-alpha form, since eps_x ~ n^(1/3)
  // and n*eps_x ~ n^(4/3), derivative = (4/3) eps_x).
  static double VX(double n, double zeta) {
    return (4.0 / 3.0) * EpsX(n, zeta);
  }

  // --- Correlation (Perdew-Wang 1992, PW92) ---
  // The plan (10-physics/12) specifies LDA-PW92 as the launch XC. PW92 is a
  // smooth, real-valued fit of the uniform-electron-gas correlation energy:
  //   eps_c^P(r_s) = -2 a (1 + alpha1 r_s) ln(1 + 1/(2 a (b1 sqrt(r_s) +
  //                                            b2 r_s + b3 r_s^1.5 + b4 r_s^2)))
  // with paramagnetic (P) constants. Spin scaling: eps_c(r_s,zeta) =
  // eps_c^P(r_s) + (eps_c^F(r_s) - eps_c^P(r_s)) f(zeta).
  static double EpsC(double n, double zeta) {
    if (n <= 0.0) return 0.0;
    const double rs = std::pow(3.0 / (4.0 * M_PI * n), 1.0/3.0);
    const double ep = EpsCParamagnetic(rs);
    if (std::fabs(zeta) < 1e-12) return ep;
    const double ef = EpsCFerromagnetic(rs);
    const double fz = SpinPolFactor(zeta);
    return ep + (ef - ep) * fz;
  }

  static double EpsCParamagnetic(double rs) {
    return tides::grid::xc::LdaPw92::EvalParamagnetic(rs).eps;
  }
  static double EpsCFerromagnetic(double rs) {
    return tides::grid::xc::LdaPw92::EvalFerromagnetic(rs).eps;
  }
  // Analytic PW92 derivatives shared with the Tier-0 device functor.
  static double DEpsCParamagneticDRs(double rs) {
    return tides::grid::xc::LdaPw92::EvalParamagnetic(rs).d_eps_d_rs;
  }
  static double DEpsCFerromagneticDRs(double rs) {
    return tides::grid::xc::LdaPw92::EvalFerromagnetic(rs).d_eps_d_rs;
  }

  // V_c = d(n eps_c)/dn. Computed via the chain rule using d(eps_c)/dn.
  // n eps_c = n [ep + (ef-ep) f]; dn-dependent part is n*ep(rs(n)) and n*... .
  // V_c = eps_c + n * d(eps_c)/dn = eps_c - (rs/3) d(eps_c)/drs.
  static double VC(double n, double zeta) {
    if (n <= 0.0) return 0.0;
    const double rs = std::pow(3.0 / (4.0 * M_PI * n), 1.0/3.0);
    const double ep = EpsCParamagnetic(rs);
    const double dep = DEpsCParamagneticDRs(rs);
    if (std::fabs(zeta) < 1e-12) {
      return ep - (rs / 3.0) * dep;
    }
    const double ef = EpsCFerromagnetic(rs);
    const double def = DEpsCFerromagneticDRs(rs);
    const double fz = SpinPolFactor(zeta);
    const double ec = ep + (ef - ep) * fz;
    const double dec_drs = dep + (def - dep) * fz;
    return ec - (rs / 3.0) * dec_drs;
  }

  // Spin-polarization interpolation factor f(zeta) = ((1+z)^(4/3)+(1-z)^(4/3)-2)/
  // (2^(4/3)-2), normalized so f(0)=0, f(1)=1.
  static double SpinPolFactor(double zeta) {
    const double fz = (std::pow(1.0 + zeta, 4.0/3.0) +
                       std::pow(1.0 - zeta, 4.0/3.0) - 2.0) /
                      (std::pow(2.0, 4.0/3.0) - 2.0);
    return std::max(0.0, std::min(1.0, fz));
  }
};

}  // namespace tides::atomgen
