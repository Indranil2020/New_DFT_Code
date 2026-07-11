#pragma once

#include "grid/xc/xc_engine.hpp"

#include "grid/xc/functionals/common.cuh"
#include "grid/xc/functionals/compose.cuh"
#include "grid/xc/functionals/lda_pw92.cuh"
#include "grid/xc/functionals/lda_slater.cuh"
#include "grid/xc/functionals/lda_vwn5.cuh"
#include "grid/xc/functionals/gga_pbe.cuh"
#include "grid/xc/functionals/gga_b88.cuh"
#include "grid/xc/functionals/gga_lyp.cuh"
#include "grid/xc/functionals/gga_rpbe.cuh"

namespace tides::grid::xc {

// Tier-0 functor dispatch.  Each Functional enum maps to a stateless functor
// with a static Eval() method and a compile-time Family.  The kernel templates
// are instantiated for these functors in xc_gga_kernel.cu.

// LDA-PW92: Slater exchange + Perdew-Wang 1992 correlation.
struct LdaPw92Functor {
  static constexpr Family kFamily = Family::kLda;
  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho) {
    return LdaSlater::Eval(rho) + LdaPw92::Eval(rho);
  }
};

// SVWN5: Slater exchange + Vosko-Wilk-Nusair 5 correlation.
struct Svwn5Functor {
  static constexpr Family kFamily = Family::kLda;
  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho) {
    return LdaSlater::Eval(rho) + LdaVwn5::Eval(rho);
  }
};

// PBE (Perdew-Burke-Ernzerhof).
using PbeFunctor = GgaPbeStandard;

// PBEsol.
using PbeSolFunctor = GgaPbe<PbeSolParameters>;

// revPBE.
using RevPbeFunctor = GgaPbe<RevPbeParameters>;

// RPBE (Hammer-Hansen-Norskov).
using RpbeFunctor = GgaRpbe;

// BLYP: Becke 88 exchange + Lee-Yang-Parr correlation.
struct BlypFunctor {
  static constexpr Family kFamily = Family::kGga;
  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho, double sigma) {
    return GgaB88::Eval(rho, sigma) + GgaLyp::Eval(rho, sigma);
  }
};

// B3LYP: 0.08 Slater + 0.72 B88 + 0.19 VWN5 + 0.81 LYP.
struct B3lypFunctor {
  static constexpr Family kFamily = Family::kGga;
  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho, double sigma) {
    return 0.08 * LdaSlater::Eval(rho) +
           0.72 * GgaB88::Eval(rho, sigma) +
           0.19 * LdaVwn5::Eval(rho) +
           0.81 * GgaLyp::Eval(rho, sigma);
  }
};

// PBE0: 0.75 PBE exchange + 1.0 PBE correlation.
struct Pbe0Functor {
  static constexpr Family kFamily = Family::kGga;
  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho, double sigma) {
    return 0.75 * GgaPbeStandard::EvalExchange(rho, sigma) +
           1.0 * GgaPbeStandard::EvalCorrelation(rho, sigma);
  }
};

}  // namespace tides::grid::xc
