#pragma once

// T3.5: libxc wrapper for exchange-correlation functionals.
//
// Provides a C++ interface to libxc for LDA and GGA (PBE) functionals.
// The wrapper handles:
// - Functional initialization and cleanup
// - LDA evaluation: eps_xc, V_xc from density rho
// - GGA evaluation: eps_xc, V_xc from density rho and reduced gradient sigma
// - Spin-unpolarized (nspin=1) and spin-polarized (nspin=2) modes
//
// libxc is installed at /home/niel/src/libxc-7.0.0-install/
// Headers: xc.h, xc_funcs.h
// Library: libxc.so, libxc.a

#include <cstddef>
#include <string>
#include <vector>

#include <xc.h>
#include <xc_funcs.h>

namespace tides::grid {

// libxc functional IDs (from xc_funcs.h)
inline constexpr int kLibxc_LDA_X = XC_LDA_X;        // 1: Slater exchange
inline constexpr int kLibxc_LDA_C_PW = XC_LDA_C_PW;  // 12: Perdew-Wang correlation
inline constexpr int kLibxc_LDA_C_VWN = XC_LDA_C_VWN;  // 7: Vosko-Wilk-Nusair (VWN5)
inline constexpr int kLibxc_GGA_X_PBE = XC_GGA_X_PBE;  // 101: PBE exchange
inline constexpr int kLibxc_GGA_C_PBE = XC_GGA_C_PBE;  // 130: PBE correlation
inline constexpr int kLibxc_GGA_C_PBE_SOL = XC_GGA_C_PBE_SOL;  // 133: PBEsol correlation
inline constexpr int kLibxc_GGA_X_PBE_SOL = XC_GGA_X_PBE_SOL;  // 116: PBEsol exchange
inline constexpr int kLibxc_GGA_X_PBE_R = XC_GGA_X_PBE_R;  // 102: revPBE exchange
inline constexpr int kLibxc_GGA_X_RPBE = XC_GGA_X_RPBE;  // 117: RPBE exchange
inline constexpr int kLibxc_GGA_X_B88 = XC_GGA_X_B88;  // 106: Becke 88 exchange
inline constexpr int kLibxc_GGA_C_LYP = XC_GGA_C_LYP;  // 131: Lee-Yang-Parr correlation
inline constexpr int kLibxc_MGGA_X_TPSS = XC_MGGA_X_TPSS;  // 201: TPSS exchange
inline constexpr int kLibxc_MGGA_C_TPSS = XC_MGGA_C_TPSS;  // 202: TPSS correlation
inline constexpr int kLibxc_MGGA_X_SCAN = XC_MGGA_X_SCAN;  // 263: SCAN exchange
inline constexpr int kLibxc_MGGA_C_SCAN = XC_MGGA_C_SCAN;  // 267: SCAN correlation
inline constexpr int kLibxc_MGGA_X_R2SCAN = XC_MGGA_X_R2SCAN;  // 497: r2SCAN exchange
inline constexpr int kLibxc_MGGA_C_R2SCAN = XC_MGGA_C_R2SCAN;  // 498: r2SCAN correlation
inline constexpr int kLibxc_HYB_MGGA_X_M06_2X = XC_HYB_MGGA_X_M06_2X;  // 450: M06-2X exchange
inline constexpr int kLibxc_MGGA_C_M06_2X = XC_MGGA_C_M06_2X;  // 236: M06-2X correlation
inline constexpr int kLibxc_HYB_GGA_XC_HSE06 = XC_HYB_GGA_XC_HSE06;  // 428: HSE06
inline constexpr int kLibxc_HYB_GGA_XC_WB97X = XC_HYB_GGA_XC_WB97X;  // 464: wB97X

// RAII wrapper for a libxc functional.
class LibxcFunctional {
 public:
  LibxcFunctional() = default;

  ~LibxcFunctional() {
    if (func_ != nullptr) {
      xc_func_end(func_);
      delete func_;
    }
  }

  LibxcFunctional(const LibxcFunctional&) = delete;
  LibxcFunctional& operator=(const LibxcFunctional&) = delete;

  LibxcFunctional(LibxcFunctional&& other) noexcept : func_(other.func_) {
    other.func_ = nullptr;
  }

  LibxcFunctional& operator=(LibxcFunctional&& other) noexcept {
    if (this != &other) {
      if (func_ != nullptr) {
        xc_func_end(func_);
        delete func_;
      }
      func_ = other.func_;
      other.func_ = nullptr;
    }
    return *this;
  }

  // Initialize a functional by ID. nspin = 1 (unpolarized) or 2 (polarized).
  bool Init(int functional_id, int nspin) {
    func_ = xc_func_alloc();
    if (func_ == nullptr) return false;
    if (xc_func_init(func_, functional_id, nspin) != 0) {
      delete func_;
      func_ = nullptr;
      return false;
    }
    return true;
  }

  // Get the functional family (XC_FAMILY_LDA=1, XC_FAMILY_GGA=2, etc.)
  int Family() const {
    if (func_ == nullptr || func_->info == nullptr) return -1;
    return xc_func_info_get_family(func_->info);
  }

  // Exposes the initialized threshold so Tier-0 can mirror the pinned libxc
  // oracle rather than copying an assumption into a device functor.
  double DensityThreshold() const {
    return (func_ != nullptr) ? func_->dens_threshold : 0.0;
  }

  // Get the functional name.
  std::string Name() const {
    if (func_ == nullptr || func_->info == nullptr) return "unknown";
    return xc_func_info_get_name(func_->info);
  }

  // Evaluate LDA functional: eps_xc and V_xc from density.
  // rho: density array (size np for unpol, 2*np for pol)
  // Returns eps_xc (per particle) and vrho (dE/drho).
  struct LdaResult {
    std::vector<double> eps_xc;  // per-particle energy
    std::vector<double> vrho;    // potential dE/drho
  };

  LdaResult EvalLDA(const std::vector<double>& rho, std::size_t np) const {
    LdaResult res;
    res.eps_xc.resize(np, 0.0);
    const int nspin = (func_ != nullptr) ? func_->nspin : 1;
    res.vrho.resize(static_cast<std::size_t>(nspin) * np, 0.0);
    if (func_ == nullptr) return res;
    xc_lda_exc_vxc(func_, np, rho.data(), res.eps_xc.data(), res.vrho.data());
    return res;
  }

  // T-X4.5: Evaluate LDA second derivatives (f_xc, order 2) for TDDFT/Hessian.
  // Returns v2rho2 = d^2(eps*rho)/d rho^2 (the kernel of the XC Hessian).
  struct LdaOrder2Result {
    std::vector<double> v2rho2;  // d^2E/d rho^2
  };

  LdaOrder2Result EvalLDAOrder2(const std::vector<double>& rho,
                                 std::size_t np) const {
    LdaOrder2Result res;
    const int nspin = (func_ != nullptr) ? func_->nspin : 1;
    res.v2rho2.resize(static_cast<std::size_t>(nspin) * np, 0.0);
    if (func_ == nullptr) return res;
    // xc_lda_fxc computes the second derivative of the energy.
    xc_lda_fxc(func_, np, rho.data(), res.v2rho2.data());
    return res;
  }

  // Evaluate GGA functional: eps_xc and V_xc from density and sigma.
  // rho: density array (size np for unpol, 2*np for pol)
  // sigma: |grad rho|^2 array (size np for unpol, 3*np for pol)
  // Returns eps_xc, vrho (dE/drho), vsigma (dE/dsigma).
  struct GgaResult {
    std::vector<double> eps_xc;   // per-particle energy
    std::vector<double> vrho;     // dE/drho
    std::vector<double> vsigma;   // dE/dsigma
  };

  GgaResult EvalGGA(const std::vector<double>& rho,
                     const std::vector<double>& sigma,
                     std::size_t np) const {
    GgaResult res;
    res.eps_xc.resize(np, 0.0);
    const int nspin = (func_ != nullptr) ? func_->nspin : 1;
    const int nsigma = (nspin == 2) ? 3 : 1;
    res.vrho.resize(static_cast<std::size_t>(nspin) * np, 0.0);
    res.vsigma.resize(static_cast<std::size_t>(nsigma) * np, 0.0);
    if (func_ == nullptr) return res;
    xc_gga_exc_vxc(func_, np, rho.data(), sigma.data(),
                   res.eps_xc.data(), res.vrho.data(), res.vsigma.data());
    return res;
  }

  // T-X4.5: Evaluate GGA second derivatives (f_xc, order 2) for TDDFT/Hessian.
  // Returns v2rho2, v2rhosigma, v2sigma2 — the kernel of the XC Hessian for GGA.
  struct GgaOrder2Result {
    std::vector<double> v2rho2;       // d^2E/d rho^2
    std::vector<double> v2rhosigma;   // d^2E/d rho d sigma
    std::vector<double> v2sigma2;     // d^2E/d sigma^2
  };

  GgaOrder2Result EvalGGAOrder2(const std::vector<double>& rho,
                                 const std::vector<double>& sigma,
                                 std::size_t np) const {
    GgaOrder2Result res;
    const int nspin = (func_ != nullptr) ? func_->nspin : 1;
    const int nsigma = (nspin == 2) ? 3 : 1;
    res.v2rho2.resize(static_cast<std::size_t>(nspin) * np, 0.0);
    res.v2rhosigma.resize(static_cast<std::size_t>(nspin * nsigma) * np, 0.0);
    res.v2sigma2.resize(static_cast<std::size_t>(nsigma * nsigma) * np, 0.0);
    if (func_ == nullptr) return res;
    xc_gga_fxc(func_, np, rho.data(), sigma.data(),
               res.v2rho2.data(), res.v2rhosigma.data(), res.v2sigma2.data());
    return res;
  }

  // Evaluate mGGA functional: eps_xc and V_xc from rho, sigma, lapl, and tau.
  // rho: density array (size np for unpol, 2*np for pol)
  // sigma: |grad rho|^2 array (size np for unpol, 3*np for pol)
  // lapl: laplacian of density array (size np for unpol, 2*np for pol)
  // tau: kinetic energy density array (size np for unpol, 2*np for pol)
  struct MggaResult {
    std::vector<double> eps_xc;   // per-particle energy
    std::vector<double> vrho;     // dE/drho
    std::vector<double> vsigma;   // dE/dsigma
    std::vector<double> vlapl;    // dE/dlapl
    std::vector<double> vtau;     // dE/dtau
  };

  MggaResult EvalMGGA(const std::vector<double>& rho,
                      const std::vector<double>& sigma,
                      const std::vector<double>& lapl,
                      const std::vector<double>& tau,
                      std::size_t np) const {
    MggaResult res;
    res.eps_xc.resize(np, 0.0);
    const int nspin = (func_ != nullptr) ? func_->nspin : 1;
    const int nsigma = (nspin == 2) ? 3 : 1;
    res.vrho.resize(static_cast<std::size_t>(nspin) * np, 0.0);
    res.vsigma.resize(static_cast<std::size_t>(nsigma) * np, 0.0);
    res.vlapl.resize(static_cast<std::size_t>(nspin) * np, 0.0);
    res.vtau.resize(static_cast<std::size_t>(nspin) * np, 0.0);
    if (func_ == nullptr) return res;
    xc_mgga_exc_vxc(func_, np, rho.data(), sigma.data(), lapl.data(), tau.data(),
                    res.eps_xc.data(), res.vrho.data(), res.vsigma.data(),
                    res.vlapl.data(), res.vtau.data());
    return res;
  }

  // Compute sigma = |grad rho|^2 for a 3D uniform grid via central FD.
  // Returns sigma array (size np, unpolarized).
  static std::vector<double> ComputeSigma(
      std::size_t n0, std::size_t n1, std::size_t n2,
      double h0, double h1, double h2,
      const std::vector<double>& rho) {
    std::size_t np = n0 * n1 * n2;
    std::vector<double> sigma(np, 0.0);
    auto idx = [n0, n1](std::size_t x, std::size_t y, std::size_t z) {
      return x + n0 * (y + n1 * z);
    };
    for (std::size_t iz = 0; iz < n2; ++iz)
      for (std::size_t iy = 0; iy < n1; ++iy)
        for (std::size_t ix = 0; ix < n0; ++ix) {
          const std::size_t g = idx(ix, iy, iz);
          if (rho[g] < 1e-14) { sigma[g] = 0.0; continue; }
          double dnx = 0.0, dny = 0.0, dnz = 0.0;
          if (ix > 0 && ix + 1 < n0)
            dnx = (rho[idx(ix + 1, iy, iz)] - rho[idx(ix - 1, iy, iz)]) / (2.0 * h0);
          if (iy > 0 && iy + 1 < n1)
            dny = (rho[idx(ix, iy + 1, iz)] - rho[idx(ix, iy - 1, iz)]) / (2.0 * h1);
          if (iz > 0 && iz + 1 < n2)
            dnz = (rho[idx(ix, iy, iz + 1)] - rho[idx(ix, iy, iz - 1)]) / (2.0 * h2);
          sigma[g] = dnx * dnx + dny * dny + dnz * dnz;
        }
    return sigma;
  }

  // Convenience: evaluate PBE (X + C) on a grid.
  // Returns combined eps_xc and V_xc for the PBE functional.
  struct PBEResult {
    std::vector<double> eps_xc;
    std::vector<double> vxc;
  };

  static PBEResult EvalPBEOnGrid(
      std::size_t n0, std::size_t n1, std::size_t n2,
      double h0, double h1, double h2,
      const std::vector<double>& rho) {
    std::size_t np = n0 * n1 * n2;
    std::vector<double> rho_safe(np);
    for (std::size_t i = 0; i < np; ++i)
      rho_safe[i] = std::max(rho[i], 0.0);

    auto sigma = ComputeSigma(n0, n1, n2, h0, h1, h2, rho_safe);

    // PBE exchange
    LibxcFunctional fx, fc;
    fx.Init(kLibxc_GGA_X_PBE, XC_UNPOLARIZED);
    fc.Init(kLibxc_GGA_C_PBE, XC_UNPOLARIZED);

    auto rx = fx.EvalGGA(rho_safe, sigma, np);
    auto rc = fc.EvalGGA(rho_safe, sigma, np);

    PBEResult res;
    res.eps_xc.resize(np, 0.0);
    res.vxc.resize(np, 0.0);
    for (std::size_t i = 0; i < np; ++i) {
      res.eps_xc[i] = rx.eps_xc[i] + rc.eps_xc[i];
      // For GGA, V_xc = vrho - 2 * div(vsigma * grad rho) / rho
      // The full potential requires the divergence term. For the grid
      // evaluation, we return vrho + vsigma contribution. The full
      // V_xc with gradient terms is computed in the SCF loop.
      res.vxc[i] = rx.vrho[i] + rc.vrho[i];
    }
    return res;
  }

  // Convenience: evaluate LDA-PW92 (X + C) on a grid using libxc.
  static PBEResult EvalLDAOnGrid(const std::vector<double>& rho) {
    std::size_t np = rho.size();
    std::vector<double> rho_safe(np);
    for (std::size_t i = 0; i < np; ++i)
      rho_safe[i] = std::max(rho[i], 0.0);

    LibxcFunctional fx, fc;
    fx.Init(kLibxc_LDA_X, XC_UNPOLARIZED);
    fc.Init(kLibxc_LDA_C_PW, XC_UNPOLARIZED);

    auto rx = fx.EvalLDA(rho_safe, np);
    auto rc = fc.EvalLDA(rho_safe, np);

    PBEResult res;
    res.eps_xc.resize(np, 0.0);
    res.vxc.resize(np, 0.0);
    for (std::size_t i = 0; i < np; ++i) {
      res.eps_xc[i] = rx.eps_xc[i] + rc.eps_xc[i];
      res.vxc[i] = rx.vrho[i] + rc.vrho[i];
    }
    return res;
  }

 private:
  xc_func_type* func_ = nullptr;
};

}  // namespace tides::grid
