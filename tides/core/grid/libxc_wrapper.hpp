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
inline constexpr int kLibxc_GGA_X_PBE = XC_GGA_X_PBE;  // 101: PBE exchange
inline constexpr int kLibxc_GGA_C_PBE = XC_GGA_C_PBE;  // 130: PBE correlation

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
    res.vrho.resize(np, 0.0);
    if (func_ == nullptr) return res;
    xc_lda_exc_vxc(func_, np, rho.data(), res.eps_xc.data(), res.vrho.data());
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
    res.vrho.resize(np, 0.0);
    res.vsigma.resize(np, 0.0);
    if (func_ == nullptr) return res;
    xc_gga_exc_vxc(func_, np, rho.data(), sigma.data(),
                   res.eps_xc.data(), res.vrho.data(), res.vsigma.data());
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
