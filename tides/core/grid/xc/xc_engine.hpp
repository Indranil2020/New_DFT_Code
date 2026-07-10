#pragma once

// AUDIT T-X0.3: XC Engine public contract.
//
// This is the single entry point for all XC evaluations in TIDES.
// No code path may bypass xc_engine.hpp (audit standing rule #2).
//
// XcSpec:  describes which functional to evaluate (family + parameters).
// XcGridIn:  device-resident density/gradient data (SoA layout).
// XcGridOut: device-resident V_xc, eps_xc, and reduced E_xc.
//
// The host dispatch (xc_engine.cu) routes XcSpec to:
//   Tier 0: header-only __device__ functors (this engine) — LDA + PBE
//   Tier 1: pinned libxc maple2c sources compiled with nvcc (future)
//   Tier 2: CPU fallback for exotics (future)

#include <cstdint>
#include <string>
#include <vector>

namespace tides::grid::xc {

// XC functional family.
enum class XcFamily : int {
  kLda = 0,   // Rung 1: LDA (density only)
  kGga = 1,   // Rung 2: GGA (density + gradient)
  kMgga = 2,  // Rung 3: meta-GGA (density + gradient + kinetic)
  kHybrid = 3, // Rung 4: hybrid (local part only for Tier 0)
};

// Functional identifier — maps to the Top-15 table.
enum class XcFunctionalId : int {
  kLdaPw92 = 0,    // LDA_X + LDA_C_PW (Slater + Perdew-Wang)
  kLdaVwn5 = 1,    // LDA_X + LDA_C_VWN
  kPbe = 2,        // GGA_X_PBE + GGA_C_PBE
  kPbesol = 3,     // GGA_X_PBE_SOL + GGA_C_PBE_SOL
  kRevPbe = 4,     // GGA_X_PBE_R + GGA_C_PBE
  kRpbe = 5,       // GGA_X_RPBE + GGA_C_PBE
  kBlyp = 6,       // GGA_X_B88 + GGA_C_LYP
  kPbe0Local = 7,  // PBE with a_x=0.25 scaling (local part)
  kB3lypLocal = 8, // B3LYP local part
  kHse06Local = 9, // SR-omegaPBE local part
  kTpss = 10,      // mGGA
  kR2scan = 11,    // mGGA
  kScan = 12,      // mGGA (FP64-only)
  kWb97xLocal = 13, // RSH-GGA local part
  kM062xLocal = 14, // hyb-mGGA local part
};

// Specification of an XC functional for evaluation.
struct XcSpec {
  XcFunctionalId id = XcFunctionalId::kLdaPw92;
  XcFamily family = XcFamily::kLda;
  bool spin_polarized = false;
  // Optional parameters for hybrids/RSH (local part scaling).
  double exchange_fraction = 1.0;  // 1.0 for pure, 0.25 for PBE0, etc.
  double omega = 0.0;              // range-separation parameter for HSE06
};

// Input grid data (SoA layout for coalesced GPU access).
// For LDA: only rho is used.
// For GGA: rho + grad_rho_x/y/z → sigma = |grad_rho|^2.
// For mGGA: rho + grad_rho + tau.
struct XcGridIn {
  const double* rho = nullptr;       // density (np elements)
  const double* grad_rho_x = nullptr; // d(rho)/dx (np elements, GGA+)
  const double* grad_rho_y = nullptr;
  const double* grad_rho_z = nullptr;
  const double* tau = nullptr;       // kinetic energy density (mGGA+)
  std::size_t np = 0;                // number of grid points
  double grid_weight = 0.0;          // integration weight (dv)
};

// Output grid data.
struct XcGridOut {
  double* vxc = nullptr;       // V_xc(rho) — d(E_xc)/d(rho) (np elements)
  double* vsigma = nullptr;    // d(E_xc)/d(sigma) (np elements, GGA+)
  double* vtau = nullptr;      // d(E_xc)/d(tau) (np elements, mGGA+)
  double* eps_xc = nullptr;    // energy density per particle (np elements)
  double xc_energy = 0.0;      // reduced total E_xc = sum(w * eps_xc * rho)
  double kernel_ms = 0.0;      // kernel execution time
};

// Host dispatch: evaluate XC functional on the grid.
// Routes to Tier-0 (fused device kernels), Tier-1 (libxc device), or Tier-2 (CPU).
//
// For Tier-0 LDA: loads rho → eval functor → write w·v_ρ → in-kernel E_xc reduction.
// For Tier-0 GGA: loads rho + ∇ρ → compute σ → eval functor → write w·v_ρ and 2w·v_σ∇ρ.
//
// Returns true on success, false on failure (with error_msg set).
bool XcEval(const XcSpec& spec, const XcGridIn& in, XcGridOut& out,
            std::string& error_msg);

// Get the functional name as a string.
std::string XcFunctionalName(XcFunctionalId id);

// Check if a functional is implemented in Tier-0.
bool IsTier0(XcFunctionalId id);

}  // namespace tides::grid::xc
