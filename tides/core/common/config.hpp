#pragma once

// TIDES configuration: runtime parameters and physical constants.
//
// All internal calculations use Hartree atomic units (hbar = m_e = e = 4*pi*eps0 = 1).
// Conversion factors to other unit systems are provided here for I/O and reporting.

#include <cmath>

namespace tides::config {

// --- Physical constants (Hartree atomic units) ---
inline constexpr double kHartreeToEv = 27.211386245988;   // 1 Ha = 27.211... eV
inline constexpr double kBohrToAngstrom = 0.529177210903;  // 1 a0 = 0.529... Å
inline constexpr double kBohrToNm = 0.0529177210903;
inline constexpr double kEvToHartree = 1.0 / kHartreeToEv;
inline constexpr double kAngstromToBohr = 1.0 / kBohrToAngstrom;

// --- SCF convergence defaults ---
struct SCFConfig {
  double energy_tolerance = 1e-8;       // Ha, total energy convergence
  double density_tolerance = 1e-6;      // RMS density change
  int max_iterations = 100;
  double mixing_alpha = 0.4;            // linear mixing factor
  int mixing_history = 10;              // Pulay / DIIS history
  bool use_pulay = true;
  double fermi_temperature = 0.01;      // Ha, Fermi-Dirac smearing
  int sp2_purge_tolerance = 12;         // SP2 truncation threshold (bits)
};

// --- Grid defaults ---
struct GridConfig {
  double h_default = 0.2;               // Bohr, default grid spacing
  double r_max_default = 10.0;          // Bohr, default grid extent
  int poisson_fft_threshold = 32;       // min grid size for FFT Poisson
  bool use_dual_grid = true;            // coarse/fine dual-grid strategy
  int halo_width = 3;                   // halo cells for parallel decomposition
};

// --- XC functional selection ---
enum class XCMethod {
  kLDA_PW92,   // Slater X + PW92 C (default)
  kPBE,        // PBE GGA (needs libxc)
  kPBE0,       // PBE0 hybrid (Phase B)
  kBLYP,       // BLYP GGA (needs libxc)
};

struct XCConfig {
  XCMethod method = XCMethod::kLDA_PW92;
  bool spin_polarized = false;
  double libxc_dens_threshold = 1e-14;  // skip XC eval below this density
};

// --- Precision / mixed-precision settings ---
struct PrecisionConfig {
  bool use_ozaki_f64e = true;           // T1.4: Ozaki f64 emulation via TF32
  int ozaki_exponent_bits = 10;         // Ozaki mantissa bits
  bool use_tensor_cores = true;         // T1.1: enable TF32/tensor-core GEMM
  double sp2_truncation_threshold = 3e-15;  // SP2 idempotency target
  bool deterministic_reductions = true; // enforce deterministic atomics
};

// --- Default configurations ---
inline SCFConfig DefaultSCF() { return {}; }
inline GridConfig DefaultGrid() { return {}; }
inline XCConfig DefaultXC() { return {}; }
inline PrecisionConfig DefaultPrecision() { return {}; }

}  // namespace tides::config
