#pragma once

// TIDES unit conversion utilities.
//
// Internal units: Hartree atomic units (Ha, Bohr, etc.).
// This header provides conversion functions for I/O and reporting.

#include "common/config.hpp"

namespace tides::units {

// --- Energy conversions ---
inline double HartreeToEv(double e_ha) { return e_ha * config::kHartreeToEv; }
inline double EvToHartree(double e_ev) { return e_ev * config::kEvToHartree; }
inline double HartreeToRydberg(double e_ha) { return e_ha * 2.0; }
inline double RydbergToHartree(double e_ry) { return e_ry * 0.5; }
inline double HartreeToKcalMol(double e_ha) {
  return e_ha * 27.211386245988 * 23.060549028;
}

// --- Length conversions ---
inline double BohrToAngstrom(double r_bohr) {
  return r_bohr * config::kBohrToAngstrom;
}
inline double AngstromToBohr(double r_ang) {
  return r_ang * config::kAngstromToBohr;
}

// --- Force conversions ---
// 1 Ha/Bohr = 51.4220 eV/Å
inline double HaPerBohrToEvPerAngstrom(double f) {
  return f * config::kHartreeToEv / config::kBohrToAngstrom;
}

// --- Pressure conversions ---
// 1 Ha/Bohr^3 = 294.21025 GPa
inline double HaPerBohr3ToGPa(double p) {
  return p * 294.21025;
}

}  // namespace tides::units
