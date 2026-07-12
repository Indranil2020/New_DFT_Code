#pragma once

// Counterpoise (Boys-Bernardi) correction for Basis Set Superposition Error.
//
// BSSE arises when monomers in a complex use each other's basis functions,
// artificially lowering the interaction energy. The counterpoise correction
// computes each monomer's energy in the full (super-system) basis and
// subtracts the spurious stabilization.
//
// The counterpoise-corrected interaction energy:
//   E_int^CP = E_AB(AB) - E_A(AB) - E_B(AB)
// where E_A(AB) is monomer A computed in the dimer basis (with ghost atoms B).
//
// The BSSE correction itself:
//   E_BSSE = sum_A [E_A(A basis) - E_A(full basis)]
//   E_corrected = E_total + E_BSSE  (note: BSSE is typically negative)
//
// Ghost atoms (Z=0) contribute basis functions but no electrons or potential.
//
// §3.1.1: Counterpoise/BSSE tooling for interaction energies.

#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace tides::basis {

struct CounterpoiseResult {
  double bsse_correction = 0.0;    // E_BSSE = sum_A [E_A(own) - E_A(full)]
  double corrected_energy = 0.0;   // E_total + E_BSSE
  double total_energy = 0.0;       // E_total (uncorrected)
  std::vector<double> monomer_energies_full;    // E_A in full basis (with ghosts)
  std::vector<double> monomer_energies_isolated; // E_A in own basis
  bool ok = true;
  std::string error;
};

class BSSECorrection {
 public:
  // Compute the counterpoise correction for a set of fragments.
  //
  //   Z:          atomic numbers of all atoms (n_atoms)
  //   positions:  positions of all atoms in Bohr (3*n_atoms)
  //   fragments:  list of fragments, each a vector of atom indices
  //   energy_fn:  function(Z, positions) -> total energy of the given atoms
  //
  // For each fragment A:
  //   - E_A(isolated): run energy_fn with only fragment A's atoms
  //   - E_A(full): run energy_fn with all atoms, but fragment B's atoms
  //     have Z=0 (ghost atoms), so only their basis functions contribute
  //
  // The energy_fn must handle Z=0 atoms (ghosts) by providing basis functions
  // but no nuclear charge or electrons.
  static CounterpoiseResult Compute(
      const std::vector<int>& Z,
      const std::vector<double>& positions,
      const std::vector<std::vector<std::size_t>>& fragments,
      const std::function<double(const std::vector<int>&,
                                  const std::vector<double>&)>& energy_fn) {
    CounterpoiseResult res;
    const std::size_t n_atoms = Z.size();
    const std::size_t n_frag = fragments.size();

    if (n_atoms == 0 || n_frag == 0) {
      res.ok = false;
      res.error = "no atoms or fragments";
      return res;
    }

    // Validate fragment indices.
    for (std::size_t f = 0; f < n_frag; ++f) {
      for (std::size_t idx : fragments[f]) {
        if (idx >= n_atoms) {
          res.ok = false;
          res.error = "fragment index out of range";
          return res;
        }
      }
    }

    // Total energy of the full system.
    res.total_energy = energy_fn(Z, positions);

    // For each fragment, compute energy in own basis and in full basis.
    res.bsse_correction = 0.0;
    res.monomer_energies_full.resize(n_frag, 0.0);
    res.monomer_energies_isolated.resize(n_frag, 0.0);

    for (std::size_t f = 0; f < n_frag; ++f) {
      // E_A(isolated): only fragment A's atoms.
      {
        std::vector<int> Z_iso;
        std::vector<double> pos_iso;
        for (std::size_t idx : fragments[f]) {
          Z_iso.push_back(Z[idx]);
          pos_iso.push_back(positions[3 * idx]);
          pos_iso.push_back(positions[3 * idx + 1]);
          pos_iso.push_back(positions[3 * idx + 2]);
        }
        res.monomer_energies_isolated[f] = energy_fn(Z_iso, pos_iso);
      }

      // E_A(full): all atoms, but other fragments are ghosts (Z=0).
      // Positions stay the same; only Z is modified.
      {
        std::vector<int> Z_ghost = Z;  // copy all
        for (std::size_t g = 0; g < n_frag; ++g) {
          if (g == f) continue;
          for (std::size_t idx : fragments[g]) {
            Z_ghost[idx] = 0;  // ghost atom
          }
        }
        res.monomer_energies_full[f] = energy_fn(Z_ghost, positions);
      }

      // BSSE contribution: E_A(own) - E_A(full)
      // (E_A(full) is more negative because of extra basis functions,
      //  so BSSE = E_A(own) - E_A(full) > 0, correction = -BSSE)
      res.bsse_correction += res.monomer_energies_isolated[f] -
                             res.monomer_energies_full[f];
    }

    // Corrected energy: E_total - BSSE (BSSE is positive, so we subtract it
    // to remove the artificial stabilization).
    // Actually: E_corrected = E_total + BSSE where BSSE = sum(E_own - E_full)
    // Since E_full < E_own (more basis = lower energy), BSSE > 0,
    // and we ADD it to E_total to correct (remove artificial lowering).
    res.corrected_energy = res.total_energy + res.bsse_correction;

    return res;
  }

  // Convenience: compute BSSE for a simple dimer (2 fragments defined by
  // a split point: atoms [0, split) are fragment A, [split, n) are B.
  static CounterpoiseResult ComputeDimer(
      const std::vector<int>& Z,
      const std::vector<double>& positions,
      std::size_t split,
      const std::function<double(const std::vector<int>&,
                                  const std::vector<double>&)>& energy_fn) {
    std::vector<std::size_t> frag_a, frag_b;
    for (std::size_t i = 0; i < split; ++i) frag_a.push_back(i);
    for (std::size_t i = split; i < Z.size(); ++i) frag_b.push_back(i);
    return Compute(Z, positions, {frag_a, frag_b}, energy_fn);
  }
};

}  // namespace tides::basis
