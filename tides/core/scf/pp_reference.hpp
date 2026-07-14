#pragma once

// CPU reference implementations for the pseudopotential grid builds
// (Step 7 / 7b of NaoDriver::Run). These are the parity oracles for the
// device kernels in grid/pp_build_gpu.hpp and the production fallback when
// CUDA (or TIDES_PP_DEVICE) is unavailable.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "basis/pseudo/pseudopotential.hpp"
#include "grid/dual_grid.hpp"
#include "grid/pp_build_gpu.hpp"

namespace tides::scf {

// v_ext(r) = sum_a v_loc,a(|r - R_a|) on the grid: PP v_local linearly
// interpolated on the PP radial grid when a PP entry exists for the atom,
// all-electron -Z/r otherwise. Extracted verbatim from the NaoDriver Step-7
// loop so driver, tests, and device kernel share one oracle.
inline std::vector<double> BuildVlocGridReference(
    const grid::UniformGrid3D& grid,
    const std::vector<std::array<double, 3>>& positions,
    const std::vector<int>& charges,
    const std::vector<basis::Pseudopotential>* pseudopotentials) {
  const auto [n0, n1, n2] = grid.n;
  std::vector<double> v_ext_grid(n0 * n1 * n2, 0.0);
  for (std::size_t ix = 0; ix < n0; ++ix) {
    for (std::size_t iy = 0; iy < n1; ++iy) {
      for (std::size_t iz = 0; iz < n2; ++iz) {
        const std::size_t g = grid.flatten(ix, iy, iz);
        auto [x, y, z] = grid.coord(ix, iy, iz);
        double v = 0.0;
        for (std::size_t a = 0; a < positions.size(); ++a) {
          const double dx = x - positions[a][0];
          const double dy = y - positions[a][1];
          const double dz = z - positions[a][2];
          const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
          if (pseudopotentials && a < pseudopotentials->size()) {
            const auto& pp = (*pseudopotentials)[a];
            if (r < 1e-10) {
              v += pp.v_local.empty() ? 0.0 : pp.v_local[0];
            } else if (!pp.r_grid.empty() && r <= pp.r_grid.back() &&
                       !pp.v_local.empty()) {
              // Linear interpolation of v_local on the PP radial grid.
              const auto& rg = pp.r_grid;
              const auto& vl = pp.v_local;
              auto it = std::upper_bound(rg.begin(), rg.end(), r);
              if (it != rg.begin() && it != rg.end()) {
                std::size_t j = static_cast<std::size_t>(it - rg.begin() - 1);
                double t = (r - rg[j]) / (rg[j + 1] - rg[j]);
                v += (1.0 - t) * vl[j] + t * vl[j + 1];
              } else if (it == rg.end()) {
                v += vl.back();
              }
            }
          } else {
            if (r > 1e-10) v -= static_cast<double>(charges[a]) / r;
          }
        }
        v_ext_grid[g] = v;
      }
    }
  }
  return v_ext_grid;
}

// Flatten per-atom pseudopotentials into deduped per-species v_loc tables for
// device upload. Atoms without a PP entry (pseudopotentials null or index
// beyond its size) get species -1 (all-electron -Z/r fallback). A PP whose
// v_local length does not match its r_grid gets an empty table range and
// contributes zero, matching BuildVlocGridReference's empty-v_local handling.
inline grid::PpVlocTablesHost FlattenVlocTables(
    const std::vector<std::array<double, 3>>& positions,
    const std::vector<int>& charges,
    const std::vector<basis::Pseudopotential>* pseudopotentials) {
  grid::PpVlocTablesHost tables;
  const std::size_t n_atoms = positions.size();
  tables.atom_pos.resize(3 * n_atoms, 0.0);
  tables.atom_charge.resize(n_atoms, 0.0);
  tables.atom_species.assign(n_atoms, -1);
  tables.species_offset.push_back(0);

  std::vector<const basis::Pseudopotential*> species;
  for (std::size_t a = 0; a < n_atoms; ++a) {
    tables.atom_pos[3 * a] = positions[a][0];
    tables.atom_pos[3 * a + 1] = positions[a][1];
    tables.atom_pos[3 * a + 2] = positions[a][2];
    tables.atom_charge[a] =
        (a < charges.size()) ? static_cast<double>(charges[a]) : 0.0;
    if (!pseudopotentials || a >= pseudopotentials->size()) continue;
    const auto& pp = (*pseudopotentials)[a];
    int sp = -1;
    for (std::size_t k = 0; k < species.size(); ++k) {
      if (species[k]->element == pp.element &&
          species[k]->r_grid == pp.r_grid &&
          species[k]->v_local == pp.v_local) {
        sp = static_cast<int>(k);
        break;
      }
    }
    if (sp < 0) {
      sp = static_cast<int>(species.size());
      species.push_back(&pp);
      if (!pp.v_local.empty() && pp.v_local.size() == pp.r_grid.size()) {
        tables.r_tab.insert(tables.r_tab.end(), pp.r_grid.begin(),
                            pp.r_grid.end());
        tables.v_tab.insert(tables.v_tab.end(), pp.v_local.begin(),
                            pp.v_local.end());
      }
      tables.species_offset.push_back(static_cast<int>(tables.r_tab.size()));
    }
    tables.atom_species[a] = sp;
  }
  return tables;
}

}  // namespace tides::scf
