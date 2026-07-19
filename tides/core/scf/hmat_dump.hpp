#pragma once

// P0.6 per-term Hamiltonian dump (roadmap §3 P0.6).
//
// When the environment variable TIDES_DUMP_HMAT_DIR names a directory, the
// NaoDriver writes each Hamiltonian term there so external oracles
// (tides/verification/runners/) can compare TIDES against PySCF or a dense
// quadrature reference, and so CPU/GPU pipelines can be diffed matrix-by-
// matrix on identical inputs.
//
// Schema (consumed by verification/runners/compare_pyscf_terms.py):
//   meta.json                 n_basis, grid_h, atoms (Z, pos_bohr), basis map
//   S.txt T.txt V_ext.txt V_nl.txt V_H_iter1.txt V_xc_iter1.txt
//                             n*n doubles, row-major, one per line, %.17g
//   radial_<atom>_<fn>.txt    two columns: r R(r)

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace tides::scf::dump {

inline const char* HmatDumpDir() { return std::getenv("TIDES_DUMP_HMAT_DIR"); }

inline void WriteMatrixTxt(const std::string& dir, const char* name,
                           const std::vector<double>& m) {
  if (m.empty()) return;
  const std::string path = dir + "/" + name;
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return;
  for (double v : m) std::fprintf(f, "%.17g\n", v);
  std::fclose(f);
}

// atoms_z/pos: per atom; basis_*: per basis function (TIDES ordering).
inline void WriteMeta(const std::string& dir, std::size_t n_basis,
                      double grid_h, const std::vector<int>& atoms_z,
                      const std::vector<std::array<double, 3>>& atoms_pos,
                      const std::vector<std::size_t>& basis_atom,
                      const std::vector<int>& basis_l,
                      const std::vector<int>& basis_m,
                      const std::vector<std::size_t>& basis_fn) {
  const std::string path = dir + "/meta.json";
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return;
  std::fprintf(f, "{\n  \"n_basis\": %zu,\n  \"grid_h\": %.17g,\n  \"atoms\": [",
               n_basis, grid_h);
  for (std::size_t a = 0; a < atoms_z.size(); ++a) {
    std::fprintf(f, "%s\n    {\"Z\": %d, \"pos_bohr\": [%.17g, %.17g, %.17g]}",
                 a ? "," : "", atoms_z[a], atoms_pos[a][0], atoms_pos[a][1],
                 atoms_pos[a][2]);
  }
  std::fprintf(f, "\n  ],\n  \"basis\": [");
  for (std::size_t b = 0; b < basis_atom.size(); ++b) {
    std::fprintf(f, "%s\n    {\"atom\": %zu, \"l\": %d, \"m\": %d, \"fn\": %zu}",
                 b ? "," : "", basis_atom[b], basis_l[b], basis_m[b],
                 basis_fn[b]);
  }
  std::fprintf(f, "\n  ]\n}\n");
  std::fclose(f);
}

inline void WriteRadial(const std::string& dir, std::size_t atom_idx,
                        std::size_t fn_idx, const std::vector<double>& r,
                        const std::vector<double>& R) {
  char name[64];
  std::snprintf(name, sizeof(name), "radial_%zu_%zu.txt", atom_idx, fn_idx);
  const std::string path = dir + "/" + name;
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return;
  const std::size_t n = std::min(r.size(), R.size());
  for (std::size_t k = 0; k < n; ++k)
    std::fprintf(f, "%.17g %.17g\n", r[k], R[k]);
  std::fclose(f);
}

}  // namespace tides::scf::dump
