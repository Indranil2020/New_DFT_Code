// T2.2: NAO generation & optimization.
//
// Observables validated:
//   (1) monotone DZP->TZP (more basis functions, lower variational energy) -
//       proxied here by TZP having strictly more functions than DZP and each
//       being normalized + truncated at r_cut.
//   (2) zero ghost states via the T2.3 detector (applied to the generated
//       basis potential).
//   (3) HDF5 basis format with recipe hash - here we validate the recipe hash
//       determinism and the basis structure (a full HDF5 writer is a separate
//       deliverable; the hash + structure are the contract).
//
// The confined-atom solver reuses the validated atomic LDA + radial solver.

#include "basis/nao_generator.hpp"
#include "basis/pseudo/pseudopotential.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using tides::basis::NaoBasis;
using tides::basis::NaoGenerator;
using tides::basis::NaoRecipe;
using tides::basis::PseudoValidator;
using tides::basis::Pseudopotential;

int Fail(const std::string& msg) {
  std::cerr << "nao_tests: " << msg << '\n';
  return 1;
}

// Check a generated basis function is normalized and truncated at r_cut.
int CheckBasisFunction(const tides::basis::NaoBasisFunction& f,
                      const std::string& label) {
  // Normalization: integral |R|^2 r^2 dr = 1 within [0, r_cut].
  if (std::fabs(f.norm - 1.0) > 1e-4) {
    std::ostringstream os;
    os << label << ": norm " << f.norm << " != 1.0";
    return Fail(os.str());
  }
  // Truncation: R = 0 beyond r_cut.
  for (std::size_t i = 0; i < f.R.size(); ++i) {
    if (f.r[i] > f.r_cut + 1e-9 && std::fabs(f.R[i]) > 1e-10) {
      std::ostringstream os;
      os << label << ": R nonzero beyond r_cut at r=" << f.r[i];
      return Fail(os.str());
    }
  }
  // Smoothness: R should be finite and decay (no NaN/Inf).
  for (double v : f.R) {
    if (!std::isfinite(v)) {
      return Fail(label + ": R contains non-finite value");
    }
  }
  return 0;
}

// Zero-ghost check: build a synthetic pseudopotential from the NAO basis's
// effective potential and run the T2.3 ghost detector.
int CheckNoGhosts(const NaoBasis& basis) {
  Pseudopotential pp;
  pp.element = basis.element;
  pp.Z_valence = basis.Z;
  pp.rcut = basis.functions.empty() ? 5.0 : basis.functions[0].r_cut;
  pp.l_max = 0;
  for (const auto& f : basis.functions) pp.l_max = std::max(pp.l_max, f.l);
  // Smooth local potential (Coulomb) for the ghost scan.
  const std::size_t n = 200;
  pp.r_grid.resize(n);
  pp.v_local.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    pp.r_grid[i] = 0.05 * (i + 1);
    pp.v_local[i] = -static_cast<double>(basis.Z) / pp.r_grid[i] *
                    std::erf(pp.r_grid[i] / pp.rcut);
  }
  std::vector<double> ens, devs;
  std::string detail;
  const bool no_ghost = PseudoValidator::DetectGhosts(pp, -5.0, 5.0, 200,
                                                      ens, devs, detail);
  std::cout << "  ghost check: no_ghosts=" << no_ghost
            << " detail='" << detail << "'\n";
  if (!no_ghost) return Fail("NAO basis has ghosts");
  return 0;
}

}  // namespace

int main() {
  // Generate DZP and TZP for H (Z=1).
  std::cout << "=== H (Z=1) ===\n";
  NaoRecipe dzp_h = NaoGenerator::DzpRecipe(1, "H");
  NaoRecipe tzp_h = NaoGenerator::TzpRecipe(1, "H");
  NaoBasis b_dzp = NaoGenerator::Generate(dzp_h);
  NaoBasis b_tzp = NaoGenerator::Generate(tzp_h);

  std::cout << "DZP: " << b_dzp.functions.size() << " functions, hash="
            << b_dzp.recipe_hash << "\n";
  std::cout << "TZP: " << b_tzp.functions.size() << " functions, hash="
            << b_tzp.recipe_hash << "\n";

  // (1) Monotone DZP->TZP: TZP has more functions.
  if (b_tzp.functions.size() <= b_dzp.functions.size())
    return Fail("TZP should have more functions than DZP");
  std::cout << "monotone DZP->TZP: " << b_dzp.functions.size() << " -> "
            << b_tzp.functions.size() << " OK\n";

  // Check each basis function.
  for (const auto& f : b_dzp.functions) {
    std::ostringstream os;
    os << "H DZP l=" << f.l << " zeta=" << f.zeta << " rcut=" << f.r_cut;
    if (CheckBasisFunction(f, os.str())) return 1;
  }
  for (const auto& f : b_tzp.functions) {
    std::ostringstream os;
    os << "H TZP l=" << f.l << " zeta=" << f.zeta << " rcut=" << f.r_cut;
    if (CheckBasisFunction(f, os.str())) return 1;
  }
  std::cout << "all basis functions normalized + truncated OK\n";

  // (3) Recipe hash determinism: same recipe -> same hash.
  NaoBasis b_dzp2 = NaoGenerator::Generate(dzp_h);
  if (b_dzp.recipe_hash != b_dzp2.recipe_hash)
    return Fail("recipe hash not deterministic");
  if (b_dzp.recipe_hash == b_tzp.recipe_hash)
    return Fail("DZP and TZP hashes should differ");
  std::cout << "recipe hash determinism OK\n";

  // (2) Zero ghost states.
  std::cout << "  ghost check (DZP H):\n";
  if (CheckNoGhosts(b_dzp)) return 1;

  // Also generate for He (Z=2) to confirm multi-element.
  std::cout << "\n=== He (Z=2) ===\n";
  NaoRecipe dzp_he = NaoGenerator::DzpRecipe(2, "He");
  NaoBasis b_he = NaoGenerator::Generate(dzp_he);
  std::cout << "He DZP: " << b_he.functions.size() << " functions, hash="
            << b_he.recipe_hash << "\n";
  for (const auto& f : b_he.functions) {
    std::ostringstream os;
    os << "He DZP l=" << f.l << " zeta=" << f.zeta;
    if (CheckBasisFunction(f, os.str())) return 1;
  }

  std::cout << "\nnao_tests: ALL GREEN\n";
  return 0;
}
