#pragma once

// Pseudopotential loader: maps element symbols to UPF2 files on disk and
// loads them into Pseudopotential structs. Supports the PseudoDojo/SG15 ONCV
// PBE SR library bundled in external/pseudopotentials/pseudodojo-pbe-sr/.
//
// Usage:
//   auto pp = PpLoader::Load("Si");
//   if (pp.ok()) { ... use pp.value() ... }
//
//   // Load a set of elements:
//   auto pps = PpLoader::LoadMany({"H", "O"}, "PBE-SR");
//   // Then pass &pps to NaoDriver::Run.

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "basis/pseudo/pseudopotential.hpp"
#include "basis/pseudo/upf2_reader.hpp"
#include "common/status.hpp"

namespace tides::basis {

class PpLoader {
 public:
  // Default library path relative to the tides source root.
  // Can be overridden via the TIDES_PP_DIR environment variable.
  static std::string DefaultPpDir() {
    const char* env = std::getenv("TIDES_PP_DIR");
    if (env && env[0] != '\0') return std::string(env);
    // Try common locations relative to the source tree.
    namespace fs = std::filesystem;
    // 1. tides/external/pseudopotentials/pseudodojo-pbe-sr
    const char* src_dir = std::getenv("TIDES_SRC_DIR");
    if (src_dir && src_dir[0] != '\0') {
      fs::path p = fs::path(src_dir) / "external" / "pseudopotentials" / "pseudodojo-pbe-sr";
      if (fs::exists(p)) return p.string();
    }
    // 2. Hardcoded fallback (common build/source layout).
    return "external/pseudopotentials/pseudodojo-pbe-sr";
  }

  // Load a single pseudopotential by element symbol (e.g. "H", "Si").
  // Searches for {Element}_ONCV_PBE_SR.upf in the PP directory.
  static tides::Result<Pseudopotential> Load(const std::string& element,
                                              const std::string& pp_dir = "") {
    const std::string dir = pp_dir.empty() ? DefaultPpDir() : pp_dir;
    const std::string filename = element + "_ONCV_PBE_SR.upf";
    namespace fs = std::filesystem;
    const fs::path filepath = fs::path(dir) / filename;

    if (!fs::exists(filepath)) {
      return tides::Status::IoError(
          "Pseudopotential file not found: " + filepath.string() +
          " (element=" + element + ", dir=" + dir + ")");
    }

    auto result = Upf2Reader::Read(filepath.string());
    if (!result.ok()) {
      return tides::Status::CorruptData(
          "Failed to parse UPF2 file " + filepath.string() + ": " +
          result.status().message());
    }

    // Basic sanity checks.
    auto& pp = result.value();
    if (pp.r_grid.empty()) {
      return tides::Status::CorruptData(
          "Pseudopotential has empty radial grid: " + filepath.string());
    }
    if (pp.Z_valence <= 0) {
      return tides::Status::CorruptData(
          "Pseudopotential has non-positive Z_valence: " + filepath.string());
    }

    return result;
  }

  // Load pseudopotentials for a list of element symbols.
  // Returns a vector of Pseudopotential in the same order.
  // If any element fails to load, returns an empty vector and sets error_msg.
  static std::vector<Pseudopotential> LoadMany(
      const std::vector<std::string>& elements,
      const std::string& pp_dir = "",
      std::string* error_msg = nullptr) {
    std::vector<Pseudopotential> pps;
    pps.reserve(elements.size());
    for (const auto& el : elements) {
      auto result = Load(el, pp_dir);
      if (!result.ok()) {
        if (error_msg) *error_msg = result.status().message();
        return {};
      }
      pps.push_back(std::move(result.value()));
    }
    return pps;
  }

  // Load pseudopotentials for a list of atomic numbers.
  // Uses the built-in Z → symbol map for elements up to Z=86.
  static std::vector<Pseudopotential> LoadByAtomicNumbers(
      const std::vector<int>& atomic_numbers,
      const std::string& pp_dir = "",
      std::string* error_msg = nullptr) {
    std::vector<std::string> symbols;
    symbols.reserve(atomic_numbers.size());
    for (int z : atomic_numbers) {
      const char* sym = ElementSymbol(z);
      if (sym[0] == '\0') {
        if (error_msg) *error_msg = "Unknown atomic number: " + std::to_string(z);
        return {};
      }
      symbols.push_back(sym);
    }
    return LoadMany(symbols, pp_dir, error_msg);
  }

  // Map atomic number to element symbol (H=1 through Rn=86).
  static const char* ElementSymbol(int z) {
    static const char* table[] = {
      "",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
      "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc",
      "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
      "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc",
      "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
      "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
      "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os",
      "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"
    };
    if (z < 1 || z > 86) return "";
    return table[z];
  }

  // Check if a pseudopotential file is available for the given element.
  static bool IsAvailable(const std::string& element,
                           const std::string& pp_dir = "") {
    const std::string dir = pp_dir.empty() ? DefaultPpDir() : pp_dir;
    namespace fs = std::filesystem;
    return fs::exists(fs::path(dir) / (element + "_ONCV_PBE_SR.upf"));
  }
};

}  // namespace tides::basis
