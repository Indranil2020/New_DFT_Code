#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "basis/pseudo/pseudopotential.hpp"
#include "common/status.hpp"

namespace tides::basis {

// Minimal PSML (Pseudopotential Markup Language) reader.
//
// PSML is an XML-based pseudopotential interchange format used by SIESTA,
// Abinit, and other codes. It stores norm-conserving pseudopotentials in a
// format-independent representation.
//
// The PSML structure:
//   <pseudo ...>                      root element, attributes: atomic-number,
//                                                                z-valence, ...
//     <radial_grid ...>                grid definition (n, a, b, or explicit)
//       <data> ... </data>             (or inline grid points)
//     </radial_grid>
//     <radial_function l="0" ...>      pseudo wavefunction (per angular momentum)
//       <data> ... </data>
//     </radial_function>
//     <projector l="0" ...>            Kleinman-Bylander projector
//       <data> ... </data>
//     </projector>
//     <valence_charge>                 valence charge density
//       <data> ... </data>
//     </valence_charge>
//     <core_charge>                    core charge density (for NLCC)
//       <data> ... </data>
//     </core_charge>
//     <local_potential>                local part of the PP
//       <data> ... </data>
//     </local_potential>
//   </pseudo>
//
// This is a lightweight XML extractor (not a full DOM parser), mirroring the
// approach in Upf2Reader. PSML files are well-structured and we only need
// specific tags. Units in PSML are typically atomic units (Bohr, Hartree);
// no energy conversion is needed (unlike UPF2 which uses Ry).
class PsmlReader {
 public:
  // Parse a PSML file from disk. Returns Status::Ok on success.
  static tides::Result<Pseudopotential> Read(const std::string& path) {
    std::ifstream f(path);
    if (!f) return tides::Status::IoError("cannot open PSML file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();
    f.close();
    return Parse(content);
  }

  // Parse PSML XML content from a string.
  static tides::Result<Pseudopotential> Parse(const std::string& xml) {
    Pseudopotential pp;
    pp.format = "PSML";

    // Parse attributes from the <pseudo ...> root element.
    const std::string root = ExtractTagOpening(xml, "pseudo");
    pp.element = Trim(ExtractAttr(root, "element"));
    pp.Z_valence = ExtractIntAttr(root, "z-valence");
    // Fall back to atomic-number attribute for Z_valence if z-valence absent.
    if (pp.Z_valence == 0)
      pp.Z_valence = ExtractIntAttr(root, "z-val");
    if (pp.Z_valence == 0)
      pp.Z_valence = ExtractIntAttr(root, "atomic-number");
    // Atomic number for metadata.
    int atomic_number = ExtractIntAttr(root, "atomic-number");
    if (pp.element.empty() && atomic_number > 0)
      pp.element = ElementFromZ(atomic_number);

    // Radial grid. PSML can use explicit <data> or a grid specification
    // (n, a, b, delta). We try explicit data first.
    pp.r_grid = ParseDoubles(ExtractElementData(xml, "radial_grid"));
    if (pp.r_grid.empty()) {
      // Fall back: generate grid from n/a/b/delta attributes.
      pp.r_grid = BuildGridFromAttrs(ExtractTagOpening(xml, "radial_grid"));
    }
    if (pp.r_grid.empty())
      return tides::Status::CorruptData("PSML: no radial grid found");

    // Local potential (already in Hartree — no Ry conversion needed).
    pp.v_local = ParseDoubles(ExtractElementData(xml, "local_potential"));
    if (pp.v_local.empty())
      pp.v_local = ParseDoubles(ExtractElementData(xml, "vlocal"));
    if (pp.v_local.empty())
      pp.v_local = ParseDoubles(ExtractElementData(xml, "local"));
    // PSML may store local potential with sign conventions matching the PP;
    // no unit conversion needed (Hartree native).

    // Parse radial functions (pseudo wavefunctions) — these give us the
    // reference energies and angular momentum channels.
    std::vector<int> l_values;
    std::vector<double> eigenvalues;
    ParseRadialFunctions(xml, pp.r_grid, l_values, eigenvalues);

    // Parse all projectors and KB coefficients.
    struct RawProj {
      int l = 0;
      std::vector<double> raw_projector;
      double kb_coeff = 0.0;  // KB coefficient (Hartree)
      int index = 0;
    };
    std::vector<RawProj> projs;
    std::size_t pos = 0;
    int proj_idx = 0;
    while (true) {
      const std::size_t start = xml.find("<projector", pos);
      if (start == std::string::npos) break;
      if (start > 0 && xml[start - 1] == '/') { pos = start + 1; continue; }
      const std::size_t tag_end = xml.find('>', start);
      if (tag_end == std::string::npos) break;
      const std::string tag = xml.substr(start, tag_end - start + 1);

      // Find closing </projector>
      const std::size_t end = xml.find("</projector>", tag_end);
      if (end == std::string::npos) break;
      const std::string body = xml.substr(tag_end + 1, end - tag_end - 1);

      RawProj p;
      p.index = proj_idx++;
      p.l = ExtractIntAttr(tag, "l");
      if (p.l == 0) p.l = ExtractIntAttr(tag, "angular_momentum");
      p.kb_coeff = ExtractDoubleAttr(tag, "kb_coeff");
      if (p.kb_coeff == 0.0)
        p.kb_coeff = ExtractDoubleAttr(tag, "eigenvalue");
      // Extract <data> from within the projector body.
      p.raw_projector = ParseDoubles(ExtractTagContent(body, "data"));
      if (p.raw_projector.empty())
        p.raw_projector = ParseDoubles(Trim(body));

      projs.push_back(std::move(p));
      pos = end + 12;  // length of "</projector>"
    }

    // Build per-l KB channels with multi-projector Dij blocks.
    if (!projs.empty()) {
      // Sort by index.
      std::sort(projs.begin(), projs.end(),
                [](const RawProj& a, const RawProj& b) {
                  return a.index < b.index;
                });

      // Group by angular momentum l.
      std::map<int, std::vector<std::size_t>> l_groups;
      for (std::size_t i = 0; i < projs.size(); ++i)
        l_groups[projs[i].l].push_back(i);

      for (auto& [l, idxs] : l_groups) {
        Pseudopotential::KBChannel ch;
        ch.l = l;
        for (std::size_t i : idxs) {
          std::vector<double> beta(pp.r_grid.size(), 0.0);
          for (std::size_t k = 0; k < projs[i].raw_projector.size() &&
               k < beta.size(); ++k) {
            beta[k] = projs[i].raw_projector[k];
          }
          ch.projectors.push_back(std::move(beta));
        }
        // Dij matrix: diagonal with KB coefficients.
        const std::size_t n = idxs.size();
        ch.Dij.assign(n, std::vector<double>(n, 0.0));
        for (std::size_t a = 0; a < n; ++a)
          ch.Dij[a][a] = projs[idxs[a]].kb_coeff;
        // Legacy: first projector + first diagonal coefficient.
        if (!ch.projectors.empty()) {
          ch.projector = ch.projectors[0];
          ch.kb_coeff = ch.Dij[0][0];
        }
        // Eigenvalue from radial function if available.
        for (std::size_t li = 0; li < l_values.size(); ++li) {
          if (l_values[li] == l) {
            ch.eiganvalue = eigenvalues[li];
            break;
          }
        }
        pp.channels.push_back(std::move(ch));
      }
    }

    // l_max: maximum angular momentum from channels or radial functions.
    pp.l_max = 0;
    if (!pp.channels.empty()) {
      for (const auto& ch : pp.channels)
        pp.l_max = std::max(pp.l_max, ch.l);
    }
    if (!l_values.empty()) {
      for (int l : l_values)
        pp.l_max = std::max(pp.l_max, l);
    }

    pp.rcut = pp.r_grid.empty() ? 0.0 : pp.r_grid.back();
    pp.md5_checksum = SimpleHash(xml);
    return pp;
  }

 private:
  static std::string Trim(const std::string& s) {
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
  }

  static std::string ExtractAttr(const std::string& s,
                                 const std::string& attr) {
    std::string pat = attr + "=\"";
    std::size_t p = s.find(pat);
    if (p == std::string::npos) return "";
    p += pat.size();
    std::size_t q = s.find('"', p);
    if (q == std::string::npos) return "";
    return s.substr(p, q - p);
  }

  static int ExtractIntAttr(const std::string& s, const std::string& attr) {
    std::string v = ExtractAttr(s, attr);
    if (v.empty()) return 0;
    try { return std::stoi(v); } catch (...) { return 0; }
  }

  static double ExtractDoubleAttr(const std::string& s,
                                  const std::string& attr) {
    std::string v = ExtractAttr(s, attr);
    if (v.empty()) return 0.0;
    try { return std::stod(v); } catch (...) { return 0.0; }
  }

  // Extract the opening tag of an element, including attributes.
  // e.g. <pseudo atomic-number="1" ...>
  static std::string ExtractTagOpening(const std::string& xml,
                                        const std::string& tag) {
    std::string open = "<" + tag;
    std::size_t p = xml.find(open);
    if (p == std::string::npos) return "";
    // Ensure we matched the full tag name (not a prefix).
    if (p + open.size() < xml.size()) {
      char next = xml[p + open.size()];
      if (next != ' ' && next != '\t' && next != '\n' && next != '>' &&
          next != '/') return "";
    }
    std::size_t q = xml.find(">", p);
    if (q == std::string::npos) return "";
    return xml.substr(p, q - p + 1);
  }

  // Extract the text content between <tag> and </tag>, stripping nested tags.
  static std::string ExtractTagContent(const std::string& xml,
                                        const std::string& tag) {
    std::string open = "<" + tag;
    std::size_t p = xml.find(open);
    if (p == std::string::npos) return "";
    p = xml.find('>', p);
    if (p == std::string::npos) return "";
    ++p;
    std::size_t q = xml.find("</" + tag, p);
    if (q == std::string::npos) return "";
    return Trim(xml.substr(p, q - p));
  }

  // Extract the <data> content from within an element (e.g. <radial_grid>,
  // <radial_function>, <projector>). PSML stores numerical arrays inside
  // <data>...</data> sub-elements.
  static std::string ExtractElementData(const std::string& xml,
                                         const std::string& tag) {
    // Find the opening tag of the element.
    std::string open = "<" + tag;
    std::size_t p = xml.find(open);
    if (p == std::string::npos) return "";
    // Ensure full tag name match.
    if (p + open.size() < xml.size()) {
      char next = xml[p + open.size()];
      if (next != ' ' && next != '\t' && next != '\n' && next != '>' &&
          next != '/') return "";
    }
    // Find closing tag.
    std::size_t tag_end = xml.find('>', p);
    if (tag_end == std::string::npos) return "";
    std::size_t close_tag = xml.find("</" + tag, tag_end);
    if (close_tag == std::string::npos) return "";

    std::string body = xml.substr(tag_end + 1, close_tag - tag_end - 1);

    // Extract <data> from within the element body.
    std::string data = ExtractTagContent(body, "data");
    if (data.empty()) {
      // If no <data> subtag, the element body itself may contain the numbers.
      data = Trim(body);
    }
    return data;
  }

  // Parse all <radial_function> elements to get l values and eigenvalues.
  // PSML radial functions are pseudo wavefunctions; the eigenvalue is stored
  // as an attribute (eigenvalue, epsilon, or energy).
  static void ParseRadialFunctions(const std::string& xml,
                                    const std::vector<double>& /*r_grid*/,
                                    std::vector<int>& l_values,
                                    std::vector<double>& eigenvalues) {
    l_values.clear();
    eigenvalues.clear();
    std::size_t pos = 0;
    while (true) {
      const std::size_t start = xml.find("<radial_function", pos);
      if (start == std::string::npos) break;
      if (start > 0 && xml[start - 1] == '/') { pos = start + 1; continue; }
      const std::size_t tag_end = xml.find('>', start);
      if (tag_end == std::string::npos) break;
      const std::string tag = xml.substr(start, tag_end - start + 1);

      int l = ExtractIntAttr(tag, "l");
      if (l == 0) l = ExtractIntAttr(tag, "angular_momentum");
      double eig = ExtractDoubleAttr(tag, "eigenvalue");
      if (eig == 0.0) eig = ExtractDoubleAttr(tag, "epsilon");
      if (eig == 0.0) eig = ExtractDoubleAttr(tag, "energy");

      l_values.push_back(l);
      eigenvalues.push_back(eig);

      const std::size_t end = xml.find("</radial_function>", tag_end);
      pos = (end == std::string::npos) ? tag_end + 1 : end + 18;
    }
  }

  // Build a radial grid from n, a, b, delta attributes (PSML grid spec).
  // PSML uses: r(i) = a * (exp(b * i) - 1)  for i = 0, ..., n-1
  // or r(i) = delta * i for linear grids.
  static std::vector<double> BuildGridFromAttrs(const std::string& tag_opening) {
    int n = ExtractIntAttr(tag_opening, "n");
    if (n <= 0) n = ExtractIntAttr(tag_opening, "size");
    if (n <= 0) return {};

    double a = ExtractDoubleAttr(tag_opening, "a");
    double b = ExtractDoubleAttr(tag_opening, "b");
    double delta = ExtractDoubleAttr(tag_opening, "delta");
    double scale = ExtractDoubleAttr(tag_opening, "scale");

    std::vector<double> grid(n);
    if (std::fabs(b) > 1e-15 && std::fabs(a) > 1e-15) {
      // Logarithmic grid: r(i) = a * (exp(b * i) - 1)
      for (int i = 0; i < n; ++i)
        grid[i] = a * (std::exp(b * static_cast<double>(i)) - 1.0);
    } else if (std::fabs(delta) > 1e-15) {
      // Linear grid: r(i) = delta * i
      for (int i = 0; i < n; ++i)
        grid[i] = delta * static_cast<double>(i);
    } else if (std::fabs(scale) > 1e-15) {
      // Scaled linear: r(i) = scale * i / (n - 1) * r_max
      double r_max = ExtractDoubleAttr(tag_opening, "rmax");
      if (r_max == 0.0) r_max = 20.0;
      for (int i = 0; i < n; ++i)
        grid[i] = scale * static_cast<double>(i) / static_cast<double>(n - 1) * r_max;
    } else {
      // Default: linear 0 to 20 Bohr.
      for (int i = 0; i < n; ++i)
        grid[i] = 20.0 * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return grid;
  }

  static std::vector<double> ParseDoubles(const std::string& s) {
    std::vector<double> out;
    std::istringstream iss(s);
    double v;
    while (iss >> v) out.push_back(v);
    return out;
  }

  static std::string ElementFromZ(int Z) {
    static const char* elements[] = {
        "",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc",
        "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
        "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc",
        "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
        "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"};
    if (Z >= 0 && Z <= 86) return elements[Z];
    return "X" + std::to_string(Z);
  }

  static std::string SimpleHash(const std::string& s) {
    std::uint64_t h = 14695981039346656037ULL;
    for (char c : s) {
      h ^= static_cast<std::uint8_t>(c);
      h *= 1099511628211ULL;
    }
    std::ostringstream os;
    os << std::hex << h;
    return os.str();
  }
};

}  // namespace tides::basis
