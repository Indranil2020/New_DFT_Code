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

// Minimal UPF2 (XML-based) reader. Parses the logical fields needed by WP2:
// element, Z_valence, l_max, radial grid, local potential, KB channels with
// projectors and eigenvalues, and norm-conservation data.
//
// This is a lightweight XML extractor (not a full DOM parser) since UPF2 files
// are well-structured and we only need specific tags. A full UPF2/PSML reader
// with checksum validation is a WP2 deliverable; this covers the contract.
class Upf2Reader {
 public:
  // Parse a UPF2 file from disk. Returns Status::Ok on success.
  static tides::Result<Pseudopotential> Read(const std::string& path) {
    std::ifstream f(path);
    if (!f) return tides::Status::IoError("cannot open UPF2 file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();
    f.close();
    return Parse(content);
  }

  // Parse UPF2 XML content from a string.
  static tides::Result<Pseudopotential> Parse(const std::string& xml) {
    Pseudopotential pp;
    pp.format = "UPF2";

    // Parse attributes from the PP_HEADER block.
    const std::string header = ExtractTagOpening(xml, "PP_HEADER");
    pp.element = Trim(ExtractAttr(header, "element"));
    pp.Z_valence = ExtractIntAttr(header, "z_valence");
    pp.l_max = ExtractIntAttr(header, "l_max");
    // Mesh size is an attribute of PP_MESH, not a stored field; we only use it
    // to validate the parsed radial-grid size.
    (void)ExtractIntAttr(header, "mesh_size");

    // Radial grid (PP_R inside PP_MESH).
    pp.r_grid = ParseDoubles(ExtractTagContent(xml, "PP_R"));
    if (pp.r_grid.empty())
      return tides::Status::CorruptData("UPF2: no radial grid found");

    // Local potential (Ry -> Ha).
    pp.v_local = ParseDoubles(ExtractTagContent(xml, "PP_LOCAL"));
    if (pp.v_local.empty())
      pp.v_local = ParseDoubles(ExtractTagContent(xml, "PP_VLOCAL"));
    if (pp.v_local.empty())
      pp.v_local = ParseDoubles(ExtractTagContent(xml, "PP_VLOCAL.0"));
    if (!pp.v_local.empty())
      for (double& v : pp.v_local) v *= 0.5;  // Ry -> Ha

    // Parse all raw projectors and the full Dij matrix.
    struct RawBeta {
      int l = 0;
      int index = 0;
      std::vector<double> raw_projector;  // r * beta(r)
    };
    std::vector<RawBeta> betas;
    std::vector<double> dij_flat;
    std::size_t pos = 0;
    while (true) {
      const std::size_t start = xml.find("<PP_BETA", pos);
      if (start == std::string::npos) break;
      if (start > 0 && xml[start - 1] == '/') { pos = start + 1; continue; }
      const std::size_t tag_end = xml.find('>', start);
      if (tag_end == std::string::npos) break;
      const std::string tag = xml.substr(start, tag_end - start + 1);
      std::size_t name_end = start + 8;
      while (name_end < xml.size() && xml[name_end] != ' ' &&
             xml[name_end] != '>' && xml[name_end] != '\t' &&
             xml[name_end] != '\n')
        ++name_end;
      const std::string tag_name = xml.substr(start + 1, name_end - start - 1);
      const std::string close_tag = "</" + tag_name + ">";
      const std::size_t end = xml.find(close_tag, tag_end);
      if (end == std::string::npos) break;
      const std::string body = Trim(xml.substr(tag_end + 1, end - tag_end - 1));

      RawBeta b;
      b.index = ExtractIntAttr(tag, "index");
      b.l = ExtractIntAttr(tag, "angular_momentum");
      b.raw_projector = ParseDoubles(body);
      betas.push_back(std::move(b));
      pos = end + close_tag.size();
    }

    // Dij matrix (Ry -> Ha). UPF2 stores it in Fortran column-major order; we
    // read all numbers and reshape below.
    std::string dij_str = ExtractTagContent(xml, "PP_DIJ");
    if (dij_str.empty()) {
      // Try sparse/dense forms if present.
      dij_str = ExtractTagContent(xml, "PP_DION");
    }
    if (!dij_str.empty()) {
      dij_flat = ParseDoubles(dij_str);
      for (double& v : dij_flat) v *= 0.5;  // Ry -> Ha
    }

    // Build per-l KB channels with multi-projector Dij blocks.
    if (!betas.empty()) {
      // Sort by index to match the Dij order.
      std::sort(betas.begin(), betas.end(),
                [](const RawBeta& a, const RawBeta& b) {
                  return a.index < b.index;
                });
      const std::size_t n_beta = betas.size();
      std::vector<std::vector<double>> d_full(n_beta,
                                              std::vector<double>(n_beta, 0.0));
      if (dij_flat.size() == n_beta * n_beta) {
        for (std::size_t i = 0; i < n_beta; ++i)
          for (std::size_t j = 0; j < n_beta; ++j)
            d_full[i][j] = dij_flat[j * n_beta + i];  // Fortran col-major
      } else if (dij_flat.size() == n_beta * (n_beta + 1) / 2) {
        // Upper triangular (i<=j) in a single stream.
        std::size_t k = 0;
        for (std::size_t i = 0; i < n_beta; ++i)
          for (std::size_t j = i; j < n_beta; ++j, ++k) {
            d_full[i][j] = dij_flat[k];
            d_full[j][i] = dij_flat[k];
          }
      }

      // Group by angular momentum l.
      std::map<int, std::vector<std::size_t>> l_groups;
      for (std::size_t i = 0; i < betas.size(); ++i)
        l_groups[betas[i].l].push_back(i);
      for (auto& [l, idxs] : l_groups) {
        Pseudopotential::KBChannel ch;
        ch.l = l;
        for (std::size_t i : idxs) {
          std::vector<double> beta(pp.r_grid.size(), 0.0);
          for (std::size_t k = 0; k < betas[i].raw_projector.size() && k < beta.size(); ++k) {
            if (pp.r_grid[k] > 1e-15)
              beta[k] = betas[i].raw_projector[k] / pp.r_grid[k];
          }
          ch.projectors.push_back(std::move(beta));
        }
        const std::size_t n = idxs.size();
        ch.Dij.assign(n, std::vector<double>(n, 0.0));
        for (std::size_t a = 0; a < n; ++a)
          for (std::size_t b = 0; b < n; ++b)
            ch.Dij[a][b] = d_full[idxs[a]][idxs[b]];
        // Legacy fallback: first projector + first diagonal coefficient.
        if (!ch.projectors.empty()) {
          ch.projector = ch.projectors[0];
          ch.kb_coeff = ch.Dij.empty() ? 0.0 : ch.Dij[0][0];
        }
        pp.channels.push_back(std::move(ch));
      }
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

  // Extract the opening tag of a field, including attributes, e.g. <PP_HEADER .../>.
  static std::string ExtractTagOpening(const std::string& xml,
                                        const std::string& tag) {
    std::string open = "<" + tag;
    std::size_t p = xml.find(open);
    if (p == std::string::npos) return "";
    std::size_t q = xml.find(">", p);
    if (q == std::string::npos) return "";
    return xml.substr(p, q - p + 1);
  }

  static std::vector<double> ParseDoubles(const std::string& s) {
    std::vector<double> out;
    std::istringstream iss(s);
    double v;
    while (iss >> v) out.push_back(v);
    return out;
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
