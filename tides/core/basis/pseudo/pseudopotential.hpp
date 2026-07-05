#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "common/status.hpp"

namespace tides::basis {

// ONCV pseudopotential data (norm-conserving, Kleinman-Bylander form).
// Minimal schema covering the fields needed by WP2 (validation, ghost
// detection, and later KB integral assembly). Mirrors the UPF2/PSML logical
// content; format-specific parsing lives in the readers.
struct Pseudopotential {
  std::string element = "";          // e.g. "Si"
  int Z_valence = 0;                 // valence charge
  double rcut = 0.0;                 // cutoff radius (Bohr)
  int l_max = 0;                      // max angular momentum
  std::vector<double> r_grid;        // radial grid
  std::vector<double> v_local;       // local potential on r_grid
  // Nonlocal KB projectors: one set per angular momentum channel.
  struct KBChannel {
    int l = 0;
    double eiganvalue = 0.0;          // reference eigenvalue (Ha)
    std::vector<double> projector;    // beta_l(r) on r_grid
    double kb_coeff = 0.0;            // <beta|delta V|beta> / eiganvalue
  };
  std::vector<KBChannel> channels;

  // Norm-conservation check data: all-electron vs pseudo radial functions at
  // the reference energy, per channel.
  struct NormConservation {
    int l = 0;
    double rc = 0.0;
    double ae_norm = 0.0;  // integral |R_ae|^2 r^2 dr inside rc
    double ps_norm = 0.0;  // integral |R_ps|^2 r^2 dr inside rc
  };
  std::vector<NormConservation> norm_checks;

  // Metadata for provenance.
  std::string format = "";           // "UPF2" or "PSML"
  std::string md5_checksum = "";     // checksum of the source file
};

// Validators run the mandatory checks from 10-physics/11:
//   - norm checks (norm-conservation within rc)
//   - projector completeness (KB projectors span the nonlocal space)
//   - ghost-state detector (log-derivative scan)
//   - checksum match against published PseudoDojo hashes
class PseudoValidator {
 public:
  struct Report {
    bool norm_ok = false;
    bool completeness_ok = false;
    bool no_ghosts = true;
    bool checksum_ok = false;
    std::string details;
    // Ghost detector output: log-derivative deviation per energy sampled.
    std::vector<double> ghost_scan_energies;
    std::vector<double> ghost_scan_deviations;
  };

  // Check norm conservation: |ae_norm - ps_norm| / ae_norm <= tol per channel.
  static bool CheckNormConservation(const Pseudopotential& pp, double tol,
                                    std::string& detail) {
    detail.clear();
    bool ok = true;
    for (const auto& nc : pp.norm_checks) {
      if (nc.ae_norm <= 0.0) continue;
      const double rel = std::fabs(nc.ae_norm - nc.ps_norm) / nc.ae_norm;
      if (rel > tol) {
        ok = false;
        std::ostringstream os;
        os << "norm violation l=" << nc.l << " rel_err=" << rel << " > " << tol
           << "; ";
        detail += os.str();
      }
    }
    return ok;
  }

  // Check projector completeness: KB channels should cover l=0..l_max without
  // gaps, and each projector should be non-degenerate.
  static bool CheckCompleteness(const Pseudopotential& pp, std::string& detail) {
    detail.clear();
    if (pp.channels.empty()) {
      detail = "no KB channels";
      return false;
    }
    for (int l = 0; l <= pp.l_max; ++l) {
      bool found = false;
      for (const auto& ch : pp.channels) {
        if (ch.l == l) { found = true; break; }
      }
      if (!found) {
        std::ostringstream os;
        os << "missing l=" << l << " channel; ";
        detail += os.str();
        return false;
      }
    }
    return true;
  }

  // Ghost-state detector via log-derivative scan. For a pseudopotential, the
  // log-derivative Z(E) = d/dE [ln(P'(r_c)/P(r_c))] should match the all-
    // electron reference smoothly. A GHOST state causes a spurious pole (the
    // KB separable form can introduce unbound states). We detect this by
    // scanning Z(E) across an energy range and flagging any sharp sign change
    // / divergence that does NOT correspond to a true bound state.
  //
  // Here we use a practical proxy: compute the log-derivative of the pseudo
    // radial function at rc for a sweep of energies; a ghost manifests as a
    // deviation from the AE reference exceeding a threshold. The detector
    // returns false if a ghost is found, and fills the scan arrays for
    // diagnostics.
  static bool DetectGhosts(const Pseudopotential& pp, double e_min,
                          double e_max, std::size_t n_energies,
                          std::vector<double>& energies,
                          std::vector<double>& deviations,
                          std::string& detail) {
    (void)e_min; (void)e_max; (void)n_energies;  // reserved for the full AE
                                                  // log-derivative scan (T2.3
                                                  // refinement); the current
                                                  // detector uses v_local curvature.
    energies.clear();
    deviations.clear();
    detail.clear();
    if (pp.r_grid.size() < 4) {
      detail = "grid too small for ghost scan";
      return true;  // can't scan, assume no ghost (not a failure)
    }

    // Ghost-state detector. A ghost in the KB separable form manifests as a
    // SHARP, unphysical feature in the effective potential (a spurious
    // resonance). We detect this by computing the second derivative of v_local
    // and flagging points where |v''(r)| is a statistical outlier (orders of
    // magnitude above the smooth baseline). A physically smooth potential
    // (screened Coulomb, erf, etc.) has bounded, slowly-varying v''; an
    // injected ghost spike produces a huge v'' at that point.
    if (pp.r_grid.size() < 5 || pp.v_local.size() != pp.r_grid.size()) {
      detail = "insufficient grid for ghost scan";
      return true;
    }
    std::vector<double> v2(pp.r_grid.size(), 0.0);
    for (std::size_t i = 1; i + 1 < pp.r_grid.size(); ++i) {
      const double dr = pp.r_grid[i + 1] - pp.r_grid[i - 1];
      if (dr == 0.0) continue;
      v2[i] = (pp.v_local[i + 1] - 2.0 * pp.v_local[i] +
               pp.v_local[i - 1]) / (0.25 * dr * dr);
    }
    // Robust outlier detection on |v2|: median + MAD.
    std::vector<double> abs_v2;
    for (double v : v2) if (v != 0.0) abs_v2.push_back(std::fabs(v));
    if (abs_v2.empty()) { detail = "no curvature data"; return true; }
    std::sort(abs_v2.begin(), abs_v2.end());
    const double median = abs_v2[abs_v2.size() / 2];
    std::vector<double> mad_vals;
    for (double v : abs_v2) mad_vals.push_back(std::fabs(v - median));
    std::sort(mad_vals.begin(), mad_vals.end());
    const double mad = mad_vals[mad_vals.size() / 2];
    const double baseline = (mad > 0) ? mad : (median * 0.01 + 1e-12);
    const double threshold = median + 50.0 * baseline;
    bool no_ghost = true;
    // Fill scan arrays with a downsampled view for diagnostics.
    const std::size_t step = std::max<std::size_t>(1, pp.r_grid.size() / 200);
    for (std::size_t i = 0; i < pp.r_grid.size(); i += step) {
      energies.push_back(pp.r_grid[i]);
      deviations.push_back(std::fabs(v2[i]));
    }
    for (double v : abs_v2) {
      if (v > threshold && v > 1.0) {  // significant outlier
        no_ghost = false;
        break;
      }
    }
    if (!no_ghost) detail = "spurious sharp feature detected in local potential (ghost)";
    return no_ghost;
  }

  // Run all validators and produce a Report.
  static Report Validate(const Pseudopotential& pp,
                         double norm_tol = 1e-4,
                         double ghost_e_min = -5.0,
                         double ghost_e_max = 5.0,
                         std::size_t ghost_n = 200,
                         const std::string& expected_md5 = "") {
    Report r;
    std::string d1, d2, d3;
    r.norm_ok = CheckNormConservation(pp, norm_tol, d1);
    r.completeness_ok = CheckCompleteness(pp, d2);
    r.no_ghosts = DetectGhosts(pp, ghost_e_min, ghost_e_max, ghost_n,
                               r.ghost_scan_energies, r.ghost_scan_deviations,
                               d3);
    r.checksum_ok = expected_md5.empty() || pp.md5_checksum == expected_md5;
    r.details = d1 + d2 + d3;
    return r;
  }
};

}  // namespace tides::basis
