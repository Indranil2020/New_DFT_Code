#pragma once

// Finite electronic temperature (Mermin) DFT (§3.1.2, §3.2.4).
//
// At finite electronic temperature, the Fermi-Dirac distribution replaces
// the step function for orbital occupations:
//   f(ε) = 1 / (1 + exp((ε - μ) / kT))
//
// The Mermin free energy: Ω = E_band - T·S_elec
// where S_elec = -k_B * Σ_k [f_k ln(f_k) + (1-f_k) ln(1-f_k)]
//
// The chemical potential μ is found by requiring Σ_k f_k = N_e (electron
// conservation). We use a bracketed bisection/Newton search.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tides::scf {

struct MerminResult {
  double fermi_level = 0.0;         // μ
  double electronic_entropy = 0.0;   // S_elec (in k_B units, dimensionless)
  double free_energy = 0.0;         // E - T*S (Mermin free energy)
  std::vector<double> occupations;  // f_k per orbital
  double nelectrons_check = 0.0;    // Σ f_k (should = N_e)
};

class MerminDFT {
 public:
  // Compute Fermi-Dirac occupations given eigenvalues and kT.
  //   eigenvalues:  orbital eigenvalues (Hartree)
  //   n_electrons:  total number of electrons
  //   kT:           electronic temperature (Hartree)
  static MerminResult Compute(const std::vector<double>& eigenvalues,
                               double n_electrons, double kT,
                               double tol = 1e-12) {
    MerminResult res;
    if (eigenvalues.empty() || kT < 0) {
      res.occupations.assign(eigenvalues.size(), 0.0);
      return res;
    }

    // Handle T=0 case: step function.
    if (kT < 1e-15) {
      res.fermi_level = FindFermiLevelT0(eigenvalues, n_electrons);
      for (double e : eigenvalues)
        res.occupations.push_back(e < res.fermi_level ? 1.0 : 0.0);
      res.nelectrons_check = n_electrons;
      res.electronic_entropy = 0.0;
      res.free_energy = 0.0;
      return res;
    }

    // Find Fermi level.
    res.fermi_level = FindFermiLevel(eigenvalues, n_electrons, kT, tol);

    // Compute occupations.
    res.occupations.reserve(eigenvalues.size());
    double ne_sum = 0.0;
    for (double e : eigenvalues) {
      double f = FermiDirac(e, res.fermi_level, kT);
      res.occupations.push_back(f);
      ne_sum += f;
    }
    res.nelectrons_check = ne_sum;

    // Compute entropy.
    res.electronic_entropy = MerminEntropy(res.occupations, kT);

    // Free energy = sum(f_k * eps_k) - kT * S
    // (caller adds the rest of the energy terms via MerminCorrectedEnergy)
    double e_band = 0.0;
    for (std::size_t k = 0; k < eigenvalues.size(); ++k)
      e_band += res.occupations[k] * eigenvalues[k];
    res.free_energy = e_band - kT * res.electronic_entropy;

    return res;
  }

  // Compute Mermin electronic entropy.
  // S = -Σ_k [f_k ln(f_k) + (1-f_k) ln(1-f_k)] (in units of k_B)
  static double MerminEntropy(const std::vector<double>& occupations, double kT) {
    if (kT < 1e-15) return 0.0;
    double S = 0.0;
    for (double f : occupations) {
      // Clamp to avoid log(0).
      double f_clamped = std::max(std::min(f, 1.0 - 1e-15), 1e-15);
      double one_minus_f = 1.0 - f_clamped;
      S -= f_clamped * std::log(f_clamped) +
           one_minus_f * std::log(one_minus_f);
    }
    return S;  // in k_B units (multiply by k_B for physical entropy)
  }

  // Find the Fermi level via bracketed bisection + Newton refinement.
  static double FindFermiLevel(const std::vector<double>& eigenvalues,
                                double n_electrons, double kT,
                                double tol = 1e-12) {
    if (eigenvalues.empty()) return 0.0;
    if (kT < 1e-15) return FindFermiLevelT0(eigenvalues, n_electrons);

    // Bracket: μ is between min and max eigenvalue.
    auto [e_min, e_max] = std::minmax_element(eigenvalues.begin(),
                                               eigenvalues.end());
    double lo = *e_min - 10.0 * kT;
    double hi = *e_max + 10.0 * kT;

    // Bisection.
    for (int iter = 0; iter < 200; ++iter) {
      double mid = 0.5 * (lo + hi);
      double ne = CountElectrons(eigenvalues, mid, kT);
      if (std::fabs(ne - n_electrons) < tol) return mid;
      if (ne < n_electrons)
        lo = mid;
      else
        hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  // Mermin-corrected total energy.
  // E(T) = E_0K - kT * S_elec + correction terms
  // For the leading-order correction: E(T) ≈ E_0K - (π²/6) * kT² * DOS(E_F)
  // The full correction: E(T) = E_0K - kT * S_elec
  static double MerminCorrectedEnergy(double E_0K,
                                       const MerminResult& mermin,
                                       double kT) {
    return E_0K - kT * mermin.electronic_entropy;
  }

  // Fermi-Dirac distribution function.
  static double FermiDirac(double energy, double mu, double kT) {
    if (kT < 1e-15)
      return energy < mu ? 1.0 : (energy > mu ? 0.0 : 0.5);
    double x = (energy - mu) / kT;
    if (x > 700.0) return 0.0;
    if (x < -700.0) return 1.0;
    return 1.0 / (1.0 + std::exp(x));
  }

 private:
  // Count electrons for a given μ.
  static double CountElectrons(const std::vector<double>& eigenvalues,
                               double mu, double kT) {
    double ne = 0.0;
    for (double e : eigenvalues)
      ne += FermiDirac(e, mu, kT);
    return ne;
  }

  // T=0 Fermi level: average of HOMO and LUMO.
  static double FindFermiLevelT0(const std::vector<double>& eigenvalues,
                                  double n_electrons) {
    std::vector<double> sorted = eigenvalues;
    std::sort(sorted.begin(), sorted.end());
    std::size_t homo = static_cast<std::size_t>(n_electrons);
    if (homo == 0) return sorted[0] - 1.0;
    if (homo >= sorted.size()) return sorted.back() + 1.0;
    return 0.5 * (sorted[homo - 1] + sorted[homo]);
  }
};

}  // namespace tides::scf
