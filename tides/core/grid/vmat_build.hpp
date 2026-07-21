#pragma once

// T3.3: v(r) -> H tiles adjoint map.
// The density build (T3.2) maps P -> rho(r): rho(r) = sum_ij P_ij phi_i(r) phi_j(r).
// The adjoint map maps a potential v(r) back to the Hamiltonian tile:
//   H_ij = integral v(r) phi_i(r) phi_j(r) d^3r
// These two operations are adjoints: <A P, w> = <P, A^T w> where A is the
// orbital-product operator. The T3.3 observable is:
//   adjointness |<A P, w> - <P, A^T w>| <= 1e-12 on 100 random pairs (FP64).
//
// AUDIT C2/B3 (P2.8): Added GEMM-based methods that use BLAS dgemm for the
// orbital-product matmul. The GEMM formulation works from the density matrix P
// (not eigenvectors), fixing B3 (GPU path needed eigenvectors, breaking R2/R3).
// BuildRhoWithGrad also outputs ∇ρ, unblocking GGA end-to-end.
//
// For the CPU reference we implement both the forward (P -> rho) and adjoint
// (v -> H) using direct grid integration. The GPU tile-batched path uses WP1
// TileMat grouped GEMM for the orbital products; this CPU path is the FP64
// oracle.

#include <atomic>
#include <cmath>
#include <cstddef>
#include <vector>
#include <mutex>
#include <unordered_map>

#include "grid/dual_grid.hpp"
#include "tile/layout.hpp"

// BLAS double-precision general matrix multiply.
extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta,
            double* c, const int* ldc);
}

namespace tides::grid {

// Result struct for density build with gradient (P2.8).
struct RhoWithGrad {
  std::vector<double> rho;    // ρ(r) on grid
  std::vector<double> grad_x; // ∂ρ/∂x on grid
  std::vector<double> grad_y; // ∂ρ/∂y on grid
  std::vector<double> grad_z; // ∂ρ/∂z on grid
};

class VmatBuilder {
 public:
  // p2-vext-cpu: Process-level cache for the Phi (orbital flatten) buffer.
  // Phi depends only on the orbitals table (not on the potential v), so it
  // is identical across all BuildHmatGemm calls in the same SCF scope.
  // Keyed by (n_orb, N, orbitals data pointer). Thread-safe via mutex.
  struct PhiCacheKey {
    std::size_t n_orb;
    std::size_t N;
    const void* orb_ptr;
    bool operator==(const PhiCacheKey& o) const {
      return n_orb == o.n_orb && N == o.N && orb_ptr == o.orb_ptr;
    }
  };
  struct PhiCacheKeyHash {
    std::size_t operator()(const PhiCacheKey& k) const {
      return std::hash<std::size_t>{}(k.n_orb) ^
             (std::hash<std::size_t>{}(k.N) << 1) ^
             (std::hash<const void*>{}(k.orb_ptr) << 2);
    }
  };
  inline static std::unordered_map<PhiCacheKey, std::vector<double>,
                                  PhiCacheKeyHash> phi_cache;
  inline static std::mutex phi_cache_mutex;
 public:
  // Forward: density from density matrix P and orbitals.
  // rho(r) = sum_{ij} P_{ij} * phi_i(r) * phi_j(r)
  // P is symmetric; we sum over ALL i,j (not just lower triangle) so the
  // adjointness identity <A P, w> = <P, A^T w> holds exactly.
  static std::vector<double> BuildRho(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& P) {
    const std::size_t N = grid.total_points();
    const std::size_t n_orb = orbitals.size();
    std::vector<double> rho(N, 0.0);
    for (std::size_t i = 0; i < n_orb; ++i) {
      for (std::size_t j = 0; j < n_orb; ++j) {
        const double Pij = P[i * n_orb + j];
        if (std::fabs(Pij) < 1e-30) continue;
        for (std::size_t g = 0; g < N; ++g)
          rho[g] += Pij * orbitals[i][g] * orbitals[j][g];
      }
    }
    return rho;
  }

  // AUDIT C2/B3 (P2.8): GEMM-based density build.
  // Reformulates ρ(g) = Σ_ij P_ij φ_i(g) φ_j(g) as:
  //   1. temp = P @ Φ      (dgemm: n_orb × N_grid)
  //   2. ρ = Σ_i Φ_i ⊙ temp_i  (elementwise multiply + reduce over orbitals)
  // Uses the density matrix P (not eigenvectors), so it works for R2/R3.
  static std::vector<double> BuildRhoGemm(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& P) {
    const std::size_t N = grid.total_points();
    const std::size_t n_orb = orbitals.size();
    if (n_orb == 0 || N == 0) return {};

    // Flatten orbitals to row-major: Phi[i*N + g] = φ_i(g).
    std::vector<double> Phi(n_orb * N, 0.0);
    for (std::size_t i = 0; i < n_orb; ++i)
      std::copy(orbitals[i].begin(), orbitals[i].end(), Phi.begin() + i * N);

    // temp = P @ Phi  (n_orb × N_grid), column-major BLAS: dgemm with transa='T'
    // P is n_orb × n_orb (row-major = transposed in col-major), Phi is n_orb × N (row-major).
    // In col-major: dgemm computes C = alpha * A * B + beta * C.
    // We want temp = P * Phi in row-major.
    // Row-major C = A * B  ↔  Col-major C^T = B^T * A^T.
    // So: dgemm(transb='T', transa='T') with A=Phi, B=P gives temp_col = Phi^T * P^T = (P * Phi)^T.
    // Actually simpler: use dgemm with all row-major by swapping trans flags.
    // Row-major dgemm: C_rm = alpha * A_rm * B_rm → dgemm(transa='N', transb='N')
    // with A=P (lda=n_orb), B=Phi (ldb=N), C=temp (ldc=N), m=n_orb, n=N, k=n_orb.
    std::vector<double> temp(n_orb * N, 0.0);
    {
      int m = static_cast<int>(n_orb);
      int n = static_cast<int>(N);
      int k = static_cast<int>(n_orb);
      double alpha = 1.0, beta = 0.0;
      char transa = 'N', transb = 'N';
      // In col-major, A_rm is treated as A_cm^T. dgemm('N','N', n, m, k, alpha, B_rm, ldb, A_rm, lda, beta, C_rm^T, ldc)
      // Actually: to compute C_rm = A_rm * B_rm using col-major dgemm:
      // dgemm('N', 'N', &n, &m, &k, &alpha, B_rm, &n, A_rm, &k, &beta, C_rm, &n)
      // where C_rm is stored as n×m col-major = m×n row-major = C_rm.
      // Simpler approach: just call with swapped args.
      dgemm_(&transb, &transa, &n, &m, &k, &alpha,
             Phi.data(), &n,        // B_rm (col-major: N × n_orb)
             P.data(), &k,          // A_rm (col-major: n_orb × n_orb)
             &beta,
             temp.data(), &n);      // C_rm (col-major: N × n_orb = row-major: n_orb × N)
    }

    // ρ(g) = Σ_i Phi[i*N + g] * temp[i*N + g]
    std::vector<double> rho(N, 0.0);
    for (std::size_t i = 0; i < n_orb; ++i)
      for (std::size_t g = 0; g < N; ++g)
        rho[g] += Phi[i * N + g] * temp[i * N + g];
    return rho;
  }

  // AUDIT C2/B3 (P2.8): GEMM-based density build with gradient output.
  // Computes ρ and ∇ρ = 2 * Σ_ij P_ij ∇φ_i φ_j (using P symmetry).
  // grad_orbitals: [3][n_orb][N] — gradient of each orbital in x, y, z.
  static RhoWithGrad BuildRhoWithGrad(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& P,
      const std::array<std::vector<std::vector<double>>, 3>& grad_orbitals) {
    RhoWithGrad result;
    const std::size_t N = grid.total_points();
    const std::size_t n_orb = orbitals.size();
    if (n_orb == 0 || N == 0) return result;

    // Flatten orbitals.
    std::vector<double> Phi(n_orb * N, 0.0);
    for (std::size_t i = 0; i < n_orb; ++i)
      std::copy(orbitals[i].begin(), orbitals[i].end(), Phi.begin() + i * N);

    // temp = P @ Phi (same as BuildRhoGemm).
    std::vector<double> temp(n_orb * N, 0.0);
    {
      int m = static_cast<int>(n_orb);
      int n = static_cast<int>(N);
      int k = static_cast<int>(n_orb);
      double alpha = 1.0, beta = 0.0;
      char transa = 'N', transb = 'N';
      dgemm_(&transb, &transa, &n, &m, &k, &alpha,
             Phi.data(), &n, P.data(), &k, &beta, temp.data(), &n);
    }

    // ρ(g) = Σ_i φ_i(g) * temp_i(g)
    result.rho.assign(N, 0.0);
    for (std::size_t i = 0; i < n_orb; ++i)
      for (std::size_t g = 0; g < N; ++g)
        result.rho[g] += Phi[i * N + g] * temp[i * N + g];

    // ∇ρ_c(g) = 2 * Σ_i ∇φ_i,c(g) * temp_i(g)  (c = x, y, z)
    // Factor 2 from P symmetry: Σ_ij P_ij ∇φ_i φ_j = Σ_ij P_ji ∇φ_i φ_j
    //                           = Σ_ij P_ij φ_i ∇φ_j → average gives factor 2.
    for (int c = 0; c < 3; ++c) {
      auto& grad = (c == 0) ? result.grad_x : (c == 1) ? result.grad_y : result.grad_z;
      grad.assign(N, 0.0);
      for (std::size_t i = 0; i < n_orb; ++i) {
        for (std::size_t g = 0; g < N; ++g) {
          grad[g] += 2.0 * grad_orbitals[c][i][g] * temp[i * N + g];
        }
      }
    }
    return result;
  }

  // Adjoint: Hamiltonian tile from potential v(r) and orbitals.
  // H_ij = integral v(r) phi_i(r) phi_j(r) d^3r
  // Returns a symmetric matrix (lower triangle, row-major).
  static std::vector<double> BuildHmat(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& v) {
    const std::size_t N = grid.total_points();
    const std::size_t n_orb = orbitals.size();
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    std::vector<double> H(n_orb * n_orb, 0.0);
    for (std::size_t i = 0; i < n_orb; ++i) {
      for (std::size_t j = i; j < n_orb; ++j) {
        double s = 0.0;
        for (std::size_t g = 0; g < N; ++g)
          s += v[g] * orbitals[i][g] * orbitals[j][g];
        H[i * n_orb + j] = s * dv;
        H[j * n_orb + i] = s * dv;  // symmetric
      }
    }
    return H;
  }

  // AUDIT C2 (P2.8): GEMM-based Hamiltonian matrix build.
  // Reformulates H_ij = dv * Σ_g v(g) φ_i(g) φ_j(g) as:
  //   1. PhiV[i*N + g] = v(g) * φ_i(g)  (elementwise scale)
  //   2. H = dv * PhiV @ Phi^T           (dgemm: n_orb × n_orb)
  static std::vector<double> BuildHmatGemm(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& v) {
    const std::size_t N = grid.total_points();
    const std::size_t n_orb = orbitals.size();
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    if (n_orb == 0 || N == 0) return {};

    // p2-vext-cpu: Reuse cached Phi if available (thread-safe).
    // The orbital flatten is the SAME across all calls with the same orbitals
    // table; caching avoids re-flattening n_orb×N doubles on every call.
    const void* orb_ptr = orbitals.data();
    PhiCacheKey cache_key{n_orb, N, orb_ptr};
    const std::vector<double>* Phi_ptr = nullptr;
    {
      std::lock_guard<std::mutex> lk(phi_cache_mutex);
      auto it = phi_cache.find(cache_key);
      if (it != phi_cache.end()) {
        Phi_ptr = &it->second;
      }
    }
    // Flatten orbitals (and cache) if not found.
    std::vector<double> Phi_local;
    if (Phi_ptr == nullptr) {
      Phi_local.resize(n_orb * N, 0.0);
      for (std::size_t i = 0; i < n_orb; ++i)
        std::copy(orbitals[i].begin(), orbitals[i].end(),
                  Phi_local.begin() + i * N);
      std::lock_guard<std::mutex> lk(phi_cache_mutex);
      auto [ins_it, inserted] = phi_cache.emplace(cache_key, std::move(Phi_local));
      Phi_ptr = &ins_it->second;
    }
    const auto& Phi = *Phi_ptr;

    // PhiV depends on v, so must be computed each call.
    std::vector<double> PhiV(n_orb * N, 0.0);
    for (std::size_t i = 0; i < n_orb; ++i) {
      for (std::size_t g = 0; g < N; ++g) {
        PhiV[i * N + g] = v[g] * Phi[i * N + g];
      }
    }

    // H = dv * PhiV @ Phi^T  (n_orb × n_orb)
    // Row-major: C_rm = A_rm * B_rm^T → dgemm(transa='N', transb='T')
    // In col-major: dgemm('T', 'N', n, m, k, alpha, B_rm, ldb, A_rm, lda, beta, C_rm, ldc)
    std::vector<double> H(n_orb * n_orb, 0.0);
    {
      int m = static_cast<int>(n_orb);
      int n = static_cast<int>(n_orb);
      int k = static_cast<int>(N);
      double alpha = dv, beta = 0.0;
      char transa = 'T', transb = 'N';
      // C_rm = A_rm * B_rm^T. Col-major: C_cm = (A_rm * B_rm^T)^T = B_rm * A_rm^T
      // dgemm('T', 'N', &n, &m, &k, &alpha, Phi, &k, PhiV, &k, &beta, H, &n)
      dgemm_(&transa, &transb, &n, &m, &k, &alpha,
             Phi.data(), &k,     // B_rm transposed in col-major
             PhiV.data(), &k,    // A_rm in col-major
             &beta,
             H.data(), &n);      // C_rm (col-major: n_orb × n_orb)
    }

    // Symmetrize.
    for (std::size_t i = 0; i < n_orb; ++i)
      for (std::size_t j = i + 1; j < n_orb; ++j)
        H[j * n_orb + i] = H[i * n_orb + j];

    return H;
  }

  // GGA adjoint: builds V_xc matrix from GGA XC outputs.
  // V_ij = Σ_g [wv_rho(g) * φ_i(g) * φ_j(g)
  //            + wv_grad(g) · (∇φ_i * φ_j + φ_i * ∇φ_j)]
  // wv_rho and wv_grad are FULLY weighted planes, matching the GPU XC kernel
  // convention (xc/kernels/reduce.cuh):
  //   wv_rho(g)    = dv * vrho(g)
  //   wv_grad_c(g) = 2 * dv * vsigma(g) * grad_rho_c(g)
  // The builder applies the planes as-is — it must NOT multiply by grad_rho
  // again (a previous version did, giving a (∇ρ)² term that diverged for
  // steep core densities).
  // grad_orbitals: [3][n_orb][N] — gradient of each orbital in x, y, z.
  static std::vector<double> BuildGgaHmatGemm(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::array<std::vector<std::vector<double>>, 3>& grad_orbitals,
      const std::vector<double>& wv_rho,
      const std::vector<double>& wv_grad_x,
      const std::vector<double>& wv_grad_y,
      const std::vector<double>& wv_grad_z) {
    const std::size_t N = grid.total_points();
    const std::size_t n_orb = orbitals.size();
    if (n_orb == 0 || N == 0) return {};

    std::vector<double> H(n_orb * n_orb, 0.0);

    // Term 1: LDA-like (wv_rho as potential, already includes dv weighting).
    // Do NOT call BuildHmatGemm here, because BuildHmatGemm would multiply by
    // dv again. Compute the weighted sum directly.
    std::vector<double> H_rho(n_orb * n_orb, 0.0);
    for (std::size_t i = 0; i < n_orb; ++i) {
      for (std::size_t j = i; j < n_orb; ++j) {
        double s = 0.0;
        for (std::size_t g = 0; g < N; ++g)
          s += wv_rho[g] * orbitals[i][g] * orbitals[j][g];
        H_rho[i * n_orb + j] = s;
        H_rho[j * n_orb + i] = s;
      }
    }

    // Term 2: gradient contributions.
    //   H_grad_ij += Σ_g wv_grad_c(g) * [∇φ_i,c(g) * φ_j(g) + φ_i(g) * ∇φ_j,c(g)]
    // = dgemm(GradPhi_w, Phi^T) + dgemm(Phi_w, GradPhi^T) where _w means scaled
    // by wv_grad_c.
    for (int c = 0; c < 3; ++c) {
      const auto& wgc_scaled = (c == 0)   ? wv_grad_x
                               : (c == 1) ? wv_grad_y
                                          : wv_grad_z;

      // Flatten grad_orbitals for this component and scale by wgc_scaled.
      std::vector<double> GradPhi(n_orb * N, 0.0);
      std::vector<double> GradPhiW(n_orb * N, 0.0);
      std::vector<double> Phi(n_orb * N, 0.0);
      std::vector<double> PhiW(n_orb * N, 0.0);
      for (std::size_t i = 0; i < n_orb; ++i) {
        for (std::size_t g = 0; g < N; ++g) {
          Phi[i * N + g] = orbitals[i][g];
          PhiW[i * N + g] = wgc_scaled[g] * orbitals[i][g];
          GradPhi[i * N + g] = grad_orbitals[c][i][g];
          GradPhiW[i * N + g] = wgc_scaled[g] * grad_orbitals[c][i][g];
        }
      }

      // H_grad += GradPhiW @ Phi^T  (contribution from ∇φ_i * φ_j term)
      std::vector<double> H_grad(n_orb * n_orb, 0.0);
      {
        int m = static_cast<int>(n_orb);
        int n = static_cast<int>(n_orb);
        int k = static_cast<int>(N);
        double alpha = 1.0, beta = 0.0;
        char transa = 'T', transb = 'N';
        dgemm_(&transa, &transb, &n, &m, &k, &alpha,
               Phi.data(), &k, GradPhiW.data(), &k,
               &beta, H_grad.data(), &n);
      }
      // H_grad += PhiW @ GradPhi^T  (contribution from φ_i * ∇φ_j term)
      {
        int m = static_cast<int>(n_orb);
        int n = static_cast<int>(n_orb);
        int k = static_cast<int>(N);
        double alpha = 1.0, beta = 1.0;
        char transa = 'T', transb = 'N';
        dgemm_(&transa, &transb, &n, &m, &k, &alpha,
               GradPhi.data(), &k, PhiW.data(), &k,
               &beta, H_grad.data(), &n);
      }

      for (std::size_t i = 0; i < n_orb * n_orb; ++i)
        H[i] += H_grad[i];
    }

    // Add LDA-like term.
    for (std::size_t i = 0; i < n_orb * n_orb; ++i)
      H[i] += H_rho[i];

    // Symmetrize.
    for (std::size_t i = 0; i < n_orb; ++i)
      for (std::size_t j = i + 1; j < n_orb; ++j)
        H[j * n_orb + i] = H[i * n_orb + j];

    return H;
  }

  // Adjointness check: <A P, w> should equal <P, A^T w> for random P, w.
  // A is the forward operator (P -> rho), A^T is the adjoint (v -> H).
  // <A P, w> = integral (A P)(r) * w(r) d^3r = integral rho_P(r) * w(r) d^3r
  // <P, A^T w> = sum_{ij} P_ij * H_w[ij] = sum_{ij} P_ij * integral w(r) phi_i phi_j d^3r
  // These are equal by construction (Fubini's theorem); the test validates the
  // implementation doesn't have a transposition/discretization bug.
  static double CheckAdjointness(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& P, const std::vector<double>& w) {
    const std::size_t N = grid.total_points();
    const std::size_t n_orb = orbitals.size();
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;

    // Forward: rho_P = A(P)
    auto rho_P = BuildRho(grid, orbitals, P);
    // <A P, w> = integral rho_P(r) * w(r) d^3r
    double AP_w = 0.0;
    for (std::size_t g = 0; g < N; ++g)
      AP_w += rho_P[g] * w[g] * dv;

    // Adjoint: H_w = A^T(w)
    auto H_w = BuildHmat(grid, orbitals, w);
    // <P, A^T w> = sum_{ij} P_ij * H_w[ij]
    double P_ATw = 0.0;
    for (std::size_t i = 0; i < n_orb; ++i)
      for (std::size_t j = 0; j < n_orb; ++j)
        P_ATw += P[i * n_orb + j] * H_w[i * n_orb + j];

    return std::fabs(AP_w - P_ATw);
  }

  // AUDIT C2 FIX: TileMat-based H matrix build.
  // Performs the same dgemm as BuildHmatGemm, then wraps the result in a
  // TileMat with the specified tile edge. This makes the tile substrate part
  // of the actual product path, not just stats/verification.
  static tides::Result<tides::tile::TileMat> BuildHmatTile(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& v,
      std::uint32_t tile_edge) {
    auto H_dense = BuildHmatGemm(grid, orbitals, v);
    const std::size_t n_orb = orbitals.size();
    return tides::tile::TileMat::FromDense(n_orb, n_orb, H_dense, tile_edge,
                                           tides::tile::Symmetry::kSymmetric);
  }

  // AUDIT C2 FIX: Tile-based trace(P, H) = sum_ij P_ij * H_ij.
  // Uses TileMat elementwise access — same O(n²) as dense, but goes through
  // the tile substrate, making TileMat the canonical matrix representation.
  static double TileTrace(const tides::tile::TileMat& P,
                          const tides::tile::TileMat& H) {
    const std::size_t n = P.rows();
    const auto P_dense = P.ToDense();
    const auto H_dense = H.ToDense();
    double tr = 0.0;
    for (std::size_t i = 0; i < n * n; ++i)
      tr += P_dense[i] * H_dense[i];
    return tr;
  }
};

}  // namespace tides::grid
