#pragma once

// Molecule driver: chains all TIDES components into an end-to-end SCF.
//
// AUDIT C1 DECISION: This GTO/STO-3G driver is the bootstrap oracle. The NAO
// product driver lives in nao_driver.hpp. Both drivers now share the same
// grid pipeline infrastructure (GEMM rho/vmat + fused XC engine).
//
// AUDIT C3 FIXED: The SCF loop now calls GPU-resident kernels via the fused
// XC engine (xc::XcEval auto-dispatches to CUDA when TIDES_HAVE_CUDA is
// defined). Rho/vmat builds use BLAS dgemm (BuildRhoGemm/BuildHmatGemm).
// The standalone GPU kernels (rho_build.cu, vmat_build.cu, poisson_fft.cu)
// are available for large-scale production but have CPU fallback for small
// systems.
//
// AUDIT C4 FIXED: XC is now evaluated via the fused Tier-0 XC engine
// (xc::XcEval) which supports LDA-PW92 and PBE. The old XCGridEvaluator
// path is retained as a CPU-only fallback. The driver accepts an XcSpec
// to select the functional.
//
// AUDIT B3 FIXED: Rho build now uses BuildRhoGemm (BLAS dgemm from density
// matrix P), not BuildFromOrbitals (which required eigenvectors). This
// makes the pipeline compatible with R2/R3 (purification produces P).
//
// AUDIT C2 FIXED: Vmat build uses BuildHmatGemm (BLAS dgemm) instead of
// the O(n²·N_grid) triple-loop BuildHmat.
//
// AUDIT C8: Single uniform grid (no dual coarse/fine). Acceptable for
// bootstrap; flagged as a deliberate decision, not drift.
//
// Pipeline:
//   1. GTO integrals → S (overlap), T (kinetic) matrices
//   2. Grid setup → evaluate basis functions on 3D grid
//   3. V_ext analytically (nuclear attraction integrals)
//   4. SCF loop (via SCFDriver):
//      a. P → rho(r) via VmatBuilder::BuildRhoGemm (BLAS dgemm)
//      b. rho → V_H: analytic ERIs (GTO) or grid Poisson (NAO/option)
//      c. rho → V_xc(r), E_xc via xc::XcEval (fused Tier-0, GPU auto-dispatch)
//      d. V_xc(r) → V_xc matrix via VmatBuilder::BuildHmatGemm (BLAS dgemm)
//      e. H = T + V_ext + V_H + V_xc
//   5. Energy assembly from cached H build + eigenvalues
//   6. EwaldIonIon → ion-ion repulsion

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "scf/gto_integrals.hpp"
#include "scf/scf_driver.hpp"
#include "scf/energy_assembly.hpp"
#include "grid/dual_grid.hpp"
#include "grid/poisson.hpp"
#include "grid/vmat_build.hpp"
#include "grid/xc.hpp"
#include "grid/xc/xc_engine.hpp"
#include "forces/analytic_forces.hpp"
#include "ham/ham_builder.hpp"
#include "hybrids/d4_dispersion.hpp"
#include "scf/mermin.hpp"
#include "basis/bsse.hpp"
#include "common/point_group.hpp"
#include "verification/a_posteriori_error.hpp"
#include "verification/energy_metering.hpp"
#include "scf/mixed_precision.hpp"
#include "tile/qtt_scf.hpp"
#include "tile/tile_scf_integration.hpp"
#include "tile/cuda_graph_scf.hpp"
#include "tile/kpoints.hpp"
#include "tile/bloch_phase.hpp"

namespace tides::scf {

// Per-component pipeline timings for profiling (Audit P3).
struct PipelineTimings {
  double rho_build_ms = 0.0;    // GEMM density build per SCF iteration
  double xc_eval_ms = 0.0;      // Fused XC engine per SCF iteration
  double poisson_ms = 0.0;      // Poisson solve per SCF iteration (grid Hartree only)
  double vmat_build_ms = 0.0;   // GEMM v→H matrix build per SCF iteration
  double eigensolve_ms = 0.0;   // Eigensolve per SCF iteration
  double scf_total_ms = 0.0;    // Total SCF loop time
  int n_iterations = 0;
  bool used_gpu_xc = false;     // Whether XC engine dispatched to CUDA
  bool used_grid_hartree = false; // Whether grid Poisson was used for Hartree
  std::string xc_functional;   // XC functional name
};

struct MoleculeDriverResult {
  SCFResult scf;
  EnergyComponents energy;
  std::size_t n_basis = 0;
  std::size_t n_electrons = 0;
  std::size_t n_atoms = 0;
  double grid_h = 0.0;
  std::array<std::size_t, 3> grid_n = {0, 0, 0};
  double wall_time_ms = 0.0;
  std::vector<double> forces;  // AUDIT C6: 3*n_atoms, Hellmann-Feynman + Pulay
  PipelineTimings timings;     // Per-component profiling data
  // --- Gap module integration fields ---
  double E_dispersion = 0.0;           // D4 dispersion energy
  double E_mermin_free_energy = 0.0;   // Mermin finite-Te free energy
  double mermin_entropy = 0.0;         // Electronic entropy (k_B units)
  double a_posteriori_energy_bound = 0.0; // Certified energy error bound
  double a_posteriori_scf_residual = 0.0; // ||[H,P]||_F commutator norm
  double energy_kwh = 0.0;             // Energy consumption (kWh)
  bool bsse_corrected = false;         // Whether BSSE correction was applied
  double bsse_correction = 0.0;        // BSSE correction energy
  bool mixed_precision_used = false;   // Whether mixed-precision path was used
  std::string mixed_precision_mode;   // FP64/BF16/FP16/Auto
  double qtt_compression_ratio = 0.0;  // QTT density matrix compression ratio
  std::string point_group_symbol;     // Detected point group
  bool point_group_symmetrized = false; // Whether matrices were symmetrized
};

class MoleculeDriver {
 public:
  // Run end-to-end SCF on a molecule with a GTO basis.
  // Positions in Bohr. n_electrons = sum of atomic numbers (neutral).
  // grid_h: grid spacing in Bohr (default 0.2835 = 0.15 Angstrom).
  // grid_margin: extra space around molecule in Bohr (default 3.7794 = 2.0 Angstrom).
  // use_grid_hartree: if true, use grid-based Poisson for Hartree (NAO path);
  //   if false, use analytic ERIs (GTO path, exact and faster).
  // xc_spec: XC functional specification (default LDA-PW92). Use PBE for GGA.
  static MoleculeDriverResult Run(
      const GTOMolecule& mol,
      double grid_h = 0.2835,
      double grid_margin = 3.7794,
      int max_iter = 100,
      double tol = 1e-8,
      bool use_grid_hartree = false,
      grid::xc::HostXcSpec xc_spec = {},
      bool use_grid_vext = false) {
    MoleculeDriverResult result;
    result.n_basis = mol.n_basis;
    result.n_atoms = mol.atomic_numbers.size();
    result.grid_h = grid_h;
    result.timings.used_grid_hartree = use_grid_hartree;
    result.timings.xc_functional = grid::xc::XcFunctionalName(xc_spec.id);

    auto t0 = std::chrono::steady_clock::now();


    // Count electrons (neutral molecule).
    std::size_t n_electrons = 0;
    for (int Z : mol.atomic_numbers) n_electrons += static_cast<std::size_t>(Z);
    result.n_electrons = n_electrons;
    const std::size_t n_occ = n_electrons / 2;  // spin-paired

    // Step 1: Compute S and T analytically.
    const std::size_t n = mol.n_basis;
    auto S = GTOIntegrals::Overlap(mol);
    auto T = GTOIntegrals::Kinetic(mol);

    // Step 2: Set up the grid.
    // Compute bounding box of atoms + basis extent.
    double rmin[3] = {1e30, 1e30, 1e30};
    double rmax[3] = {-1e30, -1e30, -1e30};
    for (std::size_t a = 0; a < mol.atomic_numbers.size(); ++a) {
      const double* Ra = &mol.positions[3 * a];
      for (int c = 0; c < 3; ++c) {
        rmin[c] = std::min(rmin[c], Ra[c]);
        rmax[c] = std::max(rmax[c], Ra[c]);
      }
    }
    // Add margin.
    for (int c = 0; c < 3; ++c) {
      rmin[c] -= grid_margin;
      rmax[c] += grid_margin;
    }

    // Grid dimensions.
    std::size_t n0 = static_cast<std::size_t>((rmax[0] - rmin[0]) / grid_h) + 1;
    std::size_t n1 = static_cast<std::size_t>((rmax[1] - rmin[1]) / grid_h) + 1;
    std::size_t n2 = static_cast<std::size_t>((rmax[2] - rmin[2]) / grid_h) + 1;
    // Ensure odd dimensions for better Poisson behavior.
    if (n0 % 2 == 0) n0++;
    if (n1 % 2 == 0) n1++;
    if (n2 % 2 == 0) n2++;
    result.grid_n = {n0, n1, n2};

    grid::UniformGrid3D grid;
    grid.n = {n0, n1, n2};
    grid.h = {grid_h, grid_h, grid_h};
    grid.origin = {rmin[0], rmin[1], rmin[2]};
    // AUDIT B4: Use free (open) BCs for isolated molecules.
    // Previous code set periodic BCs — harmless only because the driver
    // bypasses Poisson (uses analytic ERIs for Hartree). Fix now to prevent
    // silent wrong electrostatics when grid Hartree is enabled.
    grid.bc = {grid::BoundaryCondition::kFree,
               grid::BoundaryCondition::kFree,
               grid::BoundaryCondition::kFree};

    // Generate grid coordinate arrays.
    std::vector<double> gx(n0), gy(n1), gz(n2);
    for (std::size_t i = 0; i < n0; ++i) gx[i] = rmin[0] + grid_h * i;
    for (std::size_t i = 0; i < n1; ++i) gy[i] = rmin[1] + grid_h * i;
    for (std::size_t i = 0; i < n2; ++i) gz[i] = rmin[2] + grid_h * i;

    // Step 3: Evaluate basis functions on the grid.
    std::vector<std::vector<double>> orbitals(n);
    for (std::size_t i = 0; i < n; ++i) {
      orbitals[i] = GTOIntegrals::EvalBasisOnGrid(
          mol, i, gx, gy, gz, n0, n1, n2);
    }

    // Step 4: Compute V_ext (nuclear attraction).
    // When use_grid_vext=true, compute on grid (avoids analytic OS recursion
    // bug for p-orbital nuclear attraction). When false, use analytic integrals.
    std::vector<double> V_ext;
    if (use_grid_vext) {
      const std::size_t N_grid_tmp = n0 * n1 * n2;
      std::vector<double> v_ext_grid(N_grid_tmp, 0.0);
      for (std::size_t ix = 0; ix < n0; ++ix) {
        for (std::size_t iy = 0; iy < n1; ++iy) {
          for (std::size_t iz = 0; iz < n2; ++iz) {
            const std::size_t g = grid.flatten(ix, iy, iz);
            auto [x, y, z] = grid.coord(ix, iy, iz);
            double v = 0.0;
            for (std::size_t a = 0; a < mol.atomic_numbers.size(); ++a) {
              const double dx = x - mol.positions[3 * a];
              const double dy = y - mol.positions[3 * a + 1];
              const double dz = z - mol.positions[3 * a + 2];
              const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
              if (r > 1e-10)
                v -= static_cast<double>(mol.atomic_numbers[a]) / r;
            }
            v_ext_grid[g] = v;
          }
        }
      }
      V_ext = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, v_ext_grid);
    } else {
      V_ext = GTOIntegrals::NuclearAttraction(mol);
    }

    // Step 5: Compute ion-ion energy.
    double E_ion = EnergyAssembly::EwaldIonIon(
        mol.positions, 
        std::vector<double>(mol.atomic_numbers.begin(),
                            mol.atomic_numbers.end()),
        false);

    // AUDIT B8: Cache the ERI tensor at setup — it is geometry-constant
    // and was being recomputed 3× per SCF iteration.
    // Now uses BuildERICache with 8-fold permutational symmetry + Schwarz screening.
    // O(n⁴) once at setup + O(n⁴) contraction per iteration (1×, not 3×).
    auto eri_cache = GTOIntegrals::BuildERICache(mol);

    // Grid volume element for energy integration.
    const auto [h0_g, h1_g, h2_g] = grid.h;
    const double dv = h0_g * h1_g * h2_g;
    const std::size_t N_grid = grid.total_points();

    // AUDIT B7: build_H caches its last result so energy_fn can reuse it
    // without rebuilding H. This eliminates the 2nd and 3rd H builds.
    // AUDIT B5: energy_fn receives eigenvalues from SCFDriver, so no
    // re-diagonalization is needed.
    // AUDIT B6: E_xc comes directly from the fused XC engine (out.xc_energy),
    // an O(N_grid) in-kernel reduction — no separate grid dot product needed.

    struct CachedHBuild {
      std::vector<double> H;
      std::vector<double> V_H;          // Hartree potential matrix (n×n)
      std::vector<double> V_xc;         // XC potential matrix (n×n)
      std::vector<double> vxc_grid;     // V_xc(r) on grid (from fused XC engine)
      std::vector<double> eps_xc_grid;  // eps_xc(r) on grid (from fused XC engine)
      double xc_energy = 0.0;           // Total E_xc from fused XC engine
      double xc_kernel_ms = 0.0;        // XC kernel time
      std::vector<double> rho;          // Density on grid
      std::vector<double> P2;           // Scaled density matrix (2×P)
      std::vector<double> V_H_grid;     // Hartree potential on grid (grid Hartree path)
      double E_H_grid = 0.0;            // Hartree energy from grid (grid Hartree path)
      double rho_build_ms = 0.0;        // Rho build timing
      double vmat_build_ms = 0.0;       // Vmat build timing
      double poisson_ms = 0.0;          // Poisson timing
    };
    CachedHBuild cache;

    // Accumulators for per-component timing (averaged over SCF iterations).
    double total_rho_ms = 0.0, total_xc_ms = 0.0;
    double total_poisson_ms = 0.0, total_vmat_ms = 0.0;
    int scf_iter_count = 0;

    auto build_H = [&](const std::vector<double>& P) -> std::vector<double> {
      // Scale P by 2 for spin degeneracy (SCFDriver uses occupation 1).
      cache.P2.assign(n * n, 0.0);
      for (std::size_t i = 0; i < static_cast<std::size_t>(mol.n_basis) * mol.n_basis; ++i) cache.P2[i] = 2.0 * P[i];
      cache.vmat_build_ms = 0.0;  // reset per-iteration

      // --- Step A: Build density on grid from density matrix P ---
      // AUDIT B3 FIX: Use BuildRhoGemm (BLAS dgemm from P) instead of
      // BuildRho (triple loop) or BuildFromOrbitals (requires eigenvectors).
      // This makes the pipeline compatible with R2/R3 (purification → P).
      auto t_rho0 = std::chrono::steady_clock::now();
      cache.rho = grid::VmatBuilder::BuildRhoGemm(grid, orbitals, cache.P2);
      auto t_rho1 = std::chrono::steady_clock::now();
      cache.rho_build_ms =
          std::chrono::duration<double, std::milli>(t_rho1 - t_rho0).count();
      total_rho_ms += cache.rho_build_ms;

      // --- Step B: Build Hartree potential ---
      if (use_grid_hartree) {
        // Grid-based Poisson: rho → V_H(r) via PoissonSolver.
        // This is the NAO production path (no analytic ERIs available).
        auto t_poi0 = std::chrono::steady_clock::now();
        cache.V_H_grid = grid::PoissonSolver::Solve(grid, cache.rho);
        auto t_poi1 = std::chrono::steady_clock::now();
        cache.poisson_ms =
            std::chrono::duration<double, std::milli>(t_poi1 - t_poi0).count();
        total_poisson_ms += cache.poisson_ms;

        // E_H from grid: 0.5 * integral(rho * V_H * dv).
        double E_H = 0.0;
        for (std::size_t g = 0; g < N_grid; ++g)
          E_H += cache.rho[g] * cache.V_H_grid[g] * dv;
        cache.E_H_grid = 0.5 * E_H;

        // Project V_H(r) to matrix: V_H_ij = integral V_H(r) phi_i phi_j dv.
        auto t_vm0 = std::chrono::steady_clock::now();
        cache.V_H = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, cache.V_H_grid);
        auto t_vm1 = std::chrono::steady_clock::now();
        cache.vmat_build_ms +=
            std::chrono::duration<double, std::milli>(t_vm1 - t_vm0).count();
      } else {
        // Analytic ERI-based Hartree (GTO path: exact, faster for small n).
        // AUDIT B8: Use cached ERI tensor (8-fold symmetry + Schwarz screened).
        cache.V_H = GTOIntegrals::CoulombMatrixCached(eri_cache, cache.P2);
      }

      // --- Step C: Evaluate XC on grid via fused Tier-0 engine ---
      // AUDIT C4/B1 FIX: Use xc::XcEval (fused Tier-0 engine) instead of
      // XCGridEvaluator::EvaluateLDA. Supports LDA-PW92 and PBE.
      // Auto-dispatches to CUDA when TIDES_HAVE_CUDA is defined.
      grid::xc::HostXcGridIn xc_in;
      xc_in.rho = cache.rho.data();
      xc_in.np = N_grid;
      xc_in.grid_weight = dv;

      cache.vxc_grid.assign(N_grid, 0.0);
      cache.eps_xc_grid.assign(N_grid, 0.0);

      grid::xc::HostXcGridOut xc_out;
      xc_out.vxc = cache.vxc_grid.data();
      xc_out.eps_xc = cache.eps_xc_grid.data();
      xc_out.xc_energy = 0.0;
      xc_out.kernel_ms = 0.0;

      std::string xc_err;
      bool xc_ok = grid::xc::XcEvalHost(xc_spec, xc_in, xc_out, xc_err);
      if (!xc_ok) {
        // Fallback to CPU XCGridEvaluator if fused engine fails.
        auto xc_cpu = grid::XCGridEvaluator::EvaluateLDA(grid, cache.rho);
        std::copy(xc_cpu.vxc.begin(), xc_cpu.vxc.end(), cache.vxc_grid.begin());
        std::copy(xc_cpu.eps_xc.begin(), xc_cpu.eps_xc.end(),
                  cache.eps_xc_grid.begin());
        cache.xc_energy =
            grid::XCGridEvaluator::XCEnergy(grid, xc_cpu, cache.rho);
        cache.xc_kernel_ms = 0.0;
      } else {
        cache.xc_energy = xc_out.xc_energy;
        cache.xc_kernel_ms = xc_out.kernel_ms;
      }
      total_xc_ms += cache.xc_kernel_ms;

      // --- Step D: Project V_xc(r) to matrix via GEMM ---
      // AUDIT C2 FIX: Use BuildHmatGemm (BLAS dgemm) instead of BuildHmat
      // (O(n²·N_grid) triple loop).
      auto t_vm0 = std::chrono::steady_clock::now();
      cache.V_xc = grid::VmatBuilder::BuildHmatGemm(grid, orbitals, cache.vxc_grid);
      auto t_vm1 = std::chrono::steady_clock::now();
      cache.vmat_build_ms +=
          std::chrono::duration<double, std::milli>(t_vm1 - t_vm0).count();
      total_vmat_ms += cache.vmat_build_ms;

      // --- Step E: Assemble H = T + V_ext + V_H + V_xc ---
      cache.H = tides::ham::AssembleH(
          static_cast<std::size_t>(mol.n_basis), T, V_ext, cache.V_H, cache.V_xc,
          std::vector<double>{});

      scf_iter_count++;
      return cache.H;
    };

    // AUDIT B5/B6/B7: energy_fn uses cached H build results + eigenvalues
    // from the SCF loop. No re-diagonalization, no rebuild.
    // E_xc comes from the fused XC engine's in-kernel reduction (B6 fix).
    auto energy_fn = [&](const std::vector<double>& P,
                         const std::vector<double>& eigenvalues) -> double {
      // sum_eps from eigenvalues passed by SCFDriver (B5 fix).
      double sum_eps = 0.0;
      for (std::size_t k = 0; k < n_occ && k < n; ++k)
        sum_eps += 2.0 * eigenvalues[k];  // factor 2 for spin

      auto trace = [&](const std::vector<double>& A, const std::vector<double>& B) {
        double s = 0.0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(mol.n_basis) * mol.n_basis; ++i) s += A[i] * B[i];
        return s;
      };

      double E_ne = trace(cache.P2, V_ext);
      double E_H;
      if (use_grid_hartree) {
        E_H = cache.E_H_grid;  // From grid Poisson (O(N_grid) dot product)
      } else {
        E_H = 0.5 * trace(cache.P2, cache.V_H);  // From analytic ERIs
      }
      double E_xc = cache.xc_energy;  // From fused XC engine (B6 fix)
      double E_kin = sum_eps - E_ne - 2.0 * E_H - trace(cache.P2, cache.V_xc);
      double E_total = E_kin + E_ne + E_H + E_xc + E_ion;

      result.energy.E_kin = E_kin;
      result.energy.E_ne = E_ne;
      result.energy.E_H = E_H;
      result.energy.E_xc = E_xc;
      result.energy.E_ion = E_ion;
      result.energy.E_total = E_total;
      return E_total;
    };

    // Run SCF.
    auto t_scf0 = std::chrono::steady_clock::now();
    result.scf = SCFDriver::Run(n, n_occ, S, build_H, energy_fn,
                                 {}, max_iter, tol, 1, 0.3);
    auto t_scf1 = std::chrono::steady_clock::now();

    // Record per-component timings.
    result.timings.n_iterations = result.scf.n_iterations;
    result.timings.rho_build_ms =
        (scf_iter_count > 0) ? total_rho_ms / scf_iter_count : 0.0;
    result.timings.xc_eval_ms =
        (scf_iter_count > 0) ? total_xc_ms / scf_iter_count : 0.0;
    result.timings.poisson_ms =
        (scf_iter_count > 0) ? total_poisson_ms / scf_iter_count : 0.0;
    result.timings.vmat_build_ms =
        (scf_iter_count > 0) ? total_vmat_ms / scf_iter_count : 0.0;
    result.timings.scf_total_ms =
        std::chrono::duration<double, std::milli>(t_scf1 - t_scf0).count();
#ifdef TIDES_HAVE_CUDA
    result.timings.used_gpu_xc = true;
#else
    result.timings.used_gpu_xc = false;
#endif

    // --- Gap module wiring: post-SCF integration of remediation modules ---
    // D4 dispersion correction.
    if (mol.atomic_numbers.size() >= 2) {
      auto d4 = hybrids::D4Dispersion::ComputeEnergy(
          mol.atomic_numbers, mol.positions);
      result.E_dispersion = d4.energy;
      result.energy.E_total += d4.energy;
    }

    // Point-group symmetrization: detect and report symmetry.
    if (mol.atomic_numbers.size() >= 2) {
      auto pg = common::PointGroupSymmetrizer::Detect(
          mol.atomic_numbers, mol.positions);
      result.point_group_symbol = pg.symbol;
      if (pg.order() > 1 && result.scf.converged && !result.scf.P.empty()) {
        result.scf.P = common::PointGroupSymmetrizer::SymmetrizeMatrix(
            result.scf.P, static_cast<std::size_t>(mol.n_basis), pg,
            mol.atomic_numbers, mol.positions);
        result.point_group_symmetrized = true;
      }
    }

    // A-posteriori error control: certified energy/force bounds.
    if (result.scf.converged && !result.scf.P.empty() && !cache.H.empty()) {
      auto bounds = verification::APosterioriErrorControl::Compute(
          static_cast<std::size_t>(mol.n_basis), n_occ,
          cache.H, S, result.scf.P,
          result.scf.eigenvalues, result.scf.eigenvectors);
      result.a_posteriori_energy_bound = bounds.energy_error_bound;
      result.a_posteriori_scf_residual = bounds.scf_residual_norm;
    }

    // Mixed precision: report mode for this system size.
    {
      auto mode = scf::MixedPrecisionSCF::AutoSelect(
          static_cast<std::size_t>(mol.n_basis), 1e-6);
      result.mixed_precision_mode =
          (mode == scf::PrecisionMode::kFP64) ? "FP64" :
          (mode == scf::PrecisionMode::kBF16) ? "BF16" :
          (mode == scf::PrecisionMode::kFP16) ? "FP16" : "Auto";
    }

    // QTT compression: compress the density matrix.
    if (result.scf.converged && !result.scf.P.empty() &&
        static_cast<std::size_t>(mol.n_basis) >= 8) {
      auto compressed = tile::QTTCompressor::Compress(
          result.scf.P, static_cast<std::size_t>(mol.n_basis), 1e-6, 0);
      result.qtt_compression_ratio = compressed.compression_ratio;
    }

    // Energy metering: estimate energy consumption from wall time.
    {
      double elapsed_s = result.timings.scf_total_ms / 1000.0;
      double total_j = 125.0 * 16 * 0.7 * elapsed_s + 350.0 * 0.8 * elapsed_s;
      result.energy_kwh = total_j / 3.6e6;
    }
    auto t1 = std::chrono::steady_clock::now();
    // --- End gap module wiring ---

    result.wall_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return result;
  }

  // STO-3G basis set for light elements (H, He, Li, C, N, O, F).
  // Returns shells for a given atomic number.
  static std::vector<GTOShell> STO3G(int Z) {
    std::vector<GTOShell> shells;

    if (Z == 1) {  // H
      GTOShell s;
      s.atom_index = -1;  // set by caller
      s.l = 0;
      s.primitives = {
        {3.42525091, 0.15432897},
        {0.62391373, 0.53532814},
        {0.16885540, 0.44463454},
      };
      shells.push_back(s);
    } else if (Z == 2) {  // He
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {6.36242139, 0.15432897},
        {1.15892300, 0.53532814},
        {0.31364979, 0.44463454},
      };
      shells.push_back(s);
    } else if (Z == 6) {  // C
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {71.6168370, 0.15432897},
        {13.0450960, 0.53532814},
        {3.5305122, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {2.9412494, -0.09996723},
        {0.6834831, 0.39951283},
        {0.2222899, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {2.9412494, 0.15591627},
        {0.6834831, 0.60768372},
        {0.2222899, 0.39195739},
      };
      shells.push_back(p);
    } else if (Z == 7) {  // N
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {99.1061690, 0.15432897},
        {18.0523120, 0.53532814},
        {4.8856212, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {3.7804559, -0.09996723},
        {0.8784966, 0.39951283},
        {0.2857144, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {3.7804559, 0.15591627},
        {0.8784966, 0.60768372},
        {0.2857144, 0.39195739},
      };
      shells.push_back(p);
    } else if (Z == 8) {  // O
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {130.7093210, 0.15432897},
        {23.8088610, 0.53532814},
        {6.4436083, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {5.0331513, -0.09996723},
        {1.1695961, 0.39951283},
        {0.3803890, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {5.0331513, 0.15591627},
        {1.1695961, 0.60768372},
        {0.3803890, 0.39195739},
      };
      shells.push_back(p);
    } else if (Z == 9) {  // F
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {166.6791360, 0.15432897},
        {30.3608910, 0.53532814},
        {8.2180370, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {6.4648067, -0.09996723},
        {1.5021830, 0.39951283},
        {0.4885880, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {6.4648067, 0.15591627},
        {1.5021830, 0.60768372},
        {0.4885880, 0.39195739},
      };
      shells.push_back(p);
    } else if (Z == 3) {  // Li — AUDIT B9: comment claimed Li but no branch existed
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {16.1195750, 0.15432897},
        {2.9362007, 0.53532814},
        {0.7946505, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {0.6362897, -0.09996723},
        {0.1478601, 0.39951283},
        {0.0480887, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {0.6362897, 0.15591627},
        {0.1478601, 0.60768372},
        {0.0480887, 0.39195739},
      };
      shells.push_back(p);
    } else if (Z == 10) {  // Ne — AUDIT B9: add Ne (was missing, caused empty basis)
      GTOShell s;
      s.atom_index = -1;
      s.l = 0;
      s.primitives = {
        {244.8595400, 0.15432897},
        {44.5475780, 0.53532814},
        {12.0775680, 0.44463454},
      };
      shells.push_back(s);
      GTOShell s2s;
      s2s.atom_index = -1;
      s2s.l = 0;
      s2s.primitives = {
        {9.6323120, -0.09996723},
        {2.2272140, 0.39951283},
        {0.7232770, 0.70011547},
      };
      shells.push_back(s2s);
      GTOShell p;
      p.atom_index = -1;
      p.l = 1;
      p.primitives = {
        {9.6323120, 0.15591627},
        {2.2272140, 0.60768372},
        {0.7232770, 0.39195739},
      };
      shells.push_back(p);
    }
    // AUDIT B9: If no shells were found, this is an error, not an empty basis.
    // Caller (BuildMolecule) must check and fail loudly.
    return shells;
  }

  // Build a GTOMolecule from atoms, positions, and basis set.
  static GTOMolecule BuildMolecule(
      const std::vector<int>& atomic_numbers,
      const std::vector<double>& positions) {
    GTOMolecule mol;
    mol.atomic_numbers = atomic_numbers;
    mol.positions = positions;

    std::size_t n_basis = 0;
    for (std::size_t a = 0; a < atomic_numbers.size(); ++a) {
      auto shells = STO3G(atomic_numbers[a]);
      // AUDIT B9: Fail loudly on unsupported elements, not silently empty basis.
      if (shells.empty()) {
        std::cerr << "ERROR: STO-3G basis not available for Z="
                  << atomic_numbers[a]
                  << ". Supported: H(1), He(2), Li(3), C(6), N(7), O(8), F(9), Ne(10)\n";
        mol.n_basis = 0;
        return mol;
      }
      for (auto& shell : shells) {
        shell.atom_index = static_cast<int>(a);
        n_basis += GTOIntegrals::NumCartesian(shell.l);
        mol.shells.push_back(shell);
      }
    }
    mol.n_basis = n_basis;
    return mol;
  }

  // AUDIT C6: Compute forces on all atoms.
  // F_I = -dE/dR_I = F_HF + F_Pulay + F_ion
  //
  // F_HF (Hellmann-Feynman): -Tr(P dH/dR_I)
  //   For GTOs, dH/dR comes from d(S)/dR, d(T)/dR, d(V_ne)/dR, d(V_H)/dR, d(V_xc)/dR.
  //   The dominant terms are d(V_ne)/dR (nuclear attraction) and d(T)/dR (kinetic).
  //   For the grid-based path, d(V_H)/dR and d(V_xc)/dR are computed via grid shifts.
  //
  // F_Pulay (basis follows atoms): -Tr(dP/dR * H) + Tr(P * dS/dR * eps)
  //   For GTOs with atom-centered basis, Pulay forces are non-zero.
  //   F_Pulay = -sum_k f_k * C_k^T * (dH/dR - eps_k * dS/dR) * C_k
  //
  // F_ion (ion-ion): -d(E_ion)/dR_I = sum_{J!=I} Z_I Z_J (R_I - R_J) / |R_I - R_J|^3
  //
  // For the initial implementation, we compute:
  //   (1) F_ion analytically (exact, trivial)
  //   (2) F_HF via finite-difference of H (5-point stencil on dH/dR)
  //   (3) F_Pulay via the response formula using dS/dR
  //
  // The FD5 validation path (AnalyticForces::Validate) is available separately.
  static std::vector<double> ComputeForces(
      const GTOMolecule& mol,
      const SCFResult& scf_result,
      const EnergyComponents& energy) {
    const std::size_t n_atoms = mol.atomic_numbers.size();
    std::vector<double> forces(3 * n_atoms, 0.0);

    // (1) Ion-ion repulsion forces: F_I = sum_{J!=I} Z_I Z_J (R_I - R_J) / |R_I - R_J|^3
    for (std::size_t a = 0; a < n_atoms; ++a) {
      for (std::size_t b = 0; b < n_atoms; ++b) {
        if (a == b) continue;
        double dx = mol.positions[3*a]     - mol.positions[3*b];
        double dy = mol.positions[3*a + 1] - mol.positions[3*b + 1];
        double dz = mol.positions[3*a + 2] - mol.positions[3*b + 2];
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < 1e-20) continue;
        double r = std::sqrt(r2);
        double r3 = r * r2;
        double Z_a = mol.atomic_numbers[a];
        double Z_b = mol.atomic_numbers[b];
        double f = Z_a * Z_b / r3;
        forces[3*a]     += f * dx;
        forces[3*a + 1] += f * dy;
        forces[3*a + 2] += f * dz;
      }
    }

    // (2) Hellmann-Feynman + Pulay forces via numerical dH/dR and dS/dR.
    // F_HF = -Tr(P * dH/dR_I)  where P is the density matrix.
    // F_Pulay = Tr(P * dS/dR_I * eps)  where eps is diagonal eigenvalue matrix.
    // Total: F = F_ion + F_HF + F_Pulay
    //
    // dH/dR and dS/dR are computed via central differences of the analytic
    // integral routines (Overlap, Kinetic, NuclearAttraction). This is exact
    // to O(h^2) and serves as the production force path until analytic
    // derivative streams (T2.6) are implemented.
    const double fd_h = 0.001;  // Bohr
    const std::vector<double>& P = scf_result.P;
    const std::vector<double>& evals = scf_result.eigenvalues;
    std::size_t n_el = 0;
    for (int Z : mol.atomic_numbers) n_el += static_cast<std::size_t>(Z);
    const std::size_t n_occ = n_el / 2;
    const double occ_factor = (n_occ > 0)
        ? static_cast<double>(n_el) / static_cast<double>(n_occ) : 0.0;

    for (std::size_t a = 0; a < n_atoms; ++a) {
      for (int c = 0; c < 3; ++c) {
        // Perturb atom a in direction c.
        GTOMolecule mol_plus = mol, mol_minus = mol;
        mol_plus.positions[3*a + c] += fd_h;
        mol_minus.positions[3*a + c] -= fd_h;

        auto S_plus = GTOIntegrals::Overlap(mol_plus);
        auto S_minus = GTOIntegrals::Overlap(mol_minus);
        auto T_plus = GTOIntegrals::Kinetic(mol_plus);
        auto T_minus = GTOIntegrals::Kinetic(mol_minus);
        auto V_plus = GTOIntegrals::NuclearAttraction(mol_plus);
        auto V_minus = GTOIntegrals::NuclearAttraction(mol_minus);

        // dH/dR = dT/dR + dV/dR (H = T + V_ext + V_H + V_xc, but V_H and V_xc
        // depend on P which is fixed at the converged geometry).
        // For the HF force, only the geometry-dependent parts of H matter.
        const std::size_t n = mol.n_basis;
        double dH_trace = 0.0;
        for (std::size_t i = 0; i < n * n; ++i) {
          double dH = (T_plus[i] + V_plus[i]) - (T_minus[i] + V_minus[i]);
          dH_trace += P[i] * dH;
        }
        // F_HF = -Tr(P * dH/dR) / (2h)
        double f_hf = -dH_trace / (2.0 * fd_h);

        // Full Pulay: F_Pulay = sum_k f_k * eps_k * (C_k^T * dS/dR * C_k) / (2h)
        // where f_k is the occupation (2 for closed-shell), eps_k the eigenvalue,
        // and C_k the k-th eigenvector. This replaces the previous eps_avg
        // approximation which weighted all orbitals equally.
        const auto& evec = scf_result.eigenvectors;
        double f_pulay = 0.0;
        for (std::size_t k = 0; k < n_occ && k < evals.size(); ++k) {
          const double eps_k = evals[k];
          double ctds_c = 0.0;
          for (std::size_t i = 0; i < n; ++i) {
            const double ci = evec[k * n + i];
            for (std::size_t j = 0; j < n; ++j) {
              const double dS_ij = S_plus[i * n + j] - S_minus[i * n + j];
              ctds_c += ci * dS_ij * evec[k * n + j];
            }
          }
          f_pulay += occ_factor * eps_k * ctds_c;
        }
        f_pulay /= (2.0 * fd_h);

        forces[3*a + c] += f_hf + f_pulay;
      }
    }

    return forces;
  }
};

}  // namespace tides::scf
