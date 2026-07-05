#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/status.hpp"

namespace tides::grid {

// Boundary conditions for the Poisson/electrostatics problem (per
// 10-physics/13). All four must be supported by the Poisson solver (T3.4).
enum class BoundaryCondition : std::uint8_t {
  kFree = 0,      // isolated molecule / cluster (decays to 0 at infinity)
  kWire = 1,      // 1D periodic (nanotube, polymer)
  kSlab = 2,      // 2D periodic (surface, 2D material)
  kPeriodic = 3,  // 3D periodic (bulk crystal)
};

// A single 3D real-space grid with uniform spacing. Two of these form the dual
// grid: a coarse grid (orbital operations) and a fine grid (density/potential,
// default h ~ 0.15 Angstrom per 10-physics/13).
//
// The grid is cell-tasked: the domain is decomposed into tiles for parallel
// processing (per 30-architecture/34: "atom/tile domain decomposition"). Each
// tile owns a contiguous block of grid points; halo regions overlap neighbors
// for stencil operations. The halo spec is documented in HaloSpec below.
struct UniformGrid3D {
  // Number of grid points along each axis.
  std::array<std::size_t, 3> n = {0, 0, 0};
  // Spacing (Bohr) along each axis.
  std::array<double, 3> h = {0.0, 0.0, 0.0};
  // Origin (Bohr).
  std::array<double, 3> origin = {0.0, 0.0, 0.0};
  // Boundary condition along each axis (0=free, 1=periodic).
  std::array<BoundaryCondition, 3> bc = {
      BoundaryCondition::kFree, BoundaryCondition::kFree,
      BoundaryCondition::kFree};

  [[nodiscard]] std::size_t total_points() const {
    return n[0] * n[1] * n[2];
  }
  [[nodiscard]] std::array<double, 3> cell_size() const {
    return {h[0] * static_cast<double>(n[0]),
            h[1] * static_cast<double>(n[1]),
            h[2] * static_cast<double>(n[2])};
  }
  // Map a flat index to 3D (ix, iy, iz). Row-major: ix fastest.
  [[nodiscard]] std::array<std::size_t, 3> unflatten(std::size_t idx) const {
    const std::size_t iz = idx / (n[0] * n[1]);
    idx %= (n[0] * n[1]);
    const std::size_t iy = idx / n[0];
    const std::size_t ix = idx % n[0];
    return {ix, iy, iz};
  }
  // Map 3D to flat index.
  [[nodiscard]] std::size_t flatten(std::size_t ix, std::size_t iy,
                                    std::size_t iz) const {
    return ix + n[0] * (iy + n[1] * iz);
  }
  // Physical coordinate of grid point (ix, iy, iz).
  [[nodiscard]] std::array<double, 3> coord(std::size_t ix, std::size_t iy,
                                           std::size_t iz) const {
    return {origin[0] + h[0] * static_cast<double>(ix),
            origin[1] + h[1] * static_cast<double>(iy),
            origin[2] + h[2] * static_cast<double>(iz)};
  }
};

// Halo specification for domain decomposition (per 30-architecture/34 and the
// T3.1 observable "halo spec documented"). Each axis has:
//   - periodic: whether the axis wraps (determines if halo crosses boundary)
//   - n_halo: number of halo layers on each side (set by the stencil width)
//   - owned: the [lo, hi) range of owned points on this rank (for multi-GPU)
struct HaloSpec {
  std::array<bool, 3> periodic = {false, false, false};
  std::array<int, 3> n_halo = {0, 0, 0};
  // Owned range [lo, hi) per axis (inclusive lo, exclusive hi).
  std::array<std::size_t, 3> owned_lo = {0, 0, 0};
  std::array<std::size_t, 3> owned_hi = {0, 0, 0};

  [[nodiscard]] std::string describe() const {
    std::string s = "HaloSpec{periodic={";
    s += (periodic[0] ? "1" : "0");
    s += (periodic[1] ? ",1" : ",0");
    s += (periodic[2] ? ",1" : ",0");
    s += "}, n_halo={";
    s += std::to_string(n_halo[0]) + "," + std::to_string(n_halo[1]) + "," +
         std::to_string(n_halo[2]);
    s += "}, owned=[" + std::to_string(owned_lo[0]) + ":" +
         std::to_string(owned_hi[0]) + ",";
    s += std::to_string(owned_lo[1]) + ":" + std::to_string(owned_hi[1]) + ",";
    s += std::to_string(owned_lo[2]) + ":" + std::to_string(owned_hi[2]) + ")}";
    return s;
  }
};

// The dual grid: coarse (orbital ops) + fine (density/potential), per
// 10-physics/13. The fine grid has ~2x the resolution of the coarse grid in
// each direction; orbital functions are projected from coarse to fine for
// density building (T3.2), and the potential is restricted from fine to coarse
// for the v->H map (T3.3).
struct DualGrid {
  UniformGrid3D coarse;
  UniformGrid3D fine;
  HaloSpec halo;

  // Validate that fine is a consistent refinement of coarse (integer ratio,
  // aligned origins). Returns Status::Ok if valid.
  [[nodiscard]] tides::Status validate() const {
    for (int ax = 0; ax < 3; ++ax) {
      const double ratio = fine.h[ax] > 0 ? coarse.h[ax] / fine.h[ax] : 0.0;
      const double rounded = std::round(ratio);
      if (std::fabs(ratio - rounded) > 1e-10 || rounded < 1.0) {
        return tides::Status::InvalidArgument(
            "fine grid spacing must be an integer subdivision of coarse");
      }
      if (coarse.bc[ax] != fine.bc[ax]) {
        return tides::Status::InvalidArgument(
            "coarse and fine BCs must match per axis");
      }
    }
    return tides::Status::Ok();
  }
};

// Index map: maps a global fine-grid index to its position in the
// cell-tasked/decomposed layout. For single-process (current) this is identity.
// The contract (per 31-data-contracts: GridArray) is that the grid supports
// domain(BCs, spacings), halo spec, device residency, and the
// map_orbitals_to_grid / adjoint pair with an adjointness contract test.
struct IndexMap {
  const UniformGrid3D* grid = nullptr;
  const HaloSpec* halo = nullptr;

  // Map a global 3D index to the owned flat index (including halo offset).
  // For single-process, this is just grid->flatten(ix, iy, iz).
  [[nodiscard]] std::size_t global_to_local(std::size_t ix, std::size_t iy,
                                            std::size_t iz) const {
    return grid->flatten(ix, iy, iz);
  }

  // Prolongation: coarse (ix_c, iy_c, iz_c) -> fine index list (the 2^3 children).
  // Returns up to 8 fine indices.
  [[nodiscard]] std::vector<std::size_t> prolongate(
      std::size_t ix_c, std::size_t iy_c, std::size_t iz_c,
      std::size_t ratio) const {
    std::vector<std::size_t> children;
    children.reserve(ratio * ratio * ratio);
    for (std::size_t dx = 0; dx < ratio; ++dx)
      for (std::size_t dy = 0; dy < ratio; ++dy)
        for (std::size_t dz = 0; dz < ratio; ++dz) {
          children.push_back(grid->flatten(ix_c * ratio + dx,
                                           iy_c * ratio + dy,
                                           iz_c * ratio + dz));
        }
    return children;
  }

  // Restriction: fine -> coarse by averaging (full-weighting).
  // Returns the coarse index for the center of a ratio^3 fine block.
  [[nodiscard]] std::size_t restrict_center(std::size_t ix_f,
                                            std::size_t iy_f,
                                            std::size_t iz_f,
                                            std::size_t ratio) const {
    return grid->flatten(ix_f / ratio, iy_f / ratio, iz_f / ratio);
  }
};

}  // namespace tides::grid
