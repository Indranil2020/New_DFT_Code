// T3.1: Dual-grid layout + decomposition structs + index-map + halo spec.
// Observables: index-map unit tests; halo spec documented.

#include "grid/dual_grid.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using tides::grid::BoundaryCondition;
using tides::grid::DualGrid;
using tides::grid::HaloSpec;
using tides::grid::IndexMap;
using tides::grid::UniformGrid3D;

int Fail(const std::string& msg) {
  std::cerr << "dual_grid_tests: " << msg << '\n';
  return 1;
}

int CheckFlattenUnflatten() {
  UniformGrid3D g;
  g.n = {4, 3, 2};
  g.h = {0.5, 0.5, 0.5};
  // Test round-trip: flatten -> unflatten -> flatten
  for (std::size_t iz = 0; iz < 2; ++iz)
    for (std::size_t iy = 0; iy < 3; ++iy)
      for (std::size_t ix = 0; ix < 4; ++ix) {
        const std::size_t idx = g.flatten(ix, iy, iz);
        const auto [ix2, iy2, iz2] = g.unflatten(idx);
        if (ix != ix2 || iy != iy2 || iz != iz2) {
          std::ostringstream os;
          os << "flatten/unflatten mismatch at (" << ix << "," << iy << "," << iz
             << ")";
          return Fail(os.str());
        }
      }
  // Total points
  if (g.total_points() != 24) return Fail("total_points wrong");
  std::cout << "flatten/unflatten round-trip: OK (4x3x2 = 24 pts)\n";
  return 0;
}

int CheckCoord() {
  UniformGrid3D g;
  g.n = {5, 5, 5};
  g.h = {0.2, 0.2, 0.2};
  g.origin = {-0.5, -0.5, -0.5};
  auto [x, y, z] = g.coord(0, 0, 0);
  if (std::fabs(x + 0.5) > 1e-15 || std::fabs(y + 0.5) > 1e-15 ||
      std::fabs(z + 0.5) > 1e-15)
    return Fail("coord(0,0,0) wrong");
  auto [x2, y2, z2] = g.coord(4, 4, 4);
  if (std::fabs(x2 - 0.3) > 1e-15) return Fail("coord(4,...) wrong");
  std::cout << "coord mapping: OK\n";
  return 0;
}

int CheckDualGridValidate() {
  DualGrid dg;
  dg.coarse.n = {16, 16, 16};
  dg.coarse.h = {0.4, 0.4, 0.4};
  dg.fine.n = {32, 32, 32};
  dg.fine.h = {0.2, 0.2, 0.2};
  dg.coarse.bc = {BoundaryCondition::kFree, BoundaryCondition::kFree,
                  BoundaryCondition::kFree};
  dg.fine.bc = dg.coarse.bc;
  auto s = dg.validate();
  if (!s.ok()) return Fail("dual grid validation failed: " + s.message());
  std::cout << "dual grid validation (2:1 refinement): OK\n";

  // Invalid: non-integer ratio
  dg.fine.h = {0.15, 0.15, 0.15};
  s = dg.validate();
  if (s.ok()) return Fail("should reject non-integer refinement ratio");
  std::cout << "non-integer refinement rejected: OK\n";
  return 0;
}

int CheckHaloSpec() {
  HaloSpec halo;
  halo.periodic = {false, false, true};
  halo.n_halo = {2, 2, 2};
  halo.owned_lo = {0, 0, 0};
  halo.owned_hi = {16, 16, 16};
  std::string desc = halo.describe();
  std::cout << "halo spec: " << desc << '\n';
  if (desc.find("periodic={0,0,1}") == std::string::npos)
    return Fail("halo spec describe() wrong");
  return 0;
}

int CheckProlongationRestriction() {
  UniformGrid3D coarse;
  coarse.n = {4, 4, 4};
  IndexMap im{&coarse, nullptr};
  // Prolongate coarse (1,1,1) with ratio 2 -> 8 fine children.
  auto children = im.prolongate(1, 1, 1, 2);
  if (children.size() != 8) return Fail("prolongate should return 8 children");
  // Restrict fine (3,3,3) with ratio 2 -> coarse (1,1,1).
  auto c = im.restrict_center(3, 3, 3, 2);
  if (c != im.restrict_center(2, 2, 2, 2))
    return Fail("restrict_center inconsistency");
  std::cout << "prolongation/restriction: OK (8 children, center consistent)\n";
  return 0;
}

}  // namespace

int main() {
  if (CheckFlattenUnflatten()) return 1;
  if (CheckCoord()) return 1;
  if (CheckDualGridValidate()) return 1;
  if (CheckHaloSpec()) return 1;
  if (CheckProlongationRestriction()) return 1;
  std::cout << "\ndual_grid_tests: ALL GREEN\n";
  return 0;
}
