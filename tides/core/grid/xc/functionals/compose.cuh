#pragma once

#include "grid/xc/functionals/common.cuh"

namespace tides::grid::xc {

// Weighted combine for GgaEvaluation objects.  Used by the hybrid and
// composite functionals (BLYP, B3LYP, PBE0, etc.).
inline constexpr GgaEvaluation operator*(GgaEvaluation eval, double scale) {
  return {eval.eps * scale, eval.vrho * scale, eval.vsigma * scale};
}
inline constexpr GgaEvaluation operator*(double scale, GgaEvaluation eval) {
  return eval * scale;
}
inline constexpr GgaEvaluation operator+(GgaEvaluation lhs, GgaEvaluation rhs) {
  return {lhs.eps + rhs.eps, lhs.vrho + rhs.vrho, lhs.vsigma + rhs.vsigma};
}

}  // namespace tides::grid::xc
