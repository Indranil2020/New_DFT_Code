#pragma once

#include "grid/xc/xc_engine.hpp"

namespace tides::grid::xc {

// Enqueues the unpolarized FP64 PBE fast reduction.  XcEval owns contract
// validation and zeroing exc_per_system before this launch.
[[nodiscard]] Status LaunchPbeGgaKernel(const XcGridIn& input,
                                        XcGridOut& output,
                                        cudaStream_t stream,
                                        bool deterministic);

}  // namespace tides::grid::xc
