#pragma once

#include "grid/xc/xc_engine.hpp"

namespace tides::grid::xc {

// Enqueues the unpolarized FP64 evaluation for the given Tier-0 functional.
// XcEval owns contract validation and zeroing exc_per_system before this call.
[[nodiscard]] Status LaunchXcFunctional(const XcSpec& spec,
                                        const XcGridIn& input,
                                        XcGridOut& output,
                                        cudaStream_t stream);

}  // namespace tides::grid::xc
