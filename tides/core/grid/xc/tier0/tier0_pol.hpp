#pragma once

#include "grid/xc/xc_engine.hpp"

namespace tides::grid::xc::tier0 {

// Launches the polarized (nspin=2) Tier-0 LDA/GGA kernel for the given
// functional.  Dispatches to the correct PolFunctor based on the Functional
// enum.  XcEval owns contract validation before this call.
[[nodiscard]] Status LaunchTier0Pol(const XcSpec& spec,
                                    const XcGridIn& input,
                                    XcGridOut& output,
                                    cudaStream_t stream);

}  // namespace tides::grid::xc::tier0
