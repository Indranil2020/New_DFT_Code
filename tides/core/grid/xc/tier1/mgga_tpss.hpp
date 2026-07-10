#pragma once

#include "grid/xc/xc_engine.hpp"

namespace tides::grid::xc::tier1 {

Status LaunchMggaTpss(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin = 1);

}  // namespace tides::grid::xc::tier1
