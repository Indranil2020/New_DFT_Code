#pragma once

#include "grid/xc/xc_engine.hpp"
#include "common/status.hpp"

#include <cuda_runtime.h>

namespace tides::grid::xc::tier1 {

Status LaunchMggaM06_2x(const XcGridIn& input, XcGridOut& output,
                        cudaStream_t stream, bool deterministic, int nspin = 1);

}  // namespace tides::grid::xc::tier1
