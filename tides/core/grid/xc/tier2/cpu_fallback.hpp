#pragma once

// T-X4.3: Tier-2 CPU libxc fallback for exotic or mixed XC functionals.
//
// This is a synchronous host fallback: it copies device grid quantities to the
// host, evaluates the requested functional(s) with libxc, and copies the
// weighted potentials back to the device.  It is intentionally scoped to
// nspin=1 / nsys=1 / FP64; anything else remains unimplemented.

#include "grid/xc/xc_engine.hpp"

namespace tides::grid::xc::tier2 {

[[nodiscard]] Status LaunchCpuFallback(const XcSpec& spec, const XcGridIn& input,
                                       XcGridOut& output, cudaStream_t stream);

}  // namespace tides::grid::xc::tier2
