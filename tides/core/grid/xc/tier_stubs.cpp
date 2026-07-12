// Stub implementations for tier0/tier1 XC functions when libxc maple2c
// submodule is not available. These return Unimplemented status.
#include "common/status.hpp"
#include "grid/xc/xc_engine.hpp"
#include <cuda_runtime.h>

namespace tides::grid::xc::tier0 {
[[nodiscard]] Status LaunchTier0Pol(const XcSpec& spec,
                                    const XcGridIn& input,
                                    XcGridOut& output,
                                    cudaStream_t stream) {
  return Status::Unimplemented("tier0_pol not available: libxc maple2c submodule not initialized");
}
}  // namespace tides::grid::xc::tier0

namespace tides::grid::xc::tier1 {
Status LaunchMggaTpss(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("mgga_tpss not available");
}
Status LaunchMggaScan(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("mgga_scan not available");
}
Status LaunchMggaR2scan(const XcGridIn& input, XcGridOut& output,
                        cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("mgga_r2scan not available");
}
Status LaunchMggaM06_2x(const XcGridIn& input, XcGridOut& output,
                        cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("mgga_m06_2x not available");
}
Status LaunchRshHse06(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("rsh_hse06 not available");
}
Status LaunchRshWb97x(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("rsh_wb97x not available");
}
}  // namespace tides::grid::xc::tier1
