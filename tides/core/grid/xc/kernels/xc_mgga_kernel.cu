// T-X4.1: mGGA and range-separated-hybrid kernel launch file.
//
// Tier-1 functionals (mGGA and RSH) are delegated to the libxc device adapters
// in tier1/*.cu.  This file is the dispatch boundary for those families.

#include "grid/xc/kernels/reduce.cuh"
#include "grid/xc/tier1/mgga_tpss.hpp"
#include "grid/xc/tier1/mgga_scan.hpp"
#include "grid/xc/tier1/mgga_r2scan.hpp"
#include "grid/xc/tier1/mgga_m06_2x.hpp"
#include "grid/xc/tier1/rsh_hse06.hpp"
#include "grid/xc/tier1/rsh_wb97x.hpp"

namespace tides::grid::xc {

Status LaunchMggaFunctional(const XcSpec& spec, const XcGridIn& input,
                            XcGridOut& output, cudaStream_t stream) {
  switch (spec.terms[0].functional) {
    case Functional::kTpss:
      return tier1::LaunchMggaTpss(input, output, stream, spec.deterministic, spec.nspin);
    case Functional::kScan:
      return tier1::LaunchMggaScan(input, output, stream, spec.deterministic, spec.nspin);
    case Functional::kR2scan:
      return tier1::LaunchMggaR2scan(input, output, stream, spec.deterministic, spec.nspin);
    case Functional::kM06_2x:
      return tier1::LaunchMggaM06_2x(input, output, stream, spec.deterministic, spec.nspin);
    default:
      return Status::Unimplemented(
          "mGGA functional not implemented in xc_mgga_kernel.cu");
  }
}

Status LaunchRshFunctional(const XcSpec& spec, const XcGridIn& input,
                           XcGridOut& output, cudaStream_t stream) {
  switch (spec.terms[0].functional) {
    case Functional::kHse06:
      return tier1::LaunchRshHse06(input, output, stream, spec.deterministic, spec.nspin);
    case Functional::kWb97x:
      return tier1::LaunchRshWb97x(input, output, stream, spec.deterministic, spec.nspin);
    default:
      return Status::Unimplemented(
          "RSH functional not implemented in xc_mgga_kernel.cu");
  }
}

}  // namespace tides::grid::xc
