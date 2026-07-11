// T-X4.1: GGA kernel launch file and XC functional dispatcher.
//
// xc_gga_kernel.cu is the dispatch hub for the family-specific launch files
// (xc_lda_kernel.cu, xc_mgga_kernel.cu) and hosts the GGA-specific kernel
// instantiations and the FP32/stress dispatchers.

#include "grid/xc/kernels/reduce.cuh"
#include "grid/xc/functional_dispatch.hpp"
#include "grid/xc/tier0/tier0_pol.hpp"

namespace tides::grid::xc {

using kernels::LaunchFunctionalKernel;
using kernels::LaunchFunctionalKernelFp32;
using kernels::LaunchStressKernel;

Status LaunchGgaFunctional(const XcSpec& spec, const XcGridIn& input,
                           XcGridOut& output, cudaStream_t stream) {
  switch (spec.terms[0].functional) {
    case Functional::kPbe:
      return LaunchFunctionalKernel<PbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kPbeSol:
      return LaunchFunctionalKernel<PbeSolFunctor>(input, output, stream, spec.deterministic);
    case Functional::kRevPbe:
      return LaunchFunctionalKernel<RevPbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kRpbe:
      return LaunchFunctionalKernel<RpbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kBlyp:
      return LaunchFunctionalKernel<BlypFunctor>(input, output, stream, spec.deterministic);
    case Functional::kB3lyp:
      return LaunchFunctionalKernel<B3lypFunctor>(input, output, stream, spec.deterministic);
    case Functional::kPbe0:
      return LaunchFunctionalKernel<Pbe0Functor>(input, output, stream, spec.deterministic);
    default:
      return Status::Unimplemented(
          "GGA functional not implemented in xc_gga_kernel.cu");
  }
}

Status LaunchGgaFunctionalFp32(const XcSpec& spec, const XcGridInFp32& input,
                               XcGridOutFp32& output, cudaStream_t stream) {
  switch (spec.terms[0].functional) {
    case Functional::kPbe:
      return LaunchFunctionalKernelFp32<PbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kPbeSol:
      return LaunchFunctionalKernelFp32<PbeSolFunctor>(input, output, stream, spec.deterministic);
    case Functional::kRevPbe:
      return LaunchFunctionalKernelFp32<RevPbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kRpbe:
      return LaunchFunctionalKernelFp32<RpbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kBlyp:
      return LaunchFunctionalKernelFp32<BlypFunctor>(input, output, stream, spec.deterministic);
    case Functional::kB3lyp:
      return LaunchFunctionalKernelFp32<B3lypFunctor>(input, output, stream, spec.deterministic);
    case Functional::kPbe0:
      return LaunchFunctionalKernelFp32<Pbe0Functor>(input, output, stream, spec.deterministic);
    default:
      return Status::Unimplemented(
          "FP32 GGA functional not implemented in xc_gga_kernel.cu");
  }
}

Status LaunchGgaStress(const XcSpec& spec, const XcGridIn& input,
                       XcStressOut& stress_out, cudaStream_t stream) {
  switch (spec.terms[0].functional) {
    case Functional::kPbe:
      return LaunchStressKernel<PbeFunctor>(spec, input, stress_out, stream);
    case Functional::kPbeSol:
      return LaunchStressKernel<PbeSolFunctor>(spec, input, stress_out, stream);
    case Functional::kRevPbe:
      return LaunchStressKernel<RevPbeFunctor>(spec, input, stress_out, stream);
    case Functional::kRpbe:
      return LaunchStressKernel<RpbeFunctor>(spec, input, stress_out, stream);
    case Functional::kBlyp:
      return LaunchStressKernel<BlypFunctor>(spec, input, stress_out, stream);
    default:
      return Status::Unimplemented(
          "GGA stress tensor not implemented in xc_gga_kernel.cu");
  }
}

// The public dispatchers declared in xc_gga_kernel.hpp.

Status LaunchXcFunctional(const XcSpec& spec, const XcGridIn& input,
                          XcGridOut& output, cudaStream_t stream) {
  if (spec.nspin == 2) {
    switch (spec.terms[0].functional) {
      case Functional::kLdaPw92:
      case Functional::kSvwn5:
      case Functional::kPbe:
      case Functional::kPbeSol:
      case Functional::kRevPbe:
      case Functional::kRpbe:
      case Functional::kBlyp:
      case Functional::kB3lyp:
      case Functional::kPbe0:
        return tier0::LaunchTier0Pol(spec, input, output, stream);
      default:
        break;
    }
  }
  switch (spec.terms[0].functional) {
    case Functional::kLdaPw92:
    case Functional::kSvwn5:
      return LaunchLdaFunctional(spec, input, output, stream);
    case Functional::kPbe:
    case Functional::kPbeSol:
    case Functional::kRevPbe:
    case Functional::kRpbe:
    case Functional::kBlyp:
    case Functional::kB3lyp:
    case Functional::kPbe0:
      return LaunchGgaFunctional(spec, input, output, stream);
    case Functional::kTpss:
    case Functional::kScan:
    case Functional::kR2scan:
    case Functional::kM06_2x:
      return LaunchMggaFunctional(spec, input, output, stream);
    case Functional::kHse06:
    case Functional::kWb97x:
      return LaunchRshFunctional(spec, input, output, stream);
    default:
      return Status::Unimplemented("Functional not yet implemented in Tier-0");
  }
}

Status LaunchXcFunctionalFp32(const XcSpec& spec, const XcGridInFp32& input,
                              XcGridOutFp32& output, cudaStream_t stream) {
  switch (spec.terms[0].functional) {
    case Functional::kLdaPw92:
    case Functional::kSvwn5:
      return LaunchLdaFunctionalFp32(spec, input, output, stream);
    case Functional::kPbe:
    case Functional::kPbeSol:
    case Functional::kRevPbe:
    case Functional::kRpbe:
    case Functional::kBlyp:
    case Functional::kB3lyp:
    case Functional::kPbe0:
      return LaunchGgaFunctionalFp32(spec, input, output, stream);
    default:
      return Status::Unimplemented(
          "FP32 mid-SCF path supports Tier-0 LDA/GGA only. "
          "mGGA/RSH functionals require FP64 (SCAN alpha, erfc hazards).");
  }
}

Status LaunchXcStress(const XcSpec& spec, const XcGridIn& input,
                      XcStressOut& stress_out, cudaStream_t stream) {
  if (spec.nspin != 1) {
    return Status::Unimplemented("Stress tensor supports unpolarized only.");
  }
  switch (spec.terms[0].functional) {
    case Functional::kLdaPw92:
    case Functional::kSvwn5:
      return LaunchLdaStress(spec, input, stress_out, stream);
    case Functional::kPbe:
    case Functional::kPbeSol:
    case Functional::kRevPbe:
    case Functional::kRpbe:
    case Functional::kBlyp:
      return LaunchGgaStress(spec, input, stress_out, stream);
    default:
      return Status::Unimplemented(
          "Stress tensor not yet implemented for this functional.");
  }
}

}  // namespace tides::grid::xc
