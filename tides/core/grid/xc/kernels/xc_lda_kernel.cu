// T-X4.1: LDA kernel launch file.
//
// Instantiates the generic FunctionalKernel for the unpolarized Tier-0 LDA
// functors and exposes the per-family LaunchLda* functions.

#include "grid/xc/kernels/reduce.cuh"
#include "grid/xc/functional_dispatch.hpp"

namespace tides::grid::xc {

using kernels::LaunchFunctionalKernel;
using kernels::LaunchFunctionalKernelFp32;
using kernels::LaunchStressKernel;

Status LaunchLdaFunctional(const XcSpec& spec, const XcGridIn& input,
                           XcGridOut& output, cudaStream_t stream) {
  switch (spec.terms[0].functional) {
    case Functional::kLdaPw92:
      return LaunchFunctionalKernel<LdaPw92Functor>(input, output, stream, spec.deterministic);
    case Functional::kSvwn5:
      return LaunchFunctionalKernel<Svwn5Functor>(input, output, stream, spec.deterministic);
    default:
      return Status::Unimplemented(
          "LDA functional not implemented in xc_lda_kernel.cu");
  }
}

Status LaunchLdaFunctionalFp32(const XcSpec& spec, const XcGridInFp32& input,
                               XcGridOutFp32& output, cudaStream_t stream) {
  switch (spec.terms[0].functional) {
    case Functional::kLdaPw92:
      return LaunchFunctionalKernelFp32<LdaPw92Functor>(input, output, stream, spec.deterministic);
    case Functional::kSvwn5:
      return LaunchFunctionalKernelFp32<Svwn5Functor>(input, output, stream, spec.deterministic);
    default:
      return Status::Unimplemented(
          "FP32 LDA functional not implemented in xc_lda_kernel.cu");
  }
}

Status LaunchLdaStress(const XcSpec& spec, const XcGridIn& input,
                       XcStressOut& stress_out, cudaStream_t stream) {
  switch (spec.terms[0].functional) {
    case Functional::kLdaPw92:
      return LaunchStressKernel<LdaPw92Functor>(spec, input, stress_out, stream);
    case Functional::kSvwn5:
      return LaunchStressKernel<Svwn5Functor>(spec, input, stress_out, stream);
    default:
      return Status::Unimplemented(
          "LDA stress tensor not implemented in xc_lda_kernel.cu");
  }
}

}  // namespace tides::grid::xc
