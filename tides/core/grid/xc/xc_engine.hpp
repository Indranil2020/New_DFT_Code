#pragma once

// The only Tier-0 XC entry point. All pointers in the grid views are device
// pointers in 256-byte-aligned structure-of-arrays storage.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "common/status.hpp"
#include "grid/xc/functionals/common.cuh"

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

namespace tides::grid::xc {

enum class Functional : std::uint16_t {
  kLdaPw92,
  kSvwn5,
  kPbe,
  kPbeSol,
  kRevPbe,
  kRpbe,
  kBlyp,
  kB3lyp,
  kPbe0,
  kHse06,
  kTpss,
  kR2scan,
  kScan,
  kWb97x,
  kM06_2x
};
enum class PrecisionPolicy : std::uint8_t { kFloat64, kFloat32MidScf };

struct XcTerm {
  Functional functional = Functional::kLdaPw92;
  double coefficient = 1.0;
};

struct XcSpec {
  Family family = Family::kLda;
  int nspin = 1;
  std::vector<XcTerm> terms;
  PrecisionPolicy precision = PrecisionPolicy::kFloat64;
  bool deterministic = false;
};

struct XcGridIn {
  // All SoA planes use point_stride, not np.  np is the logical count and
  // point_stride is the padded physical stride owned by XcArena or the grid
  // producer; this keeps producer/consumer plane offsets unambiguous.
  const double* rho = nullptr;          // [nspin][point_stride]
  const double* grad = nullptr;         // [nspin][3][point_stride], GGA+
  const double* tau = nullptr;          // [nspin][point_stride], mGGA
  const double* w = nullptr;            // [np]
  std::int64_t np = 0;
  std::int64_t point_stride = 0;
  int nsys = 1;
  const std::int64_t* sys_offsets = nullptr;  // [nsys + 1], optional for nsys=1
};

struct XcGridOut {
  double* wv_rho = nullptr;             // [nspin][input.point_stride]
  double* wv_grad = nullptr;             // [nspin][3][input.point_stride], GGA+
  double* wv_tau = nullptr;             // [nspin][input.point_stride], mGGA
  double* exc_per_system = nullptr;     // [nsys], FP64 device accumulators
};

static_assert(std::is_standard_layout_v<XcGridIn>);
static_assert(std::is_standard_layout_v<XcGridOut>);
static_assert(alignof(XcGridIn) >= alignof(const double*));
static_assert(alignof(XcGridOut) >= alignof(double*));
static_assert(offsetof(XcGridIn, rho) == 0);
static_assert(offsetof(XcGridOut, wv_rho) == 0);
static_assert(sizeof(XcGridIn) % alignof(XcGridIn) == 0);
static_assert(sizeof(XcGridOut) % alignof(XcGridOut) == 0);

// The call enqueues work only: it neither allocates nor synchronizes. Unsupported
// functionals fail explicitly; they never take a hidden host fallback route.
[[nodiscard]] Status XcEval(const XcSpec& spec, const XcGridIn& input,
                            XcGridOut& output, cudaStream_t stream);

}  // namespace tides::grid::xc
