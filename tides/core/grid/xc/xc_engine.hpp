#pragma once

// The only Tier-0 XC entry point. All pointers in the grid views are device
// pointers in 256-byte-aligned structure-of-arrays storage.

#include <cstddef>
#include <cstdint>
#include <string>
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

// ---- Host-oriented convenience API (for GTO molecule driver) ----
// These types provide a simpler interface for host-side code that doesn't
// need device-resident data management.  The device-resident API above is
// the production path for the NAO pipeline; this is the GTO oracle path.

enum class XcFunctionalId : int {
  kLdaPw92 = 0, kLdaVwn5 = 1, kPbe = 2, kPbesol = 3, kRevPbe = 4,
  kRpbe = 5, kBlyp = 6, kPbe0Local = 7, kB3lypLocal = 8, kHse06Local = 9,
  kTpss = 10, kR2scan = 11, kScan = 12, kWb97xLocal = 13, kM062xLocal = 14,
};

enum class XcFamily : int {
  kLda = 0, kGga = 1, kMgga = 2, kHybrid = 3,
};

struct HostXcSpec {
  XcFunctionalId id = XcFunctionalId::kLdaPw92;
  XcFamily family = XcFamily::kLda;
  bool spin_polarized = false;
  double exchange_fraction = 1.0;
  double omega = 0.0;
};

struct HostXcGridIn {
  const double* rho = nullptr;
  const double* grad_rho_x = nullptr;
  const double* grad_rho_y = nullptr;
  const double* grad_rho_z = nullptr;
  const double* tau = nullptr;
  std::size_t np = 0;
  double grid_weight = 0.0;
};

struct HostXcGridOut {
  double* vxc = nullptr;
  double* vsigma = nullptr;
  double* vtau = nullptr;
  double* eps_xc = nullptr;
  double xc_energy = 0.0;
  double kernel_ms = 0.0;
};

bool XcEvalHost(const HostXcSpec& spec, const HostXcGridIn& in,
                HostXcGridOut& out, std::string& error_msg);
std::string XcFunctionalName(XcFunctionalId id);
bool IsTier0(XcFunctionalId id);

}  // namespace tides::grid::xc
