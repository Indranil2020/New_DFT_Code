// HIP runtime probe — validates that HIP is functional.
// Returns 77 (skip) if no HIP device is available, 0 on success.

#include "tile/hip_compat.hpp"

#include <cstdio>
#include <iostream>

int main() {
#ifdef TIDES_USE_HIP
  int device_count = 0;
  auto err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess || device_count == 0) {
    std::cout << "HIP: No devices found, skipping.\n";
    return 77;
  }

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  std::cout << "HIP: Device 0 = " << prop.name
            << " (SM=" << prop.major << "." << prop.minor << ")"
            << " cores=" << prop.multiProcessorCount
            << " warp=" << prop.warpSize << '\n';

  // Simple kernel test.
  double* d_ptr = nullptr;
  hipMalloc(&d_ptr, sizeof(double));
  double val = 42.0;
  hipMemcpy(d_ptr, &val, sizeof(double), hipMemcpyHostToDevice);
  double back = 0.0;
  hipMemcpy(&back, d_ptr, sizeof(double), hipMemcpyDeviceToHost);
  hipFree(d_ptr);

  if (back != 42.0) {
    std::cerr << "HIP: Memory copy failed (got " << back << ")\n";
    return 1;
  }

  std::cout << "HIP: Runtime OK (memcpy round-trip verified)\n";
  return 0;
#else
  std::cout << "HIP: Not compiled with TIDES_USE_HIP, skipping.\n";
  return 77;
#endif
}
