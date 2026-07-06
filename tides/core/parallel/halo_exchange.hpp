#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "parallel/graph_partitioner.hpp"

namespace tides::parallel {

// Halo exchange (T8.3): exchange boundary data between partitions.
//
// Per 34-parallelism-io: "halo exchange overlapped with compute (exposed
// comm <= 15% of step time on 4096-H2O proxy)."
//
// For the CPU reference, we implement the halo exchange logic: given a
// partition assignment and a grid, each partition packs its boundary data,
// sends it to neighbors, and unpacks received halos. The actual communication
// (NCCL for GPU, MPI for multi-node) is the production path.
//
// Observable (T8.3): halo exchange correctly fills ghost cells.

struct HaloExchangeResult {
  std::vector<double> data;       // data with halos filled
  int n_exchanges = 0;
  double exchange_volume = 0.0;  // bytes exchanged (for profiling)
  bool ok = false;
};

class HaloExchange {
 public:
  // Exchange halos for a 1D partitioned array.
  //   local_data: data owned by this partition (no halos)
  //   left_neighbor_data: last n_halo elements from the left neighbor
  //   right_neighbor_data: first n_halo elements from the right neighbor
  //   n_halo: number of halo layers on each side
  // Returns the extended data: [left_halo | local | right_halo].
  static HaloExchangeResult Exchange1D(
      const std::vector<double>& local_data,
      const std::vector<double>& left_halo,
      const std::vector<double>& right_halo,
      std::size_t n_halo) {
    HaloExchangeResult res;
    const std::size_t n_local = local_data.size();
    res.data.resize(n_halo + n_local + n_halo, 0.0);

    // Fill left halo.
    for (std::size_t i = 0; i < n_halo && i < left_halo.size(); ++i)
      res.data[i] = left_halo[left_halo.size() - n_halo + i];

    // Fill local data.
    for (std::size_t i = 0; i < n_local; ++i)
      res.data[n_halo + i] = local_data[i];

    // Fill right halo.
    for (std::size_t i = 0; i < n_halo && i < right_halo.size(); ++i)
      res.data[n_halo + n_local + i] = right_halo[i];

    res.n_exchanges = 2;  // left + right
    res.exchange_volume = 2.0 * n_halo * sizeof(double);
    res.ok = true;
    return res;
  }

  // Exchange halos for a 3D grid partition (simplified: only the axis
  // being partitioned). This is the pattern used in the tile-graph halo.
  static HaloExchangeResult Exchange3D(
      const std::vector<double>& local_grid,
      std::size_t nx, std::size_t ny, std::size_t nz,
      const std::vector<double>& left_face,
      const std::vector<double>& right_face,
      std::size_t n_halo) {
    HaloExchangeResult res;
    const std::size_t ext_nx = nx + 2 * n_halo;
    res.data.resize(ext_nx * ny * nz, 0.0);

    // Copy local data into the interior.
    for (std::size_t k = 0; k < nz; ++k)
      for (std::size_t j = 0; j < ny; ++j)
        for (std::size_t i = 0; i < nx; ++i)
          res.data[(k * ny + j) * ext_nx + (i + n_halo)] =
              local_grid[(k * ny + j) * nx + i];

    // Fill left halo (x < n_halo).
    for (std::size_t k = 0; k < nz; ++k)
      for (std::size_t j = 0; j < ny; ++j)
        for (std::size_t i = 0; i < n_halo; ++i)
          if (left_face.size() > (k * ny + j) * n_halo + i)
            res.data[(k * ny + j) * ext_nx + i] =
                left_face[(k * ny + j) * n_halo + i];

    // Fill right halo (x >= nx + n_halo).
    for (std::size_t k = 0; k < nz; ++k)
      for (std::size_t j = 0; j < ny; ++j)
        for (std::size_t i = 0; i < n_halo; ++i)
          if (right_face.size() > (k * ny + j) * n_halo + i)
            res.data[(k * ny + j) * ext_nx + (nx + n_halo + i)] =
                right_face[(k * ny + j) * n_halo + i];

    res.n_exchanges = 2;
    res.exchange_volume = 2.0 * n_halo * ny * nz * sizeof(double);
    res.ok = true;
    return res;
  }

  // Profile the exchange: compute the fraction of step time spent on
  // communication (simplified model).
  static double CommFraction(double exchange_volume_bytes,
                             double step_time_ms,
                             double bandwidth_gbps = 50.0) {
    // Bandwidth: Gbps -> bytes/ms. 50 Gbps = 50e9 bit/s = 6.25e9 B/s
    // = 6.25e6 B/ms. So bandwidth_B_per_ms = (bandwidth_gbps / 8) * 1e6.
    const double bandwidth_B_per_ms = (bandwidth_gbps / 8.0) * 1e6;
    const double comm_time_ms = exchange_volume_bytes / bandwidth_B_per_ms;
    if (step_time_ms < 1e-30) return 0.0;
    return comm_time_ms / step_time_ms;
  }
};

}  // namespace tides::parallel
