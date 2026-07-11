#pragma once

// T8.1: Single-node 2-GPU data model (NCCL).
//
// Defines the data model for distributing DFT work across 2 GPUs on a
// single node using NCCL for inter-GPU communication.
//
// The model:
//   1. Atoms are partitioned across GPUs using the graph partitioner (T8.2).
//   2. Each GPU owns a subset of atoms and their associated basis functions.
//   3. Tiles that span the partition boundary require halo exchange.
//   4. NCCL all-reduce is used for global reductions (energy, density).
//   5. NCCL send/recv is used for tile halo exchange between GPUs.
//
// This header defines the data structures and communication plan.
// The actual NCCL calls are guarded by TIDES_HAVE_NCCL and fall back to
// CPU simulation for testing.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/status.hpp"
#include "parallel/graph_partitioner.hpp"
#include "parallel/halo_exchange.hpp"

namespace tides::parallel {

// Configuration for a 2-GPU run.
struct MultiGPUConfig {
  int n_gpus = 2;
  int gpu_ids[2] = {0, 1};
  // Whether to use NCCL or CPU simulation.
  bool use_nccl = false;
  // NVLink bandwidth (GB/s) for profiling.
  double nvlink_bandwidth_gbps = 600.0;
};

// Per-GPU data partition.
struct GPUPartition {
  int gpu_id = 0;
  std::vector<std::size_t> owned_atoms;       // atoms owned by this GPU
  std::vector<std::size_t> owned_basis;       // basis function indices owned
  std::vector<std::size_t> halo_atoms;        // atoms needed from other GPU(s)
  std::vector<std::size_t> halo_basis;        // basis functions needed from other GPU(s)
  std::size_t n_local_tiles = 0;              // tiles fully on this GPU
  std::size_t n_halo_tiles = 0;               // tiles spanning the boundary
  double estimated_work_fraction = 0.0;       // fraction of total work
};

// Communication plan for one SCF step across 2 GPUs.
struct CommPlan {
  // Halo exchange: tiles that need to be sent to the other GPU.
  std::vector<std::size_t> send_tiles;        // tile indices to send
  std::vector<std::size_t> recv_tiles;        // tile indices to receive
  std::size_t exchange_bytes = 0;             // total bytes to exchange

  // Global reductions: values that need all-reduce.
  int n_allreduces = 0;                       // number of all-reduce calls
  std::size_t allreduce_bytes = 0;            // total bytes for all-reduces

  // Estimated communication time (ms).
  double halo_time_ms = 0.0;
  double allreduce_time_ms = 0.0;
  double total_comm_time_ms = 0.0;
  double comm_fraction = 0.0;                 // fraction of step time
};

// Result of a 2-GPU SCF step simulation.
struct MultiGPUStepResult {
  double energy = 0.0;          // total energy (after all-reduce)
  std::vector<double> density;  // density matrix (local portion)
  CommPlan comm_plan;
  bool ok = false;
};

// Build the 2-GPU partition from atom positions and neighbor list.
//
// @param n_atoms         Number of atoms
// @param atom_positions  3*n_atoms positions (Bohr)
// @param neighbor_list   Per-atom neighbor lists
// @param config          Multi-GPU configuration
// @return                Vector of GPUPartition (one per GPU)
[[nodiscard]] inline std::vector<GPUPartition> BuildPartitions(
    std::size_t n_atoms,
    const std::vector<double>& atom_positions,
    const std::vector<std::vector<std::size_t>>& neighbor_list,
    const MultiGPUConfig& config) {
  std::vector<GPUPartition> partitions(config.n_gpus);
  if (n_atoms == 0 || config.n_gpus <= 0) return partitions;

  // Use the graph partitioner to assign atoms to GPUs.
  auto part = GraphPartitioner::Partition(
      atom_positions, neighbor_list, n_atoms, config.n_gpus);

  // Build per-GPU ownership and halo lists.
  for (int g = 0; g < config.n_gpus; ++g) {
    partitions[g].gpu_id = config.gpu_ids[g];
  }

  for (std::size_t a = 0; a < n_atoms; ++a) {
    int owner = part.assignment[a];
    if (owner >= 0 && owner < config.n_gpus)
      partitions[owner].owned_atoms.push_back(a);
  }

  // Determine halo atoms: atoms in neighbor_list that are owned by another GPU.
  for (int g = 0; g < config.n_gpus; ++g) {
    std::vector<bool> is_owned(n_atoms, false);
    for (std::size_t a : partitions[g].owned_atoms)
      is_owned[a] = true;

    for (std::size_t a : partitions[g].owned_atoms) {
      for (std::size_t nb : neighbor_list[a]) {
        if (nb < n_atoms && !is_owned[nb]) {
          // This neighbor is on another GPU — it's a halo atom.
          if (std::find(partitions[g].halo_atoms.begin(),
                        partitions[g].halo_atoms.end(), nb) ==
              partitions[g].halo_atoms.end()) {
            partitions[g].halo_atoms.push_back(nb);
          }
        }
      }
    }

    // Estimate work fraction.
    double total = static_cast<double>(n_atoms);
    double local = static_cast<double>(partitions[g].owned_atoms.size());
    partitions[g].estimated_work_fraction = (total > 0) ? local / total : 0.0;

    // Estimate tile counts (simplified: n_local = owned^2, n_halo = owned * halo).
    std::size_t n_own = partitions[g].owned_atoms.size();
    std::size_t n_halo = partitions[g].halo_atoms.size();
    partitions[g].n_local_tiles = n_own * n_own;
    partitions[g].n_halo_tiles = n_own * n_halo;
  }

  return partitions;
}

// Build the communication plan for one SCF step.
//
// @param partitions  Per-GPU partitions
// @param config      Multi-GPU configuration
// @param tile_bytes  Size of each tile in bytes (default: 64*64*8 = 32768)
// @param step_time_ms Estimated compute time per step (ms)
// @return            Communication plan
[[nodiscard]] inline CommPlan BuildCommPlan(
    const std::vector<GPUPartition>& partitions,
    const MultiGPUConfig& config,
    std::size_t tile_bytes = 64 * 64 * sizeof(double),
    double step_time_ms = 10.0) {
  CommPlan plan;
  if (partitions.empty() || config.n_gpus <= 0) return plan;

  // Halo exchange: each GPU sends its halo tiles to the other.
  for (const auto& p : partitions) {
    plan.send_tiles.push_back(p.n_halo_tiles);
    plan.recv_tiles.push_back(p.n_halo_tiles);
    plan.exchange_bytes += p.n_halo_tiles * tile_bytes;
  }

  // Global reductions: energy (1 double), density diagonal (n_atoms doubles),
  // and Pulay mixing vectors (a few doubles per SCF iteration).
  plan.n_allreduces = 3;  // energy, density trace, Pulay residual
  plan.allreduce_bytes = static_cast<std::size_t>(partitions[0].owned_atoms.size()) *
                         sizeof(double) * 2;  // density + forces

  // Estimate communication time.
  // NVLink bandwidth: GB/s -> bytes/ms.
  double bw_B_per_ms = (config.nvlink_bandwidth_gbps * 1e9) / 1e3;
  plan.halo_time_ms = static_cast<double>(plan.exchange_bytes) / bw_B_per_ms;
  // All-reduce: ring all-reduce = 2 * (n_gpus-1) * data_size / bandwidth.
  plan.allreduce_time_ms =
      2.0 * (config.n_gpus - 1) *
      static_cast<double>(plan.allreduce_bytes) / bw_B_per_ms;
  plan.total_comm_time_ms = plan.halo_time_ms + plan.allreduce_time_ms;
  if (step_time_ms > 1e-30)
    plan.comm_fraction = plan.total_comm_time_ms / step_time_ms;

  return plan;
}

// Simulate a 2-GPU SCF step (CPU reference for testing).
// Each "GPU" computes its local energy contribution, then the results
// are combined (simulating NCCL all-reduce).
[[nodiscard]] inline MultiGPUStepResult SimulateSCFStep(
    const std::vector<GPUPartition>& partitions,
    const std::vector<double>& atom_energies,
    const MultiGPUConfig& config,
    double step_time_ms = 10.0) {
  MultiGPUStepResult res;
  if (partitions.empty()) return res;

  // Each GPU computes its local energy sum.
  double total_energy = 0.0;
  for (const auto& p : partitions) {
    double local_E = 0.0;
    for (std::size_t a : p.owned_atoms) {
      if (a < atom_energies.size())
        local_E += atom_energies[a];
    }
    // Simulate all-reduce: sum across GPUs.
    total_energy += local_E;
  }

  res.energy = total_energy;
  res.comm_plan = BuildCommPlan(partitions, config, 64 * 64 * sizeof(double),
                                 step_time_ms);
  res.ok = true;
  return res;
}

// Check load balance: the imbalance between GPU partitions.
[[nodiscard]] inline double LoadImbalance(
    const std::vector<GPUPartition>& partitions) {
  if (partitions.size() < 2) return 0.0;
  std::size_t max_load = 0, min_load = SIZE_MAX;
  for (const auto& p : partitions) {
    max_load = std::max(max_load, p.owned_atoms.size());
    min_load = std::min(min_load, p.owned_atoms.size());
  }
  double avg = static_cast<double>(max_load + min_load) / 2.0;
  if (avg < 1e-30) return 0.0;
  return static_cast<double>(max_load - min_load) / avg;
}

}  // namespace tides::parallel
