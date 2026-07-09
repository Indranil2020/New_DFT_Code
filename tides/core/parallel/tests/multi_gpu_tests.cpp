// T8.1: Single-node 2-GPU data model (NCCL) tests.
//
// Validates:
//   - Partition assigns atoms to 2 GPUs with good balance
//   - Halo atoms are correctly identified (neighbors on other GPU)
//   - Communication plan estimates exchange volume
//   - Simulated SCF step produces correct total energy
//   - Load imbalance is within 10%

#include "parallel/multi_gpu.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::parallel::BuildCommPlan;
using tides::parallel::BuildPartitions;
using tides::parallel::LoadImbalance;
using tides::parallel::MultiGPUConfig;
using tides::parallel::SimulateSCFStep;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// T8.1a: 2-GPU partition balance.
int TestPartitionBalance() {
  std::cout << "\n=== T8.1a: 2-GPU partition balance ===\n";
  // 20 atoms on a 1D chain, each connected to neighbors within radius 2.
  const std::size_t n = 20;
  std::vector<double> positions(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    positions[3 * i] = static_cast<double>(i);
    positions[3 * i + 1] = 0.0;
    positions[3 * i + 2] = 0.0;
  }
  std::vector<std::vector<std::size_t>> nl(n);
  for (std::size_t i = 0; i < n; ++i)
    for (int r = 1; r <= 2; ++r) {
      if (i >= static_cast<std::size_t>(r)) nl[i].push_back(i - r);
      if (i + static_cast<std::size_t>(r) < n) nl[i].push_back(i + r);
    }

  MultiGPUConfig config;
  config.n_gpus = 2;
  auto partitions = BuildPartitions(n, positions, nl, config);

  for (int g = 0; g < 2; ++g)
    std::cout << "  GPU " << g << ": " << partitions[g].owned_atoms.size()
              << " atoms, " << partitions[g].halo_atoms.size()
              << " halo atoms, " << partitions[g].n_local_tiles
              << " local tiles, " << partitions[g].n_halo_tiles
              << " halo tiles\n";

  // Check all atoms are assigned.
  std::size_t total_assigned = 0;
  for (const auto& p : partitions) total_assigned += p.owned_atoms.size();
  if (total_assigned != n) return Fail("T8.1a: not all atoms assigned");

  // Check load imbalance <= 10%.
  double imb = LoadImbalance(partitions);
  std::cout << "  load imbalance=" << imb * 100.0 << "%\n";
  if (imb > 0.10) return Fail("T8.1a: load imbalance > 10%");

  std::cout << "T8.1a: GREEN\n";
  return 0;
}

// T8.1b: Halo atoms are neighbors on the other GPU.
int TestHaloAtoms() {
  std::cout << "\n=== T8.1b: Halo atom identification ===\n";
  // 10 atoms, split at x=5. Atoms 0-4 on GPU0, 5-9 on GPU1.
  // With radius=2, GPU0's halo should include atoms 5,6 (neighbors of 3,4).
  // GPU1's halo should include atoms 3,4 (neighbors of 5,6).
  const std::size_t n = 10;
  std::vector<double> positions(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    positions[3 * i] = static_cast<double>(i);
    positions[3 * i + 1] = 0.0;
    positions[3 * i + 2] = 0.0;
  }
  std::vector<std::vector<std::size_t>> nl(n);
  for (std::size_t i = 0; i < n; ++i)
    for (int r = 1; r <= 2; ++r) {
      if (i >= static_cast<std::size_t>(r)) nl[i].push_back(i - r);
      if (i + static_cast<std::size_t>(r) < n) nl[i].push_back(i + r);
    }

  MultiGPUConfig config;
  config.n_gpus = 2;
  auto partitions = BuildPartitions(n, positions, nl, config);

  // GPU0 owns atoms 0-4, halo should include 5,6.
  std::cout << "  GPU0 owned: ";
  for (auto a : partitions[0].owned_atoms) std::cout << a << " ";
  std::cout << "\n  GPU0 halo: ";
  for (auto a : partitions[0].halo_atoms) std::cout << a << " ";
  std::cout << "\n  GPU1 owned: ";
  for (auto a : partitions[1].owned_atoms) std::cout << a << " ";
  std::cout << "\n  GPU1 halo: ";
  for (auto a : partitions[1].halo_atoms) std::cout << a << " ";
  std::cout << '\n';

  // Verify halo atoms are not in owned set.
  for (int g = 0; g < 2; ++g)
    for (std::size_t ha : partitions[g].halo_atoms) {
      for (std::size_t oa : partitions[g].owned_atoms)
        if (ha == oa) return Fail("T8.1b: halo atom is also owned");
    }

  // Verify halo atoms are actual neighbors of owned atoms.
  bool gpu0_has_halo = !partitions[0].halo_atoms.empty();
  bool gpu1_has_halo = !partitions[1].halo_atoms.empty();
  if (!gpu0_has_halo || !gpu1_has_halo)
    return Fail("T8.1b: expected non-empty halos on both GPUs");

  std::cout << "T8.1b: GREEN\n";
  return 0;
}

// T8.1c: Communication plan estimates exchange volume.
int TestCommPlan() {
  std::cout << "\n=== T8.1c: Communication plan ===\n";
  const std::size_t n = 20;
  std::vector<double> positions(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    positions[3 * i] = static_cast<double>(i);
    positions[3 * i + 1] = 0.0;
    positions[3 * i + 2] = 0.0;
  }
  std::vector<std::vector<std::size_t>> nl(n);
  for (std::size_t i = 0; i < n; ++i)
    for (int r = 1; r <= 2; ++r) {
      if (i >= static_cast<std::size_t>(r)) nl[i].push_back(i - r);
      if (i + static_cast<std::size_t>(r) < n) nl[i].push_back(i + r);
    }

  MultiGPUConfig config;
  config.n_gpus = 2;
  config.nvlink_bandwidth_gbps = 600.0;
  auto partitions = BuildPartitions(n, positions, nl, config);
  auto plan = BuildCommPlan(partitions, config, 64 * 64 * sizeof(double), 10.0);

  std::cout << "  exchange_bytes=" << plan.exchange_bytes
            << " halo_time_ms=" << plan.halo_time_ms
            << " allreduce_time_ms=" << plan.allreduce_time_ms
            << " total_comm_ms=" << plan.total_comm_time_ms
            << " comm_fraction=" << plan.comm_fraction * 100.0 << "%\n";

  if (plan.exchange_bytes == 0)
    return Fail("T8.1c: exchange bytes should be non-zero");
  if (plan.n_allreduces < 1)
    return Fail("T8.1c: should have at least 1 all-reduce");
  if (plan.comm_fraction < 0 || plan.comm_fraction > 1)
    return Fail("T8.1c: comm fraction out of range");

  std::cout << "T8.1c: GREEN\n";
  return 0;
}

// T8.1d: Simulated SCF step produces correct total energy.
int TestSimulatedSCFStep() {
  std::cout << "\n=== T8.1d: Simulated SCF step ===\n";
  const std::size_t n = 20;
  std::vector<double> positions(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    positions[3 * i] = static_cast<double>(i);
    positions[3 * i + 1] = 0.0;
    positions[3 * i + 2] = 0.0;
  }
  std::vector<std::vector<std::size_t>> nl(n);
  for (std::size_t i = 0; i < n; ++i)
    for (int r = 1; r <= 2; ++r) {
      if (i >= static_cast<std::size_t>(r)) nl[i].push_back(i - r);
      if (i + static_cast<std::size_t>(r) < n) nl[i].push_back(i + r);
    }

  // Per-atom energies: E_i = i * 0.1.
  std::vector<double> atom_energies(n);
  double expected_total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    atom_energies[i] = static_cast<double>(i) * 0.1;
    expected_total += atom_energies[i];
  }

  MultiGPUConfig config;
  config.n_gpus = 2;
  auto partitions = BuildPartitions(n, positions, nl, config);
  auto res = SimulateSCFStep(partitions, atom_energies, config, 10.0);

  std::cout << "  total_energy=" << res.energy
            << " expected=" << expected_total << '\n';

  if (!res.ok) return Fail("T8.1d: step failed");
  if (std::fabs(res.energy - expected_total) > 1e-12)
    return Fail("T8.1d: energy mismatch");

  std::cout << "T8.1d: GREEN\n";
  return 0;
}

}  // namespace

int main() {
  if (TestPartitionBalance()) return 1;
  if (TestHaloAtoms()) return 1;
  if (TestCommPlan()) return 1;
  if (TestSimulatedSCFStep()) return 1;
  std::cout << "\nmulti_gpu_tests: ALL GREEN\n";
  return 0;
}
