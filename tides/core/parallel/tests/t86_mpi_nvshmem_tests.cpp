// T8.6: MPI + NVSHMEM multi-node tests (Phase C simulation).
//
// Validates the distributed SCF driver interface with CPU-only stubs.

#include "parallel/mpi_nvshmem.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {

using tides::parallel::CommBackend;
using tides::parallel::DistributedConfig;
using tides::parallel::DistributedSCFDriver;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

int TestSinglePartition() {
  std::cout << "\n=== T8.6: Single-partition SCF ===\n";

  DistributedConfig cfg;
  cfg.backend = CommBackend::None;
  cfg.n_partitions = 1;
  cfg.rank = 0;

  auto result = DistributedSCFDriver::Run(cfg);

  if (!result.converged) return Fail("SCF did not converge");
  if (result.n_iterations != 3) return Fail("wrong iteration count");
  if (result.energy >= -1.0) return Fail("energy should be negative");

  std::cout << "  Energy: " << result.energy << '\n';
  std::cout << "  Iterations: " << result.n_iterations << '\n';
  std::cout << "PASS\n";
  return 0;
}

int TestMPIBackend() {
  std::cout << "\n=== T8.6: MPI backend simulation ===\n";

  DistributedConfig cfg;
  cfg.backend = CommBackend::MPI;
  cfg.n_partitions = 1;
  cfg.rank = 0;

  auto result = DistributedSCFDriver::Run(cfg);

  if (!result.converged) return Fail("SCF did not converge with MPI backend");

  std::cout << "  Energy: " << result.energy << '\n';
  std::cout << "PASS\n";
  return 0;
}

int TestNVSHMEMBackend() {
  std::cout << "\n=== T8.6: NVSHMEM backend simulation ===\n";

  DistributedConfig cfg;
  cfg.backend = CommBackend::NVSHMEM;
  cfg.n_partitions = 1;
  cfg.rank = 0;
  cfg.nvshmem_heap_size_mb = 256;

  auto result = DistributedSCFDriver::Run(cfg);

  if (!result.converged) return Fail("SCF did not converge with NVSHMEM backend");

  std::cout << "  Energy: " << result.energy << '\n';
  std::cout << "  (NVSHMEM stub: no actual GPU communication)\n";
  std::cout << "PASS\n";
  return 0;
}

int TestNCCLBackend() {
  std::cout << "\n=== T8.6: NCCL backend simulation ===\n";

  DistributedConfig cfg;
  cfg.backend = CommBackend::NCCL;
  cfg.n_partitions = 1;
  cfg.rank = 0;

  auto result = DistributedSCFDriver::Run(cfg);

  if (!result.converged) return Fail("SCF did not converge with NCCL backend");

  std::cout << "  Energy: " << result.energy << '\n';
  std::cout << "PASS\n";
  return 0;
}

}  // namespace

int main() {
  if (TestSinglePartition()) return 1;
  if (TestMPIBackend()) return 1;
  if (TestNVSHMEMBackend()) return 1;
  if (TestNCCLBackend()) return 1;

  std::cout << "\nt86_mpi_nvshmem: ALL GREEN\n";
  return 0;
}
