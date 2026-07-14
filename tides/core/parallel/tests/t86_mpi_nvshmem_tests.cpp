// T8.6: MPI + NVSHMEM multi-node tests (Phase C simulation).
//
// Validates the distributed SCF driver interface with CPU-only stubs.
// M6: Also validates actual MPI communication when run under mpirun.

#include "parallel/mpi_nvshmem.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::parallel::CommBackend;
using tides::parallel::DistributedConfig;
using tides::parallel::DistributedSCFDriver;
using tides::parallel::MPIOrchestrator;
using tides::parallel::NVSHMEMHaloExchange;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// M6: Test actual MPI rank/size retrieval.
int TestMPIRankSize() {
  std::cout << "\n=== M6: MPI rank/size ===\n";

  bool ok = MPIOrchestrator::Init();
  if (!ok) {
    std::cout << "  MPI not available — skipping\n";
    std::cout << "PASS (no MPI)\n";
    return 0;
  }

  int rank = -1, size = -1;
  MPIOrchestrator::GetRank(rank, size);

  std::cout << "  rank=" << rank << " size=" << size << "\n";

  if (rank < 0) return Fail("rank is negative");
  if (size < 1) return Fail("size < 1");
  if (rank >= size) return Fail("rank >= size");

  std::cout << "PASS\n";
  return 0;
}

// M6: Test actual MPI all-reduce sum.
int TestMPIAllReduce() {
  std::cout << "\n=== M6: MPI all-reduce ===\n";

  bool ok = MPIOrchestrator::Init();
  if (!ok) {
    std::cout << "  MPI not available — skipping\n";
    std::cout << "PASS (no MPI)\n";
    return 0;
  }

  int rank = 0, size = 1;
  MPIOrchestrator::GetRank(rank, size);

  // Each rank contributes its rank index + 1.
  double local = static_cast<double>(rank + 1);
  double global = MPIOrchestrator::AllReduceSum(local);

  // Expected: sum(1..size) = size*(size+1)/2.
  double expected = static_cast<double>(size) * (size + 1) / 2.0;
  std::cout << "  rank=" << rank << " local=" << local
            << " global=" << global << " expected=" << expected << "\n";

  if (std::fabs(global - expected) > 1e-12)
    return Fail("all-reduce sum mismatch: got " + std::to_string(global) +
                " expected " + std::to_string(expected));

  // Test vector all-reduce.
  std::vector<double> vec = {static_cast<double>(rank),
                              static_cast<double>(rank * 2)};
  MPIOrchestrator::AllReduceSum(vec);

  double expected_v0 = static_cast<double>(size * (size - 1) / 2);
  double expected_v1 = static_cast<double>(size * (size - 1));
  if (std::fabs(vec[0] - expected_v0) > 1e-12)
    return Fail("vector all-reduce[0] mismatch");
  if (std::fabs(vec[1] - expected_v1) > 1e-12)
    return Fail("vector all-reduce[1] mismatch");

  std::cout << "  vector reduce: [" << vec[0] << ", " << vec[1] << "]\n";
  std::cout << "PASS\n";
  return 0;
}

// M6: Test actual MPI halo exchange via Sendrecv.
int TestMPIHaloExchange() {
  std::cout << "\n=== M6: MPI halo exchange ===\n";

  bool ok = MPIOrchestrator::Init();
  if (!ok) {
    std::cout << "  MPI not available — skipping\n";
    std::cout << "PASS (no MPI)\n";
    return 0;
  }

  int rank = 0, size = 1;
  MPIOrchestrator::GetRank(rank, size);

  if (size < 2) {
    std::cout << "  Need >=2 ranks for halo exchange — skipping (size=" << size << ")\n";
    std::cout << "PASS (single rank)\n";
    return 0;
  }

  // 1D ring: rank i exchanges with (i+1)%size and (i-1+size)%size.
  const int next = (rank + 1) % size;
  const int prev = (rank - 1 + size) % size;

  // Send my rank to next, receive from prev.
  std::vector<double> send = {static_cast<double>(rank)};
  std::vector<double> recv(1, -1.0);
  bool sent = MPIOrchestrator::SendRecv(send, next, recv, prev);

  if (!sent) return Fail("SendRecv failed");

  std::cout << "  rank=" << rank << " sent=" << send[0]
            << " recv=" << recv[0] << " (from rank " << prev << ")\n";

  // Should receive the rank of the previous process.
  if (std::fabs(recv[0] - static_cast<double>(prev)) > 1e-12)
    return Fail("halo exchange: expected " + std::to_string(prev) +
                " got " + std::to_string(recv[0]));

  // Test via NVSHMEMHaloExchange::Exchange with MPI backend.
  DistributedConfig cfg;
  cfg.backend = CommBackend::MPI;
  cfg.n_partitions = size;
  cfg.rank = rank;

  // halo_data: [send_half | recv_half], 4 elements each.
  std::vector<double> halo(8, 0.0);
  for (int i = 0; i < 4; ++i) halo[i] = static_cast<double>(rank * 10 + i);
  std::vector<int> neighbors = {next};
  NVSHMEMHaloExchange::Exchange(halo, neighbors, cfg);

  std::cout << "  Exchange: send=[" << halo[0] << "," << halo[1]
            << "] recv=[" << halo[4] << "," << halo[5] << "]\n";

  // Should receive first elements from next rank's send half.
  if (std::fabs(halo[4] - static_cast<double>(next * 10)) > 1e-12)
    return Fail("Exchange: expected " + std::to_string(next * 10) +
                " got " + std::to_string(halo[4]));

  std::cout << "PASS\n";
  return 0;
}

// M6: Test MPI barrier synchronization.
int TestMPIBarrier() {
  std::cout << "\n=== M6: MPI barrier ===\n";

  bool ok = MPIOrchestrator::Init();
  if (!ok) {
    std::cout << "  MPI not available — skipping\n";
    std::cout << "PASS (no MPI)\n";
    return 0;
  }

  // Just call barrier — if it hangs, the test will timeout.
  MPIOrchestrator::Barrier();
  std::cout << "  Barrier completed\n";
  std::cout << "PASS\n";
  return 0;
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
  int failures = 0;

  // M6: Real MPI communication tests.
  failures += TestMPIRankSize();
  failures += TestMPIAllReduce();
  failures += TestMPIHaloExchange();
  failures += TestMPIBarrier();

  // Original T8.6 simulation tests.
  failures += TestSinglePartition();
  failures += TestMPIBackend();
  failures += TestNVSHMEMBackend();
  failures += TestNCCLBackend();

  // Finalize MPI if it was initialized.
  MPIOrchestrator::Finalize();

  if (failures == 0) {
    std::cout << "\nt86_mpi_nvshmem: ALL GREEN\n";
    return 0;
  }
  std::cout << "\nt86_mpi_nvshmem: " << failures << " FAILED\n";
  return failures;
}
