// T8.6: MPI + NVSHMEM multi-node interface stubs (Phase C).
//
// This file provides the interface definitions for multi-node distributed
// execution using MPI for CPU-side orchestration and NVSHMEM for GPU-side
// one-sided communication. The actual implementations require NVSHMEM
// and MPI installations and are compiled conditionally.
//
// Key design:
//   - MPI for process management, CPU-side reductions, checkpointing
//   - NVSHMEM for GPU-to-GPU one-sided halo exchange (PGAS model)
//   - NCCL as fallback for single-node multi-GPU
//
// This prototype validates the interface with a CPU-only simulation.

#ifdef TIDES_HAVE_MPI
#include <mpi.h>
#endif

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace tides::parallel {

// Communication backend selection.
enum class CommBackend {
  None,     // Single-process (no communication)
  NCCL,     // Single-node multi-GPU
  NVSHMEM,  // Multi-node GPU (PGAS, one-sided)
  MPI,      // CPU-only fallback
};

// Configuration for distributed execution.
struct DistributedConfig {
  CommBackend backend = CommBackend::None;
  int n_partitions = 1;
  int rank = 0;
  std::string nvshmem_bootstrap = "MPI";  // or "SHMEM"
  int nvshmem_heap_size_mb = 512;
  bool use_async_exchange = true;
};

// NVSHMEM-backed halo exchange interface.
// In the actual implementation, this uses nvshmem_put/nvshmem_get
// for one-sided communication between GPU partitions.
class NVSHMEMHaloExchange {
 public:
  // Initialize NVSHMEM with MPI bootstrap.
  // Requires: MPI_Init already called, CUDA context set on each GPU.
  static bool Init(const DistributedConfig& cfg) {
#ifdef TIDES_HAVE_NVSHMEM
    // Actual implementation:
    // nvshmem_init();
    // nvshmem_malloc for symmetric heap;
    // setup neighbor mappings;
    return true;
#else
    // Stub: no NVSHMEM available.
    return false;
#endif
  }

  // Exchange halo data between partitions.
  // In NVSHMEM, this uses nvshmem_put_nbi (non-blocking) for each
  // neighbor partition, then nvshmem_quiet to wait.
  // With MPI, uses MPI_Sendrecv for each neighbor pair.
  static void Exchange(std::vector<double>& halo_data,
                        const std::vector<int>& neighbors,
                        const DistributedConfig& cfg) {
#ifdef TIDES_HAVE_NVSHMEM
    // Actual implementation:
    // for each neighbor: nvshmem_double_put_nbi(target, source, count, pe);
    // nvshmem_quiet();
    if (cfg.n_partitions == 1) return;
#elif defined(TIDES_HAVE_MPI)
    // M6: MPI fallback for halo exchange.
    // For each neighbor, use MPI_Sendrecv to exchange boundary data.
    if (cfg.n_partitions == 1) return;
    const int rank = cfg.rank;
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
      const int neighbor = neighbors[i];
      if (neighbor < 0 || neighbor == rank) continue;
      // Pack send buffer (first half of halo_data is send, second half is recv).
      const std::size_t half = halo_data.size() / 2;
      std::vector<double> send_buf(halo_data.begin(), halo_data.begin() + half);
      std::vector<double> recv_buf(half, 0.0);
      MPI_Sendrecv(send_buf.data(), static_cast<int>(half), MPI_DOUBLE,
                   neighbor, 0,
                   recv_buf.data(), static_cast<int>(half), MPI_DOUBLE,
                   neighbor, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // Unpack into second half of halo_data.
      for (std::size_t j = 0; j < half; ++j)
        halo_data[half + j] = recv_buf[j];
    }
#else
    // Stub: no communication available.
    if (cfg.n_partitions == 1) return;
#endif
  }

  // All-reduce via NVSHMEM or NCCL.
  static double AllReduceSum(double local_value, const DistributedConfig& cfg) {
#ifdef TIDES_HAVE_NVSHMEM
    // nvshmem_double_allreduce_sum
    return local_value;  // placeholder
#elif defined(TIDES_HAVE_MPI)
    double global = 0.0;
    MPI_Allreduce(&local_value, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
#else
    // Single-partition: no reduction needed.
    return local_value;
#endif
  }

  // Finalize NVSHMEM.
  static void Finalize() {
#ifdef TIDES_HAVE_NVSHMEM
    // nvshmem_finalize();
#endif
  }
};

// MPI-backed orchestration interface.
class MPIOrchestrator {
 public:
  // Initialize MPI. Returns true if MPI is available (or already initialized).
  static bool Init() {
#ifdef TIDES_HAVE_MPI
    int provided = 0;
    int mpi_init_flag = 0;
    MPI_Initialized(&mpi_init_flag);
    if (!mpi_init_flag) {
      // Request MPI_THREAD_SINGLE — we don't use threaded MPI.
      int err = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided);
      if (err != MPI_SUCCESS) return false;
    }
    return true;
#else
    return false;
#endif
  }

  // Get rank and size.
  static void GetRank(int& rank, int& size) {
#ifdef TIDES_HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
    rank = 0;
    size = 1;
#endif
  }

  // Barrier.
  static void Barrier() {
#ifdef TIDES_HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  // All-reduce sum (scalar double).
  static double AllReduceSum(double local) {
#ifdef TIDES_HAVE_MPI
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
#else
    return local;
#endif
  }

  // All-reduce sum (vector of doubles, in-place).
  static void AllReduceSum(std::vector<double>& local) {
#ifdef TIDES_HAVE_MPI
    if (local.empty()) return;
    std::vector<double> global(local.size(), 0.0);
    MPI_Allreduce(local.data(), global.data(),
                  static_cast<int>(local.size()),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    local = global;
#endif
  }

  // Sendrecv: exchange data with a neighbor rank.
  // Sends sendbuf to dst, receives into recvbuf from src.
  static bool SendRecv(const std::vector<double>& sendbuf, int dst,
                       std::vector<double>& recvbuf, int src) {
#ifdef TIDES_HAVE_MPI
    if (sendbuf.size() != recvbuf.size()) return false;
    MPI_Sendrecv(const_cast<double*>(sendbuf.data()),
                 static_cast<int>(sendbuf.size()), MPI_DOUBLE, dst, 0,
                 recvbuf.data(),
                 static_cast<int>(recvbuf.size()), MPI_DOUBLE, src, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return true;
#else
    return false;
#endif
  }

  // Bcast: broadcast data from root to all ranks.
  static void Bcast(std::vector<double>& data, int root) {
#ifdef TIDES_HAVE_MPI
    if (data.empty()) return;
    MPI_Bcast(data.data(), static_cast<int>(data.size()),
              MPI_DOUBLE, root, MPI_COMM_WORLD);
#endif
  }

  // Finalize.
  static void Finalize() {
#ifdef TIDES_HAVE_MPI
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
      MPI_Finalize();
    }
#endif
  }
};

// Distributed SCF driver using MPI + NVSHMEM.
class DistributedSCFDriver {
 public:
  struct Result {
    double energy = 0.0;
    int n_iterations = 0;
    bool converged = false;
    double comm_time = 0.0;
    double compute_time = 0.0;
  };

  static Result Run(const DistributedConfig& cfg) {
    Result result;

    // Initialize communication.
    bool have_mpi = MPIOrchestrator::Init();
    bool have_nvshmem = NVSHMEMHaloExchange::Init(cfg);

    int rank, size;
    MPIOrchestrator::GetRank(rank, size);

    if (rank == 0) {
      std::cout << "Distributed SCF: " << size << " partitions"
                << ", backend=" << static_cast<int>(cfg.backend)
                << ", NVSHMEM=" << (have_nvshmem ? "yes" : "no")
                << ", MPI=" << (have_mpi ? "yes" : "no") << '\n';
    }

    // Simulate SCF iterations.
    for (int iter = 0; iter < 3; ++iter) {
      // Local computation (simulated).
      double local_energy = -1.0 - 0.01 * iter;

      // Global reduction.
      double global_energy = MPIOrchestrator::AllReduceSum(local_energy);
      global_energy = NVSHMEMHaloExchange::AllReduceSum(global_energy, cfg);

      if (rank == 0)
        std::cout << "  Iter " << iter << ": E = " << global_energy << '\n';

      result.energy = global_energy;
      result.n_iterations = iter + 1;
    }

    result.converged = true;

    // Finalize NVSHMEM (MPI is finalized by the caller).
    NVSHMEMHaloExchange::Finalize();

    return result;
  }
};

}  // namespace tides::parallel
