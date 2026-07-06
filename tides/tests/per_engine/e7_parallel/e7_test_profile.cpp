// E7: Parallel Engine — Comprehensive Test & Profile Suite
//
// Tests and profiles:
//   1. Graph partitioner (recursive bisection)
//   2. Halo exchange (1D and 3D)
//   3. Communication fraction model

#include "parallel/graph_partitioner.hpp"
#include "parallel/halo_exchange.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::parallel::GraphPartitioner;
using tides::parallel::PartitionResult;
using tides::parallel::HaloExchange;
using tides::parallel::HaloExchangeResult;

struct ProfileEntry {
  std::string kernel;
  std::string variant;
  std::string size_label;
  double time_ms = 0.0;
  double error = 0.0;
  std::string status;
};

std::vector<ProfileEntry> g_log;

void Log(const std::string& kernel, const std::string& variant,
         const std::string& size_label, double time_ms, double error,
         const std::string& status) {
  ProfileEntry e{kernel, variant, size_label, time_ms, error, status};
  g_log.push_back(e);
  std::cout << "  " << std::left << std::setw(18) << kernel
            << std::setw(12) << variant
            << std::setw(14) << size_label
            << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_ms
            << std::setw(14) << std::scientific << std::setprecision(3) << error
            << "  " << status << '\n';
}

void PrintHeader() {
  std::cout << std::left << std::setw(18) << "Kernel"
            << std::setw(12) << "Variant"
            << std::setw(14) << "Size"
            << std::right << std::setw(10) << "Time(ms)"
            << std::setw(14) << "Error"
            << "  Status\n";
  std::cout << std::string(82, '-') << '\n';
}

// --- Test 1: Graph partitioner ---
int TestGraphPartitioner() {
  std::cout << "\n=== E7.1: Graph Partitioner (recursive bisection) ===\n";
  PrintHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto [n_verts, n_parts] : std::vector<std::pair<int, int>>{
           {100, 2}, {1000, 4}, {4000, 8}, {10000, 16}}) {
    // Generate random 3D coordinates.
    std::uniform_real_distribution<double> dist(0.0, 10.0);
    std::vector<double> coords(3 * n_verts);
    for (auto& v : coords) v = dist(rng);

    // Build adjacency: each vertex connected to nearest 4.
    std::vector<std::vector<std::size_t>> adj(n_verts);
    for (int i = 0; i < n_verts; ++i) {
      std::vector<std::pair<double, int>> dists;
      for (int j = 0; j < n_verts; ++j) {
        if (i == j) continue;
        double dx = coords[3*i] - coords[3*j];
        double dy = coords[3*i+1] - coords[3*j+1];
        double dz = coords[3*i+2] - coords[3*j+2];
        dists.push_back({dx*dx + dy*dy + dz*dz, j});
      }
      std::sort(dists.begin(), dists.end());
      for (int k = 0; k < std::min(4, n_verts - 1); ++k)
        adj[i].push_back(dists[k].second);
    }

    auto t0 = std::chrono::steady_clock::now();
    auto result = GraphPartitioner::Partition(coords, adj, n_verts, n_parts);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Check imbalance <= 10%.
    std::string status = (result.imbalance <= 0.10) ? "PASS" : "FAIL";
    if (result.imbalance > 0.10) failures++;
    Log("Partitioner", "RCB",
        std::to_string(n_verts) + "/" + std::to_string(n_parts),
        ms, result.imbalance, status);
  }
  return failures;
}

// --- Test 2: Halo exchange ---
int TestHaloExchange() {
  std::cout << "\n=== E7.2: Halo Exchange ===\n";
  PrintHeader();
  int failures = 0;

  // 1D halo exchange.
  for (auto [n_local, n_halo] : std::vector<std::pair<int, int>>{
           {100, 2}, {1000, 4}, {10000, 8}}) {
    std::vector<double> local(n_local);
    for (int i = 0; i < n_local; ++i) local[i] = static_cast<double>(i);

    std::vector<double> left_halo(n_halo, -1.0);
    std::vector<double> right_halo(n_halo, -2.0);

    auto t0 = std::chrono::steady_clock::now();
    auto result = HaloExchange::Exchange1D(local, left_halo, right_halo, n_halo);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Verify: [left_halo | local | right_halo].
    double err = 0.0;
    if (!result.ok) {
      err = 1.0;
    } else {
      for (int i = 0; i < n_halo; ++i)
        err = std::max(err, std::abs(result.data[i] - left_halo[i]));
      for (int i = 0; i < n_local; ++i)
        err = std::max(err, std::abs(result.data[n_halo + i] - local[i]));
      for (int i = 0; i < n_halo; ++i)
        err = std::max(err, std::abs(result.data[n_halo + n_local + i] - right_halo[i]));
    }

    std::string status = (err < 1e-15) ? "PASS" : "FAIL";
    if (err >= 1e-15) failures++;
    Log("HaloExchange", "1D",
        std::to_string(n_local) + "+" + std::to_string(n_halo),
        ms, err, status);
  }

  // 3D halo exchange.
  {
    int nx = 10, ny = 10, nz = 10;
    std::vector<double> local(nx * ny * nz);
    for (int i = 0; i < nx * ny * nz; ++i) local[i] = static_cast<double>(i);

    int n_halo = ny * nz;
    std::vector<double> left_face(n_halo, -1.0);
    std::vector<double> right_face(n_halo, -2.0);

    auto t0 = std::chrono::steady_clock::now();
    auto result = HaloExchange::Exchange3D(local, nx, ny, nz,
                                            left_face, right_face, n_halo);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::string status = (result.ok) ? "PASS" : "FAIL";
    if (!result.ok) failures++;
    Log("HaloExchange", "3D",
        "10^3+halo",
        ms, 0, status);
  }

  return failures;
}

// --- Test 3: Communication fraction model ---
int TestCommFraction() {
  std::cout << "\n=== E7.3: Communication Fraction Model ===\n";
  PrintHeader();
  int failures = 0;

  // Test the communication fraction model.
  double vol = 1e6;  // 1 MB
  double step_ms = 10.0;
  double frac = HaloExchange::CommFraction(vol, step_ms, 50.0);

  // 50 Gbps = 6.25 GB/s = 6.25e6 B/ms.
  // Time = 1e6 / 6.25e6 = 0.16 ms. Fraction = 0.16/10 = 0.016.
  double expected = (1e6 / (50.0 * 1e9 / 8.0 / 1e3)) / step_ms;
  double err = std::abs(frac - expected);

  std::string status = (err < 1e-10) ? "PASS" : "FAIL";
  if (err >= 1e-10) failures++;
  Log("CommFraction", "model",
    "1MB/10ms",
    0, err, status);

  return failures;
}

void PrintSummary(int total_failures) {
  std::cout << "\n=== E7 Summary ===\n";
  std::cout << "Total profile entries: " << g_log.size() << '\n';
  if (total_failures == 0) {
    std::cout << "ALL E7 TESTS PASSED\n";
  } else {
    std::cout << total_failures << " E7 TEST(S) FAILED\n";
  }
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║   E7: Parallel Engine — Test & Profile Suite                 ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestGraphPartitioner();
  failures += TestHaloExchange();
  failures += TestCommFraction();

  PrintSummary(failures);
  return failures;
}
