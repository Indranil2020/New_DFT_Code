// E6: CUDA graph SCF tests.
// Verifies graph capture, replay, and overhead estimation.
#include "tile/cuda_graph_scf.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using tides::tile::CudaGraphSCF;

int Fail(const std::string& msg) {
  std::cerr << "cuda_graph_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestCaptureAndReplay() {
  std::cout << "\n=== E6: Graph Capture and Replay ===\n";
  CudaGraphSCF graph;
  int counter = 0;

  graph.BeginCapture();
  graph.Record("op1", [&]() { counter += 1; });
  graph.Record("op2", [&]() { counter += 10; });
  graph.Record("op3", [&]() { counter += 100; });
  graph.EndCapture();

  if (!graph.IsCaptured()) return Fail("Graph should be captured");
  if (graph.OperationCount() != 3) return Fail("Should have 3 operations");

  int executed = graph.Replay();
  std::cout << "  Executed " << executed << " ops, counter = " << counter << "\n";
  if (executed != 3) return Fail("Should execute 3 operations");
  if (counter != 111) return Fail("Counter should be 111");
  std::cout << "  PASS\n";
  return 0;
}

int TestNames() {
  std::cout << "\n=== E6: Operation Names ===\n";
  CudaGraphSCF graph;
  graph.BeginCapture();
  graph.Record("build_rho", [](){});
  graph.Record("poisson", [](){});
  graph.Record("xc_eval", [](){});
  graph.EndCapture();

  auto names = graph.OperationNames();
  std::cout << "  Operations: ";
  for (const auto& n : names) std::cout << n << " ";
  std::cout << "\n";
  if (names.size() != 3) return Fail("Should have 3 names");
  if (names[0] != "build_rho") return Fail("First op name wrong");
  std::cout << "  PASS\n";
  return 0;
}

int TestSpeedupEstimate() {
  std::cout << "\n=== E6: Speedup Estimate ===\n";
  double speedup = CudaGraphSCF::EstimateSpeedup(100);
  std::cout << "  Estimated speedup for 100 ops: " << speedup << "x\n";
  if (speedup < 1.0) return Fail("Speedup should be > 1");
  if (speedup > 100.0) return Fail("Speedup unrealistically high");
  std::cout << "  PASS\n";
  return 0;
}

int TestReplayWithoutCapture() {
  std::cout << "\n=== E6: Replay Without Capture ===\n";
  CudaGraphSCF graph;
  int n = graph.Replay();
  if (n != 0) return Fail("Should execute 0 ops without capture");
  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== E6: CUDA Graph SCF Tests ===\n";
  int failures = 0;
  failures += TestCaptureAndReplay();
  failures += TestNames();
  failures += TestSpeedupEstimate();
  failures += TestReplayWithoutCapture();
  if (failures == 0) std::cout << "\nALL CUDA GRAPH TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
