#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#ifdef TIDES_HAVE_METIS
extern "C" {
#include <metis.h>
}
#endif

namespace tides::parallel {

// Graph partitioner (T8.2): partition the tile graph into n_parts with
// imbalance <= 10% (per WP8 spec).
//
// Two backends:
//   1. RCB (recursive coordinate bisection) — always available, no external
//      dependency.  Deterministic, good balance on geometric graphs.
//   2. METIS — high-quality multilevel k-way partitioner.  Linked when
//      TIDES_HAVE_METIS is defined at compile time (CMake finds libmetis).
//
// Observable (T8.2): imbalance <= 10% on gauntlet graphs.

// Partitioning algorithm selection.
enum class PartitionMethod {
  RCB,    ///< Recursive coordinate bisection (always available).
  METIS,  ///< METIS multilevel k-way (requires TIDES_HAVE_METIS).
};

struct PartitionResult {
  std::vector<int> assignment;  // part index for each vertex [0, n_parts)
  double imbalance = 0.0;       // max |part_size - avg| / avg
  int n_parts = 0;
  int edge_cut = 0;             // number of cut edges (METIS only; 0 for RCB)
  PartitionMethod method = PartitionMethod::RCB;
};

class GraphPartitioner {
 public:
  // Partition n_vertices vertices into n_parts.
  //   coords:    vertex coordinates (n_vertices * 3, x,y,z per vertex)
  //   adjacency: adjacency list (vertex -> list of neighbor vertices)
  //   n_vertices: number of vertices
  //   n_parts:   number of partitions
  //   method:    RCB (default) or METIS (silently falls back to RCB if
  //              METIS is not compiled in)
  static PartitionResult Partition(
      const std::vector<double>& coords,
      const std::vector<std::vector<std::size_t>>& adjacency,
      std::size_t n_vertices, int n_parts,
      PartitionMethod method = PartitionMethod::RCB) {
    if (method == PartitionMethod::METIS) {
#ifdef TIDES_HAVE_METIS
      auto res = PartitionMetis(adjacency, n_vertices, n_parts);
      if (res.n_parts > 0) return res;  // METIS succeeded
      // Fall through to RCB on METIS failure.
#endif
    }
    return PartitionRCB(coords, adjacency, n_vertices, n_parts);
  }

 private:
  // --- RCB backend (always available) ---------------------------------------

  static PartitionResult PartitionRCB(
      const std::vector<double>& coords,
      const std::vector<std::vector<std::size_t>>& adjacency,
      std::size_t n_vertices, int n_parts) {
    PartitionResult res;
    res.method = PartitionMethod::RCB;
    res.assignment.assign(n_vertices, 0);
    res.n_parts = n_parts;
    if (n_vertices == 0 || n_parts <= 0) return res;
    if (n_parts == 1) { res.imbalance = 0.0; return res; }

    // Recursive coordinate bisection along the longest axis.
    std::vector<std::size_t> indices(n_vertices);
    for (std::size_t i = 0; i < n_vertices; ++i) indices[i] = i;

    int actual_parts = 1;
    while (actual_parts < n_parts) actual_parts *= 2;

    BisectRecursive(coords, indices, 0, actual_parts, res.assignment);

    // Compute imbalance.
    res.imbalance = ComputeImbalance(res.assignment, n_vertices, n_parts);
    res.edge_cut = CountCutEdges(res.assignment, adjacency);
    return res;
  }

#ifdef TIDES_HAVE_METIS
  // --- METIS backend --------------------------------------------------------

  static PartitionResult PartitionMetis(
      const std::vector<std::vector<std::size_t>>& adjacency,
      std::size_t n_vertices, int n_parts) {
    // Use METIS types (idx_t = int32_t, real_t = float on this system).
    using idx_t = int;

    PartitionResult res;
    res.method = PartitionMethod::METIS;
    res.assignment.assign(n_vertices, 0);
    res.n_parts = n_parts;
    if (n_vertices == 0 || n_parts <= 0) return res;
    if (n_parts == 1) {
      res.imbalance = 0.0;
      return res;
    }

    // Build CSR adjacency (xadj/adjncy) from the adjacency list.
    std::vector<idx_t> xadj(n_vertices + 1, 0);
    std::vector<idx_t> adjncy;
    adjncy.reserve(n_vertices * 6);  // typical 6-neighbour grid
    for (std::size_t v = 0; v < n_vertices; ++v) {
      xadj[v + 1] = xadj[v] + static_cast<idx_t>(adjacency[v].size());
      for (std::size_t j = 0; j < adjacency[v].size(); ++j)
        adjncy.push_back(static_cast<idx_t>(adjacency[v][j]));
    }

    // METIS parameters.
    idx_t nvtxs = static_cast<idx_t>(n_vertices);
    idx_t ncon = 1;              // 1 constraint (vertex weight)
    idx_t* vwgt = nullptr;       // no vertex weights (uniform)
    idx_t* vsize = nullptr;      // no vertex sizes
    idx_t* adjwgt = nullptr;      // no edge weights (uniform)
    idx_t nparts = static_cast<idx_t>(n_parts);

    // Target partition weights: equal.
    std::vector<float> tpwgts(n_parts, 1.0f / static_cast<float>(n_parts));

    // Imbalance tolerance (1.05 = 5% allowed, stricter than the 10% WP8 spec
    // to ensure we stay well within bounds even for non-power-of-2 partition counts).
    std::vector<float> ubvec(ncon, 1.05f);

    // Options.
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;  // C-style, 0-based
    options[METIS_OPTION_SEED] = 42;      // deterministic
    options[METIS_OPTION_CONTIG] = 1;      // produce contiguous partitions

    idx_t edgecut = 0;
    std::vector<idx_t> part(n_vertices, 0);

    // Use k-way partitioning for n_parts > 2, recursive for 2.
    int status;
    if (n_parts == 2) {
      status = METIS_PartGraphRecursive(
          &nvtxs, &ncon, xadj.data(), adjncy.data(),
          vwgt, vsize, adjwgt, &nparts,
          tpwgts.data(), ubvec.data(), options,
          &edgecut, part.data());
    } else {
      status = METIS_PartGraphKway(
          &nvtxs, &ncon, xadj.data(), adjncy.data(),
          vwgt, vsize, adjwgt, &nparts,
          tpwgts.data(), ubvec.data(), options,
          &edgecut, part.data());
    }

    if (status != METIS_OK) {
      // METIS failed — signal failure by returning empty result.
      res.n_parts = 0;
      return res;
    }

    // Copy results.
    for (std::size_t v = 0; v < n_vertices; ++v)
      res.assignment[v] = static_cast<int>(part[v]);
    res.edge_cut = static_cast<int>(edgecut);
    res.imbalance = ComputeImbalance(res.assignment, n_vertices, n_parts);
    return res;
  }
#endif  // TIDES_HAVE_METIS

  // --- Shared helpers --------------------------------------------------------

  static double ComputeImbalance(const std::vector<int>& assignment,
                                  std::size_t n_vertices, int n_parts) {
    std::vector<std::size_t> counts(n_parts, 0);
    for (std::size_t i = 0; i < n_vertices; ++i)
      if (assignment[i] >= 0 && assignment[i] < n_parts)
        counts[assignment[i]]++;
    const double avg = static_cast<double>(n_vertices) / n_parts;
    double max_imb = 0.0;
    for (int p = 0; p < n_parts; ++p) {
      const double imb = std::fabs(static_cast<double>(counts[p]) - avg) / avg;
      max_imb = std::max(max_imb, imb);
    }
    return max_imb;
  }

  // Count edges that cross partition boundaries (edge cut).
  static int CountCutEdges(const std::vector<int>& assignment,
                            const std::vector<std::vector<std::size_t>>& adjacency) {
    int cut = 0;
    for (std::size_t v = 0; v < assignment.size(); ++v) {
      for (std::size_t nb : adjacency[v]) {
        if (nb < assignment.size() && assignment[v] != assignment[nb])
          ++cut;
      }
    }
    // Each cut edge is counted twice (once from each endpoint).
    return cut / 2;
  }

  // Recursively bisect a set of vertices along the longest coordinate axis.
  static void BisectRecursive(
      const std::vector<double>& coords,
      std::vector<std::size_t>& indices, int part_id, int n_parts,
      std::vector<int>& assignment) {
    if (n_parts == 1) {
      for (auto idx : indices) assignment[idx] = part_id;
      return;
    }

    // Find the axis with the largest spread.
    double min_val[3] = {1e30, 1e30, 1e30};
    double max_val[3] = {-1e30, -1e30, -1e30};
    for (auto idx : indices) {
      for (int c = 0; c < 3; ++c) {
        const double v = coords[3 * idx + c];
        min_val[c] = std::min(min_val[c], v);
        max_val[c] = std::max(max_val[c], v);
      }
    }
    int best_axis = 0;
    double best_spread = 0;
    for (int c = 0; c < 3; ++c) {
      if (max_val[c] - min_val[c] > best_spread) {
        best_spread = max_val[c] - min_val[c];
        best_axis = c;
      }
    }

    // Sort indices along the best axis.
    std::sort(indices.begin(), indices.end(),
              [&](std::size_t a, std::size_t b) {
                return coords[3 * a + best_axis] < coords[3 * b + best_axis];
              });

    // Split at the median.
    const std::size_t mid = indices.size() / 2;
    std::vector<std::size_t> left(indices.begin(), indices.begin() + mid);
    std::vector<std::size_t> right(indices.begin() + mid, indices.end());

    BisectRecursive(coords, left, part_id, n_parts / 2, assignment);
    BisectRecursive(coords, right, part_id + n_parts / 2, n_parts / 2, assignment);
  }
};

}  // namespace tides::parallel
