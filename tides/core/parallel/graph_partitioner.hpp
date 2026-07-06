#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::parallel {

// Graph partitioner (T8.2): partition the tile graph into n_parts with
// imbalance <= 10% (per WP8 spec).
//
// For the CPU reference, we implement recursive coordinate bisection (RCB) —
// a simple, deterministic partitioner that achieves good balance on
// geometric graphs. The production path uses METIS (Scotch-METIS available as
// a runtime lib; dev headers not installed).
//
// Observable (T8.2): imbalance <= 10% on gauntlet graphs.

struct PartitionResult {
  std::vector<int> assignment;  // part index for each vertex [0, n_parts)
  double imbalance = 0.0;       // max |part_size - avg| / avg
  int n_parts = 0;
};

class GraphPartitioner {
 public:
  // Partition n_vertices vertices into n_parts using recursive bisection.
  //   coords:    vertex coordinates (n_vertices * 3, x,y,z per vertex)
  //   adjacency: adjacency list (vertex -> list of neighbor vertices)
  //   n_vertices: number of vertices
  //   n_parts:   number of partitions (must be a power of 2 for RCB)
  static PartitionResult Partition(
      const std::vector<double>& coords,
      const std::vector<std::vector<std::size_t>>& adjacency,
      std::size_t n_vertices, int n_parts) {
    PartitionResult res;
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
    std::vector<std::size_t> counts(n_parts, 0);
    for (std::size_t i = 0; i < n_vertices; ++i)
      if (res.assignment[i] < n_parts) counts[res.assignment[i]]++;
    const double avg = static_cast<double>(n_vertices) / n_parts;
    double max_imb = 0.0;
    for (int p = 0; p < n_parts; ++p) {
      const double imb = std::fabs(static_cast<double>(counts[p]) - avg) / avg;
      max_imb = std::max(max_imb, imb);
    }
    res.imbalance = max_imb;
    return res;
  }

 private:
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
