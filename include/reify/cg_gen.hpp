#pragma once

// cg_gen — random DAG call-graph generator for rylink.
//
// Nodes are numbered [0, N). Node 0 is the entry. Edges (i, j) require
// i < j so the graph is necessarily acyclic (no recursion). Edge density
// is controlled by hyperparameters::kPEdge and capped by kMaxOutDegree.

#include <cstddef>
#include <random>
#include <vector>

namespace refractir::reify {

  struct RyCG {
    int nNodes = 0;
    // Adjacency list: outEdges[i] = nodes that node i calls. Always sorted
    // ascending, all entries > i.
    std::vector<std::vector<int>> outEdges;

    [[nodiscard]] constexpr int entry() const noexcept { return 0; }
  };

  struct CGGenConfig {
    int nNodes = 4;
    double pEdge = 0.45;
    int maxOutDegree = 3;
  };

  [[nodiscard]] RyCG genCallGraph(std::mt19937 &rng, const CGGenConfig &cfg);

} // namespace refractir::reify
