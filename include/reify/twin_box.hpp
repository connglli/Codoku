#pragma once

// The widest set of entry states a region's trace can be proven over.
//
// analysis/state_set.hpp answers the question for a given set: propagate an
// interval per leaf through the body and check that nothing the run depends on
// can differ. This header is the search on top of it — start from the single
// profiled state and widen until the proof stops going through.
//
// The search is generation policy, not analysis: how far to widen, how to
// split the remaining slack, how many passes to spend, and which leaves to
// free rather than bound are all tuning, and they read rytwin's
// hyperparameters and draw from its RNG.

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "analysis/state_set.hpp"
#include "ast/ast.hpp"
#include "reify/twin_trace.hpp"

namespace refractir::reify {

  // What a guard may say about one leaf.
  //
  //   Free   — the trace is provable with this leaf unknown, so the guard does
  //            not mention it at all. Only a proof can establish this; no
  //            amount of sampling could, since it covers every value.
  //   Ranged — provable over `range` but not beyond it: `lo <= x <= hi`.
  //   Pinned — not provable with the leaf moved at all: `x == v`.
  enum class LeafClass { Free, Ranged, Pinned };

  struct BoxLeaf {
    std::string key;
    LeafClass cls = LeafClass::Pinned;
    Interval range; // meaningful when Ranged
  };

  // The states a guard admits, one entry per integer leaf. Floats and pointers
  // are absent because they are always pinned: bounding a float needs
  // rounding-aware arithmetic and a pointer has no range to speak of.
  struct Box {
    std::vector<BoxLeaf> leaves;
    std::size_t passes = 0; // state-set passes spent computing it
  };

  // The branch obligations a flattened trace carries, in the form the analysis
  // takes. The conditions stay owned by `body`, which must outlive the result.
  [[nodiscard]] std::vector<BranchObligation> traceObligations(const TraceBody &body);

  // Compute the widest box the state-set analysis can prove for `body`, starting
  // from the profiled state. Leaves are freed where a proof allows, otherwise
  // widened in lockstep — every open leaf advances by the same relative step
  // each round, and a round the analysis refuses freezes only the leaves that
  // round's failing check depended on, so no leaf's width depends on the order
  // a loop visited it. `rng` settles how the remaining slack is split.
  Box computeBox(
      const FunDecl &fn, const TypeUtils::StructTable &structs, const TraceBody &body,
      const EntryState &entry, std::mt19937 &rng, const IntrinsicFold *fold = nullptr
  );

} // namespace refractir::reify
