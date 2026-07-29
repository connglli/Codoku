#pragma once

// twin_probe — run a region from an arbitrary state.
//
// rytwin's guard admits a set of live-in states, and every state it admits
// must be one the twin reproduces correctly. Deciding membership does not
// need a solver: a region behaves a certain way on a state exactly when
// *running* it on that state says so. RegionProbe is that oracle — it
// executes the region alone, from a caller-supplied state, and reports
// whether the run reached an exit without UB, which exit it took, and the
// resulting state.
//
// The harness is the region and nothing else: its blocks are lifted into a
// fresh function whose lets are the roots (seeded from the probed state),
// and every edge leaving the region is redirected to a landing block that
// returns. Execution therefore stops exactly at the region boundary, so UB
// raised by code *after* the region can never be blamed on it.
//
// The harness is built once and re-seeded per probe: growing a guard walks
// hundreds of states through the same region, and only the root
// initializers differ between them.
//
// Solver-free by construction — this is an interpreter run.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ast/ast.hpp"
#include "reify/state_profile.hpp"
#include "reify/twin_mini.hpp"

namespace refractir::reify {

  struct ProbeResult {
    // The region ran to an exit with no UB. False also covers a run that
    // exceeded the step cap, so a caller may treat `!ok` uniformly as
    // "not a usable state".
    bool ok = false;
    // Hash of the executed block-label sequence and the exit taken. Two
    // probes share a pathId iff they took the same path, so a caller can
    // group states by path without recording the sequence.
    std::uint64_t pathId = 0;
    // The out-of-region block control left to, or empty when the region
    // returned instead of branching out.
    std::string exitLabel;
    // Every root's value at the exit, in the profile's (name-sorted) order.
    std::vector<std::pair<std::string, StateValue>> effect;
  };

  // Block-step budget for one probe. A region's own blocks are few, but a
  // probed state can drive a loop far longer than the profiled one did —
  // or forever — so the run is always bounded.
  inline constexpr std::uint64_t kProbeStepCap = 4096;

  class RegionProbe {
  public:
    // Build a harness for the region `regionLabels` (entry first) of
    // `host`'s function `funcName`. `roots` supplies the shape of the
    // live-in state; the values it carries are overwritten by each run.
    RegionProbe(
        const Program &host, const std::string &funcName,
        const std::vector<std::string> &regionLabels, const std::vector<MiniRoot> &roots
    );

    // False when the region could not be lifted into a self-contained
    // harness (missing function or block, or a shape the checkers reject
    // in isolation). Every run on an invalid probe returns `ok = false`.
    bool valid() const { return valid_; }

    // Execute the region from the state in `roots`, which must match the
    // roots the harness was built with, in the same order.
    ProbeResult run(const std::vector<MiniRoot> &roots);

  private:
    Program harness_;
    StructMap structs_;
    std::size_t nRoots_ = 0;
    bool valid_ = false;
  };

} // namespace refractir::reify
