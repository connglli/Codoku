#pragma once

// TwinTransform — the equivalence-preserving region-twin rewrite behind rytwin.
//
// For the profiled entry function, TwinTransform walks the executed trace (the
// StateProfile in TransformContext). For each eligible on-path region R, with
// probability `pTwin`, it grafts an equivalent alternative:
//
//     ^X:         br call @__twg_<fn>_<X>(<state>) != 0, ^X__twin, ^X__orig
//     ^X__twin:   R'                              ->  br ^<exit>
//     ^X__orig:   R's entry block, unchanged      ->  its own terminator
//
// `^X` is the region entry's own label, taken over by the guard so that every
// predecessor still branches to `^X` and no edge needs rewriting; `__twin` /
// `__orig` are literal suffixes, which keeps the labels distinct when one
// function holds several twin sites. `^<exit>` is not generated at all — it is
// the block the region left to, already in the function under its own name.
// There is no merge block: the twin jumps straight to that exit, and the orig
// arm keeps the entry's own terminator, so the two arms rejoin at the exit.
//
// The unit is the maximal single-entry region rooted at an executed block:
// every later block the entry *dominates* on the executed path, up to the
// first block it does not (or the function's return). A whole loop collapses
// when its header is the region entry; a straight-line run collapses to one
// block, which is the degenerate case rather than a separate mode.
//
// `R'` is the region's own executed trace, flattened (see reify/twin_trace.hpp):
// the statements the profiled run performed, concatenated in execution order
// with the branches dropped and each loop iteration laid out in turn. It
// computes what the region computes for every state that follows the same path
// UB-free — established by construction rather than by search, so no solver is
// involved anywhere in rytwin.
//
// The guard is a per-site generated function `@__twg_<fn>_<label> : i1`. It
// consumes the ENTIRE definitely-initialized state at the region's entry as a
// conjunction of per-leaf equalities against `s`, the state the region sees on
// the profiled input. That conjunction is total (no UB) and collision-free, so
// the twin arm runs on the profiled input and the original runs on every other
// state: p1(i) == p2(i) for every input i. Scalar roots cross into the guard by
// value, vector roots per-lane, and aggregate roots by address (`ptr [N] T` /
// `ptr @S` parameters navigated with ptrindex/ptrfield + load — all in-bounds
// by construction, so total on every input).
//
// Note the asymmetry: soundness needs only guard-set ⊇ read-set(R) — which
// planRegion guarantees by rejecting any region whose reads are not guardable —
// and the trace body is correct on a whole path domain, so pinning every leaf
// to one state is far stricter than either requires. The guard, not the body,
// is now the narrow half of the design.
//
// Candidate regions may contain memory operations (load/store/addr/ptr
// navigation). Eligibility is decided from the bit-exact frame-state diff, in
// which store-through-pointer effects surface as diffs of the pointee root;
// the body itself needs no reconstruction, since replaying the trace re-derives
// pointer leaves by running the same navigation the region ran. Regions with
// non-intrinsic calls stay ineligible — a callee handed a pointer into an outer
// frame could mutate state the frame diff does not see.

#include <functional>
#include <memory>
#include <vector>

#include "reify/hyperparameters.hpp"
#include "reify/transform.hpp"

namespace refractir::reify {

  // Interestingness features of one candidate region, handed to a
  // SelectionPolicy. A one-block region is the degenerate case, so these
  // describe short and long regions alike.
  struct CandidateInfo {
    long loopItersCollapsed = 0; // repeated blocks in the window (a loop swallowed)
    long distinctBlocks = 0;     // region size
    long changedLeaves = 0;      // leaves the region's net effect changes
    long fanIn = 0;              // predecessors of the region entry
  };

  // A selection policy maps the candidate regions found across one program to
  // a per-region twin probability in [0,1] (index-aligned with its input). It
  // receives all candidates at once so it can normalize across the program;
  // TwinTransform then twins each region by an independent draw against its
  // probability, resolving overlaps in trace order. The policy owns its own
  // parameters, so the transform stays agnostic to how the probability is
  // chosen.
  using SelectionPolicy = std::function<std::vector<double>(const std::vector<CandidateInfo> &)>;

  // Uniform policy: every region gets probability `pTwin` (a plain coin).
  SelectionPolicy uniformPolicy(double pTwin);

  // Interestingness policy: score = 1000·itersCollapsed + 10·blocks +
  // 5·changedLeaves + fanIn, normalized to [0,1] program-wide, then
  // `p = pTwin ^ exp((0.5 - norm) / temp)` — monotone in the score, `1` at
  // `pTwin=1`, `0` at `pTwin=0`, and → the uniform `pTwin` coin as
  // `temp → ∞`. `temp` must be > 0.
  SelectionPolicy interestingPolicy(double pTwin, double temp = rytwin::hp::kInterestingTemp);

  // Build the twin transform. `select` scores candidate regions into twin
  // probabilities (see SelectionPolicy).
  std::unique_ptr<Transform> makeTwinTransform(SelectionPolicy select);

} // namespace refractir::reify
