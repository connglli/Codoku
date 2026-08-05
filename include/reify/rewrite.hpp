#pragma once

// Peephole rewrite tier (lower tier).
//
// A *rewrite* is a local, pattern-directed edit inside one function: find a
// position that matches, decide whether it may fire there, splice the
// replacement. This header owns the tier — its contract, its shared result
// type, and its candidate ordering — while each engine keeps its own rule
// interface, because what a rule matches against differs fundamentally
// between them:
//
//   CallRewriteRule (reify/call_realize.hpp) matches a site against a callee
//                   descriptor and a pinned realization index.
//   AntiOptRule     (reify/antiopt.hpp) matches a run of statements against
//                   the declarations it may add to and a predicate that says
//                   whether the rewritten body is still acceptable.
//
// The tier above is reify/transform.hpp: a `Transform` is a whole-program
// rewrite driven through a `TransformPipeline`. A Transform may own a set of
// rewrite rules and drive them (CallRealizeTransform, TwinTransform do), but
// the two tiers are separate — a rule never sees the Program, only the unit
// it edits.
//
// ---------------------------------------------------------------------------
// The contract every rewrite rule must satisfy
// ---------------------------------------------------------------------------
//
//   R1  Purity of matching. Deciding whether a rule applies never mutates
//       the unit, so an engine may probe candidates freely, in any order,
//       and discard the answer.
//
//   R2  Local soundness. `apply()` on its own preserves the meaning the
//       engine cares about, under the rule's own declared precondition —
//       the callee's solved value for call realization, the statement's
//       value on the guarded state box for twin disguise.
//
//   R3  Composition is NOT implied by R2, and the ENGINE owns it. Two
//       individually sound rewrites can be unsound stacked. This is the
//       tier's sharpest edge, and both engines have already been cut by it:
//
//         - Call realization: RefractIR expressions evaluate left to right
//           with no parentheses, so rewriting a literal `c` into
//           `f1() + (c - r1)` and then rewriting *that* literal again
//           yields `f1() + f2() + ...`, whose left-prefix sum can wrap even
//           though each rewrite is BV-sound on its own. The engine's answer
//           is a consumed-site set: one splice per site, for the lifetime of
//           the transform.
//
//         - Twin disguise: each identity is value-preserving, but stacking
//           two can introduce an intermediate that overflows on states the
//           original body handled. The engine's answer is to re-run the
//           interval pass over the whole body after each application and
//           roll back the ones that fail.
//
//       A new rule author is responsible for R1 and R2. A new *engine*
//       author must say, explicitly, how it discharges R3.

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

namespace refractir::reify {

  // Outcome of driving a rule set over one unit. `found` counts candidates
  // that matched, `applied` those actually spliced; the difference is rules
  // declining late (a type the rule cannot handle, a license that fails, a
  // rewrite rolled back), which is the number worth watching when a tool
  // suddenly stops producing rewrites.
  struct RewriteReport {
    int found = 0;
    int applied = 0;

    int declined() const { return found - applied; }
  };

  // Shuffle candidates into a random order and keep at most `cap` of them.
  //
  // Every engine in this tier wants the same thing: rules propose more
  // candidates than the engine intends to spend work on, and picking the
  // first few in discovery order would bias rewrites toward the top of the
  // function. Shuffling first makes the choice uniform; capping afterwards
  // bounds the work. `cap == 0` means no cap.
  //
  // Note for callers that report `found`: count the candidates *before*
  // capping, otherwise the report describes the budget rather than the code.
  template<typename T>
  void shuffleAndCap(std::vector<T> &cands, std::mt19937 &rng, std::size_t cap) {
    std::shuffle(cands.begin(), cands.end(), rng);
    if (cap != 0 && cands.size() > cap)
      cands.resize(cap);
  }

} // namespace refractir::reify
