#pragma once

// twin_interval — does a whole set of entry states run a trace the same way?
//
// A twin guard admits a set of live-in states, and every state it admits must
// reach the region's exit computing what the region computed. Deciding that by
// running each state is fine for a handful of them and hopeless for a range, so
// this is the other oracle: propagate an *interval* per leaf through the
// flattened trace once, and check at every step that nothing the run depends on
// can differ.
//
// Two obligations, both checked here:
//
//   UB-freedom — every trapping operation stays safe for the whole input
//   interval: `+ - * <<` cannot leave the type's range, a divisor cannot
//   contain 0 (nor the INT_MIN / -1 pair), a shift amount stays in [0, N), an
//   index stays in bounds, and a `require` cannot fail.
//
//   Path agreement — every branch the trace recorded (twin_trace.hpp) is still
//   decided the way the profiled run decided it. The body has no branches left,
//   so a state that would have gone the other way would silently compute the
//   wrong thing.
//
// Intervals over-approximate, so a verdict of `ok` covers every state in the
// input box: one pass proves the whole set. The converse does not hold — a
// verdict of "not ok" means *not proven*, never "unsafe" — so a caller may only
// use it to decline widening, never to conclude a state is bad.
//
// Precision is deliberately modest in this first version. Integer scalars are
// tracked exactly through `+ - * ~ <<` and casts; bitwise and shift results
// widen to the type's full range unless their operands are known constants;
// floats, vectors, pointers and anything reached through memory are unknown
// from the start. Unknown is sound but useless: a check on an unknown value
// cannot be proven, so such regions simply do not widen and keep the guard they
// have today. Sharpening any of this is a local change to one transfer
// function.
//
// Solver-free by construction — this is arithmetic on bounds.

#include <cstdint>
#include <string>
#include <unordered_map>

#include "ast/ast.hpp"
#include "reify/twin_mini.hpp"
#include "reify/twin_trace.hpp"

namespace refractir::reify {

  // An inclusive range of signed values. `unknown` is the full range of the
  // value's type and carries no information.
  struct Interval {
    std::int64_t lo = 0;
    std::int64_t hi = 0;
    bool unknown = false;

    bool isConst() const { return !unknown && lo == hi; }
  };

  // The entry state to check a trace against: one interval per scalar leaf,
  // keyed as twin_mini's `leafKey` names them (`%a`, `%a[1]`, `%s.f0`). A leaf
  // the map does not mention is unknown, so an empty environment is the
  // "nothing is pinned" extreme.
  using IntervalEnv = std::unordered_map<std::string, Interval>;

  struct IntervalVerdict {
    bool ok = false;
    // Why the trace could not be proven — the failing operation and what it
    // needed. Empty when ok.
    std::string reason;
  };

  // Check `body` over every state in `entry`. `fn` supplies the declared types
  // of the locals the body touches (widths bound the arithmetic) and `structs`
  // resolves field types.
  IntervalVerdict checkTrace(
      const FunDecl &fn, const StructMap &structs, const TraceBody &body, const IntervalEnv &entry
  );

} // namespace refractir::reify
