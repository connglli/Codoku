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
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

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
    // The declared width of the leaf this interval describes, when it is one
    // (0 otherwise). A leaf can never hold a value its own type cannot
    // represent, so a search widening it must stop there however much the
    // arithmetic downstream would tolerate.
    std::uint32_t bits = 0;
    // Which entry leaves this value was computed from, one bit each. A check
    // that fails names them, so a search widening the box knows which leaves
    // to freeze rather than freezing all of them. Only integer leaves get a
    // bit: floats and pointers are pinned, so they can never be the reason a
    // widened box fails. Callers leave this alone — the pass assigns the bits.
    std::uint64_t deps = 0;

    bool isConst() const { return !unknown && lo == hi; }
  };

  // The entry state to check a trace against: one interval per scalar leaf,
  // keyed as twin_mini's `leafKey` names them (`%a`, `%a[1]`, `%s.f0`). A leaf
  // the map does not mention is unknown, so an empty environment is the
  // "nothing is pinned" extreme.
  using IntervalEnv = std::unordered_map<std::string, Interval>;

  // Where each pointer leaf points at region entry, keyed the same way. An
  // empty optional is the null pointer; a pointer the caller cannot resolve is
  // left out, and everything reached through it stays unknown. Without this a
  // pointer set up *before* the region — the usual case — would have no known
  // target, and every load through it would be a dead end.
  using PtrEnv = std::unordered_map<std::string, std::optional<LValue>>;

  // Floats are carried as exact values rather than ranges. A guard pins every
  // float leaf it mentions (bounding a float needs rounding-aware arithmetic,
  // which buys nothing while they stay pinned), and a pinned float is a
  // constant — so the domain that fits them is equality, not an interval.
  using FloatEnv = std::unordered_map<std::string, double>;

  // The state a trace is checked against: what each leaf may hold at region
  // entry, per kind. A leaf missing from all three is simply unknown.
  struct EntryState {
    IntervalEnv ints;
    FloatEnv floats;
    PtrEnv ptrs;
  };

  struct IntervalVerdict {
    bool ok = false;
    // Why the trace could not be proven — the failing operation and what it
    // needed. Empty when ok.
    std::string reason;
    // The entry leaves the failing check depended on. Empty when the failure
    // involved nothing the caller can widen (an untracked value, say), which
    // means narrowing the box cannot help.
    std::vector<std::string> blame;
  };

  // Check `body` over every state in `entry`. `fn` supplies the declared types
  // of the locals the body touches (widths bound the arithmetic) and `structs`
  // resolves field types.
  IntervalVerdict checkTrace(
      const FunDecl &fn, const StructMap &structs, const TraceBody &body, const EntryState &entry
  );

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
    std::size_t passes = 0; // interval passes spent computing it
  };

  // Compute the widest box the interval pass can prove for `body`, starting
  // from the profiled state. Leaves are freed where a proof allows, otherwise
  // widened in lockstep — every open leaf advances by the same relative step
  // each round, and a round the pass refuses freezes only the leaves that
  // round's failing check depended on, so no leaf's width depends on the order
  // a loop visited it. `rng` settles how the remaining slack is split.
  Box computeBox(
      const FunDecl &fn, const StructMap &structs, const TraceBody &body, const EntryState &entry,
      std::mt19937 &rng
  );

} // namespace refractir::reify
