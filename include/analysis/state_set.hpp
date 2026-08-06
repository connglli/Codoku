#pragma once

// State-set analysis over a straight-line body: does every state in a set run
// it the same way?
//
// The question this answers is not about one state but about a whole set of
// them. Running the body on each is fine for a handful and hopeless for a
// range, so instead an interval per scalar leaf is propagated through the body
// once, and at every step the analysis checks that nothing the run depends on
// can differ across the set. Intervals are the domain, not the point: what is
// being proven is a property of the whole set, and sharpening the domain would
// not change the question. Two obligations, both checked:
//
//   UB-freedom — every trapping operation stays safe for the whole input
//   interval: `+ - * <<` cannot leave the type's range, a divisor cannot
//   contain 0 (nor the INT_MIN / -1 pair), a shift amount stays in [0, N), an
//   index stays in bounds, and a `require` cannot fail.
//
//   Branch agreement — every branch the caller declares must still be decided
//   the way the caller says. A straight-line body has no branches left, so a
//   state that would have gone the other way would silently compute something
//   else.
//
// Intervals over-approximate, so a verdict of `ok` covers every state in the
// input box: one pass proves the whole set. The converse does not hold. A
// verdict of "not ok" means *not proven*, never "unsafe", so a caller may use
// it to decline to widen a set but never to conclude that a state is bad.
//
// Precision is deliberately modest, but nothing is left *unbounded* that has a
// bound. Integer scalars are tracked exactly through `+ - * ~ <<` and casts
// that keep the value; a cast that does not is still inside the destination
// type's range, since narrowing truncates rather than trapping. Shifts by a
// known amount, `x & m` for a non-negative m, `x | y` and `x ^ y` over
// non-negative ranges, and `/` and `%` (which cannot grow a value) all carry
// bounds. Floats, pointers, intrinsic results and anything reached through
// memory are unknown from the start. Unknown is sound but useless: a check on
// an unknown value cannot be proven. Sharpening any of this is a local change
// to one transfer function.
//
// Vectors need no value of their own. A vector statement is N scalar
// statements, one per lane, so the analysis walks it that way and every lane is
// an ordinary tracked cell. A lane written through an index that cannot be
// pinned could have landed anywhere, and the whole vector is forgotten.
//
// Solver-free by construction: this is arithmetic on bounds.

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "analysis/type_utils.hpp"
#include "ast/ast.hpp"

namespace refractir {

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
    // that fails names them, so a search widening the set knows which leaves
    // to freeze rather than freezing all of them. Only integer leaves get a
    // bit: floats and pointers are pinned, so they can never be the reason a
    // widened set fails. Callers leave this alone — the analysis assigns the
    // bits.
    std::uint64_t deps = 0;

    bool isConst() const { return !unknown && lo == hi; }
  };

  // The canonical name of one scalar leaf: `%a`, `%a[1]`, `%s.f0`. Every
  // environment below is keyed by it, so a caller's idea of a leaf and the
  // analysis's agree by construction rather than by both spelling it the same
  // way. The LValue form returns nullopt when an index is not a literal, since
  // such a leaf has no one name.
  [[nodiscard]] std::string leafKey(const std::string &root, const std::vector<Access> &path);
  [[nodiscard]] std::optional<std::string> leafKey(const LValue &lv);

  // One interval per scalar leaf, keyed by leafKey. A leaf the map does not
  // mention is unknown, so an empty environment is the "nothing is pinned"
  // extreme.
  using IntervalEnv = std::unordered_map<std::string, Interval>;

  // Where each pointer leaf points at entry, keyed the same way. An empty
  // optional is the null pointer; a pointer the caller cannot resolve is left
  // out, and everything reached through it stays unknown. Without this a
  // pointer set up before the body — the usual case — would have no known
  // target, and every load through it would be a dead end.
  using PtrEnv = std::unordered_map<std::string, std::optional<LValue>>;

  // Floats are carried as exact values rather than ranges. A caller pins every
  // float leaf it mentions, since bounding a float needs rounding-aware
  // arithmetic and buys nothing while they stay pinned, and a pinned float is a
  // constant — so the domain that fits them is equality, not an interval.
  using FloatEnv = std::unordered_map<std::string, double>;

  // What each leaf may hold at entry, per kind. A leaf missing from all three
  // is simply unknown.
  struct EntryState {
    IntervalEnv ints;
    FloatEnv floats;
    PtrEnv ptrs;
  };

  struct StateSetVerdict {
    bool ok = false;
    // Why the body could not be proven — the failing operation and what it
    // needed. Empty when ok.
    std::string reason;
    // The entry leaves the failing check depended on. Empty when the failure
    // involved nothing the caller can narrow (an untracked value, say), which
    // means narrowing the set cannot help.
    std::vector<std::string> blame;
  };

  // A branch the caller has already decided, and the way it goes. `afterStmt`
  // is the number of statements that precede it in the body, so the condition
  // is read in the state the body has reached at that point.
  //
  // `cond` is borrowed. The caller owns the condition and must keep it alive
  // for the call.
  struct BranchObligation {
    std::size_t afterStmt = 0;
    const Cond *cond = nullptr;
    bool taken = false;

    // The condition is read in the state reached after `afterStmt` statements,
    // which is the snapshot taken before that statement runs.
    std::size_t checkEnvIndex() const { return afterStmt; }
  };

  // What an intrinsic call computes, when every argument is a single value.
  // The analysis does not know — the interpreter does, and duplicating it here
  // is the mistake this separation exists to prevent — so a caller that can
  // answer supplies this, and one that cannot leaves every call unknown.
  // `nullopt` means "cannot say"; a caller reports UB by throwing, which the
  // analysis turns into a refusal.
  using IntrinsicFold = std::function<
      std::optional<std::int64_t>(const CallAtom &, const std::vector<std::int64_t> &)>;

  struct StateSetAnalysis {
    /**
     * Check `stmts` over every state in `entry`, requiring each obligation in
     * `branches` to be decided as it says.
     *
     * `fn` supplies the declared types of the locals the body touches, whose
     * widths bound the arithmetic, and `structs` resolves field types.
     */
    static StateSetVerdict check(
        const FunDecl &fn, const TypeUtils::StructTable &structs, const std::vector<Instr> &stmts,
        const std::vector<BranchObligation> &branches, const EntryState &entry,
        const IntrinsicFold *fold = nullptr
    );

    /**
     * What every local held *before* each statement, over every state in
     * `entry` — one environment per statement, in order.
     *
     * This is the same forward pass `check` makes, read for its annotations
     * rather than its verdict: a rewrite licensed by a value's range needs the
     * range at the point it fires. A body the pass cannot finish returns the
     * prefix it managed, so a caller reading past the end simply knows nothing.
     */
    static std::vector<IntervalEnv> snapshots(
        const FunDecl &fn, const TypeUtils::StructTable &structs, const std::vector<Instr> &stmts,
        const std::vector<BranchObligation> &branches, const EntryState &entry,
        const IntrinsicFold *fold = nullptr
    );

    /**
     * The widest range each integer leaf could possibly take, obtained by
     * pushing every check's requirement backward through the body: an addition
     * that must not overflow bounds its operands, a branch that must go one way
     * bounds what it compares, and so on back to the entry.
     *
     * These are upper bounds and nothing more. Narrowing is not exact — it
     * ignores how operands relate to one another — so a ceiling may contain
     * values that do not actually work, and a search must still prove what it
     * claims. What a ceiling is good for is not searching above it: a leaf whose
     * ceiling is a single value is pinned without a single pass, one whose
     * ceiling is its whole type is a free candidate, and everything else has a
     * bound to bisect under instead of a doubling sequence to guess at.
     */
    static std::unordered_map<std::string, Interval> ceilings(
        const FunDecl &fn, const TypeUtils::StructTable &structs, const std::vector<Instr> &stmts,
        const std::vector<BranchObligation> &branches, const EntryState &entry,
        const IntrinsicFold *fold = nullptr
    );
  };

} // namespace refractir
