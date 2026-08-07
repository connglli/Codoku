#pragma once

// antiopt — rewriting generated code so it stops looking generated.
//
// Every reify tool has the same problem from a different direction. A twin
// body is the region's own executed trace, so it reads as a copy of it. A
// rysmith leaf is assembled from one statement generator, so its functions
// read alike. In both cases the code is *correct* and *recognizable*, and the
// fix is the same: rewrite it by identities that preserve what it computes
// while changing how it says it.
//
// Hence the name. A compiler rewrites toward a normal form; every rule here
// moves away from one. That direction is the whole point — code that
// normalizes back to what it came from is code an optimizer sees through, and
// a twin that normalizes to its region is not a twin at all.
//
// The engine is deliberately free of rytwin: it takes a statement list, the
// declarations it may add to, and a predicate that says whether a rewritten
// body is still acceptable. rytwin supplies the state-set pass over its guard
// box; a generator with no such obligation can pass one that always accepts
// and rely on the rules being identities.
//
// This is the peephole tier of reify/rewrite.hpp — see that header for the
// R1-R3 contract every rule must satisfy. Two places this engine differs from
// call realization:
//
//   Rules INSERT statements, so a position is invalidated by any application.
//   The engine re-scans each round rather than holding a site list, and it
//   shifts anything that refers to a statement by index — a recorded branch,
//   say — by however much the splice moved it.
//
//   Composition is not implied by rule-level soundness (R3). Two rewrites that
//   are each value-preserving can still put an intermediate outside its type,
//   which is UB. So `accept` is consulted after every application and a body it
//   refuses is rolled back. That is why rules may be added freely: a wrong one
//   costs a rewrite, never correctness.

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "ast/ast.hpp"
#include "reify/name_alloc.hpp"
#include "reify/twin_mini.hpp"
#include "reify/twin_trace.hpp"

namespace refractir::reify {

  // Which catalog a rule belongs to. Weights are per family, and the tally a
  // run reports is per family, because that is the granularity at which the
  // catalog is worth tuning.
  enum class RuleFamily { Peephole, Mba, Structure, Licensed, Control };

  // Whether applying a rule can introduce an operation that traps. Tier0 rules
  // build only from `& | ^ ~ >> >>> cmp select` and copies, so they are safe
  // wherever they match; Tier1 rules introduce `+ - * <<` or `/ %`, which the
  // state-set pass has to clear over the whole box afterwards.
  enum class TrapTier { Tier0, Tier1 };

  // Where a rule fires: one statement of the body, addressed by index. Rules
  // that need to see more than one statement (reordering a pair, say) take the
  // body and read around this index themselves.
  struct RulePos {
    std::size_t stmt = 0;
  };

  // Every local the engine introduces starts with this, so a caller can tell
  // its own scratch from the program's state — the two arms of a twin, for
  // instance, differ on exactly these and on nothing that outlives the block.
  inline constexpr const char *kAntiOptLocalPrefix = "%__ao";

  // What is known about the values a body carries. Most rules are identities
  // and need none of this; the ones that are identities *only under a
  // condition* — a mask that is a no-op because the value is small enough, a
  // literal spelled as a read of something pinned to it — need someone to
  // stand behind the condition. That someone is the caller: rytwin answers
  // from the same state-set pass that certifies its guard, and a caller with
  // nothing to say passes nothing, which simply stops those rules firing.
  //
  // A licence granted on stale facts is not a licence, so the engine refreshes
  // before every attempt rather than once per round.
  struct ValueRange {
    std::int64_t lo = 0;
    std::int64_t hi = 0;
  };

  class AntiOptFacts {
  public:
    virtual ~AntiOptFacts() = default;

    // Recompute against the body as it now stands.
    virtual void refresh(const std::vector<Instr> &stmts) = 0;

    // What `local` may hold just before statement `at`, or nullopt when
    // nothing is known — which is also the answer past the end of what the
    // caller could follow.
    virtual std::optional<ValueRange>
    rangeBefore(std::size_t at, const std::string &local) const = 0;

    // Is this local one the caller's obligation says nothing about at all? A
    // body that reads one of those reads as a check on state the guard never
    // constrains, which is exactly what makes the dependency convincing.
    virtual bool isFree(const std::string &local) const = 0;
  };

  // What a rule may read while deciding and while rewriting.
  struct AntiOptContext {
    const FunDecl &fn;                    // the function the body belongs to
    const StructMap &structs;             //
    const std::vector<PathCheck> &checks; // branches the body assumes, if any
    NameAllocator &names;                 //
    std::vector<LetDecl> &lets;           // where fresh locals are declared
    std::mt19937 &rng;                    //
    AntiOptFacts *facts = nullptr;        // what is known, if anyone can say
  };

  class AntiOptRule {
  public:
    virtual ~AntiOptRule() = default;
    virtual const char *name() const = 0;
    virtual RuleFamily family() const = 0;
    virtual TrapTier tier() const = 0;

    // How many statements starting at the position the rule consumes. Almost
    // always one; a rule that reorders a pair takes two.
    virtual std::size_t width() const { return 1; }

    // Can this rule fire here? Must not mutate anything (R1).
    virtual bool
    matches(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx) const = 0;

    // The statements replacing the `width()` starting at `pos`. Empty declines.
    virtual std::vector<Instr>
    apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const = 0;

    // How this rule's claim is checked. `body` is RefractIR source for the
    // statements the rule fires on, written over the check's own locals:
    //
    //   %x %y   the operands the check sweeps over every i8 value
    //   %d %e   destinations
    //   %c      an i1, for the rules over comparisons
    //   %k      a cell to pin with `assume`
    //
    // The check parses it, applies the rule at `at`, and runs both versions
    // through the *interpreter*, comparing the resulting state and the UB
    // outcome. The interpreter is the authority on what a statement does;
    // restating the arithmetic in C++ here only ever checked the restatement,
    // and could not check a rule with no closed form at all.
    //
    // A rule licensed by facts says what it needs known in `assume`: each
    // entry both answers the fact and confines the sweep, so a rule that pins
    // %k to 7 is checked with %k held at 7 and the facts saying so. `free`
    // names locals the check reports as free — swept over every value all the
    // same, since that is what the claim comes to.
    struct SelfTest {
      std::string body;
      // Anything the example needs declared above the function — an
      // `intrinsic` line, for a rule that rewrites a call.
      std::string decls;
      std::size_t at = 0;
      std::vector<std::pair<std::string, ValueRange>> assume;
      std::vector<std::string> free;
    };

    virtual std::optional<SelfTest> selfTest() const { return std::nullopt; }
  };

  // How a body changed, so a sweep can be tuned by what actually survives the
  // acceptance check rather than by what was attempted. The per-rule tally is
  // the granularity that matters: a family whose rules all roll back is a
  // family whose weight is wasted, and only the names say which.
  struct AntiOptReport {
    std::size_t applied = 0;
    std::size_t rolledBack = 0;
    std::vector<std::pair<std::string, std::size_t>> byRule; // kept, first use first
    // The caller refused the body before any rewriting, so nothing could be
    // judged and only rules that cannot introduce a trap were offered.
    bool trapFreeOnly = false;
  };

  // The per-rule tally as "xor-twice x3, sub-as-add x1", or empty when nothing
  // was kept.
  std::string describeRules(const AntiOptReport &rep);

  // Is a rewritten body still acceptable? Consulted after every application;
  // `false` undoes it. A caller with a correctness obligation puts it here —
  // for rytwin, that the state-set pass still proves the body over every state
  // its guard admits. An empty predicate says the caller has no oracle at all,
  // and then only the rules that need none are offered.
  using AntiOptAccept = std::function<bool(const std::vector<Instr> &)>;

  // Rewrite `stmts` in place, shifting `checks` to follow the statements they
  // refer to. `attempts` bounds the work: the engine re-scans and applies at
  // most one rewrite per attempt, so it is also the most statements a body can
  // gain. What that should be is the caller's business and not one number for
  // everyone — a twin body is one region's trace and can take a lot of
  // rewriting, while a generator applies this to every block of every function
  // and pays for it in the size of the whole program.
  AntiOptReport antiOptimize(
      std::vector<Instr> &stmts, std::vector<PathCheck> &checks, AntiOptContext &ctx,
      const AntiOptAccept &accept, std::size_t attempts
  );

  // Self-check of the catalog: every rule that declares a `SelfTest` is
  // applied to its own example and both versions are run through the
  // interpreter over every i8 value of the operands the example sweeps. A
  // Tier0 rule must never turn a UB-free run into a trapping one; a Tier1
  // rule may, and is compared only where both runs are UB-free — that is the
  // permission the caller's proof buys it. Returns the names of the rules
  // checked, or nullopt on the first disagreement (described in `failure`).
  std::optional<std::vector<std::string>> selfTestRules(std::string &failure);

} // namespace refractir::reify
