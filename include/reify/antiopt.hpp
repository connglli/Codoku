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
// body is still acceptable. rytwin passes the interval pass over its guard
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
  // interval pass has to clear over the whole box afterwards.
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

  // Fresh names for a body being rewritten. Temporaries are needed constantly:
  // RefractIR admits at most one binary operator per atom, and the right
  // operand of `* / % & | ^ << >> >>>` must be an lvalue — so `%a << 3` has to
  // become `%k = 3; %a << %k`. Literal cells are pooled by value so a body does
  // not accumulate a dozen names for the same constant.
  class NameAllocator {
  public:
    explicit NameAllocator(std::string prefix) : prefix_(std::move(prefix)) {}

    // A fresh mutable local of `type`, declared into `lets`.
    std::string fresh(const TypePtr &type, std::vector<LetDecl> &lets);

    // A local holding `value` at `type`, reused when one already exists.
    std::string literal(std::int64_t value, const TypePtr &type, std::vector<LetDecl> &lets);

  private:
    std::string prefix_;
    std::size_t next_ = 0;
    std::vector<std::tuple<std::int64_t, std::string, std::string>> pool_; // value, type key, name
  };

  // What a rule may read while deciding and while rewriting.
  struct AntiOptContext {
    const FunDecl &fn;                    // the function the body belongs to
    const StructMap &structs;             //
    const std::vector<PathCheck> &checks; // branches the body assumes, if any
    NameAllocator &names;                 //
    std::vector<LetDecl> &lets;           // where fresh locals are declared
    std::mt19937 &rng;                    //
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

    // The arithmetic a value rule claims, as the two sides of its identity —
    // checked exhaustively at i8 so a wrong identity is caught once, at the
    // rule, rather than as a mysterious rollback later. `nullopt` from either
    // side means the operation is undefined for those operands and the pair is
    // skipped; `nullopt` from the rule means it has no closed form (a
    // structural rule), and the body-level re-check covers it instead.
    struct SelfTest {
      std::function<std::optional<std::int64_t>(std::int64_t, std::int64_t)> original;
      std::function<std::optional<std::int64_t>(std::int64_t, std::int64_t)> rewritten;
    };

    virtual std::optional<SelfTest> selfTest() const { return std::nullopt; }
  };

  // How a body changed, so a sweep can be tuned by what actually survives the
  // acceptance check rather than by what was attempted.
  struct AntiOptReport {
    std::size_t applied = 0;
    std::size_t rolledBack = 0;
  };

  // Is a rewritten body still acceptable? Consulted after every application;
  // `false` undoes it. A caller with a correctness obligation puts it here —
  // for rytwin, that the interval pass still proves the body over every state
  // its guard admits.
  using AntiOptAccept = std::function<bool(const std::vector<Instr> &)>;

  // Rewrite `stmts` in place, shifting `checks` to follow the statements they
  // refer to. A body `accept` refuses before any rewriting is left alone —
  // there would be nothing to judge later applications against.
  AntiOptReport antiOptimize(
      std::vector<Instr> &stmts, std::vector<PathCheck> &checks, AntiOptContext &ctx,
      const AntiOptAccept &accept
  );

  // Exhaustive self-check of every value rule at i8: for all 2^16 operand
  // pairs, the rewritten form computes what the original did. Returns the
  // number of rules checked, or nullopt on the first disagreement (described
  // in `failure`).
  std::optional<std::size_t> selfTestRules(std::string &failure);

} // namespace refractir::reify
