#pragma once

// DataflowPolicy — how a spliced call gets its arguments.
//
// rylink realizes a call-graph edge by replacing a value the caller already
// computes with `call @callee(args) + bias`. The callee runs on the input its
// realization was solved for, so every argument has to evaluate to that input
// where the call sits. A literal does the job and tells the compiler
// everything; an expression over the caller's own variables makes it work for
// the same answer, which is the point of routing the value through a call at
// all.
//
// A policy builds one such argument. It is handed the state the profiled run
// passes through at the splice point — one pin set per visit — and owes an
// argument that evaluates to `target` in every one of them. The obligation is
// discharged by construction, never by search: a policy that cannot state the
// target at this width declines, and the next one is asked.
//
// The result is a statement run plus a final expression rather than a bare
// expression. RefractIR admits one binary operator per atom and requires an
// lvalue on the right of `* / % & | ^ << >> >>>` (spec §5.3), so anything past
// a single operation needs temporaries to stand in.

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "ast/ast.hpp"
#include "reify/name_alloc.hpp"
#include "reify/state_profile.hpp"

namespace refractir::reify {

  // Every local a policy introduces starts with this, so the rule that splices
  // calls can tell a policy's scratch from the program's own variables and
  // never offers one back as a rewrite site.
  inline constexpr const char *kDataflowLocalPrefix = "%__df";

  // One integer variable the caller holds where the argument is built, and the
  // value it holds there. Only integer scalars appear: they are what an
  // argument expression can be built out of.
  struct Pin {
    std::string name;
    std::int64_t value = 0;
    std::uint32_t bits = 0;
  };

  // The caller's state where the argument is built, one pin set per execution
  // of that point. A policy reading the state must satisfy every set, so a
  // variable that moves between visits is not something it can lean on. An
  // empty `visits` says nothing is known, which leaves a policy with only the
  // constructions that need no state.
  struct DataflowSite {
    std::vector<std::vector<Pin>> visits;
  };

  // An argument: the locals it needs declared, the statements that compute it,
  // and the expression the call takes.
  struct DataflowResult {
    std::vector<LetDecl> lets;
    std::vector<Instr> stmts;
    Expr value;
  };

  class DataflowPolicy {
  public:
    virtual ~DataflowPolicy() = default;
    virtual const char *name() const = 0;

    // An argument of `bits` width evaluating to `target` at every visit in
    // `site`, or nullopt to decline. `names` allocates any temporaries the
    // construction needs, declaring them into the result's own `lets`.
    [[nodiscard]] virtual std::optional<DataflowResult> build(
        const DataflowSite &site, std::uint32_t bits, std::int64_t target, NameAllocator &names,
        std::mt19937 &rng
    ) = 0;
  };

  // A literal, or `%v + bias` over a variable the profiled run pins. The floor
  // of the catalog: the literal costs nothing and always applies, so a site
  // reaching this policy always leaves with an argument.
  [[nodiscard]] std::unique_ptr<DataflowPolicy> makeBaselinePolicy();

  // `k ^ %v` over a variable the profiled run pins. Total at every width, so
  // it states targets a bias has no literal to reach; declines when nothing is
  // pinned.
  [[nodiscard]] std::unique_ptr<DataflowPolicy> makeBitwisePolicy();

  // A line through the pin over a prime field, evaluated in the argument's own
  // width. The field bounds every intermediate, which is what lets the
  // construction carry a coefficient at all; it also shrinks with the width,
  // so this declines where no usable prime fits.
  [[nodiscard]] std::unique_ptr<DataflowPolicy> makeArithmeticPolicy();

  // The integer locals and parameters `fn` declares, by name and width.
  [[nodiscard]] std::unordered_map<std::string, std::uint32_t> declaredIntWidths(const FunDecl &fn);

  // The pins at `blockLabel`, one set per visit the profiled run made to it.
  // Points inside a block are ignored — an argument is built at a block's head,
  // so the state on entry is the state it sees.
  //
  // `widths` decides which variables become pins and how wide each one is. A
  // profile records the width of the value a run produced, which is not the
  // width the variable was declared with, and an argument spliced into a call
  // answers to the declaration. A recorded value outside its declared range is
  // dropped for the same reason.
  [[nodiscard]] DataflowSite pinsAtBlock(
      const StateProfile &profile, const std::string &blockLabel,
      const std::unordered_map<std::string, std::uint32_t> &widths
  );

  // Ask the policies in a uniformly random order and take the first argument
  // built. Declining is ordinary, so the order decides which construction a
  // site gets among those that can serve it.
  [[nodiscard]] std::optional<DataflowResult> buildArgument(
      const std::vector<std::unique_ptr<DataflowPolicy>> &policies, const DataflowSite &site,
      std::uint32_t bits, std::int64_t target, NameAllocator &names, std::mt19937 &rng
  );

} // namespace refractir::reify
