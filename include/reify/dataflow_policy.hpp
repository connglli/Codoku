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
#include <vector>

#include "ast/ast.hpp"
#include "reify/state_profile.hpp"

namespace refractir::reify {

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
    // `site`, or nullopt to decline.
    [[nodiscard]] virtual std::optional<DataflowResult>
    build(const DataflowSite &site, std::uint32_t bits, std::int64_t target, std::mt19937 &rng) = 0;
  };

  // A literal, or `%v + bias` over a variable the profiled run pins. The floor
  // of the catalog: the literal costs nothing and always applies, so a site
  // reaching this policy always leaves with an argument.
  [[nodiscard]] std::unique_ptr<DataflowPolicy> makeBaselinePolicy();

  // `k ^ %v` over a variable the profiled run pins. Total at every width, so
  // it states targets a bias has no literal to reach; declines when nothing is
  // pinned.
  [[nodiscard]] std::unique_ptr<DataflowPolicy> makeBitwisePolicy();

  // The pins at `blockLabel`, one set per visit the profiled run made to it.
  // Points inside a block are ignored — an argument is built at a block's head,
  // so the state on entry is the state it sees.
  [[nodiscard]] DataflowSite
  pinsAtBlock(const StateProfile &profile, const std::string &blockLabel);

  // Ask the policies in a uniformly random order and take the first argument
  // built. Declining is ordinary, so the order decides which construction a
  // site gets among those that can serve it.
  [[nodiscard]] std::optional<DataflowResult> buildArgument(
      const std::vector<std::unique_ptr<DataflowPolicy>> &policies, const DataflowSite &site,
      std::uint32_t bits, std::int64_t target, std::mt19937 &rng
  );

} // namespace refractir::reify
