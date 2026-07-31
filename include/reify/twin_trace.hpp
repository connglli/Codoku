#pragma once

// twin_trace — a region's executed path as one straight-line body.
//
// A twin must compute what the region computes. The cheapest correct way to
// obtain such a body is not to search for one: concatenate the statements the
// profiled run actually executed. Locals are function-scope and non-SSA, so
// the concatenation is already valid RefractIR, and it computes the region's
// effect for *every* state that follows the same path without UB — not only
// the profiled one. The equivalence is by construction; nothing has to prove
// it afterwards.
//
// What the flattening removes is the control flow: branches are gone (the
// path is fixed), and a loop appears as its iterations laid end to end. That
// is what makes the result a transformation rather than a copy — recognizing
// a 7-times-unrolled loop as the loop means re-deriving a closed form, not
// matching a pattern.
//
// Every region is flattenable. There is no floor on how much a trace must
// cover and no ceiling on how far a loop may unroll: the shortest trace is
// still rewritten past recognition by the disguise pass, and the longest is
// bounded by the profiled run itself.
//
// `require` / `assume` instructions are carried over unchanged. They held on
// the profiled path, and a guard that admits only states following that path
// keeps them holding; a later pass may drop the ones it can prove redundant.
//
// Solver-free by construction — this is a copy of code that already ran.

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ast/ast.hpp"

namespace refractir::reify {

  struct TraceBody {
    std::vector<Instr> stmts; // the executed statements, in execution order
    std::string exitLabel;    // where control left the region
    std::size_t steps = 0;    // block executions replayed (loop repeats counted)
    std::size_t blocks = 0;   // distinct blocks covered
  };

  // Flatten the executed block sequence `executed` (in execution order, with
  // a block appearing once per execution) into one statement list, leaving
  // the region at `exitLabel`. `byLabel` resolves labels in the function the
  // region belongs to.
  //
  // Returns nullopt only when a label does not resolve against `byLabel`,
  // with the reason in `why` when given.
  std::optional<TraceBody> flattenTrace(
      const std::vector<std::string> &executed, const std::string &exitLabel,
      const std::unordered_map<std::string, const Block *> &byLabel, std::string *why = nullptr
  );

} // namespace refractir::reify
