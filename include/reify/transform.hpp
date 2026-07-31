#pragma once

// Whole-program rewrite framework (top tier).
//
// A `Transform` is a composable whole-program rewrite over a RefractIR
// `Program`, threading a rich `TransformContext`: a deterministic RNG, the
// per-function reify metadata (input->output `descriptors` and, when
// captured, per-program-point `profiles`), and an SMT solver factory for
// synthesizing transforms. Concrete transforms either preserve the
// descriptor's input->output contract (equivalence on the profiled input)
// or document precisely what they change.
//
// This is the tier the user calls "a kind of rewrite": rytwin's
// equivalence-preserving block graft (TwinTransform) and rylink's
// call-realization (CallRealizeTransform) are both `Transform`s, driven
// through one `TransformPipeline`. Below it sits the peephole tier —
// local, pattern-directed edits inside one function — which has its own
// header, reify/rewrite.hpp, carrying the rule contract every peephole
// engine must satisfy. A Transform may own and drive a set of such rules;
// a rule never sees the Program, only the unit it edits.
//
// The context is "path-specific, profiling optional": a `FuncDescriptor`
// always carries the concretized execution `path`, so path-aware transforms
// work off `descriptors` alone; the fuller per-point `StateProfile` is
// present only when a tool captured one (rytwin), absent otherwise (rylink).

#include <iosfwd>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ast/ast.hpp"
#include "reify/func_desc.hpp"
#include "reify/state_profile.hpp"
#include "solver/solver.hpp"

namespace refractir::reify {

  // Shared, threaded context for a chain of Transforms. A single-function
  // tool (rytwin) fills one entry in each map keyed by the entry function's
  // canonical name; rylink whole-programs carry many. `rng` is a reference
  // to the driving tool's own stream, so every draw a transform makes stays
  // on that one deterministic sequence — no copy-back, no divergence.
  struct TransformContext {
    explicit TransformContext(std::mt19937 &r) : rng(r) {}

    std::mt19937 &rng;                             // the tool's deterministic stream
    SymbolicExecutor::SolverFactory solverFactory; // for synthesizing transforms; may be null
    std::unordered_map<std::string, FuncDescriptor> descriptors; // input i / oracle, by func name
    std::unordered_map<std::string, StateProfile> profiles;      // per-point state, by func name
    // When non-null, transforms may emit human-readable per-decision progress
    // here (the driver's `--verbose`). Transforms stay silent otherwise.
    std::ostream *verbose = nullptr;
  };

  // Outcome of applying one Transform (or a whole pipeline).
  struct TransformReport {
    std::size_t sites = 0; // number of rewrite sites applied
    bool ok = true;        // false aborts the pipeline
    std::string message;   // human-readable note on failure / summary
  };

  class Transform {
  public:
    virtual ~Transform() = default;
    virtual std::string_view name() const = 0;

    // A transform that needs the per-function StateProfile (e.g. rytwin)
    // says so here; the driver can then refuse to run it without a profile
    // rather than silently no-op.
    virtual bool needsProfile() const { return false; }

    virtual TransformReport apply(Program &prog, TransformContext &ctx) = 0;
  };

  // Ordered application, threading `ctx` through each stage. Stops at the
  // first transform that returns ok == false; otherwise sums the site counts.
  class TransformPipeline {
  public:
    TransformPipeline &add(std::unique_ptr<Transform> t) {
      transforms_.push_back(std::move(t));
      return *this;
    }

    TransformReport run(Program &prog, TransformContext &ctx);

  private:
    std::vector<std::unique_ptr<Transform>> transforms_;
  };

} // namespace refractir::reify
