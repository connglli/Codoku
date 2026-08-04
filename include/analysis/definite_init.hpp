#pragma once

#include <unordered_map>
#include <unordered_set>
#include "analysis/dataflow.hpp"
#include "analysis/pass_manager.hpp"

namespace refractir {

  /**
   * Performs definite initialization analysis on a function.
   * Ensures that every local variable is assigned a value before it is read.
   * Uses a forward dataflow analysis (Must-Init).
   */
  class DefiniteInitAnalysis : public refractir::FunctionPass {
  public:
    std::string name() const override { return "DefiniteInitAnalysis"; }

    /**
     * Executes the analysis on the function.
     */
    refractir::PassResult run(FunDecl &f, DiagBag &diags) override;

    /**
     * Which locals are definitely initialized on entry to each block, by
     * block label.
     *
     * This is the same must-analysis `run` checks with, exposed because a
     * consumer that emits a *read* somewhere has to agree with the checker
     * about where a read is legal — rytwin splices a guard call at a region
     * entry and passes the live state to it, and a guard reading a local the
     * checker cannot prove initialized is a program that fails re-analysis.
     * Approximating it ("declared with an initializer, or assigned in the
     * function's entry block") is sound but refuses regions the checker would
     * have accepted.
     */
    static std::unordered_map<std::string, std::unordered_set<std::string>>
    initializedAtBlockEntry(const FunDecl &f);

  private:
    using InitSet = std::unordered_map<std::string, bool>;

    /**
     * Dataflow problem definition for definite initialization.
     * The state is a map from variable name to a boolean (is initialized).
     */
    class Problem : public refractir::DataflowProblem<InitSet> {
    public:
      Problem(const FunDecl &f, DiagBag &diags);

      InitSet bottom() override;
      InitSet entryState() override;
      InitSet meet(const InitSet &lhs, const InitSet &rhs) override;
      InitSet transfer(const Block &block, const InitSet &in) override;
      bool equal(const InitSet &lhs, const InitSet &rhs) override;

    private:
      const FunDecl &f_;
      DiagBag &diags_;
    };
  };

} // namespace refractir
