#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "ast/ast.hpp"
#include "reify/cfg_gen.hpp"
#include "reify/checksum.hpp"
#include "reify/expr_gen.hpp"
#include "reify/var_catalogue.hpp"
#include "solver/solver.hpp"

namespace refractir::reify {

  struct FuncGenConfig {
    std::string funcName = "func";
    std::uint32_t seed = 0;
    int nStmts = 3;
    // Off-path blocks are never executed at the solved inputs, so their
    // volume costs the solver nothing while widening the compiler-facing
    // optimization surface. nStmts and exprCfg.{min,max}Atoms describe
    // ON-path blocks; off-path blocks scale all three by this factor
    // (--off-path-multiplier, rounded to the nearest int, default 2x).
    double offPathMultiplier = 2.0;
    bool enableInterestCoefs = true;
    bool enableInterestInits = true;
    bool enableIntrinsics = true;
    // Probability that a new on-path coef sym gets a `|c| > 2^20` require
    // rather than the `c != 0 ∧ c != 1 ∧ c != -1` triple. The triple only
    // rules out the degenerate coefficients, so the solver clusters every
    // coef at ±2 — the smallest values that survive it. The magnitude
    // require trades that floor for a spread across the type's range.
    double pLargeCoef = 0.3;
    // Magnitude threshold T for the `|c| > T` interest require, set by
    // --large-coef. Clamped per-coef to the coef's domain ∩ type range, so
    // a value wider than --coef-domain degrades to the largest in-domain
    // magnitude rather than going UNSAT.
    std::int64_t largeCoefThreshold = std::int64_t{1} << 20;
    ExprGenConfig exprCfg;
    // Sym counter domains
    std::int64_t coefLo = -8, coefHi = 8;
    std::int64_t valueLo = -128, valueHi = 127;
    std::int64_t indexLo = 1, indexHi = 30;
  };

  struct FuncGenResult {
    refractir::Program prog;
    std::vector<std::string> pathLabels; // ["^entry", "^b0", ...]
  };

  [[nodiscard]] FuncGenResult genFunction(
      const RyCFG &cfg, const std::vector<std::string> &path, const VarCatalogue &vars,
      const FuncGenConfig &fcfg
  );

  // Cycle-closing corrections for --require-nonterm. For each
  // scalar-integer mutable let the lasso cycle writes, appends a fresh
  // additive-correction symbol (`sym %?ntK`) and a `%v = %v + %?ntK;`
  // statement to the latch block (before its terminator). This gives the
  // solver the freedom to restore each leaf to its header-entry value, so
  // the header-state fixed point RequireNonterm asserts is always solvable
  // regardless of the random cycle body. `cycleLabels` / `latchLabel` are
  // ^-prefixed. No-op if the function or latch is absent, or the cycle
  // writes no scalar-int let.
  // `period` is the orbit's lap count k (1 = classic single-lap fixed point).
  // For k > 1 a modular counter is spliced in alongside the corrections, since
  // the corrections alone can only close a period-1 orbit.
  void spliceNontermCorrections(
      Program &prog, const std::string &funcName, const std::vector<std::string> &cycleLabels,
      const std::string &latchLabel, int period = 1
  );

} // namespace refractir::reify
