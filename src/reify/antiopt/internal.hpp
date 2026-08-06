#pragma once

// Shared innards of the anti-optimization catalog.
//
// One file per family (see reify/antiopt.hpp for what a family is), each
// registering its rules through the declarations at the bottom. This header
// carries only what more than one family needs: the lookups a rule makes
// about the body it is rewriting, and the dependence scan that decides whether
// two statements may trade places. The AST node constructors are shared with
// the rest of the toolchain and live in ast/build.hpp.

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <limits>
#include "ast/ast.hpp"
#include "ast/build.hpp"
#include "reify/antiopt.hpp"

namespace refractir::reify::antiopt {

  // --- lookups --------------------------------------------------------------

  // The declared type of a local, looked up in the function and in whatever
  // declarations the rewriting has added so far.
  TypePtr localType(const FunDecl &fn, const std::vector<LetDecl> &extra, const std::string &nm);

  // The signed range of a type, or nullopt when it is not a scalar integer.
  std::optional<std::pair<std::int64_t, std::int64_t>> intRange(const TypePtr &t);

  // The width the self-test works at, so a rule states its overflow condition
  // once rather than in each of its two arms.
  inline constexpr bool fitsI8(std::int64_t v) noexcept {
    return v >= std::numeric_limits<std::int8_t>::min() &&
           v <= std::numeric_limits<std::int8_t>::max();
  }

  // --- dependence -----------------------------------------------------------

  // Every local an instruction reads, the one it writes, and whether it
  // touches memory (which orders it against every other memory operation).
  struct Touches {
    std::unordered_set<std::string> reads;
    std::string writes;
    bool memory = false;
  };

  Touches touchesOf(const Instr &ins);

  // Replace every *read* of `from` with `to`. The destination of an assignment
  // is a write and is left alone, so this re-points what a statement consumes
  // without changing what it produces.
  void renameReads(Instr &ins, const std::string &from, const std::string &to);

  // --- registration ---------------------------------------------------------

  using RuleList = std::vector<std::unique_ptr<AntiOptRule>>;

  void registerPeepholeRules(RuleList &out);  // family A
  void registerMbaRules(RuleList &out);       // family B
  void registerStructureRules(RuleList &out); // family C
  void registerLicensedRules(RuleList &out);  // family D
  void registerControlRules(RuleList &out);   // family E

  // Every rule, built once. The engine draws its candidates from here and the
  // self-check walks it.
  const RuleList &catalog();

} // namespace refractir::reify::antiopt
