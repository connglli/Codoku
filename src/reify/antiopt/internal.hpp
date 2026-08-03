#pragma once

// Shared innards of the anti-optimization catalog.
//
// One file per family (see reify/antiopt.hpp for what a family is), each
// registering its rules through the declarations at the bottom. This header
// carries only what more than one family needs: the small AST builders that
// respect RefractIR's operand rules, and the dependence scan that decides
// whether two statements may trade places.

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "ast/ast.hpp"
#include "reify/antiopt.hpp"

namespace refractir::reify::antiopt {

  // --- builders -------------------------------------------------------------

  LValue localLV(const std::string &n);
  Expr simpleExpr(Atom a);
  Instr assignInstr(const LValue &lhs, Expr rhs);

  // `<left> OP %right`. RefractIR takes an id or a literal on the left of a
  // binary atom and *requires* an lvalue on the right (spec §5.3), so a
  // constant that belongs on the right has to be a local first.
  Expr opExpr(const std::string &left, AtomOpKind op, const std::string &right);

  // The declared type of a local, looked up in the function and in whatever
  // declarations the rewriting has added so far.
  TypePtr localType(const FunDecl &fn, const std::vector<LetDecl> &extra, const std::string &nm);

  // The signed range of a type, or nullopt when it is not a scalar integer.
  std::optional<std::pair<std::int64_t, std::int64_t>> intRange(const TypePtr &t);

  // --- dependence -----------------------------------------------------------

  // Every local an instruction reads, the one it writes, and whether it
  // touches memory (which orders it against every other memory operation).
  struct Touches {
    std::unordered_set<std::string> reads;
    std::string writes;
    bool memory = false;
  };

  Touches touchesOf(const Instr &ins);

  // --- registration ---------------------------------------------------------

  using RuleList = std::vector<std::unique_ptr<AntiOptRule>>;

  void registerPeepholeRules(RuleList &out);  // family A
  void registerMbaRules(RuleList &out);       // family B
  void registerStructureRules(RuleList &out); // family C

} // namespace refractir::reify::antiopt
