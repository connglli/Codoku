#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "ast/ast.hpp"
#include "solver/solver.hpp"

namespace refractir {

  // --- the solved-program header --------------------------------------------
  //
  // A concretized `.sir` file opens with a structured comment recording the
  // values the solver chose:
  //
  //     // SOLVED: %pa0=3, %pa1=-7, ret=42
  //
  // Parameter names appear verbatim in the function body — the printer does
  // not substitute them — so a consumer reads this line to re-run the program
  // on the input it was solved for, without solving it again.
  //
  // Writer and reader live together because the format is their only
  // agreement, and it is a text boundary floats cross: values go out through
  // formatDouble and come back through parseFloatLiteral, so a subnormal or a
  // negative zero survives the round trip (see docs/float.md §9).

  // One model value as the header carries it.
  [[nodiscard]] std::string formatModelValue(const SymbolicExecutor::Result::ModelVal &v);

  // Write the header, or nothing when there is nothing to record. `retText`
  // is already formatted, since a caller may have the return value as a
  // string rather than as a model value.
  void writeSolvedHeader(
      std::ostream &out,
      const std::unordered_map<std::string, SymbolicExecutor::Result::ModelVal> &paramModel,
      const std::string &retText
  );

  // Read the header back as name -> value text, keeping `ret` under that
  // name. Empty when `src` carries no header.
  [[nodiscard]] std::unordered_map<std::string, std::string>
  parseSolvedHeader(std::string_view src);

  class SIRPrinter {
  public:
    explicit SIRPrinter(std::ostream &out) : out_(out) {}

    explicit SIRPrinter(
        std::ostream &out,
        const std::unordered_map<std::string, SymbolicExecutor::Result::ModelVal> &model
    ) : out_(out), model_(model) {}

    explicit SIRPrinter(
        std::ostream &out,
        const std::unordered_map<std::string, SymbolicExecutor::Result::ModelVal> &model,
        const std::unordered_map<std::string, std::vector<SymbolicExecutor::Result::ModelVal>>
            &vecModel
    ) : out_(out), model_(model), vecModel_(vecModel) {}

    void print(const Program &p);

    // Publicly exposed so a caller that persists type information can
    // serialize a TypePtr back to its canonical SIR surface syntax
    // without rolling a private printer.
    void printType(const TypePtr &t);

    // Convenience: render a TypePtr to its canonical SIR surface
    // string without spinning up an SIRPrinter at the call site.
    static std::string typeToString(const TypePtr &t);

  private:
    std::ostream &out_;
    std::unordered_map<std::string, SymbolicExecutor::Result::ModelVal> model_;
    // Per-lane concrete values for vector syms produced by the
    // solver. References to a vec sym `%?v` are rewritten to a synthetic
    // local `%v__solved` whose init list carries these lane values.
    std::unordered_map<std::string, std::vector<SymbolicExecutor::Result::ModelVal>> vecModel_;
    int indent_level_ = 0;

    // Translate `%?v` / `@?v` to a corresponding local identifier suitable
    // for substituting concrete vector-sym references. Result starts with
    // `%` (locals are function-scoped, which is the only scope vec syms
    // live in for now).
    std::string vecSymLocalName(const std::string &symName) const;

    void indent();
    void printExpr(const Expr &e);
    void printAtom(const Atom &a);
    void printCond(const Cond &c);
    void printLValue(const LValue &lv);
    void printCoef(const Coef &c);
    void printSelectVal(const SelectVal &sv);
    void printIndex(const Index &idx);
    void printInitVal(const InitVal &iv);
    void printDomain(const Domain &d);

    std::string relOpToString(RelOp op);
    std::string atomOpToString(AtomOpKind op);
  };

} // namespace refractir
