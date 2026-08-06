#pragma once

#include <string>
#include <unordered_set>
#include "analysis/pass_manager.hpp"

namespace refractir {

  /**
   * The identity of an intrinsic declaration: its name and its parameter
   * types, as a string two declarations share exactly when they declare the
   * same intrinsic.
   *
   * RefractIR admits overloading, so the name alone does not identify one:
   * `@to_bits(f32)` and `@to_bits(f64)` lower differently, and
   * `@reduce_add(<4> i32)` and `@reduce_add(<8> i32)` are separate
   * declarations. The return type is not part of it — it follows from the
   * parameters for every intrinsic the toolchain defines, so including it
   * would split one declaration into two.
   *
   * The semantic checker rejects a program that declares one identity twice.
   * Anything else that decides whether two declarations are the same, such as
   * merging two programs' declarations, has to agree with it, which is why
   * this is the shared answer rather than each caller's own.
   */
  [[nodiscard]] std::string intrinsicSignature(const IntrinsicDecl &d);

  /**
   * Whether two struct declarations declare the same type: the same fields,
   * in the same order, with the same types.
   *
   * Order counts. Layout is packed and sequential (spec §4), so two
   * declarations that agree on names and types but not on order describe
   * different objects, and code compiled against one reads the other's fields
   * at the wrong offsets while still typechecking.
   */
  [[nodiscard]] bool sameStructDecl(const StructDecl &a, const StructDecl &b);

  /**
   * Performs semantic analysis on the RefractIR program.
   * Checks for duplicate declarations, invalid sigils, and other
   * well-formedness constraints not captured by the grammar or type checker.
   */
  class SemChecker : public refractir::ModulePass {
  public:
    std::string name() const override { return "SemChecker"; }

    /**
     * Executes the semantic checker on the program.
     */
    refractir::PassResult run(Program &prog, DiagBag &diags) override;

  private:
    void checkStruct(const StructDecl &s, DiagBag &diags);
    void checkFunction(const FunDecl &f, DiagBag &diags);
    void checkSigils(const FunDecl &f, DiagBag &diags);
    void checkDuplicates(const FunDecl &f, DiagBag &diags);
    void checkExtDecl(const ExtDecl &d, DiagBag &diags);
    void checkIntrinsicDecl(const IntrinsicDecl &d, DiagBag &diags);
  };

} // namespace refractir
