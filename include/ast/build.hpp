#pragma once

// Constructors for the small AST fragments a pass builds by hand.
//
// RefractIR's nodes are aggregates of variants and shared pointers, so
// spelling one out inline costs a line of nested braces and buries what is
// being built: `Expr{Atom{CoefAtom{Coef{IntLit{0, {}}}, {}}, {}}, {}, {}}` is
// the literal zero. Each helper here is the one-line spelling of a shape that
// passes reach for constantly.
//
// They are policy-free. A helper assembles the node the caller names and never
// decides what to build, which is what keeps this header free of any one
// pass's reasoning.
//
// Spans are left empty. A synthesized node denotes no source text, and a
// diagnostic that points at one has nothing to show; a caller splicing a node
// in place of real code may set the span to the code it replaces.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ast/ast.hpp"

namespace refractir {

  // --- type factories -------------------------------------------------------

  [[nodiscard]] inline TypePtr buildI32() {
    return std::make_shared<Type>(Type{IntType{IntType::Kind::I32, {}, {}}, {}});
  }

  [[nodiscard]] inline TypePtr buildI64() {
    return std::make_shared<Type>(Type{IntType{IntType::Kind::I64, {}, {}}, {}});
  }

  [[nodiscard]] inline TypePtr buildI1() {
    return std::make_shared<Type>(Type{IntType{IntType::Kind::ICustom, 1, {}}, {}});
  }

  // --- lvalues --------------------------------------------------------------

  [[nodiscard]] inline LValue buildLValue(const std::string &name) {
    return LValue{LocalId{name, {}}, {}, {}};
  }

  // A local reached through an access path: `%a[1]`, `%s.f0`.
  [[nodiscard]] inline LValue buildLValue(const std::string &name, std::vector<Access> accesses) {
    return LValue{LocalId{name, {}}, std::move(accesses), {}};
  }

  // --- atoms ----------------------------------------------------------------

  [[nodiscard]] inline Atom buildCoefAtom(Coef c) { return Atom{CoefAtom{std::move(c), {}}, {}}; }

  [[nodiscard]] inline Atom buildRValAtom(RValue rv) {
    return Atom{RValueAtom{std::move(rv), {}}, {}};
  }

  [[nodiscard]] inline Atom buildAddrAtom(LValue lv) {
    return Atom{AddrAtom{std::move(lv), {}}, {}};
  }

  [[nodiscard]] inline Atom buildIntAtom(std::int64_t v) {
    return buildCoefAtom(Coef{IntLit{v, {}}});
  }

  [[nodiscard]] inline Atom buildLocalAtom(const std::string &name) {
    return buildCoefAtom(Coef{LocalOrSymId{LocalId{name, {}}}});
  }

  // `<coef> OP %rval` — the only binary shape an atom can hold. RefractIR takes
  // an id or a literal on the left and *requires* an lvalue on the right (spec
  // §5.3), so a constant belonging on the right has to be a local first.
  [[nodiscard]] inline Atom buildOpAtom(Coef left, AtomOpKind op, const std::string &right) {
    return Atom{OpAtom{op, std::move(left), buildLValue(right), {}}, {}};
  }

  // `%left OP %right` — buildOpAtom with a local on both sides.
  [[nodiscard]] inline Atom
  buildBinAtom(const std::string &left, AtomOpKind op, const std::string &right) {
    return buildOpAtom(Coef{LocalOrSymId{LocalId{left, {}}}}, op, right);
  }

  // `~%x`, the only unary operator RefractIR has.
  [[nodiscard]] inline Atom buildNotAtom(const std::string &x) {
    return Atom{UnaryAtom{UnaryOpKind::Not, buildLValue(x), {}}, {}};
  }

  // --- expressions ----------------------------------------------------------

  // An expression of one atom, with no `+`/`-` tail.
  [[nodiscard]] inline Expr buildExpr(Atom a) { return Expr{std::move(a), {}, {}}; }

  [[nodiscard]] inline Expr buildIntExpr(std::int64_t v) { return buildExpr(buildIntAtom(v)); }

  [[nodiscard]] inline Expr buildFloatExpr(double v) {
    return buildExpr(buildCoefAtom(Coef{FloatLit{v, {}}}));
  }

  [[nodiscard]] inline Expr buildNullExpr() { return buildExpr(buildCoefAtom(Coef{NullLit{}})); }

  [[nodiscard]] inline Expr buildRValExpr(RValue rv) {
    return buildExpr(buildRValAtom(std::move(rv)));
  }

  [[nodiscard]] inline Expr buildAddrExpr(LValue lv) {
    return buildExpr(buildAddrAtom(std::move(lv)));
  }

  [[nodiscard]] inline Expr
  buildOpExpr(const std::string &left, AtomOpKind op, const std::string &right) {
    return buildExpr(buildBinAtom(left, op, right));
  }

  // Extend the flat `+`/`-` chain an expression already is.
  inline void appendTail(Expr &e, AddOp op, Atom a) {
    e.rest.push_back(Expr::Tail{op, std::move(a), {}});
  }

  // --- instructions ---------------------------------------------------------

  [[nodiscard]] inline Instr buildAssign(const LValue &lhs, Expr rhs) {
    return Instr{AssignInstr{lhs, std::move(rhs), {}}};
  }

} // namespace refractir
