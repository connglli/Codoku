#pragma once

// Constructors for the small AST fragments every reify generator builds by
// hand. RefractIR's AST nodes are aggregates of variants and shared pointers,
// so spelling one out inline costs three lines and hides the intent; each
// helper here is the one-line spelling of a shape the generators reach for
// constantly. They are policy-free — a helper only assembles the node the
// caller names, and never decides what to build.

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "analysis/type_utils.hpp"
#include "ast/ast.hpp"

namespace refractir::reify {

  // ---------------------------------------------------------------------------
  // Type factories
  // ---------------------------------------------------------------------------

  [[nodiscard]] inline TypePtr makeI32() {
    return std::make_shared<Type>(Type{IntType{IntType::Kind::I32, {}, {}}, {}});
  }

  [[nodiscard]] inline TypePtr makeI64() {
    return std::make_shared<Type>(Type{IntType{IntType::Kind::I64, {}, {}}, {}});
  }

  [[nodiscard]] inline TypePtr makeI1() {
    return std::make_shared<Type>(Type{IntType{IntType::Kind::ICustom, 1, {}}, {}});
  }

  // ---------------------------------------------------------------------------
  // Node construction
  // ---------------------------------------------------------------------------

  [[nodiscard]] inline LValue localLV(const std::string &name) {
    return LValue{LocalId{name, {}}, {}, {}};
  }

  [[nodiscard]] inline Atom coefAtom(Coef c) { return Atom{CoefAtom{std::move(c), {}}, {}}; }

  [[nodiscard]] inline Atom rvalAtom(RValue rv) { return Atom{RValueAtom{std::move(rv), {}}, {}}; }

  [[nodiscard]] inline Expr simpleExpr(Atom a) { return Expr{std::move(a), {}, {}}; }

  // ---------------------------------------------------------------------------
  // Intrinsic registration
  // ---------------------------------------------------------------------------

  // Declare `name` in `prog.intrinsics` unless that exact signature is already
  // there. Idempotent, so a caller may re-declare on every emission without
  // checking first.
  //
  // Signature identity is the whole signature, return type and parameter types
  // included, not just the name: RefractIR admits overloaded intrinsics, and
  // the generators use them (`@popcount` is emitted at both i32 and i64), so
  // matching on name alone would drop the second width on the floor.
  inline void ensureIntrinsicDecl(
      Program &prog, const std::string &name, TypePtr retType,
      const std::vector<std::pair<std::string, TypePtr>> &params
  ) {
    for (const auto &id: prog.intrinsics) {
      if (id.name.name != name || id.params.size() != params.size())
        continue;
      if (!TypeUtils::areTypesEqual(id.retType, retType))
        continue;
      bool sameParams = true;
      for (std::size_t i = 0; i < params.size(); ++i) {
        if (!TypeUtils::areTypesEqual(id.params[i].type, params[i].second)) {
          sameParams = false;
          break;
        }
      }
      if (sameParams)
        return;
    }
    IntrinsicDecl decl;
    decl.name = GlobalId{name, {}};
    decl.retType = std::move(retType);
    for (const auto &[pName, pType]: params) {
      ParamDecl pd;
      pd.name = LocalId{pName, {}};
      pd.type = pType;
      decl.params.push_back(std::move(pd));
    }
    prog.intrinsics.push_back(std::move(decl));
  }

  // ---------------------------------------------------------------------------
  // Bitwidth ranges
  // ---------------------------------------------------------------------------

  // Inclusive [lo, hi] range of a signed `iN`. Widths of 64 and above saturate
  // to the i64 range, which is the widest literal the AST carries.
  //
  // Requires `bits >= 1`. Width 0 is not a RefractIR type; it reaches here only
  // from a caller that read a bitwidth off a non-integer type, and the range it
  // would name is empty, so the degenerate {0, 0} keeps every downstream
  // distribution well-formed instead of shifting by a negative amount.
  [[nodiscard]] inline std::pair<std::int64_t, std::int64_t>
  signedIntRange(std::uint32_t bits) noexcept {
    assert(bits >= 1 && "signedIntRange: width 0 is not a RefractIR integer type");
    if (bits == 0)
      return {0, 0};
    if (bits >= 64)
      return {INT64_MIN, INT64_MAX};
    std::int64_t hi = (std::int64_t{1} << (bits - 1)) - 1;
    return {-hi - 1, hi};
  }

} // namespace refractir::reify
