#include "ast/clone.hpp"

#include <memory>
#include <utility>
#include <variant>

namespace refractir {

  namespace {

    // Every atom kind except SelectAtom and CallAtom is copyable as it
    // stands: their members are values, spans, or `shared_ptr<Type>` which
    // is immutable once built. Only the two owning kinds need a walk.
    SelectAtom cloneSelect(const SelectAtom &s) {
      SelectAtom out;
      if (s.cond)
        out.cond = std::make_unique<Cond>(cloneCond(*s.cond));
      if (s.maskExpr)
        out.maskExpr = std::make_unique<Expr>(cloneExpr(*s.maskExpr));
      out.vtrue = s.vtrue;
      out.vfalse = s.vfalse;
      out.span = s.span;
      return out;
    }

    CallAtom cloneCall(const CallAtom &c) {
      CallAtom out;
      out.callee = c.callee;
      out.args.reserve(c.args.size());
      // Fresh argument expressions rather than shared ones: two clones that
      // share an argument would edit each other.
      for (const auto &a: c.args)
        out.args.push_back(a ? std::make_shared<Expr>(cloneExpr(*a)) : nullptr);
      out.span = c.span;
      // The resolved overload is a pointer into the program's intrinsic
      // declarations, not into the cloned subtree, so it stays valid and
      // saves the clone a re-resolution.
      out.resolvedIntrinsic = c.resolvedIntrinsic;
      return out;
    }

  } // namespace

  Atom cloneAtom(const Atom &a) {
    Atom out;
    out.span = a.span;
    out.v = std::visit(
        [](const auto &x) -> Atom::Variant {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, SelectAtom>)
            return cloneSelect(x);
          else if constexpr (std::is_same_v<T, CallAtom>)
            return cloneCall(x);
          else
            return x;
        },
        a.v
    );
    return out;
  }

  Expr cloneExpr(const Expr &e) {
    Expr out;
    out.first = cloneAtom(e.first);
    out.rest.reserve(e.rest.size());
    for (const auto &t: e.rest)
      out.rest.push_back(Expr::Tail{t.op, cloneAtom(t.atom), t.span});
    out.span = e.span;
    return out;
  }

  Cond cloneCond(const Cond &c) {
    Cond out;
    out.lhs = cloneExpr(c.lhs);
    out.op = c.op;
    out.rhs = cloneExpr(c.rhs);
    out.span = c.span;
    return out;
  }

  Instr cloneInstr(const Instr &i) {
    return std::visit(
        [](const auto &x) -> Instr {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, AssignInstr>)
            return AssignInstr{x.lhs, cloneExpr(x.rhs), x.span};
          else if constexpr (std::is_same_v<T, AssumeInstr>)
            return AssumeInstr{cloneCond(x.cond), x.span};
          else if constexpr (std::is_same_v<T, RequireInstr>)
            return RequireInstr{cloneCond(x.cond), x.message, x.span};
          else
            return StoreInstr{cloneExpr(x.ptr), cloneExpr(x.val), x.span};
        },
        i
    );
  }

  std::vector<Instr> cloneInstrs(const std::vector<Instr> &is) {
    std::vector<Instr> out;
    out.reserve(is.size());
    for (const auto &i: is)
      out.push_back(cloneInstr(i));
    return out;
  }

} // namespace refractir
