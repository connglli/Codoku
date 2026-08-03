#pragma once

// Pattern matching over RefractIR expressions.
//
// Anything that rewrites code spends most of its lines asking the same
// question — "is this statement `%d = %x ^ %y`?" — and answering it by hand:
// get_if the instruction, check the access path is empty, get_if the atom,
// check the operator, get_if the coefficient, check it is a local, and only
// then read the two names out. Every rule that did that was mostly that, and
// the mistakes hid in it.
//
// So the question is asked declaratively instead, in the style LLVM's
// PatternMatch uses (the namespace is `pat` rather than `match`, so that the
// entry point can keep the name it wants):
//
//     std::string x, y, d;
//     if (match(ins, m_Assign(m_Local(d), m_One(m_Op(AtomOpKind::Xor,
//                                                    m_Local(x), m_Local(y))))))
//       ...                       // x, y, d are bound
//
// A pattern is any object with `bool match(const T &) const`, so combinators
// nest and binders capture by reference. Matching never mutates the subject;
// a failed match may leave binders written, so read them only on success.
//
// The grammar this mirrors is RefractIR's, which is flatter than most: an
// expression is a `+`/`-` chain of atoms, and an atom holds at most one
// binary operator whose right operand must be an lvalue (spec §5.3). That is
// why `m_One` and `m_Pair` cover so much — most statements are one or two
// atoms — and why there is no nesting combinator to write.

#include <cstdint>
#include <string>
#include <variant>

#include "ast/ast.hpp"

namespace refractir::pat {

  // --- leaves ---------------------------------------------------------------

  // Matches anything, binds nothing.
  struct AnyPattern {
    template<typename T>
    bool match(const T &) const {
      return true;
    }
  };

  inline AnyPattern m_Any() { return {}; }

  // A local, optionally binding its name. Matches a Coef holding a LocalId or
  // an LValue with no access path — the two ways a plain variable appears.
  struct LocalPattern {
    std::string *out = nullptr;

    bool match(const Coef &c) const {
      auto id = std::get_if<LocalOrSymId>(&c);
      if (!id)
        return false;
      auto loc = std::get_if<LocalId>(id);
      if (!loc)
        return false;
      if (out)
        *out = loc->name;
      return true;
    }

    bool match(const LValue &lv) const {
      if (!lv.accesses.empty())
        return false;
      if (out)
        *out = lv.base.name;
      return true;
    }
  };

  inline LocalPattern m_Local(std::string &out) { return LocalPattern{&out}; }

  inline LocalPattern m_AnyLocal() { return LocalPattern{nullptr}; }

  // An integer literal, optionally binding its value.
  struct IntPattern {
    std::int64_t *out = nullptr;

    bool match(const Coef &c) const {
      auto il = std::get_if<IntLit>(&c);
      if (!il)
        return false;
      if (out)
        *out = il->value;
      return true;
    }
  };

  inline IntPattern m_Int(std::int64_t &out) { return IntPattern{&out}; }

  inline IntPattern m_AnyInt() { return IntPattern{nullptr}; }

  // --- atoms ----------------------------------------------------------------

  // `<coef> OP <rval>` — the only binary shape an atom can hold. `kind` fixes
  // the operator; the overload taking a pointer binds it instead.
  template<typename L, typename R>
  struct OpPattern {
    AtomOpKind kind;
    bool anyKind = false;
    AtomOpKind *kindOut = nullptr;
    L lhs;
    R rhs;

    bool match(const Atom &a) const {
      auto op = std::get_if<OpAtom>(&a.v);
      if (!op)
        return false;
      if (!anyKind && op->op != kind)
        return false;
      if (!lhs.match(op->coef) || !rhs.match(op->rval))
        return false;
      if (kindOut)
        *kindOut = op->op;
      return true;
    }
  };

  template<typename L, typename R>
  OpPattern<L, R> m_Op(AtomOpKind kind, L lhs, R rhs) {
    return OpPattern<L, R>{kind, false, nullptr, lhs, rhs};
  }

  template<typename L, typename R>
  OpPattern<L, R> m_AnyOp(AtomOpKind &kindOut, L lhs, R rhs) {
    return OpPattern<L, R>{AtomOpKind::Mul, true, &kindOut, lhs, rhs};
  }

  // A bare coefficient atom — a literal or a variable standing alone.
  template<typename P>
  struct CoefPattern {
    P inner;

    bool match(const Atom &a) const {
      auto co = std::get_if<CoefAtom>(&a.v);
      return co && inner.match(co->coef);
    }
  };

  template<typename P>
  CoefPattern<P> m_Coef(P inner) {
    return CoefPattern<P>{inner};
  }

  // A read of an lvalue.
  template<typename P>
  struct RValPattern {
    P inner;

    bool match(const Atom &a) const {
      auto rv = std::get_if<RValueAtom>(&a.v);
      return rv && inner.match(rv->rval);
    }
  };

  template<typename P>
  RValPattern<P> m_RVal(P inner) {
    return RValPattern<P>{inner};
  }

  // A variable however it is spelled: RefractIR admits a plain local as either
  // a coefficient atom or an rvalue atom, and a rule almost never cares which.
  template<typename P>
  struct VarPattern {
    P inner;

    bool match(const Atom &a) const {
      if (auto co = std::get_if<CoefAtom>(&a.v))
        return inner.match(co->coef);
      if (auto rv = std::get_if<RValueAtom>(&a.v))
        return inner.match(rv->rval);
      return false;
    }
  };

  template<typename P>
  VarPattern<P> m_Var(P inner) {
    return VarPattern<P>{inner};
  }

  // --- expressions ----------------------------------------------------------

  // Exactly one atom, no `+`/`-` tail.
  template<typename P>
  struct OnePattern {
    P inner;

    bool match(const Expr &e) const { return e.rest.empty() && inner.match(e.first); }
  };

  template<typename P>
  OnePattern<P> m_One(P inner) {
    return OnePattern<P>{inner};
  }

  // Exactly two atoms joined by `+` or `-`.
  template<typename A, typename B>
  struct PairPattern {
    AddOp op;
    bool anyOp = false;
    AddOp *opOut = nullptr;
    A a;
    B b;

    bool match(const Expr &e) const {
      if (e.rest.size() != 1)
        return false;
      if (!anyOp && e.rest[0].op != op)
        return false;
      if (!a.match(e.first) || !b.match(e.rest[0].atom))
        return false;
      if (opOut)
        *opOut = e.rest[0].op;
      return true;
    }
  };

  template<typename A, typename B>
  PairPattern<A, B> m_Pair(AddOp op, A a, B b) {
    return PairPattern<A, B>{op, false, nullptr, a, b};
  }

  template<typename A, typename B>
  PairPattern<A, B> m_AnyPair(AddOp &opOut, A a, B b) {
    return PairPattern<A, B>{AddOp::Plus, true, &opOut, a, b};
  }

  // A chain of at least `n` atoms — the shape a rule splits apart rather than
  // reads operands out of.
  struct ChainPattern {
    std::size_t least;

    bool match(const Expr &e) const { return e.rest.size() + 1 >= least; }
  };

  inline ChainPattern m_ChainAtLeast(std::size_t n) { return ChainPattern{n}; }

  // --- instructions ---------------------------------------------------------

  // `<lhs> = <rhs>` where the destination is a whole local (no access path).
  template<typename D, typename R>
  struct AssignPattern {
    D dst;
    R rhs;

    bool match(const Instr &i) const {
      auto ai = std::get_if<AssignInstr>(&i);
      return ai && ai->lhs.accesses.empty() && dst.match(ai->lhs) && rhs.match(ai->rhs);
    }
  };

  template<typename D, typename R>
  AssignPattern<D, R> m_Assign(D dst, R rhs) {
    return AssignPattern<D, R>{dst, rhs};
  }

  // --- entry point ----------------------------------------------------------

  template<typename T, typename P>
  bool match(const T &subject, const P &pattern) {
    return pattern.match(subject);
  }

} // namespace refractir::pat
