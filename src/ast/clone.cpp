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

  InitVal cloneInitVal(const InitVal &iv) {
    InitVal out;
    out.kind = iv.kind;
    out.span = iv.span;
    out.value = std::visit(
        [](const auto &x) -> decltype(InitVal::value) {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, std::vector<InitValPtr>>) {
            std::vector<InitValPtr> kids;
            kids.reserve(x.size());
            for (const auto &k: x)
              kids.push_back(k ? std::make_shared<InitVal>(cloneInitVal(*k)) : nullptr);
            return kids;
          } else if constexpr (std::is_same_v<T, AtomPtr>) {
            return x ? std::make_shared<Atom>(cloneAtom(*x)) : nullptr;
          } else {
            return x;
          }
        },
        iv.value
    );
    return out;
  }

  LetDecl cloneLetDecl(const LetDecl &l) {
    LetDecl out;
    out.isMutable = l.isMutable;
    out.name = l.name;
    out.type = l.type;
    if (l.init)
      out.init = cloneInitVal(*l.init);
    out.span = l.span;
    return out;
  }

  Terminator cloneTerminator(const Terminator &t) {
    return std::visit(
        [](const auto &x) -> Terminator {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, BrTerm>) {
            BrTerm out;
            if (x.cond)
              out.cond = cloneCond(*x.cond);
            out.dest = x.dest;
            out.thenLabel = x.thenLabel;
            out.elseLabel = x.elseLabel;
            out.isConditional = x.isConditional;
            out.span = x.span;
            return out;
          } else if constexpr (std::is_same_v<T, RetTerm>) {
            RetTerm out;
            if (x.value)
              out.value = cloneExpr(*x.value);
            out.span = x.span;
            return out;
          } else {
            return x;
          }
        },
        t
    );
  }

  Block cloneBlock(const Block &b) {
    Block out;
    out.label = b.label;
    out.instrs = cloneInstrs(b.instrs);
    out.term = cloneTerminator(b.term);
    out.span = b.span;
    return out;
  }

  FunDecl cloneFunDecl(const FunDecl &f) {
    FunDecl out;
    out.name = f.name;
    out.params = f.params;
    out.retType = f.retType;
    out.syms = f.syms;
    out.lets.reserve(f.lets.size());
    for (const auto &l: f.lets)
      out.lets.push_back(cloneLetDecl(l));
    out.blocks.reserve(f.blocks.size());
    for (const auto &b: f.blocks)
      out.blocks.push_back(cloneBlock(b));
    out.span = f.span;
    out.sourceStem = f.sourceStem;
    out.attributes = f.attributes;
    return out;
  }

  namespace {

    Contract cloneContract(const Contract &c) {
      Contract out;
      out.pres.reserve(c.pres.size());
      for (const auto &p: c.pres)
        out.pres.push_back(PreClause{cloneCond(p.cond), p.message, p.span});
      out.posts.reserve(c.posts.size());
      for (const auto &p: c.posts)
        out.posts.push_back(PostClause{cloneCond(p.cond), p.message, p.span});
      out.span = c.span;
      return out;
    }

    ExtDecl cloneExtDecl(const ExtDecl &d) {
      ExtDecl out;
      out.name = d.name;
      out.params = d.params;
      out.retType = d.retType;
      if (d.contract)
        out.contract = cloneContract(*d.contract);
      // resolvedBody is deliberately not carried: it names a body the clone
      // owns its own copy of. The link resolver refills it.
      out.span = d.span;
      return out;
    }

  } // namespace

  Program cloneProgram(const Program &p) {
    Program out;
    out.structs = p.structs;
    out.funs.reserve(p.funs.size());
    for (const auto &f: p.funs)
      out.funs.push_back(cloneFunDecl(f));
    out.extDecls.reserve(p.extDecls.size());
    for (const auto &d: p.extDecls)
      out.extDecls.push_back(cloneExtDecl(d));
    out.intrinsics = p.intrinsics;
    out.span = p.span;
    return out;
  }

} // namespace refractir
