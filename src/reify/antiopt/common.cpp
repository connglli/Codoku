#include "internal.hpp"
#include "reify/ast_builder.hpp"

#include <variant>

#include "analysis/type_utils.hpp"

namespace refractir::reify::antiopt {

  Instr assignInstr(const LValue &lhs, Expr rhs) {
    return Instr{AssignInstr{lhs, std::move(rhs), {}}};
  }

  Atom localAtom(const std::string &n) {
    return Atom{CoefAtom{Coef{LocalOrSymId{LocalId{n, {}}}}, {}}, {}};
  }

  Atom intAtom(std::int64_t v) { return Atom{CoefAtom{Coef{IntLit{v, {}}}, {}}, {}}; }

  Atom opAtom(Coef left, AtomOpKind op, const std::string &right) {
    OpAtom o;
    o.op = op;
    o.coef = std::move(left);
    o.rval = localLV(right);
    return Atom{std::move(o), {}};
  }

  Atom binAtom(const std::string &left, AtomOpKind op, const std::string &right) {
    return opAtom(Coef{LocalOrSymId{LocalId{left, {}}}}, op, right);
  }

  Expr opExpr(const std::string &left, AtomOpKind op, const std::string &right) {
    return simpleExpr(binAtom(left, op, right));
  }

  Atom notAtom(const std::string &x) {
    return Atom{UnaryAtom{UnaryOpKind::Not, localLV(x), {}}, {}};
  }

  void addTail(Expr &e, AddOp op, Atom a) { e.rest.push_back(Expr::Tail{op, std::move(a), {}}); }

  TypePtr localType(const FunDecl &fn, const std::vector<LetDecl> &extra, const std::string &nm) {
    for (const auto &p: fn.params)
      if (p.name.name == nm)
        return p.type;
    for (const auto &l: fn.lets)
      if (l.name.name == nm)
        return l.type;
    for (const auto &l: extra)
      if (l.name.name == nm)
        return l.type;
    return nullptr;
  }

  std::optional<std::pair<std::int64_t, std::int64_t>> intRange(const TypePtr &t) {
    auto bits = TypeUtils::getIntBitWidth(t);
    if (!bits || *bits == 0 || *bits > 64)
      return std::nullopt;
    return signedIntRange(*bits);
  }

  namespace {

    // One walk, two readers: the dependence scan below wants the names an
    // instruction reads, and un-CSE wants to replace one of them. `onLocal`
    // sees every local read (as a mutable reference when the instruction is
    // one), `onMemory` fires for the atoms that order against other memory
    // operations. Templated on constness so neither reader needs its own copy
    // of the traversal — the shape of it is where the mistakes live.
    template<typename ExprT, typename OnLocal, typename OnMemory>
    void walkExpr(ExprT &e, OnLocal &onLocal, OnMemory &onMemory);

    template<typename SelectValT, typename OnLocal>
    void walkSelectVal(SelectValT &sv, OnLocal &onLocal) {
      if (auto rv = std::get_if<RValue>(&sv))
        onLocal(rv->base);
      else if (auto co = std::get_if<Coef>(&sv))
        if (auto id = std::get_if<LocalOrSymId>(co))
          if (auto loc = std::get_if<LocalId>(id))
            onLocal(*loc);
    }

    template<typename CoefT, typename OnLocal>
    void walkCoef(CoefT &c, OnLocal &onLocal) {
      if (auto id = std::get_if<LocalOrSymId>(&c))
        if (auto loc = std::get_if<LocalId>(id))
          onLocal(*loc);
    }

    template<typename LValueT, typename OnLocal>
    void walkAccesses(LValueT &lv, OnLocal &onLocal) {
      for (auto &acc: lv.accesses)
        if (auto ai = std::get_if<AccessIndex>(&acc))
          if (auto id = std::get_if<LocalOrSymId>(&ai->index))
            if (auto loc = std::get_if<LocalId>(id))
              onLocal(*loc);
    }

    template<typename AtomT, typename OnLocal, typename OnMemory>
    void walkAtom(AtomT &a, OnLocal &onLocal, OnMemory &onMemory) {
      std::visit(
          [&](auto &x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, CoefAtom>)
              walkCoef(x.coef, onLocal);
            else if constexpr (std::is_same_v<T, RValueAtom> || std::is_same_v<T, UnaryAtom>) {
              onLocal(x.rval.base);
              walkAccesses(x.rval, onLocal);
            } else if constexpr (std::is_same_v<T, OpAtom>) {
              walkCoef(x.coef, onLocal);
              onLocal(x.rval.base);
              walkAccesses(x.rval, onLocal);
            } else if constexpr (std::is_same_v<T, CastAtom>) {
              if (auto lv = std::get_if<LValue>(&x.src)) {
                onLocal(lv->base);
                walkAccesses(*lv, onLocal);
              }
            } else if constexpr (std::is_same_v<T, AddrAtom>) {
              onLocal(x.lv.base);
              walkAccesses(x.lv, onLocal);
              onMemory();
            } else if constexpr (std::is_same_v<T, LoadAtom> || std::is_same_v<T, PtrIndexAtom> ||
                                 std::is_same_v<T, PtrFieldAtom>) {
              onLocal(x.rval.base);
              walkAccesses(x.rval, onLocal);
              onMemory();
              if constexpr (std::is_same_v<T, PtrIndexAtom>)
                walkCoef(x.index, onLocal);
            } else if constexpr (std::is_same_v<T, CmpAtom>) {
              walkSelectVal(x.lhs, onLocal);
              walkSelectVal(x.rhs, onLocal);
            } else if constexpr (std::is_same_v<T, SelectAtom>) {
              if (x.cond) {
                walkExpr(x.cond->lhs, onLocal, onMemory);
                walkExpr(x.cond->rhs, onLocal, onMemory);
              }
              if (x.maskExpr)
                walkExpr(*x.maskExpr, onLocal, onMemory);
              walkSelectVal(x.vtrue, onLocal);
              walkSelectVal(x.vfalse, onLocal);
            } else if constexpr (std::is_same_v<T, CallAtom>) {
              onMemory(); // an intrinsic may observe or touch anything
              for (auto &arg: x.args)
                if (arg)
                  walkExpr(*arg, onLocal, onMemory);
            }
          },
          a.v
      );
    }

    template<typename ExprT, typename OnLocal, typename OnMemory>
    void walkExpr(ExprT &e, OnLocal &onLocal, OnMemory &onMemory) {
      walkAtom(e.first, onLocal, onMemory);
      for (auto &tail: e.rest)
        walkAtom(tail.atom, onLocal, onMemory);
    }

    // Every local an instruction *reads*. The destination of an assignment is
    // a write and is not visited; the indices on the way to it are reads and
    // are.
    template<typename InstrT, typename OnLocal, typename OnMemory>
    void walkInstr(InstrT &ins, OnLocal onLocal, OnMemory onMemory) {
      std::visit(
          [&](auto &x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, AssignInstr>) {
              walkExpr(x.rhs, onLocal, onMemory);
              walkAccesses(x.lhs, onLocal);
            } else if constexpr (std::is_same_v<T, StoreInstr>) {
              walkExpr(x.ptr, onLocal, onMemory);
              walkExpr(x.val, onLocal, onMemory);
              onMemory();
            } else if constexpr (std::is_same_v<T, RequireInstr> ||
                                 std::is_same_v<T, AssumeInstr>) {
              walkExpr(x.cond.lhs, onLocal, onMemory);
              walkExpr(x.cond.rhs, onLocal, onMemory);
            }
          },
          ins
      );
    }

  } // namespace

  void renameReads(Instr &ins, const std::string &from, const std::string &to) {
    walkInstr(
        ins,
        [&](LocalId &id) {
          if (id.name == from)
            id.name = to;
        },
        [] {}
    );
  }

  Touches touchesOf(const Instr &ins) {
    Touches t;
    walkInstr(ins, [&](const LocalId &id) { t.reads.insert(id.name); }, [&] { t.memory = true; });
    if (auto ai = std::get_if<AssignInstr>(&ins))
      t.writes = ai->lhs.base.name;
    return t;
  }

} // namespace refractir::reify::antiopt
