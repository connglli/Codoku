#include "internal.hpp"

#include <variant>

#include "analysis/type_utils.hpp"

namespace refractir::reify::antiopt {

  LValue localLV(const std::string &n) { return LValue{LocalId{n, {}}, {}, {}}; }

  Expr simpleExpr(Atom a) { return Expr{std::move(a), {}, {}}; }

  Instr assignInstr(const LValue &lhs, Expr rhs) {
    return Instr{AssignInstr{lhs, std::move(rhs), {}}};
  }

  Expr opExpr(const std::string &left, AtomOpKind op, const std::string &right) {
    OpAtom o;
    o.op = op;
    o.coef = Coef{LocalOrSymId{LocalId{left, {}}}};
    o.rval = localLV(right);
    return simpleExpr(Atom{std::move(o), {}});
  }

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
    if (*bits >= 64)
      return std::pair<std::int64_t, std::int64_t>{INT64_MIN, INT64_MAX};
    const std::int64_t lim = std::int64_t{1} << (*bits - 1);
    return std::pair<std::int64_t, std::int64_t>{-lim, lim - 1};
  }

  namespace {

    void scanExprLocals(const Expr &e, Touches &t);

    void scanAtomLocals(const Atom &a, Touches &t) {
      std::visit(
          [&](const auto &x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, CoefAtom>) {
              if (auto id = std::get_if<LocalOrSymId>(&x.coef))
                if (auto loc = std::get_if<LocalId>(id))
                  t.reads.insert(loc->name);
            } else if constexpr (std::is_same_v<T, RValueAtom>)
              t.reads.insert(x.rval.base.name);
            else if constexpr (std::is_same_v<T, UnaryAtom>)
              t.reads.insert(x.rval.base.name);
            else if constexpr (std::is_same_v<T, OpAtom>) {
              if (auto id = std::get_if<LocalOrSymId>(&x.coef))
                if (auto loc = std::get_if<LocalId>(id))
                  t.reads.insert(loc->name);
              t.reads.insert(x.rval.base.name);
            } else if constexpr (std::is_same_v<T, CastAtom>) {
              if (auto lv = std::get_if<LValue>(&x.src))
                t.reads.insert(lv->base.name);
            } else if constexpr (std::is_same_v<T, AddrAtom>) {
              t.reads.insert(x.lv.base.name);
              t.memory = true;
            } else if constexpr (std::is_same_v<T, LoadAtom>) {
              t.reads.insert(x.rval.base.name);
              t.memory = true;
            } else if constexpr (std::is_same_v<T, PtrIndexAtom> ||
                                 std::is_same_v<T, PtrFieldAtom>) {
              t.reads.insert(x.rval.base.name);
              t.memory = true;
            } else if constexpr (std::is_same_v<T, CmpAtom>) {
              for (const auto *sv: {&x.lhs, &x.rhs})
                if (auto rv = std::get_if<RValue>(sv))
                  t.reads.insert(rv->base.name);
                else if (auto id = std::get_if<LocalOrSymId>(&std::get<Coef>(*sv)))
                  if (auto loc = std::get_if<LocalId>(id))
                    t.reads.insert(loc->name);
            } else if constexpr (std::is_same_v<T, SelectAtom>) {
              if (x.cond) {
                scanExprLocals(x.cond->lhs, t);
                scanExprLocals(x.cond->rhs, t);
              }
              if (x.maskExpr)
                scanExprLocals(*x.maskExpr, t);
              for (const auto *sv: {&x.vtrue, &x.vfalse})
                if (auto rv = std::get_if<RValue>(sv))
                  t.reads.insert(rv->base.name);
                else if (auto id = std::get_if<LocalOrSymId>(&std::get<Coef>(*sv)))
                  if (auto loc = std::get_if<LocalId>(id))
                    t.reads.insert(loc->name);
            } else if constexpr (std::is_same_v<T, CallAtom>) {
              t.memory = true; // an intrinsic may observe or touch anything
              for (const auto &arg: x.args)
                if (arg)
                  scanExprLocals(*arg, t);
            }
          },
          a.v
      );
    }

    void scanExprLocals(const Expr &e, Touches &t) {
      scanAtomLocals(e.first, t);
      for (const auto &tail: e.rest)
        scanAtomLocals(tail.atom, t);
    }

  } // namespace

  Touches touchesOf(const Instr &ins) {
    Touches t;
    std::visit(
        [&](const auto &x) {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, AssignInstr>) {
            scanExprLocals(x.rhs, t);
            for (const auto &acc: x.lhs.accesses)
              if (auto ai = std::get_if<AccessIndex>(&acc))
                if (auto id = std::get_if<LocalOrSymId>(&ai->index))
                  if (auto loc = std::get_if<LocalId>(id))
                    t.reads.insert(loc->name);
            t.writes = x.lhs.base.name;
          } else if constexpr (std::is_same_v<T, StoreInstr>) {
            scanExprLocals(x.ptr, t);
            scanExprLocals(x.val, t);
            t.memory = true;
          } else if constexpr (std::is_same_v<T, RequireInstr> || std::is_same_v<T, AssumeInstr>) {
            scanExprLocals(x.cond.lhs, t);
            scanExprLocals(x.cond.rhs, t);
          }
        },
        ins
    );
    return t;
  }

} // namespace refractir::reify::antiopt
