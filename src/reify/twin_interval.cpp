#include "reify/twin_interval.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

#include "analysis/type_utils.hpp"
#include "reify/twin_mini.hpp"

namespace refractir::reify {

  namespace {

    using I64 = std::int64_t;

    constexpr I64 kI64Min = std::numeric_limits<I64>::min();
    constexpr I64 kI64Max = std::numeric_limits<I64>::max();

    Interval unknownOf() {
      Interval v;
      v.unknown = true;
      v.lo = kI64Min;
      v.hi = kI64Max;
      return v;
    }

    Interval constant(I64 c) { return Interval{c, c, false}; }

    // The representable range of a signed N-bit value. Widths above 64 are not
    // representable in the domain, so they are reported as unknown.
    bool rangeOf(std::uint32_t bits, I64 &lo, I64 &hi) {
      if (bits == 0 || bits > 64)
        return false;
      if (bits == 64) {
        lo = kI64Min;
        hi = kI64Max;
        return true;
      }
      lo = -(I64(1) << (bits - 1));
      hi = (I64(1) << (bits - 1)) - 1;
      return true;
    }

    bool fits(const Interval &v, std::uint32_t bits) {
      I64 lo = 0, hi = 0;
      return !v.unknown && rangeOf(bits, lo, hi) && v.lo >= lo && v.hi <= hi;
    }

    // The unsigned bit pattern of a signed value at width `bits`, and back.
    std::uint64_t toUnsigned(I64 v, std::uint32_t bits) {
      if (bits >= 64)
        return (std::uint64_t) v;
      return (std::uint64_t) v & ((std::uint64_t(1) << bits) - 1);
    }

    I64 signExtend(std::uint64_t v, std::uint32_t bits) {
      if (bits >= 64)
        return (I64) v;
      const std::uint64_t sign = std::uint64_t(1) << (bits - 1);
      return (I64) ((v ^ sign) - sign);
    }

    // Addition / multiplication that report overflow rather than wrapping, so
    // the domain never has to reason about a value it cannot represent.
    bool addOv(I64 a, I64 b, I64 &out) { return __builtin_add_overflow(a, b, &out); }

    bool subOv(I64 a, I64 b, I64 &out) { return __builtin_sub_overflow(a, b, &out); }

    bool mulOv(I64 a, I64 b, I64 &out) { return __builtin_mul_overflow(a, b, &out); }

    // --- the walk ----------------------------------------------------------

    class Checker {
    public:
      Checker(const FunDecl &fn, const StructMap &structs, const IntervalEnv &entry) :
          env_(entry), structs_(structs) {
        for (const auto &p: fn.params)
          types_[p.name.name] = p.type;
        for (const auto &l: fn.lets)
          types_[l.name.name] = l.type;
      }

      IntervalVerdict run(const TraceBody &body) {
        std::size_t next = 0; // next path check to consult
        for (std::size_t i = 0; i < body.stmts.size(); ++i) {
          // Conditions recorded *before* this statement see the state the
          // original saw when it branched.
          while (next < body.checks.size() && body.checks[next].afterStmt <= i) {
            if (!decides(body.checks[next]))
              return fail("branch could go either way: " + reason_);
            ++next;
          }
          if (!step(body.stmts[i]))
            return fail(reason_);
        }
        for (; next < body.checks.size(); ++next)
          if (!decides(body.checks[next]))
            return fail("branch could go either way: " + reason_);
        return IntervalVerdict{true, ""};
      }

    private:
      IntervalVerdict fail(const std::string &why) { return IntervalVerdict{false, why}; }

      bool reject(std::string why) {
        reason_ = std::move(why);
        return false;
      }

      // --- types -----------------------------------------------------------

      // Width of the value an lvalue names, or 0 when it is not a scalar
      // integer (floats, pointers, whole aggregates and vectors).
      std::uint32_t widthOf(const LValue &lv) const {
        auto it = types_.find(lv.base.name);
        if (it == types_.end())
          return 0;
        TypePtr t = it->second;
        for (const auto &acc: lv.accesses) {
          if (!t)
            return 0;
          if (auto af = std::get_if<AccessField>(&acc)) {
            t = fieldType(t, af->field);
            continue;
          }
          if (const auto *at = TypeUtils::asArray(t))
            t = at->elem;
          else if (const auto *vt = TypeUtils::asVec(t))
            t = vt->elem;
          else
            return 0;
        }
        auto bits = TypeUtils::getIntBitWidth(t);
        return bits ? *bits : 0;
      }

      // --- environment ------------------------------------------------------

      Interval read(const LValue &lv) const {
        auto key = leafKey(lv);
        if (!key)
          return unknownOf();
        auto it = env_.find(*key);
        return it == env_.end() ? unknownOf() : it->second;
      }

      void write(const LValue &lv, const Interval &v) {
        if (auto key = leafKey(lv))
          env_[*key] = v;
      }

      // A write whose destination cannot be named (a non-literal index) may
      // land on any leaf of its root, and a store may land anywhere at all.
      void forget(const std::string &root) {
        for (auto &[k, v]: env_)
          if (k == root || (k.size() > root.size() && k.compare(0, root.size(), root) == 0 &&
                            (k[root.size()] == '[' || k[root.size()] == '.')))
            v = unknownOf();
      }

      void forgetAll() {
        for (auto &[k, v]: env_)
          v = unknownOf();
      }

      // --- expressions ------------------------------------------------------

      // Evaluate `e` in the current environment, checking the UB conditions of
      // every operation it performs. Returns nullopt when an operation could
      // not be proven safe (`reason_` says which).
      std::optional<Interval> eval(const Expr &e, std::uint32_t bits) {
        auto acc = evalAtom(e.first, bits);
        if (!acc)
          return std::nullopt;
        for (const auto &t: e.rest) {
          auto rhs = evalAtom(t.atom, bits);
          if (!rhs)
            return std::nullopt;
          auto sum = t.op == AddOp::Plus ? addI(*acc, *rhs, bits) : subI(*acc, *rhs, bits);
          if (!sum)
            return std::nullopt;
          acc = *sum;
        }
        return acc;
      }

      std::optional<Interval> evalCoef(const Coef &c, std::uint32_t bits) {
        if (auto il = std::get_if<IntLit>(&c))
          return constant(il->value);
        if (auto id = std::get_if<LocalOrSymId>(&c)) {
          auto loc = std::get_if<LocalId>(id);
          if (!loc)
            return unknownOf(); // a symbol: twins are concrete, so this is odd
          LValue lv;
          lv.base = *loc;
          if (widthOf(lv) == 0)
            return unknownOf();
          return read(lv);
        }
        (void) bits;
        return unknownOf(); // float or null literal
      }

      std::optional<Interval> evalSelectVal(const SelectVal &sv, std::uint32_t bits) {
        if (auto rv = std::get_if<RValue>(&sv))
          return readChecked(*rv);
        return evalCoef(std::get<Coef>(sv), bits);
      }

      // Reading an lvalue evaluates its indices, which must be in bounds.
      std::optional<Interval> readChecked(const LValue &lv) {
        if (!indicesInBounds(lv))
          return std::nullopt;
        return read(lv);
      }

      std::optional<Interval> evalAtom(const Atom &a, std::uint32_t bits) {
        return std::visit(
            [&](const auto &x) -> std::optional<Interval> {
              using T = std::decay_t<decltype(x)>;
              if constexpr (std::is_same_v<T, CoefAtom>)
                return evalCoef(x.coef, bits);
              else if constexpr (std::is_same_v<T, RValueAtom>)
                return readChecked(x.rval);
              else if constexpr (std::is_same_v<T, UnaryAtom>) {
                auto v = readChecked(x.rval);
                if (!v)
                  return std::nullopt;
                if (v->unknown)
                  return unknownOf();
                return Interval{~v->hi, ~v->lo, false}; // ~ is monotone decreasing
              } else if constexpr (std::is_same_v<T, CmpAtom>) {
                // i1 true is -1 (spec §6.4), so a comparison is one of {0, -1}.
                auto l = evalSelectVal(x.lhs, 0), r = evalSelectVal(x.rhs, 0);
                if (!l || !r)
                  return std::nullopt;
                return Interval{-1, 0, false};
              } else if constexpr (std::is_same_v<T, SelectAtom>)
                return evalSelect(x, bits);
              else if constexpr (std::is_same_v<T, CastAtom>)
                return evalCast(x);
              else if constexpr (std::is_same_v<T, OpAtom>)
                return evalOp(x, bits);
              else
                // addr / load / ptrindex / ptrfield / call: the value is not
                // tracked. Loads through a pointer are the common case and the
                // main reason a memory-bearing region does not widen yet.
                return unknownOf();
            },
            a.v
        );
      }

      std::optional<Interval> evalSelect(const SelectAtom &s, std::uint32_t bits) {
        // Both arms are evaluated here even though only one runs, so an arm
        // that could trap is not provable — sound, and it costs precision only
        // on selects whose untaken arm is unsafe.
        if (s.cond && !evalCond(*s.cond))
          return std::nullopt;
        if (s.maskExpr && !eval(*s.maskExpr, bits))
          return std::nullopt;
        auto t = evalSelectVal(s.vtrue, bits), f = evalSelectVal(s.vfalse, bits);
        if (!t || !f)
          return std::nullopt;
        if (t->unknown || f->unknown)
          return unknownOf();
        return Interval{std::min(t->lo, f->lo), std::max(t->hi, f->hi), false};
      }

      std::optional<Interval> evalCast(const CastAtom &c) {
        auto bits = TypeUtils::getIntBitWidth(c.dstType);
        if (!bits)
          return unknownOf(); // to float
        std::optional<Interval> src;
        if (auto il = std::get_if<IntLit>(&c.src))
          src = constant(il->value);
        else if (auto lv = std::get_if<LValue>(&c.src))
          src = readChecked(*lv);
        else
          src = unknownOf(); // float literal or sym
        if (!src)
          return std::nullopt;
        // Widening keeps the value; narrowing truncates, which the domain does
        // not model, so it is only exact when the value already fits.
        return fits(*src, *bits) ? *src : unknownOf();
      }

      std::optional<Interval> evalOp(const OpAtom &o, std::uint32_t bits) {
        auto l = evalCoef(o.coef, bits);
        if (!l)
          return std::nullopt;
        auto r = readChecked(o.rval);
        if (!r)
          return std::nullopt;
        const std::uint32_t w = bits ? bits : widthOf(o.rval);
        switch (o.op) {
          case AtomOpKind::Mul:
            return mulI(*l, *r, w);
          case AtomOpKind::Div:
          case AtomOpKind::Mod: {
            if (r->unknown || (r->lo <= 0 && r->hi >= 0))
              return reject("divisor may be zero"), std::nullopt;
            // INT_MIN / -1 is the one division that overflows.
            I64 lo = 0, hi = 0;
            if (rangeOf(w, lo, hi) && (l->unknown || l->lo <= lo) && r->lo <= -1 && r->hi >= -1)
              return reject("division may overflow (MIN / -1)"), std::nullopt;
            if (l->isConst() && r->isConst())
              return constant(o.op == AtomOpKind::Div ? l->lo / r->lo : l->lo % r->lo);
            return unknownOf();
          }
          case AtomOpKind::Shl: {
            if (l->unknown || l->lo < 0)
              return reject("left shift of a possibly negative value"), std::nullopt;
            if (!inShiftRange(*r, w))
              return std::nullopt;
            // x << n is x * 2^n; the largest operands decide overflow.
            I64 factor = 0;
            if (r->hi >= 63 || mulOv(I64(1) << r->hi, 1, factor))
              return reject("shift amount too large to bound"), std::nullopt;
            return mulI(*l, constant(I64(1) << r->hi), w);
          }
          case AtomOpKind::Shr:
            if (!inShiftRange(*r, w))
              return std::nullopt;
            if (l->isConst() && r->isConst())
              return constant(l->lo >> r->lo);
            return unknownOf();
          case AtomOpKind::LShr:
            if (!inShiftRange(*r, w))
              return std::nullopt;
            if (l->isConst() && r->isConst())
              return constant(signExtend(toUnsigned(l->lo, w) >> r->lo, w));
            return unknownOf();
          default: {
            // & | ^ : a sound range needs bit-level reasoning, so only known
            // operands fold. Everything else widens, which is where a
            // bit-level domain would pay off first.
            if (!l->isConst() || !r->isConst())
              return unknownOf();
            const I64 a = l->lo, b = r->lo;
            switch (o.op) {
              case AtomOpKind::And:
                return constant(a & b);
              case AtomOpKind::Or:
                return constant(a | b);
              default:
                return constant(a ^ b);
            }
          }
        }
      }

      bool inShiftRange(const Interval &amount, std::uint32_t bits) {
        if (amount.unknown || amount.lo < 0 || amount.hi >= (I64) bits)
          return reject("shift amount may be out of range");
        return true;
      }

      // --- arithmetic with overflow as a proof obligation --------------------

      // The sum is bounded by summing like ends; the difference is bounded by
      // opposing ends, since subtracting the largest right operand from the
      // smallest left one gives the minimum.
      std::optional<Interval> addI(const Interval &a, const Interval &b, std::uint32_t bits) {
        return combine(a, b, a.lo, b.lo, a.hi, b.hi, bits, addOv, "addition may overflow");
      }

      std::optional<Interval> subI(const Interval &a, const Interval &b, std::uint32_t bits) {
        return combine(a, b, a.lo, b.hi, a.hi, b.lo, bits, subOv, "subtraction may overflow");
      }

      template<typename Op>
      std::optional<Interval> combine(
          const Interval &a, const Interval &b, I64 loL, I64 loR, I64 hiL, I64 hiR,
          std::uint32_t bits, Op op, const char *why
      ) {
        if (a.unknown || b.unknown)
          return reject(std::string(why) + " (an operand is not tracked)"), std::nullopt;
        I64 lo = 0, hi = 0;
        if (op(loL, loR, lo) || op(hiL, hiR, hi))
          return reject(why), std::nullopt;
        Interval out{lo, hi, false};
        if (bits && !fits(out, bits))
          return reject(why), std::nullopt;
        return out;
      }

      std::optional<Interval> mulI(const Interval &a, const Interval &b, std::uint32_t bits) {
        if (a.unknown || b.unknown)
          return reject("multiplication may overflow (an operand is not tracked)"), std::nullopt;
        const I64 ends[4][2] = {{a.lo, b.lo}, {a.lo, b.hi}, {a.hi, b.lo}, {a.hi, b.hi}};
        I64 lo = kI64Max, hi = kI64Min;
        for (const auto &e: ends) {
          I64 p = 0;
          if (mulOv(e[0], e[1], p))
            return reject("multiplication may overflow"), std::nullopt;
          lo = std::min(lo, p);
          hi = std::max(hi, p);
        }
        Interval out{lo, hi, false};
        if (bits && !fits(out, bits))
          return reject("multiplication may overflow"), std::nullopt;
        return out;
      }

      // --- conditions --------------------------------------------------------

      // The interval a condition's two sides can take, or nullopt when either
      // side could trap.
      bool evalCond(const Cond &c) {
        auto l = eval(c.lhs, 0), r = eval(c.rhs, 0);
        return l && r;
      }

      // Is the condition decided the same way for every state in the box?
      bool decides(const PathCheck &chk) {
        auto l = eval(chk.cond.lhs, 0);
        auto r = eval(chk.cond.rhs, 0);
        if (!l || !r)
          return false;
        auto always = holds(chk.cond.op, *l, *r);
        if (!always)
          return reject("condition is not settled by the entry range");
        return *always == chk.taken;
      }

      // True / false when the relation holds for every pair in the two
      // intervals; nullopt when both outcomes are possible.
      std::optional<bool> holds(RelOp op, const Interval &a, const Interval &b) {
        if (a.unknown || b.unknown)
          return std::nullopt;
        switch (op) {
          case RelOp::LT:
            if (a.hi < b.lo)
              return true;
            if (a.lo >= b.hi)
              return false;
            return std::nullopt;
          case RelOp::LE:
            if (a.hi <= b.lo)
              return true;
            if (a.lo > b.hi)
              return false;
            return std::nullopt;
          case RelOp::GT:
            if (a.lo > b.hi)
              return true;
            if (a.hi <= b.lo)
              return false;
            return std::nullopt;
          case RelOp::GE:
            if (a.lo >= b.hi)
              return true;
            if (a.hi < b.lo)
              return false;
            return std::nullopt;
          case RelOp::EQ:
            if (a.isConst() && b.isConst())
              return a.lo == b.lo;
            if (a.hi < b.lo || b.hi < a.lo)
              return false;
            return std::nullopt;
          case RelOp::NE:
            if (a.isConst() && b.isConst())
              return a.lo != b.lo;
            if (a.hi < b.lo || b.hi < a.lo)
              return true;
            return std::nullopt;
        }
        return std::nullopt;
      }

      // --- statements ---------------------------------------------------------

      bool indicesInBounds(const LValue &lv) {
        auto it = types_.find(lv.base.name);
        if (it == types_.end())
          return true; // not a tracked root; nothing to check
        TypePtr t = it->second;
        for (const auto &acc: lv.accesses) {
          if (!t)
            return true;
          if (auto af = std::get_if<AccessField>(&acc)) {
            t = fieldType(t, af->field);
            continue;
          }
          std::uint64_t n = 0;
          TypePtr elem;
          if (const auto *at = TypeUtils::asArray(t)) {
            n = at->size;
            elem = at->elem;
          } else if (const auto *vt = TypeUtils::asVec(t)) {
            n = vt->size;
            elem = vt->elem;
          } else
            return true;
          const auto &idx = std::get<AccessIndex>(acc).index;
          Interval iv = unknownOf();
          if (auto il = std::get_if<IntLit>(&idx))
            iv = constant(il->value);
          else if (auto id = std::get_if<LocalOrSymId>(&idx)) {
            if (auto loc = std::get_if<LocalId>(id)) {
              LValue base;
              base.base = *loc;
              iv = read(base);
            }
          }
          if (iv.unknown || iv.lo < 0 || iv.hi >= (I64) n)
            return reject("index may be out of bounds");
          t = elem;
        }
        return true;
      }

      bool step(const Instr &ins) {
        return std::visit(
            [&](const auto &x) -> bool {
              using T = std::decay_t<decltype(x)>;
              if constexpr (std::is_same_v<T, AssignInstr>) {
                if (!indicesInBounds(x.lhs))
                  return false;
                const std::uint32_t w = widthOf(x.lhs);
                auto v = eval(x.rhs, w);
                if (!v)
                  return false;
                if (w == 0 || !leafKey(x.lhs))
                  forget(x.lhs.base.name);
                else
                  write(x.lhs, *v);
                return true;
              } else if constexpr (std::is_same_v<T, StoreInstr>) {
                // The destination is a pointer value, which is not tracked, so
                // the store may have landed on any leaf.
                if (!eval(x.ptr, 0) || !eval(x.val, 0))
                  return false;
                forgetAll();
                return true;
              } else if constexpr (std::is_same_v<T, RequireInstr>) {
                auto l = eval(x.cond.lhs, 0), r = eval(x.cond.rhs, 0);
                if (!l || !r)
                  return false;
                auto always = holds(x.cond.op, *l, *r);
                if (!always || !*always)
                  return reject("a require is not provable over the entry range");
                return true;
              } else {
                // assume: a solver hint, not a runtime check.
                (void) x;
                return true;
              }
            },
            ins
        );
      }

      // The type of a struct field, or null when the shape does not match.
      TypePtr fieldType(const TypePtr &t, const std::string &field) const {
        const auto *st = TypeUtils::asStruct(t);
        if (!st)
          return nullptr;
        auto it = structs_.find(st->name.name);
        if (it == structs_.end())
          return nullptr;
        for (const auto &f: it->second->fields)
          if (f.name == field)
            return f.type;
        return nullptr;
      }

      IntervalEnv env_;
      const StructMap &structs_;
      std::unordered_map<std::string, TypePtr> types_;
      std::string reason_;
    };

  } // namespace

  IntervalVerdict checkTrace(
      const FunDecl &fn, const StructMap &structs, const TraceBody &body, const IntervalEnv &entry
  ) {
    return Checker(fn, structs, entry).run(body);
  }

} // namespace refractir::reify
