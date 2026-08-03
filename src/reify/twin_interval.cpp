#include "reify/twin_interval.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

#include "analysis/type_utils.hpp"
#include "reify/hyperparameters.hpp"
#include "reify/twin_mini.hpp"

namespace refractir::reify {

  namespace {

    using I64 = std::int64_t;

    constexpr I64 kI64Min = std::numeric_limits<I64>::min();
    constexpr I64 kI64Max = std::numeric_limits<I64>::max();

    Interval unknownOf(std::uint64_t deps = 0) {
      Interval v;
      v.unknown = true;
      v.lo = kI64Min;
      v.hi = kI64Max;
      v.deps = deps;
      return v;
    }

    Interval constant(I64 c) { return Interval{c, c, false, 0, 0}; }

    // Every value carries the entry leaves it came from, so a check that
    // fails can name them.
    Interval withDeps(Interval v, std::uint64_t deps) {
      v.deps = deps;
      return v;
    }

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
      Checker(const FunDecl &fn, const StructMap &structs, const EntryState &entry) :
          env_(entry.ints), floats_(entry.floats), ptrs_(entry.ptrs), structs_(structs) {
        for (const auto &p: fn.params)
          types_[p.name.name] = p.type;
        for (const auto &l: fn.lets)
          types_[l.name.name] = l.type;
        // One bit per integer entry leaf, in a stable order so two runs blame
        // the same leaves. Past 64 leaves the rest share "no bit", which costs
        // blame precision and nothing else.
        std::vector<std::string> keys;
        keys.reserve(env_.size());
        for (const auto &[k, v]: env_)
          keys.push_back(k);
        std::sort(keys.begin(), keys.end());
        for (std::size_t i = 0; i < keys.size() && i < 64; ++i) {
          bitOf_[keys[i]] = std::uint64_t(1) << i;
          leafOfBit_.push_back(keys[i]);
          env_[keys[i]].deps = std::uint64_t(1) << i;
        }
      }

      std::vector<std::string> blamed() const {
        std::vector<std::string> out;
        for (std::size_t i = 0; i < leafOfBit_.size(); ++i)
          if (blame_ & (std::uint64_t(1) << i))
            out.push_back(leafOfBit_[i]);
        return out;
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
        return IntervalVerdict{true, "", {}};
      }

    private:
      IntervalVerdict fail(const std::string &why) { return IntervalVerdict{false, why, blamed()}; }

      bool reject(std::string why, std::uint64_t deps = 0) {
        reason_ = std::move(why);
        blame_ = deps;
        return false;
      }

      // --- types -----------------------------------------------------------

      // The declared type of the cell an lvalue names, or null when the path
      // does not fit the root's shape.
      TypePtr typeOf(const LValue &lv) const {
        auto it = types_.find(lv.base.name);
        if (it == types_.end())
          return nullptr;
        TypePtr t = it->second;
        for (const auto &acc: lv.accesses) {
          if (!t)
            return nullptr;
          if (auto af = std::get_if<AccessField>(&acc)) {
            t = fieldType(t, af->field);
            continue;
          }
          if (const auto *at = TypeUtils::asArray(t))
            t = at->elem;
          else if (const auto *vt = TypeUtils::asVec(t))
            t = vt->elem;
          else
            return nullptr;
        }
        return t;
      }

      // Width of the value an lvalue names, or 0 when it is not a scalar
      // integer (floats, pointers, whole aggregates and vectors).
      std::uint32_t widthOf(const LValue &lv) const {
        auto bits = TypeUtils::getIntBitWidth(typeOf(lv));
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

      // --- pointer provenance -----------------------------------------------
      //
      // A pointer's *value* is not a number the interval domain can carry, but
      // what it points at usually has a name. Following that name is what lets
      // a value survive a round trip through memory: `store %p, v` writes the
      // leaf `%p` names, and the `load %p` that reads it back finds it there.
      // A pointer whose target cannot be named — one read out of memory, or
      // moved by an unknown amount — is simply absent from the map, and every
      // access through it falls back to "unknown".

      // Rewrite an lvalue's indices into literals, so two spellings of one
      // cell (`%a[2]` and `%a[%i]` with `%i` = 2) name the same leaf. Fails
      // when an index is not known.
      std::optional<LValue> resolve(const LValue &lv) {
        LValue out;
        out.base = lv.base;
        for (const auto &acc: lv.accesses) {
          if (auto af = std::get_if<AccessField>(&acc)) {
            out.accesses.push_back(*af);
            continue;
          }
          const auto &idx = std::get<AccessIndex>(acc).index;
          if (std::holds_alternative<IntLit>(idx)) {
            out.accesses.push_back(std::get<AccessIndex>(acc));
            continue;
          }
          auto loc = std::get_if<LocalId>(&std::get<LocalOrSymId>(idx));
          if (!loc)
            return std::nullopt;
          LValue base;
          base.base = *loc;
          const Interval v = read(base);
          if (!v.isConst())
            return std::nullopt;
          out.accesses.push_back(AccessIndex{IntLit{v.lo, {}}, {}});
        }
        return out;
      }

      // Move a target by `delta` elements, which only has a name when the
      // target is an element of something.
      std::optional<LValue> shift(const LValue &lv, I64 delta) {
        if (lv.accesses.empty())
          return std::nullopt;
        auto ai = std::get_if<AccessIndex>(&lv.accesses.back());
        if (!ai)
          return std::nullopt;
        auto il = std::get_if<IntLit>(&ai->index);
        if (!il)
          return std::nullopt;
        LValue out = lv;
        out.accesses.back() = AccessIndex{IntLit{il->value + delta, {}}, {}};
        return out;
      }

      // Where the expression points, or nullopt when the domain cannot say.
      // `null` is reported as a resolved target with no lvalue, so a
      // dereference through it can be rejected rather than merely unknown.
      std::optional<std::optional<LValue>> evalPtr(const Expr &e) {
        auto base = evalPtrAtom(e.first);
        if (!base)
          return std::nullopt;
        if (e.rest.empty())
          return base;
        if (!*base)
          return std::nullopt; // arithmetic on null
        LValue cur = **base;
        for (const auto &t: e.rest) {
          auto off = evalAtom(t.atom, 0);
          if (!off || !off->isConst())
            return std::nullopt;
          auto moved = shift(cur, t.op == AddOp::Plus ? off->lo : -off->lo);
          if (!moved)
            return std::nullopt;
          cur = *moved;
        }
        return std::optional<LValue>(cur);
      }

      std::optional<std::optional<LValue>> evalPtrAtom(const Atom &a) {
        return std::visit(
            [&](const auto &x) -> std::optional<std::optional<LValue>> {
              using T = std::decay_t<decltype(x)>;
              if constexpr (std::is_same_v<T, AddrAtom>) {
                auto lv = resolve(x.lv);
                if (!lv)
                  return std::nullopt;
                return std::optional<LValue>(*lv);
              } else if constexpr (std::is_same_v<T, CoefAtom>) {
                if (std::holds_alternative<NullLit>(x.coef))
                  return std::optional<LValue>(std::nullopt);
                if (auto id = std::get_if<LocalOrSymId>(&x.coef))
                  if (auto loc = std::get_if<LocalId>(id)) {
                    LValue lv;
                    lv.base = *loc;
                    return lookupPtr(lv);
                  }
                return std::nullopt;
              } else if constexpr (std::is_same_v<T, RValueAtom>)
                return lookupPtr(x.rval);
              else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
                auto base = lookupPtr(x.rval);
                if (!base || !*base)
                  return std::nullopt;
                LValue out = **base;
                if (std::holds_alternative<IntLit>(x.index))
                  out.accesses.push_back(AccessIndex{std::get<IntLit>(x.index), {}});
                else {
                  auto loc = std::get_if<LocalId>(&std::get<LocalOrSymId>(x.index));
                  if (!loc)
                    return std::nullopt;
                  LValue b;
                  b.base = *loc;
                  const Interval v = read(b);
                  if (!v.isConst())
                    return std::nullopt;
                  out.accesses.push_back(AccessIndex{IntLit{v.lo, {}}, {}});
                }
                return std::optional<LValue>(out);
              } else if constexpr (std::is_same_v<T, PtrFieldAtom>) {
                auto base = lookupPtr(x.rval);
                if (!base || !*base)
                  return std::nullopt;
                LValue out = **base;
                out.accesses.push_back(AccessField{x.field, {}});
                return std::optional<LValue>(out);
              } else
                // A pointer out of memory, out of a call, or out of a select:
                // no name to follow.
                return std::nullopt;
            },
            a.v
        );
      }

      std::optional<std::optional<LValue>> lookupPtr(const LValue &lv) {
        auto key = leafKey(lv);
        if (!key)
          return std::nullopt;
        auto it = ptrs_.find(*key);
        if (it == ptrs_.end())
          return std::nullopt;
        return it->second;
      }

      void setPtr(const LValue &lv, std::optional<std::optional<LValue>> target) {
        auto key = leafKey(lv);
        if (!key)
          return;
        if (target)
          ptrs_[*key] = *target;
        else
          ptrs_.erase(*key);
      }

      bool isPointer(const LValue &lv) const {
        TypePtr t = typeOf(lv);
        return t && std::holds_alternative<PtrType>(t->v);
      }

      // The cell a pointer expression names, ready to be read or written.
      // Rejects a dereference the trace could not perform safely.
      std::optional<LValue> deref(const Expr &e, const char *what) {
        auto target = evalPtr(e);
        if (!target)
          return std::nullopt; // unknown target: the caller widens instead
        if (!*target) {
          reject(std::string("a ") + what + " through null");
          return std::nullopt;
        }
        if (!indicesInBounds(**target)) {
          reject(std::string("a ") + what + " may be out of bounds");
          return std::nullopt;
        }
        return **target;
      }

      // --- floats ------------------------------------------------------------
      //
      // Exact values only, so evaluation is concrete arithmetic with the
      // strict-UB rules applied: a result that is not finite is UB (spec
      // §7.4), as is division by zero. Single precision rounds after each
      // operation, which is correctly rounded because a double holds every
      // intermediate of two floats exactly.

      static bool finite(double d) { return std::isfinite(d); }

      static double round(double d, std::uint32_t bits) {
        return bits == 32 ? (double) (float) d : d;
      }

      // Is this lvalue a float, and how wide?
      std::uint32_t floatWidthOf(const LValue &lv) const {
        TypePtr t = typeOf(lv);
        auto bits = TypeUtils::getFloatBitWidth(t);
        return bits ? *bits : 0;
      }

      std::optional<double> readFloat(const LValue &lv) {
        auto key = leafKey(lv);
        if (!key)
          return std::nullopt;
        auto it = floats_.find(*key);
        return it == floats_.end() ? std::nullopt : std::optional<double>(it->second);
      }

      // Evaluate a float expression exactly, or nullopt when a value is not
      // tracked. `bad` is set when the trace could not be proven safe, which a
      // caller must distinguish from a merely unknown value.
      std::optional<double> evalFloatExpr(const Expr &e, std::uint32_t bits, bool &bad) {
        auto acc = evalFloatAtom(e.first, bits, bad);
        if (!acc)
          return std::nullopt;
        for (const auto &t: e.rest) {
          auto rhs = evalFloatAtom(t.atom, bits, bad);
          if (!rhs)
            return std::nullopt;
          const double v = round(t.op == AddOp::Plus ? *acc + *rhs : *acc - *rhs, bits);
          if (!finite(v)) {
            reject("a float sum leaves the finite range");
            bad = true;
            return std::nullopt;
          }
          acc = v;
        }
        return acc;
      }

      std::optional<double> evalFloatCoef(const Coef &c) {
        if (auto fl = std::get_if<FloatLit>(&c))
          return fl->value;
        if (auto il = std::get_if<IntLit>(&c))
          return (double) il->value;
        if (auto id = std::get_if<LocalOrSymId>(&c))
          if (auto loc = std::get_if<LocalId>(id)) {
            LValue lv;
            lv.base = *loc;
            return readFloat(lv);
          }
        return std::nullopt;
      }

      std::optional<double> evalFloatAtom(const Atom &a, std::uint32_t bits, bool &bad) {
        return std::visit(
            [&](const auto &x) -> std::optional<double> {
              using T = std::decay_t<decltype(x)>;
              if constexpr (std::is_same_v<T, CoefAtom>)
                return evalFloatCoef(x.coef);
              else if constexpr (std::is_same_v<T, RValueAtom>) {
                if (!indicesInBounds(x.rval))
                  return bad = true, std::nullopt;
                return readFloat(x.rval);
              } else if constexpr (std::is_same_v<T, LoadAtom>) {
                Expr p;
                p.first = Atom{RValueAtom{x.rval, {}}, {}};
                auto cell = deref(p, "load");
                if (!cell)
                  return bad = !reason_.empty(), std::nullopt;
                return readFloat(*cell);
              } else if constexpr (std::is_same_v<T, CastAtom>) {
                // int -> float: exact for every value a double can hold.
                if (auto il = std::get_if<IntLit>(&x.src))
                  return round((double) il->value, bits);
                if (auto lv = std::get_if<LValue>(&x.src)) {
                  if (auto f = readFloat(*lv))
                    return round(*f, bits);
                  const Interval v = read(*lv);
                  if (v.isConst())
                    return round((double) v.lo, bits);
                }
                if (auto fl = std::get_if<FloatLit>(&x.src))
                  return round(fl->value, bits);
                return std::nullopt;
              } else if constexpr (std::is_same_v<T, OpAtom>)
                return evalFloatOp(x, bits, bad);
              else if constexpr (std::is_same_v<T, SelectAtom>) {
                auto t = evalFloatSelectVal(x.vtrue), f = evalFloatSelectVal(x.vfalse);
                // Which arm runs is not decided here, so only agreement helps.
                if (t && f && *t == *f)
                  return *t;
                return std::nullopt;
              } else
                return std::nullopt;
            },
            a.v
        );
      }

      std::optional<double> evalFloatSelectVal(const SelectVal &sv) {
        if (auto rv = std::get_if<RValue>(&sv))
          return readFloat(*rv);
        return evalFloatCoef(std::get<Coef>(sv));
      }

      std::optional<double> evalFloatOp(const OpAtom &o, std::uint32_t bits, bool &bad) {
        auto l = evalFloatCoef(o.coef);
        if (!l)
          return std::nullopt;
        if (!indicesInBounds(o.rval))
          return bad = true, std::nullopt;
        auto r = readFloat(o.rval);
        if (!r)
          return std::nullopt;
        double v = 0.0;
        switch (o.op) {
          case AtomOpKind::Mul:
            v = *l * *r;
            break;
          case AtomOpKind::Div:
            if (*r == 0.0) {
              reject("a float divisor is zero");
              bad = true;
              return std::nullopt;
            }
            v = *l / *r;
            break;
          case AtomOpKind::Mod:
            if (*r == 0.0) {
              reject("a float modulus is zero");
              bad = true;
              return std::nullopt;
            }
            v = std::fmod(*l, *r);
            break;
          default:
            return std::nullopt; // bitwise operators are not float operations
        }
        v = round(v, bits);
        if (!finite(v)) {
          reject("a float result leaves the finite range");
          bad = true;
          return std::nullopt;
        }
        return v;
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
                  return unknownOf(v->deps);
                // ~ is monotone decreasing
                return Interval{~v->hi, ~v->lo, false, 0, v->deps};
              } else if constexpr (std::is_same_v<T, CmpAtom>) {
                // i1 true is -1 (spec §6.4), so a comparison is one of {0, -1}.
                auto l = evalSelectVal(x.lhs, 0), r = evalSelectVal(x.rhs, 0);
                if (!l || !r)
                  return std::nullopt;
                return Interval{-1, 0, false, 0, l->deps | r->deps};
              } else if constexpr (std::is_same_v<T, SelectAtom>)
                return evalSelect(x, bits);
              else if constexpr (std::is_same_v<T, CastAtom>)
                return evalCast(x);
              else if constexpr (std::is_same_v<T, OpAtom>)
                return evalOp(x, bits);
              else if constexpr (std::is_same_v<T, LoadAtom>) {
                Expr p;
                p.first = Atom{RValueAtom{x.rval, {}}, {}};
                auto cell = deref(p, "load");
                if (!cell)
                  return reason_.empty() ? std::optional<Interval>(unknownOf()) : std::nullopt;
                return read(*cell);
              } else
                // addr / ptrindex / ptrfield / call: not a number.
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
        const std::uint64_t deps = t->deps | f->deps;
        if (t->unknown || f->unknown)
          return unknownOf(deps);
        return Interval{std::min(t->lo, f->lo), std::max(t->hi, f->hi), false, 0, deps};
      }

      std::optional<Interval> evalCast(const CastAtom &c) {
        auto bits = TypeUtils::getIntBitWidth(c.dstType);
        if (!bits)
          return unknownOf(); // to float
        // From a float: the value truncates toward zero, and landing outside
        // the destination's range is UB (spec §7.4).
        if (auto lv = std::get_if<LValue>(&c.src))
          if (floatWidthOf(*lv)) {
            auto f = readFloat(*lv);
            if (!f)
              return unknownOf();
            const double t = std::trunc(*f);
            I64 lo = 0, hi = 0;
            if (!rangeOf(*bits, lo, hi) || t < (double) lo || t > (double) hi)
              return reject("a float-to-integer cast may leave the range"), std::nullopt;
            return constant((I64) t);
          }
        if (auto fl = std::get_if<FloatLit>(&c.src)) {
          const double t = std::trunc(fl->value);
          I64 lo = 0, hi = 0;
          if (!rangeOf(*bits, lo, hi) || t < (double) lo || t > (double) hi)
            return reject("a float-to-integer cast may leave the range"), std::nullopt;
          return constant((I64) t);
        }
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
        return fits(*src, *bits) ? *src : unknownOf(src->deps);
      }

      std::optional<Interval> evalOp(const OpAtom &o, std::uint32_t bits) {
        auto l = evalCoef(o.coef, bits);
        if (!l)
          return std::nullopt;
        auto r = readChecked(o.rval);
        if (!r)
          return std::nullopt;
        const std::uint64_t deps = l->deps | r->deps;
        const std::uint32_t w = bits ? bits : widthOf(o.rval);
        switch (o.op) {
          case AtomOpKind::Mul:
            return mulI(*l, *r, w);
          case AtomOpKind::Div:
          case AtomOpKind::Mod: {
            if (r->unknown || (r->lo <= 0 && r->hi >= 0))
              return reject("divisor may be zero", deps), std::nullopt;
            // INT_MIN / -1 is the one division that overflows.
            I64 lo = 0, hi = 0;
            if (rangeOf(w, lo, hi) && (l->unknown || l->lo <= lo) && r->lo <= -1 && r->hi >= -1)
              return reject("division may overflow (MIN / -1)", deps), std::nullopt;
            if (l->isConst() && r->isConst())
              return withDeps(
                  constant(o.op == AtomOpKind::Div ? l->lo / r->lo : l->lo % r->lo), deps
              );
            return unknownOf(deps);
          }
          case AtomOpKind::Shl: {
            if (l->unknown || l->lo < 0)
              return reject("left shift of a possibly negative value", deps), std::nullopt;
            if (!inShiftRange(*r, w))
              return std::nullopt;
            // x << n is x * 2^n; the largest operands decide overflow.
            I64 factor = 0;
            if (r->hi >= 63 || mulOv(I64(1) << r->hi, 1, factor))
              return reject("shift amount too large to bound", deps), std::nullopt;
            return mulI(*l, constant(I64(1) << r->hi), w);
          }
          case AtomOpKind::Shr:
            if (!inShiftRange(*r, w))
              return std::nullopt;
            if (l->isConst() && r->isConst())
              return withDeps(constant(l->lo >> r->lo), deps);
            return unknownOf(deps);
          case AtomOpKind::LShr:
            if (!inShiftRange(*r, w))
              return std::nullopt;
            if (l->isConst() && r->isConst())
              return withDeps(constant(signExtend(toUnsigned(l->lo, w) >> r->lo, w)), deps);
            return unknownOf(deps);
          default: {
            // & | ^ : a sound range needs bit-level reasoning, so only known
            // operands fold. Everything else widens, which is where a
            // bit-level domain would pay off first.
            if (!l->isConst() || !r->isConst())
              return unknownOf(deps);
            const I64 a = l->lo, b = r->lo;
            switch (o.op) {
              case AtomOpKind::And:
                return withDeps(constant(a & b), deps);
              case AtomOpKind::Or:
                return withDeps(constant(a | b), deps);
              default:
                return withDeps(constant(a ^ b), deps);
            }
          }
        }
      }

      bool inShiftRange(const Interval &amount, std::uint32_t bits) {
        if (amount.unknown || amount.lo < 0 || amount.hi >= (I64) bits)
          return reject("shift amount may be out of range", amount.deps);
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
        const std::uint64_t deps = a.deps | b.deps;
        if (a.unknown || b.unknown)
          return reject(std::string(why) + " (an operand is not tracked)", deps), std::nullopt;
        I64 lo = 0, hi = 0;
        if (op(loL, loR, lo) || op(hiL, hiR, hi))
          return reject(why, deps), std::nullopt;
        Interval out{lo, hi, false, 0, deps};
        if (bits && !fits(out, bits))
          return reject(why, deps), std::nullopt;
        return out;
      }

      std::optional<Interval> mulI(const Interval &a, const Interval &b, std::uint32_t bits) {
        const std::uint64_t deps = a.deps | b.deps;
        if (a.unknown || b.unknown)
          return reject("multiplication may overflow (an operand is not tracked)", deps),
                 std::nullopt;
        const I64 ends[4][2] = {{a.lo, b.lo}, {a.lo, b.hi}, {a.hi, b.lo}, {a.hi, b.hi}};
        I64 lo = kI64Max, hi = kI64Min;
        for (const auto &e: ends) {
          I64 p = 0;
          if (mulOv(e[0], e[1], p))
            return reject("multiplication may overflow", deps), std::nullopt;
          lo = std::min(lo, p);
          hi = std::max(hi, p);
        }
        Interval out{lo, hi, false, 0, deps};
        if (bits && !fits(out, bits))
          return reject("multiplication may overflow", deps), std::nullopt;
        return out;
      }

      // --- conditions --------------------------------------------------------

      // The interval a condition's two sides can take, or nullopt when either
      // side could trap.
      bool evalCond(const Cond &c) {
        auto l = eval(c.lhs, 0), r = eval(c.rhs, 0);
        return l && r;
      }

      // A comparison of two known floats is decided exactly.
      std::optional<bool> floatHolds(const Cond &c) {
        bool bad = false;
        auto l = evalFloatExpr(c.lhs, 64, bad);
        if (bad)
          return std::nullopt;
        auto r = evalFloatExpr(c.rhs, 64, bad);
        if (bad || !l || !r)
          return std::nullopt;
        switch (c.op) {
          case RelOp::LT:
            return *l < *r;
          case RelOp::LE:
            return *l <= *r;
          case RelOp::GT:
            return *l > *r;
          case RelOp::GE:
            return *l >= *r;
          case RelOp::EQ:
            return *l == *r;
          case RelOp::NE:
            return *l != *r;
        }
        return std::nullopt;
      }

      // Is the condition decided the same way for every state in the box?
      bool decides(const PathCheck &chk) {
        if (auto f = floatHolds(chk.cond))
          return *f == chk.taken;
        auto l = eval(chk.cond.lhs, 0);
        auto r = eval(chk.cond.rhs, 0);
        if (!l || !r)
          return false;
        auto always = holds(chk.cond.op, *l, *r);
        if (!always)
          return reject("condition is not settled by the entry range", l->deps | r->deps);
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
            return reject("index may be out of bounds", iv.deps);
          t = elem;
        }
        return true;
      }

      // Assigning a float: an unproven *operation* stops the trace, while a
      // merely unknown value just makes the destination unknown.
      bool assignFloat(const LValue &lhs, const Expr &rhs, std::uint32_t fw) {
        bool bad = false;
        auto v = evalFloatExpr(rhs, fw, bad);
        if (bad)
          return false;
        auto key = leafKey(lhs);
        if (!key)
          return true;
        if (v)
          floats_[*key] = *v;
        else
          floats_.erase(*key);
        return true;
      }

      bool step(const Instr &ins) {
        return std::visit(
            [&](const auto &x) -> bool {
              using T = std::decay_t<decltype(x)>;
              if constexpr (std::is_same_v<T, AssignInstr>) {
                if (!indicesInBounds(x.lhs))
                  return false;
                if (isPointer(x.lhs)) {
                  setPtr(x.lhs, evalPtr(x.rhs));
                  return true;
                }
                if (const std::uint32_t fw = floatWidthOf(x.lhs))
                  return assignFloat(x.lhs, x.rhs, fw);
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
                auto cell = deref(x.ptr, "store");
                if (!cell && !reason_.empty())
                  return false;
                if (cell) {
                  if (const std::uint32_t fw = floatWidthOf(*cell))
                    return assignFloat(*cell, x.val, fw);
                }
                auto v = eval(x.val, cell ? widthOf(*cell) : 0);
                if (!v)
                  return false;
                // Without a name for the destination, the store may have
                // landed on any leaf.
                if (cell)
                  write(*cell, *v);
                else
                  forgetAll();
                return true;
              } else if constexpr (std::is_same_v<T, RequireInstr>) {
                if (auto f = floatHolds(x.cond))
                  return *f ? true : reject("a require is false over the entry range");
                auto l = eval(x.cond.lhs, 0), r = eval(x.cond.rhs, 0);
                if (!l || !r)
                  return false;
                auto always = holds(x.cond.op, *l, *r);
                if (!always || !*always)
                  return reject(
                      "a require is not provable over the entry range", l->deps | r->deps
                  );
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
      std::unordered_map<std::string, std::uint64_t> bitOf_;
      std::vector<std::string> leafOfBit_;
      std::uint64_t blame_ = 0;
      // Float leaf -> its exact value. Absent means unknown; there is no
      // "range of floats" here by design (see the header).
      FloatEnv floats_;
      // Pointer leaf -> the cell it names (an empty optional is the null
      // pointer). A pointer absent from the map points somewhere unknown.
      PtrEnv ptrs_;
      const StructMap &structs_;
      std::unordered_map<std::string, TypePtr> types_;
      std::string reason_;
    };

  } // namespace

  IntervalVerdict checkTrace(
      const FunDecl &fn, const StructMap &structs, const TraceBody &body, const EntryState &entry
  ) {
    return Checker(fn, structs, entry).run(body);
  }

  Box computeBox(
      const FunDecl &fn, const StructMap &structs, const TraceBody &body, const EntryState &entry,
      std::mt19937 &rng
  ) {
    Box box;
    auto judge = [&](const EntryState &st) {
      ++box.passes;
      return checkTrace(fn, structs, body, st);
    };

    // Step 1 — the floor. Everything pinned is the guard rytwin already
    // emits, so if the pass cannot even prove that, no widening is possible
    // and the answer is the floor itself.
    for (const auto &[k, v]: entry.ints)
      box.leaves.push_back(BoxLeaf{k, LeafClass::Pinned, v});
    std::sort(box.leaves.begin(), box.leaves.end(), [](const BoxLeaf &a, const BoxLeaf &b) {
      return a.key < b.key;
    });
    if (!judge(entry).ok)
      return box;

    // Step 2 — free leaves. A leaf is free exactly when the trace is provable
    // without knowing it, which is one pass with the leaf left out. This is
    // the ceiling of the whole search: nothing wider than "any value" exists,
    // so a leaf that clears it never enters the bisection below.
    EntryState st = entry;
    for (auto &leaf: box.leaves) {
      EntryState without = st;
      without.ints.erase(leaf.key);
      if (judge(without).ok) {
        leaf.cls = LeafClass::Free;
        st = std::move(without);
      }
    }

    // Step 3 — lockstep bisection over what is left. Each open leaf carries a
    // radius; every round doubles all of them at once and one pass judges the
    // result, so widening costs a pass per round rather than per leaf. A
    // refused round freezes only the leaves its failing check depended on —
    // the others keep growing, which is what stops a leaf's width from
    // depending on the order it happened to be visited in.
    struct Open {
      BoxLeaf *leaf;
      std::int64_t centre;
      std::int64_t good = 0; // widest radius proven
      std::int64_t next = 1; // radius to try
      bool frozen = false;
    };

    std::vector<Open> open;
    for (auto &leaf: box.leaves)
      if (leaf.cls != LeafClass::Free)
        open.push_back(Open{&leaf, leaf.range.lo, 0, 1, false});
    if (open.empty())
      return box;

    // A leaf cannot hold what its type cannot represent, so every proposed
    // range is clipped to its own width — otherwise the search would "prove" a
    // range the guard could not even state as a literal.
    auto clip = [](const Interval &leaf, std::int64_t lo, std::int64_t hi) {
      I64 tlo = kI64Min, thi = kI64Max;
      if (leaf.bits && leaf.bits < 64) {
        tlo = -(I64(1) << (leaf.bits - 1));
        thi = (I64(1) << (leaf.bits - 1)) - 1;
      }
      return std::pair<I64, I64>{std::max(lo, tlo), std::min(hi, thi)};
    };

    auto widen = [&](const std::vector<std::int64_t> &radii) {
      EntryState trial = st;
      for (std::size_t i = 0; i < open.size(); ++i) {
        Interval iv = open[i].leaf->range;
        const auto [lo, hi] = clip(iv, open[i].centre - radii[i], open[i].centre + radii[i]);
        iv.lo = lo;
        iv.hi = hi;
        iv.unknown = false;
        trial.ints[open[i].leaf->key] = iv;
      }
      return trial;
    };

    for (std::size_t round = 0; round < rytwin::hp::kTwinBisectMaxRounds; ++round) {
      std::vector<std::int64_t> radii;
      bool anyOpen = false;
      for (auto &o: open) {
        radii.push_back(o.frozen ? o.good : o.next);
        anyOpen = anyOpen || !o.frozen;
      }
      if (!anyOpen)
        break;
      const IntervalVerdict v = judge(widen(radii));
      if (v.ok) {
        for (auto &o: open)
          if (!o.frozen) {
            o.good = o.next;
            o.next = o.next * 2;
          }
        continue;
      }
      // Freeze what this failure actually depended on. A failure that blames
      // nothing is not about any leaf we are moving, so it freezes everything
      // — continuing would just retry the same refusal.
      bool froze = false;
      for (auto &o: open)
        if (!o.frozen && std::find(v.blame.begin(), v.blame.end(), o.leaf->key) != v.blame.end()) {
          o.frozen = true;
          froze = true;
        }
      // A refusal that names nothing still open is not about the leaves being
      // moved — retrying would only repeat it, so the round ends the search.
      if (!froze)
        for (auto &o: open)
          o.frozen = true;
    }

    // Step 4 — settle each frozen leaf between its last proven radius and the
    // one that failed, so a run does not stop at whatever power of two it
    // happened to reach.
    for (auto &o: open) {
      std::int64_t lo = o.good, hi = o.next;
      while (lo + 1 < hi) {
        const std::int64_t mid = lo + (hi - lo) / 2;
        std::vector<std::int64_t> radii;
        for (auto &p: open)
          radii.push_back(&p == &o ? mid : p.good);
        if (judge(widen(radii)).ok)
          lo = mid;
        else
          hi = mid;
      }
      o.good = lo;
    }

    // Trim each run by a small seeded amount: narrowing is always sound, and
    // maximal runs would make every guard for a given region identical.
    std::uniform_int_distribution<int> trim(0, rytwin::hp::kTwinBoxTrimPct);
    for (auto &o: open) {
      if (o.good <= 0)
        continue;
      const std::int64_t cut = o.good * trim(rng) / 100;
      o.good -= cut;
      if (o.good <= 0)
        continue;
      const auto [lo, hi] = clip(o.leaf->range, o.centre - o.good, o.centre + o.good);
      o.leaf->range.lo = lo;
      o.leaf->range.hi = hi;
      o.leaf->range.unknown = false;
      // A run covering the leaf's whole type admits every value it can hold,
      // which is what free means — and says it without two dead comparisons.
      const auto [tlo, thi] = clip(o.leaf->range, kI64Min, kI64Max);
      o.leaf->cls = (lo <= tlo && hi >= thi) ? LeafClass::Free : LeafClass::Ranged;
    }
    return box;
  }

} // namespace refractir::reify
