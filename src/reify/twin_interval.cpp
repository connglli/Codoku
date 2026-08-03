#include "reify/twin_interval.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

#include "analysis/type_utils.hpp"
#include "ast/clone.hpp"
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

    // The range of an operation that is monotone in each operand separately:
    // whatever it does in between, its extremes are at the corners of the two
    // ranges, so evaluating it four times bounds it.
    template<typename Op>
    Interval corners(const Interval &a, const Interval &b, Op op) {
      const I64 vs[4] = {op(a.lo, b.lo), op(a.lo, b.hi), op(a.hi, b.lo), op(a.hi, b.hi)};
      I64 lo = vs[0], hi = vs[0];
      for (const I64 v: vs) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
      return Interval{lo, hi, false, 0, 0};
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
      Checker(
          const FunDecl &fn, const StructMap &structs, const EntryState &entry, bool record = false
      ) :
          env_(entry.ints), record_(record), floats_(entry.floats), ptrs_(entry.ptrs),
          structs_(structs) {
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

      // What every leaf held before each statement. Only recorded on request:
      // the check itself runs hundreds of times while a box is searched, and
      // only the ceiling pass reads this back.
      const std::vector<IntervalEnv> &snapshots() const { return snaps_; }

      IntervalVerdict run(const TraceBody &body) {
        std::size_t next = 0; // next path check to consult
        for (std::size_t i = 0; i < body.stmts.size(); ++i) {
          if (record_)
            snaps_.push_back(env_);
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

      // Indices are resolved through the environment first, so `%a[%i]` with
      // `%i` known names the same cell as `%a[2]` does. Without that, every
      // array access through a variable index was a dead end — and since one
      // unknown poisons the arithmetic that reads it, a single such access
      // made a whole region unprovable.
      Interval read(const LValue &lv) {
        auto r = resolve(lv);
        if (!r)
          return unknownOf();
        auto key = leafKey(*r);
        if (!key)
          return unknownOf();
        auto it = env_.find(*key);
        return it == env_.end() ? unknownOf() : it->second;
      }

      void write(const LValue &lv, const Interval &v) {
        auto r = resolve(lv);
        if (!r)
          return forget(lv.base.name); // the destination could be any cell
        if (auto key = leafKey(*r))
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
          if (!off)
            return std::nullopt;
          // Pointer arithmetic that leaves the object is UB (§7.5), so an
          // offset the domain cannot bound is a refusal, not an unknown
          // pointer: the value may be fine and the *arithmetic* still not be.
          if (!off->isConst()) {
            reject("pointer arithmetic by an unknown amount", off->deps);
            return std::nullopt;
          }
          auto moved = shift(cur, t.op == AddOp::Plus ? off->lo : -off->lo);
          if (!moved || !indicesInBounds(*moved)) {
            if (reason_.empty())
              reject("pointer arithmetic may leave its object", off->deps);
            return std::nullopt;
          }
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
        auto r = resolve(lv);
        if (!r)
          return std::nullopt;
        auto key = leafKey(*r);
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
      // The width an expression computes in, when the caller does not already
      // know it. A condition or a require has no destination to take it from,
      // and without a width the overflow check below is skipped — which is how
      // `%a + %b > 0` came to be treated as safe for any %a and %b.
      std::uint32_t inferWidth(const Expr &e) const {
        auto ofAtom = [&](const Atom &a) -> std::uint32_t {
          return std::visit(
              [&](const auto &x) -> std::uint32_t {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, RValueAtom>)
                  return widthOf(x.rval);
                else if constexpr (std::is_same_v<T, UnaryAtom>)
                  return widthOf(x.rval);
                else if constexpr (std::is_same_v<T, OpAtom>)
                  return widthOf(x.rval);
                else if constexpr (std::is_same_v<T, CoefAtom>) {
                  if (auto id = std::get_if<LocalOrSymId>(&x.coef))
                    if (auto loc = std::get_if<LocalId>(id)) {
                      LValue lv;
                      lv.base = *loc;
                      return widthOf(lv);
                    }
                  return 0;
                } else if constexpr (std::is_same_v<T, CastAtom>) {
                  auto b = TypeUtils::getIntBitWidth(x.dstType);
                  return b ? *b : 0;
                } else
                  return 0;
              },
              a.v
          );
        };
        if (std::uint32_t w = ofAtom(e.first))
          return w;
        for (const auto &t: e.rest)
          if (std::uint32_t w = ofAtom(t.atom))
            return w;
        return 0;
      }

      std::optional<Interval> eval(const Expr &e, std::uint32_t bits) {
        if (bits == 0)
          bits = inferWidth(e);
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
              } else if constexpr (std::is_same_v<T, AddrAtom>) {
                // The value is a pointer, but the indices on the way to it are
                // evaluated all the same, and still have to be in bounds.
                if (!indicesInBounds(x.lv))
                  return std::nullopt;
                return unknownOf();
              } else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
                if (!readChecked(x.rval))
                  return std::nullopt;
                if (auto id = std::get_if<LocalOrSymId>(&x.index))
                  if (auto loc = std::get_if<LocalId>(id)) {
                    LValue lv;
                    lv.base = *loc;
                    if (!readChecked(lv))
                      return std::nullopt;
                  }
                return unknownOf();
              } else if constexpr (std::is_same_v<T, PtrFieldAtom>) {
                if (!readChecked(x.rval))
                  return std::nullopt;
                return unknownOf();
              } else if constexpr (std::is_same_v<T, CallAtom>) {
                // An intrinsic's result is not tracked, but its arguments are
                // ordinary expressions and can trap on the way in.
                for (const auto &arg: x.args)
                  if (arg && !eval(*arg, 0))
                    return std::nullopt;
                return unknownOf();
              } else
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
              // An untracked float could truncate to anything, including
              // outside the destination — which is UB, not an unknown value.
              return reject("a float-to-integer cast of an untracked value"), std::nullopt;
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
            if (r->hi >= 63)
              return reject("shift amount too large to bound", deps), std::nullopt;
            // x << n is x * 2^n, and the shift amount may itself be a range:
            // the *smallest* shift of the smallest operand is the low end and
            // the largest of the largest is the high end. Using one shift for
            // both would claim a narrower result than the trace can produce,
            // which is how a branch on a shifted value came to look settled.
            I64 lo = 0, hi = 0;
            if (mulOv(l->lo, I64(1) << r->lo, lo) || mulOv(l->hi, I64(1) << r->hi, hi))
              return reject("left shift may overflow", deps), std::nullopt;
            Interval out{lo, hi, false, 0, deps};
            if (w && !fits(out, w))
              return reject("left shift may overflow", deps), std::nullopt;
            return out;
          }
          case AtomOpKind::Shr:
            if (!inShiftRange(*r, w))
              return std::nullopt;
            // Arithmetic shift right is division by a power of two rounding
            // toward -inf, and it is monotone in each operand separately — so
            // the result's extremes are among the four corners of the two
            // ranges. Bounding it matters more than the precision: an unknown
            // poisons every addition downstream of it, and with it the leaf
            // that fed the shift.
            if (l->unknown || r->unknown)
              return unknownOf(deps);
            return withDeps(corners(*l, *r, [](I64 a, I64 b) { return a >> b; }), deps);
          case AtomOpKind::LShr: {
            if (!inShiftRange(*r, w))
              return std::nullopt;
            if (l->unknown || r->unknown)
              return unknownOf(deps);
            if (l->isConst() && r->isConst())
              return withDeps(constant(signExtend(toUnsigned(l->lo, w) >> r->lo, w)), deps);
            // On a non-negative range the logical shift is the arithmetic one.
            if (l->lo >= 0)
              return withDeps(corners(*l, *r, [](I64 a, I64 b) { return a >> b; }), deps);
            // Across zero it is not monotone at all — `-1 >>> 1` is the largest
            // value there is — so what remains is that a shift of at least one
            // clears the sign bit, which still bounds the result away from the
            // ends of the type. A shift of none is the value itself.
            if (!w || w > 64)
              return unknownOf(deps);
            const std::uint64_t umax = w >= 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << w) - 1;
            if (r->lo >= 1)
              return withDeps(Interval{0, (I64) (umax >> r->lo), false, 0, deps}, deps);
            if (r->hi == 0)
              return withDeps(Interval{l->lo, l->hi, false, 0, deps}, deps);
            return withDeps(
                Interval{
                    std::min<I64>(l->lo, 0), std::max<I64>(l->hi, (I64) (umax >> 1)), false, 0, deps
                },
                deps
            );
          }
          default: {
            // & | ^ : a sound range needs bit-level reasoning, so only known
            // operands fold. Everything else widens, which is where a
            // bit-level domain would pay off first.
            if (!l->isConst() || !r->isConst()) {
              // One exception, because it is how every mask is written: the
              // bits of `x & m` are a subset of m's, so for a non-negative m
              // the result is in [0, m] whatever x holds.
              const Interval *m = nullptr;
              if (!l->unknown && l->lo >= 0)
                m = &*l;
              if (!r->unknown && r->lo >= 0 && (!m || r->hi < m->hi))
                m = &*r;
              if (o.op == AtomOpKind::And && m)
                return withDeps(Interval{0, m->hi, false, 0, deps}, deps);
              return unknownOf(deps);
            }
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

      // The float width an expression computes in. Single precision rounds
      // after every operation, so evaluating an f32 comparison as if it were
      // f64 can decide it the other way.
      std::uint32_t inferFloatWidth(const Expr &e) const {
        auto ofAtom = [&](const Atom &a) -> std::uint32_t {
          return std::visit(
              [&](const auto &x) -> std::uint32_t {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, RValueAtom> || std::is_same_v<T, UnaryAtom>)
                  return floatWidthOf(x.rval);
                else if constexpr (std::is_same_v<T, OpAtom>)
                  return floatWidthOf(x.rval);
                else if constexpr (std::is_same_v<T, CoefAtom>) {
                  if (auto id = std::get_if<LocalOrSymId>(&x.coef))
                    if (auto loc = std::get_if<LocalId>(id)) {
                      LValue lv;
                      lv.base = *loc;
                      return floatWidthOf(lv);
                    }
                  return 0;
                } else if constexpr (std::is_same_v<T, CastAtom>) {
                  auto b = TypeUtils::getFloatBitWidth(x.dstType);
                  return b ? *b : 0;
                } else
                  return 0;
              },
              a.v
          );
        };
        if (std::uint32_t w = ofAtom(e.first))
          return w;
        for (const auto &t: e.rest)
          if (std::uint32_t w = ofAtom(t.atom))
            return w;
        return 64;
      }

      // A comparison of two known floats is decided exactly.
      std::optional<bool> floatHolds(const Cond &c) {
        bool bad = false;
        const std::uint32_t fw = inferFloatWidth(c.lhs);
        auto l = evalFloatExpr(c.lhs, fw, bad);
        if (bad)
          return std::nullopt;
        auto r = evalFloatExpr(c.rhs, fw, bad);
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
                  const std::string before = reason_;
                  auto target = evalPtr(x.rhs);
                  // An unknown target is fine — everything through it widens.
                  // A refusal is not: the arithmetic itself may be undefined.
                  if (!target && reason_ != before)
                    return false;
                  setPtr(x.lhs, target);
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
      bool record_ = false;
      // Float leaf -> its exact value. Absent means unknown; there is no
      // "range of floats" here by design (see the header).
      FloatEnv floats_;
      // Pointer leaf -> the cell it names (an empty optional is the null
      // pointer). A pointer absent from the map points somewhere unknown.
      PtrEnv ptrs_;
      const StructMap &structs_;
      std::vector<IntervalEnv> snaps_;
      std::unordered_map<std::string, std::uint64_t> bitOf_;
      std::vector<std::string> leafOfBit_;
      std::uint64_t blame_ = 0;
      std::unordered_map<std::string, TypePtr> types_;
      std::string reason_;
    };

  } // namespace

  IntervalVerdict checkTrace(
      const FunDecl &fn, const StructMap &structs, const TraceBody &body, const EntryState &entry
  ) {
    return Checker(fn, structs, entry).run(body);
  }

  // --- backward narrowing ---------------------------------------------------

  namespace {

    // A requirement on a value: where it must lie for the trace to run as the
    // profile ran it. Requirements only tighten as they travel backward.
    struct Need {
      I64 lo = kI64Min;
      I64 hi = kI64Max;

      bool none() const { return lo == kI64Min && hi == kI64Max; }
    };

    Need meet(Need a, const Need &b) {
      a.lo = std::max(a.lo, b.lo);
      a.hi = std::min(a.hi, b.hi);
      return a;
    }

    // The local an expression is a bare read of, if it is one. Narrowing only
    // follows single-operand shapes — a bare read, or one shifted by a known
    // amount. Anything else stops the requirement there, which costs a ceiling
    // its tightness and never its soundness.
    const std::string *bareLocal(const Expr &e) {
      if (!e.rest.empty())
        return nullptr;
      if (auto rv = std::get_if<RValueAtom>(&e.first.v))
        return rv->rval.accesses.empty() ? &rv->rval.base.name : nullptr;
      if (auto co = std::get_if<CoefAtom>(&e.first.v))
        if (auto id = std::get_if<LocalOrSymId>(&co->coef))
          if (auto loc = std::get_if<LocalId>(id))
            return &loc->name;
      return nullptr;
    }

    // The declared type of a local, for the width its arithmetic must fit.
    TypePtr localTypeOf(const FunDecl &fn, const std::string &nm) {
      for (const auto &p: fn.params)
        if (p.name.name == nm)
          return p.type;
      for (const auto &l: fn.lets)
        if (l.name.name == nm)
          return l.type;
      return nullptr;
    }

    // The value of a literal or a bare local at one point of the trace.
    Interval valueAt(const Atom &a, const IntervalEnv &env) {
      if (auto co = std::get_if<CoefAtom>(&a.v)) {
        if (auto il = std::get_if<IntLit>(&co->coef))
          return Interval{il->value, il->value, false, 0, 0};
        if (auto id = std::get_if<LocalOrSymId>(&co->coef))
          if (auto loc = std::get_if<LocalId>(id))
            if (auto it = env.find(loc->name); it != env.end())
              return it->second;
      }
      if (auto rv = std::get_if<RValueAtom>(&a.v))
        if (rv->rval.accesses.empty())
          if (auto it = env.find(rv->rval.base.name); it != env.end())
            return it->second;
      return unknownOf();
    }

  } // namespace

  std::unordered_map<std::string, Interval> ceilings(
      const FunDecl &fn, const StructMap &structs, const TraceBody &body, const EntryState &entry
  ) {
    // Forward first: a requirement pushed back through `%d = %x + %k` needs to
    // know what %k held there, which is what the forward pass recorded.
    Checker fwd(fn, structs, entry, /*record=*/true);
    fwd.run(body);
    const std::vector<IntervalEnv> &snaps = fwd.snapshots();

    std::unordered_map<std::string, Need> need;
    auto tighten = [&](const std::string &nm, const Need &n) {
      need[nm] = meet(need.count(nm) ? need[nm] : Need{}, n);
    };

    std::size_t check = body.checks.size();
    for (std::size_t i = body.stmts.size(); i-- > 0;) {
      const IntervalEnv &env = i < snaps.size() ? snaps[i] : entry.ints;
      // Conditions recorded after this statement are met on the way back.
      while (check > 0 && body.checks[check - 1].afterStmt > i) {
        --check;
        const PathCheck &c = body.checks[check];
        const IntervalEnv &cenv =
            c.checkEnvIndex() < snaps.size() ? snaps[c.checkEnvIndex()] : entry.ints;
        // A comparison constrains both of its sides, so each is narrowed
        // against what the other can be. Mirroring the operator is what makes
        // the second one work: `a < b` bounds a from above and b from below.
        const bool flip[2] = {false, true};
        for (bool mirror: flip) {
          const Expr &self = mirror ? c.cond.rhs : c.cond.lhs;
          const Expr &peer = mirror ? c.cond.lhs : c.cond.rhs;
          const std::string *nm = bareLocal(self);
          if (!nm || !peer.rest.empty())
            continue;
          const Interval other = valueAt(peer.first, cenv);
          if (other.unknown)
            continue;
          RelOp op = c.cond.op;
          if (mirror)
            switch (op) {
              case RelOp::LT:
                op = RelOp::GT;
                break;
              case RelOp::LE:
                op = RelOp::GE;
                break;
              case RelOp::GT:
                op = RelOp::LT;
                break;
              case RelOp::GE:
                op = RelOp::LE;
                break;
              default:
                break;
            }
          Need n;
          switch (op) {
            case RelOp::LT:
              if (c.taken)
                n.hi = other.lo - 1;
              else
                n.lo = other.hi;
              break;
            case RelOp::LE:
              if (c.taken)
                n.hi = other.lo;
              else
                n.lo = other.hi + 1;
              break;
            case RelOp::GT:
              if (c.taken)
                n.lo = other.hi + 1;
              else
                n.hi = other.lo;
              break;
            case RelOp::GE:
              if (c.taken)
                n.lo = other.hi;
              else
                n.hi = other.lo - 1;
              break;
            case RelOp::EQ:
              if (c.taken && other.isConst())
                n.lo = n.hi = other.lo;
              break;
            case RelOp::NE:
              if (!c.taken && other.isConst())
                n.lo = n.hi = other.lo;
              break;
          }
          if (!n.none())
            tighten(*nm, n);
        }
      }

      auto *ai = std::get_if<AssignInstr>(&body.stmts[i]);
      if (!ai || !ai->lhs.accesses.empty())
        continue;
      const std::string &dst = ai->lhs.base.name;
      Need have = need.count(dst) ? need[dst] : Need{};
      // Not every requirement comes from downstream: an assignment that can
      // overflow requires its own result to fit, which bounds its operands
      // just as a branch does.
      if (ai->rhs.rest.size() == 1) {
        auto bits = TypeUtils::getIntBitWidth(localTypeOf(fn, dst));
        if (bits && *bits < 64) {
          Need w;
          w.lo = -(I64(1) << (*bits - 1));
          w.hi = (I64(1) << (*bits - 1)) - 1;
          have = meet(have, w);
        }
      }
      // The destination is rewritten here, so a requirement on it reaches the
      // operands and stops.
      need.erase(dst);
      if (have.none())
        continue;
      if (const std::string *src = bareLocal(ai->rhs)) {
        tighten(*src, have);
        continue;
      }
      if (ai->rhs.rest.size() != 1)
        continue;
      const Interval k = valueAt(ai->rhs.rest[0].atom, env);
      Expr head{cloneAtom(ai->rhs.first), {}, {}};
      const std::string *src = bareLocal(head);
      if (!src || !k.isConst())
        continue;
      // `%d = %x + k` requires of %x exactly what it required of %d, shifted.
      const I64 d = ai->rhs.rest[0].op == AddOp::Plus ? k.lo : -k.lo;
      Need n;
      if (have.lo == kI64Min || __builtin_sub_overflow(have.lo, d, &n.lo))
        n.lo = kI64Min;
      if (have.hi == kI64Max || __builtin_sub_overflow(have.hi, d, &n.hi))
        n.hi = kI64Max;
      tighten(*src, n);
    }

    std::unordered_map<std::string, Interval> out;
    for (const auto &[key, iv]: entry.ints) {
      I64 lo = kI64Min, hi = kI64Max;
      if (iv.bits && iv.bits < 64) {
        lo = -(I64(1) << (iv.bits - 1));
        hi = (I64(1) << (iv.bits - 1)) - 1;
      }
      if (auto it = need.find(key); it != need.end()) {
        lo = std::max(lo, it->second.lo);
        hi = std::min(hi, it->second.hi);
      }
      Interval c = iv;
      c.lo = lo;
      c.hi = std::max(lo, hi);
      c.unknown = false;
      out[key] = c;
    }
    return out;
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

    // Step 2 — ceilings. Pushing every check's requirement backward bounds each
    // leaf before a single trial is run: one that cannot move at all is pinned
    // for free, and the rest have a bound to search under rather than a
    // doubling sequence that has to discover where to stop.
    const auto ceil = ceilings(fn, structs, body, entry);
    for (auto &leaf: box.leaves) {
      auto it = ceil.find(leaf.key);
      if (it != ceil.end() && it->second.lo == it->second.hi)
        leaf.cls = LeafClass::Pinned; // nothing else could have been proven
    }

    // Free leaves. A leaf is free exactly when the trace is provable without
    // knowing it, which is one pass with the leaf left out. Nothing is wider
    // than "any value", so a leaf that clears this never enters the search.
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
      std::int64_t good = 0;      // widest radius proven
      std::int64_t next = 1;      // radius to try
      bool frozen = false;        //
      std::int64_t cap = kI64Max; // widest the ceiling allows
    };

    std::vector<Open> open;
    for (auto &leaf: box.leaves) {
      if (leaf.cls == LeafClass::Free)
        continue;
      // How far the ceiling allows this leaf to move at all. Saturating,
      // because an unconstrained ceiling is the full width of the type and
      // subtracting its ends would overflow — which silently pinned every
      // leaf the narrowing had nothing to say about. The wider side decides:
      // a radius is clipped to the type on the way in, so a leaf sitting at
      // its type's edge can still open in the one direction it has.
      auto span = [](std::int64_t from, std::int64_t to) -> std::int64_t {
        std::int64_t d = 0;
        if (__builtin_sub_overflow(to, from, &d))
          return kI64Max;
        return d < 0 ? 0 : d;
      };
      std::int64_t cap = kI64Max;
      if (auto it = ceil.find(leaf.key); it != ceil.end()) {
        const std::int64_t centre = leaf.range.lo;
        cap = std::max(span(it->second.lo, centre), span(centre, it->second.hi));
        if (cap <= 0)
          continue; // the ceiling holds it still; no pass spent on it
      }
      open.push_back(Open{&leaf, leaf.range.lo, 0, 1, false, cap});
    }
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
            // Doubling stops at the ceiling: past it the trial would only be
            // refused, and by a check that was known in advance.
            if (o.good >= o.cap)
              o.frozen = true;
            else
              o.next = std::min(o.next * 2, o.cap);
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
