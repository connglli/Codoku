#include <optional>
#include <random>
#include <string>
#include <variant>

#include "analysis/type_utils.hpp"
#include "ast/clone.hpp"
#include "ast/match.hpp"
#include "internal.hpp"

namespace refractir::reify::antiopt {

  namespace {

    using namespace refractir::pat;
    using I64 = std::int64_t;
    using Val = std::optional<I64>;

    // Family B — arithmetic <-> bitwise crossings.
    //
    // These are the ones worth having. An optimizer simplifies within the
    // arithmetic domain or within the bitwise domain; it rarely translates
    // between them, so `x ^ y` written as `(x | y) - (x & y)` has to be
    // *reasoned* back rather than pattern-matched.
    //
    // Every crossing here is one entry in the table at the bottom of the file:
    // the shape it starts from, how to say the same thing differently, and the
    // arithmetic of both sides so the claim is checked at i8 rather than
    // believed. Adding one is adding a row.
    //
    // Most are Tier1 — they introduce `+ - *`, and whether the intermediates
    // stay inside the type is for the acceptance check to decide, not the
    // rule. In this domain that is a real filter rather than a formality: the
    // interval domain widens `& | ^` of two ranges to unknown, so a crossing
    // that leaves the bitwise domain survives only where its operands are
    // known. The two that never leave it (`mba-xor-nand`, `mba-demorgan-*`)
    // fire anywhere.
    //
    // One trap to note: the textbook form of the carry identity is
    // `(x ^ y) + ((x & y) << 1)`. That one is wrong here — RefractIR's `<<` is
    // signed arithmetic and traps on a negative left operand (spec §7.1), so
    // it is UB for any two negative operands. `2 * (x & y)` says the same
    // thing and survives.

    // The statement shapes a crossing can start from. `+` and `-` are
    // expression-level in RefractIR rather than atom operators, and `Value` is
    // for a rule that re-expresses a whole value instead of an operation.
    enum class Shape { Xor, Or, And, Add, Sub, Value };

    // What an emitter is handed: the statement it replaces, the local it
    // assigns, and the two operands it read (none, for a `Value` rule).
    struct Operands {
      const Instr *src = nullptr;
      std::string d, x, y;
    };

    using Emit = std::vector<Instr> (*)(const Operands &, AntiOptContext &);

    struct Crossing {
      const char *name;
      Shape from;
      TrapTier tier;
      Emit emit;
      Val (*lhs)(I64, I64); // what the statement computed
      Val (*rhs)(I64, I64); // what the rewrite computes, undefined where it traps
    };

    // --- emitter helpers ------------------------------------------------------

    template<typename... T>
    std::vector<Instr> list(T &&...xs) {
      std::vector<Instr> v;
      (v.push_back(std::move(xs)), ...);
      return v;
    }

    TypePtr typeOf(const AntiOptContext &ctx, const std::string &n) {
      return localType(ctx.fn, ctx.lets, n);
    }

    // `%t = ~%x` into a fresh local, which is how a complement reaches the
    // right of a binary operator at all (spec §5.3 wants an lvalue there).
    std::optional<std::string>
    complementOf(const std::string &x, AntiOptContext &ctx, std::vector<Instr> &out) {
      TypePtr ty = typeOf(ctx, x);
      if (!ty || !intRange(ty))
        return std::nullopt;
      const std::string t = ctx.names.fresh(ty, ctx.lets);
      out.push_back(assignInstr(localLV(t), simpleExpr(notAtom(x))));
      return t;
    }

    // --- the catalog ----------------------------------------------------------

    // B1  x ^ y  ==  (x | y) - (x & y)
    std::vector<Instr> emitXorAsOrSubAnd(const Operands &o, AntiOptContext &) {
      Expr e = simpleExpr(binAtom(o.x, AtomOpKind::Or, o.y));
      addTail(e, AddOp::Minus, binAtom(o.x, AtomOpKind::And, o.y));
      return list(assignInstr(localLV(o.d), std::move(e)));
    }

    // B2  x | y  ==  (x ^ y) + (x & y)
    std::vector<Instr> emitOrAsXorAddAnd(const Operands &o, AntiOptContext &) {
      Expr e = simpleExpr(binAtom(o.x, AtomOpKind::Xor, o.y));
      addTail(e, AddOp::Plus, binAtom(o.x, AtomOpKind::And, o.y));
      return list(assignInstr(localLV(o.d), std::move(e)));
    }

    // B4  x + y  ==  (x ^ y) + 2 * (x & y)
    std::vector<Instr> emitAddAsXorCarry(const Operands &o, AntiOptContext &ctx) {
      TypePtr ty = typeOf(ctx, o.y);
      if (!ty || !intRange(ty))
        return {};
      const std::string t = ctx.names.fresh(ty, ctx.lets);
      Expr e = simpleExpr(binAtom(o.x, AtomOpKind::Xor, o.y));
      addTail(e, AddOp::Plus, opAtom(Coef{IntLit{2, {}}}, AtomOpKind::Mul, t));
      return list(
          assignInstr(localLV(t), opExpr(o.x, AtomOpKind::And, o.y)),
          assignInstr(localLV(o.d), std::move(e))
      );
    }

    // B3  x & y  ==  x - (x & ~y)
    std::vector<Instr> emitAndAsSub(const Operands &o, AntiOptContext &ctx) {
      std::vector<Instr> out;
      auto ny = complementOf(o.y, ctx, out);
      if (!ny)
        return {};
      Expr e = simpleExpr(localAtom(o.x));
      addTail(e, AddOp::Minus, binAtom(o.x, AtomOpKind::And, *ny));
      out.push_back(assignInstr(localLV(o.d), std::move(e)));
      return out;
    }

    // B7  x & y  ==  (x | y) - (x ^ y)
    std::vector<Instr> emitAndAsOrSubXor(const Operands &o, AntiOptContext &) {
      Expr e = simpleExpr(binAtom(o.x, AtomOpKind::Or, o.y));
      addTail(e, AddOp::Minus, binAtom(o.x, AtomOpKind::Xor, o.y));
      return list(assignInstr(localLV(o.d), std::move(e)));
    }

    // B6  x | y  ==  x + y - (x & y)
    std::vector<Instr> emitOrAsAddSubAnd(const Operands &o, AntiOptContext &ctx) {
      TypePtr ty = typeOf(ctx, o.y);
      if (!ty || !intRange(ty))
        return {};
      const std::string t = ctx.names.fresh(ty, ctx.lets);
      Expr e = simpleExpr(localAtom(o.x));
      addTail(e, AddOp::Plus, localAtom(o.y));
      addTail(e, AddOp::Minus, localAtom(t));
      return list(
          assignInstr(localLV(t), opExpr(o.x, AtomOpKind::And, o.y)),
          assignInstr(localLV(o.d), std::move(e))
      );
    }

    // B5  x - y  ==  (x ^ y) - 2 * (~x & y)
    std::vector<Instr> emitSubAsXorBorrow(const Operands &o, AntiOptContext &ctx) {
      std::vector<Instr> out;
      auto nx = complementOf(o.x, ctx, out);
      if (!nx)
        return {};
      TypePtr ty = typeOf(ctx, o.y);
      const std::string borrow = ctx.names.fresh(ty, ctx.lets);
      out.push_back(assignInstr(localLV(borrow), opExpr(*nx, AtomOpKind::And, o.y)));
      Expr e = simpleExpr(binAtom(o.x, AtomOpKind::Xor, o.y));
      addTail(e, AddOp::Minus, opAtom(Coef{IntLit{2, {}}}, AtomOpKind::Mul, borrow));
      out.push_back(assignInstr(localLV(o.d), std::move(e)));
      return out;
    }

    // B8  x ^ y  ==  (x | y) & ~(x & y)
    std::vector<Instr> emitXorAsNand(const Operands &o, AntiOptContext &ctx) {
      TypePtr ty = typeOf(ctx, o.x);
      if (!ty || !intRange(ty))
        return {};
      const std::string both = ctx.names.fresh(ty, ctx.lets);
      std::vector<Instr> out = list(assignInstr(localLV(both), opExpr(o.x, AtomOpKind::And, o.y)));
      auto nboth = complementOf(both, ctx, out);
      if (!nboth)
        return {};
      const std::string either = ctx.names.fresh(ty, ctx.lets);
      out.push_back(assignInstr(localLV(either), opExpr(o.x, AtomOpKind::Or, o.y)));
      out.push_back(assignInstr(localLV(o.d), opExpr(either, AtomOpKind::And, *nboth)));
      return out;
    }

    // B10  x & y  ==  ~(~x | ~y)   and   x | y  ==  ~(~x & ~y)
    template<AtomOpKind Dual>
    std::vector<Instr> emitDeMorgan(const Operands &o, AntiOptContext &ctx) {
      std::vector<Instr> out;
      auto nx = complementOf(o.x, ctx, out);
      auto ny = complementOf(o.y, ctx, out);
      if (!nx || !ny)
        return {};
      TypePtr ty = typeOf(ctx, o.x);
      const std::string joined = ctx.names.fresh(ty, ctx.lets);
      out.push_back(assignInstr(localLV(joined), opExpr(*nx, Dual, *ny)));
      out.push_back(assignInstr(localLV(o.d), simpleExpr(notAtom(joined))));
      return out;
    }

    // B9  x  ==  (x >>> k << k) + (x & (2^k - 1))     for x >= 0
    //
    // The one crossing that re-expresses a value rather than an operation, so
    // it applies to any assignment: the statement stays and its result is
    // taken apart into a high half and a low half. Licensed by the sign, since
    // `<<` traps on a negative operand — which is exactly what stops it from
    // firing on a value the box cannot hold above zero.
    std::vector<Instr> emitBitSplit(const Operands &o, AntiOptContext &ctx) {
      TypePtr ty = typeOf(ctx, o.d);
      auto bits = TypeUtils::getIntBitWidth(ty);
      if (!bits || *bits < 3)
        return {};
      std::uniform_int_distribution<int> pick(1, (int) *bits - 2);
      const int k = pick(ctx.rng);
      const std::string amount = ctx.names.literal(k, ty, ctx.lets);
      const std::string mask = ctx.names.literal((I64(1) << k) - 1, ty, ctx.lets);
      const std::string high = ctx.names.fresh(ty, ctx.lets);
      const std::string low = ctx.names.fresh(ty, ctx.lets);

      Expr sum = simpleExpr(localAtom(high));
      addTail(sum, AddOp::Plus, localAtom(low));
      return list(
          cloneInstr(*o.src), assignInstr(localLV(high), opExpr(o.d, AtomOpKind::LShr, amount)),
          assignInstr(localLV(high), opExpr(high, AtomOpKind::Shl, amount)),
          assignInstr(localLV(low), opExpr(o.d, AtomOpKind::And, mask)),
          assignInstr(localLV(o.d), std::move(sum))
      );
    }

    // --- the identities, as arithmetic ---------------------------------------

    // Every `rhs` below evaluates the rewrite the way the interpreter would,
    // returning nullopt where an intermediate leaves the type — that is the
    // rewrite trapping, not the identity failing, and the self-test skips it.

    const Crossing kCrossings[] = {
        {"mba-xor", Shape::Xor, TrapTier::Tier1, emitXorAsOrSubAnd,
         [](I64 a, I64 b) { return Val(a ^ b); },
         [](I64 a, I64 b) { return fitsI8((a | b) - (a & b)) ? Val((a | b) - (a & b)) : Val{}; }},

        {"mba-or", Shape::Or, TrapTier::Tier1, emitOrAsXorAddAnd,
         [](I64 a, I64 b) { return Val(a | b); },
         [](I64 a, I64 b) { return fitsI8((a ^ b) + (a & b)) ? Val((a ^ b) + (a & b)) : Val{}; }},

        {"mba-add", Shape::Add, TrapTier::Tier1, emitAddAsXorCarry,
         [](I64 a, I64 b) { return fitsI8(a + b) ? Val(a + b) : Val{}; },
         [](I64 a, I64 b) -> Val {
           const I64 carry = 2 * (a & b);
           if (!fitsI8(carry) || !fitsI8((a ^ b) + carry))
             return {};
           return (a ^ b) + carry;
         }},

        {"mba-and-sub", Shape::And, TrapTier::Tier1, emitAndAsSub,
         [](I64 a, I64 b) { return Val(a & b); },
         [](I64 a, I64 b) { return fitsI8(a - (a & ~b)) ? Val(a - (a & ~b)) : Val{}; }},

        {"mba-and-diff", Shape::And, TrapTier::Tier1, emitAndAsOrSubXor,
         [](I64 a, I64 b) { return Val(a & b); },
         [](I64 a, I64 b) { return fitsI8((a | b) - (a ^ b)) ? Val((a | b) - (a ^ b)) : Val{}; }},

        {"mba-or-add", Shape::Or, TrapTier::Tier1, emitOrAsAddSubAnd,
         [](I64 a, I64 b) { return Val(a | b); },
         [](I64 a, I64 b) -> Val {
           // The chain runs left to right, so the sum has to fit on its own.
           if (!fitsI8(a + b) || !fitsI8(a + b - (a & b)))
             return {};
           return a + b - (a & b);
         }},

        {"mba-sub", Shape::Sub, TrapTier::Tier1, emitSubAsXorBorrow,
         [](I64 a, I64 b) { return fitsI8(a - b) ? Val(a - b) : Val{}; },
         [](I64 a, I64 b) -> Val {
           const I64 borrow = 2 * (~a & b);
           if (!fitsI8(borrow) || !fitsI8((a ^ b) - borrow))
             return {};
           return (a ^ b) - borrow;
         }},

        {"mba-xor-nand", Shape::Xor, TrapTier::Tier0, emitXorAsNand,
         [](I64 a, I64 b) { return Val(a ^ b); },
         [](I64 a, I64 b) { return Val((a | b) & ~(a & b)); }},

        {"mba-demorgan-and", Shape::And, TrapTier::Tier0, emitDeMorgan<AtomOpKind::Or>,
         [](I64 a, I64 b) { return Val(a & b); }, [](I64 a, I64 b) { return Val(~(~a | ~b)); }},

        {"mba-demorgan-or", Shape::Or, TrapTier::Tier0, emitDeMorgan<AtomOpKind::And>,
         [](I64 a, I64 b) { return Val(a | b); }, [](I64 a, I64 b) { return Val(~(~a & ~b)); }},

        {"mba-bit-split", Shape::Value, TrapTier::Tier1, emitBitSplit,
         [](I64 v, I64 k) { return k >= 1 && k <= 6 ? Val(v) : Val{}; },
         [](I64 v, I64 k) -> Val {
           if (k < 1 || k > 6 || v < 0) // the license: `<<` traps on a negative
             return {};
           const I64 high = (v >> k) << k;
           if (!fitsI8(high) || !fitsI8(high + (v & ((I64(1) << k) - 1))))
             return {};
           return high + (v & ((I64(1) << k) - 1));
         }},
    };

    class CrossingRule : public AntiOptRule {
    public:
      explicit CrossingRule(const Crossing &c) : c_(c) {}

      const char *name() const override { return c_.name; }

      RuleFamily family() const override { return RuleFamily::Mba; }

      TrapTier tier() const override { return c_.tier; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        Operands o;
        return read(stmts[pos.stmt], o, ctx);
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        Operands o;
        if (!read(stmts[pos.stmt], o, ctx))
          return {};
        return c_.emit(o, ctx);
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        t.original = c_.lhs;
        t.rewritten = c_.rhs;
        return t;
      }

    private:
      // The statement, read as this crossing's shape. Operand types have to be
      // scalar integers: every emitter needs one for its temporaries.
      bool read(const Instr &ins, Operands &o, const AntiOptContext &ctx) const {
        o.src = &ins;
        if (c_.from == Shape::Value)
          return match(ins, m_Assign(m_Local(o.d), m_Any())) &&
                 intRange(localType(ctx.fn, ctx.lets, o.d)).has_value();
        const bool ok =
            c_.from == Shape::Add || c_.from == Shape::Sub
                ? match(
                      ins, m_Assign(
                               m_Local(o.d), m_Pair(
                                                 c_.from == Shape::Add ? AddOp::Plus : AddOp::Minus,
                                                 m_Var(m_Local(o.x)), m_Var(m_Local(o.y))
                                             )
                           )
                  )
                : match(
                      ins, m_Assign(m_Local(o.d), m_One(m_Op(atomOp(), m_Local(o.x), m_Local(o.y))))
                  );
        return ok && intRange(localType(ctx.fn, ctx.lets, o.x)).has_value() &&
               intRange(localType(ctx.fn, ctx.lets, o.y)).has_value();
      }

      AtomOpKind atomOp() const {
        switch (c_.from) {
          case Shape::Xor:
            return AtomOpKind::Xor;
          case Shape::Or:
            return AtomOpKind::Or;
          default:
            return AtomOpKind::And;
        }
      }

      const Crossing &c_;
    };

  } // namespace

  void registerMbaRules(RuleList &out) {
    for (const auto &c: kCrossings)
      out.push_back(std::make_unique<CrossingRule>(c));
  }

} // namespace refractir::reify::antiopt
