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

    // Rule:  %d = e
    //   ->   %d = e;  %d = %d ^ %k;  %d = %d ^ %k
    //
    // Family A — identity insertion. The same value by x ^ k ^ k == x, for any
    // k. Trap-free: the only operator introduced is `^`, and the mask is drawn
    // fresh each time so the pair is not a recognizable constant.
    class XorTwiceRule : public AntiOptRule {
    public:
      const char *name() const override { return "xor-twice"; }

      RuleFamily family() const override { return RuleFamily::Peephole; }

      TrapTier tier() const override { return TrapTier::Tier0; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        std::string d;
        return match(stmts[pos.stmt], m_Assign(m_Local(d), m_Any())) &&
               intRange(localType(ctx.fn, ctx.lets, d)).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        std::string d;
        if (!match(stmts[pos.stmt], m_Assign(m_Local(d), m_Any())))
          return {};
        TypePtr ty = localType(ctx.fn, ctx.lets, d);
        auto range = intRange(ty);
        if (!range)
          return {};
        // The mask is a literal of the statement's own type, and a narrow type
        // holds very little: i1 is {-1, 0}, so drawing from a positive range
        // wrote `let %k: i1 = 1;` — a program the checker rejects, which took
        // down every region holding an i1 local. Drawing over the whole range
        // keeps it representable; a zero mask hides nothing, so it becomes the
        // low end instead, which is non-zero at every width.
        std::uniform_int_distribution<std::int64_t> pick(range->first, range->second);
        const std::int64_t drawn = pick(ctx.rng);
        const std::string k = ctx.names.literal(drawn ? drawn : range->first, ty, ctx.lets);

        std::vector<Instr> out;
        out.push_back(cloneInstr(stmts[pos.stmt]));
        out.push_back(assignInstr(localLV(d), opExpr(d, AtomOpKind::Xor, k)));
        out.push_back(assignInstr(localLV(d), opExpr(d, AtomOpKind::Xor, k)));
        return out;
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        t.original = [](std::int64_t v, std::int64_t) { return std::optional<std::int64_t>(v); };
        t.rewritten = [](std::int64_t v, std::int64_t k) {
          return std::optional<std::int64_t>((v ^ k) ^ k);
        };
        return t;
      }
    };

    // Rule:  %d = ... + K ...
    //   ->   %d = ... + K1 + K2 ...      (K1 + K2 == K)
    //
    // Family A — constant splitting, the reverse of the folding every compiler
    // does on the way in. Tier1: a flat chain is evaluated left to right, so
    // splitting moves the prefix sums and one of them could leave the type.
    class SplitConstantRule : public AntiOptRule {
    public:
      const char *name() const override { return "split-constant"; }

      RuleFamily family() const override { return RuleFamily::Peephole; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      // The index of a tail atom holding a literal, if the statement has one.
      static std::optional<std::size_t> literalTail(const Instr &ins) {
        auto *ai = std::get_if<AssignInstr>(&ins);
        if (!ai)
          return std::nullopt;
        for (std::size_t i = 0; i < ai->rhs.rest.size(); ++i)
          if (match(ai->rhs.rest[i].atom, m_Coef(m_AnyInt())))
            return i;
        return std::nullopt;
      }

      bool
      matches(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &) const override {
        return literalTail(stmts[pos.stmt]).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        auto idx = literalTail(stmts[pos.stmt]);
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        AssignInstr out{ai.lhs, cloneExpr(ai.rhs), ai.span};
        std::int64_t k = 0;
        if (!match(out.rhs.rest[*idx].atom, m_Coef(m_Int(k))))
          return {};

        // Both halves are literals of the statement's own type, so both have to
        // be representable in it — splitting 125 into -32 and 157 is not a
        // rewrite of an i8 statement, it is a program the checker rejects.
        auto range = intRange(localType(ctx.fn, ctx.lets, ai.lhs.base.name));
        if (!range)
          return {};
        const std::int64_t alo = std::max<std::int64_t>(range->first, -32);
        const std::int64_t ahi = std::min<std::int64_t>(range->second, 32);
        if (alo > ahi)
          return {};
        std::uniform_int_distribution<std::int64_t> pick(alo, ahi);
        const std::int64_t a = pick(ctx.rng);
        std::int64_t b = 0;
        if (__builtin_sub_overflow(k, a, &b) || b < range->first || b > range->second)
          return {};

        const AddOp op = out.rhs.rest[*idx].op;
        std::get<CoefAtom>(out.rhs.rest[*idx].atom.v).coef = Coef{IntLit{a, {}}};
        out.rhs.rest.insert(
            out.rhs.rest.begin() + (long) (*idx + 1),
            Expr::Tail{op, Atom{CoefAtom{Coef{IntLit{b, {}}}, {}}, {}}, {}}
        );
        std::vector<Instr> res;
        res.push_back(Instr{std::move(out)});
        return res;
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        // The claim is k == a + (k - a), which holds whenever every part is
        // representable — the split must not put a half outside the type.
        t.original = [](std::int64_t k, std::int64_t) { return std::optional<std::int64_t>(k); };
        t.rewritten = [](std::int64_t k, std::int64_t a) -> std::optional<std::int64_t> {
          std::int64_t b = 0;
          if (__builtin_sub_overflow(k, a, &b) || b < -128 || b > 127)
            return std::nullopt;
          return a + b;
        };
        return t;
      }
    };

    // Rule:  %d = ... - K ...   ->   %d = ... + -K ...      (K > 0)
    //
    // Family A — un-fold negation. A compiler spells an added negative literal
    // as a subtraction, so this spells it the other way. Only that direction:
    // the rewritten literal is negative and no longer matches, which is what
    // stops two rounds from swapping it back and forth forever.
    class FoldNegationRule : public AntiOptRule {
    public:
      const char *name() const override { return "fold-negation"; }

      RuleFamily family() const override { return RuleFamily::Peephole; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      // The index of a tail atom holding a positive literal, if there is one.
      static std::optional<std::size_t> positiveTail(const Instr &ins) {
        auto *ai = std::get_if<AssignInstr>(&ins);
        if (!ai)
          return std::nullopt;
        for (std::size_t i = 0; i < ai->rhs.rest.size(); ++i) {
          std::int64_t k = 0;
          if (match(ai->rhs.rest[i].atom, m_Coef(m_Int(k))) && k > 0)
            return i;
        }
        return std::nullopt;
      }

      bool
      matches(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &) const override {
        return positiveTail(stmts[pos.stmt]).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &) const override {
        auto idx = positiveTail(stmts[pos.stmt]);
        if (!idx)
          return {};
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        AssignInstr out{ai.lhs, cloneExpr(ai.rhs), ai.span};
        std::int64_t k = 0;
        if (!match(out.rhs.rest[*idx].atom, m_Coef(m_Int(k))))
          return {};
        // K was positive, so -K is representable wherever K was.
        std::get<CoefAtom>(out.rhs.rest[*idx].atom.v).coef = Coef{IntLit{-k, {}}};
        auto &op = out.rhs.rest[*idx].op;
        op = op == AddOp::Plus ? AddOp::Minus : AddOp::Plus;
        std::vector<Instr> res;
        res.push_back(Instr{std::move(out)});
        return res;
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        t.original = [](std::int64_t a, std::int64_t k) -> std::optional<std::int64_t> {
          if (k <= 0 || !fitsI8(a - k))
            return std::nullopt;
          return a - k;
        };
        t.rewritten = [](std::int64_t a, std::int64_t k) -> std::optional<std::int64_t> {
          if (k <= 0 || !fitsI8(a + -k))
            return std::nullopt;
          return a + -k;
        };
        return t;
      }
    };

    // Rule:  %d = 8 * %x   ->   %d = %x << %k3
    //        %d = 9 * %x   ->   %t = %x << %k3;  %d = %t + %x
    //
    // Family A — un-strength-reduction. Multiplying by a power of two is what
    // an optimizer turns into a shift, so the catalog goes the other way and
    // hands it back the work.
    //
    // Licensed, not merely tier-1: RefractIR's `<<` is signed arithmetic and
    // traps on a negative left operand (spec §7.1), so this is right only
    // where the guard's box proves %x >= 0. The rule offers the rewrite and
    // the interval re-check keeps it exactly there.
    class MulShiftRule : public AntiOptRule {
    public:
      MulShiftRule(bool plusOne, const char *nm) : plusOne_(plusOne), name_(nm) {}

      const char *name() const override { return name_; }

      RuleFamily family() const override { return RuleFamily::Peephole; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        std::string d, x;
        std::int64_t k = 0;
        return read(stmts[pos.stmt], d, x, k) && exponent(k) &&
               intRange(localType(ctx.fn, ctx.lets, x)).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        std::string d, x;
        std::int64_t k = 0;
        if (!read(stmts[pos.stmt], d, x, k))
          return {};
        auto e = exponent(k);
        TypePtr ty = localType(ctx.fn, ctx.lets, x);
        if (!e || !ty)
          return {};
        // The shift amount is an operand like any other, so it needs the same
        // width as what it shifts (spec §6.5) and a name of its own (§5.3).
        const std::string amount = ctx.names.literal(*e, ty, ctx.lets);

        std::vector<Instr> out;
        if (!plusOne_) {
          out.push_back(assignInstr(localLV(d), opExpr(x, AtomOpKind::Shl, amount)));
          return out;
        }
        const std::string t = ctx.names.fresh(ty, ctx.lets);
        out.push_back(assignInstr(localLV(t), opExpr(x, AtomOpKind::Shl, amount)));
        Expr sum = simpleExpr(localAtom(t));
        addTail(sum, AddOp::Plus, localAtom(x));
        out.push_back(assignInstr(localLV(d), std::move(sum)));
        return out;
      }

      std::optional<SelfTest> selfTest() const override {
        const bool plusOne = plusOne_;
        // The operands are the multiplicand and the exponent, since that is
        // what the two sides disagree about; anything that is not a shift this
        // rule would emit is skipped.
        auto multiplier = [plusOne](std::int64_t e) -> std::optional<std::int64_t> {
          if (e < 1 || e > 6)
            return std::nullopt;
          return (std::int64_t{1} << e) + (plusOne ? 1 : 0);
        };
        SelfTest t;
        t.original = [multiplier](std::int64_t x, std::int64_t e) -> std::optional<std::int64_t> {
          auto m = multiplier(e);
          if (!m || !fitsI8(*m * x))
            return std::nullopt;
          return *m * x;
        };
        t.rewritten = [multiplier,
                       plusOne](std::int64_t x, std::int64_t e) -> std::optional<std::int64_t> {
          if (!multiplier(e) || x < 0) // the license: `<<` traps on a negative
            return std::nullopt;
          const std::int64_t shifted = x << e;
          if (!fitsI8(shifted) || (plusOne && !fitsI8(shifted + x)))
            return std::nullopt;
          return plusOne ? shifted + x : shifted;
        };
        return t;
      }

    private:
      // `%d = K * %x`, the only spelling RefractIR has for a literal
      // multiplier — the right operand of `*` must be an lvalue (§5.3).
      static bool read(const Instr &ins, std::string &d, std::string &x, std::int64_t &k) {
        return match(ins, m_Assign(m_Local(d), m_One(m_Op(AtomOpKind::Mul, m_Int(k), m_Local(x)))));
      }

      // The exponent this multiplier is a shift by, if it is one at all.
      std::optional<int> exponent(std::int64_t k) const {
        const std::int64_t base = plusOne_ ? k - 1 : k;
        if (base < 2 || (base & (base - 1)) != 0)
          return std::nullopt; // 0, 1, or not a power of two
        int e = 0;
        while ((std::int64_t{1} << e) < base)
          ++e;
        return e;
      }

      bool plusOne_;
      const char *name_;
    };

    // Rule:  %d = ~%x   ->   %d = 0 - %x - 1
    //
    // Family A — complement as arithmetic. It leaves the bitwise domain, which
    // is the crossing family B is built on; here it costs one statement and no
    // temporaries. Tier1: negating INT_MIN overflows, so the box has to
    // exclude it.
    class NotComplementRule : public AntiOptRule {
    public:
      const char *name() const override { return "not-complement"; }

      RuleFamily family() const override { return RuleFamily::Peephole; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool
      matches(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &) const override {
        std::string d, x;
        return read(stmts[pos.stmt], d, x);
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &) const override {
        std::string d, x;
        if (!read(stmts[pos.stmt], d, x))
          return {};
        Expr e = simpleExpr(intAtom(0));
        addTail(e, AddOp::Minus, localAtom(x));
        addTail(e, AddOp::Minus, intAtom(1));
        std::vector<Instr> out;
        out.push_back(assignInstr(localLV(d), std::move(e)));
        return out;
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        t.original = [](std::int64_t x, std::int64_t) { return std::optional<std::int64_t>(~x); };
        t.rewritten = [](std::int64_t x, std::int64_t) -> std::optional<std::int64_t> {
          if (!fitsI8(0 - x) || !fitsI8(0 - x - 1))
            return std::nullopt;
          return 0 - x - 1;
        };
        return t;
      }

    private:
      static bool read(const Instr &ins, std::string &d, std::string &x) {
        return match(ins, m_Assign(m_Local(d), m_One(m_Not(m_Local(x)))));
      }
    };

    // Rule:  %d = %x - %y   ->   %t = ~%y;  %d = %x + %t + 1
    //
    // Family A — subtraction as addition, which is the two's-complement
    // definition of subtraction written out. Tier1 on both additions.
    class SubAsAddRule : public AntiOptRule {
    public:
      const char *name() const override { return "sub-as-add"; }

      RuleFamily family() const override { return RuleFamily::Peephole; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        std::string d, x, y;
        return read(stmts[pos.stmt], d, x, y) &&
               intRange(localType(ctx.fn, ctx.lets, y)).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        std::string d, x, y;
        if (!read(stmts[pos.stmt], d, x, y))
          return {};
        TypePtr ty = localType(ctx.fn, ctx.lets, y);
        if (!ty)
          return {};
        const std::string t = ctx.names.fresh(ty, ctx.lets);

        std::vector<Instr> out;
        out.push_back(assignInstr(
            localLV(t), simpleExpr(Atom{UnaryAtom{UnaryOpKind::Not, localLV(y), {}}, {}})
        ));
        Expr sum = simpleExpr(localAtom(x));
        addTail(sum, AddOp::Plus, localAtom(t));
        addTail(sum, AddOp::Plus, intAtom(1));
        out.push_back(assignInstr(localLV(d), std::move(sum)));
        return out;
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        t.original = [](std::int64_t x, std::int64_t y) -> std::optional<std::int64_t> {
          return fitsI8(x - y) ? std::optional<std::int64_t>(x - y) : std::nullopt;
        };
        t.rewritten = [](std::int64_t x, std::int64_t y) -> std::optional<std::int64_t> {
          const std::int64_t c = ~y;
          if (!fitsI8(x + c) || !fitsI8(x + c + 1))
            return std::nullopt;
          return x + c + 1;
        };
        return t;
      }

    private:
      static bool read(const Instr &ins, std::string &d, std::string &x, std::string &y) {
        return match(
            ins, m_Assign(m_Local(d), m_Pair(AddOp::Minus, m_Var(m_Local(x)), m_Var(m_Local(y))))
        );
      }
    };

    // Rule:  %d = %x + %x   ->   %d = 2 * %x
    //
    // Family A — un-strength-reduction again, in the shape that needs no
    // license: doubling is a multiplication whichever way it is spelled, and
    // both spellings overflow on exactly the same values.
    class DoubleAsMulRule : public AntiOptRule {
    public:
      const char *name() const override { return "double-as-mul"; }

      RuleFamily family() const override { return RuleFamily::Peephole; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool
      matches(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &) const override {
        std::string d, x;
        return read(stmts[pos.stmt], d, x);
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &) const override {
        std::string d, x;
        if (!read(stmts[pos.stmt], d, x))
          return {};
        std::vector<Instr> out;
        out.push_back(
            assignInstr(localLV(d), simpleExpr(opAtom(Coef{IntLit{2, {}}}, AtomOpKind::Mul, x)))
        );
        return out;
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        t.original = [](std::int64_t x, std::int64_t) -> std::optional<std::int64_t> {
          return fitsI8(x + x) ? std::optional<std::int64_t>(x + x) : std::nullopt;
        };
        t.rewritten = [](std::int64_t x, std::int64_t) -> std::optional<std::int64_t> {
          return fitsI8(2 * x) ? std::optional<std::int64_t>(2 * x) : std::nullopt;
        };
        return t;
      }

    private:
      static bool read(const Instr &ins, std::string &d, std::string &x) {
        std::string y;
        return match(
                   ins,
                   m_Assign(m_Local(d), m_Pair(AddOp::Plus, m_Var(m_Local(x)), m_Var(m_Local(y))))
               ) &&
               x == y;
      }
    };

    // Rule:  %d = cmp < %x, %y   ->   %d = cmp > %y, %x
    //
    // Family A — the mirrored relation. Trap-free and always applicable, and
    // it is the one rule that touches the body's comparisons at all, so
    // without it every `cmp` in a twin is the region's own.
    class SwapCompareRule : public AntiOptRule {
    public:
      const char *name() const override { return "swap-compare"; }

      RuleFamily family() const override { return RuleFamily::Peephole; }

      TrapTier tier() const override { return TrapTier::Tier0; }

      bool
      matches(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &) const override {
        return match(stmts[pos.stmt], m_Assign(m_AnyLocal(), m_One(m_AnyCmp())));
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &) const override {
        if (!match(stmts[pos.stmt], m_Assign(m_AnyLocal(), m_One(m_AnyCmp()))))
          return {};
        Instr copy = cloneInstr(stmts[pos.stmt]);
        auto &cmp = std::get<CmpAtom>(std::get<AssignInstr>(copy).rhs.first.v);
        std::swap(cmp.lhs, cmp.rhs);
        cmp.op = mirror(cmp.op);
        std::vector<Instr> out;
        out.push_back(std::move(copy));
        return out;
      }

      std::optional<SelfTest> selfTest() const override {
        // One rule, six relations: the claim is checked for all of them at
        // once by packing each relation's answer into its own bit.
        SelfTest t;
        t.original = [](std::int64_t a, std::int64_t b) {
          return std::optional<std::int64_t>(
              (a == b) | ((a != b) << 1) | ((a < b) << 2) | ((a <= b) << 3) | ((a > b) << 4) |
              ((a >= b) << 5)
          );
        };
        t.rewritten = [](std::int64_t a, std::int64_t b) {
          return std::optional<std::int64_t>(
              (b == a) | ((b != a) << 1) | ((b > a) << 2) | ((b >= a) << 3) | ((b < a) << 4) |
              ((b <= a) << 5)
          );
        };
        return t;
      }

    private:
      static RelOp mirror(RelOp op) {
        switch (op) {
          case RelOp::LT:
            return RelOp::GT;
          case RelOp::GT:
            return RelOp::LT;
          case RelOp::LE:
            return RelOp::GE;
          case RelOp::GE:
            return RelOp::LE;
          default:
            return op; // == and != read the same either way round
        }
      }
    };

    // Rule:  %d = e
    //   ->   %d = e;  %t = %d as i64;  %d = %t as iN
    //
    // Family A — cast round-trip. Widening keeps the value and narrowing back
    // is defined for every input (spec §6.4), so this is trap-free at any
    // width below 64 and needs no license at all.
    class CastRoundTripRule : public AntiOptRule {
    public:
      const char *name() const override { return "cast-roundtrip"; }

      RuleFamily family() const override { return RuleFamily::Peephole; }

      TrapTier tier() const override { return TrapTier::Tier0; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        std::string d;
        if (!match(stmts[pos.stmt], m_Assign(m_Local(d), m_Any())))
          return false;
        auto bits = TypeUtils::getIntBitWidth(localType(ctx.fn, ctx.lets, d));
        return bits && *bits < 64; // there has to be something wider to visit
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        std::string d;
        if (!match(stmts[pos.stmt], m_Assign(m_Local(d), m_Any())))
          return {};
        TypePtr narrow = localType(ctx.fn, ctx.lets, d);
        TypePtr wide = std::make_shared<Type>(Type{IntType{IntType::Kind::I64, {}, {}}, {}});
        const std::string t = ctx.names.fresh(wide, ctx.lets);

        std::vector<Instr> out;
        out.push_back(cloneInstr(stmts[pos.stmt]));
        out.push_back(
            assignInstr(localLV(t), simpleExpr(Atom{CastAtom{localLV(d), wide, {}}, {}}))
        );
        out.push_back(
            assignInstr(localLV(d), simpleExpr(Atom{CastAtom{localLV(t), narrow, {}}, {}}))
        );
        return out;
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        t.original = [](std::int64_t v, std::int64_t) { return std::optional<std::int64_t>(v); };
        t.rewritten = [](std::int64_t v, std::int64_t) {
          // Widen to 64, truncate back: the low 8 bits, sign-extended.
          return std::optional<std::int64_t>((std::int64_t) (std::int8_t) (std::int64_t) v);
        };
        return t;
      }
    };

  } // namespace

  void registerPeepholeRules(RuleList &out) {
    out.push_back(std::make_unique<XorTwiceRule>());
    out.push_back(std::make_unique<SplitConstantRule>());
    out.push_back(std::make_unique<FoldNegationRule>());
    out.push_back(std::make_unique<MulShiftRule>(/*plusOne=*/false, "mul-to-shift"));
    out.push_back(std::make_unique<MulShiftRule>(/*plusOne=*/true, "mul-to-shift-add"));
    out.push_back(std::make_unique<NotComplementRule>());
    out.push_back(std::make_unique<SubAsAddRule>());
    out.push_back(std::make_unique<DoubleAsMulRule>());
    out.push_back(std::make_unique<SwapCompareRule>());
    out.push_back(std::make_unique<CastRoundTripRule>());
  }

} // namespace refractir::reify::antiopt
