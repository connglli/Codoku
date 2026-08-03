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

  } // namespace

  void registerPeepholeRules(RuleList &out) {
    out.push_back(std::make_unique<XorTwiceRule>());
    out.push_back(std::make_unique<SplitConstantRule>());
  }

} // namespace refractir::reify::antiopt
