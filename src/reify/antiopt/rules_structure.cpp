#include <optional>
#include <string>
#include <variant>

#include "ast/clone.hpp"
#include "ast/match.hpp"
#include "internal.hpp"

namespace refractir::reify::antiopt {

  namespace {

    using namespace refractir::pat;

    // Rule:  %d = a + b + c
    //   ->   %t = a + b;  %d = %t + c
    //
    // Family C — statement expansion, which is how a body stops matching the
    // region statement for statement. Tier1: the part computed first has to fit
    // the type on its own, which the whole chain did not require.
    class SplitStatementRule : public AntiOptRule {
    public:
      const char *name() const override { return "split-statement"; }

      RuleFamily family() const override { return RuleFamily::Structure; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        std::string d;
        return match(stmts[pos.stmt], m_Assign(m_Local(d), m_ChainAtLeast(3))) &&
               intRange(localType(ctx.fn, ctx.lets, d)).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        TypePtr ty = localType(ctx.fn, ctx.lets, ai.lhs.base.name);
        if (!ty)
          return {};
        const std::string tmp = ctx.names.fresh(ty, ctx.lets);

        Expr head{cloneAtom(ai.rhs.first), {}, {}};
        head.rest.push_back(Expr::Tail{ai.rhs.rest[0].op, cloneAtom(ai.rhs.rest[0].atom), {}});
        Expr tail{Atom{CoefAtom{Coef{LocalOrSymId{LocalId{tmp, {}}}}, {}}, {}}, {}, {}};
        for (std::size_t i = 1; i < ai.rhs.rest.size(); ++i)
          tail.rest.push_back(Expr::Tail{ai.rhs.rest[i].op, cloneAtom(ai.rhs.rest[i].atom), {}});

        std::vector<Instr> out;
        out.push_back(assignInstr(localLV(tmp), std::move(head)));
        out.push_back(assignInstr(ai.lhs, std::move(tail)));
        return out;
      }
    };

    // Rule:  S1; S2
    //   ->   S2; S1        (neither reads what the other writes)
    //
    // Family C — order. Two statements that share no local and touch no memory
    // compute the same thing in either order, and a reader can no longer assume
    // the body follows the region's own sequence.
    class SwapAdjacentRule : public AntiOptRule {
    public:
      const char *name() const override { return "swap-adjacent"; }

      RuleFamily family() const override { return RuleFamily::Structure; }

      TrapTier tier() const override { return TrapTier::Tier0; }

      // Two statements in, two out — without this the engine would erase one
      // and insert two, duplicating whichever it kept.
      std::size_t width() const override { return 2; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        if (pos.stmt + 1 >= stmts.size())
          return false;
        // A branch the trace recorded is read between two statements, so a swap
        // across it would change what that condition sees.
        for (const auto &c: ctx.checks)
          if (c.afterStmt == pos.stmt + 1)
            return false;
        const Touches a = touchesOf(stmts[pos.stmt]);
        const Touches b = touchesOf(stmts[pos.stmt + 1]);
        if (a.memory || b.memory)
          return false;
        if (a.writes.empty() || b.writes.empty() || a.writes == b.writes)
          return false;
        return !b.reads.count(a.writes) && !a.reads.count(b.writes);
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &) const override {
        std::vector<Instr> out;
        out.push_back(cloneInstr(stmts[pos.stmt + 1]));
        out.push_back(cloneInstr(stmts[pos.stmt]));
        return out;
      }
    };

  } // namespace

  void registerStructureRules(RuleList &out) {
    out.push_back(std::make_unique<SplitStatementRule>());
    out.push_back(std::make_unique<SwapAdjacentRule>());
  }

} // namespace refractir::reify::antiopt
