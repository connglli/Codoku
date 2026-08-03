#include <algorithm>
#include <optional>
#include <string>
#include <variant>

#include "analysis/type_utils.hpp"
#include "ast/clone.hpp"
#include "ast/match.hpp"
#include "internal.hpp"
#include "reify/expr_gen.hpp"

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

    // Rule:  %t = <e>;  ...;  %d = f(%t)
    //   ->   %t = <e>;  ...;  %u = <e>;  %d = f(%u)
    //
    // Family C — un-CSE. A temp that later statements read is a common
    // subexpression an optimizer has already found; recomputing it hands that
    // work back, and the two copies can then take different rewrites and stop
    // looking like one value.
    //
    // The recomputation is only the same value if nothing it depends on has
    // moved since, which is what `source` checks. Tier1: the copy runs the
    // same arithmetic the original did, so it traps only where the original
    // would have — but the interval pass has the last word all the same.
    class UnCseRule : public AntiOptRule {
    public:
      const char *name() const override { return "un-cse"; }

      RuleFamily family() const override { return RuleFamily::Structure; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        return source(stmts, pos, ctx).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        auto src = source(stmts, pos, ctx);
        if (!src)
          return {};
        const auto &def = std::get<AssignInstr>(stmts[src->at]);
        TypePtr ty = localType(ctx.fn, ctx.lets, src->name);
        if (!ty)
          return {};
        const std::string copy = ctx.names.fresh(ty, ctx.lets);

        Instr user = cloneInstr(stmts[pos.stmt]);
        renameReads(user, src->name, copy);
        std::vector<Instr> out;
        out.push_back(assignInstr(localLV(copy), cloneExpr(def.rhs)));
        out.push_back(std::move(user));
        return out;
      }

    private:
      struct Source {
        std::string name; // the temp being read
        std::size_t at;   // where it was computed
      };

      // A local this statement reads whose defining statement can be replayed
      // here: it assigns a whole local, touches no memory, does not read what
      // it writes, and nothing since has written anything it read.
      std::optional<Source>
      source(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx) const {
        const Touches use = touchesOf(stmts[pos.stmt]);
        if (use.memory)
          return std::nullopt;
        std::vector<std::string> names(use.reads.begin(), use.reads.end());
        std::sort(names.begin(), names.end()); // one candidate, not one per run
        for (const auto &nm: names) {
          if (!localType(ctx.fn, ctx.lets, nm))
            continue;
          for (std::size_t j = pos.stmt; j-- > 0;) {
            const Touches def = touchesOf(stmts[j]);
            if (def.writes != nm)
              continue;
            if (def.memory || def.reads.count(nm) || !std::get_if<AssignInstr>(&stmts[j]) ||
                !std::get<AssignInstr>(stmts[j]).lhs.accesses.empty())
              break; // this definition is the one in effect; it cannot be replayed
            bool stale = false;
            for (std::size_t k = j + 1; k < pos.stmt && !stale; ++k) {
              const Touches between = touchesOf(stmts[k]);
              stale = between.memory || def.reads.count(between.writes) > 0;
            }
            if (!stale)
              return Source{nm, j};
            break;
          }
        }
        return std::nullopt;
      }
    };

    // Rule:  S   ->   S;  %dead = <generated expression over live locals>
    //
    // Family C — dead statements. They read what the body reads and compute
    // something nothing consumes, so a reader has to follow them to find out.
    // The expression comes from rysmith's own generator rather than a shape of
    // this file's invention, so the padding is written in the same hand as the
    // program around it.
    class DeadStatementRule : public AntiOptRule {
    public:
      const char *name() const override { return "dead-statement"; }

      RuleFamily family() const override { return RuleFamily::Structure; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        return pos.stmt < stmts.size() && !readable(ctx).vars.empty();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        const VarCatalogue vars = readable(ctx);
        if (vars.vars.empty())
          return {};
        std::uniform_int_distribution<std::size_t> pick(0, vars.vars.size() - 1);
        TypePtr ty = vars.vars[pick(ctx.rng)].type;
        const std::string dead = ctx.names.fresh(ty, ctx.lets);

        ExprGenConfig cfg;
        // No floats (every operation can trap and none of it is provable yet),
        // no intrinsics (each would need a declaration this rule cannot add),
        // no pointer arithmetic (the catalogue holds no pointers to do it to).
        cfg.enableFp = false;
        cfg.enableIntrinsics = false;
        cfg.enablePtrArith = false;
        cfg.maxAtoms = 2;
        Expr e = genExpr(ctx.rng, /*sym=*/nullptr, vars, ty, /*onPath=*/false, cfg, dead);

        std::vector<Instr> out;
        out.push_back(cloneInstr(stmts[pos.stmt]));
        out.push_back(assignInstr(localLV(dead), std::move(e)));
        return out;
      }

    private:
      // The scalar integers in scope, as the generator wants them. Pointers,
      // aggregates and vectors are left out: a generated load is a memory
      // operation with a real trap, and this statement is meant to be free.
      // So are locals declared `undef` — reading one before the program writes
      // it is not a dead statement, it is a program the checker rejects.
      static VarCatalogue readable(const AntiOptContext &ctx) {
        VarCatalogue vars;
        auto add = [&](const std::string &nm, const TypePtr &ty, bool isParam) {
          if (!intRange(ty))
            return;
          VarEntry v;
          v.name = nm;
          v.type = ty;
          v.isParam = isParam;
          vars.vars.push_back(std::move(v));
        };
        for (const auto &p: ctx.fn.params)
          add(p.name.name, p.type, true);
        for (const auto &l: ctx.fn.lets)
          if (l.init && l.init->kind != InitVal::Kind::Undef)
            add(l.name.name, l.type, false);
        return vars;
      }
    };

    // Rule:  %d = <e>
    //   ->   %p = addr %cell;  store %p, <e>;  %d = load %p
    //
    // Family C — memory routing. The value now travels through a cell instead
    // of a register, so undoing it means proving that nothing else can reach
    // that cell: dataflow the optimizer read for free becomes alias analysis.
    class MemoryRoutingRule : public AntiOptRule {
    public:
      const char *name() const override { return "memory-routing"; }

      RuleFamily family() const override { return RuleFamily::Structure; }

      TrapTier tier() const override { return TrapTier::Tier1; }

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
        if (!ty)
          return {};
        // The cell has to be a `let mut` for `addr` to be legal (spec §6.8),
        // which is what the allocator hands out.
        const std::string cell = ctx.names.fresh(ty, ctx.lets);
        const std::string ptr =
            ctx.names.fresh(std::make_shared<Type>(Type{PtrType{ty, {}}, {}}), ctx.lets);

        std::vector<Instr> out;
        out.push_back(assignInstr(localLV(ptr), simpleExpr(Atom{AddrAtom{localLV(cell), {}}, {}})));
        StoreInstr st;
        st.ptr = simpleExpr(localAtom(ptr));
        st.val = cloneExpr(std::get<AssignInstr>(stmts[pos.stmt]).rhs);
        out.push_back(Instr{std::move(st)});
        out.push_back(assignInstr(localLV(d), simpleExpr(Atom{LoadAtom{localLV(ptr), {}}, {}})));
        return out;
      }
    };

  } // namespace

  void registerStructureRules(RuleList &out) {
    out.push_back(std::make_unique<SplitStatementRule>());
    out.push_back(std::make_unique<SwapAdjacentRule>());
    out.push_back(std::make_unique<UnCseRule>());
    out.push_back(std::make_unique<DeadStatementRule>());
    out.push_back(std::make_unique<MemoryRoutingRule>());
  }

} // namespace refractir::reify::antiopt
