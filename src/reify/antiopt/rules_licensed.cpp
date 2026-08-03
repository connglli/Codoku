#include <algorithm>
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

    // Family D — rules licensed by what the caller knows.
    //
    // The rest of the catalog rewrites by identities that hold for every
    // input. These hold only over the states the caller's obligation covers,
    // and they read that obligation to find out where: a mask that is a no-op
    // because the value cannot be larger, a literal spelled as a read of
    // something pinned to it, an arm that never runs because a comparison is
    // settled.
    //
    // That is why no generic obfuscator can write them — they depend on facts
    // the guard is enforcing — and also why they are the ones to get wrong.
    // Two disciplines keep them honest: the engine refreshes the facts before
    // every attempt (a licence granted on a body that has since changed is no
    // licence), and a rule asks about the point it fires at rather than about
    // the body in general.
    //
    // No `selfTest` on any of them: their claim is not an identity over all
    // operands, it is an identity over a range, and the range is the caller's.
    // The body-level re-check and rytwin's spot checks cover them instead.

    // The declared width of a local, for the literals a rule wants to build.
    std::optional<std::uint32_t> widthOf(const AntiOptContext &ctx, const std::string &nm) {
      return TypeUtils::getIntBitWidth(localType(ctx.fn, ctx.lets, nm));
    }

    // Every local in scope, so a rule can look for one the facts pin.
    std::vector<std::string> localsInScope(const AntiOptContext &ctx) {
      std::vector<std::string> out;
      for (const auto &p: ctx.fn.params)
        out.push_back(p.name.name);
      for (const auto &l: ctx.fn.lets)
        out.push_back(l.name.name);
      return out;
    }

    // Every literal this statement holds, and where. A literal is either an
    // atom of the `+`/`-` chain or the left operand of a binary atom (`7 * %x`
    // — the grammar takes an id or a literal there and an lvalue on the
    // right), and a rule that replaces one has to be able to write both back.
    struct LiteralSite {
      std::size_t tail; // 0 = the first atom, else rest[tail - 1]
      bool opCoef;      // the literal is a binary atom's left operand
      I64 value;
    };

    std::vector<LiteralSite> literalSites(const AssignInstr &ai) {
      std::vector<LiteralSite> out;
      auto scan = [&](const Atom &a, std::size_t tail) {
        I64 v = 0;
        if (match(a, m_Coef(m_Int(v))))
          out.push_back({tail, false, v});
        else if (auto op = std::get_if<OpAtom>(&a.v))
          if (auto lit = std::get_if<IntLit>(&op->coef))
            out.push_back({tail, true, lit->value});
      };
      scan(ai.rhs.first, 0);
      for (std::size_t i = 0; i < ai.rhs.rest.size(); ++i)
        scan(ai.rhs.rest[i].atom, i + 1);
      return out;
    }

    // Put `c` where `site` found its literal.
    void writeCoef(AssignInstr &ai, const LiteralSite &site, Coef c) {
      Atom &atom = site.tail == 0 ? ai.rhs.first : ai.rhs.rest[site.tail - 1].atom;
      if (site.opCoef)
        std::get<OpAtom>(atom.v).coef = std::move(c);
      else
        std::get<CoefAtom>(atom.v).coef = std::move(c);
    }

    // Rule:  %d = ... 7 ...   ->   %d = ... %k ...      (%k is 7 on the box)
    //
    // D1 — a value the table pins to one number is that number on every state
    // the guard admits, so a literal can be spelled as a read of it. This is
    // the strongest rule for making a body look state-dependent where it is
    // not: in a checksum epilogue most temps are derived from pinned leaves,
    // so most of the block qualifies.
    class KnownConstantRule : public AntiOptRule {
    public:
      const char *name() const override { return "known-constant"; }

      RuleFamily family() const override { return RuleFamily::Licensed; }

      TrapTier tier() const override { return TrapTier::Tier0; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        return find(stmts, pos, ctx).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        auto hit = find(stmts, pos, ctx);
        if (!hit)
          return {};
        Instr out = cloneInstr(stmts[pos.stmt]);
        writeCoef(
            std::get<AssignInstr>(out), hit->site, Coef{LocalOrSymId{LocalId{hit->local, {}}}}
        );
        std::vector<Instr> res;
        res.push_back(std::move(out));
        return res;
      }

    private:
      struct Hit {
        LiteralSite site;
        std::string local;
      };

      // A literal atom of this statement, and a local the facts pin to its
      // value at this point. Same width, so the read means the same thing the
      // literal did.
      std::optional<Hit>
      find(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx) const {
        if (!ctx.facts)
          return std::nullopt;
        auto *ai = std::get_if<AssignInstr>(&stmts[pos.stmt]);
        if (!ai)
          return std::nullopt;
        auto width = widthOf(ctx, ai->lhs.base.name);
        if (!width)
          return std::nullopt;

        std::vector<std::string> scope = localsInScope(ctx);
        std::sort(scope.begin(), scope.end()); // one answer, not one per run
        for (const auto &site: literalSites(*ai))
          for (const auto &nm: scope) {
            if (widthOf(ctx, nm) != width)
              continue;
            auto r = ctx.facts->rangeBefore(pos.stmt, nm);
            if (r && r->lo == site.value && r->hi == site.value)
              return Hit{site, nm};
          }
        return std::nullopt;
      }
    };

    // Rule:  %d = ... %x ...   ->   %t = %x & %mask;  %d = ... %t ...
    //
    // D5 — a range no-op. Masking off bits the value cannot have set is the
    // identity on the box and nowhere else, so undoing it means re-deriving
    // the bound the guard is enforcing.
    class RangeMaskRule : public AntiOptRule {
    public:
      const char *name() const override { return "range-mask"; }

      RuleFamily family() const override { return RuleFamily::Licensed; }

      TrapTier tier() const override { return TrapTier::Tier0; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        return find(stmts, pos, ctx).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        auto hit = find(stmts, pos, ctx);
        if (!hit)
          return {};
        TypePtr ty = localType(ctx.fn, ctx.lets, hit->local);
        const std::string mask = ctx.names.literal(hit->mask, ty, ctx.lets);
        const std::string tmp = ctx.names.fresh(ty, ctx.lets);

        Instr user = cloneInstr(stmts[pos.stmt]);
        renameReads(user, hit->local, tmp);
        std::vector<Instr> out;
        out.push_back(assignInstr(localLV(tmp), opExpr(hit->local, AtomOpKind::And, mask)));
        out.push_back(std::move(user));
        return out;
      }

    private:
      struct Hit {
        std::string local;
        I64 mask;
      };

      // A local this statement reads whose range fits inside a mask of all
      // ones. The smallest such mask is the tightest claim, and it is the one
      // that says the most about what the guard admits.
      std::optional<Hit>
      find(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx) const {
        if (!ctx.facts)
          return std::nullopt;
        if (!std::get_if<AssignInstr>(&stmts[pos.stmt]))
          return std::nullopt;
        const Touches use = touchesOf(stmts[pos.stmt]);
        if (use.memory)
          return std::nullopt;
        std::vector<std::string> names(use.reads.begin(), use.reads.end());
        std::sort(names.begin(), names.end());
        for (const auto &nm: names) {
          auto width = widthOf(ctx, nm);
          auto r = ctx.facts->rangeBefore(pos.stmt, nm);
          if (!width || *width < 2 || !r || r->lo < 0)
            continue;
          for (std::uint32_t bits = 1; bits + 1 < *width; ++bits) {
            const I64 mask = (I64(1) << bits) - 1;
            if (r->hi <= mask)
              return Hit{nm, mask};
          }
        }
        return std::nullopt;
      }
    };

    // Rule:  %d = <e>   ->   %t = <e>;  %d = select <settled cond>, %t, <junk>
    //
    // D3 — an arm that never runs. The condition is one the facts settle over
    // the whole box, so the value is always the true arm; select is lazy
    // (spec §7.2), so the false arm may hold anything at all and never costs a
    // trap. A reader has to prove the condition to know that.
    class TrueSelectRule : public AntiOptRule {
    public:
      const char *name() const override { return "true-select"; }

      RuleFamily family() const override { return RuleFamily::Licensed; }

      TrapTier tier() const override { return TrapTier::Tier0; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        return find(stmts, pos, ctx).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        auto hit = find(stmts, pos, ctx);
        if (!hit)
          return {};
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        TypePtr ty = localType(ctx.fn, ctx.lets, ai.lhs.base.name);
        if (!ty)
          return {};
        const std::string kept = ctx.names.fresh(ty, ctx.lets);
        TypePtr guardTy = localType(ctx.fn, ctx.lets, hit->local);
        const std::string bound = ctx.names.literal(hit->bound, guardTy, ctx.lets);
        // The arm that never runs is still a literal of the destination's
        // type, so it has to be one that type can hold.
        auto span = intRange(ty);
        if (!span)
          return {};
        std::uniform_int_distribution<I64> pickJunk(span->first, span->second);
        const I64 junk = pickJunk(ctx.rng);

        // `%x >= lo`, which the facts say holds everywhere the guard does.
        auto cond = std::make_unique<Cond>();
        cond->lhs = simpleExpr(localAtom(hit->local));
        cond->op = RelOp::GE;
        cond->rhs = simpleExpr(localAtom(bound));

        SelectAtom sel;
        sel.cond = std::move(cond);
        sel.vtrue = SelectVal{RValue{localLV(kept)}};
        sel.vfalse = SelectVal{Coef{IntLit{junk, {}}}};

        std::vector<Instr> out;
        out.push_back(assignInstr(localLV(kept), cloneExpr(ai.rhs)));
        out.push_back(assignInstr(ai.lhs, simpleExpr(Atom{std::move(sel), {}})));
        return out;
      }

    private:
      struct Hit {
        std::string local;
        I64 bound; // what the local is known to be at least
      };

      std::optional<Hit>
      find(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx) const {
        if (!ctx.facts)
          return std::nullopt;
        auto *ai = std::get_if<AssignInstr>(&stmts[pos.stmt]);
        if (!ai || !ai->lhs.accesses.empty() ||
            !intRange(localType(ctx.fn, ctx.lets, ai->lhs.base.name)))
          return std::nullopt;
        std::vector<std::string> scope = localsInScope(ctx);
        std::sort(scope.begin(), scope.end());
        for (const auto &nm: scope) {
          if (!intRange(localType(ctx.fn, ctx.lets, nm)))
            continue;
          auto r = ctx.facts->rangeBefore(pos.stmt, nm);
          // A range whose low end is the type's own floor settles nothing a
          // reader could not have worked out without the guard.
          auto width = widthOf(ctx, nm);
          if (!r || !width || *width > 63)
            continue;
          if (r->lo <= -(I64(1) << (*width - 1)))
            continue;
          return Hit{nm, r->lo};
        }
        return std::nullopt;
      }
    };

    // Rule:  %d = <e>   ->   %d = <e>;  %d = %d ^ %f;  %d = %d ^ %f
    //
    // D4 — a fake dependency on a leaf the guard never mentions. The pair
    // cancels, so the value cannot depend on it; the guard is silent about it,
    // so a reader has no note saying it does not. That reads as an
    // under-fitted defensive check rather than as a memo of one state.
    class FreeLeafMixRule : public AntiOptRule {
    public:
      const char *name() const override { return "free-leaf-mix"; }

      RuleFamily family() const override { return RuleFamily::Licensed; }

      TrapTier tier() const override { return TrapTier::Tier0; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        return find(stmts, pos, ctx).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        auto leaf = find(stmts, pos, ctx);
        if (!leaf)
          return {};
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        const std::string d = ai.lhs.base.name;
        std::vector<Instr> out;
        out.push_back(cloneInstr(stmts[pos.stmt]));
        out.push_back(assignInstr(localLV(d), opExpr(d, AtomOpKind::Xor, *leaf)));
        out.push_back(assignInstr(localLV(d), opExpr(d, AtomOpKind::Xor, *leaf)));
        return out;
      }

    private:
      // A free local of the destination's own width — `^` wants both operands
      // the same width (spec §6.5).
      std::optional<std::string>
      find(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx) const {
        if (!ctx.facts)
          return std::nullopt;
        std::string d;
        if (!match(stmts[pos.stmt], m_Assign(m_Local(d), m_Any())))
          return std::nullopt;
        auto width = widthOf(ctx, d);
        if (!width)
          return std::nullopt;
        std::vector<std::string> scope = localsInScope(ctx);
        std::sort(scope.begin(), scope.end());
        for (const auto &nm: scope)
          if (nm != d && widthOf(ctx, nm) == width && ctx.facts->isFree(nm))
            return nm;
        return std::nullopt;
      }
    };

    // Rule:  %d = ... 2 ...   ->   %t = %x >>> %k;  %d = ... %t ...
    //
    // D6 — a constant the guard makes. %x is not pinned; it is merely bounded
    // tightly enough that every state the guard admits shifts it to the same
    // number. The literal is gone and what replaced it is constant only in
    // combination with the guard's own range.
    class BoundDerivedRule : public AntiOptRule {
    public:
      const char *name() const override { return "bound-derived"; }

      RuleFamily family() const override { return RuleFamily::Licensed; }

      TrapTier tier() const override { return TrapTier::Tier0; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        return find(stmts, pos, ctx).has_value();
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        auto hit = find(stmts, pos, ctx);
        if (!hit)
          return {};
        TypePtr ty = localType(ctx.fn, ctx.lets, hit->local);
        const std::string amount = ctx.names.literal(hit->shift, ty, ctx.lets);
        const std::string tmp = ctx.names.fresh(ty, ctx.lets);

        Instr out = cloneInstr(stmts[pos.stmt]);
        writeCoef(std::get<AssignInstr>(out), hit->site, Coef{LocalOrSymId{LocalId{tmp, {}}}});

        std::vector<Instr> res;
        res.push_back(assignInstr(localLV(tmp), opExpr(hit->local, AtomOpKind::LShr, amount)));
        res.push_back(std::move(out));
        return res;
      }

    private:
      struct Hit {
        LiteralSite site;
        std::string local;
        I64 shift;
      };

      // A literal of this statement, and a non-negative local whose whole
      // range shifts down to exactly that literal.
      std::optional<Hit>
      find(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx) const {
        if (!ctx.facts)
          return std::nullopt;
        auto *ai = std::get_if<AssignInstr>(&stmts[pos.stmt]);
        if (!ai)
          return std::nullopt;

        const auto literals = literalSites(*ai);
        if (literals.empty())
          return std::nullopt;

        auto dstWidth = widthOf(ctx, ai->lhs.base.name);
        std::vector<std::string> scope = localsInScope(ctx);
        std::sort(scope.begin(), scope.end());
        for (const auto &site: literals) {
          if (site.value <= 0)
            continue;
          for (const auto &nm: scope) {
            auto width = widthOf(ctx, nm);
            auto r = ctx.facts->rangeBefore(pos.stmt, nm);
            if (!width || width != dstWidth || *width > 63 || !r || r->lo < 0)
              continue;
            for (std::uint32_t k = 1; k < *width; ++k)
              if ((r->lo >> k) == site.value && (r->hi >> k) == site.value)
                return Hit{site, nm, (I64) k};
          }
        }
        return std::nullopt;
      }
    };

  } // namespace

  void registerLicensedRules(RuleList &out) {
    out.push_back(std::make_unique<KnownConstantRule>());
    out.push_back(std::make_unique<RangeMaskRule>());
    out.push_back(std::make_unique<TrueSelectRule>());
    out.push_back(std::make_unique<FreeLeafMixRule>());
    out.push_back(std::make_unique<BoundDerivedRule>());
  }

} // namespace refractir::reify::antiopt
