#include <memory>
#include <optional>
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

    // Family E — control, made arithmetic.
    //
    // The body is straight-line, so there is no branch to rewrite. What there
    // is instead is the *choosing*: selects, comparisons, and the intrinsics
    // that stand for a choice. Each of these has an arithmetic form, and the
    // arithmetic form is the one an optimizer has to work to recognize —
    // recovering `min` from a mask takes reasoning, while recovering it from a
    // select takes a pattern.
    //
    // Two things RefractIR gives this family for free: `i1` true is all-ones
    // (spec §6.4), so `c as iN` is a ready-made mask rather than a multiply;
    // and `cmp` takes an id or a literal on either side, so a comparison can
    // be rebuilt without materializing its operands first.
    //
    // Not here: the reverse of E1 (mask arithmetic back into a select), which
    // would undo the rule above it and leave the engine oscillating between
    // two spellings of the same body. Vectors wait on the state-set pass being
    // able to say anything per lane, and floats are out of the catalog
    // entirely — every FP operation can trap and reassociation is not
    // value-preserving under RNE.

    // Whether a select value is usable as a `cmp` operand or an arm: it always
    // is, both being SelectVal, but the type has to be known to size a temp.
    std::optional<std::string> nameOfSelectVal(const SelectVal &sv) {
      if (auto rv = std::get_if<RValue>(&sv))
        return rv->accesses.empty() ? std::optional<std::string>(rv->base.name) : std::nullopt;
      if (auto co = std::get_if<Coef>(&sv))
        if (auto id = std::get_if<LocalOrSymId>(co))
          if (auto loc = std::get_if<LocalId>(id))
            return loc->name;
      return std::nullopt;
    }

    // An atom of a select value: a read of the local, or the literal itself.
    std::optional<Atom> atomOfSelectVal(const SelectVal &sv) {
      if (auto nm = nameOfSelectVal(sv))
        return buildLocalAtom(*nm);
      if (auto co = std::get_if<Coef>(&sv))
        if (auto lit = std::get_if<IntLit>(co))
          return buildIntAtom(lit->value);
      return std::nullopt;
    }

    // `%c = <cmp op> lhs, rhs` into a fresh i1.
    std::string
    emitCmp(RelOp op, SelectVal lhs, SelectVal rhs, AntiOptContext &ctx, std::vector<Instr> &out) {
      TypePtr i1 = std::make_shared<Type>(Type{IntType{IntType::Kind::ICustom, 1, {}}, {}});
      const std::string c = ctx.names.fresh(i1, ctx.lets);
      CmpAtom cmp;
      cmp.op = op;
      cmp.lhs = std::move(lhs);
      cmp.rhs = std::move(rhs);
      out.push_back(buildAssign(buildLValue(c), buildExpr(Atom{std::move(cmp), {}})));
      return c;
    }

    // Rule:  %d = select c, %a, %b   ->   %d = %b + ((%a - %b) & (c as iN))
    //
    // E1 — the branchless form. `c as iN` is all-ones or zero, so the mask
    // picks one arm's difference and adds it back to the other. Both arms are
    // now evaluated where the select evaluated one (spec §7.2), which is why
    // this is Tier1 even though the mask itself cannot trap: the difference
    // has to fit.
    class SelectToMaskRule : public AntiOptRule {
    public:
      const char *name() const override { return "select-to-mask"; }

      RuleFamily family() const override { return RuleFamily::Control; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        return read(stmts[pos.stmt], ctx) != nullptr;
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        const SelectAtom *sel = read(stmts[pos.stmt], ctx);
        if (!sel)
          return {};
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        TypePtr ty = localType(ctx.fn, ctx.lets, ai.lhs.base.name);
        auto vtrue = atomOfSelectVal(sel->vtrue);
        auto vfalse = atomOfSelectVal(sel->vfalse);
        if (!ty || !vtrue || !vfalse)
          return {};

        std::vector<Instr> out;
        // The condition as a value. A mask-form select already has one; a
        // cond-form select has a comparison whose operands `cmp` can take.
        std::string cond;
        if (sel->maskExpr) {
          std::string nm;
          if (!match(sel->maskExpr->first, m_Var(m_Local(nm))) || !sel->maskExpr->rest.empty())
            return {};
          cond = nm;
        } else {
          std::string lhs, rhs;
          if (!match(sel->cond->lhs, m_One(m_Var(m_Local(lhs)))) ||
              !match(sel->cond->rhs, m_One(m_Var(m_Local(rhs)))))
            return {};
          cond = emitCmp(
              sel->cond->op, SelectVal{RValue{buildLValue(lhs)}},
              SelectVal{RValue{buildLValue(rhs)}}, ctx, out
          );
        }

        const std::string mask = ctx.names.fresh(ty, ctx.lets);
        const std::string diff = ctx.names.fresh(ty, ctx.lets);
        out.push_back(
            buildAssign(buildLValue(mask), buildExpr(Atom{CastAtom{buildLValue(cond), ty, {}}, {}}))
        );
        Expr sub{cloneAtom(*vtrue), {}, {}};
        appendTail(sub, AddOp::Minus, cloneAtom(*vfalse));
        out.push_back(buildAssign(buildLValue(diff), std::move(sub)));
        out.push_back(buildAssign(buildLValue(diff), buildOpExpr(diff, AtomOpKind::And, mask)));
        Expr sum{cloneAtom(*vfalse), {}, {}};
        appendTail(sum, AddOp::Plus, buildLocalAtom(diff));
        out.push_back(buildAssign(ai.lhs, std::move(sum)));
        return out;
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        t.body = "  %d = select %x > %y, %x, %y;";
        return t;
      }

    private:
      // `%d = select …, <arm>, <arm>` with everything the rewrite needs to
      // name: whole locals for the destination and the condition, and arms
      // that are locals or literals.
      static const SelectAtom *read(const Instr &ins, const AntiOptContext &ctx) {
        auto *ai = std::get_if<AssignInstr>(&ins);
        if (!ai || !ai->lhs.accesses.empty() || !ai->rhs.rest.empty())
          return nullptr;
        auto *sel = std::get_if<SelectAtom>(&ai->rhs.first.v);
        if (!sel || !intRange(localType(ctx.fn, ctx.lets, ai->lhs.base.name)))
          return nullptr;
        if (!atomOfSelectVal(sel->vtrue) || !atomOfSelectVal(sel->vfalse))
          return nullptr;
        return sel;
      }
    };

    // Rule:  %c = cmp < %x, %y   ->   %c1 = cmp <= %x, %y;
    //                                 %c2 = cmp != %x, %y;
    //                                 %c  = %c1 & %c2
    //
    // E2 — a relation as two weaker ones. Every relation is the conjunction or
    // disjunction of a pair, and i1 arithmetic is how RefractIR spells `&&`
    // and `||` (there is no short-circuit operator, and none is needed on
    // values that cannot trap). Trap-free throughout.
    class SplitCompareRule : public AntiOptRule {
    public:
      const char *name() const override { return "split-compare"; }

      RuleFamily family() const override { return RuleFamily::Control; }

      TrapTier tier() const override { return TrapTier::Tier0; }

      bool
      matches(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &) const override {
        return read(stmts[pos.stmt]) != nullptr;
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        const CmpAtom *cmp = read(stmts[pos.stmt]);
        if (!cmp)
          return {};
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        const auto parts = split(cmp->op);

        std::vector<Instr> out;
        const std::string c1 = emitCmp(parts.first_, cmp->lhs, cmp->rhs, ctx, out);
        const std::string c2 = emitCmp(parts.second_, cmp->lhs, cmp->rhs, ctx, out);
        out.push_back(buildAssign(ai.lhs, buildOpExpr(c1, parts.join, c2)));
        return out;
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        t.body = "  %c = cmp < %x, %y;\n  %d = %c as i8;";
        return t;
      }

    private:
      struct Parts {
        RelOp first_;
        RelOp second_;
        AtomOpKind join;
      };

      // `a < b` is `(a <= b) && (a != b)`, and so on round the six. Each pair
      // is weaker than what it replaces, which is the point: neither half says
      // what the original said.
      static Parts split(RelOp op) {
        switch (op) {
          case RelOp::LT:
            return {RelOp::LE, RelOp::NE, AtomOpKind::And};
          case RelOp::GT:
            return {RelOp::GE, RelOp::NE, AtomOpKind::And};
          case RelOp::LE:
            return {RelOp::LT, RelOp::EQ, AtomOpKind::Or};
          case RelOp::GE:
            return {RelOp::GT, RelOp::EQ, AtomOpKind::Or};
          case RelOp::EQ:
            return {RelOp::LE, RelOp::GE, AtomOpKind::And};
          default:
            return {RelOp::LT, RelOp::GT, AtomOpKind::Or};
        }
      }

      static const CmpAtom *read(const Instr &ins) {
        auto *ai = std::get_if<AssignInstr>(&ins);
        if (!ai || !ai->lhs.accesses.empty() || !ai->rhs.rest.empty())
          return nullptr;
        return std::get_if<CmpAtom>(&ai->rhs.first.v);
      }
    };

    // Rule:  %c = cmp < %x, %y   ->   %t = %x - %y;  %c = cmp < %t, 0
    //
    // E2 again, in the other direction: the comparison moves into arithmetic.
    // Tier1, and genuinely so — the difference has to fit, which is a fact
    // about the box and not about the relation.
    class CompareToDiffRule : public AntiOptRule {
    public:
      const char *name() const override { return "compare-to-diff"; }

      RuleFamily family() const override { return RuleFamily::Control; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        std::string x, y;
        return read(stmts[pos.stmt], ctx, x, y) != nullptr;
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        std::string x, y;
        const CmpAtom *cmp = read(stmts[pos.stmt], ctx, x, y);
        if (!cmp)
          return {};
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        TypePtr ty = localType(ctx.fn, ctx.lets, x);
        const std::string diff = ctx.names.fresh(ty, ctx.lets);

        Expr sub = buildExpr(buildLocalAtom(x));
        appendTail(sub, AddOp::Minus, buildLocalAtom(y));
        std::vector<Instr> out;
        out.push_back(buildAssign(buildLValue(diff), std::move(sub)));
        // A bare `0` in a `cmp` is an i32 literal and nothing else: the
        // comparison's operands must share a width (spec §6.5), so the zero
        // gets a cell of the operand's own type rather than being written
        // inline. It reads the same at i32 and is a type error at every other
        // width.
        const std::string zeroCell = ctx.names.literal(0, ty, ctx.lets);
        CmpAtom zero;
        zero.op = cmp->op;
        zero.lhs = SelectVal{RValue{buildLValue(diff)}};
        zero.rhs = SelectVal{RValue{buildLValue(zeroCell)}};
        out.push_back(buildAssign(ai.lhs, buildExpr(Atom{std::move(zero), {}})));
        return out;
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        t.body = "  %c = cmp < %x, %y;\n  %d = %c as i8;";
        return t;
      }

    private:
      // Both operands have to be whole integer locals of one type: the
      // difference is a value of that type, and it is what the relation is
      // restated against.
      static const CmpAtom *
      read(const Instr &ins, const AntiOptContext &ctx, std::string &x, std::string &y) {
        auto *ai = std::get_if<AssignInstr>(&ins);
        if (!ai || !ai->lhs.accesses.empty() || !ai->rhs.rest.empty())
          return nullptr;
        auto *cmp = std::get_if<CmpAtom>(&ai->rhs.first.v);
        if (!cmp)
          return nullptr;
        auto lhs = nameOfSelectVal(cmp->lhs), rhs = nameOfSelectVal(cmp->rhs);
        if (!lhs || !rhs)
          return nullptr;
        TypePtr lt = localType(ctx.fn, ctx.lets, *lhs), rt = localType(ctx.fn, ctx.lets, *rhs);
        if (!intRange(lt) || !intRange(rt) ||
            TypeUtils::getIntBitWidth(lt) != TypeUtils::getIntBitWidth(rt))
          return nullptr;
        x = *lhs;
        y = *rhs;
        return cmp;
      }
    };

    // Rule:  %d = call @min(%a, %b)   ->   %c = cmp < %a, %b;
    //                                      %d = select %c, %a, %b
    //
    // E4 — an intrinsic is a name for something the program could have said
    // itself, and the name is what a reader matches on. Spelling it out
    // removes the name; the identity is the intrinsic's own documented one
    // (docs/intrinsics.md), including where it traps — `@abs` is undefined at
    // INT_MIN and so is the negation that replaces it.
    class IntrinsicExpandRule : public AntiOptRule {
    public:
      const char *name() const override { return "intrinsic-expand"; }

      RuleFamily family() const override { return RuleFamily::Control; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool matches(
          const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &ctx
      ) const override {
        Call c;
        return read(stmts[pos.stmt], ctx, c);
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        Call c;
        if (!read(stmts[pos.stmt], ctx, c))
          return {};
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        TypePtr ty = localType(ctx.fn, ctx.lets, ai.lhs.base.name);
        std::vector<Instr> out;

        auto arg = [&](std::size_t i) { return SelectVal{RValue{buildLValue(c.args[i])}}; };
        if (c.callee == "@min" || c.callee == "@max") {
          const std::string cond =
              emitCmp(c.callee == "@min" ? RelOp::LT : RelOp::GT, arg(0), arg(1), ctx, out);
          out.push_back(buildAssign(ai.lhs, selectOn(cond, arg(0), arg(1))));
          return out;
        }
        if (c.callee == "@abs") {
          const std::string cond = emitCmp(RelOp::LT, arg(0), zeroOf(ty, ctx), ctx, out);
          const std::string neg = ctx.names.fresh(ty, ctx.lets);
          Expr sub = buildExpr(buildIntAtom(0));
          appendTail(sub, AddOp::Minus, buildLocalAtom(c.args[0]));
          out.push_back(buildAssign(buildLValue(neg), std::move(sub)));
          out.push_back(
              buildAssign(ai.lhs, selectOn(cond, SelectVal{RValue{buildLValue(neg)}}, arg(0)))
          );
          return out;
        }
        // @signum: -1, 0 or +1. i1 true is all-ones, so the two comparisons
        // cast to iN are already -1/0 and their difference is the sign.
        const std::string neg = emitCmp(RelOp::LT, arg(0), zeroOf(ty, ctx), ctx, out);
        const std::string posi = emitCmp(RelOp::GT, arg(0), zeroOf(ty, ctx), ctx, out);
        const std::string a = ctx.names.fresh(ty, ctx.lets);
        const std::string b = ctx.names.fresh(ty, ctx.lets);
        out.push_back(
            buildAssign(buildLValue(a), buildExpr(Atom{CastAtom{buildLValue(neg), ty, {}}, {}}))
        );
        out.push_back(
            buildAssign(buildLValue(b), buildExpr(Atom{CastAtom{buildLValue(posi), ty, {}}, {}}))
        );
        Expr diff = buildExpr(buildLocalAtom(a));
        appendTail(diff, AddOp::Minus, buildLocalAtom(b));
        out.push_back(buildAssign(ai.lhs, std::move(diff)));
        return out;
      }

      std::optional<SelfTest> selfTest() const override {
        SelfTest t;
        t.body = "  %d = call @min(%x, %y);";
        t.decls = "intrinsic @min(%a: i8, %b: i8) : i8;";
        return t;
      }

    private:
      struct Call {
        std::string callee;
        std::vector<std::string> args;
      };

      // As in compare-to-diff: a literal zero in a `cmp` is i32, so it takes
      // a cell of the compared type instead.
      static SelectVal zeroOf(const TypePtr &ty, AntiOptContext &ctx) {
        return SelectVal{RValue{buildLValue(ctx.names.literal(0, ty, ctx.lets))}};
      }

      static Expr selectOn(const std::string &cond, SelectVal vtrue, SelectVal vfalse) {
        SelectAtom sel;
        sel.maskExpr = std::make_unique<Expr>(buildExpr(buildLocalAtom(cond)));
        sel.vtrue = std::move(vtrue);
        sel.vfalse = std::move(vfalse);
        return buildExpr(Atom{std::move(sel), {}});
      }

      // `%d = call @<one we can write out>(<locals>)`. The resolved overload
      // has to be there: without it this is a call to whatever the program
      // named, and only an intrinsic carries the identity being relied on.
      static bool read(const Instr &ins, const AntiOptContext &ctx, Call &out) {
        auto *ai = std::get_if<AssignInstr>(&ins);
        if (!ai || !ai->lhs.accesses.empty() || !ai->rhs.rest.empty())
          return false;
        auto *call = std::get_if<CallAtom>(&ai->rhs.first.v);
        if (!call || !call->resolvedIntrinsic)
          return false;
        const std::string &nm = call->callee.name;
        const std::size_t want = nm == "@abs" || nm == "@signum" ? 1 : 2;
        if ((nm != "@abs" && nm != "@signum" && nm != "@min" && nm != "@max") ||
            call->args.size() != want)
          return false;
        if (!intRange(localType(ctx.fn, ctx.lets, ai->lhs.base.name)))
          return false;

        out.callee = nm;
        out.args.clear();
        for (const auto &a: call->args) {
          std::string an;
          if (!a || !a->rest.empty() || !match(a->first, m_Var(m_Local(an))))
            return false;
          if (!intRange(localType(ctx.fn, ctx.lets, an)))
            return false;
          out.args.push_back(an);
        }
        return true;
      }
    };

  } // namespace

  void registerControlRules(RuleList &out) {
    out.push_back(std::make_unique<SelectToMaskRule>());
    out.push_back(std::make_unique<SplitCompareRule>());
    out.push_back(std::make_unique<CompareToDiffRule>());
    out.push_back(std::make_unique<IntrinsicExpandRule>());
  }

} // namespace refractir::reify::antiopt
