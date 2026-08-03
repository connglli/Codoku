#include "reify/antiopt.hpp"

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <utility>
#include <variant>

#include "analysis/type_utils.hpp"
#include "ast/clone.hpp"
#include "ast/sir_printer.hpp"
#include "reify/hyperparameters.hpp"
#include "reify/rewrite.hpp"

namespace refractir::reify {

  // --- names ---------------------------------------------------------------

  namespace {

    std::string typeKey(const TypePtr &t) {
      auto bits = TypeUtils::getIntBitWidth(t);
      return bits ? "i" + std::to_string(*bits) : "?";
    }

    LetDecl makeLet(const std::string &name, const TypePtr &type, std::int64_t init, bool mut) {
      LetDecl d;
      d.isMutable = mut;
      d.name = LocalId{name, {}};
      d.type = type;
      d.init = InitVal{InitVal::Kind::Int, IntLit{init, {}}, {}};
      return d;
    }

  } // namespace

  namespace {

    bool nameTaken(const std::vector<LetDecl> &lets, const std::string &nm) {
      for (const auto &l: lets)
        if (l.name.name == nm)
          return true;
      return false;
    }

  } // namespace

  std::string NameAllocator::fresh(const TypePtr &type, std::vector<LetDecl> &lets) {
    // A function may hold several rewritten bodies, each with its own
    // allocator, and they all declare into the same list — so a name is only
    // fresh once nothing else has claimed it.
    std::string nm = prefix_ + std::to_string(next_++);
    while (nameTaken(lets, nm))
      nm = prefix_ + std::to_string(next_++);
    lets.push_back(makeLet(nm, type, 0, /*mut=*/true));
    return nm;
  }

  std::string
  NameAllocator::literal(std::int64_t value, const TypePtr &type, std::vector<LetDecl> &lets) {
    const std::string key = typeKey(type);
    for (const auto &[v, k, nm]: pool_)
      if (v == value && k == key)
        return nm;
    std::string nm = prefix_ + "k" + std::to_string(next_++);
    while (nameTaken(lets, nm))
      nm = prefix_ + "k" + std::to_string(next_++);
    lets.push_back(makeLet(nm, type, value, /*mut=*/false));
    pool_.emplace_back(value, key, nm);
    return nm;
  }

  // --- small AST helpers ----------------------------------------------------

  namespace {

    LValue localLV(const std::string &n) { return LValue{LocalId{n, {}}, {}, {}}; }

    Expr simpleExpr(Atom a) { return Expr{std::move(a), {}, {}}; }

    Instr assignInstr(const LValue &lhs, Expr rhs) {
      return Instr{AssignInstr{lhs, std::move(rhs), {}}};
    }

    // `%dst = <left> <op> %right` — RefractIR takes an id or literal on the
    // left of a binary atom and requires an lvalue on the right.
    Expr opExpr(const std::string &left, AtomOpKind op, const std::string &right) {
      OpAtom o;
      o.op = op;
      o.coef = Coef{LocalOrSymId{LocalId{left, {}}}};
      o.rval = localLV(right);
      return simpleExpr(Atom{std::move(o), {}});
    }

    // The declared type of a local in `fn`, or null.
    TypePtr localType(const FunDecl &fn, const std::vector<LetDecl> &extra, const std::string &nm) {
      for (const auto &p: fn.params)
        if (p.name.name == nm)
          return p.type;
      for (const auto &l: fn.lets)
        if (l.name.name == nm)
          return l.type;
      for (const auto &l: extra)
        if (l.name.name == nm)
          return l.type;
      return nullptr;
    }

    // Every local an instruction reads, and the one it writes (when it writes
    // a whole local). Used to decide whether two statements may be swapped.
    struct Touches {
      std::unordered_set<std::string> reads;
      std::string writes;
      bool memory = false; // a load or store: order with other memory is kept
    };

    void scanExprLocals(const Expr &e, Touches &t);

    void scanAtomLocals(const Atom &a, Touches &t) {
      std::visit(
          [&](const auto &x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, CoefAtom>) {
              if (auto id = std::get_if<LocalOrSymId>(&x.coef))
                if (auto loc = std::get_if<LocalId>(id))
                  t.reads.insert(loc->name);
            } else if constexpr (std::is_same_v<T, RValueAtom>)
              t.reads.insert(x.rval.base.name);
            else if constexpr (std::is_same_v<T, UnaryAtom>)
              t.reads.insert(x.rval.base.name);
            else if constexpr (std::is_same_v<T, OpAtom>) {
              if (auto id = std::get_if<LocalOrSymId>(&x.coef))
                if (auto loc = std::get_if<LocalId>(id))
                  t.reads.insert(loc->name);
              t.reads.insert(x.rval.base.name);
            } else if constexpr (std::is_same_v<T, CastAtom>) {
              if (auto lv = std::get_if<LValue>(&x.src))
                t.reads.insert(lv->base.name);
            } else if constexpr (std::is_same_v<T, AddrAtom>) {
              t.reads.insert(x.lv.base.name);
              t.memory = true;
            } else if constexpr (std::is_same_v<T, LoadAtom>) {
              t.reads.insert(x.rval.base.name);
              t.memory = true;
            } else if constexpr (std::is_same_v<T, PtrIndexAtom> ||
                                 std::is_same_v<T, PtrFieldAtom>) {
              t.reads.insert(x.rval.base.name);
              t.memory = true;
            } else if constexpr (std::is_same_v<T, CmpAtom>) {
              for (const auto *sv: {&x.lhs, &x.rhs})
                if (auto rv = std::get_if<RValue>(sv))
                  t.reads.insert(rv->base.name);
                else if (auto id = std::get_if<LocalOrSymId>(&std::get<Coef>(*sv)))
                  if (auto loc = std::get_if<LocalId>(id))
                    t.reads.insert(loc->name);
            } else if constexpr (std::is_same_v<T, SelectAtom>) {
              if (x.cond) {
                scanExprLocals(x.cond->lhs, t);
                scanExprLocals(x.cond->rhs, t);
              }
              if (x.maskExpr)
                scanExprLocals(*x.maskExpr, t);
              for (const auto *sv: {&x.vtrue, &x.vfalse})
                if (auto rv = std::get_if<RValue>(sv))
                  t.reads.insert(rv->base.name);
                else if (auto id = std::get_if<LocalOrSymId>(&std::get<Coef>(*sv)))
                  if (auto loc = std::get_if<LocalId>(id))
                    t.reads.insert(loc->name);
            } else if constexpr (std::is_same_v<T, CallAtom>) {
              t.memory = true; // an intrinsic may observe or touch anything
              for (const auto &arg: x.args)
                if (arg)
                  scanExprLocals(*arg, t);
            }
          },
          a.v
      );
    }

    void scanExprLocals(const Expr &e, Touches &t) {
      scanAtomLocals(e.first, t);
      for (const auto &tail: e.rest)
        scanAtomLocals(tail.atom, t);
    }

    Touches touchesOf(const Instr &ins) {
      Touches t;
      std::visit(
          [&](const auto &x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, AssignInstr>) {
              scanExprLocals(x.rhs, t);
              for (const auto &acc: x.lhs.accesses)
                if (auto ai = std::get_if<AccessIndex>(&acc))
                  if (auto id = std::get_if<LocalOrSymId>(&ai->index))
                    if (auto loc = std::get_if<LocalId>(id))
                      t.reads.insert(loc->name);
              t.writes = x.lhs.base.name;
            } else if constexpr (std::is_same_v<T, StoreInstr>) {
              scanExprLocals(x.ptr, t);
              scanExprLocals(x.val, t);
              t.memory = true;
            } else if constexpr (std::is_same_v<T, RequireInstr> ||
                                 std::is_same_v<T, AssumeInstr>) {
              scanExprLocals(x.cond.lhs, t);
              scanExprLocals(x.cond.rhs, t);
            }
          },
          ins
      );
      return t;
    }

  } // namespace

  // --- the catalog ----------------------------------------------------------

  namespace {

    using IdentityRewriteRuleBase = AntiOptRule;

    bool fitsI8(std::int64_t v) { return v >= -128 && v <= 127; }

    // `%d = %x OP %y` with both operands plain locals — the shape every
    // crossing below rewrites. `Add` stands for the `+` of a two-atom chain,
    // which is how RefractIR spells addition.
    // Addition is an expression-level `+` in RefractIR rather than an atom
    // operator, so the shapes a crossing can start from need their own name.
    enum class MbaFrom { Xor, Or, Add };

    struct TwoOperand {
      MbaFrom op;
      std::string lhs, rhs;
    };

    std::optional<TwoOperand> operands(const Instr &ins) {
      auto *ai = std::get_if<AssignInstr>(&ins);
      if (!ai || !ai->lhs.accesses.empty())
        return std::nullopt;
      auto localOf = [](const Coef &c) -> const std::string * {
        if (auto id = std::get_if<LocalOrSymId>(&c))
          if (auto loc = std::get_if<LocalId>(id))
            return &loc->name;
        return nullptr;
      };
      // `%x & %y` / `%x | %y` / `%x ^ %y`: one atom, one operator.
      if (ai->rhs.rest.empty()) {
        auto *op = std::get_if<OpAtom>(&ai->rhs.first.v);
        if (!op || !op->rval.accesses.empty())
          return std::nullopt;
        const std::string *l = localOf(op->coef);
        if (!l)
          return std::nullopt;
        if (op->op == AtomOpKind::Or)
          return TwoOperand{MbaFrom::Or, *l, op->rval.base.name};
        if (op->op == AtomOpKind::Xor)
          return TwoOperand{MbaFrom::Xor, *l, op->rval.base.name};
        return std::nullopt;
      }
      // `%x + %y`: two coef atoms joined by `+`.
      if (ai->rhs.rest.size() == 1 && ai->rhs.rest[0].op == AddOp::Plus) {
        auto *a = std::get_if<CoefAtom>(&ai->rhs.first.v);
        auto *b = std::get_if<CoefAtom>(&ai->rhs.rest[0].atom.v);
        if (!a || !b)
          return std::nullopt;
        const std::string *l = localOf(a->coef);
        const std::string *r = localOf(b->coef);
        if (!l || !r)
          return std::nullopt;
        return TwoOperand{MbaFrom::Add, *l, *r};
      }
      return std::nullopt;
    }

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
        auto *ai = std::get_if<AssignInstr>(&stmts[pos.stmt]);
        if (!ai || !ai->lhs.accesses.empty())
          return false;
        return TypeUtils::getIntBitWidth(localType(ctx.fn, ctx.lets, ai->lhs.base.name)) !=
               std::nullopt;
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        TypePtr ty = localType(ctx.fn, ctx.lets, ai.lhs.base.name);
        auto bits = TypeUtils::getIntBitWidth(ty);
        if (!bits)
          return {};
        // Any mask works; a varied one keeps the disguise from being a tell.
        const std::int64_t hi = *bits >= 64 ? INT64_MAX : (std::int64_t{1} << (*bits - 1)) - 1;
        std::uniform_int_distribution<std::int64_t> pick(1, std::max<std::int64_t>(1, hi));
        const std::string k = ctx.names.literal(pick(ctx.rng), ty, ctx.lets);

        std::vector<Instr> out;
        out.push_back(cloneInstr(stmts[pos.stmt]));
        out.push_back(assignInstr(ai.lhs, opExpr(ai.lhs.base.name, AtomOpKind::Xor, k)));
        out.push_back(assignInstr(ai.lhs, opExpr(ai.lhs.base.name, AtomOpKind::Xor, k)));
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
        // A branch the trace recorded is read between two statements, so a
        // swap across it would change what that condition sees.
        for (const auto &c: ctx.checks)
          if (c.afterStmt == pos.stmt + 1)
            return false;
        const Touches a = touchesOf(stmts[pos.stmt]);
        const Touches b = touchesOf(stmts[pos.stmt + 1]);
        if (a.memory || b.memory)
          return false;
        if (a.writes.empty() || b.writes.empty())
          return false;
        if (a.writes == b.writes)
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
          if (auto co = std::get_if<CoefAtom>(&ai->rhs.rest[i].atom.v))
            if (std::holds_alternative<IntLit>(co->coef))
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
        const std::int64_t k =
            std::get<IntLit>(std::get<CoefAtom>(out.rhs.rest[*idx].atom.v).coef).value;
        // Both halves are literals of the statement's own type, so both have
        // to be representable in it — splitting 125 into -32 and 157 is not a
        // rewrite of an i8 statement, it is a program the checker rejects.
        auto bits = TypeUtils::getIntBitWidth(localType(ctx.fn, ctx.lets, ai.lhs.base.name));
        if (!bits || *bits > 64)
          return {};
        const std::int64_t lim = *bits >= 64 ? INT64_MAX : (std::int64_t{1} << (*bits - 1));
        const std::int64_t tlo = *bits >= 64 ? INT64_MIN : -lim;
        const std::int64_t thi = *bits >= 64 ? INT64_MAX : lim - 1;
        // Keep the split small, and inside what both halves can hold.
        std::int64_t alo = std::max<std::int64_t>(tlo, -32);
        std::int64_t ahi = std::min<std::int64_t>(thi, 32);
        if (alo > ahi)
          return {};
        std::uniform_int_distribution<std::int64_t> pick(alo, ahi);
        const std::int64_t a = pick(ctx.rng);
        std::int64_t b = 0;
        if (__builtin_sub_overflow(k, a, &b) || b < tlo || b > thi)
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
        auto *ai = std::get_if<AssignInstr>(&stmts[pos.stmt]);
        if (!ai || !ai->lhs.accesses.empty() || ai->rhs.rest.size() < 2)
          return false;
        return TypeUtils::getIntBitWidth(localType(ctx.fn, ctx.lets, ai->lhs.base.name)) !=
               std::nullopt;
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        TypePtr ty = localType(ctx.fn, ctx.lets, ai.lhs.base.name);
        if (!ty)
          return {};
        const std::string tmp = ctx.names.fresh(ty, ctx.lets);

        // `%d = a + b + c` becomes `%t = a + b; %d = %t + c`.
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

    // Rule:  %d = %x ^ %y   ->   %d = %x | %y - %x & %y
    // Rule:  %d = %x | %y   ->   %d = %x ^ %y + %x & %y
    // Rule:  %d = %x + %y   ->   %t = %x & %y;  %d = %x ^ %y + 2 * %t
    //
    // Family B — arithmetic <-> bitwise crossings.
    //
    // These are the ones worth having. An optimizer simplifies within the
    // arithmetic domain or within the bitwise domain; it rarely translates
    // between them, so `x ^ y` written as `(x | y) - (x & y)` has to be
    // *reasoned* back rather than pattern-matched. All are Tier1: they
    // introduce `+ - *`, and whether the intermediates stay inside the type is
    // for the acceptance check to decide, not the rule.
    //
    // The textbook form of the carry identity is `(x ^ y) + ((x & y) << 1)`.
    // That one is wrong here: RefractIR's `<<` is signed arithmetic and traps
    // on a negative left operand (spec §7.1), so it is UB for any two negative
    // operands. `2 * (x & y)` says the same thing and survives.
    class MbaRule : public IdentityRewriteRuleBase {
    public:
      MbaRule(MbaFrom from, const char *nm) : from_(from), name_(nm) {}

      const char *name() const override { return name_; }

      RuleFamily family() const override { return RuleFamily::Mba; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool
      matches(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &) const override {
        auto ops = operands(stmts[pos.stmt]);
        return ops && ops->op == from_;
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        auto ops = operands(stmts[pos.stmt]);
        if (!ops)
          return {};
        const auto &ai = std::get<AssignInstr>(stmts[pos.stmt]);
        std::vector<Instr> out;
        switch (from_) {
          case MbaFrom::Xor: {
            // x ^ y == (x | y) - (x & y)
            Expr e = opExpr(ops->lhs, AtomOpKind::Or, ops->rhs);
            e.rest.push_back(
                Expr::Tail{AddOp::Minus, opExpr(ops->lhs, AtomOpKind::And, ops->rhs).first, {}}
            );
            out.push_back(assignInstr(ai.lhs, std::move(e)));
            return out;
          }
          case MbaFrom::Or: {
            // x | y == (x ^ y) + (x & y)
            Expr e = opExpr(ops->lhs, AtomOpKind::Xor, ops->rhs);
            e.rest.push_back(
                Expr::Tail{AddOp::Plus, opExpr(ops->lhs, AtomOpKind::And, ops->rhs).first, {}}
            );
            out.push_back(assignInstr(ai.lhs, std::move(e)));
            return out;
          }
          default: {
            // x + y == (x ^ y) + 2 * (x & y), the carry written out.
            TypePtr ty = localType(ctx.fn, ctx.lets, ops->rhs);
            if (!ty || !TypeUtils::getIntBitWidth(ty))
              return {};
            const std::string t = ctx.names.fresh(ty, ctx.lets);
            out.push_back(assignInstr(localLV(t), opExpr(ops->lhs, AtomOpKind::And, ops->rhs)));
            Expr e = opExpr(ops->lhs, AtomOpKind::Xor, ops->rhs);
            OpAtom twice;
            twice.op = AtomOpKind::Mul;
            twice.coef = Coef{IntLit{2, {}}};
            twice.rval = localLV(t);
            e.rest.push_back(Expr::Tail{AddOp::Plus, Atom{std::move(twice), {}}, {}});
            out.push_back(assignInstr(ai.lhs, std::move(e)));
            return out;
          }
        }
      }

      std::optional<SelfTest> selfTest() const override {
        const MbaFrom from = from_;
        SelfTest t;
        t.original = [from](std::int64_t a, std::int64_t b) -> std::optional<std::int64_t> {
          switch (from) {
            case MbaFrom::Xor:
              return a ^ b;
            case MbaFrom::Or:
              return a | b;
            default:
              return fitsI8(a + b) ? std::optional<std::int64_t>(a + b) : std::nullopt;
          }
        };
        t.rewritten = [from](std::int64_t a, std::int64_t b) -> std::optional<std::int64_t> {
          switch (from) {
            case MbaFrom::Xor:
              return fitsI8((a | b) - (a & b)) ? std::optional<std::int64_t>((a | b) - (a & b))
                                               : std::nullopt;
            case MbaFrom::Or:
              return fitsI8((a ^ b) + (a & b)) ? std::optional<std::int64_t>((a ^ b) + (a & b))
                                               : std::nullopt;
            default: {
              const std::int64_t carry = 2 * (a & b);
              const std::int64_t sum = (a ^ b) + carry;
              if (!fitsI8(carry) || !fitsI8(sum))
                return std::nullopt;
              return sum;
            }
          }
        };
        return t;
      }

    private:
      MbaFrom from_;
      const char *name_;
    };

    const std::vector<std::unique_ptr<AntiOptRule>> &catalog() {
      static const std::vector<std::unique_ptr<AntiOptRule>> rules = [] {
        std::vector<std::unique_ptr<AntiOptRule>> v;
        v.push_back(std::make_unique<XorTwiceRule>());
        v.push_back(std::make_unique<SplitConstantRule>());
        v.push_back(std::make_unique<SplitStatementRule>());
        v.push_back(std::make_unique<SwapAdjacentRule>());
        v.push_back(std::make_unique<MbaRule>(MbaFrom::Xor, "mba-xor"));
        v.push_back(std::make_unique<MbaRule>(MbaFrom::Or, "mba-or"));
        v.push_back(std::make_unique<MbaRule>(MbaFrom::Add, "mba-add"));
        return v;
      }();
      return rules;
    }

  } // namespace

  // --- the engine -----------------------------------------------------------

  AntiOptReport antiOptimize(
      std::vector<Instr> &stmts, std::vector<PathCheck> &checks, AntiOptContext &ctx,
      const AntiOptAccept &accept
  ) {
    AntiOptReport rep;
    // Without a body the caller already accepts there is nothing to judge a
    // rewrite against, so such a body is left as it is.
    if (!accept(stmts))
      return rep;

    for (std::size_t round = 0; round < rytwin::hp::kTwinRewriteRounds; ++round) {
      // Re-scanned every round: an application shifts every position after it.
      struct Cand {
        const AntiOptRule *rule;
        RulePos pos;
      };

      std::vector<Cand> cands;
      for (const auto &rule: catalog())
        for (std::size_t i = 0; i < stmts.size(); ++i)
          if (rule->matches(stmts, RulePos{i}, ctx))
            cands.push_back({rule.get(), RulePos{i}});
      if (cands.empty())
        break;
      shuffleAndCap(cands, ctx.rng, rytwin::hp::kTwinRewritesPerRound);

      for (const auto &c: cands) {
        const std::size_t width = c.rule->width();
        if (c.pos.stmt + width > stmts.size())
          continue; // an earlier application moved this position
        if (!c.rule->matches(stmts, c.pos, ctx))
          continue;
        auto replacement = c.rule->apply(stmts, c.pos, ctx);
        if (replacement.empty())
          continue;

        std::vector<Instr> before = cloneInstrs(stmts);
        // Only the indices move, and a PathCheck carries a condition that
        // cannot be copied — so the snapshot is of the indices alone.
        std::vector<std::size_t> checkIdxBefore;
        checkIdxBefore.reserve(checks.size());
        for (const auto &chk: checks)
          checkIdxBefore.push_back(chk.afterStmt);
        const std::size_t added = replacement.size();
        stmts.erase(stmts.begin() + (long) c.pos.stmt, stmts.begin() + (long) (c.pos.stmt + width));
        stmts.insert(
            stmts.begin() + (long) c.pos.stmt, std::make_move_iterator(replacement.begin()),
            std::make_move_iterator(replacement.end())
        );
        // A recorded branch names the statement it is read before, so a splice
        // that changes how many statements precede it has to move it too —
        // otherwise the condition would be judged at the wrong point of the
        // body, and judged wrong is worse than not judged at all.
        if (added != width)
          for (auto &chk: checks)
            if (chk.afterStmt > c.pos.stmt)
              chk.afterStmt = chk.afterStmt + added - width;

        // R3: each rule is sound on its own, and that says nothing about the
        // two of them together. `accept` has the last word.
        if (accept(stmts)) {
          ++rep.applied;
        } else {
          stmts = std::move(before);
          for (std::size_t i = 0; i < checks.size(); ++i)
            checks[i].afterStmt = checkIdxBefore[i];
          ++rep.rolledBack;
        }
      }
    }
    return rep;
  }

  std::optional<std::size_t> selfTestRules(std::string &failure) {
    std::size_t checked = 0;
    for (const auto &rule: catalog()) {
      auto t = rule->selfTest();
      if (!t)
        continue;
      ++checked;
      for (std::int64_t a = -128; a <= 127; ++a)
        for (std::int64_t b = -128; b <= 127; ++b) {
          auto want = t->original(a, b);
          auto got = t->rewritten(a, b);
          if (!want || !got)
            continue; // undefined for this pair on one side; nothing to compare
          if (*want != *got) {
            failure = std::string(rule->name()) + ": " + std::to_string(a) + ", " +
                      std::to_string(b) + " gives " + std::to_string(*got) + ", expected " +
                      std::to_string(*want);
            return std::nullopt;
          }
        }
    }
    return checked;
  }

} // namespace refractir::reify
