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

    bool fitsI8(std::int64_t v) { return v >= -128 && v <= 127; }

    // Addition is an expression-level `+` in RefractIR rather than an atom
    // operator, so the shapes a crossing can start from need their own name.
    enum class MbaFrom { Xor, Or, Add };

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
    class MbaRule : public AntiOptRule {
    public:
      MbaRule(MbaFrom from, const char *nm) : from_(from), name_(nm) {}

      const char *name() const override { return name_; }

      RuleFamily family() const override { return RuleFamily::Mba; }

      TrapTier tier() const override { return TrapTier::Tier1; }

      bool
      matches(const std::vector<Instr> &stmts, RulePos pos, const AntiOptContext &) const override {
        std::string d, x, y;
        return read(stmts[pos.stmt], d, x, y);
      }

      std::vector<Instr>
      apply(const std::vector<Instr> &stmts, RulePos pos, AntiOptContext &ctx) const override {
        std::string d, x, y;
        if (!read(stmts[pos.stmt], d, x, y))
          return {};
        std::vector<Instr> out;
        switch (from_) {
          case MbaFrom::Xor: {
            Expr e = opExpr(x, AtomOpKind::Or, y);
            e.rest.push_back(Expr::Tail{AddOp::Minus, opExpr(x, AtomOpKind::And, y).first, {}});
            out.push_back(assignInstr(localLV(d), std::move(e)));
            return out;
          }
          case MbaFrom::Or: {
            Expr e = opExpr(x, AtomOpKind::Xor, y);
            e.rest.push_back(Expr::Tail{AddOp::Plus, opExpr(x, AtomOpKind::And, y).first, {}});
            out.push_back(assignInstr(localLV(d), std::move(e)));
            return out;
          }
          default: {
            TypePtr ty = localType(ctx.fn, ctx.lets, y);
            if (!intRange(ty))
              return {};
            const std::string t = ctx.names.fresh(ty, ctx.lets);
            out.push_back(assignInstr(localLV(t), opExpr(x, AtomOpKind::And, y)));
            Expr e = opExpr(x, AtomOpKind::Xor, y);
            OpAtom twice;
            twice.op = AtomOpKind::Mul;
            twice.coef = Coef{IntLit{2, {}}};
            twice.rval = localLV(t);
            e.rest.push_back(Expr::Tail{AddOp::Plus, Atom{std::move(twice), {}}, {}});
            out.push_back(assignInstr(localLV(d), std::move(e)));
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
      // `%d = %x OP %y` in whichever spelling this crossing starts from.
      bool read(const Instr &ins, std::string &d, std::string &x, std::string &y) const {
        if (from_ == MbaFrom::Add)
          return match(
              ins, m_Assign(m_Local(d), m_Pair(AddOp::Plus, m_Var(m_Local(x)), m_Var(m_Local(y))))
          );
        const AtomOpKind op = from_ == MbaFrom::Xor ? AtomOpKind::Xor : AtomOpKind::Or;
        return match(ins, m_Assign(m_Local(d), m_One(m_Op(op, m_Local(x), m_Local(y)))));
      }

      MbaFrom from_;
      const char *name_;
    };

  } // namespace

  void registerMbaRules(RuleList &out) {
    out.push_back(std::make_unique<MbaRule>(MbaFrom::Xor, "mba-xor"));
    out.push_back(std::make_unique<MbaRule>(MbaFrom::Or, "mba-or"));
    out.push_back(std::make_unique<MbaRule>(MbaFrom::Add, "mba-add"));
  }

} // namespace refractir::reify::antiopt
