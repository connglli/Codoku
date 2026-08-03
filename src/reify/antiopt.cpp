#include "reify/antiopt.hpp"

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <utility>
#include <variant>

#include "analysis/type_utils.hpp"
#include "antiopt/internal.hpp"
#include "ast/clone.hpp"
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
      // A declaration's initializer has to suit its type: a scratch cell a
      // rule routes a value through is a pointer, and `let mut %p: ptr i32 =
      // 0;` is not a program the checker accepts.
      if (type && std::holds_alternative<PtrType>(type->v))
        d.init = InitVal{InitVal::Kind::Null, IntLit{0, {}}, {}};
      else if (TypeUtils::getFloatBitWidth(type))
        d.init = InitVal{InitVal::Kind::Float, FloatLit{(double) init, {}}, {}};
      else
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

  // --- the catalog ----------------------------------------------------------

  namespace {

    // One list, built once, from the per-family files in antiopt/. A family
    // registers its own rules there, so adding one touches exactly one file.
    const antiopt::RuleList &catalog() {
      static const antiopt::RuleList rules = [] {
        antiopt::RuleList v;
        antiopt::registerPeepholeRules(v);
        antiopt::registerMbaRules(v);
        antiopt::registerStructureRules(v);
        antiopt::registerLicensedRules(v);
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
      // Facts are about the body as it stands, and every application changes
      // it — so they are refreshed here and again before each attempt, never
      // carried across one.
      if (ctx.facts)
        ctx.facts->refresh(stmts);
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
        if (ctx.facts)
          ctx.facts->refresh(stmts);
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
          auto it = std::find_if(rep.byRule.begin(), rep.byRule.end(), [&](const auto &e) {
            return e.first == c.rule->name();
          });
          if (it == rep.byRule.end())
            rep.byRule.emplace_back(c.rule->name(), 1);
          else
            ++it->second;
        } else {
          stmts = std::move(before);
          for (std::size_t i = 0; i < checks.size(); ++i)
            checks[i].afterStmt = checkIdxBefore[i];
          ++rep.rolledBack;
        }
      }
    }

    // The allocator hands out names in the order the rewriting happened, so
    // declaring them in that order hands a reader the order too. Shuffling
    // them among their own slots costs nothing — every one of these
    // initializers is a literal, so no declaration depends on another — and
    // renaming them is not on offer: the caller tells its own scratch from the
    // program's state by exactly this prefix.
    std::vector<std::size_t> slots;
    for (std::size_t i = 0; i < ctx.lets.size(); ++i)
      if (ctx.lets[i].name.name.rfind(ctx.names.prefix(), 0) == 0)
        slots.push_back(i);
    if (slots.size() > 1) {
      std::vector<LetDecl> mine;
      for (std::size_t i: slots)
        mine.push_back(ctx.lets[i]);
      std::shuffle(mine.begin(), mine.end(), ctx.rng);
      for (std::size_t k = 0; k < slots.size(); ++k)
        ctx.lets[slots[k]] = std::move(mine[k]);
    }
    return rep;
  }

  std::string describeRules(const AntiOptReport &rep) {
    std::string out;
    for (const auto &[nm, n]: rep.byRule) {
      if (!out.empty())
        out += ", ";
      out += nm + " x" + std::to_string(n);
    }
    return out;
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
