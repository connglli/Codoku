#include "reify/antiopt.hpp"

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <utility>
#include <variant>

#include "analysis/type_utils.hpp"
#include "antiopt/internal.hpp"
#include "ast/clone.hpp"
#include "reify/name_alloc.hpp"
#include "reify/rewrite.hpp"

namespace refractir::reify {

  // --- the catalog ----------------------------------------------------------

  namespace antiopt {

    // One list, built once, from the per-family files in antiopt/. A family
    // registers its own rules there, so adding one touches exactly one file.
    const RuleList &catalog() {
      static const RuleList rules = [] {
        RuleList v;
        registerPeepholeRules(v);
        registerMbaRules(v);
        registerStructureRules(v);
        registerLicensedRules(v);
        registerControlRules(v);
        return v;
      }();
      return rules;
    }

  } // namespace antiopt

  // --- the engine -----------------------------------------------------------

  AntiOptReport antiOptimize(
      std::vector<Instr> &stmts, std::vector<PathCheck> &checks, AntiOptContext &ctx,
      const AntiOptAccept &accept, std::size_t attempts
  ) {
    AntiOptReport rep;
    // A body the caller refuses from the start cannot judge anything: every
    // rewrite would be measured against a body that already fails. That is not
    // a reason to leave it alone, though — it is the worst case for a caller
    // like rytwin, whose unjudged body is its region's own statements and so a
    // copy of it. What a proof buys is permission to introduce operations that
    // can trap; rules that introduce none are identities whatever the state,
    // and so is any composition of them. Those still apply, and the rest stand
    // down.
    // No predicate at all is a caller saying it has no oracle — a generator
    // rewriting concrete code, where there is no set of states to prove
    // anything over. It lands in the same place as a body the caller refuses.
    rep.trapFreeOnly = !accept || !accept(stmts);

    struct Cand {
      const AntiOptRule *rule;
      RulePos pos;
    };

    // One scan per attempt, not one per round. An application splices
    // statements in, so every position after it means something else
    // afterwards — a candidate list outlives its body by one rewrite. Scanning
    // again is also what makes the draw fair: with a catalog this size, a list
    // built once and then half-invalidated hands most of its picks to whatever
    // happened to sit early in the body.
    for (std::size_t attempt = 0; attempt < attempts; ++attempt) {
      // Facts are about the body as it stands, and every application changes
      // it, so they are never carried across one.
      if (ctx.facts)
        ctx.facts->refresh(stmts);

      std::vector<Cand> cands;
      for (const auto &rule: antiopt::catalog()) {
        if (rep.trapFreeOnly && rule->tier() != TrapTier::Tier0)
          continue;
        for (std::size_t i = 0; i < stmts.size(); ++i)
          if (rule->matches(stmts, RulePos{i}, ctx))
            cands.push_back({rule.get(), RulePos{i}});
      }
      if (cands.empty())
        break;
      shuffleAndCap(cands, ctx.rng, 1);

      {
        const Cand &c = cands.front();
        const std::size_t width = c.rule->width();
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
        // two of them together. `accept` has the last word — except where it
        // had nothing to say to begin with, and only trap-free rules ran.
        if (rep.trapFreeOnly || accept(stmts)) {
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

} // namespace refractir::reify
