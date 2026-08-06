#include "reify/twin_box.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "reify/hyperparameters.hpp"

namespace refractir::reify {

  namespace {
    using I64 = std::int64_t;
    constexpr I64 kI64Min = std::numeric_limits<I64>::min();
    constexpr I64 kI64Max = std::numeric_limits<I64>::max();
  } // namespace

  std::vector<BranchObligation> traceObligations(const TraceBody &body) {
    std::vector<BranchObligation> out;
    out.reserve(body.checks.size());
    for (const auto &c: body.checks)
      out.push_back(BranchObligation{c.afterStmt, &c.cond, c.taken});
    return out;
  }

  Box computeBox(
      const FunDecl &fn, const TypeUtils::StructTable &structs, const TraceBody &body,
      const EntryState &entry, std::mt19937 &rng, const IntrinsicFold *fold
  ) {
    Box box;
    const std::vector<BranchObligation> branches = traceObligations(body);
    auto judge = [&](const EntryState &st) {
      ++box.passes;
      return StateSetAnalysis::check(fn, structs, body.stmts, branches, st, fold);
    };

    // Step 1 — the floor. Everything pinned is the guard rytwin already
    // emits, so if the pass cannot even prove that, no widening is possible
    // and the answer is the floor itself.
    for (const auto &[k, v]: entry.ints)
      box.leaves.push_back(BoxLeaf{k, LeafClass::Pinned, v});
    std::sort(box.leaves.begin(), box.leaves.end(), [](const BoxLeaf &a, const BoxLeaf &b) {
      return a.key < b.key;
    });
    if (!judge(entry).ok)
      return box;

    // Step 2 — ceilings. Pushing every check's requirement backward bounds each
    // leaf before a single trial is run: one that cannot move at all is pinned
    // for free, and the rest have a bound to search under rather than a
    // doubling sequence that has to discover where to stop.
    const auto ceil = StateSetAnalysis::ceilings(fn, structs, body.stmts, branches, entry, fold);
    for (auto &leaf: box.leaves) {
      auto it = ceil.find(leaf.key);
      if (it != ceil.end() && it->second.lo == it->second.hi)
        leaf.cls = LeafClass::Pinned; // nothing else could have been proven
    }

    // Free leaves. A leaf is free exactly when the trace is provable without
    // knowing it, which is one pass with the leaf left out. Nothing is wider
    // than "any value", so a leaf that clears this never enters the search.
    EntryState st = entry;
    for (auto &leaf: box.leaves) {
      EntryState without = st;
      without.ints.erase(leaf.key);
      if (judge(without).ok) {
        leaf.cls = LeafClass::Free;
        st = std::move(without);
      }
    }

    // Step 3 — lockstep bisection over what is left. Each open leaf carries a
    // radius; every round doubles all of them at once and one pass judges the
    // result, so widening costs a pass per round rather than per leaf. A
    // refused round freezes only the leaves its failing check depended on —
    // the others keep growing, which is what stops a leaf's width from
    // depending on the order it happened to be visited in.
    struct Open {
      BoxLeaf *leaf;
      std::int64_t centre;
      std::int64_t good = 0;      // widest radius proven
      std::int64_t next = 1;      // radius to try
      bool frozen = false;        //
      std::int64_t cap = kI64Max; // widest the ceiling allows
    };

    std::vector<Open> open;
    for (auto &leaf: box.leaves) {
      if (leaf.cls == LeafClass::Free)
        continue;
      // How far the ceiling allows this leaf to move at all. Saturating,
      // because an unconstrained ceiling is the full width of the type and
      // subtracting its ends would overflow — which silently pinned every
      // leaf the narrowing had nothing to say about. The wider side decides:
      // a radius is clipped to the type on the way in, so a leaf sitting at
      // its type's edge can still open in the one direction it has.
      auto span = [](std::int64_t from, std::int64_t to) -> std::int64_t {
        std::int64_t d = 0;
        if (__builtin_sub_overflow(to, from, &d))
          return kI64Max;
        return d < 0 ? 0 : d;
      };
      std::int64_t cap = kI64Max;
      if (auto it = ceil.find(leaf.key); it != ceil.end()) {
        const std::int64_t centre = leaf.range.lo;
        cap = std::max(span(it->second.lo, centre), span(centre, it->second.hi));
        if (cap <= 0)
          continue; // the ceiling holds it still; no pass spent on it
      }
      open.push_back(Open{&leaf, leaf.range.lo, 0, 1, false, cap});
    }
    if (open.empty())
      return box;

    // A leaf cannot hold what its type cannot represent, so every proposed
    // range is clipped to its own width — otherwise the search would "prove" a
    // range the guard could not even state as a literal.
    auto clip = [](const Interval &leaf, std::int64_t lo, std::int64_t hi) {
      I64 tlo = kI64Min, thi = kI64Max;
      if (leaf.bits && leaf.bits < 64) {
        tlo = -(I64(1) << (leaf.bits - 1));
        thi = (I64(1) << (leaf.bits - 1)) - 1;
      }
      return std::pair<I64, I64>{std::max(lo, tlo), std::min(hi, thi)};
    };

    auto widen = [&](const std::vector<std::int64_t> &radii) {
      EntryState trial = st;
      for (std::size_t i = 0; i < open.size(); ++i) {
        Interval iv = open[i].leaf->range;
        const auto [lo, hi] = clip(iv, open[i].centre - radii[i], open[i].centre + radii[i]);
        iv.lo = lo;
        iv.hi = hi;
        iv.unknown = false;
        trial.ints[open[i].leaf->key] = iv;
      }
      return trial;
    };

    for (std::size_t round = 0; round < rytwin::hp::kTwinBisectMaxRounds; ++round) {
      std::vector<std::int64_t> radii;
      bool anyOpen = false;
      for (auto &o: open) {
        radii.push_back(o.frozen ? o.good : o.next);
        anyOpen = anyOpen || !o.frozen;
      }
      if (!anyOpen)
        break;
      const StateSetVerdict v = judge(widen(radii));
      if (v.ok) {
        for (auto &o: open)
          if (!o.frozen) {
            o.good = o.next;
            // Doubling stops at the ceiling: past it the trial would only be
            // refused, and by a check that was known in advance.
            if (o.good >= o.cap)
              o.frozen = true;
            else
              o.next = std::min(o.next * 2, o.cap);
          }
        continue;
      }
      // Freeze what this failure actually depended on. A failure that blames
      // nothing is not about any leaf we are moving, so it freezes everything
      // — continuing would just retry the same refusal.
      bool froze = false;
      for (auto &o: open)
        if (!o.frozen && std::find(v.blame.begin(), v.blame.end(), o.leaf->key) != v.blame.end()) {
          o.frozen = true;
          froze = true;
        }
      // A refusal that names nothing still open is not about the leaves being
      // moved — retrying would only repeat it, so the round ends the search.
      if (!froze)
        for (auto &o: open)
          o.frozen = true;
    }

    // Step 3b — grow each leaf alone. Lockstep moves every open leaf together
    // and a refusal freezes all the leaves that failure depended on, so a leaf
    // stops for a failure that was really about another one: where a leaf is
    // multiplied by a hundred thousand before a sum, it reaches its own limit
    // and takes every other leaf down with it, at a fraction of the width they
    // could have had. Blame cannot tell them apart — both operands of that sum
    // genuinely feed it — so this does not argue about fault. It re-opens each
    // leaf on its own, with the others held where lockstep left them, and every
    // step is proved like any other.
    auto twice = [](std::int64_t r, std::int64_t cap) {
      if (r <= 0)
        return std::min<std::int64_t>(1, cap);
      return r > cap / 2 ? cap : r * 2;
    };
    for (auto &o: open) {
      std::int64_t next = twice(o.good, o.cap);
      for (std::size_t step = 0; step < rytwin::hp::kTwinBisectMaxRounds && next > o.good; ++step) {
        std::vector<std::int64_t> radii;
        radii.reserve(open.size());
        for (auto &p: open)
          radii.push_back(&p == &o ? next : p.good);
        if (!judge(widen(radii)).ok)
          break;
        o.good = next;
        next = twice(o.good, o.cap);
      }
      // Whatever it stopped at is the bracket step 4 settles inside.
      o.next = next;
    }

    // Step 4 — settle each frozen leaf between its last proven radius and the
    // one that failed, so a run does not stop at whatever power of two it
    // happened to reach.
    for (auto &o: open) {
      std::int64_t lo = o.good, hi = o.next;
      while (lo + 1 < hi) {
        const std::int64_t mid = lo + (hi - lo) / 2;
        std::vector<std::int64_t> radii;
        for (auto &p: open)
          radii.push_back(&p == &o ? mid : p.good);
        if (judge(widen(radii)).ok)
          lo = mid;
        else
          hi = mid;
      }
      o.good = lo;
    }

    // Trim each run by a small seeded amount: narrowing is always sound, and
    // maximal runs would make every guard for a given region identical.
    std::uniform_int_distribution<int> trim(0, rytwin::hp::kTwinBoxTrimPct);
    for (auto &o: open) {
      if (o.good <= 0)
        continue;
      const std::int64_t cut = o.good * trim(rng) / 100;
      o.good -= cut;
      if (o.good <= 0)
        continue;
      const auto [lo, hi] = clip(o.leaf->range, o.centre - o.good, o.centre + o.good);
      o.leaf->range.lo = lo;
      o.leaf->range.hi = hi;
      o.leaf->range.unknown = false;
      // A run covering the leaf's whole type admits every value it can hold,
      // which is what free means — and says it without two dead comparisons.
      const auto [tlo, thi] = clip(o.leaf->range, kI64Min, kI64Max);
      o.leaf->cls = (lo <= tlo && hi >= thi) ? LeafClass::Free : LeafClass::Ranged;
    }
    return box;
  }

} // namespace refractir::reify
