#include "analysis/reducibility.hpp"
#include <variant>

namespace refractir {

  ReducibilityResult ReducibilityResult::check(const CFG &cfg, const DomTree &dt) {
    ReducibilityResult res;
    for (std::size_t u = 0; u < cfg.blocks.size(); ++u) {
      if (dt.rpoNumber[u] == DomTree::kNone)
        continue; // unreachable source: its edges constrain nothing
      for (std::size_t v: cfg.succ[u]) {
        const bool retreating = dt.rpoNumber[v] <= dt.rpoNumber[u];
        if (retreating && !dt.dominates(v, u))
          res.offenders.push_back({u, v});
      }
    }
    return res;
  }

  PassResult ReducibilityCheck::run(FunDecl &fun, DiagBag &diags) {
    CFG cfg = CFG::build(fun, diags);
    if (diags.hasErrors())
      return PassResult::Error;

    DomTree dt = DomTree::build(cfg);
    ReducibilityResult res = ReducibilityResult::check(cfg, dt);
    for (const auto &off: res.offenders) {
      const Block &src = fun.blocks[off.src];
      const Block &dst = fun.blocks[off.dst];
      SourceSpan termSpan = std::visit([](const auto &t) { return t.span; }, src.term);
      diags.error(
          "Irreducible control flow: branch from " + cfg.blocks[off.src] + " to " +
              cfg.blocks[off.dst] + " re-enters a loop whose header does not dominate " +
              cfg.blocks[off.src],
          termSpan
      );
      diags.note(
          cfg.blocks[off.dst] +
              " is reached both from above and by this retreating edge, so it is not a "
              "unique loop header",
          dst.label.span
      );
    }
    return res.reducible() ? PassResult::Success : PassResult::Error;
  }

  bool ReducibilityResult::isReducible(const FunDecl &fun) {
    // The DiagBag is throwaway: a caller reaching here has a program the
    // checkers already accepted, so the CFG builds.
    DiagBag diags;
    CFG cfg = CFG::build(fun, diags);
    return check(cfg, DomTree::build(cfg)).reducible();
  }

} // namespace refractir
