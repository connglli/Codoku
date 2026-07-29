#pragma once

// twin_gen — solver-generated twin blocks for rytwin.
//
// Given a block's entry state `s` and exit state `s'` (as concrete value
// trees per root), generateTwin synthesizes an instruction sequence whose
// net effect from `s` is exactly `s'` — the rysmith way: random statements
// with `%?` symbols (UB-safety requires spliced automatically), a fresh
// additive correction symbol per touched leaf so the target is always
// reachable, and one equality `require` per leaf pinning the final state
// to `s'`. The mini-program is solved in-process with the SMT solver,
// concretized by printing with the model and re-parsing, then cross-checked
// bit-exactly by running the interpreter (IEEE `==` conflates ±0.0; the
// bit-exact re-run does not). The equality requires are scaffolding and are
// stripped from the returned instructions.
//
// The TwinTransform consumes this through the `TwinGenFn` callback, injected by
// the rytwin driver: the pass library stays free of a solver link
// dependency (rylink links reify without the solver backend).

#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "ast/ast.hpp"
#include "reify/state_profile.hpp"
#include "reify/twin_mini.hpp"
#include "solver/solver.hpp"

namespace refractir::reify {

  // The roots a twin must model (MiniRoot / MiniPtrFix) and the conversions
  // from a captured state back into declarations and instructions live in
  // reify/twin_mini.hpp — they are shared with every other consumer that
  // materializes a region's live-in state, and stay solver-free.

  struct TwinGenConfig {
    int nStmts = 3;             // random statements per attempt
    int retries = 3;            // generation attempts before giving up
    uint32_t timeoutMs = 10000; // per-attempt SMT timeout
  };

  // A verified twin body plus the intrinsic declarations its instructions
  // call — the grafting pass merges those into the host program.
  struct TwinGenResult {
    std::vector<Instr> instrs;
    std::vector<IntrinsicDecl> intrinsics;
  };

  // Generator callback used by TwinTransform. Returns the concrete twin-block
  // body, or nullopt when no attempt verified (the pass then falls back to
  // constant reconstruction).
  using TwinGenFn = std::function<
      std::optional<TwinGenResult>(const Program &, const std::vector<MiniRoot> &, std::mt19937 &)>;

  std::optional<TwinGenResult> generateTwin(
      const Program &prog, const std::vector<MiniRoot> &roots, std::mt19937 &rng,
      const SymbolicExecutor::SolverFactory &solverFactory, const TwinGenConfig &cfg
  );

} // namespace refractir::reify
