#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include "reify/cfg_gen.hpp"

namespace refractir::reify {

  struct SamplePathParams {
    std::uint32_t seed = 0;
    int maxLoopIter = 1;
    // If > 0, the sampler guarantees that at least one back edge in the
    // returned path is traversed at least this many times. Returns nullopt
    // if the CFG has no back edges (so the caller can retry with a new CFG).
    int minLoopIter = 0;
    int maxPathLen = 50;
  };

  [[nodiscard]] std::optional<std::vector<std::string>>
  samplePath(const RyCFG &cfg, const SamplePathParams &params);

  struct SampleLassoParams {
    std::uint32_t seed = 0;
    int maxPathLen = 50;
    // Number of laps of the cycle to emit (the orbit's period k). 1 is the
    // classic single-lap fixed point; k > 1 asks the solver for an orbit that
    // returns to the header state only after k laps.
    int period = 1;
  };

  // Sample a *lasso* path for non-terminating generation:
  //
  // entry -> ... -> h -> ... -> src -> h
  // \___ stem ρ ___/\____ cycle γ ____/
  //
  // where (src -> h) is a back edge and `h` a genuine loop header (its
  // target dominates its source in a reducible CFG — the mode gates on
  // --require-reducible so every remaining back edge qualifies). The final
  // label is the header `h`; the block just before it is the latch `src`.
  // The infinite path ρ·γ^ω is represented by this finite ρ·γ witness — the
  // solver's RequireNonterm mode certifies the ω via header-state recurrence.
  // Returns nullopt if the CFG has no usable back edge or the stem/cycle
  // cannot be connected (caller retries with a fresh CFG).
  [[nodiscard]] std::optional<std::vector<std::string>>
  sampleLasso(const RyCFG &cfg, const SampleLassoParams &params);

} // namespace refractir::reify
