#pragma once

// The anti-optimization catalog as a whole-program Transform.
//
// reify/antiopt.hpp is the engine: a statement list, the declarations it may
// add to, and a predicate deciding whether a rewritten body is still
// acceptable. rytwin has such a predicate — the interval pass over its guard
// box — and gets the whole catalog. A generator does not: its programs are
// concrete, there is no set of states to prove anything over, and nothing can
// judge a rewrite after the fact.
//
// That is not a reason to leave generated code alone. What a proof buys is
// permission to introduce operations that can trap; the trap-free rules are
// identities whatever the state, and so is any composition of them. So this
// transform passes no predicate and the engine offers only those — which is
// enough to stop every function in a bundle reading like the one statement
// generator that wrote it.

#include <memory>

#include "reify/transform.hpp"

namespace refractir::reify {

  // Rewrite every function of the program by the trap-free catalog. Draws come
  // from the pipeline's rng, so a seed reproduces a bundle exactly.
  std::unique_ptr<Transform> makeAntiOptTransform();

} // namespace refractir::reify
