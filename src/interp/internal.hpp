#pragma once

// Interpreter-private helpers shared across the interpreter's translation
// units. Kept inline in this detail header (not a separate .cpp) so each
// interp TU that needs them can pull them in without an extra object file.
// Not part of the public interpreter interface — internal to src/interp.

#include <cmath>
#include <cstdint>
#include "error.hpp"

namespace refractir {

  // Enforce IEEE 754 finite-only semantics (spec §7.4 rules 6–7):
  // truncate to f32 if needed, then reject infinity or NaN.
  inline double checkFPResult(double val, std::uint32_t bits) {
    if (bits == 32)
      val = static_cast<double>(static_cast<float>(val));
    if (std::isinf(val))
      throw UndefinedBehaviorError("UB: Floating-point result is infinity");
    if (std::isnan(val))
      throw UndefinedBehaviorError("UB: Floating-point result is NaN");
    return val;
  }

  // Sign-canonicalize a 64-bit value to its declared N-bit signed width.
  inline std::int64_t canonicalize(std::int64_t val, std::uint32_t bits) {
    if (bits >= 64)
      return val;
    // Spec §6.4: i1 is a signed 1-bit integer.  The two
    // representable values are 0 (false) and -1 (true) — bit
    // pattern 1 sign-extended.  `iN as iM` widening sign-extends, so
    // an i1 true widened to i32 is -1; matches what the C backend
    // already emits and what the spec mandates.
    if (bits == 1)
      return (val & 1) ? -1 : 0;
    std::uint64_t mask = (1ULL << bits) - 1;
    std::uint64_t sign_bit = 1ULL << (bits - 1);
    std::uint64_t uval = static_cast<std::uint64_t>(val) & mask;
    if (uval & sign_bit)
      uval |= ~mask;
    return static_cast<std::int64_t>(uval);
  }

} // namespace refractir
