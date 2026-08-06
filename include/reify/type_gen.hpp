#pragma once

#include <cassert>
#include <cstdint>
#include <random>
#include <utility>
#include "analysis/type_utils.hpp"
#include "ast/ast.hpp"

namespace refractir::reify {

  struct TypeGenConfig {
    bool enableFp = true;     // --no-fp disables f32/f64
    bool enableVec = true;    // --no-vec disables <N> T generation
    bool enableAggPtr = true; // --no-agg-ptr disables ptr [N] T / ptr @S
    int maxPtrDepth = 2;      // max ptr nesting
    int maxAggNesting = 2;    // max array/struct nesting
    int maxAggElems = 3;      // max array size and struct field count
  };

  // Generate a random type. depth=0 is the top-level call.
  [[nodiscard]] TypePtr genRandomType(std::mt19937 &rng, const TypeGenConfig &cfg, int depth = 0);

  // Generate a random scalar type only (integer or float based on cfg.enableFp)
  [[nodiscard]] TypePtr genScalarType(std::mt19937 &rng, bool enableFp);

  // Generate a random vector type <N> T where N ∈ {2, 4, 8}.
  [[nodiscard]] TypePtr genVecType(std::mt19937 &rng, bool enableFp);

  // Generate a random integer scalar type (i8, i16, i32, i64)
  [[nodiscard]] TypePtr genIntType(std::mt19937 &rng);

  // Width of an integer type. Unlike TypeUtils::getIntBitWidth, which reports
  // "not an integer" as an empty optional, this asserts the caller has already
  // established int-ness — which every generator site has, since it picked the
  // type it is asking about.
  [[nodiscard]] std::uint32_t intBitWidth(const TypePtr &t);

  // The [lo, hi] range of int64 values that fit a signed `iN`, which is not
  // the range of `iN` itself: an `IntLit` is an int64, so a width of 64 or
  // more clamps to the int64 range rather than naming the type's true bounds.
  // Callers use it to keep a generated literal inside its declared type, and
  // that clamp is sound for the purpose -- every int64 fits a wider `iN`.
  //
  // Requires `bits >= 1`. Width 0 is not a RefractIR type; it reaches here
  // only from a caller that read a bitwidth off a non-integer type, and the
  // range it would name is empty, so the degenerate {0, 0} keeps every
  // downstream distribution well-formed instead of shifting by a negative
  // amount.
  [[nodiscard]] inline std::pair<std::int64_t, std::int64_t>
  signedIntRange(std::uint32_t bits) noexcept {
    assert(bits >= 1 && "signedIntRange: width 0 is not a RefractIR integer type");
    if (bits == 0)
      return {0, 0};
    if (bits >= 64)
      return {INT64_MIN, INT64_MAX};
    std::int64_t hi = (std::int64_t{1} << (bits - 1)) - 1;
    return {-hi - 1, hi};
  }

} // namespace refractir::reify
