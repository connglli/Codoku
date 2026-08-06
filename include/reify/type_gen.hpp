#pragma once

#include <cstdint>
#include <random>
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

} // namespace refractir::reify
