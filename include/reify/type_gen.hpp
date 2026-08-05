#pragma once

#include <cstdint>
#include <random>
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

  // Type predicate helpers
  [[nodiscard]] bool isIntType(const TypePtr &t);
  [[nodiscard]] bool isFpType(const TypePtr &t);
  [[nodiscard]] bool isPtrType(const TypePtr &t);
  [[nodiscard]] bool isAggType(const TypePtr &t);    // array or struct
  [[nodiscard]] bool isScalarType(const TypePtr &t); // int or fp (not ptr, not agg)
  [[nodiscard]] bool isVecType(const TypePtr &t);    // <N> T

  // Get the bitwidth of an integer type (8, 16, 32, 64, or custom bits)
  [[nodiscard]] std::uint32_t intBitWidth(const TypePtr &t);

  // Get the pointee type of a ptr type
  [[nodiscard]] TypePtr pointeeType(const TypePtr &t);

  // Two types are "assignment compatible" (same kind and width/structure)
  [[nodiscard]] bool typeEquals(const TypePtr &a, const TypePtr &b);

} // namespace refractir::reify
