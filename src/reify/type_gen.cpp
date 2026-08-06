#include "reify/type_gen.hpp"
#include "analysis/type_utils.hpp"

#include <cassert>
#include <stdexcept>
#include "reify/hyperparameters.hpp"

namespace refractir::reify {

  // ---------------------------------------------------------------------------
  // Factory helpers
  // ---------------------------------------------------------------------------

  static TypePtr makeIntTypeOfBits(uint32_t bits) {
    IntType t;
    switch (bits) {
      case 8:
        t.kind = IntType::Kind::ICustom;
        t.bits = 8;
        break;
      case 16:
        t.kind = IntType::Kind::ICustom;
        t.bits = 16;
        break;
      case 32:
        t.kind = IntType::Kind::I32;
        break;
      case 64:
        t.kind = IntType::Kind::I64;
        break;
      default:
        t.kind = IntType::Kind::ICustom;
        t.bits = (int) bits;
        break;
    }
    return std::make_shared<Type>(Type{t, {}});
  }

  // ---------------------------------------------------------------------------
  // Public helpers
  // ---------------------------------------------------------------------------

  uint32_t intBitWidth(const TypePtr &t) {
    auto bits = TypeUtils::getIntBitWidth(t);
    assert(bits && "intBitWidth: not an integer type");
    return bits.value_or(32);
  }

  // ---------------------------------------------------------------------------
  // Generators
  // ---------------------------------------------------------------------------

  // Draw an integer bitwidth: one of the standard 8/16/32/64, or — with
  // probability kPOddIntWidth — a custom width from kOddIntWidths to
  // exercise the backends' widen-and-mask paths.
  static uint32_t pickIntBits(std::mt19937 &rng, int standardPick) {
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    if (coin(rng) < rysmith::hp::kPOddIntWidth) {
      std::uniform_int_distribution<std::size_t> d(0, rysmith::hp::kOddIntWidthsSize - 1);
      return rysmith::hp::kOddIntWidths[d(rng)];
    }
    static const uint32_t widths[] = {8, 16, 32, 64};
    return widths[standardPick];
  }

  TypePtr genIntType(std::mt19937 &rng) {
    std::uniform_int_distribution<int> d(0, 3);
    return makeIntTypeOfBits(pickIntBits(rng, d(rng)));
  }

  TypePtr genScalarType(std::mt19937 &rng, bool enableFp) {
    // Integer types: i8, i16, i32, i64 — equal probability (each 1 slot)
    // Float types (f32, f64 combined) get the same probability as ONE integer type
    // Total slots: 5 if fp enabled (4 int + 1 fp bucket), 4 if not
    std::uniform_int_distribution<int> d(0, enableFp ? 4 : 3);
    int pick = d(rng);
    if (pick < 4) {
      return makeIntTypeOfBits(pickIntBits(rng, pick));
    }
    // Float bucket: pick f32 or f64
    std::uniform_int_distribution<int> fpick(0, 1);
    FloatType ft;
    ft.kind = fpick(rng) ? FloatType::Kind::F64 : FloatType::Kind::F32;
    return std::make_shared<Type>(Type{ft, {}});
  }

  TypePtr genVecType(std::mt19937 &rng, bool enableFp) {
    static const uint64_t lanes[] = {2, 4, 8};
    std::uniform_int_distribution<int> ld(0, 2);
    uint64_t N = lanes[ld(rng)];
    TypePtr elem = genScalarType(rng, enableFp);
    VecType vt;
    vt.size = N;
    vt.elem = elem;
    return std::make_shared<Type>(Type{vt, {}});
  }

  TypePtr genRandomType(std::mt19937 &rng, const TypeGenConfig &cfg, int depth) {
    // Type-kind probability buckets — see rysmith::hp::kPType*. Aggregates are zeroed
    // past maxAggNesting and pointers past maxPtrDepth, then renormalized.
    double pScalar = rysmith::hp::kPTypeScalar;
    double pArray = (depth >= cfg.maxAggNesting) ? 0.0 : rysmith::hp::kPTypeArray;
    double pStruct = (depth >= cfg.maxAggNesting) ? 0.0 : rysmith::hp::kPTypeStruct;
    double pPtr = (depth >= cfg.maxPtrDepth) ? 0.0 : rysmith::hp::kPTypePtr;
    // Vectors only at depth 0 (no nested vec, no vec in arrays/structs).
    double pVec = (depth == 0 && cfg.enableVec) ? rysmith::hp::kPTypeVec : 0.0;

    // Renormalize
    double total = pScalar + pArray + pStruct + pPtr + pVec;
    if (total <= 0.0) {
      return genScalarType(rng, cfg.enableFp);
    }

    std::uniform_real_distribution<double> prob(0.0, total);
    double r = prob(rng);

    if (r < pScalar) {
      return genScalarType(rng, cfg.enableFp);
    }
    r -= pScalar;
    if (r < pArray) {
      // Array: [N] elem — elem is recursively generated with depth+1
      std::uniform_int_distribution<int> szd(1, std::max(1, cfg.maxAggElems));
      uint64_t sz = (uint64_t) szd(rng);
      TypePtr elem = genRandomType(rng, cfg, depth + 1);
      ArrayType at;
      at.size = sz;
      at.elem = elem;
      return std::make_shared<Type>(Type{at, {}});
    }
    r -= pArray;
    if (r < pStruct) {
      StructType st;
      st.name = GlobalId{"@_pending_struct", {}};
      return std::make_shared<Type>(Type{st, {}});
    }
    r -= pStruct;
    if (r < pVec) {
      return genVecType(rng, cfg.enableFp);
    }
    r -= pVec;
    // Pointer
    TypePtr pointee = genRandomType(rng, cfg, depth + 1);
    // If enableAggPtr, allow aggregate pointees (ptr [N] T, ptr @S).
    // Otherwise fall back to scalar.
    // Pointer to vector (ptr <N> T) is always forbidden (§6.8.1).
    if (TypeUtils::isVec(pointee)) {
      pointee = genScalarType(rng, cfg.enableFp);
    }
    if (!cfg.enableAggPtr && TypeUtils::isAggregate(pointee)) {
      pointee = genScalarType(rng, cfg.enableFp);
    }
    PtrType pt;
    pt.pointee = pointee;
    return std::make_shared<Type>(Type{pt, {}});
  }

} // namespace refractir::reify
