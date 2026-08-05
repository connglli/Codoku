#pragma once

#include <set>
#include <vector>
#include "analysis/intrinsics.hpp"
#include "ast/ast.hpp"

namespace refractir::reify {

  /**
   * The curated set of intrinsic kinds rysmith is allowed to generate as
   * ordinary body calls. This is generation *policy* (which solver-friendly
   * intrinsics we want in random programs), not an intrinsic fact — every
   * signature detail (name, arity, parameter/return shape, i1-return,
   * float-admissibility) comes from the canonical table in
   * analysis/intrinsics.hpp. Excluded here: the scalar FP family and the
   * checksum primitives (the latter are emitted by reify/checksum.cpp
   * through their own path).
   *
   * Every entry is solver-friendly: it decomposes into quantifier-free
   * BV / QF_FP operations with bounded O(N) ITE chains.
   */
  inline const std::vector<IntrinsicKind> &getGeneratableIntrinsics() {
    using K = IntrinsicKind;
    static const std::vector<IntrinsicKind> list = {
        // Baseline + integer extras + bit-manipulation (§12.1–12.4).
        K::Abs,
        K::Min,
        K::Max,
        K::Popcount,
        K::Clz,
        K::Ctz,
        K::AbsDiff,
        K::Signum,
        K::Clamp,
        K::Midpoint,
        K::Parity,
        K::Bswap,
        K::Bitreverse,
        K::Rotl,
        K::Rotr,
        K::IsPow2,
        K::Ilog2,
        // Integer overflow-aware family (§12.5).
        K::WrappingAdd,
        K::WrappingSub,
        K::WrappingMul,
        K::WrappingNeg,
        K::WrappingShl,
        K::WrappingShr,
        K::SaturatingAdd,
        K::SaturatingSub,
        K::SaturatingMul,
        K::SaturatingNeg,
        K::DivEuclid,
        K::RemEuclid,
        // Horizontal reductions (§12.4).
        K::ReduceAdd,
        K::ReduceMin,
        K::ReduceMax,
        K::ReduceAnd,
        K::ReduceOr,
        K::ReduceXor,
    };
    return list;
  }

  // Append one IntrinsicDecl per used instantiation. This is the single
  // place a whitelist entry becomes a declaration; func_gen
  // both emit their `intrinsic` sections through it. The declaration is
  // reconstructed entirely from the canonical signature: each IntrinsicSigType slot
  // expands to a concrete type given the used element type and lane count.
  inline void
  appendUsedIntrinsicDecls(const std::set<IntrinsicUseKey> &used, std::vector<IntrinsicDecl> &out) {
    auto makeInt = [](uint32_t bits) -> TypePtr {
      if (bits == 32)
        return std::make_shared<Type>(Type{IntType{IntType::Kind::I32, {}, {}}, {}});
      if (bits == 64)
        return std::make_shared<Type>(Type{IntType{IntType::Kind::I64, {}, {}}, {}});
      return std::make_shared<Type>(Type{IntType{IntType::Kind::ICustom, (int) bits, {}}, {}});
    };
    auto makeScalar = [&](uint32_t bits, bool isFloat) -> TypePtr {
      if (isFloat)
        return std::make_shared<Type>(
            Type{FloatType{bits == 32 ? FloatType::Kind::F32 : FloatType::Kind::F64, {}}, {}}
        );
      return makeInt(bits);
    };
    // Expand one signature slot to a concrete type, given the element type
    // `T` (its scalar type + width) and the vector lane count.
    auto slotType = [&](IntrinsicSigType s, const TypePtr &scalarTy, uint32_t elemBits,
                        uint32_t lanes) -> TypePtr {
      switch (s) {
        case IntrinsicSigType::T:
          return scalarTy;
        case IntrinsicSigType::VecOfT:
          return std::make_shared<Type>(Type{VecType{lanes, scalarTy, {}}, {}});
        case IntrinsicSigType::I1:
          return makeInt(1);
        case IntrinsicSigType::IntWidthOfT:
          return makeInt(elemBits);
        case IntrinsicSigType::I32:
          return makeInt(32);
      }
      return scalarTy; // unreachable
    };
    for (const auto &key: used) {
      const IntrinsicInfo &info = intrinsicInfo(key.kind);
      TypePtr scalarTy = makeScalar(key.elemBits, key.elemIsFloat);
      IntrinsicDecl id;
      id.name = GlobalId{std::string(info.name), {}};
      id.retType = slotType(info.ret, scalarTy, key.elemBits, key.lanes);
      for (std::size_t pi = 0; pi < info.params.size(); pi++) {
        ParamDecl pd;
        pd.name = LocalId{"%x" + std::to_string(pi), {}};
        pd.type = slotType(info.params[pi], scalarTy, key.elemBits, key.lanes);
        id.params.push_back(std::move(pd));
      }
      out.push_back(std::move(id));
    }
  }

} // namespace refractir::reify
