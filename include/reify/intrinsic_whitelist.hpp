#pragma once

#include <compare>
#include <cstdint>
#include <set>
#include <vector>
#include "analysis/intrinsics.hpp"
#include "analysis/type_utils.hpp"
#include "ast/ast.hpp"
#include "frontend/semchecker.hpp"

namespace refractir::reify {

  // Key for tracking which intrinsic instantiations have been used, so
  // genFunction can emit one IntrinsicDecl per distinct signature. Scalar
  // intrinsics vary only by element width; the horizontal-reduction family
  // (§12.4) is additionally parameterised by lane count and element domain,
  // because `@reduce_add(<4> i32) : i32` and `@reduce_add(<8> f64) : f64`
  // are distinct overloads that each need their own declaration.
  struct IntrinsicUseKey {
    IntrinsicKind kind;
    std::uint32_t elemBits; // scalar element width (32 for i32/f32, 64 for i64/f64)
    bool elemIsFloat;       // element domain: false = iN, true = fN
    std::uint32_t lanes;    // 0 = scalar intrinsic; N = reduce over <N> T

    auto operator<=>(const IntrinsicUseKey &) const = default;
  };

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

  // Declare `name` in `prog.intrinsics` unless it is already declared.
  // Idempotent, so a caller may re-declare on every emission without checking.
  //
  // Identity is intrinsicSignature's: the name and the parameter types. That
  // is what the semantic checker rejects a program for repeating, so matching
  // any more loosely would drop a legitimate overload and any more strictly
  // would stage a program it rejects.
  inline void ensureIntrinsicDecl(
      Program &prog, const std::string &name, TypePtr retType,
      const std::vector<std::pair<std::string, TypePtr>> &params
  ) {
    IntrinsicDecl decl;
    decl.name = GlobalId{name, {}};
    decl.retType = std::move(retType);
    for (const auto &[pName, pType]: params) {
      ParamDecl pd;
      pd.name = LocalId{pName, {}};
      pd.type = pType;
      decl.params.push_back(std::move(pd));
    }
    const std::string sig = intrinsicSignature(decl);
    for (const auto &existing: prog.intrinsics)
      if (intrinsicSignature(existing) == sig)
        return;
    prog.intrinsics.push_back(std::move(decl));
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
