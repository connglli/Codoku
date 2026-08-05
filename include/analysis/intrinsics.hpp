#pragma once

#include <optional>
#include <string>
#include <vector>

namespace refractir {

  /**
   * Represents the kind of standard built-in intrinsic.
   * Standard built-in intrinsics (§12) require no user-supplied body or contracts.
   * Semantics are hardcoded inside the interpreter, solver, and codegen backends.
   *
   * v0.2.2 baseline (§12.1, §12.2): Abs, Min, Max, Popcount, Clz, Ctz.
   * v0.2.2 extra A/B (§12.3 integer extras, §12.4 bit-manipulation):
   *   AbsDiff, Signum, Clamp, Midpoint,
   *   Parity, Bswap, Bitreverse, Rotl, Rotr, IsPow2, Ilog2.
   * v0.2.2 extra C — integer overflow-aware family (§12.5):
   *   WrappingAdd, WrappingSub, WrappingMul, WrappingNeg,
   *   WrappingShl, WrappingShr,
   *   SaturatingAdd, SaturatingSub, SaturatingMul, SaturatingNeg,
   *   DivEuclid, RemEuclid.
   * v0.2.2 extra D.1 — floating-point sign / bit ops (§12.6):
   *   Fabs, Fneg, Copysign, Signbit, ToBits, FromBits.
   * v0.2.2 extra D.2 — floating-point classification predicates (§12.6):
   *   IsNormal, IsSubnormal.
   * v0.2.2 extra D.3 — floating-point min / max (§12.6): Fmin, Fmax.
   * v0.2.2 extra D.4 — correctly-rounded math (§12.6):
   *   Sqrt, Floor, Ceil, Trunc.
   * v0.2.2 extra D.5 — compositions (§12.6): Fract, Recip.
   *
   * Note: there is no @umin / @umax — RefractIR has no unsigned integer types
   * and the toolchain does not implicitly reinterpret iN bits as uN.
   * The tuple-returning members of the overflow family (`@checked_*`,
   * `@overflowing_*`) and the cross-width `@widening_mul` are deferred
   * to a follow-up batch that introduces the multi-value return ABI.
   */
  enum class IntrinsicKind {
    // v0.2.2 baseline
    Abs,
    Min,
    Max,
    Popcount,
    Clz,
    Ctz,
    // v0.2.2 extra batch A — integer extras (§12.3)
    AbsDiff,
    Signum,
    Clamp,
    Midpoint,
    // v0.2.2 extra batch B — bit-manipulation (§12.4)
    Parity,
    Bswap,
    Bitreverse,
    Rotl,
    Rotr,
    IsPow2,
    Ilog2,
    // v0.2.2 extra batch C — integer overflow-aware family (§12.5)
    WrappingAdd,
    WrappingSub,
    WrappingMul,
    WrappingNeg,
    WrappingShl,
    WrappingShr,
    SaturatingAdd,
    SaturatingSub,
    SaturatingMul,
    SaturatingNeg,
    DivEuclid,
    RemEuclid,
    // v0.2.2 extra batch D.1 — floating-point sign / bit ops (§12.6)
    Fabs,
    Fneg,
    Copysign,
    Signbit,
    ToBits,
    FromBits,
    // v0.2.2 extra batch D.2 — floating-point classification predicates (§12.6)
    IsNormal,
    IsSubnormal,
    // v0.2.2 extra batch D.3 — floating-point min / max (§12.6)
    Fmin,
    Fmax,
    // v0.2.2 extra batch D.4 — correctly-rounded math (§12.6)
    Sqrt,
    Floor,
    Ceil,
    Trunc,
    // v0.2.2 extra batch D.5 — compositions (§12.6)
    Fract,
    Recip,
    // Checksum primitives.
    //   @crc32_update(state: i32, val: iN) : i32
    //     Fold val into a running CRC32 state. Reflected polynomial
    //     0xEDB88320, no initial / final XOR, val's bytes processed
    //     LSB-first. Pure function.
    //   @check_chksum(expected: i32, actual: i32) : i32
    //     Return `actual` if it equals `expected`, otherwise abort with a
    //     diagnostic on stderr (UB in the interpreter).
    Crc32Update,
    CheckChksum,
    // v0.2.3 — observability beacon.
    //   @observe(v: T) : T   for an integer or floating-point T
    //     Value semantics is the identity (returns v unchanged), so it is a
    //     deterministic, solver-friendly, oracle-consistent pure function like
    //     every other intrinsic. Its purpose is a *lowering* property: the C
    //     backend emits a `volatile` write of v, an observable side effect the
    //     optimizer must preserve. This anchors a computation the compiler
    //     cannot prove dead — e.g. the body of a deliberately non-terminating
    //     loop. WASM / Python have no forward-progress assumption, so there it
    //     lowers to plain identity.
    Observe,
    // Horizontal vector reductions (§12.4). The sole parameter
    // is a vector `<N> T`; the result is the scalar element type `T`.
    // Add / Min / Max fold over integer or floating-point lanes; the
    // bitwise reductions And / Or / Xor are integer-only. `@reduce_mul` is
    // deliberately absent — a symbolic `N-1`-deep multiplication chain is
    // solver-hostile (§12.4, §13).
    ReduceAdd,
    ReduceMin,
    ReduceMax,
    ReduceAnd,
    ReduceOr,
    ReduceXor,
  };

  /**
   * The class of types the intrinsic's single type parameter `T` ranges
   * over. `T` is the `iN` / `fN` fixed at each declaration site.
   */
  enum class IntrinsicDomain { Int, Fp, IntOrFp };

  /**
   * One slot of a signature — the type of a parameter or of the return —
   * expressed relative to the type parameter `T`. Every intrinsic signature
   * is built compositionally from these; there is no per-intrinsic bucket.
   */
  enum class IntrinsicSigType {
    T,           // T itself                         (iN→iN, fN→fN homogeneous ops)
    VecOfT,      // <N> T                            (horizontal reductions)
    I1,          // i1, independent of T             (predicate results)
    IntWidthOfT, // the iN with N == bitwidth(T)     (@to_bits / @from_bits bridge)
    I32,         // fixed i32                        (checksum state / result)
  };

  /**
   * The one canonical descriptor for a built-in intrinsic (§12): its name,
   * the class of its type parameter, and the type form of each parameter and
   * the return. Every other fact — arity, i1-return, reduction shape,
   * float-admissibility — is a projection of this record (see the accessors
   * below). Adding an intrinsic means adding one row here (plus its per-tool
   * lowering); nothing else re-states the signature.
   */
  struct IntrinsicInfo {
    IntrinsicKind kind;
    const char *name;       // "@abs", "@reduce_add", …
    IntrinsicDomain domain; // the class T is drawn from
    IntrinsicSigType ret;   // return slot
    std::vector<IntrinsicSigType> params;
    bool widthMultipleOf8 = false; // @bswap: T's width must be a multiple of 8
  };

  inline const std::vector<IntrinsicInfo> &intrinsicTable() {
    using K = IntrinsicKind;
    using D = IntrinsicDomain;
    using S = IntrinsicSigType;
    static const std::vector<IntrinsicInfo> table = {
        // Homogeneous integer ops — T→T over an integer T (§12.1–12.5).
        {K::Abs, "@abs", D::Int, S::T, {S::T}},
        {K::Signum, "@signum", D::Int, S::T, {S::T}},
        {K::Popcount, "@popcount", D::Int, S::T, {S::T}},
        {K::Clz, "@clz", D::Int, S::T, {S::T}},
        {K::Ctz, "@ctz", D::Int, S::T, {S::T}},
        {K::Bitreverse, "@bitreverse", D::Int, S::T, {S::T}},
        {K::Ilog2, "@ilog2", D::Int, S::T, {S::T}},
        {K::WrappingNeg, "@wrapping_neg", D::Int, S::T, {S::T}},
        {K::SaturatingNeg, "@saturating_neg", D::Int, S::T, {S::T}},
        {K::Bswap, "@bswap", D::Int, S::T, {S::T}, /*widthMultipleOf8=*/true},
        {K::Min, "@min", D::Int, S::T, {S::T, S::T}},
        {K::Max, "@max", D::Int, S::T, {S::T, S::T}},
        {K::AbsDiff, "@abs_diff", D::Int, S::T, {S::T, S::T}},
        {K::Midpoint, "@midpoint", D::Int, S::T, {S::T, S::T}},
        {K::Rotl, "@rotl", D::Int, S::T, {S::T, S::T}},
        {K::Rotr, "@rotr", D::Int, S::T, {S::T, S::T}},
        {K::WrappingAdd, "@wrapping_add", D::Int, S::T, {S::T, S::T}},
        {K::WrappingSub, "@wrapping_sub", D::Int, S::T, {S::T, S::T}},
        {K::WrappingMul, "@wrapping_mul", D::Int, S::T, {S::T, S::T}},
        {K::WrappingShl, "@wrapping_shl", D::Int, S::T, {S::T, S::T}},
        {K::WrappingShr, "@wrapping_shr", D::Int, S::T, {S::T, S::T}},
        {K::SaturatingAdd, "@saturating_add", D::Int, S::T, {S::T, S::T}},
        {K::SaturatingSub, "@saturating_sub", D::Int, S::T, {S::T, S::T}},
        {K::SaturatingMul, "@saturating_mul", D::Int, S::T, {S::T, S::T}},
        {K::DivEuclid, "@div_euclid", D::Int, S::T, {S::T, S::T}},
        {K::RemEuclid, "@rem_euclid", D::Int, S::T, {S::T, S::T}},
        {K::Clamp, "@clamp", D::Int, S::T, {S::T, S::T, S::T}},
        // Integer predicates — T→i1 (§12.4).
        {K::Parity, "@parity", D::Int, S::I1, {S::T}},
        {K::IsPow2, "@is_pow2", D::Int, S::I1, {S::T}},
        // Homogeneous floating-point ops — T→T over a floating-point T (§12.6).
        {K::Fabs, "@fabs", D::Fp, S::T, {S::T}},
        {K::Fneg, "@fneg", D::Fp, S::T, {S::T}},
        {K::Sqrt, "@sqrt", D::Fp, S::T, {S::T}},
        {K::Floor, "@floor", D::Fp, S::T, {S::T}},
        {K::Ceil, "@ceil", D::Fp, S::T, {S::T}},
        {K::Trunc, "@trunc", D::Fp, S::T, {S::T}},
        {K::Fract, "@fract", D::Fp, S::T, {S::T}},
        {K::Recip, "@recip", D::Fp, S::T, {S::T}},
        {K::Copysign, "@copysign", D::Fp, S::T, {S::T, S::T}},
        {K::Fmin, "@fmin", D::Fp, S::T, {S::T, S::T}},
        {K::Fmax, "@fmax", D::Fp, S::T, {S::T, S::T}},
        // Floating-point predicates — T→i1 (§12.6).
        {K::Signbit, "@signbit", D::Fp, S::I1, {S::T}},
        {K::IsNormal, "@is_normal", D::Fp, S::I1, {S::T}},
        {K::IsSubnormal, "@is_subnormal", D::Fp, S::I1, {S::T}},
        // Bit-pattern bridge — width-matched fN ↔ iN (§12.6).
        {K::ToBits, "@to_bits", D::Fp, S::IntWidthOfT, {S::T}},
        {K::FromBits, "@from_bits", D::Fp, S::T, {S::IntWidthOfT}},
        // Checksum primitives — fixed i32 plumbing; the folded value is any iN.
        {K::Crc32Update, "@crc32_update", D::Int, S::I32, {S::I32, S::T}},
        {K::CheckChksum, "@check_chksum", D::Int, S::I32, {S::I32, S::I32}},
        // Observability beacon — identity value; observable write in the C
        // lowering. Any integer width.
        {K::Observe, "@observe", D::IntOrFp, S::T, {S::T}},
        // Horizontal reductions — <N> T → T (§12.4). add/min/max fold integer
        // or FP lanes; the bitwise reductions are integer-only.
        {K::ReduceAdd, "@reduce_add", D::IntOrFp, S::T, {S::VecOfT}},
        {K::ReduceMin, "@reduce_min", D::IntOrFp, S::T, {S::VecOfT}},
        {K::ReduceMax, "@reduce_max", D::IntOrFp, S::T, {S::VecOfT}},
        {K::ReduceAnd, "@reduce_and", D::Int, S::T, {S::VecOfT}},
        {K::ReduceOr, "@reduce_or", D::Int, S::T, {S::VecOfT}},
        {K::ReduceXor, "@reduce_xor", D::Int, S::T, {S::VecOfT}},
    };
    return table;
  }

  /** The canonical descriptor for `kind` (never null — the table is total). */
  inline const IntrinsicInfo &intrinsicInfo(IntrinsicKind kind) {
    for (const auto &info: intrinsicTable())
      if (info.kind == kind)
        return info;
    return intrinsicTable().front(); // unreachable: the table covers every kind
  }

  /** The canonical source name of `kind`, e.g. "@abs". */
  inline const char *intrinsicName(IntrinsicKind kind) { return intrinsicInfo(kind).name; }

  /**
   * Resolves a global identifier string (such as "@abs") to its IntrinsicKind,
   * or std::nullopt if it names no built-in intrinsic.
   */
  inline std::optional<IntrinsicKind> getIntrinsicKind(const std::string &name) {
    for (const auto &info: intrinsicTable())
      if (name == info.name)
        return info.kind;
    return std::nullopt;
  }

  /** Parameter count of `kind`. */
  inline std::size_t intrinsicArity(IntrinsicKind kind) {
    return intrinsicInfo(kind).params.size();
  }

  /** True iff `kind`'s result is the `i1` predicate type, independent of `T`. */
  inline bool intrinsicReturnsI1(IntrinsicKind kind) {
    return intrinsicInfo(kind).ret == IntrinsicSigType::I1;
  }

  /**
   * True iff `kind` is a horizontal vector reduction (§12.4) — the only
   * family whose parameter is a vector `<N> T` rather than a scalar. Callers
   * that special-case the vector-in / scalar-out shape (frontend WF,
   * interpreter fold, solver lane-fold, backend lowering) branch on this.
   */
  inline bool isReductionIntrinsic(IntrinsicKind kind) {
    for (IntrinsicSigType p: intrinsicInfo(kind).params)
      if (p == IntrinsicSigType::VecOfT)
        return true;
    return false;
  }

  /**
   * True iff the intrinsic admits a floating-point element type — i.e. its
   * type parameter `T` may be an `fN`. For the reductions this distinguishes
   * @reduce_add / _min / _max (integer or FP) from the integer-only bitwise
   * reductions (§12.4).
   */
  inline bool intrinsicAllowsFloat(IntrinsicKind kind) {
    return intrinsicInfo(kind).domain != IntrinsicDomain::Int;
  }

} // namespace refractir
