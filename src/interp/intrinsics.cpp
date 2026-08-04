// [v0.2.2] Interpreter-side built-in intrinsic dispatch.
//
// This file is the single source of truth for every intrinsic
// supported by the RefractIR interpreter. To add a new intrinsic:
//   1. Add its IntrinsicKind to include/analysis/intrinsics.hpp.
//   2. Implement a subclass of InterpreterIntrinsic and register it below.
//
// Width conventions:
//   - `N` is the return-type bit-width (read from intr.retType).
//   - For predicate intrinsics (@parity, @is_pow2) the return width is
//     fixed at 1 (i1) while the input width is iN. The eval API exposes
//     `intr` so impls can read per-parameter widths from the declaration.
//   - `args[i].intVal` is already sign-extended to int64 from its
//     declared width by the caller, so reading raw arg bits requires
//     re-masking against the parameter's own width — not the return's.

#include "interp/interpreter.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "analysis/intrinsics.hpp"
#include "analysis/type_utils.hpp"
#include "error.hpp"
#include "internal.hpp"

namespace refractir {

  namespace {

    /**
     * @brief Signed-bit mask for an N-bit slice of a 64-bit value.
     */
    inline int64_t maskOfN(uint32_t N) { return (N >= 64) ? -1LL : ((INT64_C(1) << N) - 1); }

    /**
     * @brief Minimum signed value representable in N bits.
     */
    inline int64_t intMinN(uint32_t N) { return (N >= 64) ? INT64_MIN : -(INT64_C(1) << (N - 1)); }

    /**
     * @brief Maximum signed value representable in N bits.
     */
    inline int64_t intMaxN(uint32_t N) {
      return (N >= 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
    }

    /**
     * @brief Sign-extend a 64-bit value as if its low N bits encode an
     * N-bit signed integer.
     */
    inline int64_t sextToInt64(int64_t v, uint32_t N) {
      if (N >= 64)
        return v;
      int64_t m = maskOfN(N);
      v &= m;
      if (v & (INT64_C(1) << (N - 1)))
        v |= ~m;
      return v;
    }

    /**
     * @brief Unsigned interpretation of an N-bit slice of a 64-bit value.
     */
    inline uint64_t uintOfN(int64_t v, uint32_t N) {
      return static_cast<uint64_t>(v) & static_cast<uint64_t>(maskOfN(N));
    }

    /**
     * @brief Read the i-th argument as a 64-bit signed value, sign-extended
     * from the *parameter's own width* (NOT the return width).
     */
    inline int64_t
    argSint(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args, size_t i) {
      if (i >= args.size())
        throw std::runtime_error("Intrinsic " + intr.name.name + ": argument count error");
      if (args[i].kind != RuntimeValue::Kind::Int)
        throw std::runtime_error("Intrinsic " + intr.name.name + ": non-integer argument");
      auto pb = TypeUtils::getIntBitWidth(intr.params[i].type);
      uint32_t pN = pb ? *pb : args[i].bits;
      // [v0.2.2] All iN — including i1 — are signed two's-complement.
      // Sign-extend bit 0 of an i1 so true reads back as -1, matching
      // the storage convention enforced by makeInt and `iN as iM`.
      return sextToInt64(args[i].intVal, pN);
    }

    /**
     * @brief Read the i-th argument as a 64-bit unsigned value, masked to
     * the parameter's own width.
     */
    inline uint64_t
    argUint(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args, size_t i) {
      auto pb = TypeUtils::getIntBitWidth(intr.params[i].type);
      uint32_t pN = pb ? *pb : args[i].bits;
      return uintOfN(argSint(intr, args, i), pN);
    }

    /**
     * @brief Construct an integer-kind RuntimeValue with the given N-bit value.
     * For N >= 2, the value is sign-extended to int64 (the iN storage
     * convention). For N == 1 (i1 / predicate), the value is stored as
     * literal 0 or 1, matching the `cmp` instruction's convention in the
     * interpreter (see src/interp/interpreter.cpp:1609).
     */
    inline RuntimeValue makeInt(uint32_t N, int64_t v) {
      RuntimeValue r;
      r.kind = RuntimeValue::Kind::Int;
      r.bits = N;
      // [v0.2.2] Spec §6.4: i1 is signed.  Sign-extension of the low
      // bit is 0 → 0 and 1 → -1, matching `iN as iM` widening
      // semantics and the C/WASM backends' i1 sign-extend.
      r.intVal = sextToInt64(v, N);
      return r;
    }

    // ── v0.2.2 extra batch D.1 — floating-point helpers ─────────────────

    /**
     * @brief Width (32 or 64) of an FP parameter or return type.
     */
    inline uint32_t fpBitsOf(const TypePtr &t) {
      auto fp = std::get_if<FloatType>(&t->v);
      return (fp && fp->kind == FloatType::Kind::F32) ? 32 : 64;
    }

    /**
     * @brief Read the i-th argument as a double; for f32 parameters, the
     * value held in `floatVal` is already at f32 precision (see the
     * argument-binding path in interpreter.cpp).
     */
    inline double
    argFloat(const IntrinsicDecl & /*intr*/, const std::vector<RuntimeValue> &args, size_t i) {
      return args[i].floatVal;
    }

    /**
     * @brief Construct a float-kind RuntimeValue. For N == 32 the value is
     * narrowed to f32 precision so subsequent reads agree with the C
     * backend's `float`-typed storage.
     */
    inline RuntimeValue makeFloat(uint32_t N, double v) {
      RuntimeValue r;
      r.kind = RuntimeValue::Kind::Float;
      r.bits = N;
      r.floatVal = (N == 32) ? static_cast<double>(static_cast<float>(v)) : v;
      return r;
    }

    /**
     * @brief Abstract base class for concrete evaluation of intrinsics.
     */
    class InterpreterIntrinsic {
    public:
      virtual ~InterpreterIntrinsic() = default;

      /**
       * @brief Concrete evaluation of the intrinsic.
       * @param intr Intrinsic declaration (for per-parameter width queries).
       * @param args Arguments, in declaration order.
       * @return The resulting RuntimeValue.
       */
      virtual RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const = 0;
    };

    // ── §12.1 Arithmetic intrinsics ─────────────────────────────────────────

    class AbsIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t x = argSint(intr, args, 0);
        if (x == intMinN(N))
          throw UndefinedBehaviorError("UB: @abs result not representable (-INT_MIN_N overflow)");
        return makeInt(N, x < 0 ? -x : x);
      }
    };

    class MinIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t a = argSint(intr, args, 0), b = argSint(intr, args, 1);
        return makeInt(N, a < b ? a : b);
      }
    };

    class MaxIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t a = argSint(intr, args, 0), b = argSint(intr, args, 1);
        return makeInt(N, a > b ? a : b);
      }
    };

    // ── §12.2 Bit-counting intrinsics ───────────────────────────────────────

    class PopcountIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t u = argUint(intr, args, 0);
        int64_t c = __builtin_popcountll(u);
        if (c > intMaxN(N))
          throw UndefinedBehaviorError("UB: @popcount result not representable in declared iN");
        return makeInt(N, c);
      }
    };

    class ClzIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t u = argUint(intr, args, 0);
        if (u == 0)
          throw UndefinedBehaviorError("UB: @clz requires non-zero input (§12.2)");
        int64_t c = 0;
        for (int b = (int) N - 1; b >= 0; --b) {
          if ((u >> b) & 1ULL)
            break;
          ++c;
        }
        return makeInt(N, c);
      }
    };

    class CtzIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t u = argUint(intr, args, 0);
        if (u == 0)
          throw UndefinedBehaviorError("UB: @ctz requires non-zero input (§12.2)");
        int64_t c = 0;
        for (uint32_t b = 0; b < N; ++b) {
          if ((u >> b) & 1ULL)
            break;
          ++c;
        }
        return makeInt(N, c);
      }
    };

    // ── §12.3 Integer extras (v0.2.2 extra batch A) ────────────────────────────────

    /**
     * @brief @abs_diff(a, b) = |a - b|, UB if the result is not representable in iN.
     * Computed via signed widening to int64 (or __int128 for N == 64) — no
     * reinterpretation as unsigned values is performed.
     */
    class AbsDiffIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t a = argSint(intr, args, 0), b = argSint(intr, args, 1);
        int64_t maxN = intMaxN(N);
        if (N < 64) {
          int64_t s = a - b;
          int64_t r = s < 0 ? -s : s;
          if (r > maxN)
            throw UndefinedBehaviorError(
                "UB: @abs_diff result not representable in declared iN (|a-b| > INT_MAX_N)"
            );
          return makeInt(N, r);
        }
#if defined(__SIZEOF_INT128__)
        __int128 s = static_cast<__int128>(a) - static_cast<__int128>(b);
        __int128 r = s < 0 ? -s : s;
        if (r > static_cast<__int128>(INT64_MAX))
          throw UndefinedBehaviorError("UB: @abs_diff result not representable in declared i64");
        return makeInt(N, static_cast<int64_t>(r));
#else
        // Portable fallback for hosts lacking __int128. Detect overflow by
        // sign analysis: a - b overflows when a and b have different signs
        // and the result has a sign opposite to a.
        bool diffSigns = ((a < 0) != (b < 0));
        int64_t s = a - b;
        if (diffSigns && ((a < 0) != (s < 0)))
          throw UndefinedBehaviorError("UB: @abs_diff result not representable in declared i64");
        int64_t r = s < 0 ? -s : s;
        if (s == INT64_MIN)
          throw UndefinedBehaviorError("UB: @abs_diff result not representable in declared i64");
        return makeInt(N, r);
#endif
      }
    };

    /**
     * @brief @signum(x) returns sign of x in {-1, 0, 1}.
     */
    class SignumIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t x = argSint(intr, args, 0);
        int64_t r = (x > 0) - (x < 0);
        return makeInt(N, r);
      }
    };

    /**
     * @brief @clamp(v, lo, hi) — signed clamp; UB if lo > hi.
     */
    class ClampIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t v = argSint(intr, args, 0);
        int64_t lo = argSint(intr, args, 1);
        int64_t hi = argSint(intr, args, 2);
        if (lo > hi)
          throw UndefinedBehaviorError("UB: @clamp requires lo <= hi");
        int64_t r = v < lo ? lo : (v > hi ? hi : v);
        return makeInt(N, r);
      }
    };

    /**
     * @brief @midpoint(a, b) — (a + b) / 2, truncating toward zero.
     */
    class MidpointIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t a = argSint(intr, args, 0), b = argSint(intr, args, 1);
        // For N <= 63, (a + b) always fits in int64. For N == 64, do the
        // sign-extend-by-1 trick via __int128 (or split). Use __int128 when
        // available; fall back to a portable computation otherwise.
        if (N <= 63) {
          int64_t s = a + b;
          int64_t r = s / 2; // truncation toward zero (C/C++ since C99)
          return makeInt(N, r);
        }
        // N == 64: compute via 128-bit if available.
#if defined(__SIZEOF_INT128__)
        __int128 s = static_cast<__int128>(a) + static_cast<__int128>(b);
        __int128 r = s / 2;
        return makeInt(N, static_cast<int64_t>(r));
#else
        // Portable fallback: ((a & b) + ((a ^ b) >> 1)) gives floor((a+b)/2);
        // adjust by +1 when the sum is negative-but-not-divisible to round
        // toward zero instead of toward -inf.
        int64_t floor_avg = (a & b) + ((a ^ b) >> 1);
        int64_t r = floor_avg;
        if ((a ^ b) & 1 && floor_avg < 0)
          r += 1;
        return makeInt(N, r);
#endif
      }
    };

    // ── §12.4 Bit-manipulation (v0.2.2 extra batch B) ──────────────────────────────

    /**
     * @brief @parity(x) -> i1; XOR of all bits.
     */
    class ParityIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint64_t u = argUint(intr, args, 0);
        int64_t p = __builtin_parityll(u);
        return makeInt(1, p & 1);
      }
    };

    /**
     * @brief @bswap(x) — reverse byte order. Declaration is rejected if N % 8 != 0.
     */
    class BswapIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t u = argUint(intr, args, 0);
        uint32_t nbytes = N / 8;
        uint64_t r = 0;
        for (uint32_t i = 0; i < nbytes; ++i) {
          uint64_t byte = (u >> (i * 8)) & 0xFFULL;
          r |= byte << ((nbytes - 1 - i) * 8);
        }
        return makeInt(N, static_cast<int64_t>(r));
      }
    };

    /**
     * @brief @bitreverse(x) — reverse all N bits.
     */
    class BitreverseIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t u = argUint(intr, args, 0);
        uint64_t r = 0;
        for (uint32_t i = 0; i < N; ++i) {
          if ((u >> i) & 1ULL)
            r |= UINT64_C(1) << (N - 1 - i);
        }
        return makeInt(N, static_cast<int64_t>(r));
      }
    };

    /**
     * @brief @rotl(x, n) — rotate x left by n positions. UB if n < 0 or n >= N.
     */
    class RotlIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t u = argUint(intr, args, 0);
        int64_t n = argSint(intr, args, 1);
        if (n < 0 || n >= static_cast<int64_t>(N))
          throw UndefinedBehaviorError("UB: @rotl requires 0 <= n < N");
        uint64_t r;
        if (n == 0)
          r = u;
        else
          r = ((u << n) | (u >> (N - n))) & static_cast<uint64_t>(maskOfN(N));
        return makeInt(N, static_cast<int64_t>(r));
      }
    };

    /**
     * @brief @rotr(x, n) — rotate x right by n positions. UB if n < 0 or n >= N.
     */
    class RotrIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t u = argUint(intr, args, 0);
        int64_t n = argSint(intr, args, 1);
        if (n < 0 || n >= static_cast<int64_t>(N))
          throw UndefinedBehaviorError("UB: @rotr requires 0 <= n < N");
        uint64_t r;
        if (n == 0)
          r = u;
        else
          r = ((u >> n) | (u << (N - n))) & static_cast<uint64_t>(maskOfN(N));
        return makeInt(N, static_cast<int64_t>(r));
      }
    };

    /**
     * @brief @is_pow2(x) -> i1; true iff x is a positive power of two.
     */
    class IsPow2Intrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        int64_t x = argSint(intr, args, 0);
        uint64_t u = static_cast<uint64_t>(x);
        int64_t r = (x > 0 && (u & (u - 1)) == 0) ? 1 : 0;
        return makeInt(1, r);
      }
    };

    /**
     * @brief @ilog2(x) — floor(log2(x)); UB if x <= 0.
     */
    class Ilog2Intrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t x = argSint(intr, args, 0);
        if (x <= 0)
          throw UndefinedBehaviorError("UB: @ilog2 requires x > 0");
        uint64_t u = static_cast<uint64_t>(x);
        int64_t r = 63 - __builtin_clzll(u);
        return makeInt(N, r);
      }
    };

    // ── §12.5 Integer overflow-aware family (v0.2.2 extra batch C) ────────
    //
    // Conventions: all members are declared at one common iN.  Width-mod
    // arithmetic is computed on int64 and then masked / sign-extended
    // back to iN via makeInt.  No reinterpretation as unsigned —
    // saturation bounds and div_euclid signs are all derived from the
    // signed semantics of iN.

    /**
     * @brief @wrapping_add(a, b) — `(a + b) mod 2^N`, sign-extended to iN.
     */
    class WrappingAddIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t a = argUint(intr, args, 0), b = argUint(intr, args, 1);
        return makeInt(N, static_cast<int64_t>(a + b));
      }
    };

    /**
     * @brief @wrapping_sub(a, b) — `(a − b) mod 2^N`, sign-extended to iN.
     */
    class WrappingSubIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t a = argUint(intr, args, 0), b = argUint(intr, args, 1);
        return makeInt(N, static_cast<int64_t>(a - b));
      }
    };

    /**
     * @brief @wrapping_mul(a, b) — `(a * b) mod 2^N`, sign-extended to iN.
     */
    class WrappingMulIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t a = argUint(intr, args, 0), b = argUint(intr, args, 1);
        // Unsigned mul on the iN-sized values; for N < 64 the high bits
        // are discarded by makeInt's mask + sign-extend.  For N == 64 the
        // low 64 bits of a 128-bit product equal `(uint64_t)(a * b)`.
        return makeInt(N, static_cast<int64_t>(a * b));
      }
    };

    /**
     * @brief @wrapping_neg(x) — `(-x) mod 2^N`.  Maps INT_MIN_N to itself.
     */
    class WrappingNegIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t a = argUint(intr, args, 0);
        return makeInt(N, static_cast<int64_t>(-a));
      }
    };

    /**
     * @brief @wrapping_shl(x, n) — left-shift modulo 2^N.  UB if `n < 0` or
     * `n >= N` (same as the OpAtom shift rule §7.1 rule 5).
     */
    class WrappingShlIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        uint64_t a = argUint(intr, args, 0);
        int64_t n = argSint(intr, args, 1);
        if (n < 0 || n >= static_cast<int64_t>(N))
          throw UndefinedBehaviorError("UB: @wrapping_shl requires 0 <= n < N");
        return makeInt(N, static_cast<int64_t>(a << n));
      }
    };

    /**
     * @brief @wrapping_shr(x, n) — arithmetic right-shift mod 2^N.  UB if
     * `n < 0` or `n >= N`.  Sign bit is preserved (matches RefractIR's signed
     * convention; `LShr` is the OpAtom path for logical shift).
     */
    class WrappingShrIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t x = argSint(intr, args, 0);
        int64_t n = argSint(intr, args, 1);
        if (n < 0 || n >= static_cast<int64_t>(N))
          throw UndefinedBehaviorError("UB: @wrapping_shr requires 0 <= n < N");
        // Arithmetic shift in C++20 is implementation-defined before
        // C++20 but well-defined as sign-preserving since.  GCC / clang
        // both perform an arithmetic shift on signed types.
        return makeInt(N, x >> n);
      }
    };

    /**
     * @brief @saturating_add(a, b) — clamp the true sum to [INT_MIN_N,
     * INT_MAX_N].
     */
    class SaturatingAddIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t a = argSint(intr, args, 0), b = argSint(intr, args, 1);
        int64_t lo = intMinN(N), hi = intMaxN(N);
#if defined(__SIZEOF_INT128__)
        __int128 s = static_cast<__int128>(a) + static_cast<__int128>(b);
        int64_t r = (s < lo) ? lo : (s > hi ? hi : static_cast<int64_t>(s));
#else
        // Fallback: detect overflow by sign analysis on the true sum.
        int64_t s = a + b;
        bool ovPos = (a > 0 && b > 0 && s < 0);
        bool ovNeg = (a < 0 && b < 0 && s >= 0);
        int64_t r = ovPos ? hi : ovNeg ? lo : (s < lo ? lo : (s > hi ? hi : s));
#endif
        return makeInt(N, r);
      }
    };

    /**
     * @brief @saturating_sub(a, b) — clamp the true difference to
     * [INT_MIN_N, INT_MAX_N].
     */
    class SaturatingSubIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t a = argSint(intr, args, 0), b = argSint(intr, args, 1);
        int64_t lo = intMinN(N), hi = intMaxN(N);
#if defined(__SIZEOF_INT128__)
        __int128 s = static_cast<__int128>(a) - static_cast<__int128>(b);
        int64_t r = (s < lo) ? lo : (s > hi ? hi : static_cast<int64_t>(s));
#else
        int64_t s = a - b;
        bool ovPos = (a >= 0 && b < 0 && s < 0);
        bool ovNeg = (a < 0 && b > 0 && s >= 0);
        int64_t r = ovPos ? hi : ovNeg ? lo : (s < lo ? lo : (s > hi ? hi : s));
#endif
        return makeInt(N, r);
      }
    };

    /**
     * @brief @saturating_mul(a, b) — clamp the true product to [INT_MIN_N,
     * INT_MAX_N].
     */
    class SaturatingMulIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t a = argSint(intr, args, 0), b = argSint(intr, args, 1);
        int64_t lo = intMinN(N), hi = intMaxN(N);
#if defined(__SIZEOF_INT128__)
        __int128 p = static_cast<__int128>(a) * static_cast<__int128>(b);
        int64_t r = (p < lo) ? lo : (p > hi ? hi : static_cast<int64_t>(p));
#else
        // No __int128: cover N <= 32 fully via int64 multiplication, and
        // fall back to a sign-only saturation for N == 64.
        if (N <= 32) {
          int64_t p = a * b;
          int64_t r = p < lo ? lo : (p > hi ? hi : p);
          return makeInt(N, r);
        }
        bool resPos = ((a >= 0) == (b >= 0));
        int64_t r;
        if (a == 0 || b == 0)
          r = 0;
        else if (resPos && a > hi / b)
          r = hi;
        else if (!resPos && a != 0 && (b < lo / a))
          r = lo;
        else
          r = a * b;
#endif
        return makeInt(N, r);
      }
    };

    /**
     * @brief @saturating_neg(x) — `-x`, clamped at INT_MAX_N when
     * `x == INT_MIN_N` (the only saturating case).
     */
    class SaturatingNegIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t x = argSint(intr, args, 0);
        int64_t r = (x == intMinN(N)) ? intMaxN(N) : -x;
        return makeInt(N, r);
      }
    };

    /**
     * @brief @div_euclid(a, b) — Euclidean division; result rounds toward
     * −∞ when the remainder would be negative.  UB on `b == 0` and on
     * `a == INT_MIN_N && b == -1` (true quotient `-INT_MIN_N` does not
     * fit in iN).
     */
    class DivEuclidIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t a = argSint(intr, args, 0), b = argSint(intr, args, 1);
        if (b == 0)
          throw UndefinedBehaviorError("UB: @div_euclid divisor is zero");
        if (a == intMinN(N) && b == -1)
          throw UndefinedBehaviorError(
              "UB: @div_euclid INT_MIN_N / -1 not representable in declared iN"
          );
        int64_t q = a / b;
        // Truncation toward zero -> Euclidean: if remainder is negative,
        // step the quotient toward -infinity.
        int64_t r = a - q * b;
        if (r < 0)
          q -= (b > 0) ? 1 : -1;
        return makeInt(N, q);
      }
    };

    /**
     * @brief @rem_euclid(a, b) — `a − @div_euclid(a, b) * b`; result is
     * always in `[0, |b|)`.  UB on `b == 0` and on `a == INT_MIN_N &&
     * b == -1` (matches @div_euclid).
     */
    class RemEuclidIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        int64_t a = argSint(intr, args, 0), b = argSint(intr, args, 1);
        if (b == 0)
          throw UndefinedBehaviorError("UB: @rem_euclid divisor is zero");
        if (a == intMinN(N) && b == -1)
          throw UndefinedBehaviorError(
              "UB: @rem_euclid INT_MIN_N rem -1 paired with non-representable quotient"
          );
        int64_t r = a % b;
        if (r < 0)
          r += (b > 0) ? b : -b;
        return makeInt(N, r);
      }
    };

    // ── §12.6 Floating-point sign / bit ops (v0.2.2 extra batch D.1) ────

    /**
     * @brief @fabs(x) = |x|. Clears the IEEE 754 sign bit; never UB on the
     * RefractIR finite-only domain.
     */
    class FabsIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        double x = argFloat(intr, args, 0);
        return makeFloat(N, std::fabs(x));
      }
    };

    /**
     * @brief @fneg(x) = -x. Flips the sign bit; preserves signed zero
     * (@fneg(+0.0) == -0.0).
     */
    class FnegIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        double x = argFloat(intr, args, 0);
        return makeFloat(N, -x);
      }
    };

    /**
     * @brief @copysign(x, y) returns x with the sign of y.
     */
    class CopysignIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        double x = argFloat(intr, args, 0), y = argFloat(intr, args, 1);
        return makeFloat(N, std::copysign(x, y));
      }
    };

    /**
     * @brief @fmin / @fmax — IEEE 754-2008 minNum/maxNum semantics, aligned
     * with WASM `fN.min` / `fN.max` (the §12.6 D.3 reference target).  Under
     * RefractIR's finite-only domain the inputs are always finite, so the only
     * subtle case is signed zero: fmin returns -0 if either operand is -0,
     * fmax returns +0 if either operand is +0.  glibc's `std::fmin` /
     * `std::fmax` are *implementation-defined* on signed-zero pairs (and in
     * practice return the first argument), so we tie-break explicitly to
     * stay bit-exact with WASM.
     */
    class FminIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        double x = argFloat(intr, args, 0), y = argFloat(intr, args, 1);
        double r;
        if (x < y)
          r = x;
        else if (y < x)
          r = y;
        else
          r = std::signbit(x) ? x : y; // (x == y) — prefer -0 over +0
        return makeFloat(N, r);
      }
    };

    class FmaxIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        double x = argFloat(intr, args, 0), y = argFloat(intr, args, 1);
        double r;
        if (x > y)
          r = x;
        else if (y > x)
          r = y;
        else
          r = std::signbit(x) ? y : x; // (x == y) — prefer +0 over -0
        return makeFloat(N, r);
      }
    };

    /**
     * @brief @signbit(x) returns 1 if x has its sign bit set (negative
     * finite or -0.0), else 0. i1 return.
     */
    class SignbitIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl & /*intr*/, const std::vector<RuntimeValue> &args) const override {
        double x = args[0].floatVal;
        return makeInt(1, std::signbit(x) ? 1 : 0);
      }
    };

    /**
     * @brief @is_normal(x) — 1 if x is a normal IEEE 754 value (biased
     * exponent in [1, max-1]), else 0.  Inside RefractIR's finite-only
     * domain the false set is {±0, subnormals}.
     */
    class IsNormalIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        // Classify at the operand's declared precision — an f32 subnormal
        // stored as a double would otherwise be classified as normal at
        // f64 precision (f64's normal range is much wider).
        uint32_t N = fpBitsOf(intr.params[0].type);
        double v = args[0].floatVal;
        if (N == 32) {
          float f = static_cast<float>(v);
          return makeInt(1, std::isnormal(f) ? 1 : 0);
        }
        return makeInt(1, std::isnormal(v) ? 1 : 0);
      }
    };

    /**
     * @brief @is_subnormal(x) — 1 if x is a subnormal IEEE 754 value
     * (biased exponent 0 and mantissa != 0), else 0.  ±0 are NOT
     * subnormal.
     */
    class IsSubnormalIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        // Operate at the operand's declared precision: an f32 subnormal
        // stored as a double would otherwise be classified as normal at
        // f64 precision.
        uint32_t N = fpBitsOf(intr.params[0].type);
        double v = args[0].floatVal;
        if (N == 32) {
          float f = static_cast<float>(v);
          return makeInt(1, (std::fpclassify(f) == FP_SUBNORMAL) ? 1 : 0);
        }
        return makeInt(1, (std::fpclassify(v) == FP_SUBNORMAL) ? 1 : 0);
      }
    };

    /**
     * @brief @to_bits(x: fN) → iN: raw IEEE 754 bit pattern reinterpreted
     * as a signed integer of equal width. f32 → i32, f64 → i64.
     */
    class ToBitsIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = *TypeUtils::getIntBitWidth(intr.retType);
        if (N == 32) {
          float f = static_cast<float>(args[0].floatVal);
          uint32_t u;
          std::memcpy(&u, &f, 4);
          return makeInt(32, static_cast<int64_t>(static_cast<int32_t>(u)));
        }
        double d = args[0].floatVal;
        uint64_t u;
        std::memcpy(&u, &d, 8);
        return makeInt(64, static_cast<int64_t>(u));
      }
    };

    /**
     * @brief @from_bits(x: iN) → fN: raw IEEE 754 bit pattern reinterpreted
     * as a float of equal width. UB if the resulting value is non-finite
     * (matches the §2.9 finite-only domain).
     */
    class FromBitsIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        if (N == 32) {
          uint32_t u = static_cast<uint32_t>(args[0].intVal & 0xFFFFFFFFu);
          float f;
          std::memcpy(&f, &u, 4);
          if (!std::isfinite(f))
            throw UndefinedBehaviorError(
                "UB: @from_bits result is non-finite (spec §2.9 finite-only domain)"
            );
          return makeFloat(32, static_cast<double>(f));
        }
        uint64_t u = static_cast<uint64_t>(args[0].intVal);
        double d;
        std::memcpy(&d, &u, 8);
        if (!std::isfinite(d))
          throw UndefinedBehaviorError(
              "UB: @from_bits result is non-finite (spec §2.9 finite-only domain)"
          );
        return makeFloat(64, d);
      }
    };

    // ── §12.6 D.4 correctly-rounded math ──────────────────────────────────
    //
    // @sqrt, @floor, @ceil, @trunc are all unary fN → fN.  Each maps to a
    // single correctly-rounded IEEE 754 operation; computing at the
    // operand's declared precision keeps the interpreter bit-exact with the
    // C builtins (`sqrtf` vs a double `sqrt` narrowed afterward would
    // double-round) and the WASM `fN.*` opcodes.  @floor/@ceil/@trunc never
    // leave the finite domain; @sqrt of a strictly-negative value is NaN,
    // which is UB under §2.9.

    class SqrtIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        double x = argFloat(intr, args, 0);
        // Round at the operand precision: sqrtf(x) for f32, sqrt(x) for f64.
        // sqrt(-0.0) == -0.0 (finite, not UB); sqrt(x<0) == NaN → UB.
        double r = (N == 32) ? static_cast<double>(std::sqrt(static_cast<float>(x))) : std::sqrt(x);
        if (!std::isfinite(r))
          throw UndefinedBehaviorError(
              "UB: @sqrt of a negative value is NaN (spec §2.9 finite-only domain)"
          );
        return makeFloat(N, r);
      }
    };

    class FloorIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        double x = argFloat(intr, args, 0);
        return makeFloat(N, std::floor(x));
      }
    };

    class CeilIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        double x = argFloat(intr, args, 0);
        return makeFloat(N, std::ceil(x));
      }
    };

    class TruncIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        double x = argFloat(intr, args, 0);
        return makeFloat(N, std::trunc(x));
      }
    };

    // ── §12.6 D.5 compositions ─────────────────────────────────────────────
    //
    // @fract(x) = x - trunc(x) is the fractional part (sign follows x); the
    // result magnitude is always < 1 so it never leaves the finite domain —
    // no UB.  @recip(x) = 1/x is UB when the result is non-finite (x = ±0.0
    // → ±∞, or a tiny |x| whose reciprocal overflows).  Both compute at the
    // operand's declared precision to stay bit-exact with the backends.

    class FractIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        double x = argFloat(intr, args, 0);
        double r;
        if (N == 32) {
          float fx = static_cast<float>(x);
          r = static_cast<double>(fx - std::trunc(fx));
        } else {
          r = x - std::trunc(x);
        }
        return makeFloat(N, r);
      }
    };

    class RecipIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        uint32_t N = fpBitsOf(intr.retType);
        double x = argFloat(intr, args, 0);
        double r = (N == 32) ? static_cast<double>(1.0f / static_cast<float>(x)) : 1.0 / x;
        if (!std::isfinite(r))
          throw UndefinedBehaviorError(
              "UB: @recip result is non-finite (division by zero or overflow, spec §2.9)"
          );
        return makeFloat(N, r);
      }
    };

    // ── Checksum machinery ────────────────────────────────────────────────
    //
    // Bit-exact counterpart of the C-side helpers emitted by
    // src/backend/c_intrinsics.cpp. Both sides implement the same byte-wise
    // CRC32 update step:
    //
    //   state' = (state >> 8) ^ crc32_tab[(state ^ byte) & 0xFF]
    //
    // Reflected polynomial 0xEDB88320, no initial / final XOR, val's bytes
    // processed LSB-first. The lookup table is a function-local static so
    // the helper is self-contained — no shared globals across helpers.

    inline const uint32_t *crc32Table() {
      static uint32_t tab[256] = {0};
      static bool inited = false;
      if (!inited) {
        const uint32_t poly = 0xEDB88320u;
        for (uint32_t i = 0; i < 256; ++i) {
          uint32_t c = i;
          for (int j = 0; j < 8; ++j)
            c = (c & 1u) ? ((c >> 1) ^ poly) : (c >> 1);
          tab[i] = c;
        }
        inited = true;
      }
      return tab;
    }

    class Crc32UpdateIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        // state: i32 (param 0); val: iN (param 1); return: i32.
        uint32_t state = static_cast<uint32_t>(argUint(intr, args, 0));
        uint64_t val = argUint(intr, args, 1);
        auto pb = TypeUtils::getIntBitWidth(intr.params[1].type);
        uint32_t valBits = pb ? *pb : args[1].bits;
        // Round up to whole bytes so an iN with N not a multiple of 8 still
        // hashes the bits in its width (extra bits are zero-padded by the
        // argUint mask).
        uint32_t nBytes = (valBits + 7) / 8;
        const uint32_t *tab = crc32Table();
        for (uint32_t b = 0; b < nBytes; ++b) {
          uint8_t byte = static_cast<uint8_t>(val & 0xFFu);
          val >>= 8;
          state = (state >> 8) ^ tab[(state ^ byte) & 0xFFu];
        }
        return makeInt(32, static_cast<int64_t>(static_cast<int32_t>(state)));
      }
    };

    class CheckChksumIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        // expected, actual: i32 → returns actual; raises UB on mismatch so
        // symiri flags the divergence the same way the C assert() does at
        // run time.
        int64_t expected = argSint(intr, args, 0);
        int64_t actual = argSint(intr, args, 1);
        if (expected != actual) {
          throw UndefinedBehaviorError(
              "@check_chksum: mismatch (expected=" + std::to_string(expected) +
              ", actual=" + std::to_string(actual) + ")"
          );
        }
        return makeInt(32, actual);
      }
    };

    // ── Observability beacon ───────────────────────────────────────────────
    //
    // @observe(v) = v. The reference oracle's value semantics is the identity;
    // the observable side effect (a volatile write) exists only in the C
    // lowering, so the interpreter simply returns the argument unchanged.
    class ObserveIntrinsic final : public InterpreterIntrinsic {
    public:
      RuntimeValue
      eval(const IntrinsicDecl &, const std::vector<RuntimeValue> &args) const override {
        return args[0];
      }
    };

    // ── §12.4 Horizontal vector reductions (v0.2.3 V1) ─────────────────────
    //
    // The sole argument is a vector `<N> T` (a RuntimeValue of kind Vec whose
    // arrayVal holds the N lane scalars); the result is the scalar element
    // type `T`.  Every reduction is the normative left-to-right fold over
    // lanes 0..N-1 — `((v[0] ⊕ v[1]) ⊕ v[2]) ⊕ …` — and reads *every* lane,
    // so an undef lane is UB (rule 22).  Each fold step is an ordinary scalar
    // operation and carries the ordinary scalar UB: @reduce_add over iN traps
    // on any partial sum outside the iN range (rule 4), over fN on a ±∞/NaN
    // intermediate (rules 6–7).  min/max fold with strict comparison (ties
    // keep the accumulator); the bitwise reductions never trap.  A single
    // parametrized class covers all six kinds — they differ only by the
    // per-step operator and the int-vs-fp element split.
    class ReduceIntrinsic final : public InterpreterIntrinsic {
    public:
      explicit ReduceIntrinsic(IntrinsicKind kind) : kind_(kind) {}

      RuntimeValue
      eval(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) const override {
        if (args.empty() || args[0].kind != RuntimeValue::Kind::Vec)
          throw std::runtime_error("Reduction " + intr.name.name + ": argument is not a vector");
        const std::vector<RuntimeValue> &lanes = args[0].arrayVal;
        if (lanes.empty())
          throw std::runtime_error("Reduction " + intr.name.name + ": empty vector");

        if (auto ib = TypeUtils::getIntBitWidth(intr.retType))
          return foldInt(intr, lanes, *ib);
        return foldFloat(intr, lanes, fpBitsOf(intr.retType));
      }

    private:
      IntrinsicKind kind_;

      // Read lane k as a sign-extended int64, raising UB on an undef lane.
      static int64_t laneInt(
          const IntrinsicDecl &intr, const std::vector<RuntimeValue> &lanes, size_t k, uint32_t W
      ) {
        const RuntimeValue &l = lanes[k];
        if (l.kind == RuntimeValue::Kind::Undef)
          throw UndefinedBehaviorError(
              "UB: " + intr.name.name + " reads an undef vector lane (rule 22)"
          );
        if (l.kind != RuntimeValue::Kind::Int)
          throw std::runtime_error("Reduction " + intr.name.name + ": non-integer lane");
        return sextToInt64(l.intVal, W);
      }

      static double
      laneFloat(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &lanes, size_t k) {
        const RuntimeValue &l = lanes[k];
        if (l.kind == RuntimeValue::Kind::Undef)
          throw UndefinedBehaviorError(
              "UB: " + intr.name.name + " reads an undef vector lane (rule 22)"
          );
        if (l.kind != RuntimeValue::Kind::Float)
          throw std::runtime_error("Reduction " + intr.name.name + ": non-float lane");
        return l.floatVal;
      }

      RuntimeValue
      foldInt(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &lanes, uint32_t W) const {
        int64_t lo = intMinN(W), hi = intMaxN(W);
        int64_t acc = laneInt(intr, lanes, 0, W);
        for (size_t k = 1; k < lanes.size(); ++k) {
          int64_t x = laneInt(intr, lanes, k, W);
          switch (kind_) {
            case IntrinsicKind::ReduceAdd: {
              __int128 s = static_cast<__int128>(acc) + static_cast<__int128>(x);
              if (s < static_cast<__int128>(lo) || s > static_cast<__int128>(hi))
                throw UndefinedBehaviorError(
                    "UB: @reduce_add partial sum not representable in the element type (rule 4)"
                );
              acc = static_cast<int64_t>(s);
              break;
            }
            case IntrinsicKind::ReduceMin:
              acc = (x < acc) ? x : acc;
              break;
            case IntrinsicKind::ReduceMax:
              acc = (x > acc) ? x : acc;
              break;
            case IntrinsicKind::ReduceAnd:
              acc &= x;
              break;
            case IntrinsicKind::ReduceOr:
              acc |= x;
              break;
            case IntrinsicKind::ReduceXor:
              acc ^= x;
              break;
            default:
              throw std::runtime_error("Reduction: unexpected integer kind");
          }
        }
        return makeInt(W, acc);
      }

      RuntimeValue foldFloat(
          const IntrinsicDecl &intr, const std::vector<RuntimeValue> &lanes, uint32_t W
      ) const {
        double acc = laneFloat(intr, lanes, 0);
        for (size_t k = 1; k < lanes.size(); ++k) {
          double x = laneFloat(intr, lanes, k);
          switch (kind_) {
            case IntrinsicKind::ReduceAdd:
              acc = checkFPResult(acc + x, W);
              break;
            // min / max fold with IEEE minNum/maxNum signed-zero tie-break
            // (min prefers -0, max prefers +0), identical to @fmin/@fmax.
            // §12.4 requires reduce_min/max to be order-independent so
            // backends may use pairwise/hardware reductions; this canonical
            // tie-break provides that (a strict `<` fold would make ±0
            // order-observable).
            case IntrinsicKind::ReduceMin:
              if (x < acc)
                acc = x;
              else if (!(acc < x))
                acc = std::signbit(acc) ? acc : x; // tie: prefer -0
              break;
            case IntrinsicKind::ReduceMax:
              if (x > acc)
                acc = x;
              else if (!(acc > x))
                acc = std::signbit(acc) ? x : acc; // tie: prefer +0
              break;
            default:
              throw std::runtime_error("Reduction: unexpected float kind (bitwise on fp?)");
          }
        }
        return makeFloat(W, acc);
      }
    };

    // ── Registry ────────────────────────────────────────────────────────────

    /**
     * @brief Centralized registry of all interpreter intrinsic implementations.
     */
    class IntrinsicRegistry {
    public:
      static const IntrinsicRegistry &get() {
        static IntrinsicRegistry instance;
        return instance;
      }

      const InterpreterIntrinsic *lookup(IntrinsicKind kind) const {
        auto it = registry_.find(kind);
        if (it != registry_.end())
          return it->second.get();
        return nullptr;
      }

    private:
      IntrinsicRegistry() {
        registry_[IntrinsicKind::Abs] = std::make_unique<AbsIntrinsic>();
        registry_[IntrinsicKind::Min] = std::make_unique<MinIntrinsic>();
        registry_[IntrinsicKind::Max] = std::make_unique<MaxIntrinsic>();
        registry_[IntrinsicKind::Popcount] = std::make_unique<PopcountIntrinsic>();
        registry_[IntrinsicKind::Clz] = std::make_unique<ClzIntrinsic>();
        registry_[IntrinsicKind::Ctz] = std::make_unique<CtzIntrinsic>();
        registry_[IntrinsicKind::AbsDiff] = std::make_unique<AbsDiffIntrinsic>();
        registry_[IntrinsicKind::Signum] = std::make_unique<SignumIntrinsic>();
        registry_[IntrinsicKind::Clamp] = std::make_unique<ClampIntrinsic>();
        registry_[IntrinsicKind::Midpoint] = std::make_unique<MidpointIntrinsic>();
        registry_[IntrinsicKind::Parity] = std::make_unique<ParityIntrinsic>();
        registry_[IntrinsicKind::Bswap] = std::make_unique<BswapIntrinsic>();
        registry_[IntrinsicKind::Bitreverse] = std::make_unique<BitreverseIntrinsic>();
        registry_[IntrinsicKind::Rotl] = std::make_unique<RotlIntrinsic>();
        registry_[IntrinsicKind::Rotr] = std::make_unique<RotrIntrinsic>();
        registry_[IntrinsicKind::IsPow2] = std::make_unique<IsPow2Intrinsic>();
        registry_[IntrinsicKind::Ilog2] = std::make_unique<Ilog2Intrinsic>();
        // §12.5 — overflow-aware family.
        registry_[IntrinsicKind::WrappingAdd] = std::make_unique<WrappingAddIntrinsic>();
        registry_[IntrinsicKind::WrappingSub] = std::make_unique<WrappingSubIntrinsic>();
        registry_[IntrinsicKind::WrappingMul] = std::make_unique<WrappingMulIntrinsic>();
        registry_[IntrinsicKind::WrappingNeg] = std::make_unique<WrappingNegIntrinsic>();
        registry_[IntrinsicKind::WrappingShl] = std::make_unique<WrappingShlIntrinsic>();
        registry_[IntrinsicKind::WrappingShr] = std::make_unique<WrappingShrIntrinsic>();
        registry_[IntrinsicKind::SaturatingAdd] = std::make_unique<SaturatingAddIntrinsic>();
        registry_[IntrinsicKind::SaturatingSub] = std::make_unique<SaturatingSubIntrinsic>();
        registry_[IntrinsicKind::SaturatingMul] = std::make_unique<SaturatingMulIntrinsic>();
        registry_[IntrinsicKind::SaturatingNeg] = std::make_unique<SaturatingNegIntrinsic>();
        registry_[IntrinsicKind::DivEuclid] = std::make_unique<DivEuclidIntrinsic>();
        registry_[IntrinsicKind::RemEuclid] = std::make_unique<RemEuclidIntrinsic>();
        // §12.6 — FP sign / bit ops (v0.2.2 extra batch D.1).
        registry_[IntrinsicKind::Fabs] = std::make_unique<FabsIntrinsic>();
        registry_[IntrinsicKind::Fneg] = std::make_unique<FnegIntrinsic>();
        registry_[IntrinsicKind::Copysign] = std::make_unique<CopysignIntrinsic>();
        registry_[IntrinsicKind::Signbit] = std::make_unique<SignbitIntrinsic>();
        registry_[IntrinsicKind::ToBits] = std::make_unique<ToBitsIntrinsic>();
        registry_[IntrinsicKind::FromBits] = std::make_unique<FromBitsIntrinsic>();
        // §12.6 D.2 — classification predicates.
        registry_[IntrinsicKind::IsNormal] = std::make_unique<IsNormalIntrinsic>();
        registry_[IntrinsicKind::IsSubnormal] = std::make_unique<IsSubnormalIntrinsic>();
        // §12.6 D.3 — min / max.
        registry_[IntrinsicKind::Fmin] = std::make_unique<FminIntrinsic>();
        registry_[IntrinsicKind::Fmax] = std::make_unique<FmaxIntrinsic>();
        // §12.6 D.4 — correctly-rounded math.
        registry_[IntrinsicKind::Sqrt] = std::make_unique<SqrtIntrinsic>();
        registry_[IntrinsicKind::Floor] = std::make_unique<FloorIntrinsic>();
        registry_[IntrinsicKind::Ceil] = std::make_unique<CeilIntrinsic>();
        registry_[IntrinsicKind::Trunc] = std::make_unique<TruncIntrinsic>();
        // §12.6 D.5 — compositions.
        registry_[IntrinsicKind::Fract] = std::make_unique<FractIntrinsic>();
        registry_[IntrinsicKind::Recip] = std::make_unique<RecipIntrinsic>();
        // Checksum primitives.
        registry_[IntrinsicKind::Crc32Update] = std::make_unique<Crc32UpdateIntrinsic>();
        registry_[IntrinsicKind::CheckChksum] = std::make_unique<CheckChksumIntrinsic>();
        registry_[IntrinsicKind::Observe] = std::make_unique<ObserveIntrinsic>();
        // §12.4 — horizontal vector reductions (v0.2.3 V1).
        registry_[IntrinsicKind::ReduceAdd] =
            std::make_unique<ReduceIntrinsic>(IntrinsicKind::ReduceAdd);
        registry_[IntrinsicKind::ReduceMin] =
            std::make_unique<ReduceIntrinsic>(IntrinsicKind::ReduceMin);
        registry_[IntrinsicKind::ReduceMax] =
            std::make_unique<ReduceIntrinsic>(IntrinsicKind::ReduceMax);
        registry_[IntrinsicKind::ReduceAnd] =
            std::make_unique<ReduceIntrinsic>(IntrinsicKind::ReduceAnd);
        registry_[IntrinsicKind::ReduceOr] =
            std::make_unique<ReduceIntrinsic>(IntrinsicKind::ReduceOr);
        registry_[IntrinsicKind::ReduceXor] =
            std::make_unique<ReduceIntrinsic>(IntrinsicKind::ReduceXor);
      }

      std::unordered_map<IntrinsicKind, std::unique_ptr<InterpreterIntrinsic>> registry_;
    };

  } // namespace

  RuntimeValue evalIntrinsic(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args) {
    auto kind = getIntrinsicKind(intr.name.name);
    if (kind)
      if (auto impl = IntrinsicRegistry::get().lookup(*kind))
        return impl->eval(intr, args);
    throw std::runtime_error("Unknown intrinsic: " + intr.name.name);
  }

  RuntimeValue Interpreter::callIntrinsic(
      const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args, SourceSpan /*callSpan*/
  ) {
    return evalIntrinsic(intr, args);
  }

} // namespace refractir
