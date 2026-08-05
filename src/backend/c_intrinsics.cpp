// C backend intrinsic helper emission.
//
// This file is the single source of truth for the C code emitted for
// every built-in RefractIR intrinsic. The helpers use a widening-and-mask
// strategy: each iN operation is widened to the next machine integer
// width (8/16/32/64), performed there, then sign-masked back to N bits.
// UB-preconditions abort via __builtin_trap.
//
// Width conventions:
//   - Return-side N/W/sty/uty are the *return* type's widening.
//   - For predicate intrinsics (@parity, @is_pow2), the return is i1
//     (W=8, sty=int8_t) but the input is iN_param with its own widening.
//     emitIntrinsicHelper computes per-param types in the helper
//     signature; impls that touch parameter widths read them via `intr`.
//
// To add a new intrinsic:
//   1. Add its IntrinsicKind to include/analysis/intrinsics.hpp.
//   2. Implement a subclass of CIntrinsic and register it below.

#include "backend/c_backend.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "analysis/intrinsics.hpp"
#include "analysis/type_utils.hpp"

namespace refractir {

  /**
   * @brief Abstract base class for C code-generation of intrinsics.
   * Subclasses generate C function bodies to emulate RefractIR standard intrinsics.
   */
  class CIntrinsic {
  public:
    virtual ~CIntrinsic() = default;

    /**
     * @brief Generates C statements inside the helper function body.
     * @param backend Reference to the code-generation engine.
     * @param intr Intrinsic declaration (for per-param widths).
     * @param N Declared return bit-width.
     * @param W Smallest native machine width fitting N (8, 16, 32, or 64).
     * @param sty Name of the signed type corresponding to W (e.g. "int32_t").
     * @param uty Name of the unsigned type corresponding to W.
     */
    virtual void emit(
        CBackend &backend, const IntrinsicDecl &intr, uint32_t N, uint32_t W,
        const std::string &sty, const std::string &uty
    ) const = 0;

    /**
     * @brief Linkage / inlining qualifier emitted in front of the helper's
     * return type. Defaults to `static inline` so the C compiler can fold
     * trivial intrinsics into their callers. Helpers whose body uses a
     * function-local static (e.g. @crc32_update's lookup table) override
     * to `static __attribute__((noinline))` so per-call-site cloning can't
     * propagate constant table contents into the caller's CCP / SCCP.
     */
    virtual std::string linkageQualifier() const { return "static inline"; }

  protected:
    static std::ostream &out(CBackend &backend);
    // Sink for UB-precondition guard lines: `out(backend)`
    // normally, but a discarding stream under --no-ub-guards, so the
    // guards vanish from the emitted helper while its computation (which
    // is only ever reached on UB-free inputs) is untouched.
    static std::ostream &guardOut(CBackend &backend);

    static uint32_t widen(uint32_t N) {
      return (N <= 8) ? 8 : (N <= 16) ? 16 : (N <= 32) ? 32 : 64;
    }

    static std::string sIntTy(uint32_t W) { return "int" + std::to_string(W) + "_t"; }

    static std::string uIntTy(uint32_t W) { return "uint" + std::to_string(W) + "_t"; }

    /**
     * @brief Parameter declared bit-width (panics if non-integer, which
     * cannot happen for v0.2.2 extra batch A&B intrinsics: semchecker enforces integer
     * parameters).
     */
    static uint32_t paramN(const IntrinsicDecl &intr, size_t i) {
      auto pb = TypeUtils::getIntBitWidth(intr.params[i].type);
      return pb ? *pb : 32;
    }

    /**
     * @brief Sign-extend an expression to occupy `W` bits, treating its low
     * `N` bits as a signed `iN` value: (sty)((uty)(expr) << (W − N)) >> (W − N).
     */
    static std::string makeSextN(
        uint32_t N, uint32_t W, const std::string &sty, const std::string &uty,
        const std::string &expr
    ) {
      if (N == W)
        return expr;
      return "(" + sty + ")((" + uty + ")(" + expr + ") << " + std::to_string(W - N) + ") >> " +
             std::to_string(W - N);
    }

    /**
     * @brief Unsigned bitmask expression that clears bits above bit `N − 1`.
     */
    static std::string
    makeMaskU(uint32_t N, uint32_t W, const std::string &uty, const std::string &expr) {
      if (N == W)
        return "(" + uty + ")(" + expr + ")";
      return "((" + uty + ")(" + expr + ") & ((" + uty + ")1 << " + std::to_string(N) + ") - 1)";
    }

    /**
     * @brief Emit `INT_MIN_N` as a C expression that cannot be parsed as
     * unary-minus on an out-of-range literal.  Writing `-9223372036854775808LL`
     * directly trips the long-long maximum: the lexer treats
     * `9223372036854775808` as `unsigned long long`, and the unary `-` on an
     * unsigned value silently produces `+9223372036854775808`, breaking
     * downstream comparisons and casts.  For `N == 64` we therefore emit
     * the canonical `(-INT64_MAX - 1)` form (and rely on <stdint.h> for
     * `INT64_MAX`); narrower widths fit cleanly in `long long`.
     */
    static std::string intMinNExpr(uint32_t N) {
      if (N == 64)
        return "(-9223372036854775807LL - 1)";
      int64_t lo = -(INT64_C(1) << (N - 1));
      return std::to_string(lo) + "LL";
    }
  };

  /**
   * @brief Friendship-mediator registry to fetch private members of CBackend.
   */
  struct CIntrinsicRegistry {
    static std::ostream &out(CBackend &backend) { return backend.out_; }

    // Guard-line sink shared by both intrinsic base classes:
    // the real stream normally, but a discarding one (null streambuf →
    // writes set badbit and emit nothing) under --no-ub-guards.
    // Emission is single-threaded per backend instance.
    static std::ostream &guardOut(CBackend &backend) {
      static std::ostream nullSink(nullptr);
      return backend.noUbGuards_ ? nullSink : backend.out_;
    }

    using CIntrinsicGenFn = std::unique_ptr<class CIntrinsic>;

    static const std::unordered_map<IntrinsicKind, CIntrinsicGenFn> &getRegistry();
  };

  std::ostream &CIntrinsic::out(CBackend &backend) { return CIntrinsicRegistry::out(backend); }

  std::ostream &CIntrinsic::guardOut(CBackend &backend) {
    return CIntrinsicRegistry::guardOut(backend);
  }

  namespace {

    // ── §12.1 Arithmetic intrinsics ─────────────────────────────────────────

    class AbsIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        guardOut(backend) << "  if (a0 == (" << sty << ")" << intMinNExpr(N)
                          << ") __builtin_trap();\n";
        if (W <= 32)
          out(backend) << "  " << sty << " r = (" << sty << ")abs((int)a0);\n";
        else
          out(backend) << "  " << sty << " r = (" << sty << ")llabs((long long)a0);\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
      }
    };

    class MinIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        out(backend) << "  " << sty << " r = a0 < a1 ? a0 : a1;\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
      }
    };

    class MaxIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        out(backend) << "  " << sty << " r = a0 > a1 ? a0 : a1;\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
      }
    };

    // ── §12.2 Bit-counting intrinsics ───────────────────────────────────────

    class PopcountIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        out(backend) << "  " << uty << " u = " << makeMaskU(N, W, uty, "a0") << ";\n";
        if (W <= 32)
          out(backend) << "  " << sty << " r = (" << sty << ")__builtin_popcount(u);\n";
        else
          out(backend) << "  " << sty << " r = (" << sty << ")__builtin_popcountll((uint64_t)u);\n";
        int64_t maxN = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        guardOut(backend) << "  if (r > " << maxN << "LL) __builtin_trap();\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
      }
    };

    class ClzIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        out(backend) << "  " << uty << " u = " << makeMaskU(N, W, uty, "a0") << ";\n";
        guardOut(backend) << "  if (u == 0) __builtin_trap();\n";
        if (W <= 32) {
          out(backend) << "  " << sty << " r = (" << sty << ")__builtin_clz((uint32_t)u) - "
                       << std::to_string(32 - N) << ";\n";
        } else {
          out(backend) << "  " << sty << " r = (" << sty << ")__builtin_clzll((uint64_t)u) - "
                       << std::to_string(64 - N) << ";\n";
        }
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
      }
    };

    class CtzIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        out(backend) << "  " << uty << " u = " << makeMaskU(N, W, uty, "a0") << ";\n";
        guardOut(backend) << "  if (u == 0) __builtin_trap();\n";
        if (W <= 32)
          out(backend) << "  " << sty << " r = (" << sty << ")__builtin_ctz((uint32_t)u);\n";
        else
          out(backend) << "  " << sty << " r = (" << sty << ")__builtin_ctzll((uint64_t)u);\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
      }
    };

    // ── §12.3 Integer extras ────────────────────────────────

    class AbsDiffIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        // Compute the subtraction in a strictly wider signed type so signed
        // overflow cannot occur, then check |s| fits in signed iN. No
        // reinterpretation as unsigned — RefractIR has no unsigned types.
        int64_t maxN = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        if (N < 64) {
          out(backend) << "  int64_t s = (int64_t)a0 - (int64_t)a1;\n";
          out(backend) << "  int64_t r = s < 0 ? -s : s;\n";
          guardOut(backend) << "  if (r > (int64_t)" << maxN << "LL) __builtin_trap();\n";
          out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")r") << ";\n";
        } else {
          // N == 64: subtraction would overflow i64; use __int128.
          out(backend) << "  __int128 s = (__int128)a0 - (__int128)a1;\n";
          out(backend) << "  __int128 r = s < 0 ? -s : s;\n";
          guardOut(backend) << "  if (r > (__int128)" << maxN << "LL) __builtin_trap();\n";
          out(backend) << "  return (" << sty << ")r;\n";
        }
      }
    };

    class SignumIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        // Predicate: return type is i1 (N=1, W=8). RefractIR's i1 convention
        // stores the boolean as the literal 0 or 1 (matching `cmp` lowering),
        // not sign-extended — so no final sextN is applied.
        if (N == 1) {
          (void) W;
          (void) uty;
          out(backend) << "  return (" << sty << ")((a0 > 0) - (a0 < 0));\n";
        } else {
          out(backend) << "  " << sty << " r = (" << sty << ")((a0 > 0) - (a0 < 0));\n";
          out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
        }
      }
    };

    class ClampIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        guardOut(backend) << "  if (a1 > a2) __builtin_trap();\n";
        out(backend) << "  " << sty << " r = (a0 < a1) ? a1 : (a0 > a2 ? a2 : a0);\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
      }
    };

    class MidpointIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        if (N <= 63) {
          out(backend) << "  int64_t s = (int64_t)a0 + (int64_t)a1;\n";
          out(backend) << "  " << sty << " r = (" << sty << ")(s / 2);\n";
          out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
        } else {
          // i64 midpoint via __int128 (GCC/Clang only — required by RefractIR
          // toolchain in this build).
          out(backend) << "  __int128 s = (__int128)a0 + (__int128)a1;\n";
          out(backend) << "  " << sty << " r = (" << sty << ")(s / 2);\n";
          out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
        }
      }
    };

    // ── §12.4 Bit-manipulation ──────────────────────────────

    class ParityIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &intr, uint32_t N, uint32_t W,
          const std::string &sty, const std::string &uty
      ) const override {
        // Predicate: return type is i1 (N=1, W=8). Read the input at the
        // *parameter's* own width, not the return's. RefractIR's i1 convention
        // stores the boolean as the literal 0 or 1 (matching `cmp` lowering),
        // not sign-extended — so no final sextN is applied.
        (void) N;
        (void) W;
        (void) uty;
        uint32_t pN = paramN(intr, 0);
        uint32_t pW = widen(pN);
        std::string puty = uIntTy(pW);
        out(backend) << "  " << puty << " u = " << makeMaskU(pN, pW, puty, "a0") << ";\n";
        if (pW <= 32)
          out(backend) << "  return (" << sty << ")(__builtin_parity(u) & 1);\n";
        else
          out(backend) << "  return (" << sty << ")(__builtin_parityll((uint64_t)u) & 1);\n";
      }
    };

    class BswapIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        // Generic byte-reverse loop. Compilers fold this for power-of-two
        // widths; for non-power-of-two byte counts (i24, i40, …) it remains a
        // short inline loop.
        out(backend) << "  " << uty << " u = " << makeMaskU(N, W, uty, "a0") << ";\n";
        out(backend) << "  " << uty << " r = 0;\n";
        out(backend) << "  for (uint32_t i = 0; i < " << (N / 8) << "; ++i) {\n";
        out(backend) << "    r |= ((u >> (i*8)) & 0xFF) << ((" << (N / 8 - 1) << " - i) * 8);\n";
        out(backend) << "  }\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")r") << ";\n";
      }
    };

    class BitreverseIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        out(backend) << "  " << uty << " u = " << makeMaskU(N, W, uty, "a0") << ";\n";
        out(backend) << "  " << uty << " r = 0;\n";
        out(backend) << "  for (uint32_t i = 0; i < " << N << "; ++i) {\n";
        out(backend) << "    if ((u >> i) & 1) r |= ((" << uty << ")1 << (" << (N - 1)
                     << " - i));\n";
        out(backend) << "  }\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")r") << ";\n";
      }
    };

    class RotlIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        guardOut(backend) << "  if (a1 < 0 || a1 >= " << N << ") __builtin_trap();\n";
        out(backend) << "  " << uty << " u = " << makeMaskU(N, W, uty, "a0") << ";\n";
        out(backend) << "  " << uty << " n = (" << uty << ")a1;\n";
        out(backend) << "  " << uty << " r = (n == 0) ? u : ((u << n) | (u >> (" << N
                     << " - n)));\n";
        if (N < W)
          out(backend) << "  r &= ((" << uty << ")1 << " << N << ") - 1;\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")r") << ";\n";
      }
    };

    class RotrIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        guardOut(backend) << "  if (a1 < 0 || a1 >= " << N << ") __builtin_trap();\n";
        out(backend) << "  " << uty << " u = " << makeMaskU(N, W, uty, "a0") << ";\n";
        out(backend) << "  " << uty << " n = (" << uty << ")a1;\n";
        out(backend) << "  " << uty << " r = (n == 0) ? u : ((u >> n) | (u << (" << N
                     << " - n)));\n";
        if (N < W)
          out(backend) << "  r &= ((" << uty << ")1 << " << N << ") - 1;\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")r") << ";\n";
      }
    };

    class IsPow2Intrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &intr, uint32_t N, uint32_t W,
          const std::string &sty, const std::string &uty
      ) const override {
        // Predicate: return i1 stored as literal 0 / 1 (RefractIR convention,
        // matches `cmp` lowering). No final sextN.
        (void) N;
        (void) W;
        (void) uty;
        uint32_t pN = paramN(intr, 0);
        uint32_t pW = widen(pN);
        std::string psty = sIntTy(pW);
        (void) pN;
        out(backend) << "  return (" << sty << ")((a0 > 0) && (((" << psty << ")a0 & ((" << psty
                     << ")a0 - 1)) == 0));\n";
      }
    };

    class Ilog2Intrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        guardOut(backend) << "  if (a0 <= 0) __builtin_trap();\n";
        out(backend) << "  " << uty << " u = " << makeMaskU(N, W, uty, "a0") << ";\n";
        if (W <= 32)
          out(backend) << "  " << sty << " r = (" << sty << ")(" << (N - 1)
                       << " - __builtin_clz((uint32_t)u) + " << (32 - N) << ");\n";
        else
          out(backend) << "  " << sty << " r = (" << sty << ")(" << (N - 1)
                       << " - __builtin_clzll((uint64_t)u) + " << (64 - N) << ");\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
      }
    };

    // ── §12.5 Integer overflow-aware family ────────
    //
    // The widening-and-mask strategy already gives us native wrapping at
    // the helper width `W`; for return widths `N < W` we mask + sign-extend
    // back into iN.  Saturating helpers fall back to a wider type (int64 /
    // __int128) so the true sum/diff/product never wraps; UB cases for
    // shifts and div_euclid abort via `__builtin_trap`.

    /**
     * @brief @wrapping_add — (a + b) mod 2^N.
     */
    class WrappingAddIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        out(backend) << "  " << uty << " ua = " << makeMaskU(N, W, uty, "a0") << ";\n";
        out(backend) << "  " << uty << " ub = " << makeMaskU(N, W, uty, "a1") << ";\n";
        out(backend) << "  " << uty << " r = (" << uty << ")(ua + ub);\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")r") << ";\n";
      }
    };

    /**
     * @brief @wrapping_sub — (a − b) mod 2^N.
     */
    class WrappingSubIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        out(backend) << "  " << uty << " ua = " << makeMaskU(N, W, uty, "a0") << ";\n";
        out(backend) << "  " << uty << " ub = " << makeMaskU(N, W, uty, "a1") << ";\n";
        out(backend) << "  " << uty << " r = (" << uty << ")(ua - ub);\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")r") << ";\n";
      }
    };

    /**
     * @brief @wrapping_mul — (a * b) mod 2^N.
     *
     * Computed in uint64 so the multiplication can never trip signed-int
     * overflow during integer promotion: for `uint16_t * uint16_t`, the
     * usual promotion converts both operands to `int`, and `65534 *
     * 65535 ≈ 4.3e9` then overshoots `INT_MAX_32`.  Widening to uint64
     * first makes the arithmetic unsigned and well-defined modularly;
     * narrowing back to `uty` wraps as required.
     */
    class WrappingMulIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        out(backend) << "  " << uty << " ua = " << makeMaskU(N, W, uty, "a0") << ";\n";
        out(backend) << "  " << uty << " ub = " << makeMaskU(N, W, uty, "a1") << ";\n";
        out(backend) << "  " << uty << " r = (" << uty << ")((uint64_t)ua * (uint64_t)ub);\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")r") << ";\n";
      }
    };

    /**
     * @brief @wrapping_neg — (-x) mod 2^N.  INT_MIN_N is its own negation.
     */
    class WrappingNegIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        out(backend) << "  " << uty << " ua = " << makeMaskU(N, W, uty, "a0") << ";\n";
        out(backend) << "  " << uty << " r = (" << uty << ")(0 - ua);\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")r") << ";\n";
      }
    };

    /**
     * @brief @wrapping_shl — left-shift with mod-2^N wrap.  UB on `n` out
     * of `[0, N)`.  The shift itself is done on the unsigned widened type
     * to avoid signed-shift UB at the host level.
     */
    class WrappingShlIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        guardOut(backend) << "  if (a1 < 0 || a1 >= " << N << ") __builtin_trap();\n";
        out(backend) << "  " << uty << " ua = " << makeMaskU(N, W, uty, "a0") << ";\n";
        out(backend) << "  " << uty << " r = (" << uty << ")(ua << a1);\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")r") << ";\n";
      }
    };

    /**
     * @brief @wrapping_shr — arithmetic right shift.  UB on `n` out of
     * `[0, N)`.
     */
    class WrappingShrIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        guardOut(backend) << "  if (a1 < 0 || a1 >= " << N << ") __builtin_trap();\n";
        // Sign-extend a0 to W bits so the host >> performs an arithmetic
        // shift over the full iN sign bit, then re-mask/sext into iN.
        out(backend) << "  " << sty << " sa = " << makeSextN(N, W, sty, uty, "a0") << ";\n";
        out(backend) << "  " << sty << " r = sa >> a1;\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
      }
    };

    /**
     * @brief @saturating_add — clamp the true sum to [INT_MIN_N,
     * INT_MAX_N].  We promote to a strictly wider integer type so the
     * true sum cannot overflow; only the final clamp matters.
     */
    class SaturatingAddIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        int64_t hi = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        std::string loE = intMinNExpr(N);
        if (N <= 32) {
          out(backend) << "  int64_t s = (int64_t)a0 + (int64_t)a1;\n";
          out(backend) << "  if (s > " << hi << "LL) s = " << hi << "LL;\n";
          out(backend) << "  if (s < " << loE << ") s = " << loE << ";\n";
          out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")s") << ";\n";
        } else {
          out(backend) << "  __int128 s = (__int128)a0 + (__int128)a1;\n";
          out(backend) << "  if (s > (__int128)" << hi << "LL) s = (__int128)" << hi << "LL;\n";
          out(backend) << "  if (s < (__int128)(" << loE << ")) s = (__int128)(" << loE << ");\n";
          out(backend) << "  return (" << sty << ")s;\n";
        }
      }
    };

    /**
     * @brief @saturating_sub — clamp the true difference.
     */
    class SaturatingSubIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        int64_t hi = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        std::string loE = intMinNExpr(N);
        if (N <= 32) {
          out(backend) << "  int64_t s = (int64_t)a0 - (int64_t)a1;\n";
          out(backend) << "  if (s > " << hi << "LL) s = " << hi << "LL;\n";
          out(backend) << "  if (s < " << loE << ") s = " << loE << ";\n";
          out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")s") << ";\n";
        } else {
          out(backend) << "  __int128 s = (__int128)a0 - (__int128)a1;\n";
          out(backend) << "  if (s > (__int128)" << hi << "LL) s = (__int128)" << hi << "LL;\n";
          out(backend) << "  if (s < (__int128)(" << loE << ")) s = (__int128)(" << loE << ");\n";
          out(backend) << "  return (" << sty << ")s;\n";
        }
      }
    };

    /**
     * @brief @saturating_mul — clamp the true product.
     */
    class SaturatingMulIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        int64_t hi = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        std::string loE = intMinNExpr(N);
        if (N <= 32) {
          out(backend) << "  int64_t s = (int64_t)a0 * (int64_t)a1;\n";
          out(backend) << "  if (s > " << hi << "LL) s = " << hi << "LL;\n";
          out(backend) << "  if (s < " << loE << ") s = " << loE << ";\n";
          out(backend) << "  return " << makeSextN(N, W, sty, uty, "(" + sty + ")s") << ";\n";
        } else {
          out(backend) << "  __int128 s = (__int128)a0 * (__int128)a1;\n";
          out(backend) << "  if (s > (__int128)" << hi << "LL) s = (__int128)" << hi << "LL;\n";
          out(backend) << "  if (s < (__int128)(" << loE << ")) s = (__int128)(" << loE << ");\n";
          out(backend) << "  return (" << sty << ")s;\n";
        }
      }
    };

    /**
     * @brief @saturating_neg — `-x`, with INT_MIN_N saturated to
     * INT_MAX_N.
     */
    class SaturatingNegIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        int64_t hi = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        std::string loE = intMinNExpr(N);
        out(backend) << "  " << sty << " r;\n";
        out(backend) << "  if (a0 == (" << sty << ")" << loE << ") r = (" << sty << ")" << hi
                     << "LL;\n";
        out(backend) << "  else r = (" << sty << ")(-(int64_t)a0);\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
      }
    };

    /**
     * @brief @div_euclid — `(a / b)` adjusted toward `−∞` when the truncated
     * remainder is negative.  UB on `b == 0` and on
     * `a == INT_MIN_N && b == -1` (true quotient -INT_MIN_N overflows).
     */
    class DivEuclidIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        std::string loE = intMinNExpr(N);
        guardOut(backend) << "  if (a1 == 0) __builtin_trap();\n";
        guardOut(backend) << "  if (a0 == (" << sty << ")" << loE << " && a1 == (" << sty
                          << ")-1) __builtin_trap();\n";
        out(backend) << "  " << sty << " q = a0 / a1;\n";
        out(backend) << "  " << sty << " r = a0 - q * a1;\n";
        out(backend) << "  if (r < 0) q -= (a1 > 0) ? 1 : -1;\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "q") << ";\n";
      }
    };

    /**
     * @brief @rem_euclid — `a − @div_euclid(a, b) * b`; result in [0, |b|).
     */
    class RemEuclidIntrinsic final : public CIntrinsic {
    public:
      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &sty,
          const std::string &uty
      ) const override {
        std::string loE = intMinNExpr(N);
        guardOut(backend) << "  if (a1 == 0) __builtin_trap();\n";
        guardOut(backend) << "  if (a0 == (" << sty << ")" << loE << " && a1 == (" << sty
                          << ")-1) __builtin_trap();\n";
        out(backend) << "  " << sty << " r = a0 % a1;\n";
        // |a1| does not fit in iN signed when a1 == INT_MIN_N (the
        // negation overflows int32 / int64), even though the final
        // result `r + |a1|` is back in iN range.  Compute the addition
        // in unsigned (uty) arithmetic — modular subtraction takes
        // care of the |.| sign analysis, and casting the unsigned sum
        // back to sty wraps to the correct bit pattern.
        out(backend) << "  if (r < 0) r = (" << sty << ")((" << uty << ")r + ((a1 > 0) ? (" << uty
                     << ")a1 : (" << uty << ")(0 - (" << uty << ")a1)));\n";
        out(backend) << "  return " << makeSextN(N, W, sty, uty, "r") << ";\n";
      }
    };

    // ── §12.6 Floating-point sign / bit ops ────

    /**
     * @brief Abstract base for FP-touching intrinsic emitters. Unlike
     * CIntrinsic (integer, takes precomputed N/W/sty/uty), CFpIntrinsic
     * receives only the IntrinsicDecl: per-arg C types are derived from
     * intr.params[i].type via cTypeOf().
     */
    class CFpIntrinsic {
    public:
      virtual ~CFpIntrinsic() = default;
      virtual void emit(CBackend &backend, const IntrinsicDecl &intr) const = 0;

      // Mirrors CIntrinsic::linkageQualifier — @observe needs `noinline` so a
      // caller cannot fold through its identity return and drop the sink write.
      virtual std::string linkageQualifier() const { return "static inline"; }

    protected:
      static std::ostream &out(CBackend &backend) { return CIntrinsicRegistry::out(backend); }

      // See CIntrinsicRegistry::guardOut — discards under --no-ub-guards.
      static std::ostream &guardOut(CBackend &backend) {
        return CIntrinsicRegistry::guardOut(backend);
      }
    };

    class FabsCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &intr) const override {
        bool isF32 = std::get<FloatType>(intr.retType->v).kind == FloatType::Kind::F32;
        out(backend) << "  return " << (isF32 ? "__builtin_fabsf" : "__builtin_fabs") << "(a0);\n";
      }
    };

    class FnegCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &) const override {
        // Plain unary negation flips the sign bit without rounding; this is
        // the RefractIR @fneg contract (no fmul -1.0).
        out(backend) << "  return -a0;\n";
      }
    };

    // Float twin of ObserveCIntrinsic — same volatile-sink anchor, typed on
    // the float so a diverging loop whose only state is FP still carries an
    // observable side effect.
    class ObserveCFpIntrinsic final : public CFpIntrinsic {
    public:
      std::string linkageQualifier() const override { return "static __attribute__((noinline))"; }

      void emit(CBackend &backend, const IntrinsicDecl &intr) const override {
        bool isF32 = std::get<FloatType>(intr.retType->v).kind == FloatType::Kind::F32;
        out(backend) << "  static volatile " << (isF32 ? "float" : "double")
                     << " __rir_observe_fsink;\n";
        out(backend) << "  __rir_observe_fsink = a0;\n";
        out(backend) << "  return a0;\n";
      }
    };

    class CopysignCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &intr) const override {
        bool isF32 = std::get<FloatType>(intr.retType->v).kind == FloatType::Kind::F32;
        out(backend) << "  return " << (isF32 ? "__builtin_copysignf" : "__builtin_copysign")
                     << "(a0, a1);\n";
      }
    };

    class SignbitCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &) const override {
        // __builtin_signbit returns nonzero (typically 1) when the sign bit
        // is set; normalise to 0/1 for the i1 return.
        out(backend) << "  return (int8_t)(__builtin_signbit(a0) ? 1 : 0);\n";
      }
    };

    // ── §12.6 D.3 min / max ───────────────────────────────────────────
    //
    // C's `fmin` / `fmax` are *implementation-defined* on signed-zero
    // operand pairs, so we emit the tie-break explicitly to stay
    // bit-exact with WASM's `fN.min` / `fN.max` (IEEE 754-2008
    // minNum/maxNum: prefer -0 over +0 for fmin, +0 over -0 for fmax).

    class FminCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &) const override {
        out(backend) << "  return (a0 < a1) ? a0\n";
        out(backend) << "       : (a1 < a0) ? a1\n";
        out(backend) << "       : __builtin_signbit(a0) ? a0 : a1;\n";
      }
    };

    class FmaxCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &) const override {
        out(backend) << "  return (a0 > a1) ? a0\n";
        out(backend) << "       : (a1 > a0) ? a1\n";
        out(backend) << "       : __builtin_signbit(a0) ? a1 : a0;\n";
      }
    };

    // ── §12.6 D.2 classification predicates ───────────────────────────

    class IsNormalCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &) const override {
        out(backend) << "  return (int8_t)(__builtin_isnormal(a0) ? 1 : 0);\n";
      }
    };

    class IsSubnormalCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &) const override {
        // x is subnormal iff (x is finite) && (!isnormal(x)) && (x != 0).
        // Defensive against any future caller that bypasses the §2.9
        // finiteness check.
        out(backend) << "  int _norm = __builtin_isnormal(a0);\n";
        out(backend) << "  int _zero = (a0 == 0.0);\n";
        out(backend) << "  int _fin = __builtin_isfinite(a0);\n";
        out(backend) << "  return (int8_t)((_fin && !_norm && !_zero) ? 1 : 0);\n";
      }
    };

    class ToBitsCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &intr) const override {
        bool isF32 = std::get<FloatType>(intr.params[0].type->v).kind == FloatType::Kind::F32;
        if (isF32) {
          out(backend) << "  int32_t r; __builtin_memcpy(&r, &a0, 4); return r;\n";
        } else {
          out(backend) << "  int64_t r; __builtin_memcpy(&r, &a0, 8); return r;\n";
        }
      }
    };

    class FromBitsCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &intr) const override {
        bool isF32 = std::get<FloatType>(intr.retType->v).kind == FloatType::Kind::F32;
        if (isF32) {
          out(backend) << "  float r; __builtin_memcpy(&r, &a0, 4);\n";
          guardOut(backend) << "  if (!__builtin_isfinite(r)) __builtin_trap();\n";
          out(backend) << "  return r;\n";
        } else {
          out(backend) << "  double r; __builtin_memcpy(&r, &a0, 8);\n";
          guardOut(backend) << "  if (!__builtin_isfinite(r)) __builtin_trap();\n";
          out(backend) << "  return r;\n";
        }
      }
    };

    // ── §12.6 D.4 correctly-rounded math ──────────────────────────────────
    //
    // Unary fN → fN.  Each lowers to the matching GCC/Clang builtin at the
    // operand precision (the f32 helper takes a `float`, so `__builtin_sqrtf`
    // is correctly rounded at single precision).  @sqrt traps on a non-finite
    // result (strictly-negative input → NaN, UB under §2.9); the integral
    // rounders never leave the finite domain.

    class SqrtCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &intr) const override {
        bool isF32 = std::get<FloatType>(intr.retType->v).kind == FloatType::Kind::F32;
        out(backend) << "  " << (isF32 ? "float" : "double")
                     << " r = " << (isF32 ? "__builtin_sqrtf" : "__builtin_sqrt") << "(a0);\n";
        guardOut(backend) << "  if (!__builtin_isfinite(r)) __builtin_trap();\n";
        out(backend) << "  return r;\n";
      }
    };

    class FloorCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &intr) const override {
        bool isF32 = std::get<FloatType>(intr.retType->v).kind == FloatType::Kind::F32;
        out(backend) << "  return " << (isF32 ? "__builtin_floorf" : "__builtin_floor")
                     << "(a0);\n";
      }
    };

    class CeilCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &intr) const override {
        bool isF32 = std::get<FloatType>(intr.retType->v).kind == FloatType::Kind::F32;
        out(backend) << "  return " << (isF32 ? "__builtin_ceilf" : "__builtin_ceil") << "(a0);\n";
      }
    };

    class TruncCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &intr) const override {
        bool isF32 = std::get<FloatType>(intr.retType->v).kind == FloatType::Kind::F32;
        out(backend) << "  return " << (isF32 ? "__builtin_truncf" : "__builtin_trunc")
                     << "(a0);\n";
      }
    };

    // ── §12.6 D.5 compositions ─────────────────────────────────────────────
    //
    // @fract(x) = x - trunc(x) — no UB.  @recip(x) = 1/x traps on a
    // non-finite result (x = ±0.0 or overflow); the explicit isfinite check
    // is required because UBSan's `-fsanitize=undefined` does not catch
    // overflow-to-∞ (and `float-divide-by-zero` is off in the xval harness),
    // matching the C backend's treatment of the `/` operator.

    class FractCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &intr) const override {
        bool isF32 = std::get<FloatType>(intr.retType->v).kind == FloatType::Kind::F32;
        out(backend) << "  return a0 - " << (isF32 ? "__builtin_truncf" : "__builtin_trunc")
                     << "(a0);\n";
      }
    };

    class RecipCIntrinsic final : public CFpIntrinsic {
    public:
      void emit(CBackend &backend, const IntrinsicDecl &intr) const override {
        bool isF32 = std::get<FloatType>(intr.retType->v).kind == FloatType::Kind::F32;
        out(backend) << "  " << (isF32 ? "float" : "double") << " r = " << (isF32 ? "1.0f" : "1.0")
                     << " / a0;\n";
        guardOut(backend) << "  if (!__builtin_isfinite(r)) __builtin_trap();\n";
        out(backend) << "  return r;\n";
      }
    };

    // ── Checksum primitives ──────────────────────────────────────────────
    //
    // Self-contained lowerings for the @crc32_update / @check_chksum
    // intrinsics declared in include/analysis/intrinsics.hpp. Each helper
    // is bit-exact with its interpreter counterpart in
    // src/interp/intrinsics.cpp so symiri and the compiled C agree byte
    // for byte.

    class Crc32UpdateIntrinsic final : public CIntrinsic {
    public:
      // The lookup table is a function-local static initialised lazily on
      // the first call. Marking the helper noinline keeps the C compiler
      // from cloning it into every caller and propagating the table
      // contents back through constant folding, which is what makes the
      // checksum opaque to the optimizer.
      std::string linkageQualifier() const override { return "static __attribute__((noinline))"; }

      void emit(
          CBackend &backend, const IntrinsicDecl &intr, uint32_t /*N*/, uint32_t /*W*/,
          const std::string & /*sty*/, const std::string & /*uty*/
      ) const override {
        // val width drives both the per-helper byte-count and the C type
        // of the second argument. semchecker has already enforced that
        // params[1] is an integer.
        uint32_t valBits = paramN(intr, 1);
        uint32_t valW = widen(valBits);
        uint32_t nBytes = (valBits + 7) / 8;
        out(backend) << "  static uint32_t tab[256];\n";
        out(backend) << "  static int inited = 0;\n";
        out(backend) << "  if (!inited) {\n";
        out(backend) << "    const uint32_t poly = 0xEDB88320u;\n";
        out(backend) << "    for (uint32_t i = 0; i < 256u; ++i) {\n";
        out(backend) << "      uint32_t c = i;\n";
        out(
            backend
        ) << "      for (int j = 0; j < 8; ++j) c = (c & 1u) ? ((c >> 1) ^ poly) : (c >> 1);\n";
        out(backend) << "      tab[i] = c;\n";
        out(backend) << "    }\n";
        out(backend) << "    inited = 1;\n";
        out(backend) << "  }\n";
        out(backend) << "  uint32_t s = (uint32_t)a0;\n";
        out(backend) << "  uint" << valW << "_t v = (uint" << valW << "_t)a1;\n";
        if (valBits % 8 != 0 && valBits < valW) {
          // Zero-extend the iN value to whole bytes per the spec: bits
          // [valBits, 8*nBytes) of the last byte must read as 0. The
          // (uintW)a1 cast preserves a1's raw high bits (a negative iN's
          // sign extension), so mask them off to match the interpreter's
          // argUint(valBits) and keep the C / interpreter oracle in sync.
          uint64_t mask = (1ULL << valBits) - 1;
          out(backend) << "  v &= (uint" << valW << "_t)" << mask << "u;\n";
        }
        if (nBytes == 1) {
          out(backend) << "  s = (s >> 8) ^ tab[(s ^ (uint32_t)(v & 0xFFu)) & 0xFFu];\n";
        } else {
          out(backend) << "  for (uint32_t b = 0; b < " << nBytes << "u; ++b) {\n";
          out(backend) << "    uint8_t byte = (uint8_t)(v & 0xFFu);\n";
          out(backend) << "    v >>= 8;\n";
          out(backend) << "    s = (s >> 8) ^ tab[(s ^ (uint32_t)byte) & 0xFFu];\n";
          out(backend) << "  }\n";
        }
        out(backend) << "  return (int32_t)s;\n";
      }
    };

    class CheckChksumIntrinsic final : public CIntrinsic {
    public:
      // Inlining @check_chksum is fine — the body is small, but it's
      // anchored by an extern fprintf + abort call so even after inlining
      // the optimizer can't fold the comparison away. The visible side
      // effect of those two calls is what makes the whole checksum chain
      // observable.
      void emit(
          CBackend &backend, const IntrinsicDecl & /*intr*/, uint32_t /*N*/, uint32_t /*W*/,
          const std::string & /*sty*/, const std::string & /*uty*/
      ) const override {
        out(backend) << "  if (a0 != a1) {\n";
        out(
            backend
        ) << "    fprintf(stderr, \"@check_chksum mismatch: expected=%d actual=%d\\n\",\n";
        out(backend) << "            (int)a0, (int)a1);\n";
        out(backend) << "    abort();\n";
        out(backend) << "  }\n";
        out(backend) << "  return a1;\n";
      }
    };

    class ObserveCIntrinsic final : public CIntrinsic {
    public:
      // The volatile store is the observable side effect the optimizer must
      // preserve; noinline keeps a caller from folding through the identity
      // return and dropping it. Together they anchor the argument's computation
      // (e.g. the body of a diverging loop) so no pass can prove it dead.
      std::string linkageQualifier() const override { return "static __attribute__((noinline))"; }

      void emit(
          CBackend &backend, const IntrinsicDecl &, uint32_t /*N*/, uint32_t /*W*/,
          const std::string &sty, const std::string & /*uty*/
      ) const override {
        out(backend) << "  static volatile " << sty << " __rir_observe_sink;\n";
        out(backend) << "  __rir_observe_sink = a0;\n";
        out(backend) << "  return a0;\n";
      }
    };

  } // namespace

  // FP registry: separate from CIntrinsicRegistry because the emit
  // signature differs (no N/W/sty/uty).
  struct CFpIntrinsicRegistry {
    using CFpIntrinsicGenFn = std::unique_ptr<CFpIntrinsic>;

    static const std::unordered_map<IntrinsicKind, CFpIntrinsicGenFn> &getRegistry() {
      static const std::unordered_map<IntrinsicKind, CFpIntrinsicGenFn> registry = []() {
        std::unordered_map<IntrinsicKind, CFpIntrinsicGenFn> r;
        r[IntrinsicKind::Observe] = std::make_unique<ObserveCFpIntrinsic>();
        r[IntrinsicKind::Fabs] = std::make_unique<FabsCIntrinsic>();
        r[IntrinsicKind::Fneg] = std::make_unique<FnegCIntrinsic>();
        r[IntrinsicKind::Copysign] = std::make_unique<CopysignCIntrinsic>();
        r[IntrinsicKind::Signbit] = std::make_unique<SignbitCIntrinsic>();
        r[IntrinsicKind::ToBits] = std::make_unique<ToBitsCIntrinsic>();
        r[IntrinsicKind::FromBits] = std::make_unique<FromBitsCIntrinsic>();
        r[IntrinsicKind::IsNormal] = std::make_unique<IsNormalCIntrinsic>();
        r[IntrinsicKind::IsSubnormal] = std::make_unique<IsSubnormalCIntrinsic>();
        r[IntrinsicKind::Fmin] = std::make_unique<FminCIntrinsic>();
        r[IntrinsicKind::Fmax] = std::make_unique<FmaxCIntrinsic>();
        r[IntrinsicKind::Sqrt] = std::make_unique<SqrtCIntrinsic>();
        r[IntrinsicKind::Floor] = std::make_unique<FloorCIntrinsic>();
        r[IntrinsicKind::Ceil] = std::make_unique<CeilCIntrinsic>();
        r[IntrinsicKind::Trunc] = std::make_unique<TruncCIntrinsic>();
        r[IntrinsicKind::Fract] = std::make_unique<FractCIntrinsic>();
        r[IntrinsicKind::Recip] = std::make_unique<RecipCIntrinsic>();
        return r;
      }();
      return registry;
    }
  };

  const std::unordered_map<IntrinsicKind, CIntrinsicRegistry::CIntrinsicGenFn> &
  CIntrinsicRegistry::getRegistry() {
    static const std::unordered_map<IntrinsicKind, CIntrinsicGenFn> registry = []() {
      std::unordered_map<IntrinsicKind, CIntrinsicGenFn> r;
      r[IntrinsicKind::Abs] = std::make_unique<AbsIntrinsic>();
      r[IntrinsicKind::Min] = std::make_unique<MinIntrinsic>();
      r[IntrinsicKind::Max] = std::make_unique<MaxIntrinsic>();
      r[IntrinsicKind::Popcount] = std::make_unique<PopcountIntrinsic>();
      r[IntrinsicKind::Clz] = std::make_unique<ClzIntrinsic>();
      r[IntrinsicKind::Ctz] = std::make_unique<CtzIntrinsic>();
      r[IntrinsicKind::AbsDiff] = std::make_unique<AbsDiffIntrinsic>();
      r[IntrinsicKind::Signum] = std::make_unique<SignumIntrinsic>();
      r[IntrinsicKind::Clamp] = std::make_unique<ClampIntrinsic>();
      r[IntrinsicKind::Midpoint] = std::make_unique<MidpointIntrinsic>();
      r[IntrinsicKind::Parity] = std::make_unique<ParityIntrinsic>();
      r[IntrinsicKind::Bswap] = std::make_unique<BswapIntrinsic>();
      r[IntrinsicKind::Bitreverse] = std::make_unique<BitreverseIntrinsic>();
      r[IntrinsicKind::Rotl] = std::make_unique<RotlIntrinsic>();
      r[IntrinsicKind::Rotr] = std::make_unique<RotrIntrinsic>();
      r[IntrinsicKind::IsPow2] = std::make_unique<IsPow2Intrinsic>();
      r[IntrinsicKind::Ilog2] = std::make_unique<Ilog2Intrinsic>();
      // §12.5 — overflow-aware family.
      r[IntrinsicKind::WrappingAdd] = std::make_unique<WrappingAddIntrinsic>();
      r[IntrinsicKind::WrappingSub] = std::make_unique<WrappingSubIntrinsic>();
      r[IntrinsicKind::WrappingMul] = std::make_unique<WrappingMulIntrinsic>();
      r[IntrinsicKind::WrappingNeg] = std::make_unique<WrappingNegIntrinsic>();
      r[IntrinsicKind::WrappingShl] = std::make_unique<WrappingShlIntrinsic>();
      r[IntrinsicKind::WrappingShr] = std::make_unique<WrappingShrIntrinsic>();
      r[IntrinsicKind::SaturatingAdd] = std::make_unique<SaturatingAddIntrinsic>();
      r[IntrinsicKind::SaturatingSub] = std::make_unique<SaturatingSubIntrinsic>();
      r[IntrinsicKind::SaturatingMul] = std::make_unique<SaturatingMulIntrinsic>();
      r[IntrinsicKind::SaturatingNeg] = std::make_unique<SaturatingNegIntrinsic>();
      r[IntrinsicKind::DivEuclid] = std::make_unique<DivEuclidIntrinsic>();
      r[IntrinsicKind::RemEuclid] = std::make_unique<RemEuclidIntrinsic>();
      // Checksum primitives
      r[IntrinsicKind::Crc32Update] = std::make_unique<Crc32UpdateIntrinsic>();
      r[IntrinsicKind::CheckChksum] = std::make_unique<CheckChksumIntrinsic>();
      r[IntrinsicKind::Observe] = std::make_unique<ObserveCIntrinsic>();
      return r;
    }();
    return registry;
  }

  // ── naming ───────────────────────────────────────────────────────────────
  std::string CBackend::intrinsicHelperName(const std::string &intrName, uint32_t bits) const {
    std::string base = intrName;
    if (!base.empty() && base[0] == '@')
      base.erase(0, 1);
    return "_refractir_" + base + "_i" + std::to_string(bits);
  }

  // FP-aware overload: mangle FP-touching intrinsics by
  // their first operand's type so e.g. @fabs(f32) and @fabs(f64) get
  // distinct helper names, and so @to_bits(f32)→i32 doesn't collide with
  // @from_bits(i32)→f32 (both would map to *_i32 under the legacy rule).
  std::string CBackend::intrinsicHelperName(const IntrinsicDecl &intr) const {
    std::string base = intr.name.name;
    if (!base.empty() && base[0] == '@')
      base.erase(0, 1);
    // Reductions overload on vector shape (lane count +
    // element type), so mangle by both — `_refractir_reduce_add_v4_i32`,
    // `_refractir_reduce_add_v2_f64` — to keep distinct declarations from
    // colliding on one helper name.
    if (!intr.params.empty() && intr.params[0].type) {
      if (auto vt = std::get_if<VecType>(&intr.params[0].type->v)) {
        std::string el = "i32";
        if (auto ib = TypeUtils::getIntBitWidth(vt->elem))
          el = "i" + std::to_string(*ib);
        else if (auto fp = std::get_if<FloatType>(&vt->elem->v))
          el = fp->kind == FloatType::Kind::F32 ? "f32" : "f64";
        return "_refractir_" + base + "_v" + std::to_string(vt->size) + "_" + el;
      }
    }
    bool paramFp = !intr.params.empty() && intr.params[0].type &&
                   std::holds_alternative<FloatType>(intr.params[0].type->v);
    bool retFp = intr.retType && std::holds_alternative<FloatType>(intr.retType->v);
    if (paramFp || retFp) {
      auto t = !intr.params.empty() ? intr.params[0].type : intr.retType;
      if (auto fp = std::get_if<FloatType>(&t->v))
        return "_refractir_" + base + "_f" + (fp->kind == FloatType::Kind::F32 ? "32" : "64");
      if (auto it = std::get_if<IntType>(&t->v)) {
        uint32_t b = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
        return "_refractir_" + base + "_i" + std::to_string(b);
      }
    }
    // For integer intrinsics whose parameter widths differ from the
    // return width, mangle by the first differing parameter's width
    // instead of the return — otherwise the per-arg width overloads of
    // a single intrinsic would all collapse onto one helper name.
    auto rb = TypeUtils::getIntBitWidth(intr.retType);
    for (const auto &p: intr.params) {
      auto pb = TypeUtils::getIntBitWidth(p.type);
      if (pb && rb && *pb != *rb)
        return "_refractir_" + base + "_i" + std::to_string(*pb);
    }
    return "_refractir_" + base + "_i" + std::to_string(rb.value_or(32));
  }

  // ── emission ─────────────────────────────────────────────────────────────

  // Return the C type-name corresponding to a RefractIR scalar type:
  // - IntType iN → "int<W>_t" where W is the smallest machine width fitting N
  // - FloatType f32/f64 → "float"/"double"
  // Used by the helper-signature emitter for batches that mix int and FP.
  static std::string cTypeOf(const TypePtr &t) {
    if (auto fp = std::get_if<FloatType>(&t->v))
      return fp->kind == FloatType::Kind::F32 ? "float" : "double";
    if (auto it = std::get_if<IntType>(&t->v)) {
      uint32_t b = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
      uint32_t W = (b <= 8) ? 8 : (b <= 16) ? 16 : (b <= 32) ? 32 : 64;
      return "int" + std::to_string(W) + "_t";
    }
    return "void";
  }

  void CBackend::emitIntrinsicHelper(const IntrinsicDecl &intr) {
    // Horizontal vector reductions (§12.4) have a vector
    // parameter, so they lower to a dedicated helper rather than the
    // scalar widening-and-mask path below.
    if (auto k = getIntrinsicKind(intr.name.name); k && isReductionIntrinsic(*k)) {
      emitReductionHelper(intr);
      return;
    }
    // Detect FP-touching intrinsics: dispatched to the FP registry instead
    // of the legacy integer-only one. Helper-signature param types follow
    // the declared RefractIR types directly (no widening for FP).
    bool anyFp = (intr.retType && std::holds_alternative<FloatType>(intr.retType->v));
    for (const auto &p: intr.params)
      anyFp = anyFp || (p.type && std::holds_alternative<FloatType>(p.type->v));

    if (anyFp) {
      // Look up the impl before the signature so its linkage qualifier can be
      // read, exactly as the integer path below does.
      const CFpIntrinsic *fpImpl = nullptr;
      if (auto kind = getIntrinsicKind(intr.name.name)) {
        const auto &fpRegistry = CFpIntrinsicRegistry::getRegistry();
        auto it = fpRegistry.find(*kind);
        if (it != fpRegistry.end())
          fpImpl = it->second.get();
      }
      std::string outTy = cTypeOf(intr.retType);
      std::string name = intrinsicHelperName(intr);
      out_ << (fpImpl ? fpImpl->linkageQualifier() : "static inline") << " " << outTy << " " << name
           << "(";
      for (size_t i = 0; i < intr.params.size(); ++i) {
        if (i)
          out_ << ", ";
        out_ << cTypeOf(intr.params[i].type) << " a" << i;
      }
      out_ << ") {\n";
      if (fpImpl) {
        fpImpl->emit(*this, intr);
        out_ << "}\n\n";
        return;
      }
      out_ << "  __builtin_trap(); /* unknown FP intrinsic */\n";
      out_ << "}\n\n";
      return;
    }

    // Legacy integer-only path (batches A/B/C and §12.1-12.2).
    auto rb = TypeUtils::getIntBitWidth(intr.retType);
    if (!rb)
      return;
    uint32_t N = *rb;
    uint32_t W = (N <= 8) ? 8 : (N <= 16) ? 16 : (N <= 32) ? 32 : 64;
    std::string sty = "int" + std::to_string(W) + "_t";
    std::string uty = "uint" + std::to_string(W) + "_t";
    std::string outTy = sty;
    std::string name = intrinsicHelperName(intr);

    // Look up the impl first so we can read its linkage qualifier before
    // emitting the helper signature. Falls back to `static inline` for
    // unregistered kinds (which then immediately trap below).
    const CIntrinsic *impl = nullptr;
    auto kind = getIntrinsicKind(intr.name.name);
    if (kind) {
      const auto &registry = CIntrinsicRegistry::getRegistry();
      auto it = registry.find(*kind);
      if (it != registry.end())
        impl = it->second.get();
    }
    std::string linkage = impl ? impl->linkageQualifier() : "static inline";

    out_ << linkage << " " << outTy << " " << name << "(";
    for (size_t i = 0; i < intr.params.size(); ++i) {
      if (i)
        out_ << ", ";
      uint32_t pN = 32;
      if (auto pb = TypeUtils::getIntBitWidth(intr.params[i].type))
        pN = *pb;
      uint32_t pW = (pN <= 8) ? 8 : (pN <= 16) ? 16 : (pN <= 32) ? 32 : 64;
      out_ << "int" << pW << "_t a" << i;
    }
    out_ << ") {\n";

    if (impl) {
      impl->emit(*this, intr, N, W, sty, uty);
      out_ << "}\n\n";
      return;
    }

    out_ << "  __builtin_trap(); /* unknown intrinsic */\n";
    out_ << "}\n\n";
  }

  // Horizontal vector reduction helper (§12.4). Emits a
  // `static inline T H(<vec> a0) { ... }` that folds the lanes left-to-right
  // into one scalar, reproducing the interpreter's per-step UB exactly:
  // @reduce_add traps on an out-of-range partial sum (int) or a non-finite
  // intermediate (fp); min/max use the @fmin/@fmax signed-zero tie-break for
  // fp; the bitwise members never trap. Lanes are read through the
  // vec-lowering strategy, so the same body serves every boundary-crossing
  // representation (vecext / structscalars / structarray). UB guard lines go
  // to a sink that vanishes under --no-ub-guards.
  void CBackend::emitReductionHelper(const IntrinsicDecl &intr) {
    const IntrinsicKind kind = *getIntrinsicKind(intr.name.name);
    const VecType &vt = std::get<VecType>(intr.params[0].type->v);
    const uint32_t M = static_cast<uint32_t>(vt.size);
    const TypePtr &elem = vt.elem;

    // The helper takes the N lanes as scalar parameters (the call site emits
    // them per-lane). This is strategy-independent: it needs no vector to
    // cross the C function boundary, so it works under every --vec-lowering
    // strategy, including `scalars` and `array`.
    const std::string retTy = cTypeOf(elem);
    out_ << "static inline " << retTy << " " << intrinsicHelperName(intr) << "(";
    for (uint32_t i = 0; i < M; ++i) {
      if (i)
        out_ << ", ";
      out_ << retTy << " a" << i;
    }
    out_ << ") {\n";

    auto lane = [&](uint32_t k) { return "a" + std::to_string(k); };
    auto guard = [&](const std::string &s) {
      if (!noUbGuards_)
        out_ << s;
    };

    if (auto ib = TypeUtils::getIntBitWidth(elem)) {
      const uint32_t N = *ib;
      const uint32_t W = (N <= 8) ? 8 : (N <= 16) ? 16 : (N <= 32) ? 32 : 64;
      const std::string sty = "int" + std::to_string(W) + "_t";
      auto intMaxN = [](uint32_t n) {
        return (n == 64) ? std::string("9223372036854775807LL")
                         : std::to_string((INT64_C(1) << (n - 1)) - 1) + "LL";
      };
      auto intMinN = [](uint32_t n) {
        return (n == 64) ? std::string("(-9223372036854775807LL - 1)")
                         : std::to_string(-(INT64_C(1) << (n - 1))) + "LL";
      };
      switch (kind) {
        case IntrinsicKind::ReduceAdd: {
          // A partial sum stays in [-2^(N-1), 2^(N-1)-1] between steps, so a
          // wide accumulator (int64 for N<=63, __int128 for N==64) cannot
          // overflow before the per-step range check fires.
          const std::string acc = (N < 64) ? "long long" : "__int128";
          const std::string minE = (N < 64) ? intMinN(N) : "(__int128)(-9223372036854775807LL - 1)";
          const std::string maxE = (N < 64) ? intMaxN(N) : "(__int128)9223372036854775807LL";
          out_ << "  " << acc << " acc = (" << acc << ")(" << lane(0) << ");\n";
          for (uint32_t k = 1; k < M; ++k) {
            out_ << "  acc += (" << acc << ")(" << lane(k) << ");";
            guard(" if (acc < " + minE + " || acc > " + maxE + ") __builtin_trap();");
            out_ << "\n";
          }
          out_ << "  return (" << retTy << ")acc;\n";
          break;
        }
        case IntrinsicKind::ReduceMin:
        case IntrinsicKind::ReduceMax: {
          const char *cmp = (kind == IntrinsicKind::ReduceMin) ? "<" : ">";
          out_ << "  " << sty << " acc = (" << sty << ")(" << lane(0) << ");\n";
          for (uint32_t k = 1; k < M; ++k)
            out_ << "  { " << sty << " x = (" << sty << ")(" << lane(k) << "); acc = (x " << cmp
                 << " acc) ? x : acc; }\n";
          out_ << "  return (" << retTy << ")acc;\n";
          break;
        }
        case IntrinsicKind::ReduceAnd:
        case IntrinsicKind::ReduceOr:
        case IntrinsicKind::ReduceXor: {
          const char *op = (kind == IntrinsicKind::ReduceAnd)  ? "&"
                           : (kind == IntrinsicKind::ReduceOr) ? "|"
                                                               : "^";
          out_ << "  " << sty << " acc = (" << sty << ")(" << lane(0) << ");\n";
          for (uint32_t k = 1; k < M; ++k)
            out_ << "  acc " << op << "= (" << sty << ")(" << lane(k) << ");\n";
          // Bitwise on sign-extended lanes stays sign-extended, so a plain
          // cast to the storage type is a valid iN representation.
          out_ << "  return (" << retTy << ")acc;\n";
          break;
        }
        default:
          throw std::runtime_error("C reduction: unexpected integer kind");
      }
    } else {
      // Floating-point element (f32 / f64).
      out_ << "  " << retTy << " acc = " << lane(0) << ";\n";
      for (uint32_t k = 1; k < M; ++k) {
        if (kind == IntrinsicKind::ReduceAdd) {
          out_ << "  acc = acc + (" << lane(k) << ");";
          guard(" if (!__builtin_isfinite(acc)) __builtin_trap();");
          out_ << "\n";
        } else {
          // fmin / fmax fold with the IEEE minNum/maxNum signed-zero
          // tie-break, identical to @fmin / @fmax.
          const bool isMin = (kind == IntrinsicKind::ReduceMin);
          const char *cmp = isMin ? "<" : ">";
          const char *tie = isMin ? "acc : x" : "x : acc"; // signbit(acc) ? ...
          out_ << "  { " << retTy << " x = " << lane(k) << "; acc = (acc " << cmp
               << " x) ? acc : (x " << cmp << " acc) ? x : (__builtin_signbit(acc) ? " << tie
               << "); }\n";
        }
      }
      out_ << "  return acc;\n";
    }
    out_ << "}\n\n";
  }

} // namespace refractir
