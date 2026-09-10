// WASM backend intrinsic helper emission.
//
// This file is the single source of truth for the WebAssembly Text Format
// (WAT) code emitted for every built-in RefractIR intrinsic. The helpers use
// a widening-and-mask strategy: each iN operation is widened to the next
// WASM native width (i32 or i64), performed there, then sign-masked back
// to N bits. UB-preconditions abort via `unreachable`.
//
// Width conventions mirror intrinsics_c.cpp:
//   - Return-side `N`/`W`/`ity` are the return type's widening.
//   - For predicate intrinsics (@parity, @is_pow2), return is i1 (W=32,
//     ity=i32) while the input is bvN with its own widening. Per-param
//     types are emitted in the helper signature by emitIntrinsicHelper;
//     impls that touch param widths read them via `intr`.
//
// RefractIR has no unsigned integer types. This file uses only:
//   - signed comparisons at the RefractIR level (iW.lt_s / iW.gt_s / iW.eq...)
//   - bit-level operations (iW.shl, iW.shr_u, iW.and, iW.or, iW.xor,
//     iW.popcnt, iW.clz, iW.ctz) which act on bit patterns, not as
//     unsigned-integer numerics.
//   - widening between i32 ↔ i64 only where required by missing native
//     ops at narrow widths (analogous to "no i3 exists, must use i8"
//     in the C backend).
//
// To add a new intrinsic:
//   1. Add its IntrinsicKind to include/analysis/intrinsics.hpp.
//   2. Implement a subclass of WasmIntrinsic and register it below.
//
// Note: float literal format intentionally diverges from refractir::formatDouble.
// WAT has its own grammar rules. See AGENTS.md §FP-serialization-invariant.

#include "backend/wasm_backend.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "analysis/intrinsics.hpp"
#include "analysis/type_utils.hpp"

namespace refractir {

  class WasmIntrinsic {
  public:
    virtual ~WasmIntrinsic() = default;

    virtual void emit(
        WasmBackend &backend, const IntrinsicDecl &intr, uint32_t N, uint32_t W,
        const std::string &ity
    ) const = 0;

    /**
     * @brief Optional: declares scratch (`local`) variables used by this impl.
     */
    virtual void declareLocals(
        WasmBackend &, const IntrinsicDecl &, uint32_t, uint32_t, const std::string &
    ) const {}

  protected:
    static std::ostream &out(WasmBackend &backend);
    static bool noUbGuards(WasmBackend &backend);
    static void indent(WasmBackend &backend);
    static void incrIndent(WasmBackend &backend);
    static void decrIndent(WasmBackend &backend);

    static uint32_t widen(uint32_t N) { return (N <= 32) ? 32 : 64; }

    static std::string ityOf(uint32_t W) { return (W == 32) ? "i32" : "i64"; }

    static uint32_t paramN(const IntrinsicDecl &intr, size_t i) {
      auto pb = TypeUtils::getIntBitWidth(intr.params[i].type);
      return pb ? *pb : 32;
    }

    static void pushArg(WasmBackend &backend, size_t i) {
      indent(backend);
      out(backend) << "local.get $a" << i << "\n";
    }

    /**
     * @brief Sign-extend the top of stack so its low N bits encode a signed
     * iN value, padded out to W bits.
     */
    static void sextN(WasmBackend &backend, uint32_t N, uint32_t W, const std::string &ity) {
      if (N == W)
        return;
      indent(backend);
      out(backend) << ity << ".const " << (W - N) << "\n";
      indent(backend);
      out(backend) << ity << ".shl\n";
      indent(backend);
      out(backend) << ity << ".const " << (W - N) << "\n";
      indent(backend);
      out(backend) << ity << ".shr_s\n";
    }

    /**
     * @brief Push an N-bit mask onto the stack (-1 if N == W, else (1 << N) - 1).
     */
    static void pushMask(WasmBackend &backend, uint32_t N, uint32_t W, const std::string &ity) {
      if (N == W) {
        indent(backend);
        out(backend) << ity << ".const -1\n";
      } else {
        indent(backend);
        out(backend) << ity << ".const " << ((uint64_t(1) << N) - 1) << "\n";
      }
    }

    static void declLocal(WasmBackend &backend, const std::string &name, const std::string &ity) {
      indent(backend);
      out(backend) << "(local $" << name << " " << ity << ")\n";
    }

    static void unreachableIfTop(WasmBackend &backend) {
      // --no-ub-guards: the UB precondition is already on the
      // stack; discard it (no trap) instead of trapping. The guard's
      // condition computation is left as (dead) stack-neutral code.
      if (noUbGuards(backend)) {
        indent(backend);
        out(backend) << "drop\n";
        return;
      }
      trapIfTop(backend);
    }

    // Unconditional twin of unreachableIfTop for non-UB divergence
    // (@check_chksum mismatch): --no-ub-guards must not elide this trap.
    static void trapIfTop(WasmBackend &backend) {
      indent(backend);
      out(backend) << "if\n";
      incrIndent(backend);
      indent(backend);
      out(backend) << "unreachable\n";
      decrIndent(backend);
      indent(backend);
      out(backend) << "end\n";
    }
  };

  struct WasmIntrinsicRegistry {
    static std::ostream &out(WasmBackend &backend) { return backend.out_; }

    static bool noUbGuards(WasmBackend &backend) { return backend.noUbGuards_; }

    static void indent(WasmBackend &backend) { backend.indent(); }

    static void incrIndent(WasmBackend &backend) { backend.indent_level_++; }

    static void decrIndent(WasmBackend &backend) { backend.indent_level_--; }

    using WasmIntrinsicGenFn = std::unique_ptr<WasmIntrinsic>;
    static const std::unordered_map<IntrinsicKind, WasmIntrinsicGenFn> &getRegistry();
  };

  bool WasmIntrinsic::noUbGuards(WasmBackend &backend) {
    return WasmIntrinsicRegistry::noUbGuards(backend);
  }

  std::ostream &WasmIntrinsic::out(WasmBackend &backend) {
    return WasmIntrinsicRegistry::out(backend);
  }

  void WasmIntrinsic::indent(WasmBackend &backend) { WasmIntrinsicRegistry::indent(backend); }

  void WasmIntrinsic::incrIndent(WasmBackend &backend) {
    WasmIntrinsicRegistry::incrIndent(backend);
  }

  void WasmIntrinsic::decrIndent(WasmBackend &backend) {
    WasmIntrinsicRegistry::decrIndent(backend);
  }

  namespace {

    // ── §12.1 Arithmetic intrinsics ─────────────────────────────────────────

    class AbsIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        int64_t int_min_N = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        pushArg(backend, 0);
        indent(backend);
        out(backend) << ity << ".const " << int_min_N << "\n";
        indent(backend);
        out(backend) << ity << ".eq\n";
        unreachableIfTop(backend);

        indent(backend);
        out(backend) << ity << ".const 0\n";
        pushArg(backend, 0);
        indent(backend);
        out(backend) << ity << ".sub\n";
        pushArg(backend, 0);
        pushArg(backend, 0);
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << ity << ".lt_s\n";
        indent(backend);
        out(backend) << "select\n";
        sextN(backend, N, W, ity);
      }
    };

    // Observability beacon — identity. WASM has no forward-progress assumption,
    // so a diverging loop is preserved without an observable side effect; the
    // helper just returns its argument.
    class ObserveIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t, uint32_t, const std::string &
      ) const override {
        pushArg(backend, 0);
      }
    };

    class MinIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        pushArg(backend, 0);
        pushArg(backend, 1);
        pushArg(backend, 0);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".lt_s\n";
        indent(backend);
        out(backend) << "select\n";
        sextN(backend, N, W, ity);
      }
    };

    class MaxIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        pushArg(backend, 0);
        pushArg(backend, 1);
        pushArg(backend, 0);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".gt_s\n";
        indent(backend);
        out(backend) << "select\n";
        sextN(backend, N, W, ity);
      }
    };

    // ── §12.2 Bit-counting intrinsics ───────────────────────────────────────

    class PopcountIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        pushArg(backend, 0);
        pushMask(backend, N, W, ity);
        indent(backend);
        out(backend) << ity << ".and\n";
        indent(backend);
        out(backend) << ity << ".popcnt\n";
        sextN(backend, N, W, ity);
      }
    };

    class ClzIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t, uint32_t, const std::string &ity
      ) const override {
        declLocal(b, "tmp0", ity);
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        pushArg(backend, 0);
        pushMask(backend, N, W, ity);
        indent(backend);
        out(backend) << ity << ".and\n";
        indent(backend);
        out(backend) << "local.tee $tmp0\n";
        indent(backend);
        out(backend) << ity << ".eqz\n";
        unreachableIfTop(backend);
        indent(backend);
        out(backend) << "local.get $tmp0\n";
        indent(backend);
        out(backend) << ity << ".clz\n";
        if (N != W) {
          indent(backend);
          out(backend) << ity << ".const " << (W - N) << "\n";
          indent(backend);
          out(backend) << ity << ".sub\n";
        }
        sextN(backend, N, W, ity);
      }
    };

    class CtzIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t, uint32_t, const std::string &ity
      ) const override {
        declLocal(b, "tmp0", ity);
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        pushArg(backend, 0);
        pushMask(backend, N, W, ity);
        indent(backend);
        out(backend) << ity << ".and\n";
        indent(backend);
        out(backend) << "local.tee $tmp0\n";
        indent(backend);
        out(backend) << ity << ".eqz\n";
        unreachableIfTop(backend);
        indent(backend);
        out(backend) << "local.get $tmp0\n";
        indent(backend);
        out(backend) << ity << ".ctz\n";
        sextN(backend, N, W, ity);
      }
    };

    // ── §12.3 Integer extras ────────────────────────────────

    /**
     * @brief @abs_diff(a, b) = |a - b|, UB if not representable in iN.
     * For N <= 32, widen both args to i64, subtract safely, then check
     * |s| > INT_MAX_N, then narrow. For N == 64, detect signed-iW overflow
     * directly without any unsigned reinterpretation.
     */
    class AbsDiffIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &ity
      ) const override {
        if (N <= 32) {
          declLocal(b, "s64", "i64");
          declLocal(b, "r64", "i64");
        } else {
          declLocal(b, "s", ity);
          declLocal(b, "r", ity);
        }
        (void) W;
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        int64_t maxN = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        if (N <= 32) {
          // i64-widened subtraction; no overflow possible since |i32 - i32| <= 2^32 - 1.
          pushArg(backend, 0);
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
          pushArg(backend, 1);
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
          indent(backend);
          out(backend) << "i64.sub\n";
          indent(backend);
          out(backend) << "local.set $s64\n";
          // r64 = |s64| via select
          indent(backend);
          out(backend) << "i64.const 0\n";
          indent(backend);
          out(backend) << "local.get $s64\n";
          indent(backend);
          out(backend) << "i64.sub\n";
          indent(backend);
          out(backend) << "local.get $s64\n";
          indent(backend);
          out(backend) << "local.get $s64\n";
          indent(backend);
          out(backend) << "i64.const 0\n";
          indent(backend);
          out(backend) << "i64.lt_s\n";
          indent(backend);
          out(backend) << "select\n";
          indent(backend);
          out(backend) << "local.tee $r64\n";
          indent(backend);
          out(backend) << "i64.const " << maxN << "\n";
          indent(backend);
          out(backend) << "i64.gt_s\n";
          unreachableIfTop(backend);
          // narrow & sext
          indent(backend);
          out(backend) << "local.get $r64\n";
          indent(backend);
          out(backend) << "i32.wrap_i64\n";
          sextN(backend, N, W, ity);
        } else {
          // N == 64 (so W = 64). Detect overflow via the two's-complement
          // identity: a - b overflows signed iff (a XOR b) has the high bit
          // set AND (a XOR s) has the high bit set, where s = a - b.
          pushArg(backend, 0);
          pushArg(backend, 1);
          indent(backend);
          out(backend) << ity << ".sub\n";
          indent(backend);
          out(backend) << "local.set $s\n";
          // overflow test
          pushArg(backend, 0);
          pushArg(backend, 1);
          indent(backend);
          out(backend) << ity << ".xor\n";
          pushArg(backend, 0);
          indent(backend);
          out(backend) << "local.get $s\n";
          indent(backend);
          out(backend) << ity << ".xor\n";
          indent(backend);
          out(backend) << ity << ".and\n";
          indent(backend);
          out(backend) << ity << ".const 0\n";
          indent(backend);
          out(backend) << ity << ".lt_s\n";
          unreachableIfTop(backend);
          // r = |s| via select
          indent(backend);
          out(backend) << ity << ".const 0\n";
          indent(backend);
          out(backend) << "local.get $s\n";
          indent(backend);
          out(backend) << ity << ".sub\n";
          indent(backend);
          out(backend) << "local.get $s\n";
          indent(backend);
          out(backend) << "local.get $s\n";
          indent(backend);
          out(backend) << ity << ".const 0\n";
          indent(backend);
          out(backend) << ity << ".lt_s\n";
          indent(backend);
          out(backend) << "select\n";
          indent(backend);
          out(backend) << "local.tee $r\n";
          // r is in [0, INT_MAX_W=INT_MAX_64]. For N=W=64, no further check
          // required. Just return.
          indent(backend);
          out(backend) << "drop\n";
          indent(backend);
          out(backend) << "local.get $r\n";
        }
      }
    };

    class SignumIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        // r = (x > 0) - (x < 0)
        // Comparison operators in WASM always return i32.
        pushArg(backend, 0);
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << ity << ".gt_s\n";
        pushArg(backend, 0);
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << ity << ".lt_s\n";
        indent(backend);
        out(backend) << "i32.sub\n";
        if (W > 32) {
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
        }
        // @parity returns i1 (W=32); sextN(1, 32) yields 0/-1 for true.
        sextN(backend, N, W, ity);
      }
    };

    class ClampIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        // UB if lo > hi
        pushArg(backend, 1);
        pushArg(backend, 2);
        indent(backend);
        out(backend) << ity << ".gt_s\n";
        unreachableIfTop(backend);
        // r = (v < lo) ? lo : ((v > hi) ? hi : v)
        pushArg(backend, 1); // lo
        // compute upper = (v > hi) ? hi : v on top
        pushArg(backend, 2); // hi
        pushArg(backend, 0); // v
        pushArg(backend, 0);
        pushArg(backend, 2);
        indent(backend);
        out(backend) << ity << ".gt_s\n";
        indent(backend);
        out(backend) << "select\n"; // upper
        pushArg(backend, 0);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".lt_s\n";
        indent(backend);
        out(backend) << "select\n"; // r
        sextN(backend, N, W, ity);
      }
    };

    class MidpointIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t N, uint32_t, const std::string &
      ) const override {
        if (N == 64)
          declLocal(b, "tmp64", "i64");
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        if (N <= 32) {
          // i64-widened add and signed division.
          pushArg(backend, 0);
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
          pushArg(backend, 1);
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
          indent(backend);
          out(backend) << "i64.add\n";
          indent(backend);
          out(backend) << "i64.const 2\n";
          indent(backend);
          out(backend) << "i64.div_s\n";
          indent(backend);
          out(backend) << "i32.wrap_i64\n";
          sextN(backend, N, W, ity);
        } else if (N <= 63) {
          // 33..63 — sum fits in i64.
          pushArg(backend, 0);
          pushArg(backend, 1);
          indent(backend);
          out(backend) << "i64.add\n";
          indent(backend);
          out(backend) << "i64.const 2\n";
          indent(backend);
          out(backend) << "i64.div_s\n";
          sextN(backend, N, W, ity);
        } else {
          // N == 64: bit-level midpoint truncating toward zero:
          //   floor_avg = (a & b) + ((a ^ b) >> 1)  ;; ashr (signed) for floor
          //   adjust by +1 if ((a ^ b) & 1) AND floor_avg < 0  (truncate ⇒ 0).
          pushArg(backend, 0);
          pushArg(backend, 1);
          indent(backend);
          out(backend) << "i64.and\n";
          pushArg(backend, 0);
          pushArg(backend, 1);
          indent(backend);
          out(backend) << "i64.xor\n";
          indent(backend);
          out(backend) << "i64.const 1\n";
          indent(backend);
          out(backend) << "i64.shr_s\n";
          indent(backend);
          out(backend) << "i64.add\n";
          indent(backend);
          out(backend) << "local.set $tmp64\n";
          // adjust: if ((a ^ b) & 1) and tmp64 < 0, tmp64 += 1
          pushArg(backend, 0);
          pushArg(backend, 1);
          indent(backend);
          out(backend) << "i64.xor\n";
          indent(backend);
          out(backend) << "i64.const 1\n";
          indent(backend);
          out(backend) << "i64.and\n";
          indent(backend);
          out(backend) << "local.get $tmp64\n";
          indent(backend);
          out(backend) << "i64.const 0\n";
          indent(backend);
          out(backend) << "i64.lt_s\n";
          indent(backend);
          out(backend) << "i64.extend_i32_u\n";
          indent(backend);
          out(backend) << "i64.and\n";
          indent(backend);
          out(backend) << "local.get $tmp64\n";
          indent(backend);
          out(backend) << "i64.add\n";
          // already i64; W == 64, no sextN.
        }
      }
    };

    // ── §12.4 Bit-manipulation ──────────────────────────────

    class ParityIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &intr, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        (void) N;
        uint32_t pN = paramN(intr, 0);
        uint32_t pW = widen(pN);
        std::string pity = ityOf(pW);
        pushArg(backend, 0);
        // mask to param-N bits (bit-level mask, not unsigned interpretation)
        if (pN < pW) {
          indent(backend);
          out(backend) << pity << ".const " << ((uint64_t(1) << pN) - 1) << "\n";
          indent(backend);
          out(backend) << pity << ".and\n";
        }
        indent(backend);
        out(backend) << pity << ".popcnt\n";
        // popcnt result is bit-level; narrow/widen to ret-ity if widths differ.
        if (pW == 64 && W == 32) {
          indent(backend);
          out(backend) << "i32.wrap_i64\n";
        } else if (pW == 32 && W == 64) {
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
        }
        indent(backend);
        out(backend) << ity << ".const 1\n";
        indent(backend);
        out(backend) << ity << ".and\n";
        // No sextN: RefractIR i1 is stored as the literal 0 / 1 (matches `cmp`),
        // not sign-extended.
      }
    };

    class BswapIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t, uint32_t, const std::string &ity
      ) const override {
        declLocal(b, "u", ity);
        declLocal(b, "r", ity);
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        pushArg(backend, 0);
        if (N < W) {
          pushMask(backend, N, W, ity);
          indent(backend);
          out(backend) << ity << ".and\n";
        }
        indent(backend);
        out(backend) << "local.set $u\n";
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << "local.set $r\n";
        uint32_t nbytes = N / 8;
        for (uint32_t i = 0; i < nbytes; ++i) {
          indent(backend);
          out(backend) << "local.get $u\n";
          indent(backend);
          out(backend) << ity << ".const " << (i * 8) << "\n";
          indent(backend);
          out(backend) << ity << ".shr_u\n";
          indent(backend);
          out(backend) << ity << ".const 255\n";
          indent(backend);
          out(backend) << ity << ".and\n";
          indent(backend);
          out(backend) << ity << ".const " << ((nbytes - 1 - i) * 8) << "\n";
          indent(backend);
          out(backend) << ity << ".shl\n";
          indent(backend);
          out(backend) << "local.get $r\n";
          indent(backend);
          out(backend) << ity << ".or\n";
          indent(backend);
          out(backend) << "local.set $r\n";
        }
        indent(backend);
        out(backend) << "local.get $r\n";
        sextN(backend, N, W, ity);
      }
    };

    class BitreverseIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t, uint32_t, const std::string &ity
      ) const override {
        declLocal(b, "u", ity);
        declLocal(b, "r", ity);
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        pushArg(backend, 0);
        if (N < W) {
          pushMask(backend, N, W, ity);
          indent(backend);
          out(backend) << ity << ".and\n";
        }
        indent(backend);
        out(backend) << "local.set $u\n";
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << "local.set $r\n";
        for (uint32_t i = 0; i < N; ++i) {
          indent(backend);
          out(backend) << "local.get $u\n";
          indent(backend);
          out(backend) << ity << ".const " << i << "\n";
          indent(backend);
          out(backend) << ity << ".shr_u\n";
          indent(backend);
          out(backend) << ity << ".const 1\n";
          indent(backend);
          out(backend) << ity << ".and\n";
          indent(backend);
          out(backend) << ity << ".const " << (N - 1 - i) << "\n";
          indent(backend);
          out(backend) << ity << ".shl\n";
          indent(backend);
          out(backend) << "local.get $r\n";
          indent(backend);
          out(backend) << ity << ".or\n";
          indent(backend);
          out(backend) << "local.set $r\n";
        }
        indent(backend);
        out(backend) << "local.get $r\n";
        sextN(backend, N, W, ity);
      }
    };

    /**
     * @brief Emit @rotl or @rotr body. Rotation is a bit-permutation; we use
     * shr_u as the bit-level shift-without-sign-propagation primitive (it is
     * not an unsigned-numeric interpretation).
     */
    static void
    emitRotation(WasmBackend &b, bool leftRot, uint32_t N, uint32_t W, const std::string &ity) {
      // UB: a1 < 0 || a1 >= N. Stack-neutral, so elided under
      // --no-ub-guards.
      if (!WasmIntrinsicRegistry::noUbGuards(b)) {
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << "local.get $a1\n";
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << ity << ".const 0\n";
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << ity << ".lt_s\n";
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << "local.get $a1\n";
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << ity << ".const " << N << "\n";
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << ity << ".ge_s\n";
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << "i32.or\n";
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << "if\n";
        WasmIntrinsicRegistry::incrIndent(b);
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << "unreachable\n";
        WasmIntrinsicRegistry::decrIndent(b);
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << "end\n";
      }

      // u = a0 [& mask if N < W]
      WasmIntrinsicRegistry::indent(b);
      WasmIntrinsicRegistry::out(b) << "local.get $a0\n";
      if (N < W) {
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << ity << ".const " << ((uint64_t(1) << N) - 1) << "\n";
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << ity << ".and\n";
      }
      WasmIntrinsicRegistry::indent(b);
      WasmIntrinsicRegistry::out(b) << "local.set $u\n";

      // left_part = u (shl|shr_u) n  [&mask if shl and N<W]
      WasmIntrinsicRegistry::indent(b);
      WasmIntrinsicRegistry::out(b) << "local.get $u\n";
      WasmIntrinsicRegistry::indent(b);
      WasmIntrinsicRegistry::out(b) << "local.get $a1\n";
      WasmIntrinsicRegistry::indent(b);
      WasmIntrinsicRegistry::out(b) << ity << (leftRot ? ".shl\n" : ".shr_u\n");
      if (leftRot && N < W) {
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << ity << ".const " << ((uint64_t(1) << N) - 1) << "\n";
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << ity << ".and\n";
      }
      // right_part = u (shr_u|shl) (N - n)
      WasmIntrinsicRegistry::indent(b);
      WasmIntrinsicRegistry::out(b) << "local.get $u\n";
      WasmIntrinsicRegistry::indent(b);
      WasmIntrinsicRegistry::out(b) << ity << ".const " << N << "\n";
      WasmIntrinsicRegistry::indent(b);
      WasmIntrinsicRegistry::out(b) << "local.get $a1\n";
      WasmIntrinsicRegistry::indent(b);
      WasmIntrinsicRegistry::out(b) << ity << ".sub\n";
      WasmIntrinsicRegistry::indent(b);
      WasmIntrinsicRegistry::out(b) << ity << (leftRot ? ".shr_u\n" : ".shl\n");
      if (!leftRot && N < W) {
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << ity << ".const " << ((uint64_t(1) << N) - 1) << "\n";
        WasmIntrinsicRegistry::indent(b);
        WasmIntrinsicRegistry::out(b) << ity << ".and\n";
      }
      WasmIntrinsicRegistry::indent(b);
      WasmIntrinsicRegistry::out(b) << ity << ".or\n";

      // For n == 0: WASM shifts reduce shift count mod W. shl by 0 → u; shr_u
      // by (N - 0) = N → either u (if N == W) or 0 (since u was masked). OR
      // yields u in both cases — no extra special case needed.
    }

    class RotlIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t, uint32_t, const std::string &ity
      ) const override {
        declLocal(b, "u", ity);
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        emitRotation(backend, true, N, W, ity);
        sextN(backend, N, W, ity);
      }
    };

    class RotrIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t, uint32_t, const std::string &ity
      ) const override {
        declLocal(b, "u", ity);
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        emitRotation(backend, false, N, W, ity);
        sextN(backend, N, W, ity);
      }
    };

    class IsPow2Intrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &intr, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        uint32_t pN = paramN(intr, 0);
        uint32_t pW = widen(pN);
        std::string pity = ityOf(pW);
        (void) N;
        (void) pN;
        (void) ity;
        // (a0 > 0) — pity.gt_s yields i32
        pushArg(backend, 0);
        indent(backend);
        out(backend) << pity << ".const 0\n";
        indent(backend);
        out(backend) << pity << ".gt_s\n";
        // (a0 & (a0 - 1)) == 0 — pity.eqz yields i32
        pushArg(backend, 0);
        pushArg(backend, 0);
        indent(backend);
        out(backend) << pity << ".const 1\n";
        indent(backend);
        out(backend) << pity << ".sub\n";
        indent(backend);
        out(backend) << pity << ".and\n";
        indent(backend);
        out(backend) << pity << ".eqz\n";
        // AND the two i32 predicates → i32 holding 0 or 1
        indent(backend);
        out(backend) << "i32.and\n";
        // @is_pow2 returns i1; sign-extend bit 0 → 0/-1.
        sextN(backend, 1, 32, "i32");
        // Defensive widening if a future change makes W == 64. Sign-extending
        // i32 -1 → i64 -1 preserves the v0.2.2 i1 convention.
        if (W == 64) {
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
        }
      }
    };

    class Ilog2Intrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t, uint32_t, const std::string &ity
      ) const override {
        declLocal(b, "u", ity);
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        // UB if a0 <= 0
        pushArg(backend, 0);
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << ity << ".le_s\n";
        unreachableIfTop(backend);

        // u = a0 [& mask if N < W]
        pushArg(backend, 0);
        if (N < W) {
          pushMask(backend, N, W, ity);
          indent(backend);
          out(backend) << ity << ".and\n";
        }
        indent(backend);
        out(backend) << "local.set $u\n";

        // ilog2 = (N - 1) - (clz(u) - (W - N)) = (W - 1) - clz(u)
        // Build (W - 1) - clz(u) on the stack:
        indent(backend);
        out(backend) << ity << ".const " << (W - 1) << "\n";
        indent(backend);
        out(backend) << "local.get $u\n";
        indent(backend);
        out(backend) << ity << ".clz\n";
        indent(backend);
        out(backend) << ity << ".sub\n";
        sextN(backend, N, W, ity);
      }
    };

    // ── §12.5 Integer overflow-aware family ────────
    //
    // The widening-and-mask discipline already supplies modular wrap, so
    // the @wrapping_* helpers are direct iW.{add,sub,mul,neg,shl,shr_s}
    // followed by `sextN`.  Saturating helpers extend to i64 (for N <= 32)
    // and clamp; N == 64 uses signed-overflow detection identities since
    // WASM has no `i128`.

    class WrappingAddIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        pushArg(backend, 0);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".add\n";
        sextN(backend, N, W, ity);
      }
    };

    class WrappingSubIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        pushArg(backend, 0);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".sub\n";
        sextN(backend, N, W, ity);
      }
    };

    class WrappingMulIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        pushArg(backend, 0);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".mul\n";
        sextN(backend, N, W, ity);
      }
    };

    class WrappingNegIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        indent(backend);
        out(backend) << ity << ".const 0\n";
        pushArg(backend, 0);
        indent(backend);
        out(backend) << ity << ".sub\n";
        sextN(backend, N, W, ity);
      }
    };

    class WrappingShlIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        // UB: a1 < 0 || a1 >= N
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << ity << ".lt_s\n";
        unreachableIfTop(backend);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".const " << N << "\n";
        indent(backend);
        out(backend) << ity << ".ge_s\n";
        unreachableIfTop(backend);
        pushArg(backend, 0);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".shl\n";
        sextN(backend, N, W, ity);
      }
    };

    class WrappingShrIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        // UB: a1 < 0 || a1 >= N
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << ity << ".lt_s\n";
        unreachableIfTop(backend);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".const " << N << "\n";
        indent(backend);
        out(backend) << ity << ".ge_s\n";
        unreachableIfTop(backend);
        // shr_s preserves the iW sign; the iN sign sits at bit N-1.  For
        // N < W we must first sign-extend a0 so the iW sign bit reflects
        // the iN value's true sign, otherwise shr_s drags the wrong bit
        // in for negative iN values.  Args already arrive sign-extended
        // by the caller (caller stores were sign-extended to the helper's
        // widened param type), so a direct shr_s is correct.
        pushArg(backend, 0);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".shr_s\n";
        sextN(backend, N, W, ity);
      }
    };

    /**
     * @brief @saturating_add — widen to i64 for N <= 32 and clamp; for
     * N == 64 use signed-overflow detection.
     */
    class SaturatingAddIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &ity
      ) const override {
        if (N <= 32) {
          declLocal(b, "s64", "i64");
        } else {
          declLocal(b, "s", ity);
        }
        (void) W;
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        int64_t hi = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        int64_t lo = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        if (N <= 32) {
          // i64-widen, add, clamp to [lo, hi], narrow.
          pushArg(backend, 0);
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
          pushArg(backend, 1);
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
          indent(backend);
          out(backend) << "i64.add\n";
          indent(backend);
          out(backend) << "local.set $s64\n";
          // clamp lo: select stack is [val1, val2, cond]; returns val1 if
          // cond is true.  We want `s < lo ? lo : s`, so val1 = lo and
          // val2 = s — push lo first (bottom), then s.
          indent(backend);
          out(backend) << "i64.const " << lo << "\n"; // val1 = lo
          indent(backend);
          out(backend) << "local.get $s64\n"; // val2 = s
          indent(backend);
          out(backend) << "local.get $s64\n";
          indent(backend);
          out(backend) << "i64.const " << lo << "\n";
          indent(backend);
          out(backend) << "i64.lt_s\n"; // cond = (s < lo)
          indent(backend);
          out(backend) << "select\n"; // cond ? lo : s
          indent(backend);
          out(backend) << "local.set $s64\n";
          // clamp hi: want `s > hi ? hi : s`.  val1 = hi (push first), val2
          // = s.
          indent(backend);
          out(backend) << "i64.const " << hi << "\n"; // val1 = hi
          indent(backend);
          out(backend) << "local.get $s64\n"; // val2 = s
          indent(backend);
          out(backend) << "local.get $s64\n";
          indent(backend);
          out(backend) << "i64.const " << hi << "\n";
          indent(backend);
          out(backend) << "i64.gt_s\n"; // cond = (s > hi)
          indent(backend);
          out(backend) << "select\n"; // cond ? hi : s
          // narrow back to i32
          indent(backend);
          out(backend) << "i32.wrap_i64\n";
          sextN(backend, N, W, ity);
        } else {
          // N == 64.  Detect signed-add overflow via the two's-complement
          // identity: (a ^ s) & (b ^ s) < 0.  When overflow, choose
          // INT_MAX if a >= 0 else INT_MIN.
          pushArg(backend, 0);
          pushArg(backend, 1);
          indent(backend);
          out(backend) << ity << ".add\n";
          indent(backend);
          out(backend) << "local.set $s\n";
          // saturation candidate
          indent(backend);
          out(backend) << ity << ".const " << hi << "\n";
          indent(backend);
          out(backend) << ity << ".const " << lo << "\n";
          pushArg(backend, 0);
          indent(backend);
          out(backend) << ity << ".const 0\n";
          indent(backend);
          out(backend) << ity << ".ge_s\n";
          indent(backend);
          out(backend) << "select\n";
          // pick s vs candidate
          indent(backend);
          out(backend) << "local.get $s\n";
          // overflow test
          pushArg(backend, 0);
          indent(backend);
          out(backend) << "local.get $s\n";
          indent(backend);
          out(backend) << ity << ".xor\n";
          pushArg(backend, 1);
          indent(backend);
          out(backend) << "local.get $s\n";
          indent(backend);
          out(backend) << ity << ".xor\n";
          indent(backend);
          out(backend) << ity << ".and\n";
          indent(backend);
          out(backend) << ity << ".const 0\n";
          indent(backend);
          out(backend) << ity << ".lt_s\n";
          // (true sat-candidate, true s, true overflow_flag) → select
          // pops s then candidate based on flag.  Note: select(cond, a, b)
          // returns `a` if cond, else `b`.  We want overflow ? candidate : s,
          // which is select(cond=overflow, a=candidate, b=s).  Adjust ordering:
          // Currently on the stack (bottom → top): candidate, s, overflow.
          // `select` pops in order: cond, b, a — so we need stack order
          // candidate, s, overflow → select takes overflow as cond, s as
          // false-arm, candidate as true-arm? In WAT `select` pops 3 vals
          // (cond, val2, val1) and returns val1 if cond else val2.  Stack:
          // …, val1, val2, cond → returns val1 if cond.  So our val1 is
          // candidate (we want when overflow), val2 is s (when no overflow).
          // We already have candidate at the bottom and overflow on top;
          // re-establish the right order by swapping below — easier: emit
          // explicit `select`.
          indent(backend);
          out(backend) << "select\n";
        }
      }
    };

    /**
     * @brief @saturating_sub — analogous to @saturating_add via i64-widen
     * (N <= 32) or signed-sub overflow detection (N == 64).
     */
    class SaturatingSubIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &ity
      ) const override {
        if (N <= 32) {
          declLocal(b, "s64", "i64");
        } else {
          declLocal(b, "s", ity);
        }
        (void) W;
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        int64_t hi = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        int64_t lo = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        if (N <= 32) {
          pushArg(backend, 0);
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
          pushArg(backend, 1);
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
          indent(backend);
          out(backend) << "i64.sub\n";
          indent(backend);
          out(backend) << "local.set $s64\n";
          // clamp lo / hi: see saturating_add comment for select operand
          // ordering.  val1 (the value returned when cond is true) is
          // pushed first.
          indent(backend);
          out(backend) << "i64.const " << lo << "\n"; // val1 = lo
          indent(backend);
          out(backend) << "local.get $s64\n"; // val2 = s
          indent(backend);
          out(backend) << "local.get $s64\n";
          indent(backend);
          out(backend) << "i64.const " << lo << "\n";
          indent(backend);
          out(backend) << "i64.lt_s\n";
          indent(backend);
          out(backend) << "select\n";
          indent(backend);
          out(backend) << "local.set $s64\n";
          indent(backend);
          out(backend) << "i64.const " << hi << "\n"; // val1 = hi
          indent(backend);
          out(backend) << "local.get $s64\n"; // val2 = s
          indent(backend);
          out(backend) << "local.get $s64\n";
          indent(backend);
          out(backend) << "i64.const " << hi << "\n";
          indent(backend);
          out(backend) << "i64.gt_s\n";
          indent(backend);
          out(backend) << "select\n";
          indent(backend);
          out(backend) << "i32.wrap_i64\n";
          sextN(backend, N, W, ity);
        } else {
          // N == 64 sat-sub.  Overflow when (a ^ b) & (a ^ s) < 0.
          pushArg(backend, 0);
          pushArg(backend, 1);
          indent(backend);
          out(backend) << ity << ".sub\n";
          indent(backend);
          out(backend) << "local.set $s\n";
          // Saturation candidate.
          indent(backend);
          out(backend) << ity << ".const " << hi << "\n";
          indent(backend);
          out(backend) << ity << ".const " << lo << "\n";
          pushArg(backend, 0);
          indent(backend);
          out(backend) << ity << ".const 0\n";
          indent(backend);
          out(backend) << ity << ".ge_s\n";
          indent(backend);
          out(backend) << "select\n";
          indent(backend);
          out(backend) << "local.get $s\n";
          // Overflow flag.
          pushArg(backend, 0);
          pushArg(backend, 1);
          indent(backend);
          out(backend) << ity << ".xor\n";
          pushArg(backend, 0);
          indent(backend);
          out(backend) << "local.get $s\n";
          indent(backend);
          out(backend) << ity << ".xor\n";
          indent(backend);
          out(backend) << ity << ".and\n";
          indent(backend);
          out(backend) << ity << ".const 0\n";
          indent(backend);
          out(backend) << ity << ".lt_s\n";
          indent(backend);
          out(backend) << "select\n";
        }
      }
    };

    /**
     * @brief @saturating_mul — widen to i64 (N <= 32) or use a
     * sign-prediction + i64-mul fallback (N == 64 — WASM has no native
     * i128 widen).  For N == 64 we fall back to splitting via the
     * sign-bit identity, but the common case (N <= 32) is the
     * straightforward path.
     */
    class SaturatingMulIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t N, uint32_t W, const std::string &ity
      ) const override {
        if (N <= 32) {
          declLocal(b, "p64", "i64");
        } else {
          declLocal(b, "p", ity);
          declLocal(b, "ov", "i32");
        }
        (void) W;
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        int64_t hi = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        int64_t lo = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        if (N <= 32) {
          pushArg(backend, 0);
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
          pushArg(backend, 1);
          indent(backend);
          out(backend) << "i64.extend_i32_s\n";
          indent(backend);
          out(backend) << "i64.mul\n";
          indent(backend);
          out(backend) << "local.set $p64\n";
          // clamp lo / hi — val1 pushed first.
          indent(backend);
          out(backend) << "i64.const " << lo << "\n"; // val1 = lo
          indent(backend);
          out(backend) << "local.get $p64\n"; // val2 = p
          indent(backend);
          out(backend) << "local.get $p64\n";
          indent(backend);
          out(backend) << "i64.const " << lo << "\n";
          indent(backend);
          out(backend) << "i64.lt_s\n";
          indent(backend);
          out(backend) << "select\n";
          indent(backend);
          out(backend) << "local.set $p64\n";
          indent(backend);
          out(backend) << "i64.const " << hi << "\n"; // val1 = hi
          indent(backend);
          out(backend) << "local.get $p64\n"; // val2 = p
          indent(backend);
          out(backend) << "local.get $p64\n";
          indent(backend);
          out(backend) << "i64.const " << hi << "\n";
          indent(backend);
          out(backend) << "i64.gt_s\n";
          indent(backend);
          out(backend) << "select\n";
          indent(backend);
          out(backend) << "i32.wrap_i64\n";
          sextN(backend, N, W, ity);
        } else {
          // N == 64.  Detect overflow with the inverse identity:
          // overflow iff (b != 0) AND (p / b != a) where p = a * b
          // (signed division).  This is one comparison + one signed
          // division which WASM has (i64.div_s).  Avoid div by zero by
          // short-circuiting `b == 0 → no overflow`.  WASM has no
          // short-circuit primitive; emit an `if` block.
          pushArg(backend, 0);
          pushArg(backend, 1);
          indent(backend);
          out(backend) << ity << ".mul\n";
          indent(backend);
          out(backend) << "local.set $p\n";
          // Compute ov = 0 default.
          indent(backend);
          out(backend) << "i32.const 0\n";
          indent(backend);
          out(backend) << "local.set $ov\n";
          // if (b != 0) { ov = (p / b != a) }
          pushArg(backend, 1);
          indent(backend);
          out(backend) << ity << ".const 0\n";
          indent(backend);
          out(backend) << ity << ".ne\n";
          indent(backend);
          out(backend) << "if\n";
          incrIndent(backend);
          indent(backend);
          out(backend) << "local.get $p\n";
          pushArg(backend, 1);
          indent(backend);
          out(backend) << ity << ".div_s\n";
          pushArg(backend, 0);
          indent(backend);
          out(backend) << ity << ".ne\n";
          indent(backend);
          out(backend) << "local.set $ov\n";
          decrIndent(backend);
          indent(backend);
          out(backend) << "end\n";
          // Saturation candidate: sign(a)==sign(b) → INT_MAX else INT_MIN.
          indent(backend);
          out(backend) << ity << ".const " << hi << "\n";
          indent(backend);
          out(backend) << ity << ".const " << lo << "\n";
          // sign(a) XOR sign(b) gives 0/1 in the sign bit; turn into bool.
          pushArg(backend, 0);
          indent(backend);
          out(backend) << ity << ".const 0\n";
          indent(backend);
          out(backend) << ity << ".ge_s\n";
          pushArg(backend, 1);
          indent(backend);
          out(backend) << ity << ".const 0\n";
          indent(backend);
          out(backend) << ity << ".ge_s\n";
          indent(backend);
          out(backend) << "i32.eq\n"; // same sign?
          indent(backend);
          out(backend) << "select\n"; // → sat candidate on stack
          indent(backend);
          out(backend) << "local.get $p\n";
          indent(backend);
          out(backend) << "local.get $ov\n";
          indent(backend);
          out(backend) << "select\n";
        }
      }
    };

    /**
     * @brief @saturating_neg — `-x`, with INT_MIN_N saturated to
     * INT_MAX_N.
     */
    class SaturatingNegIntrinsic final : public WasmIntrinsic {
    public:
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        int64_t hi = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        int64_t lo = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        // candidate INT_MAX_N
        indent(backend);
        out(backend) << ity << ".const " << hi << "\n";
        // candidate -a0
        indent(backend);
        out(backend) << ity << ".const 0\n";
        pushArg(backend, 0);
        indent(backend);
        out(backend) << ity << ".sub\n";
        // cond: a0 == INT_MIN_N
        pushArg(backend, 0);
        indent(backend);
        out(backend) << ity << ".const " << lo << "\n";
        indent(backend);
        out(backend) << ity << ".eq\n";
        indent(backend);
        out(backend) << "select\n";
        sextN(backend, N, W, ity);
      }
    };

    /**
     * @brief @div_euclid — adjust truncating quotient toward `-∞` when the
     * truncated remainder is negative.  UB on `b == 0` or
     * `a == INT_MIN_N && b == -1`.
     */
    class DivEuclidIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t, uint32_t, const std::string &ity
      ) const override {
        declLocal(b, "q", ity);
        declLocal(b, "r", ity);
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        int64_t lo = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        // UB: a1 == 0
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".eqz\n";
        unreachableIfTop(backend);
        // UB: a0 == INT_MIN_N && a1 == -1
        pushArg(backend, 0);
        indent(backend);
        out(backend) << ity << ".const " << lo << "\n";
        indent(backend);
        out(backend) << ity << ".eq\n";
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".const -1\n";
        indent(backend);
        out(backend) << ity << ".eq\n";
        indent(backend);
        out(backend) << "i32.and\n";
        unreachableIfTop(backend);
        // q = a0 / a1
        pushArg(backend, 0);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".div_s\n";
        indent(backend);
        out(backend) << "local.set $q\n";
        // r = a0 - q * a1
        pushArg(backend, 0);
        indent(backend);
        out(backend) << "local.get $q\n";
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".mul\n";
        indent(backend);
        out(backend) << ity << ".sub\n";
        indent(backend);
        out(backend) << "local.set $r\n";
        // if (r < 0) q -= sign(a1)
        indent(backend);
        out(backend) << "local.get $r\n";
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << ity << ".lt_s\n";
        indent(backend);
        out(backend) << "if\n";
        incrIndent(backend);
        // step = b > 0 ? 1 : -1
        indent(backend);
        out(backend) << "local.get $q\n";
        indent(backend);
        out(backend) << ity << ".const 1\n";
        indent(backend);
        out(backend) << ity << ".const -1\n";
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << ity << ".gt_s\n";
        indent(backend);
        out(backend) << "select\n";
        indent(backend);
        out(backend) << ity << ".sub\n";
        indent(backend);
        out(backend) << "local.set $q\n";
        decrIndent(backend);
        indent(backend);
        out(backend) << "end\n";
        indent(backend);
        out(backend) << "local.get $q\n";
        sextN(backend, N, W, ity);
      }
    };

    /**
     * @brief @rem_euclid — add `|b|` to the truncated remainder when it is
     * negative.
     */
    class RemEuclidIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &, uint32_t, uint32_t, const std::string &ity
      ) const override {
        declLocal(b, "r", ity);
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t N, uint32_t W,
          const std::string &ity
      ) const override {
        int64_t lo = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".eqz\n";
        unreachableIfTop(backend);
        pushArg(backend, 0);
        indent(backend);
        out(backend) << ity << ".const " << lo << "\n";
        indent(backend);
        out(backend) << ity << ".eq\n";
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".const -1\n";
        indent(backend);
        out(backend) << ity << ".eq\n";
        indent(backend);
        out(backend) << "i32.and\n";
        unreachableIfTop(backend);
        // r = a0 rem a1 (signed)
        pushArg(backend, 0);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".rem_s\n";
        indent(backend);
        out(backend) << "local.set $r\n";
        // if (r < 0) r += |b|
        indent(backend);
        out(backend) << "local.get $r\n";
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << ity << ".lt_s\n";
        indent(backend);
        out(backend) << "if\n";
        incrIndent(backend);
        indent(backend);
        out(backend) << "local.get $r\n";
        // |b| via select
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".const 0\n";
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".sub\n";
        pushArg(backend, 1);
        indent(backend);
        out(backend) << ity << ".const 0\n";
        indent(backend);
        out(backend) << ity << ".gt_s\n";
        indent(backend);
        out(backend) << "select\n";
        indent(backend);
        out(backend) << ity << ".add\n";
        indent(backend);
        out(backend) << "local.set $r\n";
        decrIndent(backend);
        indent(backend);
        out(backend) << "end\n";
        indent(backend);
        out(backend) << "local.get $r\n";
        sextN(backend, N, W, ity);
      }
    };

    // ── Checksum primitives ──────────────────────────────────────────────
    //
    // Bit-exact with the interpreter / C lowerings, but table-free: WASM
    // has no cheap module-level mutable table storage, so the per-byte
    // step runs the defining LFSR recurrence directly. Because the round
    // function r(s) = (s >> 1) ^ ((s & 1) * poly) is linear over GF(2),
    // eight rounds on the full register equal the table form
    // (s >> 8) ^ tab[(s ^ byte) & 0xFF] — same values, no lookup table,
    // no imports, no linear-memory footprint.

    class Crc32UpdateIntrinsic final : public WasmIntrinsic {
    public:
      void declareLocals(
          WasmBackend &b, const IntrinsicDecl &intr, uint32_t, uint32_t, const std::string &
      ) const override {
        declLocal(b, "s", "i32");
        declLocal(b, "v", ityOf(widen(paramN(intr, 1))));
        declLocal(b, "b", "i32");
        declLocal(b, "j", "i32");
      }

      void emit(
          WasmBackend &backend, const IntrinsicDecl &intr, uint32_t /*N*/, uint32_t /*W*/,
          const std::string & /*ity*/
      ) const override {
        uint32_t valBits = paramN(intr, 1);
        uint32_t valW = widen(valBits);
        std::string vty = ityOf(valW);
        uint32_t nBytes = (valBits + 7) / 8;

        pushArg(backend, 0);
        indent(backend);
        out(backend) << "local.set $s\n";
        pushArg(backend, 1);
        if (valBits % 8 != 0 && valBits < valW) {
          // Zero-pad the last byte per the spec: the (sign-extended)
          // high bits of a negative iN must read as 0 above bit valBits,
          // matching the C lowering's mask and the interpreter's
          // argUint(valBits).
          indent(backend);
          out(backend) << vty << ".const " << ((uint64_t(1) << valBits) - 1) << "\n";
          indent(backend);
          out(backend) << vty << ".and\n";
        }
        indent(backend);
        out(backend) << "local.set $v\n";

        indent(backend);
        out(backend) << "i32.const 0\n";
        indent(backend);
        out(backend) << "local.set $b\n";
        indent(backend);
        out(backend) << "(loop $bytes\n";
        incrIndent(backend);
        // s ^= v & 0xFF
        indent(backend);
        out(backend) << "local.get $s\n";
        indent(backend);
        out(backend) << "local.get $v\n";
        if (valW == 64) {
          indent(backend);
          out(backend) << "i32.wrap_i64\n";
        }
        indent(backend);
        out(backend) << "i32.const 255\n";
        indent(backend);
        out(backend) << "i32.and\n";
        indent(backend);
        out(backend) << "i32.xor\n";
        indent(backend);
        out(backend) << "local.set $s\n";
        // 8 bit rounds: s = (s >>u 1) ^ ((s & 1) * 0xEDB88320)
        indent(backend);
        out(backend) << "i32.const 0\n";
        indent(backend);
        out(backend) << "local.set $j\n";
        indent(backend);
        out(backend) << "(loop $bits\n";
        incrIndent(backend);
        indent(backend);
        out(backend) << "local.get $s\n";
        indent(backend);
        out(backend) << "i32.const 1\n";
        indent(backend);
        out(backend) << "i32.shr_u\n";
        indent(backend);
        out(backend) << "local.get $s\n";
        indent(backend);
        out(backend) << "i32.const 1\n";
        indent(backend);
        out(backend) << "i32.and\n";
        indent(backend);
        out(backend) << "i32.const 3988292384\n"; // 0xEDB88320 (reflected poly)
        indent(backend);
        out(backend) << "i32.mul\n";
        indent(backend);
        out(backend) << "i32.xor\n";
        indent(backend);
        out(backend) << "local.set $s\n";
        indent(backend);
        out(backend) << "local.get $j\n";
        indent(backend);
        out(backend) << "i32.const 1\n";
        indent(backend);
        out(backend) << "i32.add\n";
        indent(backend);
        out(backend) << "local.tee $j\n";
        indent(backend);
        out(backend) << "i32.const 8\n";
        indent(backend);
        out(backend) << "i32.lt_s\n";
        indent(backend);
        out(backend) << "br_if $bits\n";
        decrIndent(backend);
        indent(backend);
        out(backend) << ")\n";
        // v >>= 8; continue while ++b < nBytes
        indent(backend);
        out(backend) << "local.get $v\n";
        indent(backend);
        out(backend) << vty << ".const 8\n";
        indent(backend);
        out(backend) << vty << ".shr_u\n";
        indent(backend);
        out(backend) << "local.set $v\n";
        indent(backend);
        out(backend) << "local.get $b\n";
        indent(backend);
        out(backend) << "i32.const 1\n";
        indent(backend);
        out(backend) << "i32.add\n";
        indent(backend);
        out(backend) << "local.tee $b\n";
        indent(backend);
        out(backend) << "i32.const " << nBytes << "\n";
        indent(backend);
        out(backend) << "i32.lt_s\n";
        indent(backend);
        out(backend) << "br_if $bytes\n";
        decrIndent(backend);
        indent(backend);
        out(backend) << ")\n";
        indent(backend);
        out(backend) << "local.get $s\n";
      }
    };

    class CheckChksumIntrinsic final : public WasmIntrinsic {
    public:
      // Mismatch diverges via `unreachable` — the WASM-native analogue of
      // the C lowering's fprintf+abort. A trap is externally visible, so
      // the checksum chain stays observable without any host imports.
      void emit(
          WasmBackend &backend, const IntrinsicDecl &, uint32_t /*N*/, uint32_t /*W*/,
          const std::string & /*ity*/
      ) const override {
        pushArg(backend, 0);
        pushArg(backend, 1);
        indent(backend);
        out(backend) << "i32.ne\n";
        trapIfTop(backend);
        pushArg(backend, 1);
      }
    };

  } // namespace

  // ── §12.6 Floating-point sign / bit ops ────

  /**
   * @brief Abstract base for FP-touching WASM intrinsic emitters. Unlike
   * WasmIntrinsic (integer, takes precomputed N/W/ity), WasmFpIntrinsic
   * receives only the IntrinsicDecl; per-arg WAT types derive from
   * intr.params[i].type via watTypeOf().
   */
  class WasmFpIntrinsic {
  public:
    virtual ~WasmFpIntrinsic() = default;
    virtual void emit(WasmBackend &backend, const IntrinsicDecl &intr) const = 0;

    virtual void declareLocals(WasmBackend &, const IntrinsicDecl &) const {}

  protected:
    static std::ostream &out(WasmBackend &backend) { return WasmIntrinsicRegistry::out(backend); }

    static void indent(WasmBackend &backend) { WasmIntrinsicRegistry::indent(backend); }

    static void incrIndent(WasmBackend &backend) { WasmIntrinsicRegistry::incrIndent(backend); }

    static void decrIndent(WasmBackend &backend) { WasmIntrinsicRegistry::decrIndent(backend); }

    // Read the FP width (32/64) of intr.retType.
    static uint32_t retFpBits(const IntrinsicDecl &intr) {
      return std::get<FloatType>(intr.retType->v).kind == FloatType::Kind::F32 ? 32 : 64;
    }

    static uint32_t paramFpBits(const IntrinsicDecl &intr, size_t i) {
      return std::get<FloatType>(intr.params[i].type->v).kind == FloatType::Kind::F32 ? 32 : 64;
    }

    static std::string retFpTy(const IntrinsicDecl &intr) {
      return retFpBits(intr) == 32 ? "f32" : "f64";
    }

    static std::string paramFpTy(const IntrinsicDecl &intr, size_t i) {
      return paramFpBits(intr, i) == 32 ? "f32" : "f64";
    }

    // FP result-finiteness UB guard: trap unless the value in
    // local `$r` is finite (`|$r| < +inf`, false for ±inf and NaN).
    // Stack-neutral, so under --no-ub-guards the whole block is elided.
    // `ty` is the float WASM type ("f32"/"f64").
    static void finiteCheckR(WasmBackend &backend, const std::string &ty) {
      if (WasmIntrinsicRegistry::noUbGuards(backend))
        return;
      indent(backend);
      out(backend) << "local.get $r\n";
      indent(backend);
      out(backend) << ty << ".abs\n";
      indent(backend);
      out(backend) << ty << ".const inf\n";
      indent(backend);
      out(backend) << ty << ".lt\n";
      indent(backend);
      out(backend) << "i32.eqz\n";
      indent(backend);
      out(backend) << "if\n";
      incrIndent(backend);
      indent(backend);
      out(backend) << "unreachable\n";
      decrIndent(backend);
      indent(backend);
      out(backend) << "end\n";
    }

    // i1-returning FP predicates produce an i32 0/1 from a WASM
    // comparison. Convert to the RefractIR i1 storage convention (0 / -1) by
    // sign-extending bit 0.
    static void sextI1ToI32(WasmBackend &backend) {
      indent(backend);
      out(backend) << "i32.const 31\n";
      indent(backend);
      out(backend) << "i32.shl\n";
      indent(backend);
      out(backend) << "i32.const 31\n";
      indent(backend);
      out(backend) << "i32.shr_s\n";
    }
  };

  namespace {

    // Float twin of ObserveIntrinsic — identity, as in the integer registry:
    // WASM has no forward-progress assumption, so no side effect is needed.
    class ObserveWasmFpIntrinsic final : public WasmFpIntrinsic {
    public:
      void emit(WasmBackend &backend, const IntrinsicDecl &) const override {
        indent(backend);
        out(backend) << "local.get $a0\n";
      }
    };

    class FabsWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        std::string ty = retFpTy(intr);
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << ty << ".abs\n";
      }
    };

    class FnegWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        std::string ty = retFpTy(intr);
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << ty << ".neg\n";
      }
    };

    class CopysignWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        std::string ty = retFpTy(intr);
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << "local.get $a1\n";
        indent(backend);
        out(backend) << ty << ".copysign\n";
      }
    };

    // ── §12.6 D.3 min / max ───────────────────────────────────────────
    //
    // WASM has native `fN.min` / `fN.max` opcodes that follow IEEE
    // 754-2008 minNum/maxNum (prefer -0 for min, +0 for max on signed-
    // zero pairs).  That matches the interpreter and the explicit
    // tie-break emitted by the C backend, so this is just a one-opcode
    // wrapper.

    class FminWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        std::string ty = retFpTy(intr);
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << "local.get $a1\n";
        indent(backend);
        out(backend) << ty << ".min\n";
      }
    };

    class FmaxWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        std::string ty = retFpTy(intr);
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << "local.get $a1\n";
        indent(backend);
        out(backend) << ty << ".max\n";
      }
    };

    class SignbitWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        // No native WASM signbit. Reinterpret the FP value as BV and test
        // the high bit. f32: i32.lt_s of reinterpret_f32 vs 0; f64: shift
        // the high half of reinterpret_f64 by 31 to land bit 63 in bit 0.
        uint32_t bits = paramFpBits(intr, 0);
        if (bits == 32) {
          indent(backend);
          out(backend) << "local.get $a0\n";
          indent(backend);
          out(backend) << "i32.reinterpret_f32\n";
          indent(backend);
          out(backend) << "i32.const 0\n";
          indent(backend);
          out(backend) << "i32.lt_s\n"; // result is i32 0 or 1
        } else {
          indent(backend);
          out(backend) << "local.get $a0\n";
          indent(backend);
          out(backend) << "i64.reinterpret_f64\n";
          indent(backend);
          out(backend) << "i64.const 0\n";
          indent(backend);
          out(backend) << "i64.lt_s\n"; // result is already i32 0/1 (WASM rule)
        }
        // @signbit returns i1; sign-extend bit 0 → 0/-1.
        sextI1ToI32(backend);
      }
    };

    // ── §12.6 D.2 classification predicates ───────────────────────────

    /**
     * @brief @is_normal — WASM MVP has no native fN.is_normal.  Compose
     * by extracting the biased exponent from the reinterpreted bits and
     * testing it lies strictly inside [1, max-1].
     */
    class IsNormalWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void declareLocals(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        uint32_t bits = paramFpBits(intr, 0);
        indent(backend);
        out(backend) << "(local $exp " << (bits == 32 ? "i32" : "i64") << ")\n";
      }

      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        uint32_t bits = paramFpBits(intr, 0);
        if (bits == 32) {
          // exp = (reinterpret_f32(a0) >> 23) & 0xFF
          indent(backend);
          out(backend) << "local.get $a0\n";
          indent(backend);
          out(backend) << "i32.reinterpret_f32\n";
          indent(backend);
          out(backend) << "i32.const 23\n";
          indent(backend);
          out(backend) << "i32.shr_u\n";
          indent(backend);
          out(backend) << "i32.const 255\n";
          indent(backend);
          out(backend) << "i32.and\n";
          indent(backend);
          out(backend) << "local.set $exp\n";
          // result = (exp != 0) & (exp != 255)
          indent(backend);
          out(backend) << "local.get $exp\n";
          indent(backend);
          out(backend) << "i32.const 0\n";
          indent(backend);
          out(backend) << "i32.ne\n";
          indent(backend);
          out(backend) << "local.get $exp\n";
          indent(backend);
          out(backend) << "i32.const 255\n";
          indent(backend);
          out(backend) << "i32.ne\n";
          indent(backend);
          out(backend) << "i32.and\n";
        } else {
          // f64: 11-bit exponent at bits 52-62; max = 0x7FF.
          indent(backend);
          out(backend) << "local.get $a0\n";
          indent(backend);
          out(backend) << "i64.reinterpret_f64\n";
          indent(backend);
          out(backend) << "i64.const 52\n";
          indent(backend);
          out(backend) << "i64.shr_u\n";
          indent(backend);
          out(backend) << "i64.const 2047\n";
          indent(backend);
          out(backend) << "i64.and\n";
          indent(backend);
          out(backend) << "local.set $exp\n";
          indent(backend);
          out(backend) << "local.get $exp\n";
          indent(backend);
          out(backend) << "i64.const 0\n";
          indent(backend);
          out(backend) << "i64.ne\n"; // i32 result
          indent(backend);
          out(backend) << "local.get $exp\n";
          indent(backend);
          out(backend) << "i64.const 2047\n";
          indent(backend);
          out(backend) << "i64.ne\n";
          indent(backend);
          out(backend) << "i32.and\n";
        }
        // @is_normal returns i1; sign-extend bit 0 → 0/-1.
        sextI1ToI32(backend);
      }
    };

    /**
     * @brief @is_subnormal — biased exponent 0 AND mantissa != 0.
     */
    class IsSubnormalWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void declareLocals(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        uint32_t bits = paramFpBits(intr, 0);
        std::string ity = (bits == 32) ? "i32" : "i64";
        indent(backend);
        out(backend) << "(local $bp " << ity << ")\n";
      }

      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        uint32_t bits = paramFpBits(intr, 0);
        if (bits == 32) {
          // bp = reinterpret_f32(a0)
          indent(backend);
          out(backend) << "local.get $a0\n";
          indent(backend);
          out(backend) << "i32.reinterpret_f32\n";
          indent(backend);
          out(backend) << "local.set $bp\n";
          // exp_zero = (bp & 0x7F800000) == 0
          indent(backend);
          out(backend) << "local.get $bp\n";
          indent(backend);
          out(backend) << "i32.const 2139095040\n"; // 0x7F800000
          indent(backend);
          out(backend) << "i32.and\n";
          indent(backend);
          out(backend) << "i32.eqz\n";
          // mantissa_nonzero = (bp & 0x007FFFFF) != 0
          indent(backend);
          out(backend) << "local.get $bp\n";
          indent(backend);
          out(backend) << "i32.const 8388607\n"; // 0x007FFFFF
          indent(backend);
          out(backend) << "i32.and\n";
          indent(backend);
          out(backend) << "i32.const 0\n";
          indent(backend);
          out(backend) << "i32.ne\n";
          // and them
          indent(backend);
          out(backend) << "i32.and\n";
        } else {
          // f64: 11-bit exp at 52..62 (0x7FF0000000000000); mantissa mask 0x000FFFFFFFFFFFFF.
          indent(backend);
          out(backend) << "local.get $a0\n";
          indent(backend);
          out(backend) << "i64.reinterpret_f64\n";
          indent(backend);
          out(backend) << "local.set $bp\n";
          // exp_zero = (bp & 0x7FF0000000000000) == 0
          indent(backend);
          out(backend) << "local.get $bp\n";
          indent(backend);
          out(backend) << "i64.const 9218868437227405312\n"; // 0x7FF0...
          indent(backend);
          out(backend) << "i64.and\n";
          indent(backend);
          out(backend) << "i64.eqz\n"; // i32 result
          // mantissa_nonzero = (bp & 0x000FFFFFFFFFFFFF) != 0
          indent(backend);
          out(backend) << "local.get $bp\n";
          indent(backend);
          out(backend) << "i64.const 4503599627370495\n"; // 0x000FFFFFFFFFFFFF
          indent(backend);
          out(backend) << "i64.and\n";
          indent(backend);
          out(backend) << "i64.const 0\n";
          indent(backend);
          out(backend) << "i64.ne\n"; // i32 result
          indent(backend);
          out(backend) << "i32.and\n";
        }
        // @is_subnormal returns i1; sign-extend bit 0 → 0/-1.
        sextI1ToI32(backend);
      }
    };

    class ToBitsWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        uint32_t bits = paramFpBits(intr, 0);
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << (bits == 32 ? "i32.reinterpret_f32\n" : "i64.reinterpret_f64\n");
      }
    };

    class FromBitsWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void declareLocals(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        // Scratch local to hold the reinterpreted FP value for the finiteness
        // check. Width follows the return type.
        std::string ty = retFpTy(intr);
        indent(backend);
        out(backend) << "(local $r " << ty << ")\n";
      }

      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        std::string ty = retFpTy(intr);
        // Step 1: reinterpret int bits as FP, save to scratch local.
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << (retFpBits(intr) == 32 ? "f32.reinterpret_i32\n" : "f64.reinterpret_i64\n");
        indent(backend);
        out(backend) << "local.set $r\n";
        // Step 2: finiteness check — UB if result is ±∞ or NaN.
        // (|r| < +inf) is true iff r is finite. Compute it as
        // (r.abs < +inf): equivalent and avoids an explicit NaN check
        // because NaN comparisons are always false.
        finiteCheckR(backend, ty);
        // Step 3: return the FP value.
        indent(backend);
        out(backend) << "local.get $r\n";
      }
    };

    // ── §12.6 D.4 correctly-rounded math ──────────────────────────────────
    //
    // WASM has native `fN.sqrt`, `fN.floor`, `fN.ceil`, `fN.trunc` opcodes,
    // all correctly rounded per IEEE 754, so each is a one-opcode wrapper.
    // @sqrt of a strictly-negative input is NaN (UB under §2.9); the
    // integral rounders never leave the finite domain.

    class SqrtWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void declareLocals(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        // Scratch local holding the sqrt result for the finiteness check.
        indent(backend);
        out(backend) << "(local $r " << retFpTy(intr) << ")\n";
      }

      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        std::string ty = retFpTy(intr);
        // Step 1: compute the square root, save to the scratch local.
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << ty << ".sqrt\n";
        indent(backend);
        out(backend) << "local.set $r\n";
        // Step 2: UB if the result is non-finite (§2.9).  A strictly-negative
        // operand makes sqrt NaN; (|r| < +inf) is true iff r is finite (NaN
        // comparisons are always false).  Same result-finiteness guard as
        // @from_bits, keeping every backend's sqrt UB rule uniform.
        finiteCheckR(backend, ty);
        // Step 3: return the (finite) result.
        indent(backend);
        out(backend) << "local.get $r\n";
      }
    };

    // Shared one-opcode wrapper for the integral-rounding intrinsics.
    class RtiWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      explicit RtiWasmIntrinsic(std::string op) : op_(std::move(op)) {}

      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        std::string ty = retFpTy(intr);
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << ty << "." << op_ << "\n";
      }

    private:
      std::string op_;
    };

    // ── §12.6 D.5 compositions ─────────────────────────────────────────────
    //
    // @fract(x) = x - trunc(x) via native `fN.trunc` + `fN.sub`; the result
    // is always finite (|.| < 1) so there is no UB guard.  @recip(x) = 1/x
    // via `fN.div`, then the same (|r| < +inf) finiteness guard as @sqrt
    // (UB on x = ±0.0 or reciprocal overflow).

    class FractWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        std::string ty = retFpTy(intr);
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << ty << ".trunc\n";
        indent(backend);
        out(backend) << ty << ".sub\n";
      }
    };

    class RecipWasmIntrinsic final : public WasmFpIntrinsic {
    public:
      void declareLocals(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        // Scratch local holding the reciprocal for the finiteness check.
        indent(backend);
        out(backend) << "(local $r " << retFpTy(intr) << ")\n";
      }

      void emit(WasmBackend &backend, const IntrinsicDecl &intr) const override {
        std::string ty = retFpTy(intr);
        // Step 1: r = 1 / a0.
        indent(backend);
        out(backend) << ty << ".const 1\n";
        indent(backend);
        out(backend) << "local.get $a0\n";
        indent(backend);
        out(backend) << ty << ".div\n";
        indent(backend);
        out(backend) << "local.set $r\n";
        // Step 2: UB if non-finite (x = ±0.0 → ±inf, or overflow).  Same
        // (|r| < +inf) guard as @sqrt.
        finiteCheckR(backend, ty);
        // Step 3: return the (finite) reciprocal.
        indent(backend);
        out(backend) << "local.get $r\n";
      }
    };

  } // namespace

  struct WasmFpIntrinsicRegistry {
    using GenFn = std::unique_ptr<WasmFpIntrinsic>;

    static const std::unordered_map<IntrinsicKind, GenFn> &getRegistry() {
      static const std::unordered_map<IntrinsicKind, GenFn> registry = []() {
        std::unordered_map<IntrinsicKind, GenFn> r;
        r[IntrinsicKind::Observe] = std::make_unique<ObserveWasmFpIntrinsic>();
        r[IntrinsicKind::Fabs] = std::make_unique<FabsWasmIntrinsic>();
        r[IntrinsicKind::Fneg] = std::make_unique<FnegWasmIntrinsic>();
        r[IntrinsicKind::Copysign] = std::make_unique<CopysignWasmIntrinsic>();
        r[IntrinsicKind::Signbit] = std::make_unique<SignbitWasmIntrinsic>();
        r[IntrinsicKind::ToBits] = std::make_unique<ToBitsWasmIntrinsic>();
        r[IntrinsicKind::FromBits] = std::make_unique<FromBitsWasmIntrinsic>();
        r[IntrinsicKind::IsNormal] = std::make_unique<IsNormalWasmIntrinsic>();
        r[IntrinsicKind::IsSubnormal] = std::make_unique<IsSubnormalWasmIntrinsic>();
        r[IntrinsicKind::Fmin] = std::make_unique<FminWasmIntrinsic>();
        r[IntrinsicKind::Fmax] = std::make_unique<FmaxWasmIntrinsic>();
        r[IntrinsicKind::Sqrt] = std::make_unique<SqrtWasmIntrinsic>();
        r[IntrinsicKind::Floor] = std::make_unique<RtiWasmIntrinsic>("floor");
        r[IntrinsicKind::Ceil] = std::make_unique<RtiWasmIntrinsic>("ceil");
        r[IntrinsicKind::Trunc] = std::make_unique<RtiWasmIntrinsic>("trunc");
        r[IntrinsicKind::Fract] = std::make_unique<FractWasmIntrinsic>();
        r[IntrinsicKind::Recip] = std::make_unique<RecipWasmIntrinsic>();
        return r;
      }();
      return registry;
    }
  };

  const std::unordered_map<IntrinsicKind, WasmIntrinsicRegistry::WasmIntrinsicGenFn> &
  WasmIntrinsicRegistry::getRegistry() {
    static const std::unordered_map<IntrinsicKind, WasmIntrinsicGenFn> registry = []() {
      std::unordered_map<IntrinsicKind, WasmIntrinsicGenFn> r;
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
      r[IntrinsicKind::Observe] = std::make_unique<ObserveIntrinsic>();
      return r;
    }();
    return registry;
  }

  // FP-aware, decl-based naming: mirrors the C-backend rule.
  std::string WasmBackend::intrinsicHelperName(const IntrinsicDecl &intr) const {
    std::string base = intr.name.name;
    if (!base.empty() && base[0] == '@')
      base.erase(0, 1);
    // Reductions overload on vector shape (lane count + element
    // type); mangle by both so distinct declarations get distinct helpers.
    if (!intr.params.empty() && intr.params[0].type) {
      if (auto vt = std::get_if<VecType>(&intr.params[0].type->v)) {
        std::string el = "i32";
        if (auto ib = TypeUtils::getIntBitWidth(vt->elem))
          el = "i" + std::to_string(*ib);
        else if (auto fp = std::get_if<FloatType>(&vt->elem->v))
          el = fp->kind == FloatType::Kind::F32 ? "f32" : "f64";
        return "$_refractir_" + base + "_v" + std::to_string(vt->size) + "_" + el;
      }
    }
    bool paramFp = !intr.params.empty() && intr.params[0].type &&
                   std::holds_alternative<FloatType>(intr.params[0].type->v);
    bool retFp = intr.retType && std::holds_alternative<FloatType>(intr.retType->v);
    if (paramFp || retFp) {
      auto t = !intr.params.empty() ? intr.params[0].type : intr.retType;
      if (auto fp = std::get_if<FloatType>(&t->v))
        return "$_refractir_" + base + "_f" + (fp->kind == FloatType::Kind::F32 ? "32" : "64");
      if (auto it = std::get_if<IntType>(&t->v)) {
        uint32_t b = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
        return "$_refractir_" + base + "_i" + std::to_string(b);
      }
    }
    // For integer intrinsics whose parameter widths differ from the
    // return width, mangle by the first differing parameter's width
    // instead of the return (mirrors the C backend) — otherwise the
    // per-arg width overloads of one intrinsic (e.g. @crc32_update over
    // i8 / i32 / i64) would all collapse onto one helper name.
    auto rb = TypeUtils::getIntBitWidth(intr.retType);
    for (const auto &p: intr.params) {
      auto pb = TypeUtils::getIntBitWidth(p.type);
      if (pb && rb && *pb != *rb)
        return "$_refractir_" + base + "_i" + std::to_string(*pb);
    }
    return "$_refractir_" + base + "_i" + std::to_string(rb.value_or(32));
  }

  // Return the WAT scalar type-name for a RefractIR type:
  //   IntType i1..i32 → "i32", i64 → "i64"; FloatType f32 → "f32", f64 → "f64".
  static std::string watTypeOf(const TypePtr &t) {
    if (auto fp = std::get_if<FloatType>(&t->v))
      return fp->kind == FloatType::Kind::F32 ? "f32" : "f64";
    if (auto it = std::get_if<IntType>(&t->v)) {
      uint32_t b = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
      return (b <= 32) ? "i32" : "i64";
    }
    return "i32";
  }

  void WasmBackend::emitIntrinsicHelper(const IntrinsicDecl &intr) {
    // Horizontal vector reductions (§12.4) have a vector
    // parameter and lower to a dedicated helper (see emitReductionHelper).
    if (auto k = getIntrinsicKind(intr.name.name); k && isReductionIntrinsic(*k)) {
      emitReductionHelper(intr);
      return;
    }
    // FP-touching path.
    bool anyFp = (intr.retType && std::holds_alternative<FloatType>(intr.retType->v));
    for (const auto &p: intr.params)
      anyFp = anyFp || (p.type && std::holds_alternative<FloatType>(p.type->v));

    if (anyFp) {
      std::string outTy = watTypeOf(intr.retType);
      std::string name = intrinsicHelperName(intr);
      indent();
      out_ << "(func " << name;
      for (size_t i = 0; i < intr.params.size(); ++i)
        out_ << " (param $a" << i << " " << watTypeOf(intr.params[i].type) << ")";
      out_ << " (result " << outTy << ")\n";
      indent_level_++;
      auto kind = getIntrinsicKind(intr.name.name);
      if (kind) {
        const auto &fpRegistry = WasmFpIntrinsicRegistry::getRegistry();
        auto it = fpRegistry.find(*kind);
        if (it != fpRegistry.end()) {
          it->second->declareLocals(*this, intr);
          it->second->emit(*this, intr);
          indent_level_--;
          indent();
          out_ << ")\n";
          return;
        }
      }
      indent();
      out_ << "unreachable\n";
      indent_level_--;
      indent();
      out_ << ")\n";
      return;
    }

    // Legacy integer-only path.
    auto rb = getIntWidth(intr.retType);
    if (rb == 0)
      return;
    uint32_t N = rb;
    uint32_t W = (N <= 32) ? 32 : 64;
    std::string ity = (W == 32) ? "i32" : "i64";
    // Same decl-based name as the call site in emitCallAtom — the two
    // must agree or per-width overloads dangle.
    std::string name = intrinsicHelperName(intr);

    indent();
    out_ << "(func " << name;
    for (size_t i = 0; i < intr.params.size(); ++i) {
      uint32_t pN = getIntWidth(intr.params[i].type);
      if (pN == 0)
        pN = 32;
      uint32_t pW = (pN <= 32) ? 32 : 64;
      std::string pity = (pW == 32) ? "i32" : "i64";
      out_ << " (param $a" << i << " " << pity << ")";
    }
    out_ << " (result " << ity << ")\n";
    indent_level_++;

    auto kind = getIntrinsicKind(intr.name.name);
    if (kind) {
      const auto &registry = WasmIntrinsicRegistry::getRegistry();
      auto it = registry.find(*kind);
      if (it != registry.end()) {
        it->second->declareLocals(*this, intr, N, W, ity);
        it->second->emit(*this, intr, N, W, ity);
        indent_level_--;
        indent();
        out_ << ")\n";
        return;
      }
    }

    indent();
    out_ << "unreachable\n";
    indent_level_--;
    indent();
    out_ << ")\n";
  }

  // Horizontal vector reduction helper (§12.4). The vector arg
  // is passed by address (the frame-memory spill ABI is packed under every
  // vec-lowering strategy), so each lane is read directly with
  // `<elem>.load offset=k*sizeof(elem)` — the same layout the `scalars`
  // unpackParam uses. Lanes fold left-to-right into an accumulator local.
  // Consistent with the backend's stance that arithmetic UB is not
  // sanitized on WASM (docs/symirc.md), the fold emits no overflow /
  // finiteness guards; min/max use native fN.min/fN.max (correct IEEE
  // signed-zero tie-break) for floats and compare+select for integers.
  void WasmBackend::emitReductionHelper(const IntrinsicDecl &intr) {
    const IntrinsicKind kind = *getIntrinsicKind(intr.name.name);
    const VecType &vt = std::get<VecType>(intr.params[0].type->v);
    const uint32_t M = static_cast<uint32_t>(vt.size);
    const TypePtr &elem = vt.elem;
    const bool isFloat = std::holds_alternative<FloatType>(elem->v);
    const std::string retTy = watTypeOf(elem); // scalar WASM type of T
    const uint32_t elemSize = getTypeSize(elem);
    const uint32_t width =
        isFloat ? (std::get<FloatType>(elem->v).kind == FloatType::Kind::F32 ? 32u : 64u)
                : getIntWidth(elem);

    indent();
    out_ << "(func " << intrinsicHelperName(intr) << " (param $a0 i32) (result " << retTy << ")\n";
    indent_level_++;
    indent();
    out_ << "(local $acc " << retTy << ")\n";

    // Push lane k (sign-extended for narrow ints), matching the packed
    // spill layout used by every vec-lowering strategy.
    auto loadLane = [&](uint32_t k) {
      indent();
      out_ << "local.get $a0\n";
      if (k > 0) {
        indent();
        out_ << "i32.const " << (k * elemSize) << "\n";
        indent();
        out_ << "i32.add\n";
      }
      indent();
      if (isFloat)
        out_ << (width == 32 ? "f32.load\n" : "f64.load\n");
      else
        out_
            << (width <= 8    ? "i32.load8_s\n"
                : width <= 16 ? "i32.load16_s\n"
                : width <= 32 ? "i32.load\n"
                              : "i64.load\n");
    };

    // acc = lane 0.
    loadLane(0);
    indent();
    out_ << "local.set $acc\n";

    for (uint32_t k = 1; k < M; ++k) {
      if (isFloat && (kind == IntrinsicKind::ReduceMin || kind == IntrinsicKind::ReduceMax)) {
        // Native fN.min / fN.max — IEEE minNum/maxNum, order-independent.
        indent();
        out_ << "local.get $acc\n";
        loadLane(k);
        indent();
        out_ << retTy << (kind == IntrinsicKind::ReduceMin ? ".min\n" : ".max\n");
      } else if (kind == IntrinsicKind::ReduceMin || kind == IntrinsicKind::ReduceMax) {
        // Integer min/max: select(acc, x, acc <cmp> x).
        indent();
        out_ << "local.get $acc\n";
        loadLane(k);
        indent();
        out_ << "local.get $acc\n";
        loadLane(k);
        indent();
        out_ << retTy << (kind == IntrinsicKind::ReduceMin ? ".lt_s\n" : ".gt_s\n");
        indent();
        out_ << "select\n";
      } else {
        // Add / and / or / xor — one native binary op per step.
        indent();
        out_ << "local.get $acc\n";
        loadLane(k);
        indent();
        const char *op = (kind == IntrinsicKind::ReduceAdd)   ? ".add\n"
                         : (kind == IntrinsicKind::ReduceAnd) ? ".and\n"
                         : (kind == IntrinsicKind::ReduceOr)  ? ".or\n"
                                                              : ".xor\n";
        out_ << retTy << op;
      }
      indent();
      out_ << "local.set $acc\n";
    }

    indent();
    out_ << "local.get $acc\n";
    // Sign-extend an integer result to its declared width so the caller
    // sees a valid iN value (a no-op for min/max/bitwise, and for a
    // UB-free add whose result already fits).
    if (!isFloat) {
      uint32_t W = (width <= 32) ? 32 : 64;
      if (width != W) {
        indent();
        out_ << retTy << ".const " << (W - width) << "\n";
        indent();
        out_ << retTy << ".shl\n";
        indent();
        out_ << retTy << ".const " << (W - width) << "\n";
        indent();
        out_ << retTy << ".shr_s\n";
      }
    }
    indent_level_--;
    indent();
    out_ << ")\n";
  }

} // namespace refractir
