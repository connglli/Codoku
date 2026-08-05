// Solver-side SMT lowering for built-in intrinsics.
//
// This file is the single source of truth for every intrinsic
// supported by the RefractIR symbolic executor / SMT constraint generator.
// To add a new intrinsic:
//   1. Add its IntrinsicKind to include/analysis/intrinsics.hpp.
//   2. Implement a subclass of SolverIntrinsic and register it below.
//
// Width conventions:
//   - `N` is the return-type bit-width.
//   - For predicate intrinsics (@parity, @is_pow2) the return is bv1
//     while the input is bvN — impls read per-arg widths from `intr`
//     when needed.

#include "solver/solver.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "analysis/intrinsics.hpp"
#include "analysis/type_utils.hpp"
#include "internal.hpp"

namespace refractir {

  namespace {

    /**
     * @brief Helper: param bit-width from the declaration.
     */
    inline uint32_t paramBitWidth(const IntrinsicDecl &intr, size_t i) {
      auto b = TypeUtils::getIntBitWidth(intr.params[i].type);
      if (!b)
        throw std::runtime_error(
            "Solver: intrinsic " + intr.name.name + " param " + std::to_string(i) +
            " is non-integer"
        );
      return *b;
    }

    /**
     * @brief Abstract base class for lowering built-in intrinsics to SMT terms.
     */
    class SolverIntrinsic {
    public:
      virtual ~SolverIntrinsic() = default;

      /**
       * @brief Lowers the intrinsic call into a SymbolicValue.
       * @param intr Declaration (for per-arg widths).
       * @param N Declared return bit-width.
       * @param argVals Symbolic values for all inputs.
       * @param bvN Return-side bit-vector sort of width N.
       * @param solver SMT solver backend.
       * @param pc Accumulator for path conditions / UB constraints.
       */
      virtual SymbolicValue solve(
          const IntrinsicDecl &intr, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const = 0;
    };

    // ── §12.1 Arithmetic intrinsics ─────────────────────────────────────────

    class AbsIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        smt::Term x = argVals[0].term;
        int64_t int_min_N = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        auto minT = solver.make_bv_value_int64(bvN, int_min_N);
        pc.push_back(solver.make_term(smt::Kind::DISTINCT, {x, minT}));
        auto zero = solver.make_bv_value_int64(bvN, 0);
        auto cond = solver.make_term(smt::Kind::BV_SGE, {x, zero});
        auto neg = solver.make_term(smt::Kind::BV_NEG, {x});
        auto res = solver.make_term(smt::Kind::ITE, {cond, x, neg});
        return SymbolicValue(SymbolicValue::Kind::Int, res, solver.make_true());
      }
    };

    // Observability beacon — identity value; the observable write lives only in
    // the C lowering, so symbolically @observe(v) is v (no UB, no side effect).
    class ObserveIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &, std::vector<smt::Term> &
      ) const override {
        return argVals[0];
      }
    };

    class MinIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        auto cond = solver.make_term(smt::Kind::BV_SLE, {argVals[0].term, argVals[1].term});
        auto res = solver.make_term(smt::Kind::ITE, {cond, argVals[0].term, argVals[1].term});
        return SymbolicValue(SymbolicValue::Kind::Int, res, solver.make_true());
      }
    };

    class MaxIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        auto cond = solver.make_term(smt::Kind::BV_SGE, {argVals[0].term, argVals[1].term});
        auto res = solver.make_term(smt::Kind::ITE, {cond, argVals[0].term, argVals[1].term});
        return SymbolicValue(SymbolicValue::Kind::Int, res, solver.make_true());
      }
    };

    // ── §12.2 Bit-counting intrinsics ───────────────────────────────────────

    class PopcountIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        smt::Term x = argVals[0].term;
        smt::Term acc = solver.make_bv_value_int64(bvN, 0);
        for (uint32_t k = 0; k < N; ++k) {
          auto bit = solver.make_term(smt::Kind::BV_EXTRACT, {x}, {k, k});
          smt::Term ext =
              (N == 1) ? bit : solver.make_term(smt::Kind::BV_ZERO_EXTEND, {bit}, {N - 1});
          acc = solver.make_term(smt::Kind::BV_ADD, {acc, ext});
        }
        return SymbolicValue(SymbolicValue::Kind::Int, acc, solver.make_true());
      }
    };

    class ClzIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        smt::Term x = argVals[0].term;
        auto zero = solver.make_bv_value_int64(bvN, 0);
        pc.push_back(solver.make_term(smt::Kind::DISTINCT, {x, zero}));

        smt::Term result = solver.make_bv_value_int64(bvN, N);
        auto one1 = solver.make_bv_value_int64(solver.make_bv_sort(1), 1);
        for (int i = (int) N - 1; i >= 0; --i) {
          int bitPos = (int) N - 1 - i;
          auto bit =
              solver.make_term(smt::Kind::BV_EXTRACT, {x}, {(uint32_t) bitPos, (uint32_t) bitPos});
          auto cond = solver.make_term(smt::Kind::EQUAL, {bit, one1});
          auto iT = solver.make_bv_value_int64(bvN, i);
          result = solver.make_term(smt::Kind::ITE, {cond, iT, result});
        }
        return SymbolicValue(SymbolicValue::Kind::Int, result, solver.make_true());
      }
    };

    class CtzIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        smt::Term x = argVals[0].term;
        auto zero = solver.make_bv_value_int64(bvN, 0);
        pc.push_back(solver.make_term(smt::Kind::DISTINCT, {x, zero}));

        smt::Term result = solver.make_bv_value_int64(bvN, N);
        auto one1 = solver.make_bv_value_int64(solver.make_bv_sort(1), 1);
        for (int i = (int) N - 1; i >= 0; --i) {
          auto bit = solver.make_term(smt::Kind::BV_EXTRACT, {x}, {(uint32_t) i, (uint32_t) i});
          auto cond = solver.make_term(smt::Kind::EQUAL, {bit, one1});
          auto iT = solver.make_bv_value_int64(bvN, i);
          result = solver.make_term(smt::Kind::ITE, {cond, iT, result});
        }
        return SymbolicValue(SymbolicValue::Kind::Int, result, solver.make_true());
      }
    };

    // ── §12.3 Integer extras ────────────────────────────────

    class AbsDiffIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort bvN,
          smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        smt::Term a = argVals[0].term, b = argVals[1].term;
        auto diffAB = solver.make_term(smt::Kind::BV_SUB, {a, b});
        auto diffBA = solver.make_term(smt::Kind::BV_SUB, {b, a});
        auto cond = solver.make_term(smt::Kind::BV_SGE, {a, b});
        auto r = solver.make_term(smt::Kind::ITE, {cond, diffAB, diffBA});
        // UB: result must be non-negative as signed iN.
        auto zero = solver.make_bv_value_int64(bvN, 0);
        pc.push_back(solver.make_term(smt::Kind::BV_SGE, {r, zero}));
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class SignumIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort bvN,
          smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        smt::Term x = argVals[0].term;
        auto zero = solver.make_bv_value_int64(bvN, 0);
        auto one = solver.make_bv_value_int64(bvN, 1);
        auto minus1 = solver.make_bv_value_int64(bvN, -1);
        auto isNeg = solver.make_term(smt::Kind::BV_SLT, {x, zero});
        auto isZero = solver.make_term(smt::Kind::EQUAL, {x, zero});
        auto inner = solver.make_term(smt::Kind::ITE, {isZero, zero, one});
        auto r = solver.make_term(smt::Kind::ITE, {isNeg, minus1, inner});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class ClampIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        smt::Term v = argVals[0].term, lo = argVals[1].term, hi = argVals[2].term;
        pc.push_back(solver.make_term(smt::Kind::BV_SLE, {lo, hi}));
        auto vGtHi = solver.make_term(smt::Kind::BV_SGT, {v, hi});
        auto inner = solver.make_term(smt::Kind::ITE, {vGtHi, hi, v});
        auto vLtLo = solver.make_term(smt::Kind::BV_SLT, {v, lo});
        auto r = solver.make_term(smt::Kind::ITE, {vLtLo, lo, inner});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class MidpointIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        // Sign-extend by 1 bit to avoid signed overflow on add, divide by 2
        // with bvsdiv (truncation toward zero), then extract low N bits.
        smt::Term a = argVals[0].term, b = argVals[1].term;
        auto ea = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {a}, {1});
        auto eb = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {b}, {1});
        auto sum = solver.make_term(smt::Kind::BV_ADD, {ea, eb});
        auto bv2 = solver.make_bv_value_int64(solver.make_bv_sort(N + 1), 2);
        auto half = solver.make_term(smt::Kind::BV_SDIV, {sum, bv2});
        auto r = solver.make_term(smt::Kind::BV_EXTRACT, {half}, {N - 1, 0});
        (void) bvN;
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    // ── §12.4 Bit-manipulation ──────────────────────────────

    class ParityIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &intr, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        uint32_t pN = paramBitWidth(intr, 0);
        smt::Term x = argVals[0].term;
        auto bv1 = solver.make_bv_sort(1);
        // Build XOR of all bits.
        smt::Term acc = solver.make_term(smt::Kind::BV_EXTRACT, {x}, {0, 0});
        for (uint32_t k = 1; k < pN; ++k) {
          auto bit = solver.make_term(smt::Kind::BV_EXTRACT, {x}, {k, k});
          acc = solver.make_term(smt::Kind::BV_XOR, {acc, bit});
        }
        (void) bv1;
        return SymbolicValue(SymbolicValue::Kind::Int, acc, solver.make_true());
      }
    };

    class BswapIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        smt::Term x = argVals[0].term;
        if (N == 8) {
          // 1-byte bswap is the identity.
          return SymbolicValue(SymbolicValue::Kind::Int, x, solver.make_true());
        }
        // Concat the bytes of x in reverse order.
        uint32_t nbytes = N / 8;
        // Extract byte 0 first (lowest), then walk up; build CONCAT(high...low)
        // = byte_0, byte_1, ..., byte_{n-1}. We want byte_0 to land at the
        // highest position, so build by repeated CONCAT(prev, byte_k).
        auto result = solver.make_term(smt::Kind::BV_EXTRACT, {x}, {7, 0});
        for (uint32_t k = 1; k < nbytes; ++k) {
          uint32_t hi = (k + 1) * 8 - 1;
          uint32_t lo = k * 8;
          auto byte = solver.make_term(smt::Kind::BV_EXTRACT, {x}, {hi, lo});
          // CONCAT places first operand at the high end:
          //   result' = CONCAT(result_so_far, byte_k) ... wait, we want byte_0
          // to be the most-significant in the final. So we should build with
          // byte_0 as the LAST CONCAT'd at the LOW end. Let's invert:
          //   final = byte_0 :: byte_1 :: ... :: byte_{n-1}
          // CONCAT(a,b) puts a in high bits, b in low bits.
          // We start with byte_0 (intended high), then CONCAT(current, byte_k)
          // appends byte_k at the low end each iteration.
          result = solver.make_term(smt::Kind::BV_CONCAT, {result, byte});
        }
        return SymbolicValue(SymbolicValue::Kind::Int, result, solver.make_true());
      }
    };

    class BitreverseIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        smt::Term x = argVals[0].term;
        if (N == 1) {
          return SymbolicValue(SymbolicValue::Kind::Int, x, solver.make_true());
        }
        // CONCAT bit_0, bit_1, ..., bit_{N-1} — bit_0 ends up in the high
        // position because CONCAT's first operand is high.
        auto result = solver.make_term(smt::Kind::BV_EXTRACT, {x}, {0, 0});
        for (uint32_t k = 1; k < N; ++k) {
          auto bit = solver.make_term(smt::Kind::BV_EXTRACT, {x}, {k, k});
          result = solver.make_term(smt::Kind::BV_CONCAT, {result, bit});
        }
        return SymbolicValue(SymbolicValue::Kind::Int, result, solver.make_true());
      }
    };

    /**
     * @brief Common UB check + rotation construction for @rotl / @rotr.
     * @param leftRot true for rotate-left; false for rotate-right.
     */
    static smt::Term emitRotation(
        bool leftRot, uint32_t N, smt::Term x, smt::Term n, smt::Sort bvN, smt::ISolver &solver,
        std::vector<smt::Term> &pc
    ) {
      auto zero = solver.make_bv_value_int64(bvN, 0);
      auto nWide = solver.make_bv_value_int64(bvN, N);
      pc.push_back(solver.make_term(smt::Kind::BV_SGE, {n, zero}));
      pc.push_back(solver.make_term(smt::Kind::BV_SLT, {n, nWide}));
      // Use shift-and-or formulation. For n == 0, the inverse shift by N
      // would be out of range; guard with ITE.
      auto eq0 = solver.make_term(smt::Kind::EQUAL, {n, zero});
      auto leftPart = solver.make_term(leftRot ? smt::Kind::BV_SHL : smt::Kind::BV_SHR, {x, n});
      auto nInv = solver.make_term(smt::Kind::BV_SUB, {nWide, n});
      auto rightPart = solver.make_term(leftRot ? smt::Kind::BV_SHR : smt::Kind::BV_SHL, {x, nInv});
      auto combined = solver.make_term(smt::Kind::BV_OR, {leftPart, rightPart});
      return solver.make_term(smt::Kind::ITE, {eq0, x, combined});
    }

    class RotlIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        auto r = emitRotation(true, N, argVals[0].term, argVals[1].term, bvN, solver, pc);
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class RotrIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        auto r = emitRotation(false, N, argVals[0].term, argVals[1].term, bvN, solver, pc);
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class IsPow2Intrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &intr, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        uint32_t pN = paramBitWidth(intr, 0);
        auto bvP = solver.make_bv_sort(pN);
        auto bv1 = solver.make_bv_sort(1);
        smt::Term x = argVals[0].term;
        auto zeroP = solver.make_bv_value_int64(bvP, 0);
        auto oneP = solver.make_bv_value_int64(bvP, 1);
        auto isPositive = solver.make_term(smt::Kind::BV_SGT, {x, zeroP});
        auto xMinus1 = solver.make_term(smt::Kind::BV_SUB, {x, oneP});
        auto andRes = solver.make_term(smt::Kind::BV_AND, {x, xMinus1});
        auto andIsZero = solver.make_term(smt::Kind::EQUAL, {andRes, zeroP});
        auto bothTrue = solver.make_term(smt::Kind::AND, {isPositive, andIsZero});
        auto one1 = solver.make_bv_value_int64(bv1, 1);
        auto zero1 = solver.make_bv_value_int64(bv1, 0);
        auto r = solver.make_term(smt::Kind::ITE, {bothTrue, one1, zero1});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class Ilog2Intrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        smt::Term x = argVals[0].term;
        auto zero = solver.make_bv_value_int64(bvN, 0);
        // UB: x > 0 (signed).
        pc.push_back(solver.make_term(smt::Kind::BV_SGT, {x, zero}));

        // Build clz ITE chain, then return (N - 1) - clz. Because x > 0, the
        // sign bit is 0; the highest set bit is at most N-2, so result is in
        // [0, N-2].
        smt::Term clz = solver.make_bv_value_int64(bvN, N);
        auto one1 = solver.make_bv_value_int64(solver.make_bv_sort(1), 1);
        for (int i = (int) N - 1; i >= 0; --i) {
          int bitPos = (int) N - 1 - i;
          auto bit =
              solver.make_term(smt::Kind::BV_EXTRACT, {x}, {(uint32_t) bitPos, (uint32_t) bitPos});
          auto cond = solver.make_term(smt::Kind::EQUAL, {bit, one1});
          auto iT = solver.make_bv_value_int64(bvN, i);
          clz = solver.make_term(smt::Kind::ITE, {cond, iT, clz});
        }
        auto nMinus1 = solver.make_bv_value_int64(bvN, (int64_t) N - 1);
        auto r = solver.make_term(smt::Kind::BV_SUB, {nMinus1, clz});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    // ── §12.5 Integer overflow-aware family ────────

    /**
     * @brief @wrapping_add(a, b) — BV_ADD already operates modulo 2^N.
     */
    class WrappingAddIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        auto r = solver.make_term(smt::Kind::BV_ADD, {argVals[0].term, argVals[1].term});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    /**
     * @brief @wrapping_sub(a, b) — BV_SUB is modular.
     */
    class WrappingSubIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        auto r = solver.make_term(smt::Kind::BV_SUB, {argVals[0].term, argVals[1].term});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    /**
     * @brief @wrapping_mul(a, b) — BV_MUL is modular.
     */
    class WrappingMulIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        auto r = solver.make_term(smt::Kind::BV_MUL, {argVals[0].term, argVals[1].term});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    /**
     * @brief @wrapping_neg(x) — BV_NEG is modular; INT_MIN_N is fixed.
     */
    class WrappingNegIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t, const std::vector<SymbolicValue> &argVals, smt::Sort,
          smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        auto r = solver.make_term(smt::Kind::BV_NEG, {argVals[0].term});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    /**
     * @brief @wrapping_shl(x, n) — `n` is signed; UB outside [0, N).
     */
    class WrappingShlIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        smt::Term x = argVals[0].term, n = argVals[1].term;
        auto zero = solver.make_bv_value_int64(bvN, 0);
        auto nN = solver.make_bv_value_int64(bvN, (int64_t) N);
        pc.push_back(solver.make_term(smt::Kind::BV_SGE, {n, zero}));
        pc.push_back(solver.make_term(smt::Kind::BV_SLT, {n, nN}));
        auto r = solver.make_term(smt::Kind::BV_SHL, {x, n});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    /**
     * @brief @wrapping_shr(x, n) — arithmetic right shift; same UB.
     */
    class WrappingShrIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        smt::Term x = argVals[0].term, n = argVals[1].term;
        auto zero = solver.make_bv_value_int64(bvN, 0);
        auto nN = solver.make_bv_value_int64(bvN, (int64_t) N);
        pc.push_back(solver.make_term(smt::Kind::BV_SGE, {n, zero}));
        pc.push_back(solver.make_term(smt::Kind::BV_SLT, {n, nN}));
        auto r = solver.make_term(smt::Kind::BV_ASHR, {x, n});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    /**
     * @brief @saturating_add(a, b) — uses BV_SADD_OVERFLOW: when the true
     * sum overflows, return INT_MAX_N if `a >= 0` else INT_MIN_N.
     */
    class SaturatingAddIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        smt::Term a = argVals[0].term, b = argVals[1].term;
        int64_t maxN = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        int64_t minN = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        auto hi = solver.make_bv_value_int64(bvN, maxN);
        auto lo = solver.make_bv_value_int64(bvN, minN);
        auto sum = solver.make_term(smt::Kind::BV_ADD, {a, b});
        auto ov = solver.make_term(smt::Kind::BV_SADD_OVERFLOW, {a, b});
        auto zero = solver.make_bv_value_int64(bvN, 0);
        auto aNonNeg = solver.make_term(smt::Kind::BV_SGE, {a, zero});
        auto satEdge = solver.make_term(smt::Kind::ITE, {aNonNeg, hi, lo});
        auto r = solver.make_term(smt::Kind::ITE, {ov, satEdge, sum});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    /**
     * @brief @saturating_sub(a, b) — analogous to @saturating_add via
     * BV_SSUB_OVERFLOW.  When overflow fires, `a` non-negative implies the
     * difference overshoots `INT_MAX_N`.
     */
    class SaturatingSubIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        smt::Term a = argVals[0].term, b = argVals[1].term;
        int64_t maxN = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        int64_t minN = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        auto hi = solver.make_bv_value_int64(bvN, maxN);
        auto lo = solver.make_bv_value_int64(bvN, minN);
        auto diff = solver.make_term(smt::Kind::BV_SUB, {a, b});
        auto ov = solver.make_term(smt::Kind::BV_SSUB_OVERFLOW, {a, b});
        auto zero = solver.make_bv_value_int64(bvN, 0);
        auto aNonNeg = solver.make_term(smt::Kind::BV_SGE, {a, zero});
        auto satEdge = solver.make_term(smt::Kind::ITE, {aNonNeg, hi, lo});
        auto r = solver.make_term(smt::Kind::ITE, {ov, satEdge, diff});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    /**
     * @brief @saturating_mul(a, b) — uses BV_SMUL_OVERFLOW.  The sign of
     * the saturated bound is `sign(a) XOR sign(b)`: result-positive maps
     * to `INT_MAX_N`, result-negative to `INT_MIN_N`.
     */
    class SaturatingMulIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        smt::Term a = argVals[0].term, b = argVals[1].term;
        int64_t maxN = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        int64_t minN = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        auto hi = solver.make_bv_value_int64(bvN, maxN);
        auto lo = solver.make_bv_value_int64(bvN, minN);
        auto zero = solver.make_bv_value_int64(bvN, 0);
        auto prod = solver.make_term(smt::Kind::BV_MUL, {a, b});
        auto ov = solver.make_term(smt::Kind::BV_SMUL_OVERFLOW, {a, b});
        auto aNonNeg = solver.make_term(smt::Kind::BV_SGE, {a, zero});
        auto bNonNeg = solver.make_term(smt::Kind::BV_SGE, {b, zero});
        // True-product sign positive iff signs of a and b agree.
        auto sameSign = solver.make_term(smt::Kind::EQUAL, {aNonNeg, bNonNeg});
        auto satEdge = solver.make_term(smt::Kind::ITE, {sameSign, hi, lo});
        auto r = solver.make_term(smt::Kind::ITE, {ov, satEdge, prod});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    /**
     * @brief @saturating_neg(x) — `-x`, with `INT_MIN_N` saturated to
     * `INT_MAX_N`.
     */
    class SaturatingNegIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &
      ) const override {
        smt::Term x = argVals[0].term;
        int64_t maxN = (N == 64) ? INT64_MAX : ((INT64_C(1) << (N - 1)) - 1);
        int64_t minN = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        auto hi = solver.make_bv_value_int64(bvN, maxN);
        auto minT = solver.make_bv_value_int64(bvN, minN);
        auto neg = solver.make_term(smt::Kind::BV_NEG, {x});
        auto isMin = solver.make_term(smt::Kind::EQUAL, {x, minT});
        auto r = solver.make_term(smt::Kind::ITE, {isMin, hi, neg});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    /**
     * @brief @div_euclid(a, b) — adjust the truncating quotient by `-1`
     * when the truncated remainder is negative (matches Rust's signed
     * `div_euclid`).  UB on `b == 0` and on `a == INT_MIN_N && b == -1`.
     */
    class DivEuclidIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        smt::Term a = argVals[0].term, b = argVals[1].term;
        auto zero = solver.make_bv_value_int64(bvN, 0);
        auto one = solver.make_bv_value_int64(bvN, 1);
        auto minusOne = solver.make_bv_value_int64(bvN, -1);
        int64_t minN = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        auto minT = solver.make_bv_value_int64(bvN, minN);
        pc.push_back(solver.make_term(smt::Kind::DISTINCT, {b, zero}));
        auto aMin = solver.make_term(smt::Kind::EQUAL, {a, minT});
        auto bNeg1 = solver.make_term(smt::Kind::EQUAL, {b, minusOne});
        pc.push_back(
            solver.make_term(smt::Kind::NOT, {solver.make_term(smt::Kind::AND, {aMin, bNeg1})})
        );
        auto q = solver.make_term(smt::Kind::BV_SDIV, {a, b});
        auto r = solver.make_term(smt::Kind::BV_SREM, {a, b});
        auto rNeg = solver.make_term(smt::Kind::BV_SLT, {r, zero});
        auto bPos = solver.make_term(smt::Kind::BV_SGT, {b, zero});
        // Step the quotient: -1 if b > 0, +1 if b < 0.
        auto step = solver.make_term(smt::Kind::ITE, {bPos, minusOne, one});
        auto qAdj = solver.make_term(smt::Kind::BV_ADD, {q, step});
        auto out = solver.make_term(smt::Kind::ITE, {rNeg, qAdj, q});
        return SymbolicValue(SymbolicValue::Kind::Int, out, solver.make_true());
      }
    };

    /**
     * @brief @rem_euclid(a, b) — add `|b|` to the truncated remainder when
     * it is negative.
     */
    class RemEuclidIntrinsic final : public SolverIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, uint32_t N, const std::vector<SymbolicValue> &argVals,
          smt::Sort bvN, smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        smt::Term a = argVals[0].term, b = argVals[1].term;
        auto zero = solver.make_bv_value_int64(bvN, 0);
        auto minusOne = solver.make_bv_value_int64(bvN, -1);
        int64_t minN = (N == 64) ? INT64_MIN : -(INT64_C(1) << (N - 1));
        auto minT = solver.make_bv_value_int64(bvN, minN);
        pc.push_back(solver.make_term(smt::Kind::DISTINCT, {b, zero}));
        auto aMin = solver.make_term(smt::Kind::EQUAL, {a, minT});
        auto bNeg1 = solver.make_term(smt::Kind::EQUAL, {b, minusOne});
        pc.push_back(
            solver.make_term(smt::Kind::NOT, {solver.make_term(smt::Kind::AND, {aMin, bNeg1})})
        );
        auto r = solver.make_term(smt::Kind::BV_SREM, {a, b});
        auto rNeg = solver.make_term(smt::Kind::BV_SLT, {r, zero});
        auto bPos = solver.make_term(smt::Kind::BV_SGT, {b, zero});
        auto bNeg = solver.make_term(smt::Kind::BV_NEG, {b});
        auto bAbs = solver.make_term(smt::Kind::ITE, {bPos, b, bNeg});
        auto rPlus = solver.make_term(smt::Kind::BV_ADD, {r, bAbs});
        auto out = solver.make_term(smt::Kind::ITE, {rNeg, rPlus, r});
        return SymbolicValue(SymbolicValue::Kind::Int, out, solver.make_true());
      }
    };

    // ── §12.6 Floating-point sign / bit ops ────

    /**
     * @brief FP-intrinsic base class. Receives the IntrinsicDecl directly
     * so each impl can derive its return sort and per-arg sorts.
     */
    class SolverFpIntrinsic {
    public:
      virtual ~SolverFpIntrinsic() = default;
      virtual SymbolicValue solve(
          const IntrinsicDecl &intr, const std::vector<SymbolicValue> &argVals,
          smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const = 0;

    protected:
      static std::pair<uint32_t, uint32_t> fpDims(const TypePtr &t) {
        return std::get<FloatType>(t->v).kind == FloatType::Kind::F32
                   ? std::pair<uint32_t, uint32_t>{8, 24}
                   : std::pair<uint32_t, uint32_t>{11, 53};
      }

      static smt::Sort fpSortOf(const TypePtr &t, smt::ISolver &solver) {
        auto [e, s] = fpDims(t);
        return solver.make_fp_sort(e, s);
      }

      static uint32_t paramBvWidth(const IntrinsicDecl &intr, size_t i) {
        return std::get<FloatType>(intr.params[i].type->v).kind == FloatType::Kind::F32 ? 32 : 64;
      }

      // Source of unique names for fresh BV constants used by @to_bits.
      static std::atomic<uint64_t> &freshCounter() {
        static std::atomic<uint64_t> c{0};
        return c;
      }
    };

    // Float twin of ObserveIntrinsic — the beacon's value semantics is the
    // identity in every domain; only the C lowering writes the sink.
    class ObserveSolverFpIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &,
          std::vector<smt::Term> &
      ) const override {
        return argVals[0];
      }
    };

    class FabsSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
          std::vector<smt::Term> &
      ) const override {
        auto r = solver.make_term(smt::Kind::FP_ABS, {argVals[0].term});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class FnegSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
          std::vector<smt::Term> &
      ) const override {
        auto r = solver.make_term(smt::Kind::FP_NEG, {argVals[0].term});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class CopysignSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
          std::vector<smt::Term> &
      ) const override {
        // copysign(x, y) = if signbit(x) == signbit(y) then x else fp.neg(x).
        // Encoded as: ITE(EQUAL(isNeg(x), isNeg(y)), x, fp.neg(x)).
        auto sx = solver.make_term(smt::Kind::FP_IS_NEG, {argVals[0].term});
        auto sy = solver.make_term(smt::Kind::FP_IS_NEG, {argVals[1].term});
        auto sameSign = solver.make_term(smt::Kind::EQUAL, {sx, sy});
        auto negX = solver.make_term(smt::Kind::FP_NEG, {argVals[0].term});
        auto r = solver.make_term(smt::Kind::ITE, {sameSign, argVals[0].term, negX});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    // ── §12.6 D.3 min / max ───────────────────────────────────────────
    //
    // SMT-LIB's `fp.min` / `fp.max` are implementation-defined on the
    // signed-zero operand pair, so we encode the explicit IEEE 754-2008
    // tie-break that matches the C and WASM backends and the
    // interpreter: prefer -0 for fmin, +0 for fmax.

    class FminSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
          std::vector<smt::Term> &
      ) const override {
        auto x = argVals[0].term, y = argVals[1].term;
        auto xLt = solver.make_term(smt::Kind::FP_LT, {x, y});
        auto yLt = solver.make_term(smt::Kind::FP_LT, {y, x});
        auto xNeg = solver.make_term(smt::Kind::FP_IS_NEG, {x});
        // tie: pick -0 when equal — FP_IS_NEG is true for x = -0.
        auto tie = solver.make_term(smt::Kind::ITE, {xNeg, x, y});
        auto inner = solver.make_term(smt::Kind::ITE, {yLt, y, tie});
        auto r = solver.make_term(smt::Kind::ITE, {xLt, x, inner});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class FmaxSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
          std::vector<smt::Term> &
      ) const override {
        auto x = argVals[0].term, y = argVals[1].term;
        auto xGt = solver.make_term(smt::Kind::FP_GT, {x, y});
        auto yGt = solver.make_term(smt::Kind::FP_GT, {y, x});
        auto xNeg = solver.make_term(smt::Kind::FP_IS_NEG, {x});
        // tie: pick +0 when equal — when x = -0, the other operand y is +0.
        auto tie = solver.make_term(smt::Kind::ITE, {xNeg, y, x});
        auto inner = solver.make_term(smt::Kind::ITE, {yGt, y, tie});
        auto r = solver.make_term(smt::Kind::ITE, {xGt, x, inner});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class SignbitSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
          std::vector<smt::Term> &
      ) const override {
        auto bv1 = solver.make_bv_sort(1);
        auto one = solver.make_bv_value_int64(bv1, 1);
        auto zero = solver.make_bv_value_int64(bv1, 0);
        auto isNeg = solver.make_term(smt::Kind::FP_IS_NEG, {argVals[0].term});
        auto r = solver.make_term(smt::Kind::ITE, {isNeg, one, zero});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    // ── v0.2.2 extra batch D.2 — classification predicates (§12.6) ──

    class IsNormalSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
          std::vector<smt::Term> &
      ) const override {
        auto bv1 = solver.make_bv_sort(1);
        auto one = solver.make_bv_value_int64(bv1, 1);
        auto zero = solver.make_bv_value_int64(bv1, 0);
        auto pred = solver.make_term(smt::Kind::FP_IS_NORMAL, {argVals[0].term});
        auto r = solver.make_term(smt::Kind::ITE, {pred, one, zero});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class IsSubnormalSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
          std::vector<smt::Term> &
      ) const override {
        auto bv1 = solver.make_bv_sort(1);
        auto one = solver.make_bv_value_int64(bv1, 1);
        auto zero = solver.make_bv_value_int64(bv1, 0);
        auto pred = solver.make_term(smt::Kind::FP_IS_SUBNORMAL, {argVals[0].term});
        auto r = solver.make_term(smt::Kind::ITE, {pred, one, zero});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class ToBitsSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &intr, const std::vector<SymbolicValue> &argVals,
          smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        // Introduce a fresh BV `b` such that x == ((_ to_fp eb sb) b), then
        // return `b`. Works because the RefractIR finite-only domain guarantees
        // every valid FP value has a unique IEEE bit pattern.
        uint32_t width = paramBvWidth(intr, 0);
        auto [eb, sb] = fpDims(intr.params[0].type);
        auto bvSort = solver.make_bv_sort(width);
        std::string fresh = "@to_bits$" + std::to_string(freshCounter().fetch_add(1));
        auto b = solver.make_const(bvSort, fresh);
        auto reconstructed = solver.make_term(smt::Kind::FP_TO_FP_FROM_BV, {b}, {eb, sb});
        auto eq = solver.make_term(smt::Kind::FP_EQUAL, {argVals[0].term, reconstructed});
        pc.push_back(eq);
        return SymbolicValue(SymbolicValue::Kind::Int, b, solver.make_true());
      }
    };

    class FromBitsSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &intr, const std::vector<SymbolicValue> &argVals,
          smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        auto [eb, sb] = fpDims(intr.retType);
        auto r = solver.make_term(smt::Kind::FP_TO_FP_FROM_BV, {argVals[0].term}, {eb, sb});
        // UB if non-finite (§2.9): conjoin (not isInf) and (not isNaN).
        auto notInf =
            solver.make_term(smt::Kind::NOT, {solver.make_term(smt::Kind::FP_IS_INF, {r})});
        auto notNaN =
            solver.make_term(smt::Kind::NOT, {solver.make_term(smt::Kind::FP_IS_NAN, {r})});
        pc.push_back(notInf);
        pc.push_back(notNaN);
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    // ── §12.6 D.4 correctly-rounded math ──────────────────────────────────
    //
    // @sqrt maps to fp.sqrt (RNE auto-supplied by the backend); a strictly
    // negative operand yields NaN, which is UB under §2.9, so we conjoin
    // (not isNaN(result)) to the path condition.  @floor/@ceil/@trunc map to
    // fp.roundToIntegral with the matching rounding mode (RTN/RTP/RTZ) and
    // never leave the finite domain — no UB guard.

    class SqrtSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
          std::vector<smt::Term> &pc
      ) const override {
        auto r = solver.make_term(smt::Kind::FP_SQRT, {argVals[0].term});
        // UB if the result is non-finite (§2.9).  A strictly-negative operand
        // makes fp.sqrt NaN; +∞ is unreachable for sqrt but conjoined anyway
        // to enforce the same uniform "result must be finite" contract as the
        // FP arithmetic ops (float.md §10) and @from_bits.
        auto notInf =
            solver.make_term(smt::Kind::NOT, {solver.make_term(smt::Kind::FP_IS_INF, {r})});
        auto notNaN =
            solver.make_term(smt::Kind::NOT, {solver.make_term(smt::Kind::FP_IS_NAN, {r})});
        pc.push_back(notInf);
        pc.push_back(notNaN);
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    // Shared helper for the three integral-rounding intrinsics: fp.rti with a
    // fixed rounding mode supplied as the first child (so the backend does not
    // fall back to its default RNE).
    class RtiSolverIntrinsic : public SolverFpIntrinsic {
    public:
      explicit RtiSolverIntrinsic(smt::RoundingMode rm) : rm_(rm) {}

      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
          std::vector<smt::Term> &
      ) const override {
        auto rmTerm = solver.make_rm_value(rm_);
        auto r = solver.make_term(smt::Kind::FP_RTI, {rmTerm, argVals[0].term});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }

    private:
      smt::RoundingMode rm_;
    };

    // ── §12.6 D.5 compositions ─────────────────────────────────────────────
    //
    // @fract(x) = x - trunc(x); the result is always finite (|.| < 1), so —
    // like @trunc — no UB guard.  @recip(x) = 1/x reuses the FP `/` op and
    // carries the same result-finiteness guard the solver applies to every
    // division (UB on x = ±0.0 or reciprocal overflow).

    class FractSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &, const std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
          std::vector<smt::Term> &
      ) const override {
        auto rtz = solver.make_rm_value(smt::RoundingMode::RTZ);
        auto tr = solver.make_term(smt::Kind::FP_RTI, {rtz, argVals[0].term});
        auto r = solver.make_term(smt::Kind::FP_SUB, {argVals[0].term, tr});
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    class RecipSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      SymbolicValue solve(
          const IntrinsicDecl &intr, const std::vector<SymbolicValue> &argVals,
          smt::ISolver &solver, std::vector<smt::Term> &pc
      ) const override {
        auto one = solver.make_fp_value_from_real(
            fpSortOf(intr.retType, solver), 1.0, smt::RoundingMode::RNE
        );
        auto r = solver.make_term(smt::Kind::FP_DIV, {one, argVals[0].term});
        // UB if the reciprocal is non-finite (x = ±0.0 → ±∞, or overflow).
        auto notInf =
            solver.make_term(smt::Kind::NOT, {solver.make_term(smt::Kind::FP_IS_INF, {r})});
        auto notNaN =
            solver.make_term(smt::Kind::NOT, {solver.make_term(smt::Kind::FP_IS_NAN, {r})});
        pc.push_back(notInf);
        pc.push_back(notNaN);
        return SymbolicValue(SymbolicValue::Kind::Int, r, solver.make_true());
      }
    };

    // ── §12.4 Horizontal vector reductions ─────────────────────
    //
    // A reduction folds the lane terms of its sole vector argument
    // (SymbolicValue::Kind::Vec, lanes in arrayVal) into one scalar term
    // via the normative left-to-right order (§9.8: unrolled N-1-step fold,
    // linear, no fresh symbols, no quantifiers).  Every lane is read, so
    // each lane's `is_defined` is conjoined to the UB guards (rule 22).  Two
    // classes split the element domain: the integer fold (bvadd with a
    // per-step signed-overflow guard / bvand|bvor|bvxor / signed-compare
    // ITE min/max) and the floating-point fold (fp.add with a per-step
    // finiteness guard / fp-compare ITE min/max, ties keeping the
    // accumulator, matching the interpreter).  Add / Min / Max register in
    // both; the bitwise reductions are integer-only.

    inline const std::vector<SymbolicValue> &
    reductionLanes(const IntrinsicDecl &intr, const std::vector<SymbolicValue> &argVals) {
      if (argVals.empty() || argVals[0].kind != SymbolicValue::Kind::Vec ||
          argVals[0].arrayVal.empty())
        throw std::runtime_error(
            "Solver reduction " + intr.name.name + ": argument is not a non-empty vector"
        );
      return argVals[0].arrayVal;
    }

    // Pairwise fp min / max fold steps with IEEE minNum/maxNum signed-zero
    // tie-break (min prefers -0, max prefers +0), identical to @fmin/@fmax
    // (FminSolverIntrinsic / FmaxSolverIntrinsic).  §12.4 requires reduce_min
    // / reduce_max to be order-independent so backends may use pairwise /
    // hardware reductions; the canonical tie-break provides that (strict
    // comparison would make ±0 order-observable).
    inline smt::Term fpMinFold(smt::Term acc, smt::Term x, smt::ISolver &solver) {
      auto accLt = solver.make_term(smt::Kind::FP_LT, {acc, x});
      auto xLt = solver.make_term(smt::Kind::FP_LT, {x, acc});
      auto accNeg = solver.make_term(smt::Kind::FP_IS_NEG, {acc});
      auto tie = solver.make_term(smt::Kind::ITE, {accNeg, acc, x});
      auto inner = solver.make_term(smt::Kind::ITE, {xLt, x, tie});
      return solver.make_term(smt::Kind::ITE, {accLt, acc, inner});
    }

    inline smt::Term fpMaxFold(smt::Term acc, smt::Term x, smt::ISolver &solver) {
      auto accGt = solver.make_term(smt::Kind::FP_GT, {acc, x});
      auto xGt = solver.make_term(smt::Kind::FP_GT, {x, acc});
      auto accNeg = solver.make_term(smt::Kind::FP_IS_NEG, {acc});
      auto tie = solver.make_term(smt::Kind::ITE, {accNeg, x, acc});
      auto inner = solver.make_term(smt::Kind::ITE, {xGt, x, tie});
      return solver.make_term(smt::Kind::ITE, {accGt, acc, inner});
    }

    class ReduceIntSolverIntrinsic final : public SolverIntrinsic {
    public:
      explicit ReduceIntSolverIntrinsic(IntrinsicKind kind) : kind_(kind) {}

      SymbolicValue solve(
          const IntrinsicDecl &intr, uint32_t /*N*/, const std::vector<SymbolicValue> &argVals,
          smt::Sort /*bvN*/, smt::ISolver &solver, std::vector<smt::Term> &ub
      ) const override {
        const std::vector<SymbolicValue> &lanes = reductionLanes(intr, argVals);
        ub.push_back(lanes[0].is_defined);
        smt::Term acc = lanes[0].term;
        for (size_t k = 1; k < lanes.size(); ++k) {
          ub.push_back(lanes[k].is_defined);
          smt::Term x = lanes[k].term;
          switch (kind_) {
            case IntrinsicKind::ReduceAdd: {
              // Per-step signed-overflow UB (rule 4), same encoding as the
              // scalar `+` operator (src/solver/expr.cpp).
              auto ov = solver.make_term(smt::Kind::BV_SADD_OVERFLOW, {acc, x});
              ub.push_back(solver.make_term(smt::Kind::NOT, {ov}));
              acc = solver.make_term(smt::Kind::BV_ADD, {acc, x});
              break;
            }
            case IntrinsicKind::ReduceMin: {
              auto lt = solver.make_term(smt::Kind::BV_SLT, {x, acc});
              acc = solver.make_term(smt::Kind::ITE, {lt, x, acc});
              break;
            }
            case IntrinsicKind::ReduceMax: {
              auto gt = solver.make_term(smt::Kind::BV_SGT, {x, acc});
              acc = solver.make_term(smt::Kind::ITE, {gt, x, acc});
              break;
            }
            case IntrinsicKind::ReduceAnd:
              acc = solver.make_term(smt::Kind::BV_AND, {acc, x});
              break;
            case IntrinsicKind::ReduceOr:
              acc = solver.make_term(smt::Kind::BV_OR, {acc, x});
              break;
            case IntrinsicKind::ReduceXor:
              acc = solver.make_term(smt::Kind::BV_XOR, {acc, x});
              break;
            default:
              throw std::runtime_error("Solver reduction: unexpected integer kind");
          }
        }
        return SymbolicValue(SymbolicValue::Kind::Int, acc, solver.make_true());
      }

    private:
      IntrinsicKind kind_;
    };

    class ReduceFpSolverIntrinsic final : public SolverFpIntrinsic {
    public:
      explicit ReduceFpSolverIntrinsic(IntrinsicKind kind) : kind_(kind) {}

      SymbolicValue solve(
          const IntrinsicDecl &intr, const std::vector<SymbolicValue> &argVals,
          smt::ISolver &solver, std::vector<smt::Term> &ub
      ) const override {
        const std::vector<SymbolicValue> &lanes = reductionLanes(intr, argVals);
        ub.push_back(lanes[0].is_defined);
        smt::Term acc = lanes[0].term;
        auto rmRNE = solver.make_rm_value(smt::RoundingMode::RNE);
        for (size_t k = 1; k < lanes.size(); ++k) {
          ub.push_back(lanes[k].is_defined);
          smt::Term x = lanes[k].term;
          switch (kind_) {
            case IntrinsicKind::ReduceAdd:
              // Per-step RNE add with a finiteness UB guard (rules 6-7),
              // same as the scalar `+` operator.
              acc = solver.make_term(smt::Kind::FP_ADD, {rmRNE, acc, x});
              assertFPFinite(acc, solver, ub);
              break;
            case IntrinsicKind::ReduceMin:
              acc = fpMinFold(acc, x, solver);
              break;
            case IntrinsicKind::ReduceMax:
              acc = fpMaxFold(acc, x, solver);
              break;
            default:
              throw std::runtime_error("Solver reduction: unexpected float kind");
          }
        }
        return SymbolicValue(SymbolicValue::Kind::Int, acc, solver.make_true());
      }

    private:
      IntrinsicKind kind_;
    };

    class SolverFpIntrinsicRegistry {
    public:
      static const SolverFpIntrinsicRegistry &get() {
        static SolverFpIntrinsicRegistry instance;
        return instance;
      }

      const SolverFpIntrinsic *lookup(IntrinsicKind kind) const {
        auto it = registry_.find(kind);
        return it != registry_.end() ? it->second.get() : nullptr;
      }

    private:
      SolverFpIntrinsicRegistry() {
        registry_[IntrinsicKind::Observe] = std::make_unique<ObserveSolverFpIntrinsic>();
        registry_[IntrinsicKind::Fabs] = std::make_unique<FabsSolverIntrinsic>();
        registry_[IntrinsicKind::Fneg] = std::make_unique<FnegSolverIntrinsic>();
        registry_[IntrinsicKind::Copysign] = std::make_unique<CopysignSolverIntrinsic>();
        registry_[IntrinsicKind::Signbit] = std::make_unique<SignbitSolverIntrinsic>();
        registry_[IntrinsicKind::ToBits] = std::make_unique<ToBitsSolverIntrinsic>();
        registry_[IntrinsicKind::IsNormal] = std::make_unique<IsNormalSolverIntrinsic>();
        registry_[IntrinsicKind::Fmin] = std::make_unique<FminSolverIntrinsic>();
        registry_[IntrinsicKind::Fmax] = std::make_unique<FmaxSolverIntrinsic>();
        registry_[IntrinsicKind::IsSubnormal] = std::make_unique<IsSubnormalSolverIntrinsic>();
        registry_[IntrinsicKind::FromBits] = std::make_unique<FromBitsSolverIntrinsic>();
        // §12.6 D.4 — correctly-rounded math.
        registry_[IntrinsicKind::Sqrt] = std::make_unique<SqrtSolverIntrinsic>();
        registry_[IntrinsicKind::Floor] =
            std::make_unique<RtiSolverIntrinsic>(smt::RoundingMode::RTN);
        registry_[IntrinsicKind::Ceil] =
            std::make_unique<RtiSolverIntrinsic>(smt::RoundingMode::RTP);
        registry_[IntrinsicKind::Trunc] =
            std::make_unique<RtiSolverIntrinsic>(smt::RoundingMode::RTZ);
        // §12.6 D.5 — compositions.
        registry_[IntrinsicKind::Fract] = std::make_unique<FractSolverIntrinsic>();
        registry_[IntrinsicKind::Recip] = std::make_unique<RecipSolverIntrinsic>();
        // §12.4 — floating-point horizontal reductions. Only the
        // arithmetic add / min / max fold FP lanes; the bitwise reductions
        // are integer-only and live solely in the integer registry.
        registry_[IntrinsicKind::ReduceAdd] =
            std::make_unique<ReduceFpSolverIntrinsic>(IntrinsicKind::ReduceAdd);
        registry_[IntrinsicKind::ReduceMin] =
            std::make_unique<ReduceFpSolverIntrinsic>(IntrinsicKind::ReduceMin);
        registry_[IntrinsicKind::ReduceMax] =
            std::make_unique<ReduceFpSolverIntrinsic>(IntrinsicKind::ReduceMax);
      }

      std::unordered_map<IntrinsicKind, std::unique_ptr<SolverFpIntrinsic>> registry_;
    };

    // ── Registry ────────────────────────────────────────────────────────────

    class IntrinsicRegistry {
    public:
      static const IntrinsicRegistry &get() {
        static IntrinsicRegistry instance;
        return instance;
      }

      const SolverIntrinsic *lookup(IntrinsicKind kind) const {
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
        // §12.4 — integer horizontal reductions. All six kinds
        // fold integer lanes; add / min / max additionally have an FP
        // overload in the SolverFpIntrinsicRegistry above.
        registry_[IntrinsicKind::ReduceAdd] =
            std::make_unique<ReduceIntSolverIntrinsic>(IntrinsicKind::ReduceAdd);
        registry_[IntrinsicKind::ReduceMin] =
            std::make_unique<ReduceIntSolverIntrinsic>(IntrinsicKind::ReduceMin);
        registry_[IntrinsicKind::ReduceMax] =
            std::make_unique<ReduceIntSolverIntrinsic>(IntrinsicKind::ReduceMax);
        registry_[IntrinsicKind::ReduceAnd] =
            std::make_unique<ReduceIntSolverIntrinsic>(IntrinsicKind::ReduceAnd);
        registry_[IntrinsicKind::ReduceOr] =
            std::make_unique<ReduceIntSolverIntrinsic>(IntrinsicKind::ReduceOr);
        registry_[IntrinsicKind::ReduceXor] =
            std::make_unique<ReduceIntSolverIntrinsic>(IntrinsicKind::ReduceXor);
        registry_[IntrinsicKind::Observe] = std::make_unique<ObserveIntrinsic>();
      }

      std::unordered_map<IntrinsicKind, std::unique_ptr<SolverIntrinsic>> registry_;
    };

  } // namespace

  SymbolicValue SymbolicExecutor::callBuiltinIntrinsicSMT(
      const IntrinsicDecl &intr, std::vector<SymbolicValue> &argVals, smt::ISolver &solver,
      std::vector<smt::Term> & /*pc*/, std::vector<smt::Term> &ub
  ) {
    auto kind = getIntrinsicKind(intr.name.name);
    if (!kind)
      throw std::runtime_error("Solver: unknown intrinsic " + intr.name.name);

    // FP-touching dispatch first: the integer path's
    // bvN = bv_sort(retBits) only fits integer-return intrinsics; FP-return
    // and FP-param intrinsics route through their own registry.
    bool anyFp = (intr.retType && std::holds_alternative<FloatType>(intr.retType->v));
    for (const auto &p: intr.params)
      anyFp = anyFp || (p.type && std::holds_alternative<FloatType>(p.type->v));

    if (anyFp) {
      if (auto fpImpl = SolverFpIntrinsicRegistry::get().lookup(*kind))
        return fpImpl->solve(intr, argVals, solver, ub);
      throw std::runtime_error("Solver: unknown FP intrinsic " + intr.name.name);
    }

    uint32_t N = 32;
    if (auto pb = TypeUtils::getIntBitWidth(intr.retType))
      N = *pb;
    auto bvN = solver.make_bv_sort(N);
    if (auto impl = IntrinsicRegistry::get().lookup(*kind))
      return impl->solve(intr, N, argVals, bvN, solver, ub);

    throw std::runtime_error("Solver: unknown intrinsic " + intr.name.name);
  }

} // namespace refractir
