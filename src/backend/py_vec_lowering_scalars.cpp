// "scalars" python vec-lowering: a vector `<N> T` local is
// stored as N separate variables `v_0 .. v_{N-1}`. Mirrors the C
// "scalars" strategy — including the refusal of dynamic lane indices
// (a runtime index cannot select a distinct variable). Compute values
// are still lane lists; this strategy converts at read/write points.

#include <stdexcept>
#include "backend/py_vec_lowering.hpp"

namespace refractir {

  namespace {

    // Parse a fully-lowered lane-index expression as a constant. The
    // literal path of PyBackend::laneIdxExpr emits a bare decimal;
    // anything else (an `_idx(...)` guard) is a dynamic index.
    bool constIdx(const std::string &idxExpr, std::uint64_t &out) {
      if (idxExpr.empty())
        return false;
      std::uint64_t v = 0;
      for (char c: idxExpr) {
        if (c < '0' || c > '9')
          return false;
        v = v * 10 + static_cast<std::uint64_t>(c - '0');
      }
      out = v;
      return true;
    }

    class PyScalarsLowering : public PyVecLowering {
    public:
      std::string name() const override { return "scalars"; }

      void emitPreamble(std::ostream &, const std::vector<VecType> &) override {}

      // Lane list `[v_0, v_1, ...]` (raw, may contain _UNDEF).
      std::string rawList(const std::string &name, const VecType &vt) {
        std::string s = "[";
        for (std::uint64_t k = 0; k < vt.size; ++k)
          s += (k ? ", " : "") + name + "_" + std::to_string(k);
        return s + "]";
      }

      std::string declUndef(const std::string &name, const VecType &vt) override {
        // Chained assignment: every lane shares the _UNDEF sentinel
        // (a read-trap marker; never mutated).
        std::string s;
        for (std::uint64_t k = 0; k < vt.size; ++k)
          s += name + "_" + std::to_string(k) + " = ";
        return s + "_UNDEF";
      }

      std::string assignFromList(
          const std::string &name, const VecType &vt, const std::string &listExpr
      ) override {
        // Tuple-unpack the fresh N-element list into the lane vars.
        std::string lhs;
        for (std::uint64_t k = 0; k < vt.size; ++k)
          lhs += (k ? ", " : "") + name + "_" + std::to_string(k);
        return lhs + " = " + listExpr;
      }

      std::string rawListExpr(const std::string &name, const VecType &vt) override {
        return rawList(name, vt);
      }

      std::string readListExpr(const std::string &name, const VecType &vt) override {
        // Undef-check via the existing _vrd over a fresh lane list.
        return "_vrd(" + rawList(name, vt) + ", 0, " + std::to_string(vt.size) + ", 1)";
      }

      std::string
      laneRead(const std::string &name, const VecType &, const std::string &idxExpr) override {
        std::uint64_t k;
        if (!constIdx(idxExpr, k))
          throw std::runtime_error(
              "python vec-lowering 'scalars' does not support dynamic lane index"
          );
        // Undef-check the single lane via _rd over a one-element list.
        return "_rd([" + name + "_" + std::to_string(k) + "], 0)";
      }

      std::string laneWrite(
          const std::string &name, const VecType &, const std::string &idxExpr,
          const std::string &valExpr
      ) override {
        std::uint64_t k;
        if (!constIdx(idxExpr, k))
          throw std::runtime_error(
              "python vec-lowering 'scalars' does not support dynamic lane index"
          );
        return name + "_" + std::to_string(k) + " = " + valExpr;
      }

      std::string
      wholeCopy(const std::string &lhs, const std::string &rhs, const VecType &vt) override {
        // Per-lane copy of raw slots (undef stays undef): a tuple
        // assignment evaluates the whole RHS before binding, so
        // self-copy and overlap are safe.
        std::string l, r;
        for (std::uint64_t k = 0; k < vt.size; ++k) {
          l += (k ? ", " : "") + lhs + "_" + std::to_string(k);
          r += (k ? ", " : "") + rhs + "_" + std::to_string(k);
        }
        return l + " = " + r;
      }

      std::string unpackParam(const std::string &name, const VecType &vt) override {
        // The param arrives as a lane list; spread it into the vars.
        std::string lhs;
        for (std::uint64_t k = 0; k < vt.size; ++k)
          lhs += (k ? ", " : "") + name + "_" + std::to_string(k);
        return lhs + " = " + name;
      }
    };

  } // namespace

  // Registered by makePyVecLowering (src/backend/py_vec_lowering_array.cpp).
  std::unique_ptr<PyVecLowering> makePyScalarsLowering() {
    return std::make_unique<PyScalarsLowering>();
  }

} // namespace refractir
