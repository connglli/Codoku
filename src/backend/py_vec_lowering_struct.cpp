// "structarray" and "structscalars" python vec-lowering.
// A vector `<N> T` local is stored in a per-shape helper class,
// mirroring the C struct strategies:
//   structarray:   class _vec_<N>_<Telem>_arr  with a `lanes` list
//   structscalars: class _vec_<N>_<Telem>_scal with fields l0..l{N-1}
// Compute values remain lane lists; the strategy converts at
// read/write points. structarray supports dynamic lane indices
// (indexes the list); structscalars rejects them, like its C twin.

#include <set>
#include <stdexcept>
#include "backend/py_vec_lowering.hpp"

namespace refractir {

  namespace {

    std::string elemSuffix(const TypePtr &elem) {
      if (auto it = std::get_if<IntType>(&elem->v)) {
        int bits = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
        return "i" + std::to_string(bits);
      }
      if (auto ft = std::get_if<FloatType>(&elem->v))
        return ft->kind == FloatType::Kind::F32 ? "f32" : "f64";
      return "u";
    }

    std::string className(const VecType &vt, bool isArrayShape) {
      return "_vec_" + std::to_string(vt.size) + "_" + elemSuffix(vt.elem) +
             (isArrayShape ? "_arr" : "_scal");
    }

    // Parse a fully-lowered lane-index expression as a constant (see
    // the scalars strategy for the contract).
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

    // structarray: class holding a lane list.
    class PyStructArrayLowering : public PyVecLowering {
    public:
      std::string name() const override { return "structarray"; }

      void emitPreamble(std::ostream &out, const std::vector<VecType> &usedShapes) override {
        std::set<std::string> emitted;
        for (const auto &vt: usedShapes) {
          std::string cn = className(vt, /*isArrayShape=*/true);
          if (!emitted.insert(cn).second)
            continue;
          out << "\n\nclass " << cn << ":\n"
              << "    __slots__ = (\"lanes\",)\n\n"
              << "    def __init__(self, lanes):\n"
              << "        self.lanes = lanes\n";
        }
        if (!emitted.empty())
          out << "\n";
      }

      std::string declUndef(const std::string &name, const VecType &vt) override {
        return name + " = " + className(vt, true) + "([_UNDEF] * " + std::to_string(vt.size) +
               ", 1)";
      }

      std::string assignFromList(
          const std::string &name, const VecType &vt, const std::string &listExpr
      ) override {
        return name + " = " + className(vt, true) + "(" + listExpr + ")";
      }

      // Borrowed internal list (aliases); read-only consumers only.
      std::string rawListExpr(const std::string &name, const VecType &) override {
        return name + ".lanes";
      }

      std::string readListExpr(const std::string &name, const VecType &vt) override {
        return "_vrd(" + name + ".lanes, 0, " + std::to_string(vt.size) + ", 1)";
      }

      std::string
      laneRead(const std::string &name, const VecType &, const std::string &idxExpr) override {
        return "_rd(" + name + ".lanes, " + idxExpr + ")";
      }

      std::string laneWrite(
          const std::string &name, const VecType &, const std::string &idxExpr,
          const std::string &valExpr
      ) override {
        return name + ".lanes[" + idxExpr + "] = " + valExpr;
      }

      std::string
      wholeCopy(const std::string &lhs, const std::string &rhs, const VecType &vt) override {
        // Fresh object + fresh list (raw slots; undef stays undef).
        return lhs + " = " + className(vt, true) + "(list(" + rhs + ".lanes))";
      }

      std::string unpackParam(const std::string &name, const VecType &vt) override {
        // The param arrives as a lane list; wrap it (read-only, so
        // aliasing the caller's list is safe — params are immutable).
        return name + " = " + className(vt, true) + "(" + name + ")";
      }
    };

    // structscalars: class with fields l0..l{N-1}.
    class PyStructScalarsLowering : public PyVecLowering {
    public:
      std::string name() const override { return "structscalars"; }

      void emitPreamble(std::ostream &out, const std::vector<VecType> &usedShapes) override {
        std::set<std::string> emitted;
        for (const auto &vt: usedShapes) {
          std::string cn = className(vt, /*isArrayShape=*/false);
          if (!emitted.insert(cn).second)
            continue;
          std::string slots, params, body;
          for (std::uint64_t k = 0; k < vt.size; ++k) {
            std::string f = "l" + std::to_string(k);
            slots += (k ? ", " : "") + ("\"" + f + "\"");
            params += ", " + f;
            body += "        self." + f + " = " + f + "\n";
          }
          // Single-field tuple needs a trailing comma.
          if (vt.size == 1)
            slots += ",";
          out << "\n\nclass " << cn << ":\n"
              << "    __slots__ = (" << slots << ")\n\n"
              << "    def __init__(self" << params << "):\n"
              << body;
        }
        if (!emitted.empty())
          out << "\n";
      }

      std::string fieldList(const std::string &name, const VecType &vt) {
        std::string s = "[";
        for (std::uint64_t k = 0; k < vt.size; ++k)
          s += (k ? ", " : "") + name + ".l" + std::to_string(k);
        return s + "]";
      }

      std::string declUndef(const std::string &name, const VecType &vt) override {
        std::string args;
        for (std::uint64_t k = 0; k < vt.size; ++k)
          args += (k ? ", " : "") + std::string("_UNDEF");
        return name + " = " + className(vt, false) + "(" + args + ")";
      }

      std::string assignFromList(
          const std::string &name, const VecType &vt, const std::string &listExpr
      ) override {
        // Spread the fresh N-element list into the constructor.
        return name + " = " + className(vt, false) + "(*" + listExpr + ")";
      }

      std::string rawListExpr(const std::string &name, const VecType &vt) override {
        return fieldList(name, vt);
      }

      std::string readListExpr(const std::string &name, const VecType &vt) override {
        return "_vrd(" + fieldList(name, vt) + ", 0, " + std::to_string(vt.size) + ", 1)";
      }

      std::string
      laneRead(const std::string &name, const VecType &, const std::string &idxExpr) override {
        std::uint64_t k;
        if (!constIdx(idxExpr, k))
          throw std::runtime_error(
              "python vec-lowering 'structscalars' does not support dynamic lane index"
          );
        return "_rd([" + name + ".l" + std::to_string(k) + "], 0)";
      }

      std::string laneWrite(
          const std::string &name, const VecType &, const std::string &idxExpr,
          const std::string &valExpr
      ) override {
        std::uint64_t k;
        if (!constIdx(idxExpr, k))
          throw std::runtime_error(
              "python vec-lowering 'structscalars' does not support dynamic lane index"
          );
        return name + ".l" + std::to_string(k) + " = " + valExpr;
      }

      std::string
      wholeCopy(const std::string &lhs, const std::string &rhs, const VecType &vt) override {
        // Fresh object; copies field values (raw slots, undef stays).
        std::string args;
        for (std::uint64_t k = 0; k < vt.size; ++k)
          args += (k ? ", " : "") + rhs + ".l" + std::to_string(k);
        return lhs + " = " + className(vt, false) + "(" + args + ")";
      }

      std::string unpackParam(const std::string &name, const VecType &vt) override {
        return name + " = " + className(vt, false) + "(*" + name + ")";
      }
    };

  } // namespace

  std::unique_ptr<PyVecLowering> makePyStructArrayLowering() {
    return std::make_unique<PyStructArrayLowering>();
  }

  std::unique_ptr<PyVecLowering> makePyStructScalarsLowering() {
    return std::make_unique<PyStructScalarsLowering>();
  }

} // namespace refractir
