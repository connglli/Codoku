// "array" python vec-lowering: a vector local is a plain
// lane list — the historical representation, extracted verbatim so
// the strategy seam is byte-identical for the default.

#include "backend/py_vec_lowering.hpp"

namespace refractir {

  namespace {

    class PyArrayLowering : public PyVecLowering {
    public:
      std::string name() const override { return "array"; }

      void emitPreamble(std::ostream &, const std::vector<VecType> &) override {
        // Lists need no helper classes.
      }

      std::string declUndef(const std::string &name, const VecType &vt) override {
        return name + " = [_UNDEF] * " + std::to_string(vt.size);
      }

      std::string assignFromList(
          const std::string &name, const VecType &, const std::string &listExpr
      ) override {
        return name + " = " + listExpr;
      }

      std::string rawListExpr(const std::string &name, const VecType &) override { return name; }

      std::string readListExpr(const std::string &name, const VecType &vt) override {
        return "_vrd(" + name + ", 0, " + std::to_string(vt.size) + ", 1)";
      }

      std::string
      laneRead(const std::string &name, const VecType &, const std::string &idxExpr) override {
        return "_rd(" + name + ", " + idxExpr + ")";
      }

      std::string laneWrite(
          const std::string &name, const VecType &, const std::string &idxExpr,
          const std::string &valExpr
      ) override {
        return name + "[" + idxExpr + "] = " + valExpr;
      }

      std::string
      wholeCopy(const std::string &lhs, const std::string &rhs, const VecType &) override {
        return lhs + " = list(" + rhs + ")";
      }

      std::string unpackParam(const std::string &, const VecType &) override { return ""; }
    };

  } // namespace

  // Defined in the per-strategy TUs.
  std::unique_ptr<PyVecLowering> makePyScalarsLowering();
  std::unique_ptr<PyVecLowering> makePyStructArrayLowering();
  std::unique_ptr<PyVecLowering> makePyStructScalarsLowering();

  const std::vector<std::string> &pyVecLoweringNames() {
    static const std::vector<std::string> names = {
        "array", "scalars", "structscalars", "structarray"
    };
    return names;
  }

  std::unique_ptr<PyVecLowering> makePyVecLowering(const std::string &name) {
    if (name == "array" || name.empty())
      return std::make_unique<PyArrayLowering>();
    if (name == "scalars")
      return makePyScalarsLowering();
    if (name == "structarray")
      return makePyStructArrayLowering();
    if (name == "structscalars")
      return makePyStructScalarsLowering();
    return nullptr; // "vecext" and unknown names have no python analogue
  }

} // namespace refractir
