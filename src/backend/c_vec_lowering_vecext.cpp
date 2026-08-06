// VecLowering strategy: GCC/Clang vector extensions.
//
// Maps `<N> T` to a typedef of the form
//   typedef T __attribute__((vector_size(N * sizeof(T)))) <name>;
// where <name> is `_vec_<N>_<elem>` (e.g. `_vec_4_i32`). The compiler
// natively supports lane subscript and binary operators on this type,
// which makes most emissions trivially one-liner.
//
// This strategy supports `canCrossFnBoundary` (parameters and returns
// pass the vector by value in SIMD registers per the platform ABI).

#include <set>
#include <sstream>
#include "backend/c_vec_lowering.hpp"

namespace refractir {

  namespace {

    std::string elemSuffix(const TypePtr &elem) {
      if (auto it = std::get_if<IntType>(&elem->v)) {
        int bits = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
        return "i" + std::to_string(bits);
      }
      if (auto ft = std::get_if<FloatType>(&elem->v)) {
        return ft->kind == FloatType::Kind::F32 ? "f32" : "f64";
      }
      return "u"; // unknown — shouldn't happen for well-typed programs
    }

    std::string elemCType(const TypePtr &elem) {
      if (auto it = std::get_if<IntType>(&elem->v)) {
        int bits = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
        if (bits == 1)
          return "int8_t"; // i1 stored as int8 lane (vecext can't have 1-bit lanes)
        if (bits <= 8)
          return "int8_t";
        if (bits <= 16)
          return "int16_t";
        if (bits <= 32)
          return "int32_t";
        return "int64_t";
      }
      if (auto ft = std::get_if<FloatType>(&elem->v)) {
        return ft->kind == FloatType::Kind::F32 ? "float" : "double";
      }
      return "void";
    }

    std::uint64_t elemBytes(const TypePtr &elem) {
      if (auto it = std::get_if<IntType>(&elem->v)) {
        int bits = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
        if (bits == 1)
          return 1; // stored as i8
        if (bits <= 8)
          return 1;
        if (bits <= 16)
          return 2;
        if (bits <= 32)
          return 4;
        return 8;
      }
      if (auto ft = std::get_if<FloatType>(&elem->v)) {
        return ft->kind == FloatType::Kind::F32 ? 4 : 8;
      }
      return 0;
    }

    std::string typeName(const VecType &vt) {
      return "_vec_" + std::to_string(vt.size) + "_" + elemSuffix(vt.elem);
    }

  } // namespace

  class CVecExtLowering : public CVecLowering {
  public:
    std::string name() const override { return "vecext"; }

    void emitPreamble(std::ostream &out, const std::vector<VecType> &usedShapes) override {
      // De-duplicate by (N, elem-suffix).
      std::set<std::string> emitted;
      for (const auto &vt: usedShapes) {
        std::string tn = typeName(vt);
        if (!emitted.insert(tn).second)
          continue;
        std::uint64_t bytes = vt.size * elemBytes(vt.elem);
        out << "typedef " << elemCType(vt.elem) << " " << tn << " __attribute__((vector_size("
            << bytes << ")));\n";
      }
      if (!emitted.empty())
        out << "\n";
    }

    std::string typeString(const VecType &vt) override { return typeName(vt); }

    void emitLocalDecl(std::ostream &out, const std::string &name, const VecType &vt) override {
      out << typeName(vt) << " " << name;
    }

    void emitInit(
        std::ostream &out, const std::string &name, const VecType &vt, const InitVal &iv
    ) override {
      (void) name;
      // Vecext supports both brace and broadcast init via C's `(T){a,b,...}`
      // compound-literal syntax — but the caller (c_backend.cpp) already
      // walks the InitVal and emits the C braces around individual scalar
      // values. To keep this strategy thin and reuse the existing scalar
      // emitInitVal machinery, we just let the caller drive: emitInit is
      // a no-op for VecExt; the caller emits `= { … }` after the local
      // declaration. (Other strategies will override this.)
      (void) out;
      (void) vt;
      (void) iv;
    }

    std::string
    emitLaneRead(const std::string &name, const VecType &vt, const std::string &idxExpr) override {
      (void) vt;
      return name + "[" + idxExpr + "]";
    }

    void emitLaneWrite(
        std::ostream &out, const std::string &name, const VecType &vt, const std::string &idxExpr,
        const std::string &valExpr
    ) override {
      (void) vt;
      out << name << "[" << idxExpr << "] = " << valExpr;
    }

    void emitWholeCopy(
        std::ostream &out, const std::string &lhs, const std::string &rhs, const VecType &vt
    ) override {
      (void) vt;
      out << lhs << " = " << rhs;
    }

    bool canCrossFnBoundary() const override { return true; }

    bool needsLaneUnroll() const override { return false; }
  };

  // Phase-2 strategy constructors live in their own translation units.
  std::unique_ptr<CVecLowering> makeCArrayLowering();
  std::unique_ptr<CVecLowering> makeCScalarsLowering();
  std::unique_ptr<CVecLowering> makeCStructArrayLowering();
  std::unique_ptr<CVecLowering> makeCStructScalarsLowering();

  const std::vector<std::string> &cVecLoweringNames() {
    static const std::vector<std::string> names = {
        "vecext", "scalars", "array", "structscalars", "structarray"
    };
    return names;
  }

  std::unique_ptr<CVecLowering> makeCVecLowering(const std::string &name) {
    if (name == "vecext")
      return std::make_unique<CVecExtLowering>();
    if (name == "array")
      return makeCArrayLowering();
    if (name == "scalars")
      return makeCScalarsLowering();
    if (name == "structarray")
      return makeCStructArrayLowering();
    if (name == "structscalars")
      return makeCStructScalarsLowering();
    return nullptr;
  }

} // namespace refractir
