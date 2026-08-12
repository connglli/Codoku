#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "analysis/cfg.hpp"
#include "analysis/intrinsics.hpp"
#include "analysis/structurizer.hpp"
#include "ast/ast.hpp"
#include "backend/py_vec_lowering.hpp"

namespace refractir {

  /**
   * Generates Python code from a RefractIR program.
   *
   * Python has no goto and no labeled break, so this backend only
   * accepts reducible CFGs (the driver registers ReducibilityCheck for
   * --target python): each function is structured via
   * Structurizer::build + StructuredLowering::run and printed as
   * genuine while/if statements — the backend itself is a thin syntax
   * printer over the lowered ControlTree.
   *
   * Semantics: RefractIR's strict UB is checked eagerly at runtime by
   * a small helper preamble (RefractIRTrap): checked signed iN
   * arithmetic (Python ints are unbounded, so overflow must trap
   * explicitly), truncating division/remainder (Python's floor `//`
   * and `%` must not be used raw), per-operation f32 rounding via a
   * struct pack/unpack round-trip, and finiteness checks after every
   * FP operation. i1 uses the spec's canonical {0, -1} values.
   *
   * Symbols lower to `<fun>__<sym>()` provider calls that the
   * embedding (e.g. the test driver) must define in the module
   * globals before invoking the entry function.
   */
  class PyBackend {
  public:
    explicit PyBackend(std::ostream &out) : out_(out) {}

    void emit(const Program &prog);

    void setNoRequire(bool val) { noRequire_ = val; }

    /// Omit the dynamic undefined-behavior guards. Sound only
    /// for known-UB-free programs: the guards never fire on such a
    /// program, so behavior is identical. Under this mode the arithmetic
    /// operators lower to inline Python expressions (`a + b`, truncating
    /// `//`, …) instead of the checked runtime helpers, and the helper
    /// preamble drops its `_trap` guards while keeping value semantics.
    void setNoUbGuards(bool val) { noUbGuards_ = val; }

    void setNoMainMangle(bool val) { noMainMangle_ = val; }

    /// Set the vector-lowering strategy (storage form of
    /// vector locals). Takes ownership. If never called, the backend
    /// defaults to "array" (plain lane lists) on first emit.
    void setVecLowering(std::unique_ptr<PyVecLowering> vl) { vecLowering_ = std::move(vl); }

  private:
    std::ostream &out_;
    bool noRequire_ = false;
    bool noUbGuards_ = false; // see setNoUbGuards
    bool noMainMangle_ = false;
    const Program *prog_ = nullptr;
    std::unique_ptr<PyVecLowering> vecLowering_; // see py_vec_lowering.hpp
    std::string curFuncName_;
    int indent_ = 0;
    int stmtCount_ = 0; // statements in the innermost open suite ("pass" insertion)
    // Float-literal evaluation context (see F32Guard in py_internal.hpp):
    // true (the SPEC default) makes FloatLit emit as _f32(lit).
    bool f32Ctx_ = true;
    std::unordered_map<std::string, TypePtr> varTypes_;
    std::unordered_map<std::string, std::string> pyNames_; // sigiled name -> python identifier
    std::unordered_set<std::string> takenNames_;
    // Locals lowered to flat leaf-slot lists: aggregate-typed lets and
    // params, plus scalars whose address is taken. Everything else
    // stays a plain python variable for readability.
    std::unordered_set<std::string> boxedRoots_;
    // Whether the function takes the address of one of its own locals,
    // and so can hand a pointer to storage that dies when it returns.
    // Set by collectBoxedRoots alongside the roots themselves.
    bool takesAddress_ = false;
    // Struct fields in declaration order, keyed by sigiled name (@S).
    std::unordered_map<std::string, std::vector<std::pair<std::string, TypePtr>>> structFields_;

    // --- Statement emission (src/backend/py_backend.cpp) ---
    void line(const std::string &s);
    void comment(const std::string &s);

    // Runs `body` one indent level deeper; emits `pass` if it produced
    // no statements.
    template<typename Fn>
    void suite(Fn &&body) {
      int saved = stmtCount_;
      stmtCount_ = 0;
      ++indent_;
      body();
      if (stmtCount_ == 0)
        line("pass");
      --indent_;
      stmtCount_ = saved + 1;
    }

    void emitFunction(const FunDecl &f);
    void emitLet(const LetDecl &l);
    void
    emitNode(const ControlTree &tree, const ControlTree::Node &n, const FunDecl &f, const CFG &cfg);
    void emitBlockInstrs(const Block &b);
    std::string condText(const FunDecl &f, std::size_t block, bool negate);

    // --- Expression emission (src/backend/py_expr.cpp) ---
    std::string exprStr(const Expr &expr);
    std::string atomStr(const Atom &atom);
    std::string opAtomStr(const OpAtom &arg);

    // Binary-operator lowering, shared by the scalar and
    // per-lane paths. Each returns the checked-helper call by default
    // and an inline Python expression under --no-ub-guards (`a`,`b` are
    // already-emitted operand strings, `n` the bit-width string). The
    // inline forms reproduce RefractIR's value semantics exactly
    // (truncating div/rem, logical-shift masking, per-op f32 rounding);
    // only the UB traps — dead on a UB-free program — are dropped.
    std::string binInt(
        const char *helper, char op, const std::string &a, const std::string &b,
        const std::string &n
    ) const;
    std::string binIDiv(const std::string &a, const std::string &b, const std::string &n) const;
    std::string binIRem(const std::string &a, const std::string &b, const std::string &n) const;
    std::string binShift(
        const char *helper, const char *op, const std::string &a, const std::string &b,
        const std::string &n
    ) const;
    std::string binLshr(const std::string &a, const std::string &b, const std::string &n) const;
    std::string binFloat(
        const char *helper, char op, const std::string &a, const std::string &b,
        const std::string &n
    ) const;
    std::string binFFmod(const std::string &a, const std::string &b, const std::string &n) const;
    std::string castF2I(const std::string &x, const std::string &n) const;
    std::string selectAtomStr(const SelectAtom &arg);
    std::string cmpAtomStr(const CmpAtom &arg);
    std::string castAtomStr(const CastAtom &arg);
    std::string callAtomStr(const CallAtom &arg);
    std::string condStr(const Cond &cond);
    std::string coefStr(const Coef &coef);
    std::string selectValStr(const SelectVal &sv);
    // Vector-context operands: a lane list, broadcasting scalars.
    std::string vecCoefStr(const Coef &coef, std::uint64_t n);
    std::string vecSelectValStr(const SelectVal &sv, std::uint64_t n);
    // Scalar conversion of the lane variable `src` (shared by scalar
    // and per-lane cast emission).
    std::string castValueStr(const std::string &src, const TypePtr &srcTy, const TypePtr &dstTy);

    // --- LValues and the memory model (src/backend/py_lvalue.cpp) ---
    // A resolved access path into a (possibly boxed) root: `buf` is
    // the python list expression, `off` the leaf-slot offset
    // expression, and [lo, hi) the extent of the innermost enclosing
    // object (used as _Ptr provenance bounds).
    struct PathInfo {
      std::string buf;
      std::string off;
      std::string lo, hi;
      TypePtr type;
      bool boxed = false;
    };

    PathInfo resolvePath(const LValue &lv);
    // Scalar (or whole-aggregate) read of an lvalue as a python value.
    std::string lvalueStr(const LValue &lv);
    std::string indexStr(const Index &idx);
    // Final lowered lane-index expression for a vector-local
    // subscript: literal indices are validated statically, dynamic
    // ones get the `_idx` bounds guard.
    std::string laneIdxExpr(const Index &idx, std::uint64_t n);
    std::string addrAtomStr(const AddrAtom &arg);
    std::string loadAtomStr(const LoadAtom &arg);
    std::string ptrIndexAtomStr(const PtrIndexAtom &arg);
    std::string ptrFieldAtomStr(const PtrFieldAtom &arg);
    void emitAssign(const AssignInstr &ins);
    void emitStore(const StoreInstr &ins);
    // Flat leaf-slot initializer list for an aggregate local.
    std::string flattenInit(const InitVal &iv, const TypePtr &type);
    // Scalar initializer element (with f32 rounding per target type).
    std::string scalarInit(const InitVal &iv, const TypePtr &elemType);
    void collectBoxedRoots(const FunDecl &f);

    // --- Packed leaf layout (src/backend/py_types.cpp) ---
    // Number of scalar/pointer leaf slots a type occupies.
    std::uint64_t byteSize(const TypePtr &t) const;
    // Leaf offset of a struct field; also yields its type.
    std::uint64_t fieldByteOffset(
        const std::string &structName, const std::string &field, TypePtr *fieldType
    ) const;

    // --- Type queries (src/backend/py_types.cpp) ---
    TypePtr getLValueType(const LValue &lv);
    TypePtr getAtomType(const Atom &atom);
    TypePtr getExprType(const Expr &expr);
    TypePtr getCoefType(const Coef &coef);
    TypePtr getSelectValType(const SelectVal &sv);
    // Bit width of an integer type (0 if not an integer type).
    static std::uint32_t intWidth(const TypePtr &t);
    // Bit width of a float type (0 if not a float type).
    static std::uint32_t floatWidth(const TypePtr &t);
    // True if the type is or contains f64 (drives the float-literal
    // context, mirroring the C backend's isOrContainsF64).
    static bool containsF64(const TypePtr &t);
    // Rejects types the python target cannot express yet.
    void requireSupportedType(const TypePtr &t, const char *what) const;

    // --- Naming ---
    std::string mangleFun(const std::string &name) const;
    std::string symCall(const std::string &symName) const;
    std::string pyLocal(const std::string &sigiled);
    void buildNameMap(const FunDecl &f, const ControlTree &tree);
    static std::string stripSigil(const std::string &name);

    // --- Intrinsics ---
    void emitIntrinsicHelpers(const Program &prog);
    // Emit the Python `def`s for a horizontal vector reduction
    // kind (generated dynamically so the body can vary with --no-ub-guards
    // and reuse the checked/inline arithmetic of the active preamble).
    void emitReductionHelperDefs(IntrinsicKind k);
    std::string callIntrinsic(const IntrinsicDecl &intr, const std::vector<std::string> &args);
  };

} // namespace refractir
