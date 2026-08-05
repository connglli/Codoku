#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include "analysis/structurizer.hpp"
#include "ast/ast.hpp"
#include "backend/c_vec_lowering.hpp"

namespace refractir {

  /**
   * Generates C code from a RefractIR program.
   * Maps RefractIR constructs to their C equivalents, handling bitwidths
   * and symbolic variables (as externs).
   *
   * Note on Undefined Behavior: For the C target, we attempt to preserve
   * RefractIR's undefined behavior semantics as closely as possible, leveraging
   * the fact that many RefractIR UBs match native C UBs, which can be trapped
   * at runtime using sanitizers (ASan, UBSan, pointer-compare, etc.).
   */
  class CBackend {
    friend struct CIntrinsicRegistry;

  public:
    explicit CBackend(std::ostream &out) : out_(out) {}

    /**
     * Translates the entire program to C and writes it to the output stream.
     */
    void emit(const Program &prog);

    // Split emission: write `<outDir>/<stem>.c` for the
    // primary translation unit (stem = primaryStem) and one
    // `<outDir>/<libStem>.c` per distinct `FunDecl::sourceStem` that
    // arrived from an -I lib, plus `<outDir>/common.h` carrying
    // shared struct typedefs, intrinsic helpers, and extern
    // prototypes. Each per-source `.c` does `#include "common.h"`.
    // Returns the list of files written.
    std::vector<std::string>
    emitSplit(const Program &prog, const std::string &outDir, const std::string &primaryStem);

    void setNoRequire(bool val) { noRequire_ = val; }

    /// Suppress the dynamic undefined-behavior guards
    /// (null/OOB pointer traps, FP finiteness traps, integer div/rem
    /// zero traps, intrinsic-precondition traps, `unreachable`). Only
    /// sound when the program is known UB-free: on such a program the
    /// guards never fire, so the emitted code is behaviorally identical
    /// with or without them. Value semantics (truncating div, masking,
    /// etc.) are unaffected. Orthogonal to `noRequire_`.
    void setNoUbGuards(bool val) { noUbGuards_ = val; }

    void setNoMainMangle(bool val) { noMainMangle_ = val; }

    /// Set the vector-lowering strategy. Takes ownership. If
    /// never called, the backend defaults to "vecext" on first emit.
    void setVecLowering(std::unique_ptr<CVecLowering> vl) { vecLowering_ = std::move(vl); }

    /// Emit structured control flow (while/if reconstructed
    /// from the lowered control tree) instead of labels+goto. The
    /// caller must have verified reducibility (ReducibilityCheck).
    void setStructuredLowering(bool val) { structuredLowering_ = val; }

  private:
    std::ostream &out_;
    // When non-empty, restrict per-fun body emission to funs
    // whose sourceStem matches; used by emitSplit() to write one .c
    // per source-file stem.
    std::string emitOnlySourceStem_;
    // When true, suppress the global preamble (#include,
    // struct typedefs, intrinsic helpers, extern prototypes). Used
    // when emitting a non-primary `.c` whose preamble lives in
    // `common.h`.
    bool suppressPreamble_ = false;
    int indent_level_ = 0;
    bool noRequire_ = false;
    bool noUbGuards_ = false; // see setNoUbGuards
    bool noMainMangle_ = false;
    bool structuredLowering_ = false; // see setStructuredLowering
    const Program *prog_ = nullptr;   // for callee lookup in emitAtom
    std::string curFuncName_;
    std::unique_ptr<CVecLowering> vecLowering_; // strategy, see c_vec_lowering.hpp
    std::unordered_map<std::string, std::uint32_t> varWidths_;
    TypePtr curFuncRetType_;
    // ``isDoubleCtx_`` is the lowering-time evaluation context for float
    // literals: when false, literals emit with the ``f`` suffix so an
    // expression like ``-3.0 * %f32_var`` stays in single precision under
    // FLT_EVAL_METHOD == 0. Mutated transiently around assign/store/ret/
    // cond/init/cast emission via ``CtxGuard``.
    bool isDoubleCtx_ = false;
    std::unordered_map<std::string, TypePtr> varTypes_;
    // Struct field name+type in declaration order. Linear lookup by name
    // for AccessField, indexed lookup for InitVal aggregates. Structs are
    // tiny (handful of fields), so linear scan is fine.
    std::unordered_map<std::string, std::vector<std::pair<std::string, TypePtr>>> structFields_;

    // --- Emission helpers ---
    void indent();
    // Statement-level emission of one instruction, indentation
    // included. Shared by the goto block loop and the structured
    // control-tree emitter (src/backend/c_structured.cpp).
    void emitInstr(const Instr &ins);
    // `ret` terminator emission (the caller indents).
    void emitRetTerm(const RetTerm &ret);

    // --- Structured emission (src/backend/c_structured.cpp) ---
    // Function-body emission for --structured-lowering:
    // structure the (reducible) CFG and print the lowered control
    // tree as genuine while/if statements — no labels, no goto.
    void emitStructuredBody(const FunDecl &f);
    void emitCTreeNode(const ControlTree &tree, const ControlTree::Node &n, const FunDecl &f);
    // The `if (`/`while (` condition of `block`'s conditional BrTerm,
    // wrapped in `!(...)` when negate is set.
    void emitBranchCond(const FunDecl &f, std::size_t block, bool negate);
    void emitType(const TypePtr &type);
    void emitExpr(const Expr &expr);
    void emitAtom(const Atom &atom);
    // emitAtom dispatches on the Atom variant; each alternative's emission
    // lives in a dedicated emitXxxAtom helper (src/backend/c_expr.cpp). The
    // trivial CoefAtom / RValueAtom alternatives stay inline in the dispatcher.
    void emitOpAtom(const OpAtom &arg);
    void emitSelectAtom(const SelectAtom &arg);
    void emitCmpAtom(const CmpAtom &arg);
    void emitUnaryAtom(const UnaryAtom &arg);
    void emitAddrAtom(const AddrAtom &arg);
    void emitLoadAtom(const LoadAtom &arg);
    void emitPtrIndexAtom(const PtrIndexAtom &arg);
    void emitPtrFieldAtom(const PtrFieldAtom &arg);
    void emitCallAtom(const CallAtom &arg);
    void emitCastAtom(const CastAtom &arg);
    void emitCond(const Cond &cond);
    void emitLValue(const LValue &lv);
    void emitCoef(const Coef &coef);
    void emitSelectVal(const SelectVal &sv);
    void emitIndex(const Index &idx);
    void emitInitVal(const InitVal &iv, TypePtr expectedType = nullptr);

    // Helpers for new top-level decls.
    void emitIntrinsicHelper(const IntrinsicDecl &intr);
    // Emit the helper for a horizontal vector reduction
    // (@reduce_*): a `static inline` function taking the vector by the
    // vec-lowering strategy's cross-boundary representation and folding its
    // lanes to a scalar. Requires a boundary-crossing strategy.
    void emitReductionHelper(const IntrinsicDecl &intr);
    void emitExtDecl(const ExtDecl &d);
    std::string intrinsicHelperName(const std::string &intrName, std::uint32_t bits) const;
    std::string intrinsicHelperName(const IntrinsicDecl &intr) const;

    // --- Type query helpers ---
    // Resolve the static type of an lvalue / atom / expression. Used to
    // decide whether float-literal emission needs the ``f`` suffix. For
    // expressions, only the first atom is examined: RefractIR requires every
    // operand in an Expr to share a single type, so this is sound.
    //
    // NOTE on impurity: ``getCoefType`` returns a type for a bare float
    // literal that depends on the *current* ``isDoubleCtx_``. Callers
    // therefore see the literal as inheriting the outer context — which
    // is what every existing call site (emitCond / emitStore) wants. If
    // you add a new caller, be aware the answer for ``FloatLit`` reflects
    // the surrounding context, not the literal in isolation.
    TypePtr getLValueType(const LValue &lv);
    TypePtr getAtomType(const Atom &atom);
    TypePtr getExprType(const Expr &expr);
    TypePtr getCoefType(const Coef &coef);
    TypePtr getInitValType(const InitVal &iv);

    // Vector-specific statement-level emission. These handle the
    // RHS atoms that need lane-wise C emission (loop or unroll) and so
    // can't be inlined as a C expression.
    void emitVecCmpAssign(const LValue &lhs, const CmpAtom &c, const VecType &vt);
    void emitVecMaskSelectAssign(const LValue &lhs, const SelectAtom &s, const VecType &vt);
    /// Per-lane C expression for a SelectVal at lane `kExpr`.
    std::string sirSelectValLane(const SelectVal &sv, const VecType &vt, const std::string &kExpr);

    /// Lane-unroll: compute the C expression for lane k of the
    /// value produced by an Atom / Expr that yields a vector. Used by
    /// the AssignInstr handler when the active strategy needs lane-wise
    /// statement emission (scalars / array / struct*).
    std::string emitVecAtomLane(const Atom &a, const VecType &vt, std::uint64_t k);
    // Per-Atom-kind lane emission for emitVecAtomLane (src/backend/c_vec.cpp);
    // the trivial RValueAtom / UnaryAtom alternatives stay inline.
    std::string emitVecOpAtomLane(const OpAtom &arg, const VecType &vt, std::uint64_t k);
    std::string emitVecCoefAtomLane(const CoefAtom &arg, const VecType &vt, std::uint64_t k);
    std::string emitVecCastAtomLane(const CastAtom &arg, const VecType &vt, std::uint64_t k);
    std::string emitVecExprLane(const Expr &e, const VecType &vt, std::uint64_t k);
    /// Emit `lhs = rhs;` for a vector LHS, using lane-unroll if the
    /// strategy requires it; falls back to the inline (vecext) path
    /// otherwise.
    void emitVecAssign(const LValue &lhs, const Expr &rhs, const VecType &vt);

    // Look up a struct field by name; returns nullptr if either the
    // struct or the field is unknown.
    TypePtr findStructFieldType(const std::string &structName, const std::string &fieldName) const;
    // Look up a struct field by declaration-order index; returns nullptr
    // if the struct is unknown or the index is out of range.
    TypePtr getStructFieldTypeAt(const std::string &structName, size_t idx) const;

    // --- Mangling and naming helpers ---
    std::string getMangledSymbolName(const std::string &funcName, const std::string &symName);
    std::string mangleName(const std::string &name);
    std::string stripSigil(const std::string &name);
  };

} // namespace refractir
