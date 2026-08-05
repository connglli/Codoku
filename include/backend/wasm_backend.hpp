#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "analysis/structurizer.hpp"
#include "ast/ast.hpp"
#include "backend/wasm_vec_lowering.hpp"

namespace refractir {

  /**
   * Generates WebAssembly Text Format (.wat) from a RefractIR program.
   *
   * Note on Undefined Behavior: Unlike the C target (which leverages C's native
   * UB similarities and GCC sanitizers to catch and trap RefractIR UBs at runtime),
   * the WASM backend does not instrument generated WAT with check logic to detect
   * or trap undefined behavior. In compliance with the compiler's semantic refinement
   * model, behavior is only guaranteed for UB-free inputs.
   */
  class WasmBackend {
    friend struct WasmIntrinsicRegistry;
    friend class WasmVecLowering;

  public:
    explicit WasmBackend(std::ostream &out) : out_(out) {}

    /**
     * Translates the entire program to WAT and writes it to the output stream.
     */
    void emit(const Program &prog);

    void setNoModuleTags(bool val) { noModuleTags_ = val; }

    void setNoRequire(bool val) { noRequire_ = val; }

    /// Omit the dynamic undefined-behavior guards (null/OOB
    /// pointer traps, FP finiteness traps, intrinsic preconditions).
    /// Sound only for known-UB-free programs: the guards never fire on
    /// such a program, so behavior is identical. Value semantics are
    /// unaffected. Orthogonal to noRequire_. The `unreachable`
    /// *terminator* (an explicit UB marker) is kept — WASM has no
    /// no-op unreachable hint and the block still needs a terminator.
    void setNoUbGuards(bool val) { noUbGuards_ = val; }

    void setNoMainMangle(bool val) { noMainMangle_ = val; }

    /// Emit structured control flow (block/loop/if with named
    /// br labels) instead of the $__pc dispatch loop. The caller must
    /// have verified reducibility (ReducibilityCheck).
    void setStructuredLowering(bool val) { structuredLowering_ = val; }

    /// Select the vector storage strategy (default: "array").
    void setVecLowering(std::unique_ptr<WasmVecLowering> vl);

  private:
    std::ostream &out_;
    int indent_level_ = 0;
    std::string curFuncName_;
    std::unique_ptr<WasmVecLowering> vecLowering_; // set lazily to "array" in emit()
    bool noModuleTags_ = false;
    bool noRequire_ = false;
    bool noUbGuards_ = false; // see setNoUbGuards
    bool noMainMangle_ = false;
    bool structuredLowering_ = false; // see setStructuredLowering
    const Program *prog_ = nullptr;   // for callee lookup in emitAtom

    // Maps local/param names to their WASM local index or info
    struct LocalInfo {
      std::string wasmType;
      bool isParam = false;
      std::uint32_t bitwidth = 32;
      bool isAggregate = false;
      std::uint32_t offset = 0; // offset from stack pointer if aggregate
      TypePtr refractirType;
    };

    std::unordered_map<std::string, LocalInfo> locals_;
    std::unordered_map<std::string, TypePtr> syms_;
    std::uint32_t stackSize_ = 0;

    struct FieldInfo {
      std::uint32_t offset;
      std::uint32_t size;
      TypePtr type;
    };

    struct StructInfo {
      std::uint32_t totalSize;
      std::unordered_map<std::string, FieldInfo> fields;
      std::vector<std::string> fieldNames;
    };

    std::unordered_map<std::string, StructInfo> structLayouts_;

    // Vectors cross the call boundary through caller-owned frame
    // memory: every vector argument is spilled into a per-call-site
    // scratch slot and passed as an i32 address, and a vector return is
    // written by the callee through a hidden trailing `$__sret` address
    // parameter. The pre-scan in emit() sizes one scratch block per call
    // site (keyed by CallAtom node) into the frame; VecCallSlots describes
    // the block's internal layout as displacements from its base offset.
    struct VecCallSlots {
      std::vector<std::int64_t> argDisp; // per-arg displacement; -1 = non-vector arg
      std::int64_t retDisp = -1;         // sret-slot displacement; -1 = non-vector ret
      std::uint32_t totalBytes = 0;
    };

    std::unordered_map<const CallAtom *, std::uint32_t> callVecScratch_;
    const IntrinsicDecl *resolveIntrinsic(const CallAtom &arg);
    std::pair<std::vector<TypePtr>, TypePtr> calleeSignature(const CallAtom &arg);
    VecCallSlots vecCallSlots(const CallAtom &arg);
    // Emit (once) every vector-returning call nested in a vector
    // expression, filling its scratch slot, so per-lane emission can read
    // lanes without re-invoking the callee.
    void materializeVecCalls(const Expr &expr);

    // --- Emission helpers ---
    void indent();
    // Statement-level emission of one instruction. Shared by the
    // $__pc dispatch-loop body in emit() and the structured
    // control-tree emitter (src/backend/wasm_structured.cpp).
    void emitInstr(const Instr &ins);
    // `ret` terminator emission: a vector return through the hidden
    // `$__sret` slot or a scalar result, the shadow-stack teardown,
    // then `return`. Shared by both emitters likewise.
    void emitReturn(const RetTerm &ret, const FunDecl &f);

    // --- Structured emission (src/backend/wasm_structured.cpp) ---
    // Function-body emission for --structured-lowering:
    // structure the (reducible) CFG and emit genuine block/loop/if
    // WASM. WASM's multi-level `br N` consumes the *unlowered* control
    // tree directly — no $__pc dispatch loop and no guard flags — with
    // each loop and each pending join addressed by a named br label.
    void emitStructuredBody(const FunDecl &f);
    void emitCTreeNode(const ControlTree &tree, const ControlTree::Node &n, const FunDecl &f);
    void emitType(const TypePtr &type);
    std::string getWasmType(const TypePtr &type);
    std::uint32_t getTypeSize(const TypePtr &type);
    std::uint32_t getIntWidth(const TypePtr &type);
    void computeLayouts(const Program &prog);

    void emitExpr(const Expr &expr, std::uint32_t targetWidth, bool isFloat = false);
    // Emit a pointer-valued expression, scaling int offsets by pointee element size
    void emitPtrExpr(const Expr &expr, const TypePtr &ptrType);
    // Recognise `ptr - ptr` (i64 element distance per spec §6.8.6).
    bool isPtrDiff(const Expr &expr) const;
    void emitPtrDiff(const Expr &expr);
    void emitAtom(const Atom &atom, std::uint32_t targetWidth, bool isFloat = false);
    // emitAtom dispatches on the Atom variant; each alternative's emission
    // lives in a dedicated emitXxxAtom helper (src/backend/wasm_expr.cpp).
    // CoefAtom stays inline; signatures are tailored per branch so no
    // parameter is unused (wasmWidth is recomputed from targetWidth where
    // needed).
    void emitRValueAtom(const RValueAtom &arg, std::uint32_t targetWidth);
    void emitOpAtom(const OpAtom &arg, std::uint32_t targetWidth, bool isFloat);
    void emitSelectAtom(const SelectAtom &arg, std::uint32_t targetWidth, bool isFloat);
    void emitCmpAtom(const CmpAtom &arg, std::uint32_t targetWidth);
    void emitPtrIndexAtom(const PtrIndexAtom &arg);
    // sretOffset >= 0 overrides the sret target for a vector-returning
    // call: the callee writes straight into `$__old_sp - sretOffset`
    // (used when the call is the whole rhs of a vector assignment).
    void emitCallAtom(const CallAtom &arg, std::int64_t sretOffset = -1);
    void emitPtrFieldAtom(const PtrFieldAtom &arg);
    void emitUnaryAtom(const UnaryAtom &arg, std::uint32_t targetWidth, bool isFloat);
    void emitAddrAtom(const AddrAtom &arg);
    void emitLoadAtom(const LoadAtom &arg, std::uint32_t targetWidth, bool isFloat);
    void emitCastAtom(const CastAtom &arg, std::uint32_t targetWidth);
    void emitCond(const Cond &cond);
    void emitLValue(const LValue &lv, bool isStore);
    void emitCoef(const Coef &coef, std::uint32_t targetWidth, bool isFloat = false);
    void emitSelectVal(const SelectVal &sv, std::uint32_t targetWidth, bool isFloat = false);
    void emitIndex(const Index &idx);
    void emitInitVal(const InitVal &iv, const TypePtr &type, std::uint32_t offset);
    // Emit a WASM helper function for one intrinsic.
    void emitIntrinsicHelper(const IntrinsicDecl &intr);
    // Emit the helper for a horizontal vector reduction: the
    // vector arg arrives by address (the always-packed frame-memory spill
    // ABI), so lanes are read directly with `<elem>.load offset=k*size`,
    // strategy-independently, then folded to a scalar.
    void emitReductionHelper(const IntrinsicDecl &intr);
    std::string intrinsicHelperName(const IntrinsicDecl &intr) const;
    void emitCopy(
        const TypePtr &type, std::uint32_t dstOffset, const std::string &srcName,
        std::uint32_t srcOffset
    );
    void emitAddress(const LValue &lv);

    TypePtr getLValueType(const LValue &lv);
    TypePtr getSelectValType(const SelectVal &sv);
    TypePtr getAtomType(const Atom &atom);
    TypePtr getExprType(const Expr &expr);
    TypePtr getCoefType(const Coef &coef);
    // Vector compute unrolls operations lane-by-lane; storage is owned by
    // the WasmVecLowering strategy (default "vecext": native v128
    // registers per SPEC v0.2.1 §10.16, with wider shapes split across
    // registers).
    void emitVecExprLane(
        const Expr &expr, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
        bool isFloat
    );
    void emitVecAtomLane(
        const Atom &atom, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
        bool isFloat
    );
    // Per-Atom-kind lane emission for emitVecAtomLane (src/backend/wasm_vec.cpp);
    // CoefAtom / RValueAtom stay inline; signatures tailored per branch.
    void emitVecOpAtomLane(
        const OpAtom &arg, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
        bool isFloat
    );
    void emitVecSelectAtomLane(
        const SelectAtom &arg, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
        bool isFloat
    );
    void emitVecUnaryAtomLane(
        const UnaryAtom &arg, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
        bool isFloat
    );
    void emitVecCastAtomLane(
        const CastAtom &arg, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth
    );
    void emitVecCmpAtomLane(
        const CmpAtom &arg, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth
    );
    void emitVecCallLane(
        const CallAtom &arg, std::uint64_t lane, std::uint32_t targetWidth, bool isFloat
    );
    // Convert the elem-typed lane value on top of stack to targetWidth
    // (int extend/wrap, float promote/demote).
    void emitVecLaneConvert(const TypePtr &elemTy, std::uint32_t targetWidth);
    // Initialize a *register-strategy* vector local (frame-memory
    // vector locals go through emitInitVal like any aggregate).
    void emitVecLocalInit(const std::string &name, const VecType &vt, const InitVal &iv);
    void emitVecCoefLane(
        const Coef &coef, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
        bool isFloat
    );
    void emitVecLValueLane(
        const LValue &lv, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
        bool isFloat
    );
    void emitVecSelectValLane(
        const SelectVal &sv, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
        bool isFloat
    );

    // --- Naming and structure ---
    std::string mangleName(const std::string &name);
    std::string stripSigil(const std::string &name);
    std::string getMangledSymbolName(const std::string &funcName, const std::string &symName);

    void emitMask(std::uint32_t bitwidth, std::uint32_t wasmWidth);
    void emitSignExtend(std::uint32_t bitwidth, std::uint32_t wasmWidth);
  };

} // namespace refractir
