#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "ast/ast.hpp"

namespace refractir {

  class WasmBackend;

  /**
   * WasmVecLowering — strategy that controls how the WASM backend stores
   * `<N> T` vector *locals and params*. Mirrors CVecLowering
   * (include/backend/c_vec_lowering.hpp) and PyVecLowering
   * (include/backend/py_vec_lowering.hpp) with two WASM-specific
   * invariants that hold under every strategy:
   *
   *   - compute is always the per-lane unrolled stack code
   *     (emitVecExprLane); a strategy governs only where a plain vector
   *     local's lanes live between statements and how one lane is read
   *     or written there;
   *   - the call-boundary ABI is always caller-owned frame memory
   *     (spilled arguments passed by address + hidden sret), so vectors
   *     cross function boundaries under every strategy;
   *   - vectors nested inside arrays/structs stay flat leaf slots of the
   *     enclosing frame-memory aggregate under every strategy (they are
   *     the memory model, not locals).
   *
   * A frame-memory strategy ("array") answers usesFrameMemory() = true:
   * vector locals are classified as shadow-stack aggregates and every
   * access keeps the backend's address+load/store paths — the lane hooks
   * below are never invoked for it. A register strategy answers false,
   * owns the local storage it declared via declareLocals, and the
   * backend routes plain-vector-local lane traffic through the hooks.
   */
  class WasmVecLowering {
  public:
    virtual ~WasmVecLowering() = default;

    /// Strategy name; stamped into the module as a `;; vec-lowering:`
    /// traceability comment (mirrors the C backend's `// vec-lowering:`).
    virtual std::string name() const = 0;

    /// True iff plain vector locals live in shadow-stack frame memory.
    virtual bool usesFrameMemory() const = 0;

    /// Declare per-function scratch locals this strategy needs (called
    /// once per function, before the per-vector declarations). Default:
    /// none.
    virtual void declareFuncScratch(WasmBackend &) {}

    /// Declare the backing WASM locals for vector local/param `name`.
    virtual void declareLocals(WasmBackend &b, const std::string &name, const VecType &vt) = 0;

    /// Prologue: unpack an incoming pointer-ABI vector param into this
    /// strategy's storage (the param local itself holds the caller's
    /// spill address).
    virtual void unpackParam(WasmBackend &b, const std::string &name, const VecType &vt) = 0;

    /// Push lane `lane` of `name` (elem-typed, sign-extended into the
    /// element's WASM type).
    virtual void emitLaneRead(
        WasmBackend &b, const std::string &name, const VecType &vt, std::uint64_t lane
    ) = 0;

    /// Dynamic-index lane read; `emitIdx` pushes the i32 index.
    virtual void emitLaneReadDyn(
        WasmBackend &b, const std::string &name, const VecType &vt,
        const std::function<void()> &emitIdx
    ) = 0;

    /// Lane write; `emitValue` pushes the elem-typed value.
    virtual void emitLaneWrite(
        WasmBackend &b, const std::string &name, const VecType &vt, std::uint64_t lane,
        const std::function<void()> &emitValue
    ) = 0;

    /// Dynamic-index lane write (same callback contracts as above).
    virtual void emitLaneWriteDyn(
        WasmBackend &b, const std::string &name, const VecType &vt,
        const std::function<void()> &emitIdx, const std::function<void()> &emitValue
    ) = 0;

  protected:
    // Emission access for subclasses (friendship is not inherited, so the
    // befriended base class re-exports what strategies need).
    static std::ostream &out(WasmBackend &b);
    static void indent(WasmBackend &b);
    /// Backing WASM local for one lane: `$<name>__<lane>`.
    static std::string laneLocal(WasmBackend &b, const std::string &name, std::uint64_t lane);
    /// The mangled WASM local holding the pointer-ABI param address.
    static std::string paramLocal(WasmBackend &b, const std::string &name);
    static std::string wasmTypeOf(WasmBackend &b, const TypePtr &t);
    static std::uint32_t intWidthOf(WasmBackend &b, const TypePtr &t);
    static std::uint32_t typeSizeOf(WasmBackend &b, const TypePtr &t);
    static void signExtend(WasmBackend &b, std::uint32_t from, std::uint32_t to);
  };

  /**
   * Factory by name. Returns nullptr for unknown names. The WASM
   * backend's default is "vecext".
   *
   * Built-in names:
   *   "vecext"  — SIMD128 v128 locals (native WASM SIMD)
   *   "array"   — frame-memory shadow stack
   *   "scalars" — N separate scalar WASM locals
   */
  std::unique_ptr<WasmVecLowering> makeWasmVecLowering(const std::string &name);

  /**
   * The names makeWasmVecLowering accepts, in a stable order.
   *
   * A sweep over the strategies has to know what they are, and reading the
   * factory is not something a caller can do. Keep this beside the factory:
   * a name added to one and not the other is a strategy that ships and is
   * never exercised.
   *
   * The order is part of the contract. A caller that picks a strategy by
   * drawing an index reproduces its choice from a seed, so reordering the
   * existing entries changes what every past seed selects.
   */
  const std::vector<std::string> &wasmVecLoweringNames();

} // namespace refractir
