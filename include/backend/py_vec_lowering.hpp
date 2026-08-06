#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include "ast/ast.hpp"

namespace refractir {

  /**
   * PyVecLowering — abstract strategy that controls how the python
   * backend stores `<N> T` vector *locals and params*. Mirrors
   * CVecLowering (include/backend/c_vec_lowering.hpp) with one
   * architectural difference: the python backend computes vector
   * values as lane-list *expressions* (comprehensions over zip), so a
   * strategy governs only the storage form and the conversions at
   * read/write points — the lane list is the universal compute and
   * function-boundary representation. Vectors nested inside arrays /
   * structs stay flat leaf slots of the enclosing boxed root under
   * every strategy (they are the memory model, not locals).
   *
   * "vecext" has no python analogue (no native SIMD value type) and is
   * rejected by the driver.
   */
  class PyVecLowering {
  public:
    virtual ~PyVecLowering() = default;

    /// Strategy name; stamped into the module as a traceability
    /// comment (mirrors the C backend's `// vec-lowering:` line).
    virtual std::string name() const = 0;

    /// Emit per-shape helper classes after the module preamble.
    /// `usedShapes` is the deduplicated (N, T) shape set of the
    /// program (collectVecShapes).
    virtual void emitPreamble(std::ostream &out, const std::vector<VecType> &usedShapes) = 0;

    /// Statement: declare local `name` with every lane undef.
    virtual std::string declUndef(const std::string &name, const VecType &vt) = 0;

    /// Statement: declare/assign local `name` from a *fresh* lane-list
    /// expression (comprehension, flattened initializer, slice copy,
    /// or provider call result).
    virtual std::string
    assignFromList(const std::string &name, const VecType &vt, const std::string &listExpr) = 0;

    /// Expression: the storage as a lane list, possibly borrowed
    /// (aliasing); callers that store the result must copy. Used for
    /// read-only consumption (zip operands, slice sources).
    virtual std::string rawListExpr(const std::string &name, const VecType &vt) = 0;

    /// Expression: a fresh, undef-checked lane list — a whole-vector
    /// *read* per the spec (reading an undef lane is UB).
    virtual std::string readListExpr(const std::string &name, const VecType &vt) = 0;

    /// Expression: undef-checked single-lane read. `idxExpr` is the
    /// final lowered index (an `_idx(...)` guard or a validated
    /// literal). Strategies without dynamic lane storage (scalars /
    /// structscalars) throw on a non-literal index, mirroring the C
    /// strategies.
    virtual std::string
    laneRead(const std::string &name, const VecType &vt, const std::string &idxExpr) = 0;

    /// Statement: single-lane write (same idxExpr contract as
    /// laneRead).
    virtual std::string laneWrite(
        const std::string &name, const VecType &vt, const std::string &idxExpr,
        const std::string &valExpr
    ) = 0;

    /// Statement: whole-vector copy between two locals in storage
    /// form. Copies raw slots — undef lanes stay undef (copying is
    /// not a read).
    virtual std::string
    wholeCopy(const std::string &lhs, const std::string &rhs, const VecType &vt) = 0;

    /// Statement converting a vector param (which always arrives as a
    /// lane list — the function-boundary ABI) into storage form.
    /// Empty string when storage already is a list.
    virtual std::string unpackParam(const std::string &name, const VecType &vt) = 0;
  };

  /**
   * Factory by name. Returns nullptr for unknown names (including
   * "vecext"). The python backend's default is "array".
   *
   * Built-in names:
   *   "array"         — a plain lane list (the default; the historical representation)
   *   "scalars"       — N separate variables `v_0 .. v_{N-1}`
   *   "structarray"   — per-shape class holding a lane list
   *   "structscalars" — per-shape class with fields l0 .. l{N-1}
   */
  std::unique_ptr<PyVecLowering> makePyVecLowering(const std::string &name);

  /**
   * The names makePyVecLowering accepts, in a stable order.
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
  const std::vector<std::string> &pyVecLoweringNames();

} // namespace refractir
