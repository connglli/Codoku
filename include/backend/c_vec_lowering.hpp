#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include "ast/ast.hpp"

namespace refractir {

  /**
   * CVecLowering — abstract strategy that controls how the C backend lowers
   * `<N> T` vector locals and operations. Four built-in strategies are
   * provided via `makeCVecLowering`; external tools may subclass for custom
   * lowerings, and the interface is stable enough to be swapped per run.
   */
  class CVecLowering {
  public:
    virtual ~CVecLowering() = default;

    /// Human-readable name (`"vecext"`, `"struct"`, ...). Stamped into the
    /// emitted C file as a traceability comment.
    virtual std::string name() const = 0;

    /// Emit any typedefs / helpers the strategy needs at the top of the
    /// file. `usedShapes` is the deduplicated set of (N, T) shapes that
    /// appear in the program; some strategies emit one typedef per shape.
    virtual void emitPreamble(std::ostream &out, const std::vector<VecType> &usedShapes) = 0;

    /// C type-string for a vector type (e.g., for use in declarations and
    /// function signatures). Strategies that don't have a single C type
    /// (e.g., `scalars`) return an empty string; callers route those via
    /// `emitLocalDecl` instead.
    virtual std::string typeString(const VecType &vt) = 0;

    /// Emit a local variable declaration for a vector. Some strategies emit
    /// one line (vecext / struct / array); `scalars` emits N lines.
    virtual void emitLocalDecl(std::ostream &out, const std::string &name, const VecType &vt) = 0;

    /// Emit an initializer following the local declaration. iv may be
    /// Aggregate (brace), Scalar (broadcast), or Undef.
    virtual void
    emitInit(std::ostream &out, const std::string &name, const VecType &vt, const InitVal &iv) = 0;

    /// Lane read in expression position: returns a C expression string.
    /// `idxExpr` is already the C-level index expression (lexed/lowered by
    /// the caller).
    virtual std::string
    emitLaneRead(const std::string &name, const VecType &vt, const std::string &idxExpr) = 0;

    /// Lane write `name[idx] = val;` (or the strategy's equivalent).
    virtual void emitLaneWrite(
        std::ostream &out, const std::string &name, const VecType &vt, const std::string &idxExpr,
        const std::string &valExpr
    ) = 0;

    /// Whole-vector copy `lhs = rhs;`. For vecext/struct: one assignment.
    /// For scalars/array: per-lane copies. Emits statement(s) without a
    /// trailing semicolon.
    virtual void emitWholeCopy(
        std::ostream &out, const std::string &lhs, const std::string &rhs, const VecType &vt
    ) = 0;

    /// May a vector cross a C function boundary? `false` for `scalars`
    /// (multiple identifiers can't be one return value) and `array` (C
    /// decays arrays to pointers); the C backend refuses to emit a vector
    /// param or return if this is false.
    virtual bool canCrossFnBoundary() const = 0;

    /// True iff this strategy needs lane-by-lane statement emission for
    /// vector arithmetic (i.e., C arithmetic operators *don't* work
    /// natively on the lowered C type). `vecext` is the only "false";
    /// `scalars`, `array`, `structscalars`, `structarray` all return true
    /// so the C backend emits per-lane statements at AssignInstr.
    virtual bool needsLaneUnroll() const = 0;
  };

  /**
   * Factory by name. Returns nullptr for unknown names. The default in
   * `symirc` is "vecext".
   *
   * Built-in names:
   *   "vecext"        — GCC/Clang vector_size attribute (Phase 1)
   *   "array"         — T[N]                            (Phase 2)
   *   "scalars"       — N separate scalars              (Phase 2)
   *   "structarray"   — packed struct { T lanes[N]; }   (Phase 2)
   *   "structscalars" — packed struct { T l1, .., lN; } (Phase 2)
   */
  std::unique_ptr<CVecLowering> makeCVecLowering(const std::string &name);

  /**
   * The names makeCVecLowering accepts, in a stable order.
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
  const std::vector<std::string> &cVecLoweringNames();

} // namespace refractir
