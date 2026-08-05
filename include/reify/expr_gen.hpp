#pragma once

#include <compare>
#include <cstdint>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "analysis/intrinsics.hpp"
#include "ast/ast.hpp"
#include "reify/type_gen.hpp"
#include "reify/var_catalogue.hpp"

namespace refractir::reify {

  // Key for tracking which intrinsic instantiations have been used, so
  // genFunction can emit one IntrinsicDecl per distinct signature. Scalar
  // intrinsics vary only by element width; the horizontal-reduction family
  // (§12.4) is additionally parameterised by lane count and element domain,
  // because `@reduce_add(<4> i32) : i32` and `@reduce_add(<8> f64) : f64`
  // are distinct overloads that each need their own declaration.
  struct IntrinsicUseKey {
    IntrinsicKind kind;
    std::uint32_t elemBits; // scalar element width (32 for i32/f32, 64 for i64/f64)
    bool elemIsFloat;       // element domain: false = iN, true = fN
    std::uint32_t lanes;    // 0 = scalar intrinsic; N = reduce over <N> T

    auto operator<=>(const IntrinsicUseKey &) const = default;
  };

  // ---------------------------------------------------------------------------
  // SymCounter — tracks generated symbols, produces declarations
  // ---------------------------------------------------------------------------

  struct SymEntry {
    std::string name; // "%?sN"
    SymKind kind;
    TypePtr type;        // actual type of this sym (not always i32)
    std::int64_t lo, hi; // domain bounds
  };

  struct SymCounter {
    std::vector<SymEntry> entries;
    int n = 0;

    // Domains
    std::int64_t coefLo = -8, coefHi = 8;
    std::int64_t valueLo = -128, valueHi = 127;
    std::int64_t indexLo = 1, indexHi = 30;

    // Generate next sym of given kind and type
    std::string next(SymKind kind, const TypePtr &type);

    // Convenience: next coef sym of given type (most common)
    std::string nextCoef(const TypePtr &type);

    // Next value sym (i32)
    std::string nextValue();

    // Next index sym (i32)
    std::string nextIndex();

    [[nodiscard]] int countOfKind(SymKind kind) const;
    [[nodiscard]] std::vector<std::string> namesOfKindSince(SymKind kind, int since) const;
    [[nodiscard]] std::vector<SymDecl> makeDecls() const;
  };

  // ---------------------------------------------------------------------------
  // ExprGenConfig
  // ---------------------------------------------------------------------------

  struct ExprGenConfig {
    bool enableAllOps = true; // bitwise, shifts
    bool enableDiv = true;    // div/mod with concrete denominators
    bool enableSelect = true; // select ternary
    bool enableFp = true;
    bool enableIntrinsics = true;
    // Whether a `ptr T_scalar` reassign may emit an in-bounds pointer-
    // arithmetic RHS (`ptrindex %ap, b ± d`, `ptrfield %sp, f ± d`, or a
    // direct `%p = %q ± d`). Off via `--no-ptrarith`: these constraints add
    // solver work, so callers that don't want the cost can suppress them.
    bool enablePtrArith = true;
    int minAtoms = 1;
    int maxAtoms = 3;
    // Set by genCond for its arm expressions: branch/require conditions
    // are the solver's handles for driving the path, so their trivial-shape
    // replacements keep the sym-minting reroll instead of the cheap linear
    // chain. Body assignments leave this false and get sym-free, solver-
    // linear replacements.
    bool condContext = false;
    // Mutable set populated during expression generation. genFunction
    // reads it afterward to emit IntrinsicDecl entries. May be nullptr
    // (e.g. in tests) — intrinsic generation is silently skipped.
    std::set<IntrinsicUseKey> *usedIntrinsics = nullptr;
  };

  // ---------------------------------------------------------------------------
  // Core expression builder
  // ---------------------------------------------------------------------------

  // Generate an Expr of type targetType.
  // onPath=true: use symbols (sym counter). onPath=false: use concrete literals.
  // sym may be nullptr for off-path generation.
  // excludeName, when set, is the LHS local name that must NOT appear as an
  // RValue on the RHS — this defeats self-assigns (%x = %x) and reductive
  // patterns like %x = %x + 1 that fold flat under SCCP. Plumb through
  // every variable-pool pick.
  [[nodiscard]] Expr genExpr(
      std::mt19937 &rng,
      SymCounter *sym, // nullptr-safe: if nullptr, always concrete
      const VarCatalogue &vars, const TypePtr &targetType, bool onPath, const ExprGenConfig &cfg,
      const std::optional<std::string> &excludeName = std::nullopt
  );

  // Generate a branch Cond using random scalar var from available vars.
  [[nodiscard]] Cond genCond(
      std::mt19937 &rng, SymCounter *sym, const VarCatalogue &vars, bool onPath,
      const ExprGenConfig &cfg
  );

  // Generate interest requires for new coef syms since coefCountBefore.
  // Tiered: for each new coef, with probability `pLargeCoef` emit one
  // `|c| > largeCoefThreshold` require; otherwise emit nothing. The
  // threshold is clamped per-coef into the intersection of its declared
  // domain and its type's representable range, so the require is always
  // both type-valid (literal fits the coef's width) and satisfiable
  // (cannot exceed a narrow `--coef-domain`). The unconditional
  // `c != 0,1,-1` triple from earlier versions clustered the solver at
  // ±2 and starved the literal pool of magnitude diversity.
  [[nodiscard]] std::vector<Instr> interestCoefRequires(
      std::mt19937 &rng, const SymCounter &sym, int coefCountBefore, double pLargeCoef,
      std::int64_t largeCoefThreshold
  );

  // Generate N statements (assign + store mix) for a block.
  // onPath=true: uses sym and emits UB-safety requires. onPath=false:
  // concrete only, no requires (off-path blocks are never executed at the
  // solved inputs, so their UB is unreachable).
  [[nodiscard]] std::vector<Instr> genBlockStmts(
      std::mt19937 &rng,
      SymCounter *sym, // null for off-path
      const VarCatalogue &vars, int nStmts, bool onPath, const ExprGenConfig &cfg
  );

} // namespace refractir::reify
