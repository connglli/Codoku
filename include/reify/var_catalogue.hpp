#pragma once

#include <optional>
#include <random>
#include <string>
#include <vector>
#include "ast/ast.hpp"
#include "reify/type_gen.hpp"

namespace refractir::reify {

  struct VarEntry {
    std::string name; // "%v0", "%a0", "%t0", "%p0", "%pa0" (param)
    TypePtr type;
    // For pointer vars: the name of the local this points to (for addr init)
    std::optional<std::string> ptrTarget;
    // Struct type name if this var has struct type (e.g. "@struct_a3f9c2_0")
    std::string structTypeName;
    // true if this is a function parameter (immutable; appears in
    // FunDecl.params, not FunDecl.lets; never a target of `addr` or LHS).
    bool isParam = false;
  };

  struct VarCatalogue {
    std::vector<VarEntry> vars;
    std::vector<StructDecl> structDecls; // struct type declarations

    // Access by type category
    // vars of exact int/fp type
    [[nodiscard]] std::vector<const VarEntry *> scalarsOf(const TypePtr &t) const;
    // all int or fp vars
    [[nodiscard]] std::vector<const VarEntry *> allScalars() const;
    // vec vars of exact type
    [[nodiscard]] std::vector<const VarEntry *> vecsOf(const TypePtr &t) const;
    // vec vars <N> T whose lane (element) type equals `elemT`, any lane
    // count N — the operand pool for horizontal-reduction intrinsics.
    [[nodiscard]] std::vector<const VarEntry *> vecsWithElem(const TypePtr &elemT) const;
    // ptr T vars
    [[nodiscard]] std::vector<const VarEntry *> ptrsOf(const TypePtr &pointeeT) const;
    // ptr ptr T vars (any T)
    [[nodiscard]] std::vector<const VarEntry *> ptrsToPtr() const;
    // non-ptr, non-agg vars (can be addr-of to produce ptr T)
    [[nodiscard]] std::vector<const VarEntry *> addressable() const;
    // all vars that can be addr-of (including ptr T vars for ptr ptr T)
    [[nodiscard]] std::vector<const VarEntry *> allAddressable() const;

    // Find any var of a given type (for LHS of assignment)
    [[nodiscard]] const VarEntry *findAny(const TypePtr &t, std::mt19937 &rng) const;

    // Find an addressable var whose type equals T (to create ptr T)
    [[nodiscard]] const VarEntry *findAddressableOfType(const TypePtr &t, std::mt19937 &rng) const;
  };

  struct VarGenConfig {
    int nVars = 10;
    // Number of scalar parameters the function takes. Each is
    // immutable and named %p0, %p1, …. They appear in FunDecl.params
    // (not .lets) and are usable as RValues throughout the body.
    int nParams = 0;
    // Unique 6-char hex generation ID, used to namespace
    // struct names as `@struct_<id>_<funcIdx>_<j>` so multiple rysmith
    // outputs can coexist without renaming. `funcIdx` (the rysmith
    // --n-funcs index of the current function) is included so two
    // functions from the same rysmith run can't collide on struct
    // names even though they share `genId`. Without it, rylink would
    // see two distinct structs claiming the same name when bundling
    // multiple sibling funcs into one program.
    std::string genId;
    int funcIdx = 0;
    TypeGenConfig typeConfig;
  };

  // Generate a VarCatalogue.
  // Generation order: non-pointer vars first, then ptr vars (pointing to earlier
  // vars), then ptr-ptr vars. This ensures valid ptr→target relationships.
  [[nodiscard]] VarCatalogue genVarCatalogue(std::mt19937 &rng, const VarGenConfig &cfg);

} // namespace refractir::reify
