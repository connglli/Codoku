#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include "analysis/type_utils.hpp"
#include "ast/ast.hpp"

namespace refractir {

  /**
   * Pure, frame-independent type layout for the interpreter.
   *
   * Owns the program's struct registry (name -> StructDecl*, built once from
   * the Program) and answers structural layout queries: byte size of a type,
   * the scalar/pointer cell type living at a byte offset, and a field's offset
   * within its struct. Depends on nothing but the struct table, so it is
   * immutable after construction and safely shared by the Interpreter and its
   * Memory collaborator.
   *
   * Static-type queries that depend on per-frame state (getLValueType etc.)
   * deliberately stay on the Interpreter — they read the mutable typeMap_.
   */
  class TypeLayout {
  public:
    explicit TypeLayout(const Program &prog);

    /// Packed byte size of `t` (no padding; sizeof(@S) = sum of field sizes).
    std::uint64_t sizeofType(const TypePtr &t) const;

    /// Type of the scalar/pointer cell at byte `offset` into `t` (recurses
    /// through arrays and structs); returns `t` when `t` is itself a leaf.
    TypePtr getCellTypeAtOffset(TypePtr t, std::uint64_t offset) const;

    /// Byte offset of `fieldName` within struct `s` (sequential, no padding).
    std::uint64_t fieldOffset(const StructDecl &s, const std::string &fieldName) const;

    /// Struct declaration for `name`, or nullptr if unknown.
    const StructDecl *lookupStruct(const std::string &name) const;

    /// The full struct registry (for the find/end idiom at call sites).
    const TypeUtils::StructTable &structs() const { return structs_; }

  private:
    TypeUtils::StructTable structs_;
  };

} // namespace refractir
