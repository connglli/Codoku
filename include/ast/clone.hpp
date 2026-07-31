#pragma once

// Deep copy for expression and instruction ASTs.
//
// The AST is move-only: `SelectAtom` owns its condition and mask through
// `unique_ptr`, so `Expr`, `Atom`, `Cond` and `Instr` all lose their copy
// constructors, and any pass that wants a second, independently mutable copy
// of a subtree has to walk it. This header is that walk, written once.
//
// Deep means deep: `CallAtom` holds its arguments through `shared_ptr<Expr>`
// and is therefore copyable, but a shallow copy would leave the two clones
// editing one argument list. Cloning allocates fresh argument expressions so
// the result shares nothing mutable with its source.
//
// Spans are carried over unchanged. A clone denotes the same source text as
// its original, so diagnostics keep pointing at code the user wrote; a caller
// splicing a clone somewhere new is free to overwrite them.

#include <vector>

#include "ast/ast.hpp"

namespace refractir {

  Expr cloneExpr(const Expr &e);
  Cond cloneCond(const Cond &c);
  Atom cloneAtom(const Atom &a);
  Instr cloneInstr(const Instr &i);

  std::vector<Instr> cloneInstrs(const std::vector<Instr> &is);

} // namespace refractir
