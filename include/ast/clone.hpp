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
//
// The move-only property propagates upward, so `Block`, `FunDecl` and
// `Program` cannot be copied either, and the whole ladder is here. Printing a
// program and parsing the text back is not a substitute: it costs a full
// frontend round trip, and it silently depends on the printer and the parser
// agreeing about everything the AST carries — including the fields no surface
// syntax spells, which a round trip drops.
//
// The two resolved-link fields are carried differently, because what they
// point at differs. `CallAtom::resolvedIntrinsic` names an `IntrinsicDecl`,
// which is a signature and nothing else, so the source program's copy and the
// clone's answer every question identically and the clone keeps the pointer.
// `ExtDecl::resolvedBody` names a `FunDecl` that the clone owns its own
// editable copy of, so keeping it would leave the clone's declaration
// pointing at the original's body; it is dropped, and the link resolver
// refills it.
//
// The pointer a clone does keep is valid only while the program it points
// into outlives the clone.

#include <vector>

#include "ast/ast.hpp"

namespace refractir {

  Expr cloneExpr(const Expr &e);
  Cond cloneCond(const Cond &c);
  Atom cloneAtom(const Atom &a);
  Instr cloneInstr(const Instr &i);

  std::vector<Instr> cloneInstrs(const std::vector<Instr> &is);

  // An initializer owns its children through `shared_ptr` — an aggregate's
  // elements, and the atom of an atom-form init — so copying one shares them.
  // This allocates fresh children instead.
  InitVal cloneInitVal(const InitVal &iv);

  LetDecl cloneLetDecl(const LetDecl &l);
  Terminator cloneTerminator(const Terminator &t);
  Block cloneBlock(const Block &b);
  FunDecl cloneFunDecl(const FunDecl &f);

  // A whole program, including its struct, external and intrinsic
  // declarations. The result shares nothing mutable with `p`, so the two can
  // be edited independently.
  Program cloneProgram(const Program &p);

} // namespace refractir
