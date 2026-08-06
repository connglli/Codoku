#pragma once

// twin_mini — the shared scaffolding for the single-function "mini
// programs" rytwin builds out of a captured state.
//
// Any consumer that wants to *run* a region from an arbitrary state has to
// materialize that state as RefractIR first: a list of roots, each with its
// declared type and a concrete entry value, plus the conversions from a
// captured StateValue back into declarations and instructions.
//
// This header owns that description and those conversions. It is free of any
// solver dependency, so a consumer can build a mini program without linking
// the solver — which rytwin, its only user today, does not.

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "analysis/type_utils.hpp"
#include "ast/ast.hpp"
#include "reify/state_profile.hpp"

namespace refractir::reify {

  // A pointer leaf of a root, with the lvalues whose addresses reproduce
  // the captured pointer at region entry and exit (nullopt = null pointer).
  // The mini program declares the cell `null`, assigns `addr <initTarget>`
  // before the body, and reassigns `addr <finalTarget>` after it — sound
  // whatever the body did to the cell in between.
  struct MiniPtrFix {
    std::vector<Access> path; // leaf path within the root
    TypePtr type;             // static `ptr T` of the cell
    std::optional<LValue> initTarget;
    std::optional<LValue> finalTarget;
  };

  // One state root a mini program must model: its declaration shape in the
  // entry function plus its concrete entry / exit values.
  struct MiniRoot {
    std::string name;
    TypePtr type;
    bool isParam = false;             // immutable in the mini program (never written)
    StateValue init;                  // value at region entry (s)
    StateValue target;                // required value at region exit (s')
    std::vector<MiniPtrFix> ptrFixes; // every pointer leaf of the root
  };

  using StructMap = TypeUtils::StructTable;

  // --- captured state -> AST --------------------------------------------

  // StateValue -> InitVal of the matching static type. Struct fields are
  // reordered from the StateValue's name-sorted form into declaration
  // order via the StructDecl.
  InitVal stateToInit(const StateValue &v, const TypePtr &ty, const StructMap &structs);

  // The scalar type of a leaf, reconstructed from its captured value.
  // Scalar StateValues carry their exact bit-width, so this equals the
  // static type of the leaf cell.
  TypePtr leafType(const StateValue &v);

  // --- small AST builders -----------------------------------------------

  LValue leafLV(const std::string &root, const std::vector<Access> &path);

  // The canonical name of one scalar leaf: `%a`, `%a[1]`, `%s.f0`. This is the
  // key every consumer indexes leaves by, so a captured state leaf and the
  // lvalue an instruction writes agree by construction. The LValue overload
  // returns nullopt when an index is not a literal, since such a leaf has no
  // one name.
  std::string leafKey(const std::string &root, const std::vector<Access> &path);
  std::optional<std::string> leafKey(const LValue &lv);

  Expr rvalExpr(LValue lv);

  // `addr <target>` or `null` — the RHS that reproduces a pointer cell.
  Expr ptrFixExpr(const std::optional<LValue> &target);

  // --- root scaffolding -------------------------------------------------

  // Declare every root as a let initialized to its entry value, appending
  // to `lets`. Params of the source function become IMMUTABLE lets: the
  // solver treats entry-function params as free symbols, so a root must
  // never become a param of the mini program.
  void declareRoots(
      const std::vector<MiniRoot> &roots, const StructMap &structs, std::vector<LetDecl> &lets
  );

  // Assignments that set each pointer cell's entry provenance. The cells
  // are declared `null` (aggregate initializers admit `null` but not
  // `addr` atoms), so these must run before anything can read them.
  std::vector<Instr> ptrInitInstrs(const std::vector<MiniRoot> &roots);

  // Assignments that land each pointer cell on its captured exit
  // provenance, whatever the body did to it in between.
  std::vector<Instr> ptrFinalInstrs(const std::vector<MiniRoot> &roots);

} // namespace refractir::reify
