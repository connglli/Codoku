#pragma once

// Solver-private helpers shared across the symbolic executor's translation
// units. Kept inline in this detail header (not a separate .cpp) so each
// solver TU that needs them can pull them in without an extra object file.
// Not part of the public solver interface — internal to src/solver.
//
// The pointer-tag / size helpers (tagOfLocal, sizeofTagUnits,
// fieldOffsetTagUnits, typeMatch) underpin the provenance encoding and are
// candidates to fold into the Provenance collaborator later.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "analysis/type_utils.hpp"
#include "ast/ast.hpp"
#include "solver/smt.hpp"

namespace refractir {

  // Pointers are encoded as 64-bit BV tags identifying the addressed local.
  // Tag 0 is reserved for null. Tags are derived deterministically from the
  // activation and the local name via FNV-1a so they remain stable across
  // solver invocations.
  //
  // The activation, not just the name, is what a tag identifies. Locals are
  // named per function and the same name recurs across a bundle, so a tag over
  // the name alone would let a pointer into one frame's `%v0` resolve against
  // another frame's — and, once a frame has returned, against a live cell that
  // merely inherited its name. Mixing the frame in keeps a returned
  // activation's tags matching nothing, which is what makes rule 27 fall out
  // of the load/store dispatch rather than needing a check of its own.
  inline constexpr uint32_t kPtrBits = 64;

  // The entry function's activation. Named because the model decoding runs
  // after the walk, when no frame is current, and every cell it reports back
  // belongs to the entry.
  inline constexpr uint64_t kEntryFrame = 0;

  inline uint64_t tagOfLocal(uint64_t frame, const std::string &name) {
    uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a 64-bit offset basis
    for (int shift = 0; shift < 64; shift += 8) {
      h ^= (frame >> shift) & 0xFF;
      h *= 0x100000001b3ULL; // FNV-1a 64-bit prime
    }
    for (unsigned char c: name) {
      h ^= c;
      h *= 0x100000001b3ULL;
    }
    if (h == 0)
      h = 1; // never collide with null
    return h;
  }

  // Map RelOp to the appropriate SMT comparison kind, choosing BV or FP
  // variants based on isFp.
  inline smt::Kind relOpToSmtKind(RelOp op, bool isFp) {
    switch (op) {
      case RelOp::EQ:
        return smt::Kind::EQUAL;
      case RelOp::NE:
        return smt::Kind::DISTINCT;
      case RelOp::LT:
        return isFp ? smt::Kind::FP_LT : smt::Kind::BV_SLT;
      case RelOp::LE:
        return isFp ? smt::Kind::FP_LEQ : smt::Kind::BV_SLE;
      case RelOp::GT:
        return isFp ? smt::Kind::FP_GT : smt::Kind::BV_SGT;
      case RelOp::GE:
        return isFp ? smt::Kind::FP_GEQ : smt::Kind::BV_SGE;
    }
    return smt::Kind::EQUAL;
  }

  // Size of a type in **packed bytes**, so that:
  //   * `addr %arr[k]` on `[N] T` advances by `k * sizeofTagUnits(T)`
  //   * pointer arithmetic on `ptr T` scales by `sizeofTagUnits(T)`
  //   * `ptrfield`/`ptrindex` adds the right offset for nested aggregates
  //
  // This must be the *same* scale the interpreter's memory model uses, which
  // is why both route through the one authority. An earlier version counted
  // one unit per scalar leaf instead; that agrees with the byte layout only
  // while every scalar in an object has the same width, and otherwise both
  // misses real out-of-bounds pointer arithmetic and rejects valid programs
  // (see TypeUtils::packedSizeof).
  inline std::uint64_t sizeofTagUnits(
      const TypePtr &t, const std::unordered_map<std::string, const StructDecl *> &structs
  ) {
    return TypeUtils::packedSizeof(t, structs);
  }

  // Byte offset of the named struct field, on the same scale.
  inline std::uint64_t fieldOffsetTagUnits(
      const StructDecl &s, const std::string &field,
      const std::unordered_map<std::string, const StructDecl *> &structs
  ) {
    return TypeUtils::packedFieldOffset(s, field, structs);
  }

  // Compare RefractIR types for structural equality at the level we care about
  // (matters when enumerating candidate ptr targets in load/store dispatch).
  inline bool typeMatch(const TypePtr &a, const TypePtr &b) {
    if (!a || !b)
      return a == b;
    if (a->v.index() != b->v.index())
      return false;
    if (auto pa = std::get_if<IntType>(&a->v)) {
      auto pb = std::get_if<IntType>(&b->v);
      auto width = [](const IntType &t) -> uint32_t {
        switch (t.kind) {
          case IntType::Kind::I32:
            return 32;
          case IntType::Kind::I64:
            return 64;
          case IntType::Kind::ICustom:
            return t.bits.value_or(32);
        }
        return 32;
      };
      return width(*pa) == width(*pb);
    }
    if (auto pa = std::get_if<FloatType>(&a->v)) {
      return pa->kind == std::get<FloatType>(b->v).kind;
    }
    if (auto pa = std::get_if<PtrType>(&a->v)) {
      return typeMatch(pa->pointee, std::get<PtrType>(b->v).pointee);
    }
    if (auto pa = std::get_if<VecType>(&a->v)) {
      auto pb = std::get_if<VecType>(&b->v);
      return pa->size == pb->size && typeMatch(pa->elem, pb->elem);
    }
    // Aggregate types as pointees are not supported in current solver dispatch.
    return false;
  }

  // Assert that a floating-point term is finite (not +/-inf, not NaN) by
  // pushing the guards onto the path condition.
  inline void assertFPFinite(smt::Term t, smt::ISolver &solver, std::vector<smt::Term> &ub) {
    auto notInf = solver.make_term(smt::Kind::NOT, {solver.make_term(smt::Kind::FP_IS_INF, {t})});
    auto notNaN = solver.make_term(smt::Kind::NOT, {solver.make_term(smt::Kind::FP_IS_NAN, {t})});
    ub.push_back(notInf);
    ub.push_back(notNaN);
  }

} // namespace refractir
