#pragma once

// Checksum rewriting and symiri-capture utilities.
//
// Declares the post-solve CRC32 rewrite and minimal-oracle builder
// plus the shared runSymiriCaptureResult helper that was duplicated across
// rysmith.cpp and rylink.cpp. Everything here is consumed by both tools; keeping
// it in a dedicated header makes the dependency clear and avoids
// pulling in the full func_gen machinery just for the runner.

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "ast/ast.hpp"
#include "solver/solver.hpp"

namespace refractir::reify {

  // ---------------------------------------------------------------------------
  // CRC32 exit-block rewriter
  // ---------------------------------------------------------------------------

  // Post-solve exit-block rewriter.
  //
  // Walks the entry function (`@<funcName>`)'s exit block and replaces the
  // sum-based `%_chk = %_chk + <atom>` accumulator chain (emitted by
  // buildSumChecksum) with a series of
  // `%_chk = call @crc32_update(%_chk, <atom>);` calls. The `%_chk = 0;`
  // init and the trailing `ret %_chk;` are left untouched. An
  // IntrinsicDecl for `@crc32_update(state: i32, val: i32) : i32` is
  // appended to `prog.intrinsics` on first call (idempotent).
  //
  // Returns the number of `%_chk = call @crc32_update(...)` instructions
  // emitted; 0 means the exit block did not contain a recognisable sum
  // chain and the program was left unchanged. Idempotent: a second call on
  // an already-rewritten program is a no-op and returns 0.
  //
  // The solver-seen SUM checksum deliberately omits pointer-pointee loads
  // (buildSumChecksum skips ptr leaves), so the solver never constrained
  // them. When re-adding `load %p` here we must therefore only load
  // pointers whose exit-time target the solver actually resolved to a live
  // object — `letExitValues[name]` is a `Ptr` with a non-empty
  // `targetLocal`. A pointer left at an undef / cross-object value (empty
  // targetLocal) is NOT loaded: dereferencing it is UB the solver could not
  // see, which the strict interpreter would (rightly) trap on.
  [[nodiscard]] size_t rewriteExitToCrc32Checksum(
      refractir::Program &prog, const std::string &funcName,
      const std::unordered_map<std::string, refractir::SymbolicExecutor::LetExitValue>
          &letExitValues
  );

  // ---------------------------------------------------------------------------
  // Minimal CRC32 oracle builder
  // ---------------------------------------------------------------------------

  // Build a minimal "oracle" Program that computes the same checksum as
  // `full`'s rewritten entry function, but skips the body CFG entirely.
  //
  // The minimal program:
  //   - Reuses `full`'s struct decls and its @crc32_update intrinsics.
  //   - Defines `@minimal_<funcName>` with the same parameter signature
  //     and the same lets as the entry function (including any
  //     `%_pld_*` scratch slots), but each scalar / aggregate let-init
  //     is replaced by the corresponding LetExitValue from
  //     `letExitValues`. Pointer lets retain their declared `undef`
  //     init; their exit-time target is replayed in the entry block.
  //   - Has just two basic blocks: an entry block that emits
  //     `%p = addr <targetLocal>;` for every pointer let using the
  //     EXIT-time target the solver recorded (so body-side pointer
  //     retargets are captured); then `goto ^exit`. The exit block is
  //     a verbatim clone of the entry's exit-block (the CRC32 chain
  //     plus any `%_pld_N = load %p;` preamble, then `ret %_chk;`).
  //
  // Because the body bbls are gone, the lets retain their (now
  // exit-time) init values for the whole execution. The checksum then
  // collapses to a pure function of the solver model — no
  // path-dependent side effects. Useful as an independent oracle for
  // cross-checking the full program: `--validate` runs symiri on the
  // full .sir and compares its Result line against this minimal
  // program's Result line. Agreement proves that the solver's let-exit
  // values + pointer targets correctly model the interpreter's
  // path-execution end state.
  //
  // Returns an empty `Program` (no funcs) when the entry function
  // cannot be located in `full`.
  [[nodiscard]] refractir::Program buildMiniCrc32Prog(
      const refractir::Program &full, const std::string &funcName,
      const std::unordered_map<std::string, refractir::SymbolicExecutor::LetExitValue>
          &letExitValues
  );

} // namespace refractir::reify
