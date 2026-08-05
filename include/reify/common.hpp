#pragma once

// Small shared helpers used by the reify tools. Each utility here is
// generator policy — choices the *tools* need to make when driving the
// deterministic backends. The backends themselves are deterministic;
// randomness lives on this side so a multi-program sweep can vary backend
// strategies independently.

#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>
#include "ast/ast.hpp"
#include "reify/state_profile.hpp"

namespace refractir::reify {

  /// Read an entire file into a string. Throws std::runtime_error if the
  /// file cannot be opened.
  [[nodiscard]] std::string readFile(const std::filesystem::path &p);

  /**
   * Resolve a `--vec-lowering` CLI choice into a concrete strategy name for
   * `target` (`c`, `wasm` or `python`). If `requested == "random"`, picks
   * uniformly via `rng` from the strategies that target implements;
   * otherwise returns `requested` verbatim so the caller can hand it
   * straight to the backend's lowering factory. Shared across the reify
   * tools so they all sweep the same strategy set with the same odds.
   */
  [[nodiscard]] std::string
  pickVecLowering(std::mt19937 &rng, const std::string &requested, const std::string &target = "c");

  /**
   * Resolve a --structured-lowering request (true|false|random)
   * to a per-program decision. Shared between rysmith and rylink so
   * both tools flip the same coin with the same odds. Only "random"
   * consumes RNG state, so runs without the flag keep their historical
   * seed streams.
   */
  [[nodiscard]] bool pickStructuredLowering(std::mt19937 &rng, const std::string &requested);

  /**
   * Make sure `prog.intrinsics` declares `@check_chksum(i32, i32) : i32`.
   * Idempotent — only appends a new IntrinsicDecl when no matching
   * signature is already present. Called by buildMainFunction when the
   * call carries a non-empty retValue, so the AST is self-consistent
   * before SIRPrinter runs.
   */
  void ensureCheckChksumDecl(Program &prog);

  /**
   * Build a `fun @main() : i32` that calls `entryFn` once on `paramValues`
   * — decimal-int / hex-float strings, one per parameter in declaration
   * order, parsed into IntLit / FloatLit atoms.
   *
   * `retValue` is the expected return value: when non-empty the wrapper
   * asserts the result via `@check_chksum(EXPECTED, …)`, and appends that
   * intrinsic's declaration to `prog.intrinsics`. An empty `retValue`
   * skips the check, which is what a descriptor with no oracle gets (say
   * symiri failed to produce one).
   *
   * Always returns 0 on the happy path; a mismatch aborts inside
   * @check_chksum's lowering (fprintf + abort in C, UB in symiri).
   */
  [[nodiscard]] FunDecl buildMainFunction(
      Program &prog, const FunDecl &entryFn, const std::vector<std::string> &paramValues,
      const std::string &retValue
  );

  // ---------------------------------------------------------------------------
  // Shared symiri runner
  // ---------------------------------------------------------------------------

  // Run frontend and analysis passes on prog. Returns true if well-formed.
  bool runAnalysisPasses(refractir::Program &prog, bool verbose);

  // Run symiri on `sirPath` with `--main <funcName>` and capture its
  // `Result: <value>` output line. Returns the trimmed value string on
  // success; std::nullopt when symiri fails to launch, exits non-zero,
  // or doesn't produce a parseable `Result:` line.
  //
  // `paramArgs` are forwarded as positional CLI args after `--` so
  // functions with parameters can be exercised deterministically.
  //
  // When `outProfile` is non-null it is filled with the per-program-point
  // StateProfile of this very run (granularity `gran`), so a caller that
  // needs both the `Result:` value and the state trace pays for a single
  // interpret instead of two — e.g. rysmith captures the rytwin profile
  // during the same run it uses to validate the emitted program.
  std::optional<std::string> runSymiriCaptureResult(
      const std::filesystem::path &sirPath, const std::string &funcName,
      const std::vector<std::string> &paramArgs, StateProfile *outProfile = nullptr,
      StateGranularity gran = StateGranularity::Pbb
  );

  // Bounded-replay divergence check for --require-nonterm. Replays
  // `funcName` on `paramArgs` under a block-step cap (`maxBlocks`) and returns
  // true iff the program neither returns nor traps within the cap AND two
  // arrivals at the lasso header `headerLabel` (^-prefixed) `period` laps
  // apart carry bit-identical state — i.e. the header-state fixed point the
  // solver proved actually holds at runtime, so the program diverges. Returns
  // false on termination, UB, a non-recurrent header state, too few arrivals
  // within the cap, or any parse/setup failure.
  bool validateNontermDiverges(
      const std::filesystem::path &sirPath, const std::string &funcName,
      const std::vector<std::string> &paramArgs, const std::string &headerLabel, int period = 1,
      std::uint64_t maxBlocks = 256
  );

  // Returns true iff running `funcName` on `paramArgs` triggers
  // undefined behavior (the intended outcome of a --require-ub program), and
  // false on a clean return, a require-failure, or any parse/setup error. Used
  // to validate a UB-triggering whole program (rylink over a trap pool).
  bool programTraps(
      const std::filesystem::path &sirPath, const std::string &funcName,
      const std::vector<std::string> &paramArgs
  );

  // ---------------------------------------------------------------------------
  // Shared backend compiler helpers
  // ---------------------------------------------------------------------------

  /**
   * How a Program is lowered, shared by every entry point below so the four
   * of them cannot drift apart. Each field means the same thing whichever
   * backend reads it; a backend that has no use for one ignores it.
   */
  struct EmitOptions {
    // Keep `require` statements in the emitted code rather than dropping them.
    bool keepRequire = false;
    // Drop the dynamic UB guards (see CBackend::setNoUbGuards). Sound only
    // when the program is known UB-free.
    bool noUbGuards = false;
    // Vector-lowering strategy name; empty selects the backend's default
    // ("vecext" for C and WASM, "array" for python). A name the target does
    // not implement is an error, not a fallback.
    std::string vecLowering;
    // Reconstruct while/if control flow instead of C's goto or WASM's `$__pc`
    // dispatch loop. Requires a reducible program. Python always structures
    // and ignores this.
    bool structuredLowering = false;
    // Emit the program's `main` unmangled, as a real entry point.
    bool emitMain = false;
    // C only: write one `<stem>.c` per source file instead of one unit.
    bool splitBySource = false;
    bool verbose = false;
  };

  // Compile a Program to C (in-process). Writes `<outDir>/<primaryStem>.c`,
  // or one file per source stem under `opts.splitBySource`.
  bool emitCInProcess(
      refractir::Program &prog, const std::filesystem::path &outDir, const std::string &primaryStem,
      const EmitOptions &opts
  );

  // Compile a Program to WASM (in-process).
  bool emitWasmInProcess(
      refractir::Program &prog, const std::filesystem::path &outFile, const EmitOptions &opts
  );

  // Compile a Program to Python (in-process). Requires a reducible program,
  // since the backend structurizes unconditionally.
  bool emitPyInProcess(
      refractir::Program &prog, const std::filesystem::path &outFile, const EmitOptions &opts
  );

  // Parse a .sir file and compile it with the C / WASM / Python backend.
  // Returns true on success.
  bool compileSirInProcess(
      const std::filesystem::path &sirPath, const std::string &target,
      const std::filesystem::path &outPath, const EmitOptions &opts
  );

} // namespace refractir::reify
