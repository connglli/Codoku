#include "reify/common.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "analysis/dominators.hpp"
#include "analysis/reducibility.hpp"
#include "ast/build.hpp"
#include "ast/sir_printer.hpp"
#include "backend/c_backend.hpp"
#include "backend/c_vec_lowering.hpp"
#include "backend/py_backend.hpp"
#include "backend/wasm_backend.hpp"
#include "error.hpp"
#include "frontend/diagnostics.hpp"
#include "frontend/pipeline.hpp"
#include "interp/interpreter.hpp"
#include "reify/intrinsic_whitelist.hpp"
#include "reify/state_profile.hpp"

namespace fs = std::filesystem;
using namespace refractir;

namespace refractir::reify {

  std::string readFile(const std::filesystem::path &p) {
    std::ifstream ifs(p);
    if (!ifs)
      throw std::runtime_error("failed to open file: " + p.string());
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
  }

  std::string
  pickVecLowering(std::mt19937 &rng, const std::string &requested, const std::string &target) {
    if (requested != "random")
      return requested;

    // Each backend sweeps only the strategies it implements: python has no
    // native SIMD value type and so no vecext, and wasm has no struct
    // lowering. The draw stays a uniform_int_distribution<int> over the
    // chosen table so a given seed keeps picking the same strategy.
    static const char *const kC[] = {"vecext", "scalars", "array", "structscalars", "structarray"};
    static const char *const kPython[] = {"array", "scalars", "structscalars", "structarray"};
    static const char *const kWasm[] = {"vecext", "array", "scalars"};

    const char *const *pool = kC;
    int count = static_cast<int>(std::size(kC));
    if (target == "python") {
      pool = kPython;
      count = static_cast<int>(std::size(kPython));
    } else if (target == "wasm") {
      pool = kWasm;
      count = static_cast<int>(std::size(kWasm));
    }
    std::uniform_int_distribution<int> d(0, count - 1);
    return pool[d(rng)];
  }

  bool pickStructuredLowering(std::mt19937 &rng, const std::string &requested) {
    if (requested == "true")
      return true;
    if (requested != "random")
      return false;
    std::uniform_int_distribution<int> d(0, 1);
    return d(rng) == 1;
  }

  void ensureCheckChksumDecl(Program &prog) {
    ensureIntrinsicDecl(
        prog, "@check_chksum", buildI32(), {{"%expected", buildI32()}, {"%actual", buildI32()}}
    );
  }

  FunDecl buildMainFunction(
      Program &prog, const FunDecl &entryFn, const std::vector<std::string> &paramValues,
      const std::string &retValue
  ) {
    FunDecl mainFn;
    mainFn.name = GlobalId{"@main", {}};
    mainFn.retType = buildI32();

    // `%r` holds the entry-function return value. Its declared type matches
    // entryFn so a float-returning entry doesn't trip the typechecker; the
    // value is consumed immediately by @check_chksum (when present) or
    // dropped.
    LetDecl letR;
    letR.isMutable = true;
    letR.name = LocalId{"%r", {}};
    letR.type = entryFn.retType;
    letR.init = InitVal{InitVal::Kind::Undef, LocalId{}, {}};
    mainFn.lets.push_back(std::move(letR));

    Block b;
    b.label = BlockLabel{"^entry", {}};

    // %r = call @entry(arg0, arg1, ...);
    CallAtom ca;
    ca.callee = entryFn.name;
    for (std::size_t i = 0; i < entryFn.params.size() && i < paramValues.size(); ++i) {
      const auto &p = entryFn.params[i];
      const std::string &valStr = paramValues[i];
      Atom arg = p.type && std::holds_alternative<FloatType>(p.type->v)
                     ? buildCoefAtom(Coef{FloatLit{parseFloatLiteral(valStr), {}}})
                     : buildCoefAtom(Coef{IntLit{parseIntegerLiteral(valStr), {}}});
      ca.args.push_back(std::make_shared<Expr>(buildExpr(std::move(arg))));
    }
    AssignInstr callAssign;
    callAssign.lhs = buildLValue("%r");
    callAssign.rhs = buildExpr(Atom{std::move(ca), {}});
    b.instrs.push_back(std::move(callAssign));

    // %r = call @check_chksum(EXPECTED, %r);  (skipped when retValue is
    // empty — happens for descriptors that the symiri-capture step couldn't
    // fill in).
    //
    // The check is gated on an integer-returning entry: @check_chksum is
    // i32-typed and RefractIR has no implicit FP↔int cast at call
    // boundaries. Float-returning entries skip the check; reify's float
    // oracles already go through the sum/CRC32 path on the RefractIR-side
    // checksum machinery.
    if (!retValue.empty() && entryFn.retType &&
        std::holds_alternative<IntType>(entryFn.retType->v)) {
      CallAtom check;
      check.callee = GlobalId{"@check_chksum", {}};
      check.args.push_back(
          std::make_shared<Expr>(
              buildExpr(buildCoefAtom(Coef{IntLit{parseIntegerLiteral(retValue), {}}}))
          )
      );
      check.args.push_back(
          std::make_shared<Expr>(buildExpr(buildRValAtom(RValue{LocalId{"%r", {}}, {}, {}})))
      );
      AssignInstr checkAssign;
      checkAssign.lhs = buildLValue("%r");
      checkAssign.rhs = buildExpr(Atom{std::move(check), {}});
      b.instrs.push_back(std::move(checkAssign));
      ensureCheckChksumDecl(prog);
    }

    // Always exit with 0 on the happy path. Any mismatch above unwinds
    // through @check_chksum's abort() before this terminator is reached.
    RetTerm ret;
    ret.value = buildExpr(buildCoefAtom(Coef{IntLit{0, {}}}));
    b.term = std::move(ret);

    b.span = {};
    mainFn.blocks.push_back(std::move(b));

    return mainFn;
  }

  bool runAnalysisPasses(Program &prog, bool verbose) {
    DiagBag diags;
    if (checkProgram(prog, diags))
      return true;
    if (verbose) {
      std::cerr << "reify: analysis passes failed:\n";
      for (const auto &d: diags.diags)
        if (d.level == DiagLevel::Error)
          std::cerr << "  error: " << d.message << "\n";
    }
    return false;
  }

  std::optional<std::string> runSymiriCaptureResult(
      const fs::path &sirPath, const std::string &funcName,
      const std::vector<std::string> &paramArgs, StateProfile *outProfile, StateGranularity gran
  ) {
    try {
      Program prog = parseSource(readFile(sirPath));

      // Run semantics/type check passes first to ensure it's valid
      if (!runAnalysisPasses(prog, /*verbose=*/false))
        return std::nullopt;

      std::string canonical = funcName.empty() || funcName[0] == '@' ? funcName : "@" + funcName;

      // The interpreter's `Result:` line goes to a local sink rather than the
      // process-global std::cout, which races with concurrent worker threads
      // (rysmith runs one generation thread per function). The value itself
      // comes back from run().
      std::stringstream sink;
      std::optional<RuntimeValue> result;
      try {
        Interpreter interp(prog, sink);
        // Capture the state profile from this same run when requested.
        if (outProfile) {
          outProfile->func = canonical;
          outProfile->granularity = gran;
          attachStateProfile(interp, *outProfile, gran);
        }
        result = interp.run(canonical, {}, paramArgs);
      } catch (...) {
        return std::nullopt;
      }

      if (!result)
        return std::nullopt; // the entry returned nothing to record
      std::string val = formatRuntimeValue(*result);
      if (val.empty())
        return std::nullopt; // an aggregate or vector, which has no one-line form
      return val;
    } catch (...) {
      return std::nullopt;
    }
  }

  bool validateNontermDiverges(
      const fs::path &sirPath, const std::string &funcName,
      const std::vector<std::string> &paramArgs, const std::string &headerLabel, int period,
      std::uint64_t maxBlocks
  ) {
    if (headerLabel.empty())
      return false;
    try {
      Program prog = parseSource(readFile(sirPath));
      if (!runAnalysisPasses(prog, /*verbose=*/false))
        return false;
      std::string canonical = funcName.empty() || funcName[0] == '@' ? funcName : "@" + funcName;

      std::stringstream sink;
      Interpreter interp(prog, sink);
      StateProfile profile;
      profile.func = canonical;
      profile.granularity = StateGranularity::Pbb;
      attachStateProfile(interp, profile, StateGranularity::Pbb);
      interp.setMaxBlockSteps(maxBlocks);

      try {
        interp.run(canonical, {}, paramArgs);
        return false; // returned within budget => terminated, not diverging
      } catch (const StepLimitError &) {
        // Ran the whole budget without returning — the expected outcome for a
        // divergent loop. Fall through to the header-recurrence check.
      } catch (...) {
        return false; // UB / require / other => not a clean divergence
      }

      // Confirm the header-state fixed point at runtime: two arrivals at the
      // lasso header exactly `period` laps apart must carry bit-identical
      // state. Two such identical states in a deterministic program prove the
      // orbit repeats forever, which is the property we are certifying. For a
      // period-k orbit *consecutive* arrivals deliberately differ, so the gap
      // has to be k, not 1.
      //
      // The last arrivals rather than the first, because a local declared
      // `= undef` does not enter the store until it is first assigned: at the
      // header's first visit the state is both smaller and partly undefined
      // (the usual case, since the header is often the entry block that
      // initializes the pointers). Late visits are past all initialization, so
      // the comparison needs no exemption and stays strictly bit-exact.
      const int k = std::max(1, period);
      std::vector<const StatePoint *> arrivals;
      for (const auto &pt: profile.trace)
        if (pt.instr == -1 && pt.block == headerLabel)
          arrivals.push_back(&pt);
      if ((int) arrivals.size() < k + 1)
        return false; // fewer than one full orbit within the budget
      const StatePoint *last = arrivals.back();
      const StatePoint *prev = arrivals[arrivals.size() - 1 - k];
      if (prev->vars.size() != last->vars.size())
        return false;
      for (std::size_t i = 0; i < last->vars.size(); ++i) {
        if (last->vars[i].first != prev->vars[i].first)
          return false;
        if (!bitExactEq(last->vars[i].second, prev->vars[i].second))
          return false;
      }
      return true; // header state recurred after k laps => diverges
    } catch (...) {
      return false;
    }
  }

  bool programTraps(
      const fs::path &sirPath, const std::string &funcName,
      const std::vector<std::string> &paramArgs
  ) {
    try {
      Program prog = parseSource(readFile(sirPath));
      if (!runAnalysisPasses(prog, /*verbose=*/false))
        return false;
      std::string canonical = funcName.empty() || funcName[0] == '@' ? funcName : "@" + funcName;
      std::stringstream sink;
      Interpreter interp(prog, sink);
      try {
        interp.run(canonical, {}, paramArgs);
        return false; // returned cleanly — no UB
      } catch (const UndefinedBehaviorError &) {
        return true; // trapped, as a --require-ub program should
      } catch (...) {
        return false; // require-failure / other — not the UB we wanted
      }
    } catch (...) {
      return false;
    }
  }

  // Structured emission (C/WASM --structured-lowering, python)
  // is only total on reducible CFGs. Callers filter or repair upstream;
  // verify here so a violation is a clean failure instead of
  // malformed backend output.
  static bool allFunsReducible(const Program &prog, bool verbose) {
    for (const auto &f: prog.funs) {
      DiagBag diags;
      CFG cfg = CFG::build(f, diags);
      DomTree dt = DomTree::build(cfg);
      if (!ReducibilityResult::check(cfg, dt).reducible()) {
        if (verbose)
          std::cerr << "reify: structured lowering requires reducible control flow: " << f.name.name
                    << "\n";
        return false;
      }
    }
    return true;
  }

  bool emitCInProcess(
      Program &prog, const fs::path &outDir, const std::string &primaryStem, const EmitOptions &opts
  ) {
    if (!runAnalysisPasses(prog, opts.verbose))
      return false;
    if (opts.structuredLowering && !allFunsReducible(prog, opts.verbose))
      return false;

    // The split and single-file forms differ only in where the output goes and
    // which emit call makes it; how the backend lowers is identical. Under
    // --split-by-source the backend opens its own files, so `sink` stays
    // closed and unused.
    std::ofstream sink;
    if (!opts.splitBySource) {
      fs::path outFile = outDir / (primaryStem + ".c");
      sink.open(outFile);
      if (!sink) {
        if (opts.verbose)
          std::cerr << "reify: cannot open " << outFile << "\n";
        return false;
      }
    }
    CBackend cb(sink);
    cb.setNoRequire(!opts.keepRequire);
    cb.setNoUbGuards(opts.noUbGuards);
    cb.setNoMainMangle(opts.emitMain);
    cb.setStructuredLowering(opts.structuredLowering);
    cb.setVecLowering(makeCVecLowering(opts.vecLowering.empty() ? "vecext" : opts.vecLowering));
    try {
      if (opts.splitBySource)
        cb.emitSplit(prog, outDir.string(), primaryStem);
      else
        cb.emit(prog);
    } catch (const std::exception &e) {
      if (opts.verbose)
        std::cerr << "reify: CBackend failed: " << e.what() << "\n";
      return false;
    }
    return true;
  }

  bool emitWasmInProcess(Program &prog, const fs::path &outFile, const EmitOptions &opts) {
    if (!runAnalysisPasses(prog, opts.verbose))
      return false;
    if (opts.structuredLowering && !allFunsReducible(prog, opts.verbose))
      return false;
    std::ofstream ofs(outFile);
    if (!ofs) {
      if (opts.verbose)
        std::cerr << "reify: cannot open " << outFile << "\n";
      return false;
    }
    auto vl = makeWasmVecLowering(opts.vecLowering.empty() ? "vecext" : opts.vecLowering);
    if (!vl) {
      if (opts.verbose)
        std::cerr << "reify: WASM target does not support vec-lowering '" << opts.vecLowering
                  << "'\n";
      return false;
    }
    WasmBackend wb(ofs);
    wb.setNoRequire(!opts.keepRequire);
    wb.setNoUbGuards(opts.noUbGuards);
    wb.setNoMainMangle(opts.emitMain);
    wb.setStructuredLowering(opts.structuredLowering);
    wb.setVecLowering(std::move(vl));
    try {
      wb.emit(prog);
    } catch (const std::exception &e) {
      if (opts.verbose)
        std::cerr << "reify: WasmBackend failed: " << e.what() << "\n";
      return false;
    }
    return true;
  }

  bool emitPyInProcess(Program &prog, const fs::path &outFile, const EmitOptions &opts) {
    if (!runAnalysisPasses(prog, opts.verbose))
      return false;
    if (!allFunsReducible(prog, opts.verbose))
      return false;
    auto vl = makePyVecLowering(opts.vecLowering.empty() ? "array" : opts.vecLowering);
    if (!vl) {
      if (opts.verbose)
        std::cerr << "reify: python target does not support vec-lowering '" << opts.vecLowering
                  << "'\n";
      return false;
    }
    std::ofstream ofs(outFile);
    if (!ofs) {
      if (opts.verbose)
        std::cerr << "reify: cannot open " << outFile << "\n";
      return false;
    }
    PyBackend pb(ofs);
    pb.setNoRequire(!opts.keepRequire);
    pb.setNoUbGuards(opts.noUbGuards);
    pb.setNoMainMangle(opts.emitMain);
    pb.setVecLowering(std::move(vl));
    try {
      pb.emit(prog);
    } catch (const std::exception &e) {
      if (opts.verbose)
        std::cerr << "reify: PyBackend failed: " << e.what() << "\n";
      return false;
    }
    return true;
  }

  bool compileSirInProcess(
      const fs::path &sirPath, const std::string &target, const fs::path &outPath,
      const EmitOptions &opts
  ) {
    // Read separately so a missing file keeps its own message rather than
    // surfacing as a compilation exception.
    std::string src;
    try {
      src = readFile(sirPath);
    } catch (const std::exception &) {
      if (opts.verbose)
        std::cerr << "compileSirInProcess: Could not open file " << sirPath << "\n";
      return false;
    }

    try {
      Program prog = parseSource(src);

      if (target == "c") {
        // This path emits one file, whatever the caller asked for.
        EmitOptions single = opts;
        single.splitBySource = false;
        return emitCInProcess(prog, outPath.parent_path(), outPath.stem().string(), single);
      } else if (target == "wasm") {
        return emitWasmInProcess(prog, outPath, opts);
      } else if (target == "python") {
        return emitPyInProcess(prog, outPath, opts);
      } else {
        if (opts.verbose)
          std::cerr << "compileSirInProcess: Unknown target " << target << "\n";
        return false;
      }
    } catch (const std::exception &e) {
      if (opts.verbose)
        std::cerr << "compileSirInProcess: Exception during compilation: " << e.what() << "\n";
      return false;
    }
  }

} // namespace refractir::reify
