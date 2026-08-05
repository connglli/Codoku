/**
 * rytwin — equivalence-preserving RefractIR program transformer.
 *
 * Given a generated program `p1` (a rysmith leaf or a rylink whole
 * program), rytwin produces an equivalent program `p2`. It first obtains
 * `p1`'s state profile — loading the `.state.json` sidecar (rysmith
 * --emit-state) when present, and otherwise interpreting `p1` in-process:
 * from `@main` when the program has one (so callee-frame states match what
 * runtime execution sees), else on the entry's solved input (descriptor
 * realization or `// SOLVED:` header). It then grafts, into selected
 * basic blocks of any profiled function, a synthesized twin block `B'`
 * that reproduces `B`'s effect on the exact state `B` sees, guarded by a
 * check on that live-in state so `B'` runs only on it and the original
 * `B` runs otherwise. Thus `p1(i) == p2(i)` for every input `i` (see
 * docs/reify.md).
 *
 * This driver wires the CLI, infers and loads the descriptor from the
 * input path, obtains the profile, runs the transform pipeline
 * (TwinTransform), and optionally validates and compiles the result.
 */

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include "ast/sir_printer.hpp"
#include "backend/wasm_vec_lowering.hpp"
#include "cxxopts.hpp"
#include "error.hpp"
#include "frontend/lexer.hpp"
#include "frontend/parser.hpp"
#include "reify/antiopt.hpp"
#include "reify/common.hpp"
#include "reify/func_desc.hpp"
#include "reify/state_profile.hpp"
#include "reify/transform.hpp"
#include "reify/twin_transform.hpp"

namespace fs = std::filesystem;
using namespace refractir;

// Block-step budget for the in-process profiling run when no descriptor is
// present (so the leaf's outcome is unknown). A UB-free terminating program
// finishes well within it; a non-terminating one hits the cap and is rejected
// instead of hanging. Generous because it only bounds this rare no-descriptor
// path — rysmith/rylink outputs always carry a descriptor.
static constexpr std::uint64_t kNoDescProfileStepCap = 3200;
using namespace refractir::reify;

// Parse p1's `// SOLVED: %p0=…, %p1=…, ret=…` header (written by rysmith
// and symirsolve --output) into name → canonical value text. Values are
// emitted by the canonical formatters, so they are safe to pass verbatim as
// symiri positional args.
[[nodiscard]] static std::unordered_map<std::string, std::string>
parseSolvedHeader(const std::string &src) {
  std::unordered_map<std::string, std::string> kv;
  const std::string tag = "// SOLVED:";
  std::size_t pos = src.find(tag);
  if (pos == std::string::npos)
    return kv;
  std::size_t eol = src.find('\n', pos);
  std::string line = src.substr(pos + tag.size(), eol - pos - tag.size());
  std::stringstream ss(line);
  std::string part;
  while (std::getline(ss, part, ',')) {
    std::size_t eq = part.find('=');
    if (eq == std::string::npos)
      continue;
    auto trim = [](std::string s) {
      std::size_t b = s.find_first_not_of(" \t");
      std::size_t e = s.find_last_not_of(" \t");
      return b == std::string::npos ? std::string{} : s.substr(b, e - b + 1);
    };
    kv[trim(part.substr(0, eq))] = trim(part.substr(eq + 1));
  }
  return kv;
}

// Resolve the entry function's parameter values for the solved input, in
// declaration order: prefer the descriptor realization for this .sir, fall
// back to p1's SOLVED header, and default to 0. rytwin profiles p1 at these
// values, so getting them wrong is loud: rysmith programs `require`
// interest conditions on their inputs, and a wrong input traps there.
[[nodiscard]] static std::vector<std::string> resolveParamArgs(
    const FunDecl &fn, const fs::path &sirFile, const std::optional<FuncDescriptor> &desc,
    const std::string &src
) {
  const FuncDescriptor::Realization *rz = nullptr;
  if (desc)
    for (const auto &r: desc->realizations)
      if (r.file == sirFile.filename().string()) {
        rz = &r;
        break;
      }
  const auto solved = parseSolvedHeader(src);
  std::vector<std::string> args;
  for (const auto &p: fn.params) {
    std::string v = "0";
    if (rz) {
      for (const auto &pv: rz->paramValues)
        if (pv.first == p.name.name) {
          v = pv.second;
          break;
        }
    } else if (auto it = solved.find(p.name.name); it != solved.end()) {
      v = it->second;
    }
    args.push_back(v);
  }
  return args;
}

// rysmith names a concrete file `func_<id>_<i>.sir` (single init) or
// `func_<id>_<i><a..z>.sir` (multi-init); the shared descriptor is
// `func_<id>_<i>.json`. Recover the descriptor stem by dropping a trailing
// init letter when it follows a digit (so the `<i>` digit run is kept).
[[nodiscard]] static std::string descriptorStem(const std::string &sirStem) {
  if (sirStem.size() >= 2) {
    char last = sirStem.back();
    char prev = sirStem[sirStem.size() - 2];
    if (last >= 'a' && last <= 'z' && std::isdigit((unsigned char) prev))
      return sirStem.substr(0, sirStem.size() - 1);
  }
  return sirStem;
}

// The entry function of a rysmith leaf program: the sole `fun` that is not
// the optional `@main` wrapper or a `@__twg_` guard from an earlier rytwin
// run. Prefer the descriptor's name when we have one — it is authoritative.
[[nodiscard]] static std::string
findEntry(const Program &prog, const std::optional<FuncDescriptor> &desc) {
  if (desc && !desc->name.empty())
    return desc->name;
  for (const auto &f: prog.funs)
    if (f.name.name != "@main" && !f.name.name.starts_with("@__twg_"))
      return f.name.name;
  return prog.funs.empty() ? std::string{} : prog.funs.front().name.name;
}

int main(int argc, char **argv) {
  cxxopts::Options opts("rytwin", "rytwin — equivalence-preserving RefractIR transformer");
  // clang-format off
  opts.add_options()
    ("input",   "Input concrete .sir (p1). Its descriptor (func_<id>_<i>.json) and, when "
                "present, state profile (<stem>.state.json) are read from the same "
                "directory following rysmith's naming; without a sidecar the profile is "
                "computed in-process by interpreting p1 on its solved input.",
                cxxopts::value<std::string>())
    ("p-twin",  "Probability of grafting a twin for each candidate region",
                cxxopts::value<double>()->default_value("0.5"))
    ("twin-select", "Which regions to twin: random (coin per candidate) or "
                "interesting (per-region softmax probability of interestingness)",
                cxxopts::value<std::string>()->default_value("random"))
    ("seed",    "RNG seed (default: random)", cxxopts::value<std::uint32_t>())
    ("target",  "Compile p2 to a target (sir = no compilation)",
                cxxopts::value<std::string>()->default_value("sir"))
    ("keep-require", "Keep require checks in compiled output")
    ("keep-ub-guards", "Keep dynamic UB guards in compiled output (default: false — the twin is assumed UB-free)")
    ("vec-lowering", "Vec-lowering strategy for C/WASM/Python backends",
                cxxopts::value<std::string>()->default_value("vecext"))
    ("emit-main", "Keep @main un-mangled in compiled output (so p2 is runnable)")
    ("validate", "Run symiri on p1 and p2 with the profiled input and assert they agree")
    ("selftest-antiopt", "Run every anti-optimization rule's example through the interpreter and exit")
    ("v,verbose", "Log each twin decision (grafted / skipped / rejected, with reason)")
    ("o,output","Output .sir (p2)", cxxopts::value<std::string>())
    ("h,help",  "Print usage");
  opts.parse_positional({"input"});
  opts.positional_help("<p1.sir>");
  // clang-format on

  cxxopts::ParseResult result;
  try {
    result = opts.parse(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "rytwin: " << e.what() << "\n";
    return 2;
  }
  if (result.count("selftest-antiopt")) {
    std::string failure;
    auto checked = selfTestRules(failure);
    if (!checked) {
      std::cerr << "rytwin: antiopt selftest FAILED: " << failure << "\n";
      return 1;
    }
    if (result.count("verbose"))
      for (const auto &nm: *checked)
        std::cout << "  checked " << nm << "\n";
    std::cout << "rytwin: antiopt selftest OK (" << checked->size()
              << " rule(s) run through the interpreter, every i8 operand value)\n";
    return 0;
  }

  if (result.count("help") || !result.count("input") || !result.count("output")) {
    std::cout << opts.help() << "\n";
    return result.count("help") ? 0 : 2;
  }

  fs::path inputPath = result["input"].as<std::string>();
  fs::path outputPath = result["output"].as<std::string>();
  double pTwin = result["p-twin"].as<double>();
  std::uint32_t seed = result.count("seed") ? result["seed"].as<std::uint32_t>()
                                            : static_cast<std::uint32_t>(std::random_device{}());

  std::string selectStr = result["twin-select"].as<std::string>();
  if (selectStr != "random" && selectStr != "interesting") {
    std::cerr << "rytwin: --twin-select must be random or interesting (got '" << selectStr
              << "')\n";
    return 2;
  }
  // Bundle the selection knobs into a policy; the transform is agnostic to
  // how a region's twin probability is chosen.
  SelectionPolicy selectPolicy =
      selectStr == "interesting" ? interestingPolicy(pTwin) : uniformPolicy(pTwin);

  std::string target = result["target"].as<std::string>();
  if (target != "sir" && target != "c" && target != "wasm") {
    std::cerr << "rytwin: --target must be sir, c, or wasm (got '" << target << "')\n";
    return 2;
  }
  bool keepRequire = result.count("keep-require") > 0;
  // The twin is assumed UB-free — the interpreter profiling it
  // rides on would fail on any UB — so drop the backends' dynamic UB
  // guards by default. --keep-ub-guards forces them back on (e.g. to
  // catch a mis-transformed twin trapping instead of misbehaving).
  bool noUbGuards = result.count("keep-ub-guards") == 0;
  bool emitMain = result.count("emit-main") > 0;
  bool doValidate = result.count("validate") > 0;
  std::string vecLowering = result["vec-lowering"].as<std::string>();
  if (target == "wasm" && vecLowering != "random" && !makeWasmVecLowering(vecLowering)) {
    std::cerr << "rytwin: wasm target does not support --vec-lowering '" << vecLowering << "'\n";
    return 2;
  }

  // 1. Load p1. Keep the source alive: Lexer holds a std::string_view into
  // it, so a temporary would dangle.
  Program prog;
  std::string src;
  try {
    src = readFile(inputPath);
    Lexer lx(src);
    Parser ps(lx.lexAll());
    prog = ps.parseProgram();
  } catch (const std::exception &e) {
    std::cerr << "rytwin: failed to parse " << inputPath << ": " << e.what() << "\n";
    return 1;
  }
  if (!runAnalysisPasses(prog, /*verbose=*/true)) {
    std::cerr << "rytwin: analysis of " << inputPath << " failed\n";
    return 1;
  }

  // 2. Load the descriptor, inferred from the input path via rysmith's
  // naming: `<dir>/func_<id>_<i>.json`.
  const fs::path dir = inputPath.parent_path();
  const std::string stem = inputPath.stem().string();

  std::optional<FuncDescriptor> desc;
  fs::path descPath = dir / (descriptorStem(stem) + ".json");
  if (fs::exists(descPath)) {
    desc = readFuncDescriptor(descPath);
    if (!desc)
      std::cerr << "rytwin: warning: could not read descriptor " << descPath << "\n";
  }
  // rytwin only transforms UB-free terminating programs. When the descriptor
  // is present, reject a trapping (--require-ub) or non-terminating
  // (--require-nonterm) leaf up front with a clear message: profiling one would
  // trap, the other would hang. Without a descriptor the bounded profiling run
  // below catches both cases.
  if (desc && desc->outcome != FuncDescriptor::Outcome::Return) {
    const char *kind =
        desc->outcome == FuncDescriptor::Outcome::Diverge ? "non-terminating" : "UB-triggering";
    std::cerr << "rytwin: cannot twin a " << kind << " program (" << inputPath
              << "); rytwin only transforms UB-free terminating programs\n";
    return 1;
  }
  std::string entry = findEntry(prog, desc);
  if (entry.empty()) {
    std::cerr << "rytwin: no entry function found in " << inputPath << "\n";
    return 1;
  }

  // 3. Obtain the state profile TwinTransform keys its guards on. Prefer a
  // `.state.json` sidecar (rysmith --emit-state) when one is present —
  // existing pipelines keep working unchanged — and otherwise derive the
  // profile from p1 itself by interpreting it in-process on its solved
  // input. A whole program (rylink output, or a leaf with --emit-main) is
  // profiled from `@main`: it encodes the realized input at every call
  // site, so callee-frame states match what runtime execution sees.
  const FunDecl *entryFn = nullptr;
  bool hasMain = false;
  for (const auto &f: prog.funs) {
    if (f.name.name == entry)
      entryFn = &f;
    if (f.name.name == "@main")
      hasMain = true;
  }
  if (!entryFn) {
    std::cerr << "rytwin: entry function " << entry << " not found in " << inputPath << "\n";
    return 1;
  }
  const std::string profEntry = hasMain ? "@main" : entry;
  std::vector<std::string> args =
      hasMain ? std::vector<std::string>{} : resolveParamArgs(*entryFn, inputPath, desc, src);
  std::optional<StateProfile> profile;
  fs::path statePath = dir / (stem + ".state.json");
  if (fs::exists(statePath)) {
    try {
      std::string json = readFile(statePath);
      profile = readStateProfileJson(json);
    } catch (const std::exception &e) {
      std::cerr << "rytwin: warning: could not read state profile " << statePath << ": " << e.what()
                << " — falling back to in-process profiling\n";
    }
    if (!profile)
      std::cerr << "rytwin: warning: could not parse state profile " << statePath
                << " — falling back to in-process profiling\n";
  }
  if (!profile) {
    // Without a descriptor we can't rule out a trapping or non-terminating
    // program up front, so bound the profiling run: a UB-free terminating
    // program finishes well within the budget, a trapping one throws UB, and a
    // non-terminating one hits the step cap — all rejected below. With a
    // descriptor the outcome check above already gated those out, so the run
    // is unbounded.
    const std::uint64_t stepCap = desc ? 0 : kNoDescProfileStepCap;
    try {
      profile = profileProgram(prog, profEntry, args, StateGranularity::Pbb, stepCap);
    } catch (const StepLimitError &) {
      std::cerr << "rytwin: profiling " << inputPath.filename() << " exceeded " << stepCap
                << " steps — refusing to twin a possibly non-terminating program\n";
      return 1;
    } catch (const std::exception &e) {
      std::cerr << "rytwin: failed to profile " << inputPath.filename()
                << " on its recorded input: " << e.what() << "\n";
      return 1;
    }
  }

  // 4. Assemble the transform context. `rng` is the tool's own stream; the
  // context borrows it by reference so every draw stays deterministic.
  std::mt19937 rng(seed);
  TransformContext ctx(rng);
  if (result.count("verbose"))
    ctx.verbose = &std::cerr;
  if (desc)
    ctx.descriptors[entry] = *desc;
  ctx.profiles[profile->func] = *profile;
  TransformPipeline pipe;
  pipe.add(makeTwinTransform(std::move(selectPolicy), doValidate));
  TransformReport rep = pipe.run(prog, ctx);
  if (!rep.ok) {
    std::cerr << "rytwin: pass failed: " << rep.message << "\n";
    return 1;
  }

  // No twin grafted means an unchanged copy of p1 — not a useful result, and
  // silently emitting it would look like success. Report it and write
  // nothing so callers can tell the two cases apart.
  if (rep.sites == 0) {
    std::cerr << "rytwin: no twin grafted for " << entry
              << " (no eligible block, or --p-twin too low); nothing written\n";
    return 1;
  }

  // Re-check the rewritten program so a malformed graft is caught here
  // rather than downstream in symiri / the backends.
  if (!runAnalysisPasses(prog, /*verbose=*/true)) {
    std::cerr << "rytwin: internal error: rewritten program failed re-analysis\n";
    return 1;
  }

  // 5. Emit p2.
  std::ofstream ofs(outputPath);
  if (!ofs) {
    std::cerr << "rytwin: cannot open " << outputPath << " for writing\n";
    return 1;
  }
  ofs << "// rytwin: equivalent of " << inputPath.filename().string() << " (" << rep.sites
      << " twin(s) grafted)\n\n";
  SIRPrinter printer(ofs);
  printer.print(prog);
  ofs.close(); // flush before symiri / symirc read the file back

  std::cout << "rytwin: wrote " << outputPath << " (" << rep.sites << " twin(s), entry " << entry
            << ")\n";

  // 6. Validate equivalence: run p1 and p2 on the profiled input and assert
  // they agree (same Result, or both trap).
  if (doValidate) {
    auto r1 = runSymiriCaptureResult(inputPath, profEntry, args);
    StateProfile p2Prof;
    auto r2 = runSymiriCaptureResult(outputPath, profEntry, args, &p2Prof);
    bool ok = (r1.has_value() == r2.has_value()) && (!r1.has_value() || *r1 == *r2);
    // A twin that never executes on the profiled input is dead code — the
    // guard was keyed on a state the run never reaches. That is a bug in
    // the graft, not a property of p1, so fail loudly.
    std::size_t fired = 0;
    for (const auto &pt: p2Prof.trace)
      if (pt.instr == -1 && pt.block.find("__twin") != std::string::npos)
        ++fired;
    ok = ok && fired > 0;
    std::cout << "rytwin: validated: " << (ok ? "OK" : "FAIL") << " (" << fired << " twin exec(s))";
    if (!rep.message.empty())
      std::cout << "; " << rep.message;
    if (!ok)
      std::cout << " (p1=" << (r1 ? *r1 : "<trap>") << " p2=" << (r2 ? *r2 : "<trap>") << ")";
    std::cout << "\n";
    if (!ok)
      return 1;
  }

  // 7. Optionally compile p2 to C / WASM (in-process, like rysmith / rylink).
  if (target != "sir") {
    fs::path outCompiled = outputPath;
    outCompiled.replace_extension(target == "c" ? ".c" : ".wat");
    if (!compileSirInProcess(
            outputPath, target, outCompiled, keepRequire, noUbGuards, vecLowering,
            /*structuredLowering=*/false, emitMain, /*verbose=*/false
        )) {
      std::cerr << "rytwin: compile of p2 to " << target << " failed\n";
      return 1;
    }
    std::cout << "rytwin: compiled " << outCompiled << "\n";
  }

  return 0;
}
