/**
 * rylink — whole-program generator over a rysmith function pool.
 *
 * Pipeline per generated program:
 *   1. Pick K functions from the pool (K from --n-nodes, capped by pool size).
 *   2. Build a DAG call-graph over those nodes (cg_gen).
 *   3. Parse each chosen .sir and merge into one bundled Program
 *      (deduplicating struct decls by name; rysmith already namespaces
 *      structs by genID so collisions only happen across same-id picks).
 *   4. For each (caller→callee) edge in the CG, drive CallRealizeTransform
 *      to splice a `call @callee(args)` into the caller body.
 *   5. Emit `prog_<id>_<i>/program.sir` (bundled, with CG/PARAMS/RET
 *      header comments). The bundled file is the source of truth for
 *      every downstream consumer.
 *   6. (--target c) Invoke symirc --split-by-source on program.sir to
 *      emit common.h + one .c per FunDecl::sourceStem.
 *   7. (--validate) Run symiri on program.sir with the entry's solved
 *      parameter values and assert the returned value equals the entry
 *      descriptor's ret.
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "analysis/definite_init.hpp"
#include "analysis/pass_manager.hpp"
#include "analysis/reachability.hpp"
#include "analysis/unused_name.hpp"
#include "ast/sir_printer.hpp"
#include "backend/c_backend.hpp"
#include "backend/c_vec_lowering.hpp"
#include "backend/emit.hpp"
#include "backend/py_vec_lowering.hpp"
#include "backend/wasm_backend.hpp"
#include "backend/wasm_vec_lowering.hpp"
#include "cxxopts.hpp"
#include "frontend/diagnostics.hpp"
#include "frontend/pipeline.hpp"
#include "frontend/semchecker.hpp"
#include "reify/antiopt_transform.hpp"
#include "reify/call_realize.hpp"
#include "reify/cg_gen.hpp"
#include "reify/checksum.hpp"
#include "reify/common.hpp"
#include "reify/func_desc.hpp"
#include "reify/func_pool.hpp"
#include "reify/hyperparameters.hpp"
#include "reify/id_gen.hpp"
#include "reify/transform.hpp"

namespace fs = std::filesystem;
using namespace refractir;
using namespace refractir::reify;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------


// Parse a rysmith-emitted .sir file into a Program. We deliberately skip
// the heavy analysis passes (sem/type/reachability) — rysmith produced
// these files itself and they're trusted well-formed.
[[nodiscard]] static std::optional<Program> parseSir(const fs::path &p) {
  try {
    return parseSource(readFile(p));
  } catch (const std::exception &e) {
    std::cerr << "rylink: parse failed for " << p << ": " << e.what() << "\n";
    return std::nullopt;
  }
}

// ---------------------------------------------------------------------------
// Bundle merging
// ---------------------------------------------------------------------------

struct Node {
  int idx;                    // index in the CG
  std::size_t poolIdx;        // index into FuncPool::entries
  std::size_t realizationIdx; // which realization we picked
  std::string funcName;       // e.g. "@func_<id>_<i>"
  std::string stem;           // basename without ".sir" — used for sourceStem
  FunDecl *fn = nullptr;      // points into the merged bundle's funs
};

// Pick K entries from the pool without replacement (capped by pool size).
// Duplicates would require renaming support we don't have yet, so when K
// exceeds the pool we shrink K rather than reuse entries.
[[nodiscard]] static std::vector<std::size_t>
pickPoolIndices(std::mt19937 &rng, std::size_t k, std::size_t poolSize) {
  std::vector<std::size_t> all(poolSize);
  std::iota(all.begin(), all.end(), 0);
  std::shuffle(all.begin(), all.end(), rng);
  if (k > poolSize)
    k = poolSize;
  all.resize(k);
  return all;
}

// Same-name structs in two merged programs must be structurally
// identical — otherwise a callee that returns a `@struct_X` would
// access fields that mean something different in the caller. With the
// `--id` flag gone, generation IDs are seed-derived 24-bit hex; two
// runs with different seeds in a 16M-wide space *can* collide, and
// rysmith's funcIdx-based suffix only protects siblings within the
// same run. We compare by field name + SIR-surface type string so a
// collision is detected up front instead of silently keeping the
// first decl and miscompiling the second program.
[[nodiscard]] static bool structsEqual(const StructDecl &a, const StructDecl &b) {
  if (a.fields.size() != b.fields.size())
    return false;
  for (std::size_t i = 0; i < a.fields.size(); ++i) {
    if (a.fields[i].name != b.fields[i].name)
      return false;
    if (SIRPrinter::typeToString(a.fields[i].type) != SIRPrinter::typeToString(b.fields[i].type))
      return false;
  }
  return true;
}

// Merge a per-fn Program into the bundle. Returns a pointer to the
// just-moved FunDecl, or nullptr on failure. Struct decls are
// deduplicated by name and *also* validated structurally — a same-name
// struct whose fields differ from the one already in the bundle aborts
// the merge.
//
// Pointer-stability precondition: `bundle.funs` MUST have been reserved
// to its final size before the first call. The returned FunDecl* is
// held by the per-program Node table across subsequent merges; without
// the reserve a later push_back would reallocate and invalidate every
// earlier pointer. Assert rather than silently corrupt.
[[nodiscard]] static FunDecl *mergeInto(
    Program &bundle, Program src, const std::string &sourceStem,
    std::unordered_map<std::string, std::size_t> &haveStructs,
    std::unordered_set<std::string> &haveIntrinsics
) {
  assert(
      bundle.funs.size() < bundle.funs.capacity() &&
      "rylink: bundle.funs must be reserved upfront — see generateOne"
  );
  // Index-keyed dedup: a pointer into `bundle.structs` would dangle
  // on the next push_back's reallocation, but indices stay valid
  // because we only ever push.
  for (auto &s: src.structs) {
    auto it = haveStructs.find(s.name.name);
    if (it == haveStructs.end()) {
      haveStructs.emplace(s.name.name, bundle.structs.size());
      bundle.structs.push_back(std::move(s));
    } else if (!structsEqual(bundle.structs[it->second], s)) {
      std::cerr << "rylink: struct " << s.name.name
                << " conflicts across merged programs — bailing out of this bundle\n";
      return nullptr;
    }
  }
  // Intrinsic declarations: dedup by (name, param types, return type).
  // The semchecker now supports same-name intrinsics with different
  // signatures, so we must keep @popcount i32 distinct from @popcount i64.
  for (auto &id: src.intrinsics) {
    if (haveIntrinsics.emplace(intrinsicSignature(id)).second)
      bundle.intrinsics.push_back(std::move(id));
  }

  if (src.funs.size() != 1)
    return nullptr; // rysmith always emits exactly one fun per file
  bundle.funs.push_back(std::move(src.funs[0]));
  FunDecl *fn = &bundle.funs.back();
  fn->sourceStem = sourceStem;
  return fn;
}

// ---------------------------------------------------------------------------
// Emit
// ---------------------------------------------------------------------------

// Write the bundled program.sir with header comments listing the call
// graph, the entry's solved parameter values, and the entry's solved
// return value. These comments are informational — symiri and symirc
// ignore them — but make the file self-describing for inspection.
static void writeBundledSir(
    const fs::path &outPath, const Program &bundle, const RyCG &cg, const std::vector<Node> &nodes,
    const FuncDescriptor::Realization &entryRz
) {
  std::ofstream ofs(outPath);
  ofs << "// ENTRY: " << nodes[cg.entry()].funcName << "\n";
  ofs << "// CG:\n";
  for (int i = 0; i < cg.nNodes; ++i) {
    ofs << "//   n" << i << " " << nodes[i].funcName;
    if (!cg.outEdges[i].empty()) {
      ofs << " ->";
      for (int j: cg.outEdges[i])
        ofs << " n" << j;
    }
    ofs << "\n";
  }
  ofs << "// PARAMS:";
  if (entryRz.paramValues.empty())
    ofs << " (none)";
  for (const auto &pv: entryRz.paramValues)
    ofs << " " << pv.first << "=" << pv.second;
  ofs << "\n";
  ofs << "// RETURN: " << (entryRz.retValue.empty() ? "(none)" : entryRz.retValue) << "\n\n";
  SIRPrinter sp(ofs);
  sp.print(bundle);
}

// ---------------------------------------------------------------------------
// Per-program pipeline
// ---------------------------------------------------------------------------

struct PerProgConfig {
  int nNodes = 4;
  double pEdge = rylink::hp::kPEdge;
  int maxOutDeg = rylink::hp::kMaxOutDegree;
  std::string genId;
  int progIdx = 0;
  fs::path outRoot;
  std::string target;      // "sir" | "c" | "wasm" | "python"
  std::string vecLowering; // "vecext" | "scalars" | "array" |
                           // "structscalars" | "structarray" | "random"
                           // (per-program resolution against `random`
                           // happens inside generateOne so each prog
                           // sweeps independently — matching rysmith).
  // "true" | "false" | "random" — structured lowering for the
  // C (goto-free) and WASM (dispatch-free) targets, resolved per program
  // like vecLowering. true/random (and the python target) require
  // reducible seeds: the pool is filtered on the descriptors'
  // `reducible` flag before generation.
  std::string structuredLowering = "false";
  bool keepRequire = false;
  bool keepUbGuards = false; // force UB guards on even for UB-free bundles
  // The common outcome of every pool leaf (the pool is required to be
  // homogeneous). Determines the whole program's behavior and how --validate
  // checks it: Return → assert the solved ret; Trap → assert it traps; Diverge
  // → bounded-replay assert it diverges.
  FuncDescriptor::Outcome poolKind = FuncDescriptor::Outcome::Return;
  bool validate = false;
  bool verbose = false;
  bool emitMain = false;
  // When false, emit one `<progDir>/program.c` instead of the
  // default per-`FunDecl::sourceStem` split + `common.h`. Useful for
  // consumers that prefer a single translation unit (e.g. quick
  // single-file compile loops, or shipping the generated C as one
  // self-contained snippet).
  bool splitBySource = true;
  bool antiOpt = true;
  // Probabilities that each bundled callee carries the matching
  // backend hint via FunDecl::Attributes. 0.0 (the default) preserves
  // behaviour. The entry function and any `@main` wrapper are
  // never marked. The C backend translates the bits into
  // `__attribute__((noinline))` / `__attribute__((noclone))`; WASM
  // ignores them.
  double pNoinlineCallees = 0.0;
  double pNocloneCallees = 0.0;
};

// Where one bundle's compiled output goes, and how to describe it in the log.
struct EmitTarget {
  fs::path emitDir;
  std::string emitStem;
  std::string progBase;
  std::string compiledReport;
  std::string failTag;
  bool noUbGuards = false;
};

// Compile `bundle` to cfg.target. Returns false when the backend rejects it.
[[nodiscard]] static bool
emitBundle(Program &bundle, std::mt19937 &rng, const PerProgConfig &cfg, const EmitTarget &t) {
  const fs::path &emitDir = t.emitDir;
  const std::string &emitStem = t.emitStem;
  const std::string &progBase = t.progBase;
  const std::string &compiledReport = t.compiledReport;
  const std::string &failTag = t.failTag;
  const bool noUbGuards = t.noUbGuards;
  (void) progBase;
  // Target backends. Both code paths run the analysis pipeline on the
  // in-memory bundle and call CBackend / WasmBackend directly so
  // FunDecl::sourceStem (set during mergeInto) survives into emitSplit
  // — going through symirc-as-subprocess would round-trip the bundle
  // through text and collapse every sourceStem to "".
  if (cfg.target == "c") {
    // Split mode: `emitStem = "program"` keys an (empty) primary
    // translation unit; the per-source .c files carry the bodies and
    // share a `common.h`. Flat mode: `emitStem = progBase` writes a
    // single self-contained `<progBase>.c`.
    //
    // Resolve `--vec-lowering` once per program against the per-prog
    // rng so each emitted bundle stamps a single strategy — matching
    // rysmith's per-fn semantics.  `random` cycles across the five
    // strategies so a multi-prog sweep exercises every backend
    // lowering.
    std::string vecLow = reify::pickVecLowering(rng, cfg.vecLowering);
    if (cfg.verbose && !vecLow.empty())
      std::cout << "  vec-lowering: " << vecLow << "\n";
    // Per-program structured coin, drawn after the vec-lowering pick
    // (and only for "random") — matching rysmith's stream discipline.
    bool structured = reify::pickStructuredLowering(rng, cfg.structuredLowering);
    if (cfg.verbose && structured)
      std::cout << "  structured-lowering: true\n";
    EmitOptions emitOpts;
    emitOpts.keepRequire = cfg.keepRequire;
    emitOpts.noUbGuards = noUbGuards;
    emitOpts.vecLowering = vecLow;
    emitOpts.structuredLowering = structured;
    emitOpts.emitMain = cfg.emitMain;
    emitOpts.splitBySource = cfg.splitBySource;
    emitOpts.verbose = cfg.verbose;
    if (!emitC(bundle, emitDir, emitStem, emitOpts)) {
      if (cfg.verbose)
        std::cerr << "  backend FAIL (" << failTag << ")\n";
      return false;
    }
    std::cout << "  compiled: " << compiledReport << "\n";
  } else if (cfg.target == "wasm") {
    fs::path wasmOut = emitDir / (emitStem + ".wat");
    std::string vecLow = reify::pickVecLowering(rng, cfg.vecLowering, "wasm");
    if (cfg.verbose && !vecLow.empty())
      std::cout << "  vec-lowering: " << vecLow << "\n";
    // Per-program structured coin, drawn after the vec-lowering pick
    // (and only for "random") — same stream discipline as the C branch.
    bool structured = reify::pickStructuredLowering(rng, cfg.structuredLowering);
    if (cfg.verbose && structured)
      std::cout << "  structured-lowering: true\n";
    EmitOptions emitOpts;
    emitOpts.keepRequire = cfg.keepRequire;
    emitOpts.noUbGuards = noUbGuards;
    emitOpts.vecLowering = vecLow;
    emitOpts.structuredLowering = structured;
    emitOpts.emitMain = cfg.emitMain;
    emitOpts.verbose = cfg.verbose;
    if (!emitWasm(bundle, wasmOut, emitOpts)) {
      if (cfg.verbose)
        std::cerr << "  backend FAIL (" << failTag << ")\n";
      return false;
    }
    std::cout << "  compiled: " << wasmOut << "\n";
  } else if (cfg.target == "python") {
    fs::path pyOut = emitDir / (emitStem + ".py");
    // Per-program strategy pick from the python set (mirrors the C
    // branch's per-program vec-lowering resolution).
    std::string vecLow = reify::pickVecLowering(rng, cfg.vecLowering, "python");
    if (cfg.verbose && !vecLow.empty())
      std::cout << "  vec-lowering: " << vecLow << "\n";
    EmitOptions emitOpts;
    emitOpts.keepRequire = cfg.keepRequire;
    emitOpts.noUbGuards = noUbGuards;
    emitOpts.vecLowering = vecLow;
    emitOpts.emitMain = cfg.emitMain;
    emitOpts.verbose = cfg.verbose;
    if (!emitPy(bundle, pyOut, emitOpts)) {
      if (cfg.verbose)
        std::cerr << "  backend FAIL (" << failTag << ")\n";
      return false;
    }
    std::cout << "  compiled: " << pyOut << "\n";
  }
  return true;
}

static bool generateOne(const FuncPool &pool, std::mt19937 &rng, const PerProgConfig &cfg) {
  if (pool.entries.empty())
    return false;
  int k = std::min(static_cast<int>(pool.entries.size()), std::max(1, cfg.nNodes));
  auto pickIdxs = pickPoolIndices(rng, static_cast<std::size_t>(k), pool.entries.size());

  CGGenConfig cgCfg{k, cfg.pEdge, cfg.maxOutDeg};
  RyCG cg = genCallGraph(rng, cgCfg);

  // Build the bundle by parsing each chosen .sir.
  // Reserve funs upfront so Node::fn pointers stay valid across the
  // subsequent push_backs — without this, vector growth invalidates
  // every earlier pointer and the rewrite phase reads freed memory.
  // Reserve k + 1 if emitMain is true so the capacity accommodates the main function.
  Program bundle;
  bundle.funs.reserve(cfg.emitMain ? k + 1 : k);
  std::unordered_map<std::string, std::size_t> haveStructs;
  std::unordered_set<std::string> haveIntrinsics;
  std::vector<Node> nodes(k);
  for (int i = 0; i < k; ++i) {
    nodes[i].idx = i;
    nodes[i].poolIdx = pickIdxs[i];
    const auto &entry = pool.entries[pickIdxs[i]];
    std::uniform_int_distribution<std::size_t> rd(0, entry.desc.realizations.size() - 1);
    nodes[i].realizationIdx = rd(rng);
    fs::path sirPath = entry.sirPaths[nodes[i].realizationIdx];
    nodes[i].funcName = entry.desc.name;
    nodes[i].stem = sirPath.stem().string();
    auto prog = parseSir(sirPath);
    if (!prog)
      return false;
    nodes[i].fn = mergeInto(bundle, std::move(*prog), nodes[i].stem, haveStructs, haveIntrinsics);
    if (!nodes[i].fn)
      return false;
  }

  // Realize the call graph. The per-node descriptors (input/oracle + the
  // concretized execution path) ride in the shared context, keyed by func
  // name; the plan carries the composition decision (which edges, which
  // realization). The context borrows `rng` by reference so the transform's
  // draws stay on this program's deterministic stream.
  TransformContext ctx(rng);
  for (int i = 0; i < cg.nNodes; ++i)
    ctx.descriptors[nodes[i].funcName] = pool.entries[nodes[i].poolIdx].desc;

  CallRealizePlan plan;
  for (int i = 0; i < cg.nNodes; ++i)
    for (int j: cg.outEdges[i])
      plan.edges.push_back({nodes[i].funcName, nodes[j].funcName, nodes[j].realizationIdx});

  TransformPipeline pipe;
  pipe.add(makeCallRealizeTransform(std::move(plan)));
  // Last, so realization sees the code the generator wrote — its sites are
  // literals in a caller — and the rewriting only ever reworks the finished
  // bundle.
  if (cfg.antiOpt)
    pipe.add(makeAntiOptTransform());
  pipe.run(bundle, ctx);

  // Backend-hint roll for callees. The bits live on
  // FunDecl::Attributes — the backend reads them and emits the
  // matching `__attribute__((noinline))` / `__attribute__((noclone))`
  // (combined when both are set) before the function signature. We
  // intentionally exclude the entry function (and `@main`, which is
  // appended below): the marker is meant to defeat IPA-CP across
  // *callee* boundaries — the entry has no caller in the bundle that
  // would inline it, and the C entry point can't be cloned anyway.
  if (cfg.pNoinlineCallees > 0.0 || cfg.pNocloneCallees > 0.0) {
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    const int entryNode = cg.entry();
    for (int i = 0; i < cg.nNodes; ++i) {
      if (i == entryNode)
        continue;
      if (!nodes[i].fn)
        continue;
      if (coin(rng) < cfg.pNoinlineCallees)
        nodes[i].fn->attributes.noInline = true;
      if (coin(rng) < cfg.pNocloneCallees)
        nodes[i].fn->attributes.noClone = true;
    }
  }

  // Pull entry's solved param/ret values from its descriptor for header.
  const auto &entryEntry = pool.entries[nodes[cg.entry()].poolIdx];
  const auto &entryRz = entryEntry.desc.realizations[nodes[cg.entry()].realizationIdx];

  // Append main function if requested. Each rylink program bundles ONE
  // specific realization of each entry/callee descriptor (the SIR
  // body is model-substituted per realization, so different
  // realizations have different bodies), so main must call the entry
  // with exactly that realization's paramValues and assert against
  // that realization's retValue. Calling with another realization's
  // params against this body would produce a different CRC32 and
  // trip @check_chksum.
  //
  // (multiple @check_chksum call sites in main) is intentionally
  // deferred: it requires bundling multiple realization bodies as
  // distinct FunDecls so main can vary args per-call without
  // mismatching the bundled body. A follow-up can lift this.
  if (cfg.emitMain) {
    const FunDecl *entryFn = nullptr;
    for (const auto &f: bundle.funs) {
      if (f.name.name == nodes[cg.entry()].funcName) {
        entryFn = &f;
        break;
      }
    }
    if (entryFn) {
      std::vector<std::string> paramVals;
      paramVals.reserve(entryFn->params.size());
      for (const auto &p: entryFn->params) {
        std::string val = "0";
        for (const auto &pv: entryRz.paramValues) {
          if (pv.first == p.name.name) {
            val = pv.second;
            break;
          }
        }
        paramVals.push_back(std::move(val));
      }
      // A diverging entry never returns, so it has no captured ret value; use a
      // random i32 so @check_chksum stays present (the compiler can't prove the
      // loop is infinite, so it must keep the computation). Return/Trap use the
      // solved ret (a Trap entry traps before reaching the — unreachable — check).
      std::string expected = entryRz.retValue;
      if (cfg.poolKind == FuncDescriptor::Outcome::Diverge && expected.empty())
        expected = std::to_string(static_cast<std::int32_t>(rng()));
      FunDecl mainFn = reify::buildMainFunction(bundle, *entryFn, paramVals, expected);
      bundle.funs.push_back(std::move(mainFn));
    }
  }

  // Decide output layout. The default `--split-by-source`
  // mode lands every artefact inside `<outRoot>/<progBase>/` so the
  // multi-`.c` + `common.h` fanout stays self-contained per program.
  // `--no-split-by-source` flattens to `<outRoot>/<progBase>.{sir,c,wasm}`,
  // mirroring rysmith's per-function `func_<id>_<idx>.{sir,c,json}`
  // layout — useful when the bundle is meant to be consumed as a
  // single self-contained translation unit.
  std::string progBase =
      std::string(rylink::hp::kProgPrefix) + "_" + cfg.genId + "_" + std::to_string(cfg.progIdx);
  fs::path emitDir;           // directory we pass to the backend helpers
  std::string emitStem;       // basename (no extension) used by the helpers
  fs::path programSir;        // where the .sir lands
  std::string compiledReport; // path echoed on the `compiled:` log line
  std::string failTag;        // label used in backend-FAIL diagnostics
  fs::create_directories(cfg.outRoot);
  if (cfg.splitBySource) {
    fs::path progDir = cfg.outRoot / progBase;
    fs::create_directories(progDir);
    emitDir = progDir;
    emitStem = "program";
    programSir = progDir / rylink::hp::kEntrySirName;
    compiledReport = progDir.string();
    failTag = progBase;
  } else {
    emitDir = cfg.outRoot;
    emitStem = progBase;
    programSir = cfg.outRoot / (progBase + ".sir");
    compiledReport = (cfg.outRoot / (progBase + "." + cfg.target)).string();
    failTag = progBase;
  }

  writeBundledSir(programSir, bundle, cg, nodes, entryRz);
  // Always echo the bundled .sir path so the user can find it in the
  // common case (target=sir). Matches rysmith's per-artifact
  // `concrete: <path>` line.
  std::cout << "  bundled: " << programSir << "\n";

  // The bundle is UB-free iff every constituent leaf is, so the
  // backends' dynamic UB guards can be dropped only then. A leaf
  // generated with rysmith --require-ub (has_ub) may trigger UB, so its
  // presence — or any legacy descriptor conservatively defaulting to
  // has_ub=true — keeps the guards on.
  bool noUbGuards = !cfg.keepUbGuards;
  for (const auto &n: nodes)
    if (pool.entries[n.poolIdx].desc.hasUb()) {
      noUbGuards = false;
      break;
    }

  {
    EmitTarget emitTarget{emitDir, emitStem, progBase, compiledReport, failTag, noUbGuards};
    if (!emitBundle(bundle, rng, cfg, emitTarget))
      return false;
  }

  // Validate the fused program on the entry's solved input, asserting the
  // behavior the homogeneous pool's outcome dictates.
  if (cfg.validate) {
    std::vector<std::string> args;
    for (const auto &pv: entryRz.paramValues)
      args.push_back(pv.second);
    const std::string &entryName = nodes[cg.entry()].funcName;
    bool ok = false;
    std::string detail;
    switch (cfg.poolKind) {
      case FuncDescriptor::Outcome::Return: {
        // Assert the entry returns its descriptor's solved ret value.
        auto got = runSymiriCaptureResult(programSir, entryName, args);
        ok = got && (*got == entryRz.retValue);
        detail =
            "expected=" + entryRz.retValue + " got=" + (got ? *got : std::string("<no Result>"));
        break;
      }
      case FuncDescriptor::Outcome::Trap:
        // Assert the whole program triggers UB (like rysmith --require-ub).
        ok = programTraps(programSir, entryName, args);
        detail = "expected UB but the program did not trap";
        break;
      case FuncDescriptor::Outcome::Diverge: {
        // Assert the program diverges: the entry is the unmodified diverging
        // leaf, whose lasso header is the last block of its descriptor path.
        // That path also carries the orbit's period — k+1 arrivals at the
        // header means k laps between two identical states.
        const auto &dpath = entryEntry.desc.path;
        std::string header = dpath.empty() ? "" : dpath.back();
        int period = dpath.empty()
                         ? 1
                         : static_cast<int>(std::count(dpath.begin(), dpath.end(), header)) - 1;
        ok = validateNontermDiverges(programSir, entryName, args, header, period);
        detail =
            "expected divergence (header=" + header + ", period=" + std::to_string(period) + ")";
        break;
      }
    }
    if (ok) {
      std::cout << "  validated: OK (" << failTag << ")\n";
    } else {
      std::cerr << "  validated: FAIL (" << failTag << ") " << detail << "\n";
      return false;
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  cxxopts::Options opts("rylink", "rylink — whole-program generator for RefractIR");
  // clang-format off
  opts.add_options()
    ("n,n-progs", "Number of whole programs to generate",
        cxxopts::value<int>()->default_value("1"))
    // CG
    ("n-nodes", "Target number of call-graph nodes per program",
        cxxopts::value<int>()->default_value("8"))
    ("max-outdeg", "Maximum out-degree per CG node",
        cxxopts::value<int>()->default_value(std::to_string(rylink::hp::kMaxOutDegree)))
    // Input and output
    ("i,input-dir", "Directory of rysmith-emitted (.sir + .json) pairs",
        cxxopts::value<std::string>()->default_value("rysmith_out"))
    ("o,output-dir", "Root output directory; each program lands in <root>/prog_<id>_<i>/",
        cxxopts::value<std::string>()->default_value("rylink_out"))
    ("target", "sir | c | wasm | python",
        cxxopts::value<std::string>()->default_value("sir"))
    ("vec-lowering", "Vec-lowering strategy for C/WASM/Python backends "
                     "(random|vecext|scalars|array|structscalars|structarray)",
        cxxopts::value<std::string>()->default_value("random"))
    ("structured-lowering", "Structured lowering for the C (goto-free) and WASM (dispatch-free) "
                            "targets: true|false|random; "
                            "true/random discard pool seeds whose descriptor is not reducible",
        cxxopts::value<std::string>()->default_value("false"))
    ("keep-require", "Keep `require` checks in C/WASM output",
        cxxopts::value<bool>()->default_value("false"))
    ("keep-ub-guards", "Keep dynamic UB guards even when the bundle is UB-free (default: false)",
        cxxopts::value<bool>()->default_value("false"))
    ("emit-main", "Generate a main wrapper in the output program",
        cxxopts::value<bool>()->default_value("false"))
    ("p-noinline-callees", "Probability each bundled callee is marked noinline",
        cxxopts::value<double>()->default_value("0.0"))
    ("p-noclone-callees", "Probability each bundled callee is marked noclone",
        cxxopts::value<double>()->default_value("0.0"))
    ("no-split-by-source", "Emit each program flat as "
                           "<outRoot>/prog_<id>_<i>.{sir,c,wasm} (one .c per "
                           "program, no common.h) instead of the default "
                           "<outRoot>/prog_<id>_<i>/ subdirectory with one "
                           ".c per FunDecl::sourceStem",
        cxxopts::value<bool>()->default_value("false"))
    ("no-antiopt", "Skip the anti-optimization rewriting of the bundled "
                   "program (the trap-free identities that stop every "
                   "generated function reading like the one generator that "
                   "wrote it)",
        cxxopts::value<bool>()->default_value("false"))
    // Validate
    ("validate", "Run symiri on each emitted program and check semantics",
        cxxopts::value<bool>()->default_value("false"))
    // Misc
    ("seed", "RNG seed (random if omitted)",
        cxxopts::value<std::uint32_t>())
    ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Print help");
  // clang-format on

  cxxopts::ParseResult res;
  try {
    res = opts.parse(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "rylink: " << e.what() << "\n" << opts.help() << "\n";
    return 2;
  }
  if (res.count("help")) {
    std::cout << opts.help() << "\n";
    return 0;
  }

  std::uint32_t seed = res.count("seed") ? res["seed"].as<std::uint32_t>() : std::random_device{}();
  std::mt19937 rng(seed);
  // Generation ID is always derived from --seed via the
  // shared genHexId helper — no CLI override. Two runs with the same
  // seed produce identical prog_<id>_<i> directories.
  std::string genId = genHexId(rng);

  PerProgConfig pc;
  pc.outRoot = res["output-dir"].as<std::string>();
  pc.nNodes = res["n-nodes"].as<int>();
  pc.maxOutDeg = res["max-outdeg"].as<int>();
  pc.pEdge = rylink::hp::kPEdge;
  pc.genId = genId;
  pc.target = res["target"].as<std::string>();
  pc.vecLowering = res["vec-lowering"].as<std::string>();
  pc.keepRequire = res["keep-require"].as<bool>();
  pc.keepUbGuards = res["keep-ub-guards"].as<bool>();
  pc.validate = res["validate"].as<bool>();
  pc.emitMain = res["emit-main"].as<bool>();
  pc.splitBySource = !res["no-split-by-source"].as<bool>();
  pc.antiOpt = !res["no-antiopt"].as<bool>();
  pc.verbose = res["v"].as<bool>();
  pc.pNoinlineCallees = res["p-noinline-callees"].as<double>();
  pc.pNocloneCallees = res["p-noclone-callees"].as<double>();
  for (auto [name, val]:
       {std::pair{"--p-noinline-callees", pc.pNoinlineCallees},
        std::pair{"--p-noclone-callees", pc.pNocloneCallees}}) {
    if (val < 0.0 || val > 1.0) {
      std::cerr << "rylink: " << name << " must be in [0, 1] (got " << val << ")\n";
      return 2;
    }
  }

  if (pc.target != "sir" && pc.target != "c" && pc.target != "wasm" && pc.target != "python") {
    std::cerr << "rylink: --target must be sir | c | wasm | python\n";
    return 2;
  }
  pc.structuredLowering = res["structured-lowering"].as<std::string>();
  if (pc.structuredLowering != "true" && pc.structuredLowering != "false" &&
      pc.structuredLowering != "random") {
    std::cerr << "rylink: --structured-lowering must be true | false | random\n";
    return 2;
  }
  // The python backend rejects vecext (no native SIMD value type);
  // catch an explicit request up-front instead of per-program.
  if (pc.target == "python" && pc.vecLowering != "random" && !makePyVecLowering(pc.vecLowering)) {
    std::cerr << "rylink: python target does not support --vec-lowering '" << pc.vecLowering
              << "' (try random|array|scalars|structscalars|structarray)\n";
    return 2;
  }
  if (pc.target == "wasm" && pc.vecLowering != "random" && !makeWasmVecLowering(pc.vecLowering)) {
    std::cerr << "rylink: wasm target does not support --vec-lowering '" << pc.vecLowering
              << "' (try random|vecext|array|scalars)\n";
    return 2;
  }
  // Validate `--vec-lowering` up-front so a typo bites before any
  // generation work — passing through to `pickVecLowering` would
  // silently fall through to the verbatim path and then trip an
  // unrecognised strategy inside `makeVecLowering`.
  {
    static const char *known[] = {"random", "vecext",        "scalars",
                                  "array",  "structscalars", "structarray"};
    bool ok = false;
    for (const char *k: known)
      if (pc.vecLowering == k) {
        ok = true;
        break;
      }
    if (!ok) {
      std::cerr << "rylink: --vec-lowering must be one of "
                   "random|vecext|scalars|array|structscalars|structarray\n";
      return 2;
    }
  }

  // No `--target` sibling-binary discovery any more: the C
  // and WASM backends are linked directly into rylink, so all
  // target=c / target=wasm runs work without symirc on disk.

  std::cout << "rylink: master seed = " << seed << "\n";
  std::cout << "rylink: generation id = " << genId << "\n";
  FuncPool pool = loadFuncPool(res["input-dir"].as<std::string>());
  // Structuring consumers only handle reducible CFGs, and
  // seed programs (older pools, runs without --require-reducible) may
  // not be: discard every seed whose descriptor is not known
  // reducible. Descriptors predating the `reducible` field parse as
  // false and are conservatively discarded too.
  if (pc.structuredLowering != "false" || pc.target == "python") {
    std::size_t before = pool.entries.size();
    std::erase_if(pool.entries, [](const PoolEntry &e) { return !e.desc.reducible; });
    if (std::size_t discarded = before - pool.entries.size()) {
      std::cout << "rylink: discarded " << discarded
                << " non-reducible seed(s) for structured lowering\n";
    }
    if (pool.entries.empty() && before > 0) {
      std::cerr << "rylink: no reducible seeds left in pool — aborting "
                   "(regenerate with rysmith --require-reducible)\n";
      return 1;
    }
  }
  if (pool.entries.empty()) {
    std::cerr << "rylink: empty pool — aborting\n";
    return 1;
  }
  // rylink composes a HOMOGENEOUS pool: every leaf must share the same
  // runtime outcome (all Return, all Trap, or all Diverge). A mixed pool has no
  // well-defined fused behavior — e.g. a returning caller splicing a call to a
  // trapping or non-terminating callee — so it is rejected. The common outcome
  // becomes the whole program's outcome and drives @main + --validate.
  pc.poolKind = pool.entries.front().desc.outcome;
  for (const auto &e: pool.entries)
    if (e.desc.outcome != pc.poolKind) {
      std::cerr << "rylink: pool is not homogeneous — every leaf must have the same outcome "
                   "(all return, all trap, or all diverge). Regenerate the pool with a single "
                   "rysmith mode (default / --require-ub / --require-nonterm).\n";
      return 1;
    }
  std::cout << "rylink: pool size = " << pool.entries.size() << " (outcome: "
            << (pc.poolKind == FuncDescriptor::Outcome::Return    ? "return"
                : pc.poolKind == FuncDescriptor::Outcome::Diverge ? "diverge"
                                                                  : "trap")
            << ")\n";

  fs::create_directories(pc.outRoot);
  int nProgs = std::max(1, res["n-progs"].as<int>());
  int nOk = 0, nFail = 0;
  // Per-program retry budget — see rylink::hp::kMaxAttemptsPerProg for
  // the rationale. The rewrite engine itself is sound; this just
  // tolerates the occasional rng-induced downstream miss.
  constexpr int kMaxAttempts = rylink::hp::kMaxAttemptsPerProg;
  auto wallStart = std::chrono::steady_clock::now();
  for (int i = 0; i < nProgs; ++i) {
    pc.progIdx = i;
    std::string progName = "prog_" + genId + "_" + std::to_string(i);
    std::cout << "[" << (i + 1) << "/" << nProgs << "] generating " << progName << "\n";
    bool succeeded = false;
    int usedAttempts = 0;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
      ++usedAttempts;
      std::mt19937 progRng(rng());
      if (generateOne(pool, progRng, pc)) {
        succeeded = true;
        break;
      }
      if (pc.verbose)
        std::cerr << "  retry " << (attempt + 1) << "/" << kMaxAttempts << "\n";
    }
    if (succeeded) {
      ++nOk;
      // Per-program seal — matches rysmith's per-fn `compiled:` /
      // `validated:` discipline; the actual artefact paths were
      // already echoed by generateOne.
      std::cout << "  "
                << (usedAttempts > 1
                        ? "completed (after " + std::to_string(usedAttempts) + " attempts): "
                        : "completed: ")
                << progName << "\n";
    } else {
      ++nFail;
      std::cerr << "  [FAIL] " << progName << " after " << kMaxAttempts << " attempts\n";
    }
  }
  auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
  double throughput = elapsed > 0 ? nOk / elapsed : 0.0;
  std::cout << "\nDone: " << nOk << " succeeded, " << nFail << " failed (total " << nProgs << ")"
            << "  [" << elapsed << "s, " << throughput << " progs/s]\n";
  return 0;
}
