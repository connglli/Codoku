#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "analysis/cfg.hpp"
#include "ast/ast.hpp"
#include "cxxopts.hpp"
#include "error.hpp"
#include "frontend/lexer.hpp"
#include "frontend/parser.hpp"

namespace fs = std::filesystem;
using namespace refractir;

#include "analysis/definite_init.hpp"
#include "analysis/pass_manager.hpp"
#include "analysis/reachability.hpp"
#include "analysis/unused_name.hpp"
#include "frontend/semchecker.hpp"
#include "frontend/typechecker.hpp"
#include "puzzle_common.hpp"

// ---------------------------------------------------------------------------
// Check result — ordered from easiest to hardest to satisfy.
//
// Each stage is a strict prerequisite for the next.  Stages:
//
//   (1) FAIL_BASICS     — Basics (missing markers, unfilled FILL_XXX marks)
//   (2) FAIL_PARSE      — Parse (solution fails to parse)
//   (3) FAIL_REMASKING  — Re-masking (skeleton mismatch)
//   (4) FAIL_COMPILE    — Compile (solution fails typechecking/analysis)
//   (5) FAIL_CFG        — CFG topology does not match the declared edges exactly
//   (6) FAIL_PATH       — execution did not follow the prescribed path exactly
//   (7) FAIL_OUTPUT      — check_chksum reports a wrong result (non-0 exit)
//   (8) FAIL_FILL_CONST — constant budget multiset mismatch
//   (9) PASS            — all checks passed
// ---------------------------------------------------------------------------
enum class CheckResult {
  PASS,
  FAIL_BASICS,
  FAIL_PARSE,
  FAIL_REMASKING,
  FAIL_COMPILE,
  FAIL_CFG,
  FAIL_PATH,
  FAIL_OUTPUT,
  FAIL_FILL_CONST,
};

static const char *checkResultTag(CheckResult r) {
  switch (r) {
    case CheckResult::PASS:
      return "[PASS]";
    case CheckResult::FAIL_BASICS:
      return "[FAIL_BASICS]";
    case CheckResult::FAIL_PARSE:
      return "[FAIL_PARSE]";
    case CheckResult::FAIL_REMASKING:
      return "[FAIL_REMASKING]";
    case CheckResult::FAIL_COMPILE:
      return "[FAIL_COMPILE]";
    case CheckResult::FAIL_CFG:
      return "[FAIL_CFG]";
    case CheckResult::FAIL_PATH:
      return "[FAIL_PATH]";
    case CheckResult::FAIL_OUTPUT:
      return "[FAIL_OUTPUT]";
    case CheckResult::FAIL_FILL_CONST:
      return "[FAIL_FILL_CONST]";
  }
  return "[FAIL]";
}

// --- Puzzle requirements (parsed from the puzzle header's machine markers) ---
//
// rypuzmk emits three kinds of machine-readable marker lines in the header:
//
//   //@ CFG_EDGE: A -> B
//   //@ EXEC_PATH: entry -> b0 -> ... -> exit
//   //@ FILL_CONST: <value> <count>
//
// Parsing only these markers (rather than scraping prose) keeps the validator
// robust to header wording changes.
struct CfgEdge {
  std::string from, to;
};

struct PuzzleRequirements {
  std::vector<std::string> expectedPath;
  std::unordered_map<std::string, int> constCounts;
  std::vector<CfgEdge> cfgEdges; // parsed from //@ CFG_EDGE: A -> B
};

PuzzleRequirements parsePuzzleRequirements(const std::string &puzzleText) {
  PuzzleRequirements reqs;
  const std::string CONST_MARK = "//@ FILL_CONST:";
  const std::string EDGE_MARK = "//@ CFG_EDGE:";

  std::istringstream iss(puzzleText);
  std::string line;
  while (std::getline(iss, line)) {
    size_t p = line.find("//@ EXEC_PATH:");
    if (p != std::string::npos) {
      size_t mark_len = 14;
      // Strip whitespace, then split the remainder on "->".
      std::string cleaned;
      for (char c: line.substr(p + mark_len)) {
        if (!std::isspace(static_cast<unsigned char>(c)))
          cleaned.push_back(c);
      }
      size_t start = 0, arrow;
      while ((arrow = cleaned.find("->", start)) != std::string::npos) {
        reqs.expectedPath.push_back(cleaned.substr(start, arrow - start));
        start = arrow + 2;
      }
      if (start < cleaned.size())
        reqs.expectedPath.push_back(cleaned.substr(start));
    } else if (size_t c = line.find(CONST_MARK); c != std::string::npos) {
      std::istringstream ls(line.substr(c + CONST_MARK.size()));
      std::string val;
      long cnt = 0;
      if (!(ls >> val >> cnt)) {
        throw std::runtime_error("Malformed //@ FILL_CONST marker in line: " + line);
      }
      reqs.constCounts[val] = static_cast<int>(cnt);
    } else if (size_t e = line.find(EDGE_MARK); e != std::string::npos) {
      // Format: "//@ CFG_EDGE: A -> B"
      std::string rest = line.substr(e + EDGE_MARK.size());
      // Trim leading whitespace.
      size_t start = rest.find_first_not_of(' ');
      if (start != std::string::npos)
        rest = rest.substr(start);
      size_t arrow = rest.find(" -> ");
      if (arrow == std::string::npos) {
        throw std::runtime_error("Malformed //@ CFG_EDGE marker: missing ' -> ' in: " + line);
      }
      std::string from = rest.substr(0, arrow);
      std::string to = rest.substr(arrow + 4);
      // Trim trailing/leading whitespace from each token.
      auto trim = [](const std::string &s) {
        size_t f = s.find_first_not_of(" \t\r\n");
        size_t l = s.find_last_not_of(" \t\r\n");
        return (f == std::string::npos) ? "" : s.substr(f, l - f + 1);
      };
      from = trim(from);
      to = trim(to);
      if (from.empty() || to.empty()) {
        throw std::runtime_error("Malformed //@ CFG_EDGE marker: empty node name in: " + line);
      }
      reqs.cfgEdges.push_back({from, to});
    }
  }
  return reqs;
}

// --- Interpreter execution + trace ---
struct ExecResult {
  int exitCode = -1;                  // symiri process exit status (0 == success)
  std::vector<std::string> blockPath; // labels of basic blocks entered, in order
};

/**
 * @brief Runs `symiri --dump-trace` once, capturing both the program exit code
 * and the executed basic-block trace. A single run gives us both the path
 * (for FAIL_PATH) and the correctness verdict (for FAIL_OUTPUT, via the
 * embedded check_chksum that aborts on mismatch).
 */
ExecResult runSymiri(const std::string &symiriPath, const std::string &solutionPath) {
  std::string cmd = symiriPath + " --dump-trace " + solutionPath + " 2>/dev/null";
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    throw std::runtime_error("Failed to run symiri --dump-trace");
  }
  ExecResult r;
  char buffer[512];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    std::string ln(buffer);
    if (ln.rfind("^", 0) == 0) {
      size_t colon = ln.find(":");
      if (colon != std::string::npos)
        r.blockPath.push_back(ln.substr(1, colon - 1));
    }
  }
  int status = pclose(pipe);
  r.exitCode = (status == -1 || !WIFEXITED(status)) ? -1 : WEXITSTATUS(status);
  return r;
}

/**
 * @brief Normalises RefractIR text for the structural-integrity compare:
 * removes `//` comments and all whitespace, but preserves the contents of
 * string literals verbatim (so a `//` or space inside a string is not eaten).
 */
std::string stripCommentsAndWhitespace(const std::string &str) {
  std::string out;
  out.reserve(str.size());
  bool inComment = false, inString = false;
  for (size_t i = 0; i < str.size(); ++i) {
    char ch = str[i];
    if (inComment) {
      if (ch == '\n')
        inComment = false;
      continue;
    }
    if (inString) {
      out.push_back(ch);
      if (ch == '\\' && i + 1 < str.size()) {
        out.push_back(str[++i]); // keep escaped char verbatim
      } else if (ch == '"') {
        inString = false;
      }
      continue;
    }
    if (ch == '/' && i + 1 < str.size() && str[i + 1] == '/') {
      inComment = true;
      ++i;
    } else if (ch == '"') {
      inString = true;
      out.push_back(ch);
    } else if (!std::isspace(static_cast<unsigned char>(ch))) {
      out.push_back(ch);
    }
  }
  return out;
}

int main(int argc, char **argv) {
  cxxopts::Options options("rypuzchk", "RefractIR Puzzle Validator");
  // clang-format off
  options.add_options()
      ("puzzle", "Puzzle file path", cxxopts::value<std::string>())
      ("solution", "Solution file path", cxxopts::value<std::string>())
      ("symiri", "Path to symiri binary", cxxopts::value<std::string>()->default_value((fs::path(argv[0]).parent_path() / "symiri").string()))
      ("h,help", "Print usage");
  options.parse_positional({"puzzle", "solution"});
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result.count("help") || !result.count("puzzle") || !result.count("solution")) {
    std::cout << options.help() << std::endl;
    return 1;
  }

  std::string puzzlePath = result["puzzle"].as<std::string>();
  std::string solutionPath = result["solution"].as<std::string>();
  // Default symiri to the one packaged next to this binary, so a sandbox's
  // ./tools/rypuzchk finds ./tools/symiri regardless of the caller's CWD.
  std::string symiriPath;
  if (result.count("symiri")) {
    symiriPath = result["symiri"].as<std::string>();
  } else {
    std::error_code ec;
    fs::path exe = fs::canonical("/proc/self/exe", ec);
    symiriPath = (ec ? fs::path("./symiri") : exe.parent_path() / "symiri").string();
  }

  // Helper: emit a tagged failure message and return the appropriate exit code.
  auto fail = [](CheckResult r, const std::string &msg) -> int {
    std::cerr << checkResultTag(r) << " " << msg << "\n";
    return 1;
  };

  try {
    // -----------------------------------------------------------------------
    // Stage 1 — FAIL_BASICS
    // Read puzzle banner and parse machine markers.
    // A missing //@ EXEC_PATH: is a hard error before we even touch the solution.
    // -----------------------------------------------------------------------
    std::string puzzleText = readFile(puzzlePath);
    PuzzleRequirements reqs = parsePuzzleRequirements(puzzleText);
    if (reqs.expectedPath.empty()) {
      return fail(
          CheckResult::FAIL_BASICS, "Puzzle is missing a '//@ EXEC_PATH:' marker; cannot validate."
      );
    }
    if (reqs.cfgEdges.empty()) {
      return fail(
          CheckResult::FAIL_BASICS, "Puzzle is missing '//@ CFG_EDGE:' markers; cannot validate."
      );
    }

    // -----------------------------------------------------------------------
    // Stage 1 — FAIL_BASICS
    // Verify that the solution contains no unfilled FILL_XXX marks.
    // -----------------------------------------------------------------------
    std::string solutionText = readFile(solutionPath);
    std::string strippedSol = stripCommentsAndWhitespace(solutionText);
    bool hasFillMarks = false;
    for (const char *mark:
         {"FILL_VAR", "FILL_CONST", "FILL_OP", "FILL_TYPE", "FILL_LABEL", "FILL_FUNC",
          "FILL_FIELD"}) {
      if (strippedSol.find(mark) != std::string::npos) {
        hasFillMarks = true;
        break;
      }
    }
    if (hasFillMarks) {
      return fail(CheckResult::FAIL_BASICS, "Solution still contains unfilled FILL_XXX marks.");
    }

    // -----------------------------------------------------------------------
    // Stage 2 — FAIL_PARSE
    // Parse the solution file.
    // -----------------------------------------------------------------------
    Program prog;
    try {
      prog = parseProgramFromString(solutionText);
    } catch (const std::exception &e) {
      return fail(CheckResult::FAIL_PARSE, std::string("Solution fails to parse: ") + e.what());
    }

    const FunDecl *leaf = findLeafFunction(prog);
    if (!leaf) {
      return fail(CheckResult::FAIL_PARSE, "Could not find leaf function in solution.");
    }

    // -----------------------------------------------------------------------
    // Stage 3 — FAIL_REMASKING (skeleton comparison)
    // Re-mask the solution and compare the result byte-for-byte against the puzzle.
    // -----------------------------------------------------------------------
    std::unordered_set<int> maskSet = inferMaskSetFromPuzzle(*leaf, prog, puzzleText);
    int totalPositions = countMaskableStatements(*leaf);
    // If every position is masked, pass nullptr to take the cheaper "mask
    // everything" fast-path — same as what rypuzmk does for --p-mask 1.0.
    const std::unordered_set<int> *maskPtr;
    if (static_cast<int>(maskSet.size()) == totalPositions) {
      maskPtr = nullptr;
    } else {
      maskPtr = &maskSet;
    }

    std::ostringstream mss;
    SIRMaskedPrinter(mss, *leaf, /*enableMasking=*/true, maskPtr).print(prog);
    if (stripCommentsAndWhitespace(mss.str()) != stripCommentsAndWhitespace(puzzleText)) {
      return fail(
          CheckResult::FAIL_REMASKING,
          "Solution structural integrity check failed.\n"
          "  You may have changed code outside the FILL_XXX marks, or introduced\n"
          "  unauthorized variables / statements / basic blocks."
      );
    }

    // -----------------------------------------------------------------------
    // Stage 4 — FAIL_COMPILE (Typechecking and semantic checking)
    // Run the compiler analysis passes (SemChecker, TypeChecker, analyses).
    // -----------------------------------------------------------------------
    {
      DiagBag diags;
      refractir::PassManager pm(diags);
      pm.addModulePass(std::make_unique<SemChecker>());
      pm.addModulePass(std::make_unique<TypeChecker>());
      pm.addFunctionPass(std::make_unique<ReachabilityAnalysis>());
      pm.addFunctionPass(std::make_unique<DefiniteInitAnalysis>());
      pm.addFunctionPass(std::make_unique<UnusedNameAnalysis>());

      if (pm.run(prog) == refractir::PassResult::Error) {
        std::ostringstream msg;
        msg << "Solution fails typechecking or analysis:\n";
        for (const auto &d: diags.diags) {
          if (d.level == DiagLevel::Error) {
            msg << "  " << d.message << "\n";
          }
        }
        return fail(CheckResult::FAIL_COMPILE, msg.str());
      }
    }

    // -----------------------------------------------------------------------
    // Stage 5 — FAIL_CFG
    // Build the solution's CFG and compare its edge set exactly against the
    // //@ CFG_EDGE: markers declared by rypuzmk.  A solution is structurally
    // correct only when its FILL_LABEL choices produce a CFG that is
    // *isomorphic in terms of named edges* to the one the puzzle describes.
    //
    // This is strictly stronger than checking PATH edges: it verifies the
    // *complete* branching topology (including branches not taken on the
    // specified path), so a solver cannot hide a wrong CFG behind a lucky
    // path match.
    //
    // -----------------------------------------------------------------------
    {
      DiagBag cfgDiags;
      CFG cfg = CFG::build(*leaf, cfgDiags);
      if (cfgDiags.hasErrors()) {
        return fail(CheckResult::FAIL_CFG, "CFG construction failed for the solution.");
      }

      // Build the actual edge set from the solution's CFG, stripping the
      // leading '^' that CFGBuilder preserves from block labels.
      auto stripCaret = [](const std::string &s) -> std::string {
        return (s.rfind("^", 0) == 0) ? s.substr(1) : s;
      };
      auto edgeKey = [](const std::string &a, const std::string &b) { return a + "->" + b; };
      std::set<std::string> actualSet, declaredSet;
      for (size_t i = 0; i < cfg.blocks.size(); ++i) {
        std::string from = stripCaret(cfg.blocks[i]);
        for (auto sIdx: cfg.succ[i]) {
          std::string to = stripCaret(cfg.blocks[sIdx]);
          actualSet.insert(edgeKey(from, to));
        }
      }
      for (const auto &e: reqs.cfgEdges)
        declaredSet.insert(edgeKey(e.from, e.to));

      if (actualSet != declaredSet) {
        std::ostringstream msg;
        msg << "CFG topology mismatch.\n";
        // Report edges present in solution but not declared.
        for (const auto &k: actualSet)
          if (!declaredSet.count(k))
            msg << "  unexpected edge: " << k << "\n";
        // Report edges declared but missing from solution.
        for (const auto &k: declaredSet)
          if (!actualSet.count(k))
            msg << "  missing edge:    " << k << "\n";
        return fail(CheckResult::FAIL_CFG, msg.str());
      }
    }

    // -----------------------------------------------------------------------
    // Stage 6 — FAIL_PATH  +  Stage 7 — FAIL_OUTPUT
    // A single symiri --dump-trace run yields both the trace (path) and the
    // exit code (output correctness via the embedded check_chksum).  We check
    // them in order: path first so a wrong path is not misreported as a
    // wrong output.
    // -----------------------------------------------------------------------
    ExecResult exec = runSymiri(symiriPath, solutionPath);

    // --- Stage 6: path ---
    const FunDecl *mainFn = findMainFunction(prog);
    size_t mainBlocks = mainFn ? mainFn->blocks.size() : 1;
    if (exec.blockPath.size() <= mainBlocks) {
      return fail(
          CheckResult::FAIL_PATH, "Solution trace is empty or too short to contain the leaf path."
      );
    }
    std::vector<std::string> leafPath(exec.blockPath.begin() + mainBlocks, exec.blockPath.end());

    if (leafPath != reqs.expectedPath) {
      std::ostringstream msg;
      msg << "Execution path mismatch.\n  Expected: ";
      for (size_t i = 0; i < reqs.expectedPath.size(); ++i)
        msg << reqs.expectedPath[i] << (i + 1 < reqs.expectedPath.size() ? " -> " : "");
      msg << "\n  Actual:   ";
      for (size_t i = 0; i < leafPath.size(); ++i)
        msg << leafPath[i] << (i + 1 < leafPath.size() ? " -> " : "");
      return fail(CheckResult::FAIL_PATH, msg.str());
    }

    // --- Stage 7: output (check_chksum exit code) ---
    if (exec.exitCode != 0) {
      std::ostringstream msg;
      msg << "Solution output is incorrect (check_chksum mismatch; symiri exit code "
          << exec.exitCode << ").";
      return fail(CheckResult::FAIL_OUTPUT, msg.str());
    }

    // -----------------------------------------------------------------------
    // Stage 8 — FAIL_FILL_CONST
    // The re-masking check already passed, so maskPtr partitions the solution
    // exactly as the puzzle does.  Verify the constant multiset in masked
    // positions matches the budget declared in the banner.
    // -----------------------------------------------------------------------
    if (!reqs.constCounts.empty()) {
      MaskedConstantCollector collector;
      collector.selectiveMask = maskPtr;
      collector.collect(*leaf);

      for (const auto &pair: reqs.constCounts) {
        int actual = collector.counts.count(pair.first) ? collector.counts[pair.first] : 0;
        if (actual != pair.second) {
          std::ostringstream msg;
          msg << "FILL_CONST count mismatch for '" << pair.first << "'. Expected " << pair.second
              << ", got " << actual << ".";
          return fail(CheckResult::FAIL_FILL_CONST, msg.str());
        }
      }
      for (const auto &pair: collector.counts) {
        if (reqs.constCounts.find(pair.first) == reqs.constCounts.end()) {
          std::ostringstream msg;
          msg << "Off-budget constant in a FILL_CONST position: '" << pair.first
              << "' (count: " << pair.second << ").";
          return fail(CheckResult::FAIL_FILL_CONST, msg.str());
        }
      }
    }

    // Each declared intrinsic must be called somewhere (structural gate; not
    // a separate stage since re-masking already prevents removing call sites,
    // so this is effectively unreachable for well-formed puzzles).
    CallCollector callCollector;
    for (const auto &f: prog.funs)
      callCollector.collect(f);
    for (const auto &in: prog.intrinsics) {
      if (callCollector.calledFunctions.find(in.name.name) == callCollector.calledFunctions.end()) {
        std::ostringstream msg;
        msg << "Declared intrinsic '" << in.name.name << "' is never called.";
        return fail(CheckResult::FAIL_REMASKING, msg.str());
      }
    }

    std::cout << "[PASS] Solution is valid!\n";
  } catch (const std::exception &e) {
    std::cerr << "[FAIL_PARSE] " << e.what() << "\n";
    return 1;
  }

  return 0;
}
