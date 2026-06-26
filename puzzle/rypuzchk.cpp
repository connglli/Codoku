#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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

#include "puzzle_common.hpp"

// --- Puzzle requirements (parsed from the puzzle header's machine markers) ---
//
// rypuzmk emits two kinds of machine-readable marker lines in the header:
//
//   //@ PATH: entry -> b0 -> ... -> exit
//   //@ FILL_CONST: <value> <count>
//
// Parsing only these markers (rather than scraping prose) keeps the validator
// robust to header wording changes.
struct PuzzleRequirements {
  std::vector<std::string> expectedPath;
  std::unordered_map<std::string, int> constCounts;
};

PuzzleRequirements parsePuzzleRequirements(const std::string &puzzleText) {
  PuzzleRequirements reqs;
  const std::string PATH_MARK = "//@ PATH:";
  const std::string CONST_MARK = "//@ FILL_CONST:";

  std::istringstream iss(puzzleText);
  std::string line;
  while (std::getline(iss, line)) {
    if (size_t p = line.find(PATH_MARK); p != std::string::npos) {
      // Strip whitespace, then split the remainder on "->".
      std::string cleaned;
      for (char c: line.substr(p + PATH_MARK.size())) {
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
      if (ls >> val >> cnt)
        reqs.constCounts[val] = static_cast<int>(cnt);
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
 * and the executed basic-block trace. A single run gives us correctness (exit
 * code: the embedded check_chksum aborts on mismatch) and the path together.
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
  options
      .add_options()("puzzle", "Puzzle file path", cxxopts::value<std::string>())("solution", "Solution file path", cxxopts::value<std::string>())("symiri", "Path to symiri binary (default: 'symiri' next to this binary)", cxxopts::value<std::string>())(
          "h,help", "Print usage"
      );

  options.parse_positional({"puzzle", "solution"});
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

  try {
    // 1. Read and parse puzzle requirements (machine markers in the header).
    std::string puzzleText = readFile(puzzlePath);
    PuzzleRequirements reqs = parsePuzzleRequirements(puzzleText);
    if (reqs.expectedPath.empty()) {
      std::cerr << "[FAIL] Puzzle is missing a '//@ PATH:' marker; cannot validate.\n";
      return 1;
    }

    // 2. Read and parse the solution.
    std::string solutionText = readFile(solutionPath);
    Program prog = parseProgramFromString(solutionText);

    const FunDecl *leaf = findLeafFunction(prog);
    if (!leaf) {
      std::cerr << "[FAIL] Could not find leaf function in solution.\n";
      return 1;
    }

    // 3. Execution + trace in a single symiri run. The embedded check_chksum
    //    aborts (non-zero exit) on a wrong result, so exit 0 == correct value.
    ExecResult exec = runSymiri(symiriPath, solutionPath);
    if (exec.exitCode != 0) {
      std::cerr << "[FAIL] Solution fails to compile or execute under symiri (exit code "
                << exec.exitCode << ").\n";
      return 1;
    }

    // 4. Execution-path verification. The trace is [@main's blocks..., leaf
    //    blocks...]; @main is a fixed single-block wrapper that calls the leaf
    //    and never re-enters a block afterwards, so drop exactly @main's blocks.
    const FunDecl *mainFn = findMainFunction(prog);
    size_t mainBlocks = mainFn ? mainFn->blocks.size() : 1;
    if (exec.blockPath.size() <= mainBlocks) {
      std::cerr << "[FAIL] Solution trace is empty or too short to contain the leaf path.\n";
      return 1;
    }
    std::vector<std::string> leafPath(exec.blockPath.begin() + mainBlocks, exec.blockPath.end());

    if (leafPath != reqs.expectedPath) {
      std::cerr << "[FAIL] Execution path mismatch.\n  Expected: ";
      for (size_t i = 0; i < reqs.expectedPath.size(); ++i)
        std::cerr << reqs.expectedPath[i] << (i + 1 < reqs.expectedPath.size() ? " -> " : "");
      std::cerr << "\n  Actual:   ";
      for (size_t i = 0; i < leafPath.size(); ++i)
        std::cerr << leafPath[i] << (i + 1 < leafPath.size() ? " -> " : "");
      std::cerr << "\n";
      return 1;
    }

    // Infer the selective masking set from the puzzle body. The puzzle file
    // contains FILL_XXX tokens (not parseable as RefractIR), so we derive
    // which positions were masked by comparing sentinel-annotated renders of
    // the solution (full-masked vs. plain) against the stripped puzzle text.
    // inferMaskSetFromPuzzle() returns the indices that are masked: the empty
    // set when nothing is masked, and the full index range [0, N) when every
    // position is masked.
    std::unordered_set<int> maskSet = inferMaskSetFromPuzzle(*leaf, prog, puzzleText);
    int totalPositions = countMaskableStatements(*leaf);
    // If every position is masked, pass nullptr to MaskedConstantCollector and
    // SIRMaskedPrinter to take their cheaper "mask everything" fast-path (which
    // is also what rypuzmk does for the default --p-mask 1.0). Otherwise hand
    // them the inferred set so only the masked positions are considered.
    const std::unordered_set<int> *maskPtr;
    if (static_cast<int>(maskSet.size()) == totalPositions) {
      maskPtr = nullptr;
    } else {
      maskPtr = &maskSet;
    }

    // 5. Structural integrity (anti-cheating): re-masking the solution must
    //    reproduce the puzzle skeleton exactly (modulo comments/whitespace).
    //
    //    This is the authoritative structural gate and MUST run before the
    //    budget check below. Re-masking emits exactly one chunk per maskable
    //    statement of the *solution*, so a solution whose statement count or
    //    skeleton differs from the puzzle can never reproduce it -- regardless
    //    of how the mask set was inferred. Conversely, once this passes the
    //    solution is structurally identical to the puzzle in every non-FILL
    //    position, which means the inferred maskPtr is exact and the budget
    //    check operates on a trustworthy partition. Running it first also keeps
    //    a structural edit (e.g. an inserted statement, which would otherwise
    //    drift the mask inference) reported as a structural failure rather than
    //    an incidental "FILL_CONST count mismatch".
    std::ostringstream mss;
    SIRMaskedPrinter(mss, *leaf, /*enableMasking=*/true, maskPtr).print(prog);
    if (stripCommentsAndWhitespace(mss.str()) != stripCommentsAndWhitespace(puzzleText)) {
      std::cerr << "[FAIL] Solution structural integrity check failed.\n"
                << "  You may have changed code outside the FILL_XXX marks, or introduced\n"
                << "  unauthorized variables / statements / basic blocks.\n";
      return 1;
    }

    // 6. FILL_CONST budget: the exact multiset of constants in masked positions
    //    must equal the puzzle's budget (no missing, no extra, no off-budget).
    //    Reaching here means step 5 passed, so maskPtr partitions the solution
    //    exactly as the puzzle does.
    MaskedConstantCollector collector;
    collector.selectiveMask = maskPtr;
    collector.collect(*leaf);

    for (const auto &pair: reqs.constCounts) {
      int actual = collector.counts.count(pair.first) ? collector.counts[pair.first] : 0;
      if (actual != pair.second) {
        std::cerr << "[FAIL] FILL_CONST count mismatch for '" << pair.first << "'. Expected "
                  << pair.second << ", got " << actual << ".\n";
        return 1;
      }
    }
    for (const auto &pair: collector.counts) {
      if (reqs.constCounts.find(pair.first) == reqs.constCounts.end()) {
        std::cerr << "[FAIL] Off-budget constant in a FILL_CONST position: '" << pair.first
                  << "' (count: " << pair.second << ").\n";
        return 1;
      }
    }

    // 7. Each declared intrinsic must be called somewhere.
    CallCollector callCollector;
    for (const auto &f: prog.funs)
      callCollector.collect(f);
    for (const auto &in: prog.intrinsics) {
      if (callCollector.calledFunctions.find(in.name.name) == callCollector.calledFunctions.end()) {
        std::cerr << "[FAIL] Declared intrinsic '" << in.name.name << "' is never called.\n";
        return 1;
      }
    }

    std::cout << "[PASS] Solution is valid!\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
