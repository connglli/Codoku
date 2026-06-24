#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
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

// --- Rysmith loop execution ---
/**
 * @brief Runs rysmith in a loop with loop constraints until a concrete
 * test case is successfully generated and output.
 * @param rysmithPath Path to the rysmith executable.
 * @param seed Master seed; incremented on each retry until generation succeeds.
 * @return String path to the successfully generated concrete .sir file.
 */
std::string runRysmithAndGetSir(
    const std::string &rysmithPath, uint32_t seed, const fs::path &tmpDir, uint32_t minLoopIter,
    uint32_t nBbls, uint32_t nStmts
) {
  fs::create_directories(tmpDir);

  int attempt = 0;
  while (attempt < 100) {
    // Clean temporary directory before each run
    for (auto &entry: fs::directory_iterator(tmpDir)) {
      fs::remove_all(entry.path());
    }

    std::string cmd = rysmithPath + " -n 1" + " --min-loop-iter " + std::to_string(minLoopIter) +
                      " --n-bbls " + std::to_string(nBbls) + " --n-stmts " +
                      std::to_string(nStmts) + " --emit-main -o " + tmpDir.string();
    cmd += " --seed " + std::to_string(seed);
    cmd += " > /dev/null 2>&1";

    int ret = std::system(cmd.c_str());
    (void) ret; // Suppress unused-result warning

    for (auto &entry: fs::directory_iterator(tmpDir)) {
      if (entry.path().extension() == ".sir" &&
          entry.path().filename().string().find("_sym") == std::string::npos) {
        return entry.path().string();
      }
    }

    seed++;
    attempt++;
  }
  throw std::runtime_error("Failed to generate a test case via rysmith after 100 attempts");
}

/**
 * @brief Parses the output stream from rysmith to extract the execution path comment.
 */
std::string extractPath(const std::string &src) {
  std::istringstream iss(src);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.rfind("// PATH:", 0) == 0) {
      return line.substr(8); // skip "// PATH:"
    }
  }
  return "";
}

// --- Puzzle header template ---
//
// The instruction banner that precedes every puzzle. Keep it here as one
// editable block rather than scattered `<<` calls. Placeholders of the form
// `{{NAME}}` are filled in by renderHeader():
//
//   {{LEAF_NAME}}   leaf function name (e.g. @func_ab12_3)
//   {{CFG}}         CFG adjacency listing (multi-line, each line "//  a -> b")
//   {{PATH}}        execution path "entry -> b0 -> ... -> exit"
//   {{FILL_CONST}}  FILL_CONST budget, one "//@ FILL_CONST: <value> <count>" per line
//
// Lines beginning with `//@` are *machine-readable markers* parsed by rypuzchk;
// everything else is for the human/agent solver. Keep the `//@` markers stable.
static const char *PUZZLE_HEADER_TEMPLATE = R"TPL(//
//
// {{LEAF_NAME}}() is a function of the following CFG:
//
{{CFG}}//
// ------------------------------------------------
// Task
// ------------------------------------------------
//
// Replace all occurrences of FILL_XXX with appropriate code to make
// the function return the expected value for the test case in @main
// following the below execution path:
//
//@ PATH: {{PATH}}
//
// ------------------------------------------------
// Validation
// ------------------------------------------------
//
// Use the following command to verify your solution:
//
//   ./rypuzchk [this_puzzle_file].sir [your_solution].sir
//
// ------------------------------------------------
// Materials
// ------------------------------------------------
//
// * Grammar: ./references/SPEC.md.
// * Good Examples: ./references/examples/
// * Good and Bad Examples:
//   * Each example is with "// EXPECT: PASS" or "// EXPECT: FAIL" in the header comment.
//   * "// EXPECT: PASS" means the code is valid and should be accepted by the interpreter.
//   * "// EXPECT: FAIL" means the code is invalid and should be rejected by the interpreter.
//
// Read the grammar and good examples to understand the syntax and semantics of the language.
// Use the bad examples to understand common mistakes and pitfalls.
//
// ------------------------------------------------
// General Requirements
// ------------------------------------------------
//
// 0. Each FILL_XXX mark must be filled out with a corresponding element.
// 1. Each intrinsic function is used at least once.
// 2. You have access to all common command line tools and the toolchain in ./:
//    - ./symiri:     The interpreter
//    - ./symirc:     The compiler
//    - ./symirsolve: The solver
//    - ./z3, ./cvc5, ./bitwuzla: SMT solvers
// 3. Do NOT change any code except for the FILL_XXX marks.
// 4. Do NOT introduce any new code, variables, or basic blocks.
// 5. Do NOT access the internet.
//
// ------------------------------------------------
// Requirements for FILL_CONST
// ------------------------------------------------
//
// The lines below list every constant the FILL_CONST marks must carry, as
// "<value> <count>" pairs. Across your whole solution each <value> must appear
// in FILL_CONST positions exactly <count> times -- no more, no fewer -- and no
// other constant may appear in any FILL_CONST position. Constants already shown
// in the fixed (entry/exit) code do not count toward this budget.
//
{{FILL_CONST}}//
//

)TPL";

/**
 * @brief Replaces every occurrence of `key` in `s` with `val` (in place).
 */
void replaceAll(std::string &s, const std::string &key, const std::string &val) {
  size_t pos = 0;
  while ((pos = s.find(key, pos)) != std::string::npos) {
    s.replace(pos, key.size(), val);
    pos += val.size();
  }
}

/**
 * @brief Trims leading/trailing ASCII whitespace from a string.
 */
std::string trim(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos)
    return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

/**
 * @brief Renders the puzzle header banner from its template and dynamic parts.
 */
std::string renderHeader(
    const std::string &leafName, const std::string &cfgStr, const std::string &pathStr,
    const std::string &fillConstStr
) {
  std::string out = PUZZLE_HEADER_TEMPLATE;
  replaceAll(out, "{{LEAF_NAME}}", leafName);
  replaceAll(out, "{{CFG}}", cfgStr);
  replaceAll(out, "{{PATH}}", pathStr.empty() ? "[unknown]" : pathStr);
  replaceAll(out, "{{FILL_CONST}}", fillConstStr);
  return out;
}

int main(int argc, char **argv) {
  cxxopts::Options options("rypuzmk", "RefractIR Puzzle Creator");
  options.add_options()
      ("input", "Optional concrete .sir file to mask", cxxopts::value<std::string>())
      ("o,output", "Output puzzle file path", cxxopts::value<std::string>())
      // Difficulty-control options
      ("L,min-loop-iter", "Minimum loop iterations constraint for rysmith", cxxopts::value<uint32_t>()->default_value("2"))
      ("B,n-bbls", "Number of basic blocks for rysmith", cxxopts::value<uint32_t>()->default_value("5"))
      ("S,n-stmts", "Number of statements per block on path for rysmith", cxxopts::value<uint32_t>()->default_value("3"))
      // Other options
      ("keep-ground-truth", "Save the unmasked ground-truth concrete .sir file as <puzzle>.gt.sir", cxxopts::value<bool>()->default_value("false"))
      ("rysmith", "Path to rysmith binary", cxxopts::value<std::string>()->default_value("./rysmith"))
      ("s,seed", "Seed for rysmith", cxxopts::value<uint32_t>())
      ("h,help", "Print usage");

  options.parse_positional({"input"});
  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  std::string inputPath;
  bool deleteTemp = false;
  fs::path tmpDir;

  try {
    if (result.count("input")) {
      inputPath = result["input"].as<std::string>();
    } else {
      std::string rysmithPath = result["rysmith"].as<std::string>();
      std::optional<uint32_t> seed;
      if (result.count("seed"))
        seed = result["seed"].as<uint32_t>();

      uint32_t minLoopIter = result["min-loop-iter"].as<uint32_t>();
      uint32_t nBbls = result["n-bbls"].as<uint32_t>();
      uint32_t nStmts = result["n-stmts"].as<uint32_t>();

      std::random_device rd;
      std::mt19937 gen(rd());
      uint32_t runSeed = seed.value_or(gen());
      tmpDir = fs::temp_directory_path() / ("rypuz_tmp_" + std::to_string(runSeed));

      inputPath = runRysmithAndGetSir(rysmithPath, runSeed, tmpDir, minLoopIter, nBbls, nStmts);
      deleteTemp = true;
    }

    // Read the generated or input RefractIR source code file
    std::string src = readFile(inputPath);

    // Parse the RefractIR source code to AST
    Program prog = parseProgramFromString(src);

    const FunDecl *leaf = findLeafFunction(prog);
    if (!leaf) {
      std::cerr << "Error: Could not find leaf function in " << inputPath << "\n";
      return 1;
    }

    // Masking is keyed off block position (entry = first, exit = last, body in
    // between). Reject any leaf whose shape breaks that assumption rather than
    // silently emitting a degenerate puzzle.
    if (std::string why = validateLeafShape(*leaf); !why.empty()) {
      std::cerr << "Error: malformed leaf function in " << inputPath << ": " << why << "\n";
      return 1;
    }

    // Build CFG representation
    DiagBag diags;
    CFG cfg = CFG::build(*leaf, diags);
    if (diags.hasErrors()) {
      std::cerr << "Error: CFG building failed for " << leaf->name.name << "\n";
      return 1;
    }

    // Collect the exact FILL_CONST budget (every constant the masking hides,
    // counted only at masked positions).
    MaskedConstantCollector collector;
    collector.collect(*leaf);

    // Extract path comment
    std::string pathComment = extractPath(src);

    // Build the CFG adjacency listing for the header (one "//  a -> b ..." line per block).
    std::ostringstream cfgStr;
    for (size_t i = 0; i < cfg.blocks.size(); ++i) {
      std::string labelName = cfg.blocks[i];
      if (labelName.rfind("^", 0) == 0)
        labelName = labelName.substr(1);
      cfgStr << "//   " << labelName;
      if (!cfg.succ[i].empty()) {
        cfgStr << " ->";
        for (auto sIdx: cfg.succ[i]) {
          std::string succName = cfg.blocks[sIdx];
          if (succName.rfind("^", 0) == 0)
            succName = succName.substr(1);
          cfgStr << " " << succName;
        }
      }
      cfgStr << "\n";
    }

    // Build the FILL_CONST budget as machine-readable "//@ FILL_CONST: <value> <count>" lines.
    std::vector<std::pair<std::string, int>> sortedConsts(
        collector.counts.begin(), collector.counts.end()
    );
    std::sort(sortedConsts.begin(), sortedConsts.end());
    std::ostringstream fillConstStr;
    for (const auto &p: sortedConsts) {
      fillConstStr << "//@ FILL_CONST: " << p.first << " " << p.second << "\n";
    }

    std::string header =
        renderHeader(leaf->name.name, cfgStr.str(), trim(pathComment), fillConstStr.str());

    // Render the masked puzzle body and the unmasked ground truth from the same
    // program, so we can both self-check and (optionally) save the ground truth.
    std::ostringstream puzzleBody;
    SIRMaskedPrinter(puzzleBody, *leaf, /*enableMasking=*/true).print(prog);
    std::ostringstream gtBody;
    SIRMaskedPrinter(gtBody, *leaf, /*enableMasking=*/false).print(prog);

    // Self-check: the ground truth is the canonical solution, so re-masking it
    // must reproduce the puzzle body. This catches printer/parser round-trip
    // asymmetries before a broken (unsolvable) puzzle is shipped.
    {
      Program gtProg = parseProgramFromString(gtBody.str());
      const FunDecl *gtLeaf = findLeafFunction(gtProg);
      if (!gtLeaf) {
        std::cerr << "Error: self-check failed: regenerated ground truth has no leaf function.\n";
        return 1;
      }
      std::ostringstream remasked;
      SIRMaskedPrinter(remasked, *gtLeaf, /*enableMasking=*/true).print(gtProg);
      if (remasked.str() != puzzleBody.str()) {
        std::cerr << "Error: self-check failed: ground truth does not re-mask to the puzzle "
                     "(printer/parser round-trip mismatch).\n";
        return 1;
      }
    }

    // Save the unmasked ground-truth program if requested.
    if (result["keep-ground-truth"].as<bool>()) {
      if (!result.count("output")) {
        std::cerr << "Error: --keep-ground-truth requires --output to be specified.\n";
        return 1;
      }
      std::string outputPath = result["output"].as<std::string>();
      std::string gtPath = std::filesystem::path(outputPath).replace_extension(".gt.sir").string();
      std::ofstream gtFile(gtPath);
      if (!gtFile) {
        std::cerr << "Error: Could not open ground-truth file for writing: " << gtPath << "\n";
        return 1;
      }
      gtFile << gtBody.str();
    }

    std::ostream *out = &std::cout;
    std::ofstream fileOut;
    if (result.count("output")) {
      fileOut.open(result["output"].as<std::string>());
      if (!fileOut) {
        std::cerr << "Error: Could not open output file " << result["output"].as<std::string>()
                  << "\n";
        return 1;
      }
      out = &fileOut;
    }

    *out << header << puzzleBody.str();

    if (deleteTemp) {
      std::error_code ec;
      fs::remove_all(tmpDir, ec);
    }

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
