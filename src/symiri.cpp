#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "analysis/cfg.hpp"
#include "cxxopts.hpp"
#include "error.hpp"
#include "frontend/link_resolver.hpp"
#include "frontend/pipeline.hpp"
#include "interp/interpreter.hpp"

int main(int argc, char **argv) {
  using namespace refractir;

  cxxopts::Options options("symiri", "RefractIR Reference Interpreter");

  // clang-format off
  options.add_options()
    ("input", "Input .sir file", cxxopts::value<std::string>())
    ("args", "Argument values for the entry function's parameters", cxxopts::value<std::vector<std::string>>())
    ("main", "Entry function to execute", cxxopts::value<std::string>()->default_value("@main"))
    ("sym", "Bind a symbol (name=value)", cxxopts::value<std::vector<std::string>>())
    ("check", "Check semantics only (do not execute)", cxxopts::value<bool>()->default_value("false"))
    ("dump-trace", "Dump executed blocks and variable updates", cxxopts::value<bool>()->default_value("false"))
    ("max-bbl-steps", "Abort after entering this many basic blocks (0 = unlimited); bounds a possibly non-terminating program", cxxopts::value<uint64_t>()->default_value("0"))
    ("w", "Inhibit all warning messages", cxxopts::value<bool>()->default_value("false"))
    ("Werror", "Make all warnings into errors", cxxopts::value<bool>()->default_value("false"))
    ("I", "Include path for resolving link-form `decl`s (may repeat)", cxxopts::value<std::vector<std::string>>())
    ("h,help", "Print usage");
  options.parse_positional({"input", "args"});
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (!result.count("input")) {
    std::cerr << "Error: No input file specified." << std::endl;
    std::cerr << options.help() << std::endl;
    return 1;
  }

  std::string inputPath = result["input"].as<std::string>();
  std::string mainFunc = result["main"].as<std::string>();
  Interpreter::SymBindings symBindings;

  if (result.count("sym")) {
    // cxxopts splits each occurrence's value on commas, so a vector
    // lane list like `--sym %?v=1,2,3,4` arrives as ["%?v=1", "2",
    // "3", "4"]. Entries without '=' are comma-continuations of the
    // previous binding — unambiguous, since values are numeric
    // literals and never contain '='.
    std::vector<std::string> binds;
    for (const auto &piece: result["sym"].as<std::vector<std::string>>()) {
      if (piece.find('=') != std::string::npos) {
        binds.push_back(piece);
      } else if (!binds.empty()) {
        binds.back() += "," + piece;
      } else {
        std::cerr << "Error: Invalid symbol binding format (expected name=value): " << piece
                  << "\n";
        return 1;
      }
    }
    for (const auto &bind: binds) {
      size_t eq = bind.find('=');
      std::string name = bind.substr(0, eq);
      std::string valStr = bind.substr(eq + 1);
      // A single scalar, or a comma-joined lane list for vector syms
      // (one value per lane; a single value splats).
      std::vector<Interpreter::SymScalar> vals;
      try {
        size_t pos = 0;
        while (true) {
          size_t comma = valStr.find(',', pos);
          vals.push_back(parseNumberLiteral(
              valStr.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos)
          ));
          if (comma == std::string::npos)
            break;
          pos = comma + 1;
        }
      } catch (...) {
        std::cerr << "Error: Invalid number value for symbol " << name << ": " << valStr << "\n";
        return 1;
      }
      symBindings[name] = std::move(vals);
    }
  }

  std::ifstream ifs(inputPath);
  if (!ifs) {
    std::cerr << "Error: Could not open file " << inputPath << "\n";
    return 1;
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  std::string src = ss.str();

  try {
    // 1. Frontend: Lex & Parse
    Program prog = parseSource(src);

    // 1b. Load -I libraries and resolve link-form `decl`s.
    std::vector<Program> libs;
    if (result.count("I")) {
      libs = loadIncludeDirs(result["I"].as<std::vector<std::string>>());
    }
    resolveLinkDecls(prog, libs);

    // 2. Analysis
    CheckOptions checkOpts;
    checkOpts.warningsAreErrors = result["Werror"].as<bool>();

    bool werror = checkOpts.warningsAreErrors;
    bool nowarn = result["w"].as<bool>();

    DiagBag diags;
    if (!checkProgram(prog, diags, checkOpts)) {
      std::cerr << "Errors:\n";
      for (const auto &d: diags.diags) {
        if (d.level == DiagLevel::Error || (werror && d.level == DiagLevel::Warning)) {
          printMessage(std::cerr, src, d.span, d.message, d.level);
        }
      }
      return ExitCode::StaticError;
    }

    if (!nowarn) {
      for (const auto &d: diags.diags) {
        if (d.level == DiagLevel::Warning) {
          printMessage(std::cout, src, d.span, d.message, d.level);
        }
      }
    }

    if (result["check"].as<bool>()) {
      return 0;
    }

    // 4. Interpret
    std::vector<std::string> paramArgs;
    if (result.count("args"))
      paramArgs = result["args"].as<std::vector<std::string>>();
    Interpreter interp(prog);
    interp.setMaxBlockSteps(result["max-bbl-steps"].as<uint64_t>());
    interp.run(mainFunc, symBindings, paramArgs, result["dump-trace"].as<bool>());

  } catch (const StepLimitError &e) {
    std::cerr << "Step limit exceeded: " << e.what() << "\n";
    return ExitCode::Error;
  } catch (const UndefinedBehaviorError &e) {
    std::cerr << e.what() << "\n";
    return ExitCode::UndefinedBehavior;
  } catch (const RequireViolationError &e) {
    std::cerr << "Requirement failed: " << e.what() << "\n";
    return ExitCode::RequireViolation;
  } catch (const LexError &e) {
    printMessage(std::cerr, src, e.span, e.what(), DiagLevel::Error);
    return ExitCode::LexError;
  } catch (const ParseError &e) {
    printMessage(std::cerr, src, e.span, e.what(), DiagLevel::Error);
    return ExitCode::ParseError;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return ExitCode::Error;
  }

  return 0;
}
