#pragma once

// The solved-program header.
//
// A concretized `.sir` file opens with a structured comment recording the
// values the solver chose:
//
//     // SOLVED: %pa0=3, %pa1=-7, ret=42
//
// A file with more than one example repeats the line, one per example; the
// first is the solved input. Parameter names appear verbatim in the function
// body, so a consumer reads the line to re-run the program on one example,
// without solving it again.
//
// Writer and reader live together because the format is their only agreement,
// and it is a text boundary floats cross: values go out through formatDouble
// and come back through parseFloatLiteral, so a subnormal or a negative zero
// survives the round trip (see docs/float.md §9).
//
// It lives with the solver because a model value is what it carries. A reader
// needs the format, not the symbolic executor, so nothing here calls into one.

#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "solver/solver.hpp"

namespace refractir {

  // One model value as the header carries it.
  [[nodiscard]] std::string formatModelValue(const SymbolicExecutor::Result::ModelVal &v);

  // Write one header line from ordered (name, value-text) pairs, or nothing
  // when there is nothing to record. Declaration order makes the line
  // byte-identical across runs. `retText` is already formatted (a caller
  // may hold the return value as text, not a model value).
  void writeSolvedHeader(
      std::ostream &out, const std::vector<std::pair<std::string, std::string>> &paramValues,
      const std::string &retText
  );

  // Same line from a param model; values go out via formatModelValue.
  void writeSolvedHeader(
      std::ostream &out,
      const std::unordered_map<std::string, SymbolicExecutor::Result::ModelVal> &paramModel,
      const std::string &retText
  );

  // Read the header back as name -> value text, keeping `ret` under that name.
  // Only the first `// SOLVED:` line is read, so a multi-example file reads as
  // its solved input. Empty when `src` carries no header.
  [[nodiscard]] std::unordered_map<std::string, std::string>
  parseSolvedHeader(std::string_view src);

} // namespace refractir
