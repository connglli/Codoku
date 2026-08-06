#pragma once

// The solved-program header.
//
// A concretized `.sir` file opens with a structured comment recording the
// values the solver chose:
//
//     // SOLVED: %pa0=3, %pa1=-7, ret=42
//
// Parameter names appear verbatim in the function body — the printer does not
// substitute them — so a consumer reads this line to re-run the program on the
// input it was solved for, without solving it again.
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

#include "solver/solver.hpp"

namespace refractir {

  // One model value as the header carries it.
  [[nodiscard]] std::string formatModelValue(const SymbolicExecutor::Result::ModelVal &v);

  // Write the header, or nothing when there is nothing to record. `retText` is
  // already formatted, since a caller may have the return value as a string
  // rather than as a model value.
  void writeSolvedHeader(
      std::ostream &out,
      const std::unordered_map<std::string, SymbolicExecutor::Result::ModelVal> &paramModel,
      const std::string &retText
  );

  // Read the header back as name -> value text, keeping `ret` under that name.
  // Empty when `src` carries no header.
  [[nodiscard]] std::unordered_map<std::string, std::string>
  parseSolvedHeader(std::string_view src);

} // namespace refractir
