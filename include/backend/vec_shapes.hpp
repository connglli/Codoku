#pragma once

#include <vector>
#include "ast/ast.hpp"

namespace refractir {

  // Walk the program collecting every (N, T) vector shape
  // used, so a vec-lowering strategy can emit its per-shape preamble
  // (C typedefs / struct decls, python helper classes). Shapes may
  // repeat; strategies deduplicate. Shared by the C and python
  // backends (lifted from c_backend.cpp).
  std::vector<VecType> collectVecShapes(const Program &prog);

} // namespace refractir
