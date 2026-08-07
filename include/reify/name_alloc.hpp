#pragma once

// Fresh names for a body being rewritten.
//
// Temporaries are needed constantly: RefractIR admits at most one binary
// operator per atom, and the right operand of `* / % & | ^ << >> >>>` must be
// an lvalue — so `%a << 3` has to become `%k = 3; %a << %k`. A pass that
// builds anything longer than one operation allocates its way through this.
//
// Literal cells are pooled by value, so a body does not accumulate a dozen
// names for the same constant.
//
// A name is fresh against two lists: the declarations being built and, when a
// caller supplies it, the ones the function already carries. Passing only the
// first is right when a pass declares straight into the function; a pass that
// collects its declarations aside has to name the function too, or two of them
// will mint the same name and the program will not parse.

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "ast/ast.hpp"

namespace refractir::reify {

  class NameAllocator {
  public:
    // `taken`, when given, is borrowed and must outlive the allocator.
    explicit NameAllocator(std::string prefix, const std::vector<LetDecl> *taken = nullptr) :
        prefix_(std::move(prefix)), taken_(taken) {}

    // A fresh mutable local of `type`, declared into `lets`.
    std::string fresh(const TypePtr &type, std::vector<LetDecl> &lets);

    // A local holding `value` at `type`, reused when one already exists.
    std::string literal(std::int64_t value, const TypePtr &type, std::vector<LetDecl> &lets);

    const std::string &prefix() const { return prefix_; }

  private:
    std::string prefix_;
    const std::vector<LetDecl> *taken_ = nullptr;
    std::size_t next_ = 0;
    std::vector<std::tuple<std::int64_t, std::string, std::string>> pool_; // value, type key, name
  };

} // namespace refractir::reify
