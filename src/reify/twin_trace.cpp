#include "reify/twin_trace.hpp"

#include <unordered_set>
#include <utility>

#include "ast/clone.hpp"

namespace refractir::reify {

  std::optional<TraceBody> flattenTrace(
      const std::vector<std::string> &executed, const std::string &exitLabel,
      const std::unordered_map<std::string, const Block *> &byLabel, std::string *why
  ) {
    auto reject = [&](std::string r) -> std::optional<TraceBody> {
      if (why)
        *why = std::move(r);
      return std::nullopt;
    };

    TraceBody out;
    out.exitLabel = exitLabel;
    out.steps = executed.size();

    std::unordered_set<std::string> distinct;
    for (const auto &label: executed) {
      auto it = byLabel.find(label);
      if (it == byLabel.end())
        return reject("block not found: " + label);
      distinct.insert(label);

      // The run reached the next block entry, so this block ran to its
      // terminator: every instruction it holds executed, in this order.
      for (const auto &ins: it->second->instrs)
        out.stmts.push_back(cloneInstr(ins));
    }
    out.blocks = distinct.size();

    return out;
  }

} // namespace refractir::reify
