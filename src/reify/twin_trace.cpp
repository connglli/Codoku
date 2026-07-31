#include "reify/twin_trace.hpp"

#include <unordered_set>
#include <utility>
#include <variant>

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
    for (std::size_t k = 0; k < executed.size(); ++k) {
      const std::string &label = executed[k];
      auto it = byLabel.find(label);
      if (it == byLabel.end())
        return reject("block not found: " + label);
      distinct.insert(label);

      // The run reached the next block entry, so this block ran to its
      // terminator: every instruction it holds executed, in this order.
      for (const auto &ins: it->second->instrs)
        out.stmts.push_back(cloneInstr(ins));

      // Where the run went next — the following step, or out of the region.
      const std::string &next = k + 1 < executed.size() ? executed[k + 1] : exitLabel;
      const auto *br = std::get_if<BrTerm>(&it->second->term);
      if (!br)
        return reject("trace continues past a non-branch terminator in " + label);
      if (!br->isConditional)
        continue;
      PathCheck chk;
      chk.afterStmt = out.stmts.size();
      chk.cond = cloneCond(*br->cond);
      if (next == br->thenLabel.name)
        chk.taken = true;
      else if (next == br->elseLabel.name)
        chk.taken = false;
      else
        return reject("branch in " + label + " does not lead to " + next);
      out.checks.push_back(std::move(chk));
    }
    out.blocks = distinct.size();

    return out;
  }

} // namespace refractir::reify
