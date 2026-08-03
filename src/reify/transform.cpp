#include "reify/transform.hpp"

namespace refractir::reify {

  TransformReport TransformPipeline::run(Program &prog, TransformContext &ctx) {
    TransformReport total;
    for (auto &t: transforms_) {
      TransformReport r = t->apply(prog, ctx);
      total.sites += r.sites;
      if (!r.ok) {
        total.ok = false;
        total.message = std::string(t->name()) + ": " + r.message;
        return total;
      }
      // A transform may summarize what it did, not only why it failed, so a
      // successful note is carried out rather than dropped.
      if (!r.message.empty()) {
        if (!total.message.empty())
          total.message += "; ";
        total.message += r.message;
      }
    }
    return total;
  }

} // namespace refractir::reify
