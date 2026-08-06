#include "reify/antiopt_transform.hpp"

#include <string>
#include <vector>

#include "reify/antiopt.hpp"
#include "reify/hyperparameters.hpp"
#include "reify/twin_mini.hpp"

namespace refractir::reify {

  namespace {

    class AntiOptTransform : public Transform {
    public:
      std::string_view name() const override { return "AntiOptTransform"; }

      TransformReport apply(Program &prog, TransformContext &ctx) override {
        TransformReport rep;
        const StructMap structs = TypeUtils::buildStructTable(prog);
        for (auto &fn: prog.funs) {
          // One allocator per function: its names are declared into that
          // function's lets, and they should read like the locals around them
          // rather than restarting per block.
          NameAllocator names(kAntiOptLocalPrefix);
          for (auto &block: fn.blocks) {
            // A block of a concrete program assumes no branch conditions the
            // way a twin body does — control still runs through its own
            // terminator — so there is nothing to shift and nothing to check.
            std::vector<PathCheck> noChecks;
            AntiOptContext actx{fn, structs, noChecks, names, fn.lets, ctx.rng, nullptr};
            // No acceptance predicate: nothing here can judge a rewritten
            // block, so the engine offers only the rules that need no
            // judgement. See antiopt_transform.hpp.
            const AntiOptReport r = antiOptimize(
                block.instrs, noChecks, actx, {}, rylink::hp::kAntiOptAttemptsPerBlock
            );
            rep.sites += r.applied;
          }
        }
        if (ctx.verbose)
          *ctx.verbose << "  antiopt: " << rep.sites << " rewrites (trap-free only)\n";
        return rep;
      }
    };

  } // namespace

  std::unique_ptr<Transform> makeAntiOptTransform() { return std::make_unique<AntiOptTransform>(); }

} // namespace refractir::reify
