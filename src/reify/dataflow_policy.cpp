#include "reify/dataflow_policy.hpp"

#include <algorithm>
#include <limits>

#include "ast/build.hpp"
#include "reify/hyperparameters.hpp"

namespace refractir::reify {

  namespace {

    // The values an `iN` literal can carry. A bias outside it has no spelling
    // at that width, which is what makes this a limit on the construction
    // rather than a guard against overflow.
    struct SignedRange {
      std::int64_t lo;
      std::int64_t hi;
    };

    [[nodiscard]] SignedRange signedRange(std::uint32_t bits) {
      if (bits == 0 || bits >= 64)
        return {std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max()};
      const std::int64_t hi = (std::int64_t{1} << (bits - 1)) - 1;
      return {-hi - 1, hi};
    }

    // Variables of `bits` width that hold one value across every visit. A
    // policy leaning on a variable that moves has nothing to pin to, and a
    // variable missing from a visit is not live there at all.
    [[nodiscard]] std::vector<Pin> stablePins(const DataflowSite &site, std::uint32_t bits) {
      if (site.visits.empty())
        return {};
      std::vector<Pin> stable;
      for (const auto &pin: site.visits.front()) {
        if (pin.bits != bits)
          continue;
        const bool everywhere = std::all_of(
            site.visits.begin() + 1, site.visits.end(), [&](const std::vector<Pin> &visit) {
              return std::any_of(visit.begin(), visit.end(), [&](const Pin &other) {
                return other.name == pin.name && other.value == pin.value;
              });
            }
        );
        if (everywhere)
          stable.push_back(pin);
      }
      return stable;
    }

    // `%v + bias`, or bare `%v` when the variable already holds the target.
    [[nodiscard]] Expr biasExpr(const Pin &pin, std::int64_t bias) {
      Expr e = buildExpr(buildLocalAtom(pin.name));
      if (bias > 0)
        appendTail(e, AddOp::Plus, buildIntAtom(bias));
      else if (bias < 0)
        appendTail(e, AddOp::Minus, buildIntAtom(-bias));
      return e;
    }

    class BaselinePolicy : public DataflowPolicy {
    public:
      const char *name() const override { return "literal-or-bias"; }

      std::optional<DataflowResult> build(
          const DataflowSite &site, std::uint32_t bits, std::int64_t target, std::mt19937 &rng
      ) override {
        DataflowResult result;
        result.value = biasedOrLiteral(site, bits, target, rng);
        return result;
      }

    private:
      // A pinned variable carried to the target by a bias, falling back to the
      // target spelled out. The fallback is why this policy never declines.
      //
      // The coin keeps plain literals in the mix. An argument the compiler can
      // read straight off is the case every other construction is measured
      // against, so a generator that never emits one has lost its control.
      [[nodiscard]] static Expr biasedOrLiteral(
          const DataflowSite &site, std::uint32_t bits, std::int64_t target, std::mt19937 &rng
      ) {
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        if (coin(rng) >= rylink::hp::kPVarBiasArg)
          return buildIntExpr(target);
        std::vector<Pin> pins = stablePins(site, bits);
        const SignedRange range = signedRange(bits);
        std::shuffle(pins.begin(), pins.end(), rng);
        for (const Pin &pin: pins) {
          std::int64_t bias = 0;
          if (__builtin_sub_overflow(target, pin.value, &bias))
            continue;
          if (bias < range.lo || bias > range.hi)
            continue;
          return biasExpr(pin, bias);
        }
        return buildIntExpr(target);
      }
    };

  } // namespace

  std::unique_ptr<DataflowPolicy> makeBaselinePolicy() {
    return std::make_unique<BaselinePolicy>();
  }

  DataflowSite pinsAtBlock(const StateProfile &profile, const std::string &blockLabel) {
    DataflowSite site;
    for (const StatePoint &point: profile.trace) {
      if (point.block != blockLabel || point.instr != -1)
        continue;
      std::vector<Pin> pins;
      for (const auto &[name, value]: point.vars) {
        if (value.kind == StateValue::Kind::Int)
          pins.push_back(Pin{name, value.intVal, value.bits});
      }
      site.visits.push_back(std::move(pins));
    }
    return site;
  }

  std::optional<DataflowResult> buildArgument(
      const std::vector<std::unique_ptr<DataflowPolicy>> &policies, const DataflowSite &site,
      std::uint32_t bits, std::int64_t target, std::mt19937 &rng
  ) {
    std::vector<DataflowPolicy *> order;
    order.reserve(policies.size());
    for (const auto &policy: policies)
      order.push_back(policy.get());
    std::shuffle(order.begin(), order.end(), rng);
    for (DataflowPolicy *policy: order) {
      if (auto result = policy->build(site, bits, target, rng))
        return result;
    }
    return std::nullopt;
  }

} // namespace refractir::reify
