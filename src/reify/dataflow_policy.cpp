#include "reify/dataflow_policy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "analysis/type_utils.hpp"
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

    // The signed `iN` value whose low `bits` bits are `value`'s. Bitwise work
    // is done on 64-bit patterns and has to come back as a literal the
    // argument's own width can carry.
    [[nodiscard]] std::int64_t signExtend(std::int64_t value, std::uint32_t bits) {
      if (bits == 0 || bits >= 64)
        return value;
      const std::uint64_t low =
          static_cast<std::uint64_t>(value) & ((std::uint64_t{1} << bits) - 1);
      const std::uint64_t signBit = std::uint64_t{1} << (bits - 1);
      return static_cast<std::int64_t>((low ^ signBit) - signBit);
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
          const DataflowSite &site, std::uint32_t bits, std::int64_t target, NameAllocator &,
          std::mt19937 &rng
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
          // What reaches the program is the literal, and a negative bias is
          // spelled `%v - |bias|`, so it is the negation that has to fit. The
          // most negative value of a width has no positive counterpart there:
          // at 64 bits negating it is undefined and hands back the same value,
          // which emits a subtraction that overflows at run time.
          if (bias < -range.hi || bias > range.hi)
            continue;
          return biasExpr(pin, bias);
        }
        return buildIntExpr(target);
      }
    };

    // Variables of `bits` width live at every visit. Their values may differ
    // between visits — that is what a construction reading several states has
    // to cope with, and what makes them worth reading.
    [[nodiscard]] std::vector<std::string> liveVars(const DataflowSite &site, std::uint32_t bits) {
      if (site.visits.empty())
        return {};
      std::vector<std::string> live;
      for (const Pin &pin: site.visits.front()) {
        if (pin.bits != bits)
          continue;
        const bool everywhere = std::all_of(
            site.visits.begin() + 1, site.visits.end(), [&](const std::vector<Pin> &visit) {
              return std::any_of(visit.begin(), visit.end(), [&](const Pin &other) {
                return other.name == pin.name;
              });
            }
        );
        if (everywhere)
          live.push_back(pin.name);
      }
      return live;
    }

    [[nodiscard]] std::int64_t valueAt(const std::vector<Pin> &visit, const std::string &name) {
      const auto it = std::find_if(visit.begin(), visit.end(), [&](const Pin &pin) {
        return pin.name == name;
      });
      return it == visit.end() ? 0 : it->value;
    }

    // Draw a non-empty subset, capped so an argument stays readable.
    [[nodiscard]] std::vector<std::string>
    pickSubset(std::vector<std::string> vars, std::size_t cap, std::mt19937 &rng) {
      std::shuffle(vars.begin(), vars.end(), rng);
      std::uniform_int_distribution<std::size_t> howMany(1, std::min(cap, vars.size()));
      vars.resize(howMany(rng));
      return vars;
    }

    // `target ^ (⋀ⱼ (g ^ gⱼ))`, with `g` the XOR of a chosen set of the
    // caller's variables and `gⱼ` the values it takes at each recorded visit.
    //
    // One factor is zero at every visit, so the chain collapses and the target
    // is what remains — at each visit, whatever the variables hold there. That
    // is what lets the set include variables that move between visits: each
    // distinct value they drive `g` to simply contributes another factor.
    //
    // Every operation is `^` or `&`, neither of which can trap at any width, so
    // the construction owes no magnitude argument at all.
    class BitwisePolicy : public DataflowPolicy {
    public:
      const char *name() const override { return "xor-mask"; }

      std::optional<DataflowResult> build(
          const DataflowSite &site, std::uint32_t bits, std::int64_t target, NameAllocator &names,
          std::mt19937 &rng
      ) override {
        const std::vector<std::string> live = liveVars(site, bits);
        if (live.empty())
          return std::nullopt;
        const std::vector<std::string> probe = pickSubset(live, rylink::hp::kMaxProbeVars, rng);

        std::vector<std::int64_t> seen;
        for (const auto &visit: site.visits) {
          std::int64_t g = 0;
          for (const auto &name: probe)
            g ^= valueAt(visit, name);
          seen.push_back(signExtend(g, bits));
        }
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        if (seen.empty() || !maskVaries(seen, bits, rng))
          return std::nullopt;

        DataflowResult result;
        const TypePtr type = buildIntType(static_cast<int>(bits));
        const std::string g = foldProbe(probe, type, names, result);
        const std::string mask = names.fresh(type, result.lets);
        emit(result, mask, buildOpAtom(Coef{IntLit{seen.front(), {}}}, AtomOpKind::Xor, g));
        if (seen.size() > 1) {
          const std::string factor = names.fresh(type, result.lets);
          for (std::size_t i = 1; i < seen.size(); ++i) {
            emit(result, factor, buildOpAtom(Coef{IntLit{seen[i], {}}}, AtomOpKind::Xor, g));
            emit(result, mask, buildBinAtom(mask, AtomOpKind::And, factor));
          }
        }
        result.value = buildExpr(buildOpAtom(Coef{IntLit{target, {}}}, AtomOpKind::Xor, mask));
        return result;
      }

    private:
      // The name holding `g`. A single variable is already the fold, so only a
      // longer probe needs a cell of its own.
      [[nodiscard]] static std::string foldProbe(
          const std::vector<std::string> &probe, const TypePtr &type, NameAllocator &names,
          DataflowResult &out
      ) {
        if (probe.size() == 1)
          return probe.front();
        const std::string g = names.fresh(type, out.lets);
        emit(out, g, buildLocalAtom(probe.front()));
        for (std::size_t i = 1; i < probe.size(); ++i)
          emit(out, g, buildBinAtom(g, AtomOpKind::Xor, probe[i]));
        return g;
      }

      // Is the chain non-zero somewhere? A mask that vanishes everywhere leaves
      // the target spelled out, which is the one thing this is not for. Only a
      // sample can answer, since the domain is the whole width.
      [[nodiscard]] static bool
      maskVaries(const std::vector<std::int64_t> &seen, std::uint32_t bits, std::mt19937 &rng) {
        std::uniform_int_distribution<std::uint64_t> anyValue;
        for (int attempt = 0; attempt < rylink::hp::kMaskSamples; ++attempt) {
          const std::int64_t g = signExtend(static_cast<std::int64_t>(anyValue(rng)), bits);
          std::int64_t mask = ~std::int64_t{0};
          for (std::int64_t value: seen)
            mask &= g ^ value;
          if (signExtend(mask, bits) != 0)
            return true;
        }
        return false;
      }

      static void emit(DataflowResult &out, const std::string &dst, Atom rhs) {
        out.stmts.push_back(buildAssign(buildLValue(dst), buildExpr(std::move(rhs))));
      }
    };

    // The largest prime `p` with `p*p` inside the signed range of `bits`, or 0
    // when none is worth using. Squaring is the bound that matters: every
    // product the evaluation forms is between two residues, so `p*p` in range
    // means no step can overflow whatever the caller's variable holds.
    //
    // The field therefore shrinks with the width — 46337 at `i32`, 181 at
    // `i16`, 11 at `i8` — and below `kMinModulus` there are too few residues
    // for a line through them to be worth emitting.
    [[nodiscard]] std::int64_t fieldFor(std::uint32_t bits) {
      const auto isPrime = [](std::int64_t n) {
        if (n < 2)
          return false;
        for (std::int64_t d = 2; d * d <= n; ++d)
          if (n % d == 0)
            return false;
        return true;
      };
      // Largest `p` with `p*p <= hi`, from a floating-point root corrected
      // both ways so the bound holds exactly at every width.
      const std::int64_t hi = signedRange(bits).hi;
      std::int64_t ceiling = static_cast<std::int64_t>(std::sqrt(static_cast<double>(hi)));
      while (ceiling > 0 && ceiling > hi / ceiling)
        --ceiling;
      while ((ceiling + 1) <= hi / (ceiling + 1))
        ++ceiling;
      for (std::int64_t p = ceiling; p >= rylink::hp::kMinModulus; --p)
        if (isPrime(p))
          return p;
      return 0;
    }

    // Truncated `%`, as RefractIR and C both define it, so the coefficients are
    // solved against the arithmetic the emitted statements actually perform.
    [[nodiscard]] std::int64_t truncMod(std::int64_t a, std::int64_t p) { return a % p; }

    // The residue in `[0, p)`, which is where field arithmetic wants its
    // representatives and truncated `%` does not put them.
    [[nodiscard]] std::int64_t residue(std::int64_t a, std::int64_t p) {
      const std::int64_t r = truncMod(a, p);
      return r < 0 ? r + p : r;
    }

    // A line through the pin, over the field: `P(r) = (a1*r + a0) mod p`, with
    // `r` the probe reduced into the field. `a1` is drawn non-zero, which is
    // what makes `P` non-constant — the off-point a general interpolation adds
    // to square its system is, for a line, just the slope being chosen rather
    // than solved. `a0` then follows in closed form.
    //
    // The statements mirror the arithmetic the coefficients were solved
    // against, step for step, including that `%` truncates toward zero.
    class ArithmeticPolicy : public DataflowPolicy {
    public:
      const char *name() const override { return "prime-line"; }

      std::optional<DataflowResult> build(
          const DataflowSite &site, std::uint32_t bits, std::int64_t target, NameAllocator &names,
          std::mt19937 &rng
      ) override {
        const std::int64_t p = fieldFor(bits);
        if (p == 0)
          return std::nullopt;
        std::vector<Pin> pins = stablePins(site, bits);
        if (pins.empty())
          return std::nullopt;
        std::uniform_int_distribution<std::size_t> pickPin(0, pins.size() - 1);
        const Pin &pin = pins[pickPin(rng)];

        const std::int64_t r = residue(pin.value, p);
        const std::int64_t cm = residue(target, p);
        std::uniform_int_distribution<std::int64_t> pickSlope(1, p - 1);
        const std::int64_t a1 = pickSlope(rng);
        const std::int64_t a0 = residue(cm - a1 * r, p);

        DataflowResult result;
        const TypePtr type = buildIntType(static_cast<int>(bits));
        const std::string mod = names.literal(p, type, result.lets);
        const std::string acc = names.fresh(type, result.lets);

        // r = ((%v % p) + p) % p, so the residue is non-negative before it is
        // scaled — a truncated `%` on a negative variable would otherwise put
        // the line's input outside the field.
        emit(result, acc, buildBinAtom(pin.name, AtomOpKind::Mod, mod));
        emitTail(result, acc, AddOp::Plus, p);
        emit(result, acc, buildBinAtom(acc, AtomOpKind::Mod, mod));
        // P(r), reduced after the product so the sum that follows stays under
        // 2p rather than p*p + p.
        emit(result, acc, buildOpAtom(Coef{IntLit{a1, {}}}, AtomOpKind::Mul, acc));
        emit(result, acc, buildBinAtom(acc, AtomOpKind::Mod, mod));
        emitTail(result, acc, AddOp::Plus, a0);
        emit(result, acc, buildBinAtom(acc, AtomOpKind::Mod, mod));

        // `- cm + target` rather than one folded constant: `target - cm` can
        // fall outside the width when the target sits near its end, while the
        // two steps each stay inside it.
        result.value = buildExpr(buildLocalAtom(acc));
        if (cm != 0)
          appendTail(result.value, AddOp::Minus, buildIntAtom(cm));
        if (target != 0)
          appendTail(result.value, AddOp::Plus, buildIntAtom(target));
        return result;
      }

    private:
      static void emit(DataflowResult &out, const std::string &dst, Atom rhs) {
        out.stmts.push_back(buildAssign(buildLValue(dst), buildExpr(std::move(rhs))));
      }

      static void emitTail(DataflowResult &out, const std::string &dst, AddOp op, std::int64_t v) {
        Expr e = buildExpr(buildLocalAtom(dst));
        appendTail(e, op, buildIntAtom(v));
        out.stmts.push_back(buildAssign(buildLValue(dst), std::move(e)));
      }
    };

  } // namespace

  std::unique_ptr<DataflowPolicy> makeArithmeticPolicy() {
    return std::make_unique<ArithmeticPolicy>();
  }

  std::unique_ptr<DataflowPolicy> makeBaselinePolicy() {
    return std::make_unique<BaselinePolicy>();
  }

  std::unique_ptr<DataflowPolicy> makeBitwisePolicy() { return std::make_unique<BitwisePolicy>(); }

  std::optional<std::int64_t> stableValue(const DataflowSite &site, const std::string &name) {
    std::optional<std::int64_t> held;
    for (const auto &visit: site.visits) {
      const auto it = std::find_if(visit.begin(), visit.end(), [&](const Pin &pin) {
        return pin.name == name;
      });
      if (it == visit.end())
        return std::nullopt;
      if (held && *held != it->value)
        return std::nullopt;
      held = it->value;
    }
    return held;
  }

  std::unordered_map<std::string, std::uint32_t> declaredIntWidths(const FunDecl &fn) {
    std::unordered_map<std::string, std::uint32_t> widths;
    const auto record = [&widths](const std::string &name, const TypePtr &type) {
      if (auto bits = TypeUtils::getIntBitWidth(type))
        widths.emplace(name, static_cast<std::uint32_t>(*bits));
    };
    for (const auto &param: fn.params)
      record(param.name.name, param.type);
    for (const auto &let: fn.lets)
      record(let.name.name, let.type);
    return widths;
  }

  DataflowSite pinsAtBlock(
      const StateProfile &profile, const std::string &blockLabel,
      const std::unordered_map<std::string, std::uint32_t> &widths
  ) {
    DataflowSite site;
    for (const StatePoint &point: profile.trace) {
      if (point.block != blockLabel || point.instr != -1)
        continue;
      std::vector<Pin> pins;
      for (const auto &[name, value]: point.vars) {
        if (value.kind != StateValue::Kind::Int)
          continue;
        const auto width = widths.find(name);
        if (width == widths.end())
          continue;
        const SignedRange range = signedRange(width->second);
        if (value.intVal < range.lo || value.intVal > range.hi)
          continue;
        pins.push_back(Pin{name, value.intVal, width->second});
      }
      site.visits.push_back(std::move(pins));
    }
    return site;
  }

  std::optional<DataflowResult> buildArgument(
      const std::vector<std::unique_ptr<DataflowPolicy>> &policies, const DataflowSite &site,
      std::uint32_t bits, std::int64_t target, NameAllocator &names, std::mt19937 &rng
  ) {
    std::vector<DataflowPolicy *> order;
    order.reserve(policies.size());
    for (const auto &policy: policies)
      order.push_back(policy.get());
    std::shuffle(order.begin(), order.end(), rng);
    for (DataflowPolicy *policy: order) {
      if (auto result = policy->build(site, bits, target, names, rng))
        return result;
    }
    return std::nullopt;
  }

} // namespace refractir::reify
