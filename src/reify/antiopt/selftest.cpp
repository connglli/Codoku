// The catalog's self-check.
//
// Every rule claims that two spellings of a body compute the same thing. The
// authority on what a RefractIR statement computes is the interpreter, so the
// check runs both spellings through it rather than restating the arithmetic
// here: a restatement only ever checks itself, and it cannot check a rule with
// no closed form — reordering statements, routing a value through memory —
// which was half the catalog.
//
// A rule supplies its own example as source text. The check parses a harness
// around it, applies the rule to one copy, and probes both from the same
// state, sweeping the operands over every i8 value. `twin_probe` already does
// exactly this for regions, so the harness is the same machinery a guard's
// spot checks run on.

#include <algorithm>
#include <climits>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ast/sir_printer.hpp"
#include "frontend/pipeline.hpp"
#include "internal.hpp"
#include "reify/common.hpp"
#include "reify/twin_probe.hpp"

namespace refractir::reify {

  namespace {

    using I64 = std::int64_t;

    // The locals every example is written over. All one width, so any rule can
    // combine them, and narrow enough that sweeping two of them covers every
    // pair of values there is.
    const char *const kCheckLocals[] = {"%x", "%y", "%d", "%e", "%k"};
    constexpr const char *kCheckBool = "%c";

    std::string harnessSource(const AntiOptRule::SelfTest &t, int bits) {
      std::string checkType = "i" + std::to_string(bits);
      std::string src = t.decls.empty() ? "" : t.decls + "\n\n";
      src += std::string("fun @check") + "(%pa0: " + checkType + ") : " + checkType + " {\n";
      for (const char *nm: kCheckLocals)
        src += std::string("  let mut ") + nm + ": " + checkType + " = 0;\n";
      src += std::string("  let mut ") + kCheckBool + ": i1 = 0;\n";
      src += "^body:\n";
      src += t.body;
      src += "\n  br ^done;\n^done:\n  ret %d;\n}\n";
      return src;
    }

    std::optional<Program> parseHarness(const std::string &src) {
      Program prog = parseSource(src);
      if (prog.funs.empty() || prog.funs.front().blocks.empty())
        return std::nullopt;
      // The checkers, so the example is a program and not merely text — and so
      // a call in it carries the overload the type checker picked, which is
      // what a rule over intrinsics matches on.
      if (!runAnalysisPasses(prog, /*verbose=*/false))
        return std::nullopt;
      return prog;
    }

    // What the rule is told, straight from its own declaration. The facts a
    // rule leans on are its precondition; whether they hold at a real site is
    // the caller's proof to make, and this check is about what follows *given*
    // them.
    class DeclaredFacts : public AntiOptFacts {
    public:
      explicit DeclaredFacts(const AntiOptRule::SelfTest &t) : t_(t) {}

      void refresh(const std::vector<Instr> &) override {}

      std::optional<ValueRange> rangeBefore(std::size_t, const std::string &local) const override {
        for (const auto &[nm, r]: t_.assume)
          if (nm == local)
            return r;
        return std::nullopt;
      }

      bool isFree(const std::string &local) const override {
        for (const auto &nm: t_.free)
          if (nm == local)
            return true;
        return false;
      }

    private:
      const AntiOptRule::SelfTest &t_;
    };

    // The value range the sweep gives a local: whatever the rule assumed, or
    // the whole of the harness width `bits`.
    std::pair<I64, I64>
    sweepRange(const AntiOptRule::SelfTest &t, const std::string &nm, int bits) {
      for (const auto &[name, r]: t.assume)
        if (name == nm)
          return {r.lo, r.hi};
      if (bits >= 64)
        return {INT64_MIN, INT64_MAX};
      if (bits > 0 && bits < 64) {
        int64_t hi = static_cast<int64_t>((1ULL << (bits - 1)) - 1ULL);
        int64_t lo = -static_cast<int64_t>(1ULL << (bits - 1));
        return {lo, hi};
      }
      return {-128, 127};
    }

    StateValue intState(I64 v, int bits) {
      StateValue sv;
      sv.kind = StateValue::Kind::Int;
      sv.intVal = v;
      sv.bits = bits;
      return sv;
    }

    // One root per check local, in the order `kCheckLocals` names them. The
    // i1 is left out: no example needs it seeded, and its width would have to
    // be modelled separately.
    std::vector<MiniRoot> checkRoots(const FunDecl &fn, int bits) {
      std::vector<MiniRoot> roots;
      for (const char *nm: kCheckLocals)
        for (const auto &l: fn.lets)
          if (l.name.name == nm) {
            roots.push_back(MiniRoot{nm, l.type, false, intState(0, bits), intState(0, bits), {}});
            break;
          }
      return roots;
    }

    // The two runs are compared on the state the program had before either of
    // them: the rewritten body also carries whatever scratch the rule declared
    // for itself, which nothing outside the body reads and which the original
    // has no counterpart for.
    bool sameEffect(const ProbeResult &a, const ProbeResult &b) {
      std::unordered_map<std::string, const StateValue *> mine;
      for (const auto &[nm, v]: b.effect)
        mine[nm] = &v;
      for (const auto &[nm, v]: a.effect) {
        if (nm.rfind(kAntiOptLocalPrefix, 0) == 0)
          continue;
        auto it = mine.find(nm);
        if (it == mine.end() || it->second->kind != v.kind || it->second->intVal != v.intVal)
          return false;
      }
      return true;
    }

    std::string describe(const std::vector<MiniRoot> &roots) {
      std::string out;
      for (const auto &r: roots) {
        if (!out.empty())
          out += ", ";
        out += r.name + "=" + std::to_string(r.init.intVal);
      }
      return out;
    }

    void dumpRuleTransform(const AntiOptRule &rule, const Program &before, const Program &after) {
      std::ostringstream osBefore, osAfter;
      SIRPrinter prBefore(osBefore);
      prBefore.print(before);
      SIRPrinter prAfter(osAfter);
      prAfter.print(after);
      std::string beforeStr = osBefore.str();
      std::string afterStr = osAfter.str();
      while (!beforeStr.empty() && (beforeStr.back() == '\n' || beforeStr.back() == '\r' ||
                                    beforeStr.back() == ' ' || beforeStr.back() == '\t'))
        beforeStr.pop_back();
      while (!afterStr.empty() && (afterStr.back() == '\n' || afterStr.back() == '\r' ||
                                   afterStr.back() == ' ' || afterStr.back() == '\t'))
        afterStr.pop_back();
      std::cout << rule.name() << ":\n" << beforeStr << "\n=>\n" << afterStr << "\n\n";
    }

    // Apply `rule` to the harness in `prog`, returning false when it declines
    // its own example — a rule that cannot fire on the case it nominated has
    // nothing to say about it.
    bool applyToHarness(
        const AntiOptRule &rule, const AntiOptRule::SelfTest &t, Program &prog, std::mt19937 &rng,
        std::string &failure
    ) {
      FunDecl &fn = prog.funs.front();
      Block &body = fn.blocks.front();
      StructMap structs = TypeUtils::buildStructTable(prog);
      std::vector<PathCheck> noChecks;
      NameAllocator names(kAntiOptLocalPrefix);
      DeclaredFacts facts(t);
      AntiOptContext ctx{fn, structs, noChecks, names, fn.lets, rng, &facts};

      if (t.at >= body.instrs.size()) {
        failure =
            std::string(rule.name()) + ": example has no statement at " + std::to_string(t.at);
        return false;
      }
      if (!rule.matches(body.instrs, RulePos{t.at}, ctx)) {
        failure = std::string(rule.name()) + ": does not fire on its own example";
        return false;
      }
      auto replacement = rule.apply(body.instrs, RulePos{t.at}, ctx);
      if (replacement.empty()) {
        failure = std::string(rule.name()) + ": declined its own example";
        return false;
      }
      const std::size_t width = rule.width();
      body.instrs.erase(
          body.instrs.begin() + (long) t.at, body.instrs.begin() + (long) (t.at + width)
      );
      body.instrs.insert(
          body.instrs.begin() + (long) t.at, std::make_move_iterator(replacement.begin()),
          std::make_move_iterator(replacement.end())
      );
      return true;
    }

    // Run one rule's example both ways over the sweep.
    bool checkRule(const AntiOptRule &rule, std::string &failure, int bits, bool verbose) {
      auto t = rule.selfTest(bits);
      if (!t)
        return true;

      const std::string src = harnessSource(*t, bits);
      auto before = parseHarness(src), after = parseHarness(src);
      if (!before || !after) {
        failure = std::string(rule.name()) + ": example does not parse";
        return false;
      }
      std::mt19937 rng(1234);
      if (!applyToHarness(rule, *t, *after, rng, failure))
        return false;

      std::vector<MiniRoot> roots = checkRoots(before->funs.front(), bits);
      if (roots.empty()) {
        failure = std::string(rule.name()) + ": harness has no roots";
        return false;
      }
      RegionProbe orig(*before, "@check", {"^body"}, roots);
      RegionProbe twin(*after, "@check", {"^body"}, roots);
      if (!orig.valid() || !twin.valid()) {
        failure = std::string(rule.name()) + (orig.valid()
                                                  ? ": the rewritten example is not a valid program"
                                                  : ": the example is not a valid program");
        return false;
      }
      if (verbose)
        dumpRuleTransform(rule, *before, *after);

      // Sweep what the bodies read. A rule whose example never mentions %y
      // is not tested by giving %y 256 values, and the difference is a
      // 256-fold one.
      std::unordered_set<std::string> read;
      for (const Program *p: {&*before, &*after})
        for (const auto &ins: p->funs.front().blocks.front().instrs)
          for (const auto &nm: antiopt::touchesOf(ins).reads)
            read.insert(nm);
      auto [xLo, xHi] = sweepRange(*t, "%x", bits);
      auto [yLo, yHi] = sweepRange(*t, "%y", bits);
      if (!read.count("%x"))
        xHi = xLo;
      if (!read.count("%y"))
        yHi = yLo;
      // Everything the sweep does not drive is held at the value the rule
      // assumed for it, or zero.
      for (auto &r: roots)
        r.init = intState(sweepRange(*t, r.name, bits).first, bits);

      for (I64 x = xLo; x <= xHi; ++x)
        for (I64 y = yLo; y <= yHi; ++y) {
          for (auto &r: roots) {
            if (r.name == "%x")
              r.init = intState(x, bits);
            else if (r.name == "%y")
              r.init = intState(y, bits);
          }
          const ProbeResult a = orig.run(roots);
          const ProbeResult b = twin.run(roots);
          if (!a.ok)
            continue; // the original traps here; the rewrite owes nothing
          if (!b.ok) {
            // Introducing a trap is what a Tier1 rule needs a proof for, and
            // what a Tier0 rule promised never to do.
            if (rule.tier() == TrapTier::Tier0) {
              failure = std::string(rule.name()) + ": trap-free rule traps at " + describe(roots);
              return false;
            }
            continue;
          }
          if (!sameEffect(a, b)) {
            failure = std::string(rule.name()) + ": disagrees at " + describe(roots);
            return false;
          }
        }
      return true;
    }

  } // namespace

  std::optional<std::vector<std::string>>
  selfTestRules(std::string &failure, int bits, bool verbose) {
    std::vector<std::string> checked;
    for (const auto &rule: antiopt::catalog()) {
      if (!rule->selfTest(bits))
        continue;
      if (!checkRule(*rule, failure, bits, verbose))
        return std::nullopt;
      checked.push_back(rule->name());
    }
    return checked;
  }

} // namespace refractir::reify
