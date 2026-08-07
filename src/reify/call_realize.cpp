#include "reify/call_realize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "analysis/type_utils.hpp"
#include "ast/build.hpp"
#include "ast/sir_printer.hpp"
#include "reify/dataflow_policy.hpp"
#include "reify/hyperparameters.hpp"

namespace refractir::reify {

  // ---------------------------------------------------------------------------
  // Descriptor-value parsing
  //
  // Descriptors stringify scalars via std::to_string (ints) or a printf
  // float format (floats). std::stoll / std::stod accept both, so
  // round-tripping is straight. Both helpers return nullopt on any
  // partial parse so a malformed descriptor field just disables one
  // rewrite opportunity rather than crashing the engine.
  // ---------------------------------------------------------------------------
  static std::optional<std::int64_t> parseI64(const std::string &s) {
    try {
      size_t pos = 0;
      long long v = std::stoll(s, &pos);
      if (pos != s.size())
        return std::nullopt;
      return static_cast<std::int64_t>(v);
    } catch (...) {
      return std::nullopt;
    }
  }

  static std::optional<double> parseF64(const std::string &s) {
    // Delegate to the canonical RefractIR float parser so descriptor values
    // round-trip identically to lexed float literals — same subnormal
    // handling, same signed-zero behaviour, same overflow rule.
    try {
      return parseFloatLiteral(s);
    } catch (...) {
      return std::nullopt;
    }
  }

  // Width of an `iN` SIR type string, or 0 for anything else. Descriptors
  // carry parameter types as surface syntax, and the width is all a dataflow
  // policy needs to know about an integer one.
  static int intTypeBits(const std::string &sirType) {
    if (sirType.size() < 2 || sirType[0] != 'i')
      return 0;
    return std::max(0, std::atoi(sirType.c_str() + 1));
  }

  // ---------------------------------------------------------------------------
  // Build an Expr that evaluates to a scalar literal of the given SIR
  // type. Used to construct call arguments from descriptor.paramValues.
  // Returns nullopt for types we can't synthesize (pointers, vectors,
  // structs) — the rule then declines to apply.
  // ---------------------------------------------------------------------------
  static std::optional<Expr> makeScalarLitExpr(const std::string &sirType, const std::string &val) {
    Atom a;
    if (sirType.size() >= 1 && sirType[0] == 'i') {
      auto iv = parseI64(val);
      if (!iv)
        return std::nullopt;
      IntLit lit;
      lit.value = *iv;
      Coef coef = lit;
      CoefAtom ca;
      ca.coef = coef;
      a.v = ca;
    } else if (sirType == "f32" || sirType == "f64") {
      auto fv = parseF64(val);
      if (!fv)
        return std::nullopt;
      FloatLit lit;
      lit.value = *fv;
      Coef coef = lit;
      CoefAtom ca;
      ca.coef = coef;
      a.v = ca;
    } else {
      return std::nullopt;
    }
    Expr e;
    e.first = std::move(a);
    return e;
  }

  // Does anything take `%var`'s address? A variable reached through a pointer
  // can change without an assignment naming it, so a rewrite that relies on
  // its value has to see that first.
  struct AddrChecker {
    const std::string &varName;
    bool addressTaken = false;

    AddrChecker(const std::string &vn) : varName(vn) {}

    void check(const Atom &atom) {
      if (addressTaken)
        return;
      if (auto addr = std::get_if<AddrAtom>(&atom.v)) {
        if (addr->lv.base.name == varName) {
          addressTaken = true;
          return;
        }
      }
      if (auto select = std::get_if<SelectAtom>(&atom.v)) {
        if (select->cond)
          check(*select->cond);
        if (select->maskExpr)
          check(*select->maskExpr);
      } else if (auto call = std::get_if<CallAtom>(&atom.v)) {
        for (auto &arg: call->args) {
          if (arg)
            check(*arg);
        }
      }
    }

    void check(const Expr &expr) {
      if (addressTaken)
        return;
      check(expr.first);
      for (auto &tail: expr.rest) {
        check(tail.atom);
      }
    }

    void check(const Cond &cond) {
      if (addressTaken)
        return;
      check(cond.lhs);
      check(cond.rhs);
    }

    void check(const Instr &instr) {
      if (addressTaken)
        return;
      std::visit(
          [this](auto &arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, AssignInstr>) {
              check(arg.rhs);
            } else if constexpr (std::is_same_v<T, AssumeInstr> ||
                                 std::is_same_v<T, RequireInstr>) {
              check(arg.cond);
            } else if constexpr (std::is_same_v<T, StoreInstr>) {
              check(arg.ptr);
              check(arg.val);
            }
          },
          instr
      );
    }

    void check(const Terminator &term) {
      if (addressTaken)
        return;
      std::visit(
          [this](auto &arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, BrTerm>) {
              if (arg.cond)
                check(*arg.cond);
            } else if constexpr (std::is_same_v<T, RetTerm>) {
              if (arg.value)
                check(*arg.value);
            }
          },
          term
      );
    }
  };

  // Does anything put a value in `%var` that its declaration no longer
  // describes — an assignment naming it, or a write through its address?
  static bool isMutatedOrAddressTaken(const FunDecl &caller, const std::string &varName) {
    AddrChecker checker(varName);
    for (const auto &block: caller.blocks) {
      for (const auto &instr: block.instrs) {
        if (auto assign = std::get_if<AssignInstr>(&instr))
          if (assign->lhs.base.name == varName)
            return true;
        checker.check(instr);
      }
      checker.check(block.term);
    }
    return checker.addressTaken;
  }

  // Can control reach this block again? Everything in a block on a cycle runs
  // once per iteration.
  static bool isPartOfLoop(const FunDecl &caller, std::size_t startIdx) {
    const auto successors = [&caller](std::size_t bi) {
      std::vector<std::size_t> succs;
      const auto addLabel = [&](const std::string &label) {
        for (std::size_t i = 0; i < caller.blocks.size(); ++i)
          if (caller.blocks[i].label.name == label) {
            succs.push_back(i);
            break;
          }
      };
      if (auto br = std::get_if<BrTerm>(&caller.blocks[bi].term)) {
        if (br->isConditional) {
          addLabel(br->thenLabel.name);
          addLabel(br->elseLabel.name);
        } else {
          addLabel(br->dest.name);
        }
      }
      return succs;
    };

    std::vector<bool> seen(caller.blocks.size(), false);
    std::vector<std::size_t> stack = successors(startIdx);
    while (!stack.empty()) {
      const std::size_t curr = stack.back();
      stack.pop_back();
      if (curr == startIdx)
        return true;
      if (seen[curr])
        continue;
      seen[curr] = true;
      for (std::size_t s: successors(curr))
        stack.push_back(s);
    }
    return false;
  }

  // ---------------------------------------------------------------------------
  // The spliced expression
  //
  // Both rules produce the same thing — `call @callee(args) + (target - o)`,
  // which evaluates to `target` because the callee returns its solved `o` — and
  // differ only in where they put it and what it displaces. This builds it,
  // along with the statements the arguments need.
  // ---------------------------------------------------------------------------

  namespace {

    struct SplicedCall {
      std::vector<LetDecl> lets;
      std::vector<Instr> stmts;
      Expr value;
    };

    // `target` is the value the caller had here and must still have. Returns
    // nullopt when an argument has no expressible form, or when the offset has
    // no literal at the destination's width.
    [[nodiscard]] std::optional<SplicedCall> buildSplicedCall(
        const FuncDescriptor &callee, std::size_t realizationIdx, const DataflowSite &pins,
        const std::vector<std::unique_ptr<DataflowPolicy>> &policies, std::int64_t target,
        int destBits, NameAllocator &names, std::mt19937 &rng
    ) {
      const auto &rz = callee.realizations[realizationIdx];
      if (rz.paramValues.size() != callee.params.size())
        return std::nullopt;
      const auto retVal = parseI64(rz.retValue);
      if (!retVal)
        return std::nullopt;

      SplicedCall out;
      CallAtom ca;
      GlobalId gid;
      gid.name = callee.name;
      ca.callee = gid;
      std::uniform_real_distribution<double> coin(0.0, 1.0);
      for (std::size_t i = 0; i < callee.params.size(); ++i) {
        const auto &paramType = callee.params[i].type;
        const auto &paramValStr = rz.paramValues[i].second;
        std::optional<Expr> argExpr;
        const int paramBits = intTypeBits(paramType);
        // Whether this argument is stated in terms of the caller's state at
        // all. The solved literal is the alternative, and keeping some of them
        // is what leaves a compiler something to compare against.
        const bool replace = coin(rng) < rylink::hp::kPReplaceParam;
        if (auto want = replace && paramBits > 0 ? parseI64(paramValStr) : std::nullopt) {
          if (auto built = buildArgument(
                  policies, pins, static_cast<std::uint32_t>(paramBits), *want, names, rng
              )) {
            out.lets.insert(
                out.lets.end(), std::make_move_iterator(built->lets.begin()),
                std::make_move_iterator(built->lets.end())
            );
            out.stmts.insert(
                out.stmts.end(), std::make_move_iterator(built->stmts.begin()),
                std::make_move_iterator(built->stmts.end())
            );
            argExpr = std::move(built->value);
          }
        }
        if (!argExpr)
          argExpr = makeScalarLitExpr(paramType, paramValStr);
        if (!argExpr)
          return std::nullopt;
        ca.args.push_back(std::make_shared<Expr>(std::move(*argExpr)));
      }

      // `target - o` under bit-vector arithmetic reproduces `target`, but the C
      // backend emits the site as a signed addition and UBSan trips when the
      // intermediate leaves the destination's range even though the wrapped
      // result is right. Declining costs one splice; the engine tries another.
      std::int64_t offset = 0;
      if (__builtin_sub_overflow(target, *retVal, &offset))
        return std::nullopt;
      if (destBits > 0 && destBits < 64) {
        const std::int64_t maxv = (std::int64_t{1} << (destBits - 1)) - 1;
        if (offset < -maxv - 1 || offset > maxv)
          return std::nullopt;
      }

      Atom callAtom;
      callAtom.v = std::move(ca);
      out.value = buildExpr(std::move(callAtom));
      if (offset != 0)
        appendTail(out.value, AddOp::Plus, buildIntAtom(offset));
      return out;
    }

  } // namespace

  // ---------------------------------------------------------------------------
  // LiteralToCallRule
  // ---------------------------------------------------------------------------

  namespace {

    class LiteralToCallRule : public CallRewriteRule {
    public:
      LiteralToCallRule() {
        policies_.push_back(makeBaselinePolicy());
        policies_.push_back(makeBitwisePolicy());
        policies_.push_back(makeArithmeticPolicy());
      }

      const char *name() const override { return "LiteralToCall"; }

      std::vector<CallRewriteSite>
      findSites(const FunDecl &caller, const FuncDescriptor &) override {
        std::vector<CallRewriteSite> sites;
        for (size_t i = 0; i < caller.lets.size(); ++i) {
          const auto &ld = caller.lets[i];
          if (!ld.init)
            continue;
          // Apply emits an AssignInstr to ld.name; only `let mut` is
          // assignable per spec §3.5.2. rysmith currently emits every
          // let as mutable so this filter is defensive, but it
          // protects the rule against future generators that mix in
          // immutable lets.
          if (!ld.isMutable)
            continue;
          // Skip the checksum accumulator — it is unconditionally
          // overwritten to `0` at the top of the exit block (see
          // func_gen::buildSumChecksum), so any splice into its let-init
          // is dead computation. Filtering it out at the source rule
          // saves a wasted attempt budget per edge.
          if (ld.name.name == "%_chk")
            continue;
          // A local an earlier splice introduced to compute an argument. Its
          // initializer is scaffolding, not a constant the program computes.
          if (ld.name.name.rfind(kDataflowLocalPrefix, 0) == 0)
            continue;
          // Only scalar literal initializers — Atom-form inits are
          // skipped (they're already calls/loads/etc.) and aggregate
          // inits are out of scope for v1.
          if (ld.init->kind == InitVal::Kind::Int) {
            CallRewriteSite s;
            s.kind = CallRewriteSite::Kind::LetInitIntLit;
            s.letIdx = static_cast<int>(i);
            s.intVal = std::get<IntLit>(ld.init->value).value;
            s.sirType = SIRPrinter::typeToString(ld.type);
            sites.push_back(s);
          } else if (ld.init->kind == InitVal::Kind::Float) {
            CallRewriteSite s;
            s.kind = CallRewriteSite::Kind::LetInitFloatLit;
            s.letIdx = static_cast<int>(i);
            s.floatVal = std::get<FloatLit>(ld.init->value).value;
            s.sirType = SIRPrinter::typeToString(ld.type);
            sites.push_back(s);
          }
        }
        return sites;
      }

      bool matchCallee(
          const CallRewriteSite &site, const FuncDescriptor &callee, std::size_t fixedRealizationIdx
      ) override {
        // Only the ret-type has to match: we splice in `call + (c - ret)`
        // so semantics are preserved regardless of whether the callee
        // happens to solve to the same value as the literal. Float
        // literals are excluded because IEEE 754 subtraction is not
        // generally lossless (c - ret + ret ≠ c when rounding fires),
        // which would break --validate equivalence.
        if (callee.retType != site.sirType)
          return false;
        if (site.kind == CallRewriteSite::Kind::LetInitFloatLit)
          return false;
        if (fixedRealizationIdx >= callee.realizations.size())
          return false;
        const auto &rz = callee.realizations[fixedRealizationIdx];
        return !rz.retValue.empty() && parseI64(rz.retValue).has_value();
      }

      bool apply(
          FunDecl &caller, const FuncDescriptor &callerDesc, const StateProfile *callerProfile,
          const TypeUtils::StructTable &structs, const CallRewriteSite &site,
          const FuncDescriptor &callee, std::size_t realizationIdx, std::mt19937 &rng
      ) override {
        if (site.letIdx < 0 || (size_t) site.letIdx >= caller.lets.size())
          return false;
        const auto &rz = callee.realizations[realizationIdx];
        if (rz.paramValues.size() != callee.params.size())
          return false;
        auto retVal = parseI64(rz.retValue);
        if (!retVal)
          return false;

        const auto target = chooseTarget(caller, callerDesc, callerProfile, structs, site, rng);
        if (!target)
          return false;
        const std::size_t targetBlock = target->blockIdx;
        const DataflowSite &pins = target->pins;

        // Build the call atom. An integer argument goes to the dataflow
        // policies, which state the callee's solved value in terms of what the
        // caller holds where the call lands; anything else is spelled out.
        CallAtom ca;
        GlobalId gid;
        gid.name = callee.name;
        ca.callee = gid;
        std::vector<LetDecl> argLets;
        std::vector<Instr> argStmts;
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        NameAllocator names(
            std::string(kDataflowLocalPrefix) + "l" + std::to_string(spliceSeq_++) + "_",
            &caller.lets
        );
        for (size_t i = 0; i < callee.params.size(); ++i) {
          const auto &paramType = callee.params[i].type;
          const auto &paramValStr = rz.paramValues[i].second;
          std::optional<Expr> argExpr;
          const int paramBits = intTypeBits(paramType);
          // Whether this argument is stated in terms of the caller's state at
          // all. The solved literal is the alternative, and keeping some of
          // them is what leaves a compiler something to compare against.
          const bool replace = coin(rng) < rylink::hp::kPReplaceParam;
          if (auto target = replace && paramBits > 0 ? parseI64(paramValStr) : std::nullopt) {
            if (auto built = buildArgument(
                    policies_, pins, static_cast<std::uint32_t>(paramBits), *target, names, rng
                )) {
              argLets.insert(
                  argLets.end(), std::make_move_iterator(built->lets.begin()),
                  std::make_move_iterator(built->lets.end())
              );
              argStmts.insert(
                  argStmts.end(), std::make_move_iterator(built->stmts.begin()),
                  std::make_move_iterator(built->stmts.end())
              );
              argExpr = std::move(built->value);
            }
          }
          if (!argExpr)
            argExpr = makeScalarLitExpr(paramType, paramValStr);
          if (!argExpr)
            return false; // unsupported arg type — bail without mutating
          ca.args.push_back(std::make_shared<Expr>(std::move(*argExpr)));
        }
        Atom callAtom;
        callAtom.v = std::move(ca);

        // Offset literal: `c - ret` under bitvector arithmetic. RefractIR
        // `T + lit` truncates with wraparound (§6.5), so the BV-side
        // math always reproduces `c`. The C backend, however, emits
        // the call site as a signed-int addition, and UBSan trips
        // when the intermediate `ret + offset` overflows the let's
        // signed range — even though the wrapped result equals `c`.
        //
        // Decline the rewrite when the offset doesn't fit in the
        // let's signed range. Concretely: compute `c - ret` in 64-bit
        // signed (rejecting the i64-vs-i64 overflow case via the
        // builtin), then range-check against the destination width.
        //
        // The check is conservative — some perfectly fine programs
        // are skipped — but the alternative (BV-correct C output) is
        // a deep change to the backend; declining costs at most a
        // missed splice on a single edge and the rewrite engine just
        // tries another candidate site.
        int parsedBits = 0;
        if (site.sirType.size() < 2 || site.sirType[0] != 'i' ||
            (parsedBits = std::atoi(site.sirType.c_str() + 1)) <= 0)
          return false;
        std::int64_t offsetVal = 0;
        if (__builtin_sub_overflow(target->restore, (std::int64_t) *retVal, &offsetVal))
          return false;
        if (parsedBits < 64) {
          std::int64_t maxv = (std::int64_t{1} << (parsedBits - 1)) - 1;
          std::int64_t minv = -(std::int64_t{1} << (parsedBits - 1));
          if (offsetVal < minv || offsetVal > maxv)
            return false;
        }
        IntLit offsetLit;
        offsetLit.value = offsetVal;

        // Build `call (+ offset)?` as the RHS of an AssignInstr that
        // we prepend to the target block: `%x = call @callee(args) (+
        // offset)?;`. The original literal init stays in place — it
        // runs before any block, then the prepended assign overwrites
        // %x with the semantically-equivalent call expression. We
        // can't fold the call straight into the let-init because
        // InitVal::Atom holds a single Atom and we need an Expr (call
        // + offset).
        Expr rhs;
        rhs.first = std::move(callAtom);
        if (offsetLit.value != 0) {
          Coef offsetCoef = offsetLit;
          CoefAtom offsetCa;
          offsetCa.coef = offsetCoef;
          Atom offsetAtom;
          offsetAtom.v = std::move(offsetCa);
          Expr::Tail tail;
          tail.op = AddOp::Plus;
          tail.atom = std::move(offsetAtom);
          rhs.rest.push_back(std::move(tail));
        }

        AssignInstr ai;
        ai.lhs.base = caller.lets[site.letIdx].name;
        ai.rhs = std::move(rhs);

        // The argument statements run ahead of the call, and both go to the
        // head of the block: an argument is stated in terms of the state on
        // entry, so nothing the block itself does may run first.
        argStmts.push_back(Instr{std::move(ai)});
        auto &instrs = caller.blocks[targetBlock].instrs;
        instrs.insert(
            instrs.begin(), std::make_move_iterator(argStmts.begin()),
            std::make_move_iterator(argStmts.end())
        );
        caller.lets.insert(
            caller.lets.end(), std::make_move_iterator(argLets.begin()),
            std::make_move_iterator(argLets.end())
        );
        return true;
      }

    private:
      // Where a call can be spliced, what the assignment has to restore there,
      // and the state a policy builds its arguments against.
      struct Target {
        std::size_t blockIdx = 0;
        std::int64_t restore = 0;
        DataflowSite pins;
      };

      // Pick an executed block to splice into.
      //
      // The assignment overwrites the site's variable, so it may only land
      // where it puts back what the variable already holds. The profiled run
      // answers that directly: a block qualifies when the variable holds one
      // value across every visit made to it, and that value is what the offset
      // restores. Mutation and loops need no rule of their own — a variable
      // reassigned earlier simply holds a different value at the block, and one
      // that moves between iterations holds no single value there at all.
      //
      // Without a profile the declaration's initializer is the only value
      // known, and it holds at a block that no mutation precedes and no cycle
      // repeats.
      [[nodiscard]] std::optional<Target> chooseTarget(
          const FunDecl &caller, const FuncDescriptor &callerDesc,
          const StateProfile *callerProfile, const TypeUtils::StructTable &structs,
          const CallRewriteSite &site, std::mt19937 &rng
      ) const {
        const std::string &varName = caller.lets[site.letIdx].name.name;
        const std::unordered_set<std::string> onPath(
            callerDesc.path.begin(), callerDesc.path.end()
        );
        std::vector<std::size_t> executed;
        for (std::size_t i = 0; i < caller.blocks.size(); ++i)
          if (onPath.count(caller.blocks[i].label.name))
            executed.push_back(i);
        std::shuffle(executed.begin(), executed.end(), rng);

        if (!callerProfile) {
          if (isMutatedOrAddressTaken(caller, varName))
            return std::nullopt;
          for (std::size_t blockIdx: executed)
            if (!isPartOfLoop(caller, blockIdx))
              return Target{blockIdx, site.intVal, DataflowSite{}};
          return std::nullopt;
        }

        // `executed` is shuffled, so the first qualifying block is a uniform
        // draw among them. Spreading splices over the path is worth more than
        // steering them toward any one kind of block.
        const auto leaves = declaredLeafTypes(caller, structs);
        for (std::size_t blockIdx: executed) {
          DataflowSite pins =
              pinsAtBlock(*callerProfile, caller.blocks[blockIdx].label.name, leaves);
          if (auto held = stableValue(pins, varName))
            return Target{blockIdx, *held, std::move(pins)};
        }
        return std::nullopt;
      }

      // Drawn from per argument. The order they are asked in is the draw; each
      // one either states the target or passes.
      std::vector<std::unique_ptr<DataflowPolicy>> policies_;
      // Distinguishes one splice's scratch locals from the next's, since each
      // allocator only sees the declarations it makes itself.
      std::size_t spliceSeq_ = 0;
    };

  } // namespace

  std::unique_ptr<CallRewriteRule> makeLiteralToCallRule() {
    return std::make_unique<LiteralToCallRule>();
  }

  // ---------------------------------------------------------------------------
  // Engine
  // ---------------------------------------------------------------------------

  static std::optional<std::string>
  getVarTypeString(const std::string &name, const FunDecl &caller) {
    for (const auto &p: caller.params) {
      if (p.name.name == name) {
        return SIRPrinter::typeToString(p.type);
      }
    }
    for (const auto &l: caller.lets) {
      if (l.name.name == name) {
        return SIRPrinter::typeToString(l.type);
      }
    }
    for (const auto &s: caller.syms) {
      if (s.name.name == name) {
        return SIRPrinter::typeToString(s.type);
      }
    }
    return std::nullopt;
  }

  static std::optional<std::string> inferExprType(const Expr &expr, const FunDecl &caller) {
    auto getAtomType = [&](const Atom &atom) -> std::optional<std::string> {
      if (auto coef = std::get_if<CoefAtom>(&atom.v)) {
        if (auto varId = std::get_if<LocalOrSymId>(&coef->coef)) {
          std::string varName;
          if (auto loc = std::get_if<LocalId>(varId))
            varName = loc->name;
          else if (auto sym = std::get_if<SymId>(varId))
            varName = sym->name;
          return getVarTypeString(varName, caller);
        }
      } else if (auto rval = std::get_if<RValueAtom>(&atom.v)) {
        if (rval->rval.accesses.empty()) {
          return getVarTypeString(rval->rval.base.name, caller);
        }
      }
      return std::nullopt;
    };

    if (auto t = getAtomType(expr.first))
      return t;
    for (const auto &tail: expr.rest) {
      if (auto t = getAtomType(tail.atom))
        return t;
    }
    return std::nullopt;
  }

  static std::optional<std::string> getExpectedTypeForExpr(
      const Expr &expr, const FunDecl &caller, const std::optional<std::string> &contextType
  ) {
    if (auto inferred = inferExprType(expr, caller)) {
      return inferred;
    }
    return contextType;
  }

  // ---------------------------------------------------------------------------
  // CallReplacer
  //
  // An AST mutator that walks expression trees, conditions, instructions,
  // and terminators in search of variables or literals that match the callee's
  // return type. It collects pointers to all matching atoms as candidates.
  //
  // Expected types of literals are inferred from their context (e.g. assignment
  // LHS, condition partner, or store target) to guarantee type checker safety.
  // ---------------------------------------------------------------------------
  struct CallReplacer {
    const TypePtr &expectedType;
    const FunDecl &caller;
    std::string expectedTypeStr;
    std::vector<Atom *> candidates;

    CallReplacer(const TypePtr &et, const FunDecl &c) : expectedType(et), caller(c) {
      if (expectedType) {
        expectedTypeStr = SIRPrinter::typeToString(expectedType);
      }
    }

    void collect(Atom &atom, const std::optional<std::string> &contextType) {
      if (expectedTypeStr.empty())
        return;

      if (auto coefAtom = std::get_if<CoefAtom>(&atom.v)) {
        bool match = false;
        if (std::holds_alternative<IntLit>(coefAtom->coef)) {
          if (contextType && *contextType == expectedTypeStr && expectedTypeStr[0] == 'i') {
            match = true;
          }
        } else if (std::holds_alternative<FloatLit>(coefAtom->coef)) {
          if (contextType && *contextType == expectedTypeStr &&
              (expectedTypeStr == "f32" || expectedTypeStr == "f64")) {
            match = true;
          }
        } else if (std::holds_alternative<NullLit>(coefAtom->coef)) {
          if (contextType && *contextType == expectedTypeStr &&
              expectedTypeStr.rfind("ptr", 0) == 0) {
            match = true;
          }
        } else if (auto varId = std::get_if<LocalOrSymId>(&coefAtom->coef)) {
          std::string varName = std::holds_alternative<LocalId>(*varId)
                                    ? std::get<LocalId>(*varId).name
                                    : std::get<SymId>(*varId).name;
          if (auto tStr = getVarTypeString(varName, caller)) {
            if (*tStr == expectedTypeStr)
              match = true;
          }
        }
        if (match) {
          candidates.push_back(&atom);
        }
      } else if (auto rvalAtom = std::get_if<RValueAtom>(&atom.v)) {
        if (rvalAtom->rval.accesses.empty()) {
          if (auto tStr = getVarTypeString(rvalAtom->rval.base.name, caller)) {
            if (*tStr == expectedTypeStr) {
              candidates.push_back(&atom);
            }
          }
        }
      }

      // Traverse nested
      if (auto select = std::get_if<SelectAtom>(&atom.v)) {
        if (select->cond)
          collect(*select->cond);
        if (select->maskExpr)
          collect(*select->maskExpr, std::nullopt);
      } else if (auto call = std::get_if<CallAtom>(&atom.v)) {
        for (auto &arg: call->args) {
          if (arg)
            collect(*arg, std::nullopt);
        }
      }
    }

    void collect(Expr &expr, const std::optional<std::string> &contextType) {
      auto effType = getExpectedTypeForExpr(expr, caller, contextType);
      collect(expr.first, effType);
      for (auto &tail: expr.rest) {
        collect(tail.atom, effType);
      }
    }

    void collect(Cond &cond) {
      auto lhsType = inferExprType(cond.lhs, caller);
      auto rhsType = inferExprType(cond.rhs, caller);
      collect(cond.lhs, rhsType);
      collect(cond.rhs, lhsType);
    }

    void collect(Instr &instr) {
      std::visit(
          [this](auto &arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, AssignInstr>) {
              std::optional<std::string> lhsType;
              if (arg.lhs.accesses.empty()) {
                lhsType = getVarTypeString(arg.lhs.base.name, caller);
              }
              collect(arg.rhs, lhsType);
            } else if constexpr (std::is_same_v<T, AssumeInstr> ||
                                 std::is_same_v<T, RequireInstr>) {
              collect(arg.cond);
            } else if constexpr (std::is_same_v<T, StoreInstr>) {
              std::optional<std::string> valType;
              if (auto ptrType = inferExprType(arg.ptr, caller)) {
                if (ptrType->rfind("ptr ", 0) == 0) {
                  valType = ptrType->substr(4);
                }
              }
              collect(arg.ptr, std::nullopt);
              collect(arg.val, valType);
            }
          },
          instr
      );
    }

    void collect(Terminator &term) {
      std::visit(
          [this](auto &arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, BrTerm>) {
              if (arg.cond)
                collect(*arg.cond);
            } else if constexpr (std::is_same_v<T, RetTerm>) {
              if (arg.value) {
                std::optional<std::string> retTypeStr = SIRPrinter::typeToString(caller.retType);
                collect(*arg.value, retTypeStr);
              }
            }
          },
          term
      );
    }
  };

  // ---------------------------------------------------------------------------
  // Build arguments for call realization.
  //
  // - If `randomize` is true (for unexecuted blocks), we synthesize random
  //   arguments or pick matching variables since they are never executed and cannot trigger UB.
  // - If `randomize` is false, we use only exact literal constants
  //   representing the callee's solved param values. Arguments stated in terms
  //   of caller variables are the dataflow policies' business, and they answer
  //   to a splice point whose state is known — which an arbitrary block is not.
  // ---------------------------------------------------------------------------
  static std::optional<std::vector<std::shared_ptr<Expr>>> makeCallArgs(
      const FunDecl &caller, const FunDecl &calleeFn, const FuncDescriptor &callee,
      const FuncDescriptor::Realization &rz, std::mt19937 &rng, bool randomize
  ) {
    std::vector<std::shared_ptr<Expr>> args;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_int_distribution<int> randInt(0, 100);
    std::uniform_real_distribution<double> randFloat(0.0, 100.0);

    for (size_t i = 0; i < callee.params.size(); ++i) {
      const auto &paramType = callee.params[i].type;
      const auto &paramValStr = rz.paramValues[i].second;
      std::optional<Expr> argExpr;

      if (randomize) {
        std::vector<std::string> matchingVars;
        for (const auto &p: caller.params) {
          if (SIRPrinter::typeToString(p.type) == paramType) {
            matchingVars.push_back(p.name.name);
          }
        }
        for (const auto &l: caller.lets) {
          if (SIRPrinter::typeToString(l.type) == paramType) {
            matchingVars.push_back(l.name.name);
          }
        }
        for (const auto &s: caller.syms) {
          if (SIRPrinter::typeToString(s.type) == paramType) {
            matchingVars.push_back(s.name.name);
          }
        }

        bool chooseVar = false;
        if (!matchingVars.empty()) {
          std::uniform_int_distribution<int> coin(0, 1);
          if (coin(rng) == 0) {
            chooseVar = true;
          }
        }

        if (chooseVar) {
          std::uniform_int_distribution<size_t> pick(0, matchingVars.size() - 1);
          std::string chosenVar = matchingVars[pick(rng)];
          if (chosenVar.size() >= 2 && chosenVar[1] == '?') {
            SymId sid;
            sid.name = chosenVar;
            sid.span = calleeFn.name.span;
            CoefAtom ca;
            ca.coef = LocalOrSymId{sid};
            ca.span = calleeFn.name.span;
            Atom a;
            a.v = ca;
            a.span = calleeFn.name.span;
            Expr e;
            e.first = std::move(a);
            e.span = calleeFn.name.span;
            argExpr = std::move(e);
          } else {
            LocalId lid;
            lid.name = chosenVar;
            lid.span = calleeFn.name.span;
            LValue lv;
            lv.base = lid;
            lv.span = calleeFn.name.span;
            RValueAtom rva;
            rva.rval = lv;
            rva.span = calleeFn.name.span;
            Atom a;
            a.v = rva;
            a.span = calleeFn.name.span;
            Expr e;
            e.first = std::move(a);
            e.span = calleeFn.name.span;
            argExpr = std::move(e);
          }
        } else {
          if (paramType.size() >= 2 && paramType[0] == 'i') {
            int bits = std::atoi(paramType.c_str() + 1);
            if (bits <= 0)
              bits = 32;

            IntLit lit;
            lit.value = randInt(rng) % (1LL << std::min(bits, 30));
            Coef coef = lit;
            CoefAtom ca;
            ca.coef = coef;
            Atom a;
            a.v = ca;
            Expr e;
            e.first = std::move(a);
            argExpr = std::move(e);
          } else if (paramType == "f32" || paramType == "f64") {
            FloatLit lit;
            lit.value = randFloat(rng);
            Coef coef = lit;
            CoefAtom ca;
            ca.coef = coef;
            Atom a;
            a.v = ca;
            Expr e;
            e.first = std::move(a);
            argExpr = std::move(e);
          } else if (paramType.rfind("ptr", 0) == 0) {
            NullLit lit;
            Coef coef = lit;
            CoefAtom ca;
            ca.coef = coef;
            Atom a;
            a.v = ca;
            Expr e;
            e.first = std::move(a);
            argExpr = std::move(e);
          }
        }
      }

      if (!argExpr) {
        argExpr = makeScalarLitExpr(paramType, paramValStr);
      }
      if (!argExpr) {
        return std::nullopt;
      }
      args.push_back(std::make_shared<Expr>(std::move(*argExpr)));
    }
    return args;
  }

  // ---------------------------------------------------------------------------
  // AtomToCallRule
  //
  // A value the caller computes inside a statement the run executes can be
  // carried by a call. Two kinds qualify: an integer literal, whose value is
  // written down, and a read of a variable the profiled run holds steady —
  // steady across the visits to the block, and untouched within it, so the
  // value at the read is the value on entry. It cannot stand where either
  // stands: a flat expression evaluates left to right with no
  // parentheses, so `a + k * b` with `call + (k - o)` put in place of `k` would
  // reassociate everything after it. The call is hoisted into a cell above the
  // statement instead, and the literal becomes a read of that cell.
  //
  // Hoisting also settles composition. The call lands in an expression of its
  // own rather than joining someone else's `+` chain, so no prefix sum can
  // wrap, and the atom it replaced is no longer a literal for a later edge to
  // find.
  // ---------------------------------------------------------------------------

  namespace {

    class AtomToCallRule : public CallRewriteRule {
    public:
      AtomToCallRule() {
        policies_.push_back(makeBaselinePolicy());
        policies_.push_back(makeBitwisePolicy());
        policies_.push_back(makeArithmeticPolicy());
      }

      const char *name() const override { return "AtomToCall"; }

      // One site per (executed block, integer width) that holds a literal of
      // that width. Which literal is settled when the rewrite fires, since an
      // earlier splice into the same block will have moved them.
      std::vector<CallRewriteSite>
      findSites(const FunDecl &caller, const FuncDescriptor &callerDesc) override {
        const std::unordered_set<std::string> onPath(
            callerDesc.path.begin(), callerDesc.path.end()
        );
        std::vector<CallRewriteSite> sites;
        for (const auto &block: caller.blocks) {
          if (!onPath.count(block.label.name))
            continue;
          std::unordered_set<std::string> widths;
          for (const auto &type: knownTypes(caller, block))
            widths.insert(type);
          for (const auto &type: widths) {
            CallRewriteSite s;
            s.kind = CallRewriteSite::Kind::OnPathIntAtom;
            s.blockLabel = block.label.name;
            s.sirType = type;
            s.consumesSite = false;
            sites.push_back(std::move(s));
          }
        }
        return sites;
      }

      bool matchCallee(
          const CallRewriteSite &site, const FuncDescriptor &callee, std::size_t realizationIdx
      ) override {
        if (site.kind != CallRewriteSite::Kind::OnPathIntAtom)
          return false;
        if (callee.retType != site.sirType)
          return false;
        if (realizationIdx >= callee.realizations.size())
          return false;
        const auto &rz = callee.realizations[realizationIdx];
        return !rz.retValue.empty() && parseI64(rz.retValue).has_value();
      }

      bool apply(
          FunDecl &caller, const FuncDescriptor &, const StateProfile *callerProfile,
          const TypeUtils::StructTable &structs, const CallRewriteSite &site,
          const FuncDescriptor &callee, std::size_t realizationIdx, std::mt19937 &rng
      ) override {
        if (!callerProfile)
          return false;
        Block *block = nullptr;
        for (auto &b: caller.blocks)
          if (b.label.name == site.blockLabel)
            block = &b;
        if (!block)
          return false;

        const int bits = intTypeBits(site.sirType);
        NameAllocator names(
            std::string(kDataflowLocalPrefix) + "a" + std::to_string(spliceSeq_++) + "_",
            &caller.lets
        );
        const DataflowSite pins =
            pinsAtBlock(*callerProfile, site.blockLabel, declaredLeafTypes(caller, structs));
        auto found = pickKnown(caller, *block, site.sirType, steadyIn(caller, *block, pins), rng);
        if (!found)
          return false;
        auto spliced = buildSplicedCall(
            callee, realizationIdx, pins, policies_, found->value, bits, names, rng
        );
        if (!spliced)
          return false;

        // Everything that can fail has failed by here, so the two mutations
        // below cannot leave a half-rewritten body. The atom is overwritten
        // first: inserting into the statement list reallocates it and every
        // pointer into it, this one included.
        //
        // The call goes at the head of the block, not just above the statement
        // that reads it. Its arguments are stated against the state on entry,
        // and by the middle of a block the caller's variables have moved on;
        // the cell is fresh, so nothing between the two points can disturb it.
        const std::string cell = names.fresh(buildIntType(bits), spliced->lets);
        found->atom->v = RValueAtom{buildLValue(cell), {}};
        spliced->stmts.push_back(buildAssign(buildLValue(cell), std::move(spliced->value)));
        block->instrs.insert(
            block->instrs.begin(), std::make_move_iterator(spliced->stmts.begin()),
            std::make_move_iterator(spliced->stmts.end())
        );
        caller.lets.insert(
            caller.lets.end(), std::make_move_iterator(spliced->lets.begin()),
            std::make_move_iterator(spliced->lets.end())
        );
        return true;
      }

    private:
      struct Found {
        Atom *atom;
        std::int64_t value;
      };

      // The SIR types of the integer literals this block holds, by the type
      // each one is read at rather than its own — a literal takes the type of
      // where it stands.
      // The widths at which this block offers a value worth carrying. Only
      // literals count here: whether a variable holds still needs the profile,
      // which arrives with the rewrite rather than with the search.
      [[nodiscard]] static std::vector<std::string>
      knownTypes(const FunDecl &caller, const Block &block) {
        std::vector<std::string> types;
        Block &mutableBlock = const_cast<Block &>(block);
        const std::unordered_map<std::string, std::int64_t> none;
        for (auto &instr: mutableBlock.instrs)
          for (const auto &type: kIntTypes(caller))
            if (!collect(caller, instr, type, none).empty())
              types.push_back(type);
        return types;
      }

      // Integer widths any of the caller's own declarations use. A callee
      // returning some other width has nowhere to land anyway.
      [[nodiscard]] static std::vector<std::string> kIntTypes(const FunDecl &caller) {
        std::vector<std::string> types;
        std::unordered_set<std::string> seen;
        const auto add = [&](const TypePtr &t) {
          if (!TypeUtils::getIntBitWidth(t))
            return;
          const std::string s = SIRPrinter::typeToString(t);
          if (seen.insert(s).second)
            types.push_back(s);
        };
        for (const auto &p: caller.params)
          add(p.type);
        for (const auto &l: caller.lets)
          add(l.type);
        return types;
      }

      // Atoms in one instruction read at `sirType` whose value is known: a
      // literal, or a read of a variable in `steady`.
      [[nodiscard]] static std::vector<std::pair<Atom *, std::int64_t>> collect(
          const FunDecl &caller, Instr &instr, const std::string &sirType,
          const std::unordered_map<std::string, std::int64_t> &steady
      ) {
        const int bits = intTypeBits(sirType);
        if (bits <= 0)
          return {};
        CallReplacer replacer(buildIntType(bits), caller);
        replacer.collect(instr);
        std::vector<std::pair<Atom *, std::int64_t>> known;
        for (Atom *a: replacer.candidates) {
          if (auto coef = std::get_if<CoefAtom>(&a->v)) {
            if (auto lit = std::get_if<IntLit>(&coef->coef)) {
              known.emplace_back(a, lit->value);
              continue;
            }
            if (auto id = std::get_if<LocalOrSymId>(&coef->coef))
              if (auto local = std::get_if<LocalId>(id)) {
                const auto it = steady.find(local->name);
                if (it != steady.end())
                  known.emplace_back(a, it->second);
              }
            continue;
          }
          if (auto rval = std::get_if<RValueAtom>(&a->v))
            if (rval->rval.accesses.empty()) {
              const auto it = steady.find(rval->rval.base.name);
              if (it != steady.end())
                known.emplace_back(a, it->second);
            }
        }
        return known;
      }

      // Variables whose value at any point in `block` is the value on entry:
      // the same at every visit the run made, and changed by nothing the block
      // itself does. A variable the block assigns holds something else by the
      // time a later statement reads it, and a profile taken per block cannot
      // say what.
      [[nodiscard]] static std::unordered_map<std::string, std::int64_t>
      steadyIn(const FunDecl &caller, const Block &block, const DataflowSite &pins) {
        std::unordered_set<std::string> written;
        for (const auto &instr: block.instrs)
          if (auto assign = std::get_if<AssignInstr>(&instr))
            written.insert(assign->lhs.base.name);
        std::unordered_map<std::string, std::int64_t> steady;
        if (pins.visits.empty())
          return steady;
        for (const Pin &pin: pins.visits.front()) {
          // A plain integer local only: what stands in the statement is a bare
          // read, so a leaf reached through a path or a float needing a cast
          // has no atom here to take the place of.
          if (!pin.path.empty() || pin.viaFloat)
            continue;
          if (written.count(pin.root) || isMutatedOrAddressTaken(caller, pin.root))
            continue;
          if (auto held = stableValue(pins, pin.key))
            steady.emplace(pin.root, *held);
        }
        return steady;
      }

      [[nodiscard]] static std::optional<Found> pickKnown(
          const FunDecl &caller, Block &block, const std::string &sirType,
          const std::unordered_map<std::string, std::int64_t> &steady, std::mt19937 &rng
      ) {
        std::vector<Found> all;
        for (auto &instr: block.instrs)
          for (const auto &[atom, value]: collect(caller, instr, sirType, steady))
            all.push_back(Found{atom, value});
        if (all.empty())
          return std::nullopt;
        std::uniform_int_distribution<std::size_t> pick(0, all.size() - 1);
        return all[pick(rng)];
      }

      std::vector<std::unique_ptr<DataflowPolicy>> policies_;
      std::size_t spliceSeq_ = 0;
    };

  } // namespace

  std::unique_ptr<CallRewriteRule> makeAtomToCallRule() {
    return std::make_unique<AtomToCallRule>();
  }

  // ---------------------------------------------------------------------------
  // insertCallInUnexecBlock
  //
  // Attempts to realize a call edge in a block of the caller.
  // Rather than prepending dummy let-assignments, this method uses CallReplacer
  // to substitute an existing variable or type-safe literal inside the block
  // with the call. Returns true on successful replacement; false if no match.
  // ---------------------------------------------------------------------------
  static bool insertCallInUnexecBlock(
      FunDecl &caller, const FunDecl &calleeFn, const FuncDescriptor &callee,
      const FuncDescriptor::Realization &rz, std::mt19937 &rng, std::size_t blockIdx, bool randomize
  ) {
    if (blockIdx >= caller.blocks.size())
      return false;

    auto argsOpt = makeCallArgs(caller, calleeFn, callee, rz, rng, randomize);
    if (!argsOpt)
      return false;

    CallAtom ca;
    GlobalId gid;
    gid.name = callee.name;
    ca.callee = gid;
    ca.args = std::move(*argsOpt);
    ca.span = calleeFn.name.span;

    CallReplacer replacer(calleeFn.retType, caller);

    // Collect all matching candidates in the block
    for (auto &instr: caller.blocks[blockIdx].instrs) {
      replacer.collect(instr);
    }
    replacer.collect(caller.blocks[blockIdx].term);

    // If we have candidates, pick one at random and replace it
    if (!replacer.candidates.empty()) {
      std::uniform_int_distribution<size_t> pick(0, replacer.candidates.size() - 1);
      Atom *chosen = replacer.candidates[pick(rng)];
      chosen->v = std::move(ca);
      return true;
    }

    return false;
  }

  RewriteReport CallRealizeTransform::rewriteEdge(
      FunDecl &caller, const FuncDescriptor &callerDesc, const StateProfile *callerProfile,
      const TypeUtils::StructTable &structs, const FunDecl &calleeFn, const FuncDescriptor &callee,
      std::size_t fixedRealizationIdx, std::mt19937 &rng
  ) {
    RewriteReport res;

    // Collect (rule, site) pairs.
    struct Candidate {
      CallRewriteRule *rule;
      CallRewriteSite site;
    };

    std::vector<Candidate> cands;
    for (auto &rule: rules_) {
      auto sites = rule->findSites(caller, callerDesc);
      for (auto &s: sites) {
        // Skip sites already consumed by an earlier rewriteEdge call —
        // stacking calls on the same let-init produces left-to-right
        // chains like `f1() + f2() + ...` whose prefix sums can wrap
        // in unintended ways even though each individual rewrite is
        // BV-sound. See CallRealizeTransform class header for the full note.
        if (s.consumesSite && consumed_.count({&caller, s.letIdx}))
          continue;
        cands.push_back({rule.get(), s});
      }
    }
    // Filter to the candidates whose callee actually matches the site before
    // rolling any dice. matchCallee is a pure predicate (ret-type
    // compatibility + a parseable realization value), so doing it up front is
    // free and stops the accept coin and the apply() work budget below from
    // being spent on sites that could never be spliced. Iterating sites and
    // checking the match inline (the previous shape) let kMaxAttemptsPerEdge
    // and the coin be burned on non-matching sites, so an edge could "fail"
    // even though a perfect, appliable match sat further down the shuffle.
    std::vector<Candidate> matched;
    matched.reserve(cands.size());
    for (auto &c: cands) {
      if (c.rule->matchCallee(c.site, callee, fixedRealizationIdx))
        matched.push_back(c);
    }

    res.found = static_cast<int>(matched.size());

    if (!matched.empty()) {
      // Accept probability is keyed on the match count (pRewriteForMatches) so
      // the expected number of rewrites per edge stays bounded no matter how
      // many sites match, while a lone match is always taken. With the per-
      // edge break removed, several *distinct* sites may be spliced on one
      // edge — each on a different let-init, never stacking on the same one.
      // Keyed on the pre-cap count, so the budget below cannot inflate it.
      const double pAccept = rylink::hp::pRewriteForMatches(matched.size());

      shuffleAndCap(matched, rng, static_cast<std::size_t>(rylink::hp::kMaxAttemptsPerEdge));
      std::uniform_real_distribution<double> uni(0.0, 1.0);

      for (auto &c: matched) {
        // A site consumed earlier — on a prior edge or earlier in this very
        // loop — must not be rewritten again: stacking calls on one let-init
        // builds an `f1()+f2()+…` left-prefix sum that can wrap. See the
        // CallRealizeTransform header note. Without break we now re-check here.
        // A workaround is to introduce new variables and statements first.
        // A complete solution would land when RefractIR support parentheses.
        if (c.site.consumesSite && consumed_.count({&caller, c.site.letIdx}))
          continue;
        if (uni(rng) >= pAccept)
          continue;
        if (c.rule->apply(
                caller, callerDesc, callerProfile, structs, c.site, callee, fixedRealizationIdx, rng
            )) {
          if (c.site.consumesSite)
            consumed_.insert({&caller, c.site.letIdx});
          ++res.applied;
        }
      }
    }

    const auto &rz = callee.realizations[fixedRealizationIdx];

    // Target unexecuted blocks safely. The execution path (block labels)
    // is recorded in callerDesc.path. Any block in the caller not found in this
    // path is unexecuted under the solved model, making it safe to populate with
    // additional calls using randomized arguments. We collect all unexecuted blocks,
    // shuffle them, and attempt to realize the call edge in one at random.
    std::vector<std::size_t> unexecutedIndices;
    std::unordered_set<std::string> pathSet(callerDesc.path.begin(), callerDesc.path.end());
    for (std::size_t i = 0; i < caller.blocks.size(); ++i) {
      if (pathSet.find(caller.blocks[i].label.name) == pathSet.end()) {
        unexecutedIndices.push_back(i);
      }
    }

    std::shuffle(unexecutedIndices.begin(), unexecutedIndices.end(), rng);
    for (std::size_t i: unexecutedIndices) {
      if (insertCallInUnexecBlock(caller, calleeFn, callee, rz, rng, i, true)) {
        ++res.applied;
        break;
      }
    }

    return res;
  }

  TransformReport CallRealizeTransform::apply(Program &prog, TransformContext &ctx) {
    TransformReport rep;

    // Resolve functions by name once — names are unique across a rylink
    // bundle (pool picks are without replacement), so a plain map suffices.
    std::unordered_map<std::string, FunDecl *> byName;
    for (auto &f: prog.funs)
      byName[f.name.name] = &f;

    TypeUtils::StructTable structs;
    for (const auto &sd: prog.structs)
      structs.emplace(sd.name.name, &sd);

    for (const auto &e: plan_.edges) {
      auto callerIt = byName.find(e.caller);
      auto calleeIt = byName.find(e.callee);
      if (callerIt == byName.end() || calleeIt == byName.end())
        continue;
      auto callerDesc = ctx.descriptors.find(e.caller);
      auto calleeDesc = ctx.descriptors.find(e.callee);
      if (callerDesc == ctx.descriptors.end() || calleeDesc == ctx.descriptors.end())
        continue;

      auto callerProfile = ctx.profiles.find(e.caller);
      RewriteReport r = rewriteEdge(
          *callerIt->second, callerDesc->second,
          callerProfile == ctx.profiles.end() ? nullptr : &callerProfile->second, structs,
          *calleeIt->second, calleeDesc->second, e.calleeRealizationIdx, ctx.rng
      );
      rep.sites += static_cast<std::size_t>(r.applied);
    }

    return rep;
  }

  std::unique_ptr<Transform> makeCallRealizeTransform(CallRealizePlan plan) {
    auto t = std::make_unique<CallRealizeTransform>(std::move(plan));
    t->addRule(makeLiteralToCallRule());
    t->addRule(makeAtomToCallRule());
    return t;
  }

} // namespace refractir::reify
