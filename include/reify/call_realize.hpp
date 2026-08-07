#pragma once

// Call-realization transform for rylink.
//
// CallRealizeTransform is the whole-program Transform that turns rylink's
// chosen call-graph into real `call @callee(args)` sites. For each planned
// caller->callee edge it finds semantically-safe rewrite sites in the caller
// and substitutes them with a call whose solved arguments reproduce the
// original value. The site-finding is a small peephole engine over pluggable
// CallRewriteRules; ships LiteralToCallRule, which rewrites scalar-literal
// `let` initializers. Additional rules plug in without touching the transform.
//
// This is one instance of the peephole tier defined in reify/rewrite.hpp;
// see that header for the R1-R3 rule contract, and in particular for why
// composition safety is this engine's job rather than a rule's.

#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "analysis/type_utils.hpp"
#include "ast/ast.hpp"
#include "reify/func_desc.hpp"
#include "reify/rewrite.hpp"
#include "reify/state_profile.hpp"
#include "reify/transform.hpp"

namespace refractir::reify {

  // A candidate location inside a FunDecl where a rewrite could fire.
  // Opaque to the engine — only the rule that produced it knows how to
  // apply it. Kind discriminates so the engine can sort or filter by
  // category.
  struct CallRewriteSite {
    // LetInit*: a declaration's initializer, realized by an assignment
    // prepended to a block. OnPathIntAtom: an integer literal inside a
    // statement the run executes, realized by hoisting the call into a cell
    // above that statement and reading the cell where the literal stood.
    enum class Kind { LetInitIntLit, LetInitFloatLit, OnPathIntAtom };
    Kind kind;
    // OnPathIntAtom: the block holding the statement. The atom itself is found
    // again when the rewrite fires, since any earlier splice into the same
    // block has moved it.
    std::string blockLabel;
    // Whether rewriting this site once rules it out for good. A literal spliced
    // into becomes a call, and stacking a second on it would build a
    // left-to-right chain whose prefix sums can wrap; an atom replaced by a
    // read is no longer a literal and cannot be found again, so it needs no
    // bookkeeping.
    bool consumesSite = true;
    // Index into FunDecl::lets. The rule that emitted this site is
    // responsible for re-validating the index before applying (cheap
    // because literal rewriting mutates the init value, not the lets vector).
    int letIdx = 0;
    // The literal's value and its SIR-surface type-string. Used by the
    // engine to filter callees whose retType / ret value match without
    // re-walking the AST.
    std::int64_t intVal = 0;
    double floatVal = 0.0;
    std::string sirType;
  };

  // Peephole rule for call realization (reify/rewrite.hpp tier). `findSites`
  // and `matchCallee` are the tier's R1 purity obligation: neither may mutate
  // the caller.
  class CallRewriteRule {
  public:
    virtual ~CallRewriteRule() = default;
    [[nodiscard]] virtual const char *name() const = 0;
    // `callerDesc` carries the concretized path, so a rule that only rewrites
    // code the run executes can say so here rather than proposing candidates
    // the engine spends its budget rejecting.
    [[nodiscard]] virtual std::vector<CallRewriteSite>
    findSites(const FunDecl &caller, const FuncDescriptor &callerDesc) = 0;
    // Decide whether the site can be rewritten by calling into
    // `callee` using `fixedRealizationIdx` as the bundled realization
    // (the engine has already locked which realization runs at the
    // call site; rules cannot pick a different one). Returns true when
    // the rule can produce an `apply()` for this combination.
    virtual bool matchCallee(
        const CallRewriteSite &site, const FuncDescriptor &callee, std::size_t fixedRealizationIdx
    ) = 0;
    // Splice the call in. Returns true on success; false if some
    // late-stage check fails (e.g. a param type the rule can't handle).
    // `callerProfile` is the state the caller's own run passes through, which
    // is what an argument built from the caller's variables is stated against;
    // null when the tool captured none, leaving the rule its literal path.
    // `rng` is passed through so the rule can make randomised sub-decisions
    // (e.g. which dataflow policy states an argument) and still play nicely
    // with the engine's shuffled-candidates order.
    virtual bool apply(
        FunDecl &caller, const FuncDescriptor &callerDesc, const StateProfile *callerProfile,
        const TypeUtils::StructTable &structs, const CallRewriteSite &site,
        const FuncDescriptor &callee, std::size_t realizationIdx, std::mt19937 &rng
    ) = 0;
  };

  // Rule for scalar Int/Float literal let initializers.
  [[nodiscard]] std::unique_ptr<CallRewriteRule> makeLiteralToCallRule();

  // Rule for integer literals inside the statements the run executes. The call
  // is hoisted into a cell of its own above the statement, because RefractIR
  // evaluates a flat expression left to right: splicing `call + (k - o)` into
  // the middle of one would reassociate everything after it.
  [[nodiscard]] std::unique_ptr<CallRewriteRule> makeAtomToCallRule();

  // One planned call-graph edge: realize a call from `caller` into `callee`,
  // pinned to `calleeRealizationIdx`. Functions are named (canonical "@...")
  // rather than pointed-to so the plan is pure data — the transform resolves
  // the FunDecls out of the Program and the FuncDescriptors out of the
  // TransformContext at apply() time.
  struct CallRealizeEdge {
    std::string caller;
    std::string callee;
    std::size_t calleeRealizationIdx = 0;
  };

  // rylink's composition decision: which caller->callee edges to realize,
  // in application order. This is the tool's plan (which functions, which
  // realization); the per-function metadata it needs (descriptors, incl. the
  // concretized path) rides in the shared TransformContext.
  struct CallRealizePlan {
    std::vector<CallRealizeEdge> edges;
  };

  // Whole-program call-realization transform. Owns the peephole rules and
  // walks `plan.edges` in order, realizing each. Build one with
  // makeCallRealizeTransform (pre-loaded with LiteralToCallRule) and run it
  // through a TransformPipeline like any other Transform.
  class CallRealizeTransform : public Transform {
  public:
    explicit CallRealizeTransform(CallRealizePlan plan) : plan_(std::move(plan)) {}

    void addRule(std::unique_ptr<CallRewriteRule> r) { rules_.push_back(std::move(r)); }

    [[nodiscard]] std::string_view name() const override { return "CallRealizeTransform"; }

    // Resolve each planned edge's caller/callee (FunDecl from `prog`,
    // FuncDescriptor from `ctx.descriptors`) and realize it, drawing all
    // randomness from `ctx.rng`. Edges whose endpoints are missing are
    // skipped defensively. `TransformReport::sites` accumulates the spliced
    // call sites.
    TransformReport apply(Program &prog, TransformContext &ctx) override;

  private:
    // Realize sites in `caller` that call into `callee`. Filters sites to
    // those whose callee matches, then rolls a count-keyed acceptance coin
    // (pRewriteForMatches) per match under `rng`, splicing every
    // accepted+appliable site — possibly several distinct sites per edge —
    // up to the per-edge attempts cap. Each site is consumed once (see the
    // composition-safety note below), so a given let-init is never spliced
    // twice. Returns counters for telemetry.
    //
    // `fixedRealizationIdx` pins which realization of the callee the
    // rule must use — it has to be the one whose .sir was actually
    // merged into the bundle, otherwise the call would return a
    // different solved value than the rewrite expression assumes and
    // --validate would fail.
    //
    // Composition safety is this engine's discharge of the tier's R3
    // obligation (reify/rewrite.hpp): each rule is individually UB-free (the
    // call expression evaluates to the original literal under BV arithmetic),
    // but *stacking* rewrites on the same site is not, so the transform marks
    // each (caller, site) it successfully rewrites as consumed and skips it on
    // subsequent edges — one splice per site for the lifetime of the
    // transform.
    // `callerDesc` supplies the metadata of the caller function (including
    // the concretized execution path to target unexecuted blocks safely).
    RewriteReport rewriteEdge(
        FunDecl &caller, const FuncDescriptor &callerDesc, const StateProfile *callerProfile,
        const TypeUtils::StructTable &structs, const FunDecl &calleeFn,
        const FuncDescriptor &callee, std::size_t fixedRealizationIdx, std::mt19937 &rng
    );

    CallRealizePlan plan_;
    std::vector<std::unique_ptr<CallRewriteRule>> rules_;
    // Identity = (caller FunDecl pointer, site letIdx). Caller pointers
    // are stable across one rylink program (the bundle's funs vector is
    // reserved upfront — see rylink.cpp generateOne). Stored as a
    // std::set of the bare pair so two different (caller, letIdx)
    // combinations cannot collide on a hash — the previous version
    // packed both into a uint64 via shift+xor, which loses the top
    // bits of the pointer and could (very rarely) alias two distinct
    // sites in the same rylink run.
    std::set<std::pair<const FunDecl *, int>> consumed_;
  };

  // Build a CallRealizeTransform pre-loaded with LiteralToCallRule.
  // This is the entry point rylink adds to its pipeline.
  [[nodiscard]] std::unique_ptr<Transform> makeCallRealizeTransform(CallRealizePlan plan);

} // namespace refractir::reify
