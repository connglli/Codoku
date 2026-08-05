// Structured WASM body emission (--structured-lowering).
//
// WebAssembly has no goto; the default backend encodes arbitrary CFGs
// with a $__pc + br_table dispatch loop. When the CFG is reducible we
// can instead reconstruct genuine `block`/`loop`/`if` control flow —
// smaller, idiomatic, and directly optimizable by every WASM engine.
//
// The driver registers ReducibilityCheck whenever this mode is on, so
// every function reaching emitStructuredBody is reducible and the
// structuring pipeline is total. Unlike the C and Python structured
// emitters — which run StructuredLowering first because their targets
// have only single-level break/continue — WASM consumes the *unlowered*
// control tree: its native multi-level `br N` expresses every transfer
// directly, so no guard flags are ever needed.
//
// The mapping is the dominator-tree-to-WASM translation of Ramsey
// ("Beyond Relooper", ICFP 2022):
//   - each natural loop        -> a `(loop $__cont<h>)` whose label is
//                                 the loop's *continue* target;
//   - each pending join block  -> a `(block $__jn<b>)` whose end sits
//                                 right before the join's subtree, so a
//                                 forward `br $__jn<b>` lands there;
//   - Continue -> `br $__cont<header>`,  Break/JumpJoin -> `br $__jn<t>`,
//     FallThrough -> nothing (control falls out of the enclosing block).
// Named labels make the transfers' `levels` fields redundant: the WAT
// grammar resolves `$__jn<b>` / `$__cont<h>` to the right depth.
//
// A join's block scope wraps everything emitted *before* the join at its
// nesting level; because the Structurizer lays out a block's followers
// as the trailing items of the Seq it produces (items[0] is the block's
// own BlockStmts or its Loop node), those followers are exactly the Seq
// items past the "content" prefix. Statement-level emission (emitInstr /
// emitReturn / emitCond) is shared with the dispatch-loop emitter in
// wasm_backend.cpp.

#include <cassert>
#include <string>
#include "analysis/cfg.hpp"
#include "analysis/dominators.hpp"
#include "analysis/loop_info.hpp"
#include "analysis/structurizer.hpp"
#include "backend/wasm_backend.hpp"

namespace refractir {

  namespace {
    // Named br targets. Distinct prefixes keep loop-continue and join
    // labels apart; the block index guarantees uniqueness so a
    // multi-level `br` binds to the intended enclosing scope (labels
    // live in their own WAT index space, separate from locals).
    std::string contLabel(std::size_t header) { return "$__cont" + std::to_string(header); }

    std::string joinLabel(std::size_t block) { return "$__jn" + std::to_string(block); }

    // The entry block of a follower subtree — a Seq whose items[0] is
    // the follower's own BlockStmts (or its Loop node). Its index names
    // the follower's `$__jn` scope.
    std::size_t entryBlock(const ControlTree::Node &n) {
      if (auto *seq = std::get_if<ControlTree::Seq>(&n.v))
        return entryBlock(*seq->items.front());
      if (auto *bs = std::get_if<ControlTree::BlockStmts>(&n.v))
        return bs->block;
      if (auto *lp = std::get_if<ControlTree::Loop>(&n.v))
        return lp->header;
      assert(false && "follower subtree without a block/loop head");
      return 0;
    }
  } // namespace

  void WasmBackend::emitStructuredBody(const FunDecl &f) {
    DiagBag diags;
    CFG cfg = CFG::build(f, diags);
    DomTree dt = DomTree::build(cfg);
    LoopInfo li = LoopInfo::build(cfg, dt);
    ControlTree tree = Structurizer::build(f, cfg, dt, li);

    if (tree.root)
      emitCTreeNode(tree, *tree.root, f);

    // The Structurizer terminates every path in a Return/Trap (or an
    // infinite loop), so control never actually falls off the tree. But
    // a value-returning function's implicit end must still be
    // stack-valid under WASM validation; a trailing `unreachable` makes
    // it polymorphic and is dead code in practice.
    bool retVec = f.retType && std::holds_alternative<VecType>(f.retType->v);
    if (f.retType && !retVec) {
      indent();
      out_ << "unreachable\n";
    }
  }

  void WasmBackend::emitCTreeNode(
      const ControlTree &tree, const ControlTree::Node &n, const FunDecl &f
  ) {
    std::visit(
        [&](const auto &node) {
          using T = std::decay_t<decltype(node)>;
          if constexpr (std::is_same_v<T, ControlTree::Seq>) {
            const auto &items = node.items;
            const std::size_t n2 = items.size();
            // items[0] is the block's own content (a Loop node, or a
            // BlockStmts followed at items[1] by the lowered terminator);
            // everything past that prefix is a pending-join follower.
            std::size_t contentCount =
                std::holds_alternative<ControlTree::Loop>(items[0]->v) ? 1 : 2;
            if (contentCount > n2)
              contentCount = n2;

            // Open one `(block $__jn<f>)` per follower, outermost first.
            for (std::size_t j = n2; j-- > contentCount;) {
              indent();
              out_ << "(block " << joinLabel(entryBlock(*items[j])) << "\n";
              indent_level_++;
            }
            // Content is emitted inside the innermost join block.
            for (std::size_t j = 0; j < contentCount; ++j)
              emitCTreeNode(tree, *items[j], f);
            // Close each join block (innermost first) and emit the
            // follower's subtree right after its `end`, where forward
            // branches to it land.
            for (std::size_t j = contentCount; j < n2; ++j) {
              indent_level_--;
              indent();
              out_ << ")\n";
              emitCTreeNode(tree, *items[j], f);
            }
          } else if constexpr (std::is_same_v<T, ControlTree::BlockStmts>) {
            const Block &b = f.blocks[node.block];
            if (!b.instrs.empty()) {
              indent();
              out_ << ";; " << b.label.name << "\n";
            }
            for (const auto &ins: b.instrs)
              emitInstr(ins);
          } else if constexpr (std::is_same_v<T, ControlTree::If>) {
            const auto *br = std::get_if<BrTerm>(&f.blocks[node.block].term);
            assert(br && br->isConditional && br->cond && "If node on a non-branch block");
            emitCond(*br->cond);
            if (node.negate) {
              indent();
              out_ << "i32.eqz\n";
            }
            indent();
            out_ << "if\n";
            indent_level_++;
            if (node.thenBr)
              emitCTreeNode(tree, *node.thenBr, f);
            indent_level_--;
            indent();
            out_ << "else\n";
            indent_level_++;
            if (node.elseBr)
              emitCTreeNode(tree, *node.elseBr, f);
            indent_level_--;
            indent();
            out_ << "end\n";
          } else if constexpr (std::is_same_v<T, ControlTree::Loop>) {
            indent();
            out_ << "(loop " << contLabel(node.header) << "\n";
            indent_level_++;
            if (node.body)
              emitCTreeNode(tree, *node.body, f);
            indent_level_--;
            indent();
            out_ << ")\n";
          } else if constexpr (std::is_same_v<T, ControlTree::Break>) {
            // Leave `levels` loops and land at the join `target`; the
            // named label encodes the depth, so `levels` is unused.
            indent();
            out_ << "br " << joinLabel(node.target) << "\n";
          } else if constexpr (std::is_same_v<T, ControlTree::Continue>) {
            indent();
            out_ << "br " << contLabel(node.header) << "\n";
          } else if constexpr (std::is_same_v<T, ControlTree::JumpJoin>) {
            // A forward jump skipping intermediate joins: `br` to the
            // target's block exits every scope in between for free.
            indent();
            out_ << "br " << joinLabel(node.target) << "\n";
          } else if constexpr (std::is_same_v<T, ControlTree::FallThrough>) {
            // Control falls out of the enclosing join block into the
            // target's subtree — no instruction needed.
          } else if constexpr (std::is_same_v<T, ControlTree::Return>) {
            const auto *ret = std::get_if<RetTerm>(&f.blocks[node.block].term);
            assert(ret && "Return node on a block without ret");
            emitReturn(*ret, f);
          } else if constexpr (std::is_same_v<T, ControlTree::Trap>) {
            // Executing `unreachable` is UB; WASM's `unreachable` traps,
            // the closest analogue (and keeps the block's stack valid).
            indent();
            out_ << "unreachable\n";
          } else {
            // CondLoop/DoWhile/SetFlag/... are produced only by
            // StructuredLowering, which the WASM path never runs.
            assert(false && "lowering-only node reached the structured WASM emitter");
          }
        },
        n.v
    );
  }

} // namespace refractir
