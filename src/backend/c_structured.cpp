// Structured C body emission (--structured-lowering).
//
// The driver registers ReducibilityCheck whenever this mode is on, so
// every function reaching emitStructuredBody has a reducible CFG and
// the structuring pipeline below is total. The backend consumes the
// *lowered* control tree — single-level break/continue plus one-shot
// guard flags — rather than lowering JumpJoin back to goto: the whole
// point of the mode is goto-free output. Statement-level emission
// (emitInstr / emitRetTerm / emitCond) is shared with the default
// goto emitter in c_backend.cpp.

#include <cassert>
#include "analysis/dominators.hpp"
#include "analysis/loop_info.hpp"
#include "analysis/structured_lowering.hpp"
#include "backend/c_backend.hpp"
#include "c_internal.hpp"

namespace refractir {

  void CBackend::emitStructuredBody(const FunDecl &f) {
    DiagBag diags;
    CFG cfg = CFG::build(f, diags);
    DomTree dt = DomTree::build(cfg);
    LoopInfo li = LoopInfo::build(cfg, dt);
    ControlTree tree = StructuredLowering::run(Structurizer::build(f, cfg, dt, li), f, cfg);

    // One-shot guard flags for multi-level exits: false except between
    // their SetFlag and the final cascade/reset.
    for (const auto &flag: tree.flagNames) {
      indent();
      out_ << "bool " << flag << " = false;\n";
    }
    if (tree.root)
      emitCTreeNode(tree, *tree.root, f);
  }

  void CBackend::emitBranchCond(const FunDecl &f, std::size_t block, bool negate) {
    const auto *br = std::get_if<BrTerm>(&f.blocks[block].term);
    assert(br && br->isConditional && br->cond && "structured condition on a non-branch block");
    if (negate)
      out_ << "!(";
    emitCond(*br->cond);
    if (negate)
      out_ << ")";
  }

  void
  CBackend::emitCTreeNode(const ControlTree &tree, const ControlTree::Node &n, const FunDecl &f) {
    std::visit(
        [&](const auto &node) {
          using T = std::decay_t<decltype(node)>;
          if constexpr (std::is_same_v<T, ControlTree::Seq>) {
            for (const auto &item: node.items)
              if (item)
                emitCTreeNode(tree, *item, f);
          } else if constexpr (std::is_same_v<T, ControlTree::BlockStmts>) {
            const Block &b = f.blocks[node.block];
            if (!b.instrs.empty()) {
              indent();
              out_ << "// " << b.label.name << "\n";
            }
            for (const auto &ins: b.instrs)
              emitInstr(ins);
          } else if constexpr (std::is_same_v<T, ControlTree::If>) {
            indent();
            out_ << "if (";
            emitBranchCond(f, node.block, node.negate);
            out_ << ") {\n";
            indent_level_++;
            if (node.thenBr)
              emitCTreeNode(tree, *node.thenBr, f);
            indent_level_--;
            if (node.elseBr) {
              indent();
              out_ << "} else {\n";
              indent_level_++;
              emitCTreeNode(tree, *node.elseBr, f);
              indent_level_--;
            }
            indent();
            out_ << "}\n";
          } else if constexpr (std::is_same_v<T, ControlTree::Loop>) {
            indent();
            out_ << "for (;;) {\n";
            indent_level_++;
            if (node.body)
              emitCTreeNode(tree, *node.body, f);
            indent_level_--;
            indent();
            out_ << "}\n";
          } else if constexpr (std::is_same_v<T, ControlTree::CondLoop>) {
            indent();
            out_ << "while (";
            emitBranchCond(f, node.header, node.negate);
            out_ << ") {\n";
            indent_level_++;
            if (node.body)
              emitCTreeNode(tree, *node.body, f);
            indent_level_--;
            indent();
            out_ << "}\n";
          } else if constexpr (std::is_same_v<T, ControlTree::DoWhile>) {
            indent();
            out_ << "do {\n";
            indent_level_++;
            if (node.body)
              emitCTreeNode(tree, *node.body, f);
            indent_level_--;
            indent();
            out_ << "} while (";
            emitBranchCond(f, node.latch, node.negate);
            out_ << ");\n";
          } else if constexpr (std::is_same_v<T, ControlTree::Break>) {
            assert(node.levels == 1 && "multi-level break survived lowering");
            indent();
            out_ << "break;\n";
          } else if constexpr (std::is_same_v<T, ControlTree::Continue>) {
            assert(node.levels == 0 && "multi-level continue survived lowering");
            indent();
            out_ << "continue;\n";
          } else if constexpr (std::is_same_v<T, ControlTree::SetFlag>) {
            indent();
            out_ << tree.flagNames[node.flag] << " = true;\n";
          } else if constexpr (std::is_same_v<T, ControlTree::FlagBreak>) {
            const std::string &flag = tree.flagNames[node.flag];
            indent();
            if (node.isFinal) {
              out_ << "if (" << flag << ") {\n";
              indent_level_++;
              indent();
              out_ << flag << " = false;\n";
              indent();
              out_ << "break;\n";
              indent_level_--;
              indent();
              out_ << "}\n";
            } else {
              out_ << "if (" << flag << ") break;\n";
            }
          } else if constexpr (std::is_same_v<T, ControlTree::FlagContinue>) {
            const std::string &flag = tree.flagNames[node.flag];
            indent();
            out_ << "if (" << flag << ") {\n";
            indent_level_++;
            indent();
            out_ << flag << " = false;\n";
            indent();
            out_ << "continue;\n";
            indent_level_--;
            indent();
            out_ << "}\n";
          } else if constexpr (std::is_same_v<T, ControlTree::Guarded>) {
            indent();
            out_ << "if (";
            bool first = true;
            for (int fl: node.flags) {
              if (!first)
                out_ << " && ";
              out_ << "!" << tree.flagNames[fl];
              first = false;
            }
            out_ << ") {\n";
            indent_level_++;
            if (node.body)
              emitCTreeNode(tree, *node.body, f);
            indent_level_--;
            indent();
            out_ << "}\n";
          } else if constexpr (std::is_same_v<T, ControlTree::ResetFlag>) {
            indent();
            out_ << tree.flagNames[node.flag] << " = false;\n";
          } else if constexpr (std::is_same_v<T, ControlTree::Return>) {
            const auto *ret = std::get_if<RetTerm>(&f.blocks[node.block].term);
            assert(ret && "Return node on a block without ret");
            indent();
            emitRetTerm(*ret);
          } else if constexpr (std::is_same_v<T, ControlTree::Trap>) {
            // Executing `unreachable` is UB; trapping is the C target's
            // closest analogue (and keeps -Wreturn-type quiet where the
            // goto emitter's bare comment would not). Under --no-ub-guards
            // we assume the block is never reached and emit the pure
            // optimization hint instead (also -Wreturn-type-quiet).
            indent();
            out_
                << (noUbGuards_ ? "__builtin_unreachable(); // unreachable\n"
                                : "__builtin_trap(); // unreachable\n");
          } else {
            // FallThrough / JumpJoin are eliminated by StructuredLowering.
            assert(false && "unlowered transfer node reached the structured C emitter");
          }
        },
        n.v
    );
  }

} // namespace refractir
