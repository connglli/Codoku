#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "analysis/cfg.hpp"
#include "ast/ast.hpp"
#include "frontend/lexer.hpp"
#include "frontend/parser.hpp"

namespace refractir {

  /**
   * @brief Generic AST visitor for RefractIR functions.
   * Walks the AST of a function definition and provides virtual hooks for subclass handlers.
   * This class centralizes AST traversal logic to prevent duplicate walking code in analysis tools.
   */
  class ASTVisitor {
  public:
    virtual ~ASTVisitor() = default;

    /**
     * @brief Traverses the entire AST of the given function declaration.
     * Walks local declarations (lets), instructions, and basic block terminators.
     */
    void visitFun(const FunDecl &f) {
      for (const auto &l: f.lets) {
        if (l.init) {
          visitInitVal(*l.init);
        }
      }
      for (const auto &b: f.blocks) {
        for (const auto &ins: b.instrs) {
          // Skip require/assume instructions entirely: both are dropped from
          // puzzles and ground truth (they are pure assertions, not part of the
          // observable computation), so analyses must ignore them too.
          if (std::holds_alternative<RequireInstr>(ins) ||
              std::holds_alternative<AssumeInstr>(ins)) {
            continue;
          }
          std::visit([this](auto &&arg) { visitInstr(arg); }, ins);
        }
        std::visit([this](auto &&arg) { visitTerminator(arg); }, b.term);
      }
    }

    /**
     * @brief Walks initialization values and dispatches recursively.
     */
    virtual void visitInitVal(const InitVal &iv) {
      switch (iv.kind) {
        case InitVal::Kind::Int:
          visitIntLit(std::get<IntLit>(iv.value));
          break;
        case InitVal::Kind::Float:
          visitFloatLit(std::get<FloatLit>(iv.value));
          break;
        case InitVal::Kind::Aggregate: {
          for (const auto &elem: std::get<std::vector<InitValPtr>>(iv.value)) {
            visitInitVal(*elem);
          }
          break;
        }
        case InitVal::Kind::Atom:
          visitAtom(*std::get<AtomPtr>(iv.value));
          break;
        default:
          break;
      }
    }

    /**
     * @brief Virtual hooks for processing integer and floating-point literals.
     */
    virtual void visitIntLit(const IntLit &) {}

    virtual void visitFloatLit(const FloatLit &) {}

    /**
     * @brief Dispatches assignment, assume, require, and store instructions.
     */
    virtual void visitInstr(const AssignInstr &ins) {
      visitLValue(ins.lhs);
      visitExpr(ins.rhs);
    }

    virtual void visitInstr(const AssumeInstr &ins) { visitCond(ins.cond); }

    virtual void visitInstr(const RequireInstr &ins) { visitCond(ins.cond); }

    virtual void visitInstr(const StoreInstr &ins) {
      visitExpr(ins.ptr);
      visitExpr(ins.val);
    }

    /**
     * @brief Dispatches branch, return, and unreachable terminators.
     */
    virtual void visitTerminator(const BrTerm &term) {
      if (term.isConditional) {
        visitCond(*term.cond);
      }
    }

    virtual void visitTerminator(const RetTerm &term) {
      if (term.value) {
        visitExpr(*term.value);
      }
    }

    virtual void visitTerminator(const UnreachableTerm &) {}

    /**
     * @brief Walks an expression consisting of atoms combined by operators.
     */
    virtual void visitExpr(const Expr &e) {
      visitAtom(e.first);
      for (const auto &t: e.rest) {
        visitAtom(t.atom);
      }
    }

    /**
     * @brief Walks relational comparisons between two expressions.
     */
    virtual void visitCond(const Cond &c) {
      visitExpr(c.lhs);
      visitExpr(c.rhs);
    }

    /**
     * @brief Walks lvalue definitions and their index/field accesses.
     */
    virtual void visitLValue(const LValue &lv) {
      for (const auto &acc: lv.accesses) {
        if (auto ai = std::get_if<AccessIndex>(&acc)) {
          visitIndex(ai->index);
        }
      }
    }

    /**
     * @brief Walks indexing operations.
     */
    virtual void visitIndex(const Index &idx) {
      if (auto ilit = std::get_if<IntLit>(&idx)) {
        visitIntLit(*ilit);
      }
    }

    /**
     * @brief Walks numerical and symbol coefficients.
     */
    virtual void visitCoef(const Coef &c) {
      if (auto ilit = std::get_if<IntLit>(&c)) {
        visitIntLit(*ilit);
      } else if (auto flit = std::get_if<FloatLit>(&c)) {
        visitFloatLit(*flit);
      }
    }

    /**
     * @brief Walks values passed as branch conditions or select options.
     */
    virtual void visitSelectVal(const SelectVal &sv) {
      if (auto c = std::get_if<Coef>(&sv)) {
        visitCoef(*c);
      } else if (auto lv = std::get_if<RValue>(&sv)) {
        visitLValue(*lv);
      }
    }

    /**
     * @brief Dispatches the various RefractIR Atom structures recursively.
     */
    virtual void visitAtom(const Atom &a) {
      std::visit(
          [this](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, OpAtom>) {
              visitCoef(arg.coef);
              visitLValue(arg.rval);
            } else if constexpr (std::is_same_v<T, SelectAtom>) {
              if (arg.cond)
                visitCond(*arg.cond);
              if (arg.maskExpr)
                visitExpr(*arg.maskExpr);
              visitSelectVal(arg.vtrue);
              visitSelectVal(arg.vfalse);
            } else if constexpr (std::is_same_v<T, CoefAtom>) {
              visitCoef(arg.coef);
            } else if constexpr (std::is_same_v<T, RValueAtom>) {
              visitLValue(arg.rval);
            } else if constexpr (std::is_same_v<T, CastAtom>) {
              std::visit(
                  [this](auto &&src) {
                    using S = std::decay_t<decltype(src)>;
                    if constexpr (std::is_same_v<S, IntLit>) {
                      visitIntLit(src);
                    } else if constexpr (std::is_same_v<S, FloatLit>) {
                      visitFloatLit(src);
                    } else if constexpr (std::is_same_v<S, LValue>) {
                      visitLValue(src);
                    }
                  },
                  arg.src
              );
            } else if constexpr (std::is_same_v<T, UnaryAtom>) {
              visitLValue(arg.rval);
            } else if constexpr (std::is_same_v<T, AddrAtom>) {
              visitLValue(arg.lv);
            } else if constexpr (std::is_same_v<T, LoadAtom>) {
              visitLValue(arg.rval);
            } else if constexpr (std::is_same_v<T, CmpAtom>) {
              visitSelectVal(arg.lhs);
              visitSelectVal(arg.rhs);
            } else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
              visitLValue(arg.rval);
              visitIndex(arg.index);
            } else if constexpr (std::is_same_v<T, PtrFieldAtom>) {
              visitLValue(arg.rval);
            } else if constexpr (std::is_same_v<T, CallAtom>) {
              visitCall(arg);
              for (const auto &arg_expr: arg.args) {
                visitExpr(*arg_expr);
              }
            }
          },
          a.v
      );
    }

    /**
     * @brief Virtual hook to capture function or intrinsic calls.
     */
    virtual void visitCall(const CallAtom &) {}
  };

  /**
   * @brief Collects the exact multiset of constants that masking renders as
   * `FILL_CONST`, i.e. the constants the solver must distribute across the
   * `FILL_CONST` slots of a puzzle.
   *
   * This is *not* a "magic number" collector: it tracks **every** integer,
   * floating-point and `null` constant (no magnitude threshold), but only at
   * the positions that actually become `FILL_CONST` in the masked output:
   *
   *  - inside masked basic blocks (strictly between the first/entry and the
   *    last/exit block), skipping dropped `require`/`assume` instructions;
   *  - inside masked `let` initialisers (every constant except the `0`/`1`
   *    sentinels the printer leaves visible, and except `_`-prefixed scratch
   *    locals whose initialisers are never masked).
   *
   * Constants absorbed into an `FILL_VAR` (an lvalue base or its subscript /
   * field accesses) are deliberately *not* counted, because the printer masks
   * the whole lvalue to a single `FILL_VAR` — so `visitLValue` is a no-op here.
   *
   * Keeping the producer (rypuzmk header) and the consumer (rypuzchk check) on
   * this single definition guarantees they stay in sync.
   */
  struct MaskedConstantCollector : public ASTVisitor {
    // value-string -> number of FILL_CONST slots that must carry that value
    std::unordered_map<std::string, int> counts;

    // Selective masking (the `--p-mask` knob). nullptr => every maskable
    // statement is masked (the default, full-masking behaviour). Otherwise a
    // statement is masked iff its canonical index (see collect()) is in the
    // set; constants in *un*masked statements stay on screen and are therefore
    // excluded from the budget. Producer (rypuzmk) and consumer (rypuzchk) must
    // pass the identical set, derived by inferMaskSetFromPuzzle() in rypuzchk.
    const std::unordered_set<int> *selectiveMask = nullptr;

    // Whole lvalues mask to a single FILL_VAR; their bases/subscripts/fields
    // never surface as FILL_CONST, so do not descend into them.
    void visitLValue(const LValue &) override {}

    void visitIntLit(const IntLit &ilit) override { counts[std::to_string(ilit.value)]++; }

    void visitFloatLit(const FloatLit &flit) override { counts[formatDouble(flit.value)]++; }

    // A bare `null` coefficient also masks to FILL_CONST.
    void visitCoef(const Coef &c) override {
      if (auto i = std::get_if<IntLit>(&c)) {
        visitIntLit(*i);
      } else if (auto f = std::get_if<FloatLit>(&c)) {
        visitFloatLit(*f);
      } else if (std::holds_alternative<NullLit>(c)) {
        counts["null"]++;
      }
      // LocalOrSymId coefficients mask to FILL_VAR — not counted.
    }

    /**
     * @brief Walk a leaf function and collect its FILL_CONST budget.
     *
     * The walk visits maskable statements in the canonical order shared with
     * SIRMaskedPrinter and countMaskableStatements(): every `let` initialiser
     * (in declaration order), then, for each body block in order, its
     * instructions (skipping the dropped require/assume) followed by its
     * terminator. Each such position consumes one index; with `selectiveMask`
     * set, only positions in the set contribute constants.
     */
    void collect(const FunDecl &f) {
      int idx = 0;
      // Masked let initialisers.
      for (const auto &l: f.lets) {
        if (l.init) {
          if (isMasked(idx++)) {
            collectLetInit(*l.init, l.name.name);
          }
        }
      }
      // Body blocks carry maskable statements: strictly between entry (first)
      // and exit (last). Whether each statement is actually masked is decided
      // per-position by isMasked(idx).
      for (size_t i = 0; i < f.blocks.size(); ++i) {
        bool blockBody = (i > 0) && (i + 1 < f.blocks.size());
        if (!blockBody) {
          continue;
        }
        const auto &b = f.blocks[i];
        for (const auto &ins: b.instrs) {
          if (std::holds_alternative<RequireInstr>(ins) ||
              std::holds_alternative<AssumeInstr>(ins)) {
            continue;
          }
          if (isMasked(idx++)) {
            std::visit([this](auto &&arg) { visitInstr(arg); }, ins);
          }
        }
        if (isMasked(idx++)) {
          std::visit([this](auto &&arg) { visitTerminator(arg); }, b.term);
        }
      }
    }

  private:
    // Whether the maskable statement at canonical index `idx` is masked.
    bool isMasked(int idx) const { return selectiveMask ? (selectiveMask->count(idx) > 0) : true; }

    // Mirror SIRMaskedPrinter::printMaskedInitVal: `_`-prefixed locals keep
    // their initialisers verbatim, and the `0`/`1` sentinels stay visible.
    void collectLetInit(const InitVal &iv, const std::string &varName) {
      if (varName.rfind("_", 0) == 0) {
        return;
      }
      switch (iv.kind) {
        case InitVal::Kind::Int: {
          int64_t v = std::get<IntLit>(iv.value).value;
          if (v != 0 && v != 1) {
            counts[std::to_string(v)]++;
          }
          break;
        }
        case InitVal::Kind::Float: {
          double v = std::get<FloatLit>(iv.value).value;
          if (v != 0.0 && v != 1.0) {
            counts[formatDouble(v)]++;
          }
          break;
        }
        case InitVal::Kind::Aggregate:
          for (const auto &elem: std::get<std::vector<InitValPtr>>(iv.value)) {
            collectLetInit(*elem, varName);
          }
          break;
        case InitVal::Kind::Atom:
          // Atom-form initialisers are masked exactly like block expressions.
          visitAtom(*std::get<AtomPtr>(iv.value));
          break;
        default:
          break;
      }
    }
  };

  /**
   * @brief Counts the maskable statements of a leaf function, in the canonical
   * order shared by SIRMaskedPrinter and MaskedConstantCollector.
   *
   * A *maskable statement* is one of: a `let` initialiser, a body-block
   * instruction (excluding the dropped require/assume), or a body-block
   * terminator -- exactly the positions that turn into FILL_XXX under full
   * masking. `rypuzmk` flips one coin per position to build the `--p-mask`
   * masked set; this is the size of the canonical index space.
   */
  inline int countMaskableStatements(const FunDecl &leaf) {
    int n = 0;
    for (const auto &l: leaf.lets) {
      if (l.init) {
        ++n;
      }
    }
    for (size_t i = 0; i < leaf.blocks.size(); ++i) {
      bool blockBody = (i > 0) && (i + 1 < leaf.blocks.size());
      if (!blockBody) {
        continue;
      }
      for (const auto &ins: leaf.blocks[i].instrs) {
        if (std::holds_alternative<RequireInstr>(ins) || std::holds_alternative<AssumeInstr>(ins)) {
          continue;
        }
        ++n;
      }
      ++n; // terminator
    }
    return n;
  }

  /**
   * @brief Concrete walker to collect names of all called functions and intrinsics.
   */
  struct CallCollector : public ASTVisitor {
    // Set of all function/intrinsic names that are called in the function
    std::unordered_set<std::string> calledFunctions;

    /**
     * @brief Entry point to walk and collect function calls from a function definition.
     */
    void collect(const FunDecl &f) { visitFun(f); }

    /**
     * @brief Override to record the name of the callee when a function call is encountered.
     */
    void visitCall(const CallAtom &call) override { calledFunctions.insert(call.callee.name); }
  };

  /**
   * @brief Reads the entire contents of a file into a string.
   * Throws std::runtime_error on failure.
   */
  inline std::string readFile(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) {
      throw std::runtime_error("Could not open file: " + path);
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
  }

  /**
   * @brief Parse RefractIR source code from a string into a Program AST.
   */
  inline Program parseProgramFromString(const std::string &src) {
    Lexer lx(src);
    auto toks = lx.lexAll();
    Parser ps(std::move(toks));
    return ps.parseProgram();
  }

  /**
   * @brief Custom printer class to serialize a RefractIR program back
   * to string representation, with optional masking enabled for the
   * intermediate blocks and lets in a designated leaf function.
   */
  class SIRMaskedPrinter {
  public:
    /**
     * @param selectiveMask The `--p-mask` masked set (see
     * MaskedConstantCollector::selectiveMask). nullptr => mask every maskable
     * statement (full masking). Only consulted when `enableMasking` is set.
     * @param sentinelMode When true, emits '\x01' immediately before the
     * first character of each maskable position's contribution to the output
     * (the let initialiser value or the instruction/terminator tokens). This
     * never appears in real RefractIR source and is used by
     * inferMaskSetFromPuzzle() to split the output into per-position segments
     * without parsing the puzzle (which contains unparseable FILL_XXX tokens).
     */
    explicit SIRMaskedPrinter(
        std::ostream &out, const FunDecl &leafFunc, bool enableMasking = true,
        const std::unordered_set<int> *selectiveMask = nullptr, bool sentinelMode = false
    ) :
        out_(out), leafFunc_(&leafFunc), enableMasking_(enableMasking),
        selectiveMask_(selectiveMask), sentinelMode_(sentinelMode) {}

    /**
     * @brief Prints the entire RefractIR Program, masking the leaf function as required.
     */
    void print(const Program &p) {
      // 1. Struct declarations (never masked)
      for (const auto &s: p.structs) {
        out_ << "struct " << s.name.name << " {\n";
        indent_level_++;
        for (const auto &f: s.fields) {
          indent();
          out_ << f.name << ": ";
          printType(f.type);
          out_ << ";\n";
        }
        indent_level_--;
        out_ << "} \n\n";
      }

      // 2. Functions (leaf function gets its body masked)
      for (const auto &f: p.funs) {
        bool isLeaf = (&f == leafFunc_);
        out_ << "fun " << f.name.name << "(";
        for (size_t i = 0; i < f.params.size(); ++i) {
          out_ << f.params[i].name.name << ": ";
          printType(f.params[i].type);
          if (i + 1 < f.params.size())
            out_ << ", ";
        }
        out_ << ") : ";
        printType(f.retType);
        out_ << " {\n";
        indent_level_++;

        // 2a. Print local lets
        for (const auto &l: f.lets) {
          indent();
          out_ << "let " << (l.isMutable ? "mut " : "") << l.name.name << ": ";
          printType(l.type);
          if (l.init) {
            out_ << " = ";
            // Each leaf let initialiser is a maskable position. In masking mode
            // takeLeafMask() advances the index and decides masked/revealed.
            // In sentinel mode the '\x01' marks the position boundary so that
            // inferMaskSetFromPuzzle() can split the output per-position.
            if (isLeaf && enableMasking_) {
              bool doMask = takeLeafMask();
              if (sentinelMode_)
                out_ << '\x01';
              if (doMask) {
                printMaskedInitVal(*l.init, l.name.name);
              } else {
                printInitVal(*l.init);
              }
            } else {
              if (isLeaf && sentinelMode_)
                out_ << '\x01';
              printInitVal(*l.init);
            }
          }
          out_ << ";\n";
        }

        // 2b. Print CFG blocks
        for (size_t i = 0; i < f.blocks.size(); ++i) {
          const auto &b = f.blocks[i];
          out_ << b.label.name << ":\n";

          // Only blocks of the leaf strictly between entry (0) and exit (last)
          // carry maskable statements. `blockBody` captures this structurally;
          // `blockMaskable` also requires masking to be enabled.
          bool blockBody = isLeaf && (i > 0) && (i < f.blocks.size() - 1);
          bool blockMaskable = blockBody && enableMasking_;

          for (const auto &ins: b.instrs) {
            // Drop require/assume from both the puzzle and the ground truth.
            // They are pure assertions (path conditions / properties) and not
            // part of the observable computation the solver must reconstruct.
            if (std::holds_alternative<RequireInstr>(ins) ||
                std::holds_alternative<AssumeInstr>(ins)) {
              continue;
            }
            masked_mode_ = blockMaskable && takeLeafMask();
            if (blockBody && sentinelMode_) {
              // Sentinel before indentation so stripping whitespace still leaves
              // '\x01' as the first character of this position's stripped segment.
              out_ << '\x01';
            }
            indent();
            std::visit([this](auto &&arg) { printInstr(arg); }, ins);
            out_ << ";\n";
          }
          masked_mode_ = blockMaskable && takeLeafMask();
          if (blockBody && sentinelMode_)
            out_ << '\x01';
          indent();
          std::visit([this](auto &&arg) { printTerminator(arg); }, b.term);
          out_ << ";\n";
        }

        indent_level_--;
        out_ << "} \n\n";
      }

      // 3. External Declarations
      for (const auto &d: p.extDecls) {
        out_ << "decl " << d.name.name << "(";
        for (size_t i = 0; i < d.params.size(); ++i) {
          out_ << d.params[i].name.name << ": ";
          printType(d.params[i].type);
          if (i + 1 < d.params.size())
            out_ << ", ";
        }
        out_ << ") : ";
        printType(d.retType);
        if (!d.contract) {
          out_ << ";\n\n";
          continue;
        }
        out_ << " {\n";
        indent_level_++;
        for (const auto &pre: d.contract->pres) {
          indent();
          out_ << "pre ";
          printCond(pre.cond);
          if (pre.message)
            out_ << ", \"" << *pre.message << "\"";
          out_ << ";\n";
        }
        for (const auto &post: d.contract->posts) {
          indent();
          out_ << "post ";
          printCond(post.cond);
          if (post.message)
            out_ << ", \"" << *post.message << "\"";
          out_ << ";\n";
        }
        indent_level_--;
        out_ << "};\n\n";
      }

      // 4. Intrinsics
      for (const auto &in: p.intrinsics) {
        out_ << "intrinsic " << in.name.name << "(";
        for (size_t i = 0; i < in.params.size(); ++i) {
          out_ << in.params[i].name.name << ": ";
          printType(in.params[i].type);
          if (i + 1 < in.params.size())
            out_ << ", ";
        }
        out_ << ") : ";
        printType(in.retType);
        out_ << ";\n\n";
      }
    }

  private:
    std::ostream &out_;
    const FunDecl *leafFunc_;
    bool enableMasking_ = true;
    const std::unordered_set<int> *selectiveMask_ = nullptr;
    // When true, emit '\x01' before each maskable position's content so that
    // inferMaskSetFromPuzzle() can split the output into per-position segments.
    bool sentinelMode_ = false;
    // Canonical index of the next maskable leaf statement; advanced exactly once
    // per maskable position (let initialiser, then per body statement) so it
    // stays aligned with MaskedConstantCollector.
    int leafStmtIdx_ = 0;
    int indent_level_ = 0;
    bool masked_mode_ = false;

    // Decision for the next maskable leaf statement; advances leafStmtIdx_.
    // nullptr selectiveMask_ => mask everything (full masking).
    bool takeLeafMask() {
      bool m = selectiveMask_ ? (selectiveMask_->count(leafStmtIdx_) > 0) : true;
      ++leafStmtIdx_;
      return m;
    }

    void indent() {
      for (int i = 0; i < indent_level_; ++i)
        out_ << "  ";
    }

    void printType(const TypePtr &t) {
      if (!t)
        return;
      if (masked_mode_) {
        out_ << "FILL_TYPE";
        return;
      }
      std::visit(
          [this](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, IntType>) {
              switch (arg.kind) {
                case IntType::Kind::I32:
                  out_ << "i32";
                  break;
                case IntType::Kind::I64:
                  out_ << "i64";
                  break;
                case IntType::Kind::ICustom:
                  out_ << "i" << (arg.bits ? std::to_string(*arg.bits) : "?");
                  break;
              }
            } else if constexpr (std::is_same_v<T, FloatType>) {
              switch (arg.kind) {
                case FloatType::Kind::F32:
                  out_ << "f32";
                  break;
                case FloatType::Kind::F64:
                  out_ << "f64";
                  break;
              }
            } else if constexpr (std::is_same_v<T, StructType>) {
              out_ << arg.name.name;
            } else if constexpr (std::is_same_v<T, ArrayType>) {
              out_ << "[" << arg.size << "] ";
              printType(arg.elem);
            } else if constexpr (std::is_same_v<T, PtrType>) {
              out_ << "ptr ";
              printType(arg.pointee);
            } else if constexpr (std::is_same_v<T, VecType>) {
              out_ << "<" << arg.size << "> ";
              printType(arg.elem);
            }
          },
          t->v
      );
    }

    void printInstr(const AssignInstr &ins) {
      printLValue(ins.lhs);
      out_ << " = ";
      printExpr(ins.rhs);
    }

    void printInstr(const AssumeInstr &ins) {
      out_ << "assume ";
      printCond(ins.cond);
    }

    void printInstr(const RequireInstr &ins) {
      out_ << "require ";
      printCond(ins.cond);
      if (ins.message)
        out_ << ", \"" << *ins.message << "\"";
    }

    void printInstr(const StoreInstr &ins) {
      if (masked_mode_) {
        out_ << "FILL_OP ";
      } else {
        out_ << "store ";
      }
      printExpr(ins.ptr);
      out_ << ", ";
      printExpr(ins.val);
    }

    void printTerminator(const BrTerm &term) {
      out_ << "br ";
      if (term.isConditional) {
        printCond(*term.cond);
        out_ << ", ";
        if (masked_mode_) {
          out_ << "FILL_LABEL, FILL_LABEL";
        } else {
          out_ << term.thenLabel.name << ", " << term.elseLabel.name;
        }
      } else {
        if (masked_mode_) {
          out_ << "FILL_LABEL";
        } else {
          out_ << term.dest.name;
        }
      }
    }

    void printTerminator(const RetTerm &term) {
      out_ << "ret";
      if (term.value) {
        out_ << " ";
        printExpr(*term.value);
      }
    }

    void printTerminator(const UnreachableTerm &) { out_ << "unreachable"; }

    void printExpr(const Expr &e) {
      printAtom(e.first);
      for (const auto &t: e.rest) {
        if (masked_mode_) {
          out_ << " FILL_OP ";
        } else {
          out_ << (t.op == AddOp::Plus ? " + " : " - ");
        }
        printAtom(t.atom);
      }
    }

    void printCond(const Cond &c) {
      printExpr(c.lhs);
      if (masked_mode_) {
        out_ << " FILL_OP ";
      } else {
        out_ << " " << relOpToString(c.op) << " ";
      }
      printExpr(c.rhs);
    }

    void printAtom(const Atom &a) {
      std::visit(
          [this](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, OpAtom>) {
              printCoef(arg.coef);
              if (masked_mode_) {
                out_ << " FILL_OP ";
              } else {
                out_ << " " << atomOpToString(arg.op) << " ";
              }
              printLValue(arg.rval);
            } else if constexpr (std::is_same_v<T, SelectAtom>) {
              if (masked_mode_) {
                out_ << "FILL_OP ";
              } else {
                out_ << "select ";
              }
              if (arg.cond)
                printCond(*arg.cond);
              else if (arg.maskExpr)
                printExpr(*arg.maskExpr);
              out_ << ", ";
              printSelectVal(arg.vtrue);
              out_ << ", ";
              printSelectVal(arg.vfalse);
            } else if constexpr (std::is_same_v<T, CoefAtom>) {
              printCoef(arg.coef);
            } else if constexpr (std::is_same_v<T, RValueAtom>) {
              printLValue(arg.rval);
            } else if constexpr (std::is_same_v<T, UnaryAtom>) {
              if (masked_mode_) {
                out_ << "FILL_OP ";
              } else {
                if (arg.op == UnaryOpKind::Not)
                  out_ << "~";
              }
              printLValue(arg.rval);
            } else if constexpr (std::is_same_v<T, CastAtom>) {
              std::visit(
                  [&](auto &&src) {
                    using S = std::decay_t<decltype(src)>;
                    if constexpr (std::is_same_v<S, IntLit>) {
                      if (masked_mode_)
                        out_ << "FILL_CONST";
                      else
                        out_ << src.value;
                    } else if constexpr (std::is_same_v<S, FloatLit>) {
                      if (masked_mode_)
                        out_ << "FILL_CONST";
                      else
                        out_ << formatDouble(src.value);
                    } else if constexpr (std::is_same_v<S, SymId>) {
                      if (masked_mode_)
                        out_ << "FILL_VAR";
                      else
                        out_ << src.name;
                    } else {
                      printLValue(src);
                    }
                  },
                  arg.src
              );
              if (masked_mode_) {
                out_ << " FILL_OP FILL_TYPE";
              } else {
                out_ << " as ";
                printType(arg.dstType);
              }
            } else if constexpr (std::is_same_v<T, AddrAtom>) {
              if (masked_mode_)
                out_ << "FILL_OP ";
              else
                out_ << "addr ";
              printLValue(arg.lv);
            } else if constexpr (std::is_same_v<T, LoadAtom>) {
              if (masked_mode_)
                out_ << "FILL_OP ";
              else
                out_ << "load ";
              printLValue(arg.rval);
            } else if constexpr (std::is_same_v<T, CmpAtom>) {
              if (masked_mode_) {
                out_ << "FILL_OP ";
              } else {
                out_ << "cmp " << relOpToString(arg.op) << " ";
              }
              printSelectVal(arg.lhs);
              out_ << ", ";
              printSelectVal(arg.rhs);
            } else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
              if (masked_mode_) {
                out_ << "FILL_OP ";
              } else {
                out_ << "ptrindex ";
              }
              printLValue(arg.rval);
              out_ << ", ";
              printIndex(arg.index);
            } else if constexpr (std::is_same_v<T, PtrFieldAtom>) {
              if (masked_mode_) {
                out_ << "FILL_OP ";
              } else {
                out_ << "ptrfield ";
              }
              printLValue(arg.rval);
              out_ << ", ";
              if (masked_mode_) {
                out_ << "FILL_FIELD";
              } else {
                out_ << arg.field;
              }
            } else if constexpr (std::is_same_v<T, CallAtom>) {
              if (masked_mode_) {
                out_ << "call FILL_FUNC(";
              } else {
                out_ << "call " << arg.callee.name << "(";
              }
              for (size_t i = 0; i < arg.args.size(); ++i) {
                if (i)
                  out_ << ", ";
                printExpr(*arg.args[i]);
              }
              out_ << ")";
            }
          },
          a.v
      );
    }

    void printLValue(const LValue &lv) {
      if (masked_mode_) {
        out_ << "FILL_VAR";
        return;
      }
      out_ << lv.base.name;
      for (const auto &acc: lv.accesses) {
        if (auto ai = std::get_if<AccessIndex>(&acc)) {
          out_ << "[";
          printIndex(ai->index);
          out_ << "]";
        } else if (auto af = std::get_if<AccessField>(&acc)) {
          out_ << "." << af->field;
        }
      }
    }

    void printCoef(const Coef &c) {
      std::visit(
          [this](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, IntLit>) {
              if (masked_mode_)
                out_ << "FILL_CONST";
              else
                out_ << arg.value;
            } else if constexpr (std::is_same_v<T, FloatLit>) {
              if (masked_mode_)
                out_ << "FILL_CONST";
              else
                out_ << formatDouble(arg.value);
            } else if constexpr (std::is_same_v<T, NullLit>) {
              if (masked_mode_)
                out_ << "FILL_CONST";
              else
                out_ << "null";
            } else {
              std::visit(
                  [this](auto &&id) {
                    if (masked_mode_)
                      out_ << "FILL_VAR";
                    else
                      out_ << id.name;
                  },
                  arg
              );
            }
          },
          c
      );
    }

    void printIndex(const Index &idx) {
      std::visit(
          [this](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, IntLit>) {
              if (masked_mode_)
                out_ << "FILL_CONST";
              else
                out_ << arg.value;
            } else {
              std::visit(
                  [this](auto &&id) {
                    if (masked_mode_)
                      out_ << "FILL_VAR";
                    else
                      out_ << id.name;
                  },
                  arg
              );
            }
          },
          idx
      );
    }

    void printSelectVal(const SelectVal &sv) {
      if (std::holds_alternative<RValue>(sv)) {
        printLValue(std::get<RValue>(sv));
      } else {
        printCoef(std::get<Coef>(sv));
      }
    }

    void printMaskedInitVal(const InitVal &iv, const std::string &varName) {
      if (varName.rfind("_", 0) == 0) {
        printInitVal(iv);
        return;
      }

      switch (iv.kind) {
        case InitVal::Kind::Int: {
          int64_t val = std::get<IntLit>(iv.value).value;
          if (val == 0 || val == 1) {
            out_ << val;
          } else {
            out_ << "FILL_CONST";
          }
          break;
        }
        case InitVal::Kind::Float: {
          double val = std::get<FloatLit>(iv.value).value;
          if (val == 0.0 || val == 1.0) {
            out_ << formatDouble(val);
          } else {
            out_ << "FILL_CONST";
          }
          break;
        }
        case InitVal::Kind::Aggregate: {
          out_ << "{";
          const auto &elements = std::get<std::vector<InitValPtr>>(iv.value);
          for (size_t i = 0; i < elements.size(); ++i) {
            printMaskedInitVal(*elements[i], varName);
            if (i + 1 < elements.size())
              out_ << ", ";
          }
          out_ << "}";
          break;
        }
        case InitVal::Kind::Atom: {
          bool old_mode = masked_mode_;
          masked_mode_ = true;
          printAtom(*std::get<AtomPtr>(iv.value));
          masked_mode_ = old_mode;
          break;
        }
        default:
          printInitVal(iv);
          break;
      }
    }

    void printInitVal(const InitVal &iv) {
      switch (iv.kind) {
        case InitVal::Kind::Int:
          out_ << std::get<IntLit>(iv.value).value;
          break;
        case InitVal::Kind::Float:
          out_ << formatDouble(std::get<FloatLit>(iv.value).value);
          break;
        case InitVal::Kind::Sym:
          out_ << std::get<SymId>(iv.value).name;
          break;
        case InitVal::Kind::Local:
          out_ << std::get<LocalId>(iv.value).name;
          break;
        case InitVal::Kind::Undef:
          out_ << "undef";
          break;
        case InitVal::Kind::Null:
          out_ << "null";
          break;
        case InitVal::Kind::Aggregate: {
          out_ << "{";
          const auto &elements = std::get<std::vector<InitValPtr>>(iv.value);
          for (size_t i = 0; i < elements.size(); ++i) {
            printInitVal(*elements[i]);
            if (i + 1 < elements.size())
              out_ << ", ";
          }
          out_ << "}";
          break;
        }
        case InitVal::Kind::Atom:
          printAtom(*std::get<AtomPtr>(iv.value));
          break;
      }
    }

    std::string relOpToString(RelOp op) {
      switch (op) {
        case RelOp::EQ:
          return "==";
        case RelOp::NE:
          return "!=";
        case RelOp::LT:
          return "<";
        case RelOp::LE:
          return "<=";
        case RelOp::GT:
          return ">";
        case RelOp::GE:
          return ">=";
      }
      return "?";
    }

    std::string atomOpToString(AtomOpKind op) {
      switch (op) {
        case AtomOpKind::Mul:
          return "*";
        case AtomOpKind::Div:
          return "/";
        case AtomOpKind::Mod:
          return "%";
        case AtomOpKind::And:
          return "&";
        case AtomOpKind::Or:
          return "|";
        case AtomOpKind::Xor:
          return "^";
        case AtomOpKind::Shl:
          return "<<";
        case AtomOpKind::Shr:
          return ">>";
        case AtomOpKind::LShr:
          return ">>>";
      }
      return "?";
    }
  };

  /**
   * @brief Helper to locate the leaf function inside a Program (non-main function).
   */
  inline const FunDecl *findLeafFunction(const Program &prog) {
    for (const auto &f: prog.funs) {
      if (f.name.name != "@main") {
        return &f;
      }
    }
    return nullptr;
  }

  /**
   * @brief Helper to locate the `@main` wrapper function inside a Program.
   */
  inline const FunDecl *findMainFunction(const Program &prog) {
    for (const auto &f: prog.funs) {
      if (f.name.name == "@main") {
        return &f;
      }
    }
    return nullptr;
  }

  /**
   * @brief Validates that a leaf function has the shape the puzzle masker
   * assumes: at least three basic blocks, the first named `^entry`, and the
   * last named `^exit`. Masking is keyed off block position, so a malformed
   * shape would silently produce a degenerate (or wrongly masked) puzzle.
   * @return empty string if well-formed, otherwise a human-readable reason.
   */
  inline std::string validateLeafShape(const FunDecl &leaf) {
    if (leaf.blocks.size() < 3) {
      return "leaf function must have >= 3 basic blocks (entry, >=1 body, exit)";
    }
    if (leaf.blocks.front().label.name != "^entry") {
      return "first block of leaf function must be ^entry, got " + leaf.blocks.front().label.name;
    }
    if (leaf.blocks.back().label.name != "^exit") {
      return "last block of leaf function must be ^exit, got " + leaf.blocks.back().label.name;
    }
    return "";
  }

  // --- inferMaskSetFromPuzzle helpers ---
  //
  // The puzzle text is not parseable (it contains FILL_XXX tokens which are
  // not valid RefractIR). Instead of encoding the masked-statement set in a
  // `//@ MASK:` header marker, we DERIVE it from the puzzle body by:
  //
  //   1. Rendering the solution twice through SIRMaskedPrinter with
  //      `sentinelMode=true`, which inserts '\x01' before each maskable
  //      position's value text. The first render uses full masking (producing
  //      FILL_XXX tokens), the second uses no masking (producing concrete values).
  //
  //   2. Stripping whitespace and `//` comments from both renders (preserving
  //      '\x01' sentinels, which never appear in real source) and splitting at
  //      the sentinels to get per-position segments.
  //
  //   3. Stripping the puzzle text and walking it in lock-step with the
  //      per-position segments: at each position i, the stripped puzzle either
  //      starts with the full-masked segment (masked) or the plain segment
  //      (revealed). The match uniquely identifies the masking decision because
  //      full-masked segments contain FILL_ tokens (never in concrete source)
  //      while plain segments contain concrete RefractIR values.
  //
  // This keeps the puzzle header clean (no machine-readable index noise) while
  // still giving rypuzchk the exact mask set it needs for the budget check and
  // the structural re-mask.

  /**
   * @brief Like stripCommentsAndWhitespace (in rypuzchk.cpp) but preserves
   * the '\x01' position-sentinel character emitted by SIRMaskedPrinter in
   * sentinel mode. This function is self-contained so puzzle_common.hpp stays
   * independent of rypuzchk's internal helpers.
   */
  inline std::string stripKeepingSentinels(const std::string &str) {
    std::string out;
    out.reserve(str.size());
    bool inComment = false, inString = false;
    for (size_t i = 0; i < str.size(); ++i) {
      char ch = str[i];
      if (inComment) {
        if (ch == '\n')
          inComment = false;
        continue;
      }
      if (inString) {
        out.push_back(ch);
        if (ch == '\\' && i + 1 < str.size()) {
          out.push_back(str[++i]);
        } else if (ch == '"') {
          inString = false;
        }
        continue;
      }
      if (ch == '/' && i + 1 < str.size() && str[i + 1] == '/') {
        inComment = true;
        ++i;
      } else if (ch == '"') {
        inString = true;
        out.push_back(ch);
      } else if (ch == '\x01') {
        out.push_back(ch); // preserve position sentinel
      } else if (!std::isspace(static_cast<unsigned char>(ch))) {
        out.push_back(ch);
      }
    }
    return out;
  }

  /**
   * @brief Splits `s` at every '\x01' sentinel character.
   *
   * Returns N+1 segments for N sentinels:
   *  - segments[0]: text before the first sentinel (the non-maskable prefix)
   *  - segments[i+1]: text between sentinel i and sentinel i+1 (or end of
   *    string), which is the value of maskable position i plus the
   *    non-maskable inter-position text leading to position i+1.
   */
  inline std::vector<std::string> splitAtSentinels(const std::string &s) {
    std::vector<std::string> parts;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
      if (i == s.size() || s[i] == '\x01') {
        parts.push_back(s.substr(start, i - start));
        start = i + 1; // skip the sentinel itself
      }
    }
    return parts;
  }

  /**
   * @brief Infers the mask set (the set of canonical maskable-statement
   * indices that were masked by `--p-mask`) directly from the puzzle body.
   *
   * The puzzle file is not parseable (FILL_XXX tokens), so rypuzchk cannot
   * parse it to get the mask. Instead this function compares the stripped
   * puzzle text against two sentinel-annotated renders of the *solution*:
   * one with full masking and one with no masking. At each canonical position
   * the puzzle matches exactly one of the two, identifying whether that
   * position was masked in the puzzle.
   *
   * @param leaf   The leaf FunDecl from the SOLUTION (parsed, fully concrete).
   * @param prog   The full Program containing `leaf`.
   * @param puzzleText  The raw puzzle file text (may contain FILL_XXX).
   * @return  The set of canonical position indices that are masked (contain
   *          FILL_XXX in the puzzle). Empty set if nothing is masked; the set
   *          equals the full index range [0, N) if all positions are masked.
   */
  inline std::unordered_set<int>
  inferMaskSetFromPuzzle(const FunDecl &leaf, const Program &prog, const std::string &puzzleText) {
    // Render solution with sentinels + full masking.
    std::ostringstream fullOut;
    SIRMaskedPrinter(
        fullOut, leaf, /*enableMasking=*/true,
        /*selectiveMask=*/nullptr, /*sentinelMode=*/true
    )
        .print(prog);
    // Render solution with sentinels + no masking.
    std::ostringstream plainOut;
    SIRMaskedPrinter(
        plainOut, leaf, /*enableMasking=*/false,
        /*selectiveMask=*/nullptr, /*sentinelMode=*/true
    )
        .print(prog);

    // Strip each render (preserving '\x01') and split into per-position segments.
    auto fullParts = splitAtSentinels(stripKeepingSentinels(fullOut.str()));
    auto plainParts = splitAtSentinels(stripKeepingSentinels(plainOut.str()));

    int nPositions = static_cast<int>(plainParts.size()) - 1; // excludes prefix

    // Strip the puzzle text the same way (no sentinels in it, so identical to
    // the normal strip — but using stripKeepingSentinels is harmless).
    std::string strippedPuzzle = stripKeepingSentinels(puzzleText);

    std::unordered_set<int> masked;
    size_t pos = 0;

    // Consume the non-maskable prefix (identical in both renders and in the
    // puzzle — it is the struct declarations, @main function, and the leaf
    // function signature/entry/exit boilerplate).
    if (!plainParts.empty())
      pos += plainParts[0].size();

    for (int i = 0; i < nPositions; ++i) {
      const std::string &maskedSeg =
          (i + 1 < static_cast<int>(fullParts.size())) ? fullParts[i + 1] : std::string{};
      const std::string &plainSeg =
          (i + 1 < static_cast<int>(plainParts.size())) ? plainParts[i + 1] : std::string{};

      // At position i the stripped puzzle must equal either the full-masked
      // segment or the plain segment. The full-masked segment contains FILL_
      // tokens (never in concrete source), so a prefix match is unambiguous.
      if (pos + maskedSeg.size() <= strippedPuzzle.size() &&
          strippedPuzzle.substr(pos, maskedSeg.size()) == maskedSeg) {
        masked.insert(i);
        pos += maskedSeg.size();
      } else {
        // Revealed: advance by the plain segment (concrete values).
        pos += plainSeg.size();
      }
    }
    return masked;
  }

} // namespace refractir
