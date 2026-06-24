#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <memory>
#include <stdexcept>

#include "ast/ast.hpp"
#include "analysis/cfg.hpp"
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
      for (const auto &l : f.lets) {
        if (l.init) {
          visitInitVal(*l.init);
        }
      }
      for (const auto &b : f.blocks) {
        for (const auto &ins : b.instrs) {
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
          for (const auto &elem : std::get<std::vector<InitValPtr>>(iv.value)) {
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
    virtual void visitInstr(const AssumeInstr &ins) {
      visitCond(ins.cond);
    }
    virtual void visitInstr(const RequireInstr &ins) {
      visitCond(ins.cond);
    }
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
      for (const auto &t : e.rest) {
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
      for (const auto &acc : lv.accesses) {
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
      std::visit([this](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, OpAtom>) {
          visitCoef(arg.coef);
          visitLValue(arg.rval);
        } else if constexpr (std::is_same_v<T, SelectAtom>) {
          if (arg.cond) visitCond(*arg.cond);
          if (arg.maskExpr) visitExpr(*arg.maskExpr);
          visitSelectVal(arg.vtrue);
          visitSelectVal(arg.vfalse);
        } else if constexpr (std::is_same_v<T, CoefAtom>) {
          visitCoef(arg.coef);
        } else if constexpr (std::is_same_v<T, RValueAtom>) {
          visitLValue(arg.rval);
        } else if constexpr (std::is_same_v<T, CastAtom>) {
          std::visit([this](auto &&src) {
            using S = std::decay_t<decltype(src)>;
            if constexpr (std::is_same_v<S, IntLit>) {
              visitIntLit(src);
            } else if constexpr (std::is_same_v<S, FloatLit>) {
              visitFloatLit(src);
            } else if constexpr (std::is_same_v<S, LValue>) {
              visitLValue(src);
            }
          }, arg.src);
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
          for (const auto &arg_expr : arg.args) {
            visitExpr(*arg_expr);
          }
        }
      }, a.v);
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

    // Whole lvalues mask to a single FILL_VAR; their bases/subscripts/fields
    // never surface as FILL_CONST, so do not descend into them.
    void visitLValue(const LValue &) override {}

    void visitIntLit(const IntLit &ilit) override {
      counts[std::to_string(ilit.value)]++;
    }
    void visitFloatLit(const FloatLit &flit) override {
      counts[formatDouble(flit.value)]++;
    }

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
     */
    void collect(const FunDecl &f) {
      // Masked let initialisers.
      for (const auto &l : f.lets) {
        if (l.init) {
          collectLetInit(*l.init, l.name.name);
        }
      }
      // Masked basic blocks: strictly between entry (first) and exit (last).
      for (size_t i = 0; i < f.blocks.size(); ++i) {
        bool masked = (i > 0) && (i + 1 < f.blocks.size());
        if (!masked) {
          continue;
        }
        const auto &b = f.blocks[i];
        for (const auto &ins : b.instrs) {
          if (std::holds_alternative<RequireInstr>(ins) ||
              std::holds_alternative<AssumeInstr>(ins)) {
            continue;
          }
          std::visit([this](auto &&arg) { visitInstr(arg); }, ins);
        }
        std::visit([this](auto &&arg) { visitTerminator(arg); }, b.term);
      }
    }

  private:
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
          for (const auto &elem : std::get<std::vector<InitValPtr>>(iv.value)) {
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
   * @brief Concrete walker to collect names of all called functions and intrinsics.
   */
  struct CallCollector : public ASTVisitor {
    // Set of all function/intrinsic names that are called in the function
    std::unordered_set<std::string> calledFunctions;

    /**
     * @brief Entry point to walk and collect function calls from a function definition.
     */
    void collect(const FunDecl &f) {
      visitFun(f);
    }

    /**
     * @brief Override to record the name of the callee when a function call is encountered.
     */
    void visitCall(const CallAtom &call) override {
      calledFunctions.insert(call.callee.name);
    }
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
    explicit SIRMaskedPrinter(std::ostream &out, const FunDecl &leafFunc, bool enableMasking = true)
        : out_(out), leafFunc_(&leafFunc), enableMasking_(enableMasking) {}

    /**
     * @brief Prints the entire RefractIR Program, masking the leaf function as required.
     */
    void print(const Program &p) {
      // 1. Struct declarations (never masked)
      for (const auto &s : p.structs) {
        out_ << "struct " << s.name.name << " {\n";
        indent_level_++;
        for (const auto &f : s.fields) {
          indent();
          out_ << f.name << ": ";
          printType(f.type);
          out_ << ";\n";
        }
        indent_level_--;
        out_ << "} \n\n";
      }

      // 2. Functions (leaf function gets its body masked)
      for (const auto &f : p.funs) {
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
        for (const auto &l : f.lets) {
          indent();
          out_ << "let " << (l.isMutable ? "mut " : "") << l.name.name << ": ";
          printType(l.type);
          if (l.init) {
            out_ << " = ";
            if (isLeaf && enableMasking_) {
              printMaskedInitVal(*l.init, l.name.name);
            } else {
              printInitVal(*l.init);
            }
          }
          out_ << ";\n";
        }

        // 2b. Print CFG blocks
        for (size_t i = 0; i < f.blocks.size(); ++i) {
          const auto &b = f.blocks[i];
          out_ << b.label.name << ":\n";
          
          // Only mask blocks of leaf function that are strictly between entry (0) and exit (last)
          bool shouldMaskBlock = isLeaf && enableMasking_ && (i > 0) && (i < f.blocks.size() - 1);
          masked_mode_ = shouldMaskBlock;

          for (const auto &ins : b.instrs) {
            // Drop require/assume from both the puzzle and the ground truth.
            // They are pure assertions (path conditions / properties) and not
            // part of the observable computation the solver must reconstruct.
            if (std::holds_alternative<RequireInstr>(ins) ||
                std::holds_alternative<AssumeInstr>(ins)) {
              continue;
            }
            indent();
            std::visit([this](auto &&arg) { printInstr(arg); }, ins);
            out_ << ";\n";
          }
          indent();
          std::visit([this](auto &&arg) { printTerminator(arg); }, b.term);
          out_ << ";\n";
        }

        indent_level_--;
        out_ << "} \n\n";
      }

      // 3. External Declarations
      for (const auto &d : p.extDecls) {
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
        for (const auto &pre : d.contract->pres) {
          indent();
          out_ << "pre ";
          printCond(pre.cond);
          if (pre.message)
            out_ << ", \"" << *pre.message << "\"";
          out_ << ";\n";
        }
        for (const auto &post : d.contract->posts) {
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
      for (const auto &in : p.intrinsics) {
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
    int indent_level_ = 0;
    bool masked_mode_ = false;

    void indent() {
      for (int i = 0; i < indent_level_; ++i)
        out_ << "  ";
    }

    void printType(const TypePtr &t) {
      if (!t) return;
      if (masked_mode_) {
        out_ << "FILL_TYPE";
        return;
      }
      std::visit([this](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, IntType>) {
          switch (arg.kind) {
            case IntType::Kind::I32: out_ << "i32"; break;
            case IntType::Kind::I64: out_ << "i64"; break;
            case IntType::Kind::ICustom:
              out_ << "i" << (arg.bits ? std::to_string(*arg.bits) : "?");
              break;
          }
        } else if constexpr (std::is_same_v<T, FloatType>) {
          switch (arg.kind) {
            case FloatType::Kind::F32: out_ << "f32"; break;
            case FloatType::Kind::F64: out_ << "f64"; break;
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
      }, t->v);
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

    void printTerminator(const UnreachableTerm &) {
      out_ << "unreachable";
    }

    void printExpr(const Expr &e) {
      printAtom(e.first);
      for (const auto &t : e.rest) {
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
      std::visit([this](auto &&arg) {
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
            if (arg.op == UnaryOpKind::Not) out_ << "~";
          }
          printLValue(arg.rval);
        } else if constexpr (std::is_same_v<T, CastAtom>) {
          std::visit([&](auto &&src) {
            using S = std::decay_t<decltype(src)>;
            if constexpr (std::is_same_v<S, IntLit>) {
              if (masked_mode_) out_ << "FILL_CONST";
              else out_ << src.value;
            } else if constexpr (std::is_same_v<S, FloatLit>) {
              if (masked_mode_) out_ << "FILL_CONST";
              else out_ << formatDouble(src.value);
            } else if constexpr (std::is_same_v<S, SymId>) {
              if (masked_mode_) out_ << "FILL_VAR";
              else out_ << src.name;
            } else {
              printLValue(src);
            }
          }, arg.src);
          if (masked_mode_) {
            out_ << " FILL_OP FILL_TYPE";
          } else {
            out_ << " as ";
            printType(arg.dstType);
          }
        } else if constexpr (std::is_same_v<T, AddrAtom>) {
          if (masked_mode_) out_ << "FILL_OP ";
          else out_ << "addr ";
          printLValue(arg.lv);
        } else if constexpr (std::is_same_v<T, LoadAtom>) {
          if (masked_mode_) out_ << "FILL_OP ";
          else out_ << "load ";
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
            if (i) out_ << ", ";
            printExpr(*arg.args[i]);
          }
          out_ << ")";
        }
      }, a.v);
    }

    void printLValue(const LValue &lv) {
      if (masked_mode_) {
        out_ << "FILL_VAR";
        return;
      }
      out_ << lv.base.name;
      for (const auto &acc : lv.accesses) {
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
      std::visit([this](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, IntLit>) {
          if (masked_mode_) out_ << "FILL_CONST";
          else out_ << arg.value;
        } else if constexpr (std::is_same_v<T, FloatLit>) {
          if (masked_mode_) out_ << "FILL_CONST";
          else out_ << formatDouble(arg.value);
        } else if constexpr (std::is_same_v<T, NullLit>) {
          if (masked_mode_) out_ << "FILL_CONST";
          else out_ << "null";
        } else {
          std::visit([this](auto &&id) {
            if (masked_mode_) out_ << "FILL_VAR";
            else out_ << id.name;
          }, arg);
        }
      }, c);
    }

    void printIndex(const Index &idx) {
      std::visit([this](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, IntLit>) {
          if (masked_mode_) out_ << "FILL_CONST";
          else out_ << arg.value;
        } else {
          std::visit([this](auto &&id) {
            if (masked_mode_) out_ << "FILL_VAR";
            else out_ << id.name;
          }, arg);
        }
      }, idx);
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
        case RelOp::EQ: return "==";
        case RelOp::NE: return "!=";
        case RelOp::LT: return "<";
        case RelOp::LE: return "<=";
        case RelOp::GT: return ">";
        case RelOp::GE: return ">=";
      }
      return "?";
    }

    std::string atomOpToString(AtomOpKind op) {
      switch (op) {
        case AtomOpKind::Mul: return "*";
        case AtomOpKind::Div: return "/";
        case AtomOpKind::Mod: return "%";
        case AtomOpKind::And: return "&";
        case AtomOpKind::Or: return "|";
        case AtomOpKind::Xor: return "^";
        case AtomOpKind::Shl: return "<<";
        case AtomOpKind::Shr: return ">>";
        case AtomOpKind::LShr: return ">>>";
      }
      return "?";
    }
  };

  /**
   * @brief Helper to locate the leaf function inside a Program (non-main function).
   */
  inline const FunDecl* findLeafFunction(const Program &prog) {
    for (const auto &f : prog.funs) {
      if (f.name.name != "@main") {
        return &f;
      }
    }
    return nullptr;
  }

  /**
   * @brief Helper to locate the `@main` wrapper function inside a Program.
   */
  inline const FunDecl* findMainFunction(const Program &prog) {
    for (const auto &f : prog.funs) {
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
      return "first block of leaf function must be ^entry, got " +
             leaf.blocks.front().label.name;
    }
    if (leaf.blocks.back().label.name != "^exit") {
      return "last block of leaf function must be ^exit, got " +
             leaf.blocks.back().label.name;
    }
    return "";
  }

} // namespace refractir
