#include <sstream>
#include "analysis/type_utils.hpp"
#include "backend/wasm_backend.hpp"
#include "wasm_internal.hpp"

namespace refractir {

  void WasmBackend::emitExpr(const Expr &expr, std::uint32_t targetWidth, bool isFloat) {
    emitAtom(expr.first, targetWidth, isFloat);

    for (const auto &t: expr.rest) {
      emitAtom(t.atom, targetWidth, isFloat);
      indent();
      if (isFloat) {
        out_ << (targetWidth <= 32 ? "f32." : "f64.") << (t.op == AddOp::Plus ? "add\n" : "sub\n");
      } else {
        if (targetWidth <= 32)
          out_ << (t.op == AddOp::Plus ? "i32.add\n" : "i32.sub\n");
        else
          out_ << (t.op == AddOp::Plus ? "i64.add\n" : "i64.sub\n");
        emitSignExtend(targetWidth, (targetWidth <= 32 ? 32 : 64));
      }
    }
  }

  void WasmBackend::emitAtom(const Atom &atom, std::uint32_t targetWidth, bool isFloat) {
    std::visit(
        [this, targetWidth, isFloat, &atom](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, CoefAtom>) {
            emitCoef(arg.coef, targetWidth, isFloat);
          } else if constexpr (std::is_same_v<T, RValueAtom>) {
            emitRValueAtom(arg, targetWidth);
          } else if constexpr (std::is_same_v<T, OpAtom>) {
            emitOpAtom(arg, targetWidth, isFloat);
          } else if constexpr (std::is_same_v<T, SelectAtom>) {
            emitSelectAtom(arg, targetWidth, isFloat);
          } else if constexpr (std::is_same_v<T, CmpAtom>) {
            emitCmpAtom(arg, targetWidth);
          } else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
            emitPtrIndexAtom(arg);
          } else if constexpr (std::is_same_v<T, CallAtom>) {
            emitCallAtom(arg);
            // Promote an f32 call result to f64 when the surrounding
            // context expects f64 (e.g. a comparison whose other operand —
            // a float literal or an f64 value — forced 64-bit width). This
            // mirrors emitRValueAtom's promotion for f32 locals; without it
            // a `require call @f32_returning(...) == <lit>` leaves an f32 on
            // the stack against an f64 comparand and fails WASM validation.
            if (isFloat && targetWidth == 64) {
              TypePtr rt = getAtomType(atom);
              if (rt && std::holds_alternative<FloatType>(rt->v) &&
                  std::get<FloatType>(rt->v).kind == FloatType::Kind::F32) {
                indent();
                out_ << "f64.promote_f32\n";
              }
            }
          } else if constexpr (std::is_same_v<T, PtrFieldAtom>) {
            emitPtrFieldAtom(arg);
          } else if constexpr (std::is_same_v<T, UnaryAtom>) {
            emitUnaryAtom(arg, targetWidth, isFloat);
          } else if constexpr (std::is_same_v<T, AddrAtom>) {
            emitAddrAtom(arg);
          } else if constexpr (std::is_same_v<T, LoadAtom>) {
            emitLoadAtom(arg, targetWidth, isFloat);
          } else if constexpr (std::is_same_v<T, CastAtom>) {
            emitCastAtom(arg, targetWidth);
          }
        },
        atom.v
    );
  }

  void WasmBackend::emitRValueAtom(const RValueAtom &arg, std::uint32_t targetWidth) {

    emitLValue(arg.rval, false);
    if (locals_.count(arg.rval.base.name)) {
      auto const &li = locals_.at(arg.rval.base.name);
      // Walk accesses to determine the actual loaded type, not the base
      // local's type (e.g. %s.tag where tag is i64 inside an aggregate %s).
      TypePtr srcType = li.refractirType;
      for (const auto &acc: arg.rval.accesses) {
        if (std::get_if<AccessIndex>(&acc)) {
          if (auto at = std::get_if<ArrayType>(&srcType->v))
            srcType = at->elem;
          else if (auto vt = std::get_if<VecType>(&srcType->v))
            srcType = vt->elem;
        } else if (auto af = std::get_if<AccessField>(&acc)) {
          if (auto st = std::get_if<StructType>(&srcType->v)) {
            if (structLayouts_.count(st->name.name) &&
                structLayouts_.at(st->name.name).fields.count(af->field))
              srcType = structLayouts_.at(st->name.name).fields.at(af->field).type;
          }
        }
      }
      std::uint32_t srcWidth = 32;
      bool srcIsFloat = false;
      if (auto bits = TypeUtils::getIntBitWidth(srcType)) {
        srcWidth = *bits;
      } else if (srcType && std::holds_alternative<FloatType>(srcType->v)) {
        srcIsFloat = true;
        srcWidth = (std::get<FloatType>(srcType->v).kind == FloatType::Kind::F32) ? 32 : 64;
      }
      if (srcIsFloat) {
        if (srcWidth == 32 && targetWidth == 64) {
          indent();
          out_ << "f64.promote_f32\n";
        }
      } else {
        if (srcWidth <= 32 && targetWidth > 32) {
          indent();
          out_ << "i64.extend_i32_s\n";
        } else if (srcWidth > 32 && targetWidth <= 32) {
          indent();
          out_ << "i32.wrap_i64\n";
        }
      }
    }
  }

  void WasmBackend::emitOpAtom(const OpAtom &arg, std::uint32_t targetWidth, bool isFloat) {
    uint32_t wasmWidth = (targetWidth <= 32 ? 32 : 64);

    if (isFloat) {
      emitCoef(arg.coef, targetWidth, isFloat);
      emitLValue(arg.rval, false);
      if (locals_.count(arg.rval.base.name)) {
        auto const &li = locals_.at(arg.rval.base.name);
        if (std::holds_alternative<FloatType>(li.refractirType->v)) {
          if (li.bitwidth == 32 && targetWidth == 64) {
            indent();
            out_ << "f64.promote_f32\n";
          }
        }
      }
      std::string prefix = (targetWidth <= 32 ? "f32." : "f64.");
      if (arg.op == AtomOpKind::Mod) {
        // WebAssembly MVP has no f32.rem / f64.rem instruction.
        // Emit fmod inline: x - trunc(x/y) * y
        // Stack after emitCoef+emitLValue above: [x, y] — but we need
        // x and y each twice, so emit them fresh here.
        // Re-emit y (with optional f32→f64 promotion) as a helper.
        auto emitY = [&]() {
          emitLValue(arg.rval, false);
          if (locals_.count(arg.rval.base.name)) {
            auto const &li = locals_.at(arg.rval.base.name);
            if (std::holds_alternative<FloatType>(li.refractirType->v)) {
              if (li.bitwidth == 32 && targetWidth == 64) {
                indent();
                out_ << "f64.promote_f32\n";
              }
            }
          }
        };
        // The preceding emitCoef/emitLValue already pushed [x, y] onto
        // the stack (used for the initial x below). Drop those now and
        // re-emit cleanly so the stack is deterministic.
        // Actually, since we need x twice we restructure entirely:
        // emitCoef/emitLValue above left [x, y] on stack; pop y with
        // local.set into a drop, but simpler: the two emits above are
        // effectively wasted — in WASM we must balance the stack.
        // Instead we emit a `drop` for y and `drop` for x (they were
        // pushed by the standard path above), then emit the full fmod.
        indent();
        out_ << "drop\n"; // drop y (emitLValue result)
        indent();
        out_ << "drop\n"; // drop x (emitCoef result)
        // Now emit fmod(x, y) = x - trunc(x/y) * y
        emitCoef(arg.coef, targetWidth, isFloat); // x
        emitCoef(arg.coef, targetWidth, isFloat); // x  (for division)
        emitY();                                  // y
        indent();
        out_ << prefix << "div\n"; // x/y
        // Spec §2.9 intermediate-overflow trap: the inner
        // fp.div of the `%` encoding is subject to §7.4 rule 6.  If
        // x/y is ±∞ or NaN at the operand precision, the path is
        // UB.  Save the quotient into a scratch local and trap via
        // `unreachable` when it isn't finite (NaN comparisons fold
        // to false, so `|q| < +inf` is the simplest finiteness
        // test that catches both inf and NaN). Under --no-ub-guards the
        // whole (stack-neutral) check is elided; the quotient stays on
        // the stack from the `div` above.
        if (!noUbGuards_) {
          std::string qLocal = (targetWidth <= 32) ? "$__fmod_q_f32" : "$__fmod_q_f64";
          indent();
          out_ << "local.tee " << qLocal << "\n"; // stack: [x, q]
          indent();
          out_ << prefix << "abs\n";
          indent();
          out_ << prefix << "const inf\n";
          indent();
          out_ << prefix << "lt\n"; // |q| < +inf ?  1 if finite, 0 otherwise
          indent();
          out_ << "i32.eqz\n"; // not-finite ?
          indent();
          out_ << "if\n";
          indent_level_++;
          indent();
          out_ << "unreachable\n";
          indent_level_--;
          indent();
          out_ << "end\n";
          indent();
          out_ << "local.get " << qLocal << "\n"; // restore q to stack
        }
        indent();
        out_ << prefix << "trunc\n"; // trunc(x/y)
        emitY();                     // y
        indent();
        out_ << prefix << "mul\n"; // trunc(x/y)*y
        indent();
        out_ << prefix << "sub\n"; // x - trunc(x/y)*y
      } else {
        indent();
        switch (arg.op) {
          case AtomOpKind::Mul:
            out_ << prefix << "mul\n";
            break;
          case AtomOpKind::Div:
            out_ << prefix << "div\n";
            break;
          default:
            break;
        }
      }
    } else if (arg.op == AtomOpKind::LShr) {
      emitCoef(arg.coef, targetWidth, isFloat);
      emitMask(targetWidth, wasmWidth);
      emitLValue(arg.rval, false);
      indent();
      out_ << (wasmWidth == 32 ? "i32.shr_u\n" : "i64.shr_u\n");
    } else {
      emitCoef(arg.coef, targetWidth, isFloat);
      emitLValue(arg.rval, false);
      if (locals_.count(arg.rval.base.name)) {
        std::uint32_t rWidth = getIntWidth(locals_.at(arg.rval.base.name).refractirType);
        if (rWidth <= 32 && targetWidth > 32) {
          indent();
          out_ << "i64.extend_i32_s\n";
        } else if (rWidth > 32 && targetWidth <= 32) {
          indent();
          out_ << "i32.wrap_i64\n";
        }
      }

      indent();
      std::string opStr;
      std::string prefix = (targetWidth <= 32 ? "i32." : "i64.");
      switch (arg.op) {
        case AtomOpKind::Mul:
          opStr = prefix + "mul";
          break;
        case AtomOpKind::Div:
          opStr = prefix + "div_s";
          break;
        case AtomOpKind::Mod:
          opStr = prefix + "rem_s";
          break;
        case AtomOpKind::And:
          opStr = prefix + "and";
          break;
        case AtomOpKind::Or:
          opStr = prefix + "or";
          break;
        case AtomOpKind::Xor:
          opStr = prefix + "xor";
          break;
        case AtomOpKind::Shl:
          opStr = prefix + "shl";
          break;
        case AtomOpKind::Shr:
          opStr = prefix + "shr_s";
          break;
        case AtomOpKind::LShr:
          opStr = prefix + "shr_u";
          break;
      }
      out_ << opStr << "\n";
    }
    if (!isFloat)
      emitSignExtend(targetWidth, wasmWidth);
  }

  void WasmBackend::emitSelectAtom(const SelectAtom &arg, std::uint32_t targetWidth, bool isFloat) {

    if (arg.maskExpr) {
      emitExpr(*arg.maskExpr, 32, false);
    } else {
      emitCond(*arg.cond);
    }
    indent();
    std::string typePrefix;
    if (isFloat)
      typePrefix = (targetWidth <= 32 ? "f32" : "f64");
    else
      typePrefix = (targetWidth <= 32 ? "i32" : "i64");

    out_ << "if (result " << typePrefix << ")\n";
    indent_level_++;
    emitSelectVal(arg.vtrue, targetWidth, isFloat);
    indent_level_--;
    indent();
    out_ << "else\n";
    indent_level_++;
    emitSelectVal(arg.vfalse, targetWidth, isFloat);
    indent_level_--;
    indent();
    out_ << "end\n";
  }

  void WasmBackend::emitCmpAtom(const CmpAtom &arg, std::uint32_t targetWidth) {

    TypePtr lhsType = getSelectValType(arg.lhs);
    TypePtr rhsType = getSelectValType(arg.rhs);
    bool isFloat = (lhsType && std::holds_alternative<FloatType>(lhsType->v)) ||
                   (rhsType && std::holds_alternative<FloatType>(rhsType->v));
    bool is64 = false;
    if (isFloat) {
      is64 = (lhsType && std::get_if<FloatType>(&lhsType->v) &&
              std::get<FloatType>(lhsType->v).kind == FloatType::Kind::F64) ||
             (rhsType && std::get_if<FloatType>(&rhsType->v) &&
              std::get<FloatType>(rhsType->v).kind == FloatType::Kind::F64);
    } else {
      auto getWidth = [](const TypePtr &t) -> uint32_t {
        if (!t)
          return 32;
        if (auto it = std::get_if<IntType>(&t->v)) {
          if (it->kind == IntType::Kind::I64 || (it->bits && *it->bits > 32))
            return 64;
        }
        return 32;
      };
      is64 = (getWidth(lhsType) > 32 || getWidth(rhsType) > 32);
    }
    uint32_t operandWidth = is64 ? 64 : 32;
    emitSelectVal(arg.lhs, operandWidth, isFloat);
    emitSelectVal(arg.rhs, operandWidth, isFloat);

    indent();
    std::string opStr;
    if (isFloat) {
      std::string prefix = (operandWidth <= 32 ? "f32." : "f64.");
      switch (arg.op) {
        case RelOp::EQ:
          opStr = prefix + "eq";
          break;
        case RelOp::NE:
          opStr = prefix + "ne";
          break;
        case RelOp::LT:
          opStr = prefix + "lt";
          break;
        case RelOp::LE:
          opStr = prefix + "le";
          break;
        case RelOp::GT:
          opStr = prefix + "gt";
          break;
        case RelOp::GE:
          opStr = prefix + "ge";
          break;
      }
    } else {
      std::string prefix = (operandWidth <= 32 ? "i32." : "i64.");
      switch (arg.op) {
        case RelOp::EQ:
          opStr = prefix + "eq";
          break;
        case RelOp::NE:
          opStr = prefix + "ne";
          break;
        case RelOp::LT:
          opStr = prefix + "lt_s";
          break;
        case RelOp::LE:
          opStr = prefix + "le_s";
          break;
        case RelOp::GT:
          opStr = prefix + "gt_s";
          break;
        case RelOp::GE:
          opStr = prefix + "ge_s";
          break;
      }
    }
    out_ << opStr << "\n";
    // cmp returns i1; sign-extend bit 0 so true is -1.
    emitSignExtend(1, (targetWidth > 32 ? 64 : 32));
  }

  void WasmBackend::emitPtrIndexAtom(const PtrIndexAtom &arg) {

    uint64_t arrSize = 0;
    TypePtr elemType = nullptr;
    auto rvTy = getLValueType(arg.rval);
    if (rvTy) {
      if (auto pt = std::get_if<PtrType>(&rvTy->v)) {
        if (auto at = std::get_if<ArrayType>(&pt->pointee->v)) {
          arrSize = at->size;
          elemType = at->elem;
        }
      }
    }
    uint32_t elemSize = elemType ? getTypeSize(elemType) : 1;

    // The pointer and index are saved to $__ptr_temp / $__idx_temp (they
    // feed the address computation below); with guards on we also
    // null-check the pointer and range-check the index. Under
    // --no-ub-guards those checks are dropped — the `local.tee` copy is
    // balanced with a `drop`, keeping the stack identical.
    emitLValue(arg.rval, false);
    indent();
    out_ << "local.tee $__ptr_temp\n";
    if (!noUbGuards_) {
      indent();
      out_ << "i32.eqz\n";
      indent();
      out_ << "if\n";
      indent_level_++;
      indent();
      out_ << "unreachable\n";
      indent_level_--;
      indent();
      out_ << "end\n";
    } else {
      indent();
      out_ << "drop\n";
    }

    // `local.set` (not tee): the index is only ever re-read from
    // $__idx_temp below (bounds check + address math), so a stack copy
    // would be orphaned. The dispatch-loop emitter tolerates such an
    // orphan (its blocks end in `br`, which discards leftover stack),
    // but the structured emitter falls through block ends and would
    // trip WASM's "values remaining on stack" validation.
    emitIndex(arg.index);
    indent();
    out_ << "local.set $__idx_temp\n";
    if (!noUbGuards_) {
      indent();
      out_ << "local.get $__idx_temp\n";
      indent();
      out_ << "i32.const 0\n";
      indent();
      out_ << "i32.lt_s\n";
      indent();
      out_ << "local.get $__idx_temp\n";
      indent();
      out_ << "i32.const " << arrSize << "\n";
      indent();
      out_ << "i32.gt_s\n";
      indent();
      out_ << "i32.or\n";
      indent();
      out_ << "if\n";
      indent_level_++;
      indent();
      out_ << "unreachable\n";
      indent_level_--;
      indent();
      out_ << "end\n";
    }

    indent();
    out_ << "local.get $__ptr_temp\n";
    indent();
    out_ << "local.get $__idx_temp\n";
    if (elemSize > 1) {
      indent();
      out_ << "i32.const " << elemSize << "\n";
      indent();
      out_ << "i32.mul\n";
    }
    indent();
    out_ << "i32.add\n";
  }

  // Use the overload the type checker pinned onto the AST
  // node — see CallAtom::resolvedIntrinsic — with a width-based
  // fallback for un-annotated nodes.
  const IntrinsicDecl *WasmBackend::resolveIntrinsic(const CallAtom &arg) {
    const IntrinsicDecl *intr = arg.resolvedIntrinsic;
    if (!intr && prog_) {
      for (const auto &i: prog_->intrinsics) {
        if (i.name.name != arg.callee.name)
          continue;
        if (i.params.size() != arg.args.size())
          continue;
        if (!intr) {
          intr = &i;
          continue;
        }
        auto bw1 = TypeUtils::getIntBitWidth(intr->params[0].type);
        auto bw2 = TypeUtils::getIntBitWidth(i.params[0].type);
        if (!bw1 || !bw2 || *bw1 == *bw2)
          continue;
        uint32_t argBW = *bw1;
        if (!arg.args.empty()) {
          auto at = getExprType(*arg.args[0]);
          if (at) {
            if (auto b = TypeUtils::getIntBitWidth(at))
              argBW = *b;
          }
        }
        if (*bw2 == argBW && *bw1 != argBW)
          intr = &i;
      }
    }
    return intr;
  }

  std::pair<std::vector<TypePtr>, TypePtr> WasmBackend::calleeSignature(const CallAtom &arg) {
    std::vector<TypePtr> ptypes;
    TypePtr rtype;
    if (const IntrinsicDecl *intr = resolveIntrinsic(arg)) {
      for (const auto &p: intr->params)
        ptypes.push_back(p.type);
      return {std::move(ptypes), intr->retType};
    }
    if (prog_) {
      for (const auto &f: prog_->funs)
        if (f.name.name == arg.callee.name) {
          for (const auto &p: f.params)
            ptypes.push_back(p.type);
          return {std::move(ptypes), f.retType};
        }
      for (const auto &d: prog_->extDecls)
        if (d.name.name == arg.callee.name) {
          for (const auto &p: d.params)
            ptypes.push_back(p.type);
          return {std::move(ptypes), d.retType};
        }
    }
    return {std::move(ptypes), rtype};
  }

  void WasmBackend::emitCallAtom(const CallAtom &arg, std::int64_t sretOffset) {
    // Push arguments left-to-right then `call $name`. Vector arguments
    // are spilled to this call site's scratch block and passed as an i32
    // address; a vector return gets the sret address pushed last.
    const IntrinsicDecl *intr = resolveIntrinsic(arg);
    auto [ptypes, rtype] = calleeSignature(arg);
    VecCallSlots slots = vecCallSlots(arg);
    std::uint32_t base = 0;
    if (slots.totalBytes > 0)
      base = callVecScratch_.at(&arg); // pre-scan in emit() sized this block
    for (size_t i = 0; i < arg.args.size(); ++i) {
      if (i < ptypes.size() && ptypes[i] && std::holds_alternative<VecType>(ptypes[i]->v)) {
        const auto &vt = std::get<VecType>(ptypes[i]->v);
        std::int64_t off = static_cast<std::int64_t>(base) - slots.argDisp[i];
        std::uint32_t elemSize = getTypeSize(vt.elem);
        std::uint32_t width = getIntWidth(vt.elem);
        bool isFloat = std::holds_alternative<FloatType>(vt.elem->v);
        // Inner vector-returning calls evaluate once, before the lanes.
        materializeVecCalls(*arg.args[i]);
        for (std::uint64_t lane = 0; lane < vt.size; ++lane) {
          indent();
          out_ << "local.get $__old_sp\n";
          indent();
          out_ << "i32.const " << (off - static_cast<std::int64_t>(lane * elemSize)) << "\n";
          indent();
          out_ << "i32.sub\n";
          emitVecExprLane(*arg.args[i], vt, lane, width, isFloat);
          indent();
          if (isFloat) {
            out_ << (width == 32 ? "f32.store\n" : "f64.store\n");
          } else {
            out_ << (width <= 8 ? "i32.store8"
                                : (width <= 16 ? "i32.store16"
                                               : (width <= 32 ? "i32.store" : "i64.store")))
                 << "\n";
          }
        }
        indent();
        out_ << "local.get $__old_sp\n";
        indent();
        out_ << "i32.const " << off << "\n";
        indent();
        out_ << "i32.sub\n";
        continue;
      }
      uint32_t pw = 32;
      bool pf = false;
      if (i < ptypes.size() && ptypes[i]) {
        pw = getIntWidth(ptypes[i]);
        if (pw == 0) {
          if (std::holds_alternative<FloatType>(ptypes[i]->v)) {
            auto &ft = std::get<FloatType>(ptypes[i]->v);
            pw = (ft.kind == FloatType::Kind::F32) ? 32 : 64;
            pf = true;
          } else {
            pw = 32;
          }
        }
      }
      emitExpr(*arg.args[i], pw, pf);
    }
    if (rtype && std::holds_alternative<VecType>(rtype->v)) {
      std::int64_t off =
          sretOffset >= 0 ? sretOffset : static_cast<std::int64_t>(base) - slots.retDisp;
      indent();
      out_ << "local.get $__old_sp\n";
      indent();
      out_ << "i32.const " << off << "\n";
      indent();
      out_ << "i32.sub\n";
    }
    indent();
    if (intr) {
      out_ << "call " << intrinsicHelperName(*intr) << "\n";
    } else {
      out_ << "call " << mangleName(arg.callee.name) << "\n";
    }
  }

  void WasmBackend::emitPtrFieldAtom(const PtrFieldAtom &arg) {

    uint32_t fieldOffset = 0;
    auto rvTy = getLValueType(arg.rval);
    if (rvTy) {
      if (auto pt = std::get_if<PtrType>(&rvTy->v)) {
        if (auto st = std::get_if<StructType>(&pt->pointee->v)) {
          if (structLayouts_.count(st->name.name)) {
            const auto &sinfo = structLayouts_.at(st->name.name);
            if (sinfo.fields.count(arg.field)) {
              fieldOffset = sinfo.fields.at(arg.field).offset;
            }
          }
        }
      }
    }

    // Save the pointer to $__ptr_temp (used below); null-check it with
    // guards on, or balance the `local.tee` copy with a `drop` under
    // --no-ub-guards.
    emitLValue(arg.rval, false);
    indent();
    out_ << "local.tee $__ptr_temp\n";
    if (!noUbGuards_) {
      indent();
      out_ << "i32.eqz\n";
      indent();
      out_ << "if\n";
      indent_level_++;
      indent();
      out_ << "unreachable\n";
      indent_level_--;
      indent();
      out_ << "end\n";
    } else {
      indent();
      out_ << "drop\n";
    }

    if (fieldOffset > 0) {
      indent();
      out_ << "local.get $__ptr_temp\n";
      indent();
      out_ << "i32.const " << fieldOffset << "\n";
      indent();
      out_ << "i32.add\n";
    } else {
      indent();
      out_ << "local.get $__ptr_temp\n";
    }
  }

  void WasmBackend::emitUnaryAtom(const UnaryAtom &arg, std::uint32_t targetWidth, bool isFloat) {
    uint32_t wasmWidth = (targetWidth <= 32 ? 32 : 64);

    emitLValue(arg.rval, false);
    if (isFloat) {
      indent();
      out_ << (targetWidth <= 32 ? "f32.neg\n" : "f64.neg\n");
    } else {
      if (locals_.count(arg.rval.base.name)) {
        std::uint32_t srcWidth = getIntWidth(locals_.at(arg.rval.base.name).refractirType);
        if (srcWidth <= 32 && targetWidth > 32) {
          indent();
          out_ << "i64.extend_i32_s\n";
        } else if (srcWidth > 32 && targetWidth <= 32) {
          indent();
          out_ << "i32.wrap_i64\n";
        }
      }

      indent();
      if (targetWidth <= 32) {
        out_ << "i32.const -1\n";
        indent();
        out_ << "i32.xor\n";
      } else {
        out_ << "i64.const -1\n";
        indent();
        out_ << "i64.xor\n";
      }
      emitSignExtend(targetWidth, wasmWidth);
    }
  }

  void WasmBackend::emitAddrAtom(const AddrAtom &arg) {

    // Return the WASM memory address of the lvalue (must be an aggregate/spilled local)
    emitAddress(arg.lv);
  }

  void WasmBackend::emitLoadAtom(const LoadAtom &arg, std::uint32_t targetWidth, bool isFloat) {

    // Load through pointer: *ptr
    // Push ptr twice — first copy stays for the actual load,
    // second copy is used for the null check.
    const std::string &pname = arg.rval.base.name;
    // If pname was address-taken, it lives on the shadow stack rather
    // than as a WASM local; fetch its value from there.
    auto emitPushPtr = [&]() {
      if (locals_.count(pname) && locals_.at(pname).isAggregate) {
        const auto &pinfo = locals_.at(pname);
        indent();
        out_ << "local.get $__old_sp\n";
        indent();
        out_ << "i32.const " << pinfo.offset << "\n";
        indent();
        out_ << "i32.sub\n";
        indent();
        out_ << "i32.load\n";
      } else {
        indent();
        out_ << "local.get " << mangleName(pname) << "\n";
      }
    };
    // Push the pointer for the load. With guards on, push a second copy
    // and null-test it (leaving the first for the load); under
    // --no-ub-guards the single push is all that's needed.
    emitPushPtr();
    if (!noUbGuards_) {
      emitPushPtr();
      indent();
      out_ << "i32.eqz\n";
      indent();
      out_ << "if\n";
      indent_level_++;
      indent();
      out_ << "unreachable\n";
      indent_level_--;
      indent();
      out_ << "end\n";
    }
    // stack: [ptr_for_load]; determine pointee type for load instruction
    uint32_t loadWidth = targetWidth;
    bool loadIsFloat = isFloat;
    if (locals_.count(pname)) {
      const auto &info = locals_.at(pname);
      if (auto pt = std::get_if<PtrType>(&info.refractirType->v)) {
        if (auto bits = TypeUtils::getIntBitWidth(pt->pointee)) {
          loadWidth = *bits;
          loadIsFloat = false;
        } else if (pt->pointee && std::holds_alternative<FloatType>(pt->pointee->v)) {
          loadIsFloat = true;
          loadWidth = (std::get<FloatType>(pt->pointee->v).kind == FloatType::Kind::F32) ? 32 : 64;
        }
      }
    }
    indent();
    if (loadIsFloat) {
      out_ << (loadWidth <= 32 ? "f32.load\n" : "f64.load\n");
    } else {
      out_
          << (loadWidth <= 8    ? "i32.load8_s\n"
              : loadWidth <= 16 ? "i32.load16_s\n"
              : loadWidth <= 32 ? "i32.load\n"
                                : "i64.load\n");
    }
    // Sign-extend if needed
    if (!loadIsFloat && loadWidth <= 32 && targetWidth > 32) {
      indent();
      out_ << "i64.extend_i32_s\n";
    }
  }

  void WasmBackend::emitCastAtom(const CastAtom &arg, std::uint32_t targetWidth) {
    uint32_t wasmWidth = (targetWidth <= 32 ? 32 : 64);

    std::uint32_t srcWidth = 32;
    bool srcIsFloat = false;
    std::visit(
        [&](auto &&src) {
          using S = std::decay_t<decltype(src)>;
          if constexpr (std::is_same_v<S, IntLit>) {
            srcWidth = (src.value > INT32_MAX || src.value < INT32_MIN) ? 64 : 32;
            indent();
            out_ << (srcWidth <= 32 ? "i32.const " : "i64.const ") << src.value << "\n";
          } else if constexpr (std::is_same_v<S, FloatLit>) {
            srcWidth = 64;
            srcIsFloat = true;
            indent();
            out_ << "f64.const " << formatFloatLit(src.value) << "\n";
          } else if constexpr (std::is_same_v<S, SymId>) {
            indent();
            out_ << "call " << mangleName(getMangledSymbolName(curFuncName_, src.name)) << "\n";
            srcWidth = 32;
            if (syms_.count(src.name)) {
              srcWidth = getIntWidth(syms_.at(src.name));
              if (std::holds_alternative<FloatType>(syms_.at(src.name)->v)) {
                srcIsFloat = true;
                srcWidth = (std::get<FloatType>(syms_.at(src.name)->v).kind == FloatType::Kind::F32)
                               ? 32
                               : 64;
              }
            }
          } else {
            emitLValue(src, false);
            if (locals_.count(src.base.name)) {
              auto const &li = locals_.at(src.base.name);
              // Walk accesses so cast-from-field uses the field's
              // type, not the base local's (e.g. %s.tag where tag is i64).
              TypePtr at_type = li.refractirType;
              for (const auto &acc: src.accesses) {
                if (std::get_if<AccessIndex>(&acc)) {
                  if (auto at = std::get_if<ArrayType>(&at_type->v))
                    at_type = at->elem;
                  else if (auto vt = std::get_if<VecType>(&at_type->v))
                    at_type = vt->elem;
                } else if (auto af = std::get_if<AccessField>(&acc)) {
                  if (auto st = std::get_if<StructType>(&at_type->v)) {
                    if (structLayouts_.count(st->name.name) &&
                        structLayouts_.at(st->name.name).fields.count(af->field))
                      at_type = structLayouts_.at(st->name.name).fields.at(af->field).type;
                  }
                }
              }
              if (auto bits = TypeUtils::getIntBitWidth(at_type)) {
                srcWidth = *bits;
              } else if (at_type && std::holds_alternative<FloatType>(at_type->v)) {
                srcIsFloat = true;
                srcWidth = (std::get<FloatType>(at_type->v).kind == FloatType::Kind::F32) ? 32 : 64;
              }
            }
          }
        },
        arg.src
    );

    bool dstIsFloat = std::holds_alternative<FloatType>(arg.dstType->v);
    std::uint32_t dstWidth = 32;
    if (dstIsFloat) {
      dstWidth = (std::get<FloatType>(arg.dstType->v).kind == FloatType::Kind::F32) ? 32 : 64;
    } else {
      dstWidth = getIntWidth(arg.dstType);
    }

    indent();
    if (srcIsFloat && dstIsFloat) {
      if (srcWidth == 32 && dstWidth == 64)
        out_ << "f64.promote_f32\n";
      else if (srcWidth == 64 && dstWidth == 32)
        out_ << "f32.demote_f64\n";
    } else if (srcIsFloat && !dstIsFloat) {
      if (dstWidth <= 32)
        out_ << (srcWidth == 32 ? "i32.trunc_f32_s\n" : "i32.trunc_f64_s\n");
      else
        out_ << (srcWidth == 32 ? "i64.trunc_f32_s\n" : "i64.trunc_f64_s\n");
    } else if (!srcIsFloat && dstIsFloat) {
      if (srcWidth <= 32)
        out_ << (dstWidth == 32 ? "f32.convert_i32_s\n" : "f64.convert_i32_s\n");
      else
        out_ << (dstWidth == 32 ? "f32.convert_i64_s\n" : "f64.convert_i64_s\n");
    } else {
      // BV -> BV
      // Spec §6.4: i1 is signed; widening sign-extends
      // bit 0.  Emit the i1 sign-extension *first* (in i32
      // space) so a subsequent i64.extend_i32_s, if any, sees
      // the already-sign-extended low half.  Otherwise the
      // shift-pair ends up landing on a 64-bit stack top.
      if (srcWidth == 1) {
        emitSignExtend(1, 32);
      }
      if (srcWidth <= 32 && (targetWidth > 32 || dstWidth > 32)) {
        if (wasmWidth == 64) {
          out_ << "i64.extend_i32_s\n";
        }
      } else if (srcWidth > 32 && (targetWidth <= 32 || dstWidth <= 32)) {
        if (wasmWidth == 32) {
          out_ << "i32.wrap_i64\n";
        }
      }
    }
    if (!dstIsFloat && srcWidth != 1)
      emitSignExtend(targetWidth, wasmWidth);
  }

  void WasmBackend::emitCond(const Cond &cond) {
    std::uint32_t width = 32;
    bool isFloat = false;
    auto needs64 = [&](const Expr &e) {
      auto atomNeeds64 = [&](const Atom &a) {
        if (std::holds_alternative<CastAtom>(a.v)) {
          auto const &ca = std::get<CastAtom>(a.v);
          if (std::holds_alternative<FloatType>(ca.dstType->v)) {
            isFloat = true;
            if (std::get<FloatType>(ca.dstType->v).kind == FloatType::Kind::F64)
              return true;
          }
          if (getIntWidth(ca.dstType) > 32)
            return true;
        } else if (std::holds_alternative<RValueAtom>(a.v)) {
          auto &lv = std::get<RValueAtom>(a.v).rval;
          if (locals_.count(lv.base.name)) {
            auto const &li = locals_.at(lv.base.name);
            if (std::holds_alternative<FloatType>(li.refractirType->v)) {
              isFloat = true;
              if (std::get<FloatType>(li.refractirType->v).kind == FloatType::Kind::F64)
                return true;
            }
            if (li.bitwidth > 32)
              return true;
          }
        } else if (std::holds_alternative<CoefAtom>(a.v)) {
          auto &coef = std::get<CoefAtom>(a.v).coef;
          if (std::holds_alternative<FloatLit>(coef)) {
            isFloat = true;
            return true; // assume f64 for lit if 64 bits target?
          }
          if (std::holds_alternative<LocalOrSymId>(coef)) {
            auto &id = std::get<LocalOrSymId>(coef);
            auto name = std::visit([](auto &&v) { return v.name; }, id);
            if (locals_.count(name)) {
              auto const &li = locals_.at(name);
              if (std::holds_alternative<FloatType>(li.refractirType->v)) {
                isFloat = true;
                if (std::get<FloatType>(li.refractirType->v).kind == FloatType::Kind::F64)
                  return true;
              }
              if (li.bitwidth > 32)
                return true;
            } else if (syms_.count(name)) {
              auto const &st = syms_.at(name);
              if (std::holds_alternative<FloatType>(st->v)) {
                isFloat = true;
                if (std::get<FloatType>(st->v).kind == FloatType::Kind::F64)
                  return true;
              }
              if (getIntWidth(st) > 32)
                return true;
            }
          }
        } else if (std::holds_alternative<CallAtom>(a.v)) {
          // The width of a call sub-expression is its callee's return
          // type width.  Without this case, a `require call @to_bits(%v)
          // == <i64-lit>` widened the wrong side: the LHS becomes i64
          // (from to_bits), but the comparison-wide check stayed at
          // 32 because no other atom asserted 64, so the literal got
          // emitted as `i32.const <i64-value>` and produced invalid WAT.
          if (!prog_)
            return false;
          auto const &ca = std::get<CallAtom>(a.v);
          TypePtr retTy;
          if (ca.resolvedIntrinsic)
            retTy = ca.resolvedIntrinsic->retType;
          if (!retTy) {
            for (const auto &f: prog_->funs)
              if (f.name.name == ca.callee.name) {
                retTy = f.retType;
                break;
              }
          }
          if (!retTy) {
            for (const auto &d: prog_->extDecls)
              if (d.name.name == ca.callee.name) {
                retTy = d.retType;
                break;
              }
          }
          if (!retTy) {
            // Fallback to first intrinsic decl matching by name.
            for (const auto &i: prog_->intrinsics)
              if (i.name.name == ca.callee.name) {
                retTy = i.retType;
                break;
              }
          }
          if (retTy) {
            if (auto ft = std::get_if<FloatType>(&retTy->v)) {
              isFloat = true;
              if (ft->kind == FloatType::Kind::F64)
                return true;
            } else if (auto it = std::get_if<IntType>(&retTy->v)) {
              uint32_t bits = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
              if (bits > 32)
                return true;
            }
          }
        }
        return false;
      };
      if (atomNeeds64(e.first))
        return true;
      for (const auto &t: e.rest)
        if (atomNeeds64(t.atom))
          return true;
      return false;
    };
    if (needs64(cond.lhs) || needs64(cond.rhs))
      width = 64;

    emitExpr(cond.lhs, width, isFloat);
    emitExpr(cond.rhs, width, isFloat);
    indent();
    std::string opStr;
    if (isFloat) {
      std::string prefix = (width <= 32 ? "f32." : "f64.");
      switch (cond.op) {
        case RelOp::EQ:
          opStr = prefix + "eq";
          break;
        case RelOp::NE:
          opStr = prefix + "ne";
          break;
        case RelOp::LT:
          opStr = prefix + "lt";
          break;
        case RelOp::LE:
          opStr = prefix + "le";
          break;
        case RelOp::GT:
          opStr = prefix + "gt";
          break;
        case RelOp::GE:
          opStr = prefix + "ge";
          break;
      }
    } else {
      std::string prefix = (width <= 32 ? "i32." : "i64.");
      switch (cond.op) {
        case RelOp::EQ:
          opStr = prefix + "eq";
          break;
        case RelOp::NE:
          opStr = prefix + "ne";
          break;
        case RelOp::LT:
          opStr = prefix + "lt_s";
          break;
        case RelOp::LE:
          opStr = prefix + "le_s";
          break;
        case RelOp::GT:
          opStr = prefix + "gt_s";
          break;
        case RelOp::GE:
          opStr = prefix + "ge_s";
          break;
      }
    }
    out_ << opStr << "\n";
  }

  void WasmBackend::emitCoef(const Coef &coef, std::uint32_t targetWidth, bool isFloat) {
    uint32_t wasmWidth = (targetWidth <= 32 ? 32 : 64);
    std::visit(
        [this, targetWidth, wasmWidth, isFloat](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, IntLit>) {
            indent();
            if (isFloat) {
              out_ << (wasmWidth == 32 ? "f32.const " : "f64.const ") << arg.value << ".0\n";
            } else {
              out_ << (wasmWidth == 32 ? "i32.const " : "i64.const ") << arg.value << "\n";
            }
          } else if constexpr (std::is_same_v<T, FloatLit>) {
            indent();
            out_ << (wasmWidth == 32 ? "f32.const " : "f64.const ") << formatFloatLit(arg.value)
                 << "\n";
          } else if constexpr (std::is_same_v<T, NullLit>) {
            // null pointer = 0 as i32 (WASM pointers are 32-bit)
            indent();
            out_ << "i32.const 0\n";
          } else {
            std::visit(
                [this, targetWidth](auto &&id) {
                  using ID = std::decay_t<decltype(id)>;
                  if constexpr (std::is_same_v<ID, SymId>) {
                    indent();
                    out_ << "call " << mangleName(getMangledSymbolName(curFuncName_, id.name))
                         << "\n";
                    std::uint32_t srcWidth = 32;
                    bool srcIsFloat = false;
                    if (syms_.count(id.name)) {
                      srcWidth = getIntWidth(syms_.at(id.name));
                      if (std::holds_alternative<FloatType>(syms_.at(id.name)->v))
                        srcIsFloat = true;
                    }
                    if (srcIsFloat) {
                      if (srcWidth == 32 && targetWidth == 64) {
                        indent();
                        out_ << "f64.promote_f32\n";
                      }
                    } else {
                      if (srcWidth <= 32 && targetWidth > 32) {
                        indent();
                        out_ << "i64.extend_i32_s\n";
                      } else if (srcWidth > 32 && targetWidth <= 32) {
                        indent();
                        out_ << "i32.wrap_i64\n";
                      }
                    }
                  } else {
                    if (locals_.count(id.name) && locals_.at(id.name).isAggregate) {
                      emitLValue({id, {}, id.span}, false);
                    } else {
                      indent();
                      out_ << "local.get " << mangleName(id.name) << "\n";
                    }
                    if (locals_.count(id.name)) {
                      auto const &li = locals_.at(id.name);
                      std::uint32_t srcWidth = li.bitwidth;
                      if (std::holds_alternative<FloatType>(li.refractirType->v)) {
                        if (srcWidth == 32 && targetWidth == 64) {
                          indent();
                          out_ << "f64.promote_f32\n";
                        }
                      } else {
                        if (srcWidth <= 32 && targetWidth > 32) {
                          indent();
                          out_ << "i64.extend_i32_s\n";
                        } else if (srcWidth > 32 && targetWidth <= 32) {
                          indent();
                          out_ << "i32.wrap_i64\n";
                        }
                      }
                    }
                  }
                },
                arg
            );
          }
        },
        coef
    );
    if (!isFloat)
      emitSignExtend(targetWidth, wasmWidth);
  }

  void WasmBackend::emitSelectVal(const SelectVal &sv, std::uint32_t targetWidth, bool isFloat) {
    if (std::holds_alternative<RValue>(sv)) {
      const auto &lv = std::get<RValue>(sv);
      emitLValue(lv, false);
      TypePtr ty = getLValueType(lv);
      if (ty) {
        if (std::holds_alternative<FloatType>(ty->v)) {
          auto ft = std::get<FloatType>(ty->v);
          uint32_t srcWidth = (ft.kind == FloatType::Kind::F32) ? 32 : 64;
          if (srcWidth == 32 && targetWidth > 32) {
            indent();
            out_ << "f64.promote_f32\n";
          } else if (srcWidth == 64 && targetWidth <= 32) {
            indent();
            out_ << "f32.demote_f64\n";
          }
        } else {
          std::uint32_t srcWidth = getIntWidth(ty);
          if (srcWidth <= 32 && targetWidth > 32) {
            indent();
            out_ << "i64.extend_i32_s\n";
          } else if (srcWidth > 32 && targetWidth <= 32) {
            indent();
            out_ << "i32.wrap_i64\n";
          }
        }
      }
    } else {
      emitCoef(std::get<Coef>(sv), targetWidth, isFloat);
    }
  }

  void WasmBackend::emitIndex(const Index &idx) {
    std::visit(
        [this](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, IntLit>) {
            indent();
            out_ << "i32.const " << arg.value << "\n";
          } else {
            std::visit(
                [this](auto &&id) {
                  using ID = std::decay_t<decltype(id)>;
                  if constexpr (std::is_same_v<ID, SymId>) {
                    indent();
                    out_ << "call " << mangleName(getMangledSymbolName(curFuncName_, id.name))
                         << "\n";
                    // Indices must be i32. If sym is i64, wrap it.
                    std::uint32_t srcWidth = 32;
                    if (syms_.count(id.name)) {
                      srcWidth = getIntWidth(syms_.at(id.name));
                    }
                    if (srcWidth > 32) {
                      indent();
                      out_ << "i32.wrap_i64\n";
                    }
                  } else {
                    if (locals_.count(id.name) && locals_.at(id.name).isAggregate) {
                      emitLValue({id, {}, id.span}, false);
                    } else {
                      indent();
                      out_ << "local.get " << mangleName(id.name) << "\n";
                    }
                    if (locals_.count(id.name)) {
                      std::uint32_t srcWidth = getIntWidth(locals_.at(id.name).refractirType);
                      if (srcWidth > 32) {
                        indent();
                        out_ << "i32.wrap_i64\n";
                      }
                    }
                  }
                },
                arg
            );
          }
        },
        idx
    );
  }
} // namespace refractir
