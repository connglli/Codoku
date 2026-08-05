#include <sstream>
#include "analysis/type_utils.hpp"
#include "backend/wasm_backend.hpp"
#include "wasm_internal.hpp"

namespace refractir {

  void WasmBackend::emitVecExprLane(
      const Expr &expr, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
      bool isFloat
  ) {
    emitVecAtomLane(expr.first, vt, lane, targetWidth, isFloat);

    for (const auto &t: expr.rest) {
      emitVecAtomLane(t.atom, vt, lane, targetWidth, isFloat);
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

  void WasmBackend::emitVecAtomLane(
      const Atom &atom, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
      bool isFloat
  ) {
    std::visit(
        [this, &vt, lane, targetWidth, isFloat](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, CoefAtom>) {
            emitVecCoefLane(arg.coef, vt, lane, targetWidth, isFloat);
          } else if constexpr (std::is_same_v<T, RValueAtom>) {
            emitVecLValueLane(arg.rval, vt, lane, targetWidth, isFloat);
          } else if constexpr (std::is_same_v<T, OpAtom>) {
            emitVecOpAtomLane(arg, vt, lane, targetWidth, isFloat);
          } else if constexpr (std::is_same_v<T, SelectAtom>) {
            emitVecSelectAtomLane(arg, vt, lane, targetWidth, isFloat);
          } else if constexpr (std::is_same_v<T, UnaryAtom>) {
            emitVecUnaryAtomLane(arg, vt, lane, targetWidth, isFloat);
          } else if constexpr (std::is_same_v<T, CastAtom>) {
            emitVecCastAtomLane(arg, vt, lane, targetWidth);
          } else if constexpr (std::is_same_v<T, CmpAtom>) {
            emitVecCmpAtomLane(arg, vt, lane, targetWidth);
          } else if constexpr (std::is_same_v<T, CallAtom>) {
            // Materialized once by materializeVecCalls; each lane is a
            // plain load from the call's sret scratch slot.
            emitVecCallLane(arg, lane, targetWidth, isFloat);
          }
        },
        atom.v
    );
  }

  // ── Vector-returning calls in vector expressions ────────────
  //
  // Per-lane expression emission re-walks the rhs once per lane, so a
  // call atom left in place would re-invoke the callee N times — wrong
  // whenever the callee has side effects (e.g. mutates caller state
  // through a pointer parameter). The flat expression grammar confines
  // calls to top-level atom positions and to `select` mask expressions
  // (SelectVal arms and cmp operands admit no calls), so one shallow
  // walk finds every call: emit each exactly once, filling its sret
  // scratch slot, before the lane loop reads lanes back.

  void WasmBackend::materializeVecCalls(const Expr &expr) {
    auto handleAtom = [this](const Atom &a) {
      if (auto ca = std::get_if<CallAtom>(&a.v)) {
        auto [ptypes, rtype] = calleeSignature(*ca);
        (void) ptypes;
        if (rtype && std::holds_alternative<VecType>(rtype->v))
          emitCallAtom(*ca);
      } else if (auto sa = std::get_if<SelectAtom>(&a.v)) {
        if (sa->maskExpr)
          materializeVecCalls(*sa->maskExpr);
      }
    };
    handleAtom(expr.first);
    for (const auto &t: expr.rest)
      handleAtom(t.atom);
  }

  WasmBackend::VecCallSlots WasmBackend::vecCallSlots(const CallAtom &arg) {
    VecCallSlots slots;
    auto [ptypes, rtype] = calleeSignature(arg);
    auto align8 = [](std::uint32_t n) { return (n + 7u) & ~7u; };
    std::uint32_t disp = 0;
    slots.argDisp.assign(ptypes.size(), -1);
    for (size_t i = 0; i < ptypes.size(); ++i) {
      if (ptypes[i] && std::holds_alternative<VecType>(ptypes[i]->v)) {
        slots.argDisp[i] = disp;
        disp += align8(getTypeSize(ptypes[i]));
      }
    }
    if (rtype && std::holds_alternative<VecType>(rtype->v)) {
      slots.retDisp = disp;
      disp += align8(getTypeSize(rtype));
    }
    slots.totalBytes = disp;
    return slots;
  }

  void WasmBackend::emitVecLaneConvert(const TypePtr &elemTy, std::uint32_t targetWidth) {
    std::uint32_t width = getIntWidth(elemTy);
    bool valIsFloat = std::holds_alternative<FloatType>(elemTy->v);
    if (!valIsFloat) {
      if (width <= 32 && targetWidth > 32) {
        indent();
        out_ << "i64.extend_i32_s\n";
      } else if (width > 32 && targetWidth <= 32) {
        indent();
        out_ << "i32.wrap_i64\n";
      }
    } else {
      if (width == 32 && targetWidth == 64) {
        indent();
        out_ << "f64.promote_f32\n";
      } else if (width == 64 && targetWidth == 32) {
        indent();
        out_ << "f32.demote_f64\n";
      }
    }
  }

  void WasmBackend::emitVecCallLane(
      const CallAtom &arg, std::uint64_t lane, std::uint32_t targetWidth, bool isFloat
  ) {
    (void) isFloat;
    auto [ptypes, rtype] = calleeSignature(arg);
    (void) ptypes;
    if (!rtype || !std::holds_alternative<VecType>(rtype->v))
      return;
    const auto &rvt = std::get<VecType>(rtype->v);
    VecCallSlots slots = vecCallSlots(arg);
    std::int64_t off = static_cast<std::int64_t>(callVecScratch_.at(&arg)) - slots.retDisp;
    std::uint32_t elemSize = getTypeSize(rvt.elem);
    indent();
    out_ << "local.get $__old_sp\n";
    indent();
    out_ << "i32.const " << (off - static_cast<std::int64_t>(lane * elemSize)) << "\n";
    indent();
    out_ << "i32.sub\n";

    std::uint32_t width = getIntWidth(rvt.elem);
    bool valIsFloat = std::holds_alternative<FloatType>(rvt.elem->v);
    indent();
    if (valIsFloat) {
      out_ << (width == 32 ? "f32.load\n" : "f64.load\n");
    } else {
      out_
          << (width <= 8
                  ? "i32.load8_s\n"
                  : (width <= 16 ? "i32.load16_s\n" : (width <= 32 ? "i32.load\n" : "i64.load\n")));
    }
    emitVecLaneConvert(rvt.elem, targetWidth);
  }

  void WasmBackend::emitVecOpAtomLane(
      const OpAtom &arg, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
      bool isFloat
  ) {
    uint32_t wasmWidth = (targetWidth <= 32 ? 32 : 64);

    emitVecCoefLane(arg.coef, vt, lane, targetWidth, isFloat);
    emitVecLValueLane(arg.rval, vt, lane, targetWidth, isFloat);

    std::string prefix = (targetWidth <= 32 ? "i32." : "i64.");
    if (isFloat) {
      prefix = (targetWidth <= 32 ? "f32." : "f64.");
      if (arg.op == AtomOpKind::Mod) {
        auto emitX = [&]() { emitVecCoefLane(arg.coef, vt, lane, targetWidth, isFloat); };
        auto emitY = [&]() { emitVecLValueLane(arg.rval, vt, lane, targetWidth, isFloat); };

        indent();
        out_ << "drop\n";
        indent();
        out_ << "drop\n";

        emitX();
        emitX();
        emitY();
        indent();
        out_ << prefix << "div\n";
        // Same §2.9 intermediate-overflow trap as the
        // scalar fmod path — applied per lane (rule 21). Under
        // --no-ub-guards the (stack-neutral) check is elided; the
        // quotient stays on the stack from the `div` above.
        if (!noUbGuards_) {
          std::string qLocal = (targetWidth <= 32) ? "$__fmod_q_f32" : "$__fmod_q_f64";
          indent();
          out_ << "local.tee " << qLocal << "\n";
          indent();
          out_ << prefix << "abs\n";
          indent();
          out_ << prefix << "const inf\n";
          indent();
          out_ << prefix << "lt\n";
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
          indent();
          out_ << "local.get " << qLocal << "\n";
        }
        indent();
        out_ << prefix << "trunc\n";
        emitY();
        indent();
        out_ << prefix << "mul\n";
        indent();
        out_ << prefix << "sub\n";
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
      emitMask(targetWidth, wasmWidth);
      indent();
      out_ << (wasmWidth == 32 ? "i32.shr_u\n" : "i64.shr_u\n");
    } else {
      indent();
      std::string opStr;
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

  void WasmBackend::emitVecSelectAtomLane(
      const SelectAtom &arg, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
      bool isFloat
  ) {

    if (arg.maskExpr) {
      VecType maskVt{
          vt.size, std::make_shared<Type>(Type{IntType{IntType::Kind::ICustom, 1, {}}, {}}), {}
      };
      emitVecExprLane(*arg.maskExpr, maskVt, lane, 32, false);
    } else {
      emitCond(*arg.cond);
    }
    indent();
    std::string typePrefix =
        isFloat ? (targetWidth <= 32 ? "f32" : "f64") : (targetWidth <= 32 ? "i32" : "i64");
    out_ << "if (result " << typePrefix << ")\n";
    indent_level_++;
    emitVecSelectValLane(arg.vtrue, vt, lane, targetWidth, isFloat);
    indent_level_--;
    indent();
    out_ << "else\n";
    indent_level_++;
    emitVecSelectValLane(arg.vfalse, vt, lane, targetWidth, isFloat);
    indent_level_--;
    indent();
    out_ << "end\n";
  }

  void WasmBackend::emitVecUnaryAtomLane(
      const UnaryAtom &arg, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
      bool isFloat
  ) {
    uint32_t wasmWidth = (targetWidth <= 32 ? 32 : 64);

    emitVecLValueLane(arg.rval, vt, lane, targetWidth, isFloat);
    if (isFloat) {
      indent();
      out_ << (targetWidth <= 32 ? "f32.neg\n" : "f64.neg\n");
    } else {
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

  void WasmBackend::emitVecCastAtomLane(
      const CastAtom &arg, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth
  ) {
    uint32_t wasmWidth = (targetWidth <= 32 ? 32 : 64);

    bool srcIsFloat = false;
    uint32_t srcWidth = 32;
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
            out_ << "call " << mangleName(getMangledSymbolName(curFuncName_, src.name)) << "__"
                 << lane << "\n";
            srcWidth = 32;
            srcIsFloat = false;
            if (syms_.count(src.name)) {
              auto symTy = syms_.at(src.name);
              if (auto vt = std::get_if<VecType>(&symTy->v)) {
                symTy = vt->elem;
              }
              if (std::holds_alternative<FloatType>(symTy->v)) {
                srcIsFloat = true;
                srcWidth = (std::get<FloatType>(symTy->v).kind == FloatType::Kind::F32) ? 32 : 64;
              } else if (auto it = std::get_if<IntType>(&symTy->v)) {
                srcWidth = it->bits.value_or(32);
              }
            }
          } else {
            TypePtr srcTy = getLValueType(src);
            uint32_t realSrcWidth = 32;
            bool realSrcIsFloat = false;
            if (srcTy) {
              if (auto vt = std::get_if<VecType>(&srcTy->v)) {
                srcTy = vt->elem;
              }
              realSrcIsFloat = std::holds_alternative<FloatType>(srcTy->v);
              if (realSrcIsFloat) {
                realSrcWidth =
                    (std::get<FloatType>(srcTy->v).kind == FloatType::Kind::F32) ? 32 : 64;
              } else if (auto it = std::get_if<IntType>(&srcTy->v)) {
                realSrcWidth = it->bits.value_or(32);
              }
            }
            emitVecLValueLane(src, vt, lane, realSrcWidth, realSrcIsFloat);
            srcWidth = realSrcWidth;
            srcIsFloat = realSrcIsFloat;
          }
        },
        arg.src
    );

    TypePtr dstTy = arg.dstType;
    if (auto vt = std::get_if<VecType>(&dstTy->v)) {
      dstTy = vt->elem;
    }
    bool dstIsFloat = std::holds_alternative<FloatType>(dstTy->v);
    uint32_t dstWidth = getIntWidth(dstTy);
    if (dstIsFloat) {
      dstWidth = (std::get<FloatType>(dstTy->v).kind == FloatType::Kind::F32) ? 32 : 64;
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
      if (srcWidth <= 32 && dstWidth > 32)
        out_ << "i64.extend_i32_s\n";
      else if (srcWidth > 32 && dstWidth <= 32)
        out_ << "i32.wrap_i64\n";
    }
    if (!dstIsFloat)
      emitSignExtend(targetWidth, wasmWidth);
  }

  void WasmBackend::emitVecCmpAtomLane(
      const CmpAtom &arg, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth
  ) {

    TypePtr opTy = getSelectValType(arg.lhs);
    TypePtr elemTy = opTy;
    VecType opVt = vt;
    if (opTy && std::holds_alternative<VecType>(opTy->v)) {
      opVt = std::get<VecType>(opTy->v);
      elemTy = opVt.elem;
    }
    bool opIsFloat = elemTy && std::holds_alternative<FloatType>(elemTy->v);
    uint32_t opWidth = getIntWidth(elemTy);

    emitVecSelectValLane(arg.lhs, opVt, lane, opWidth, opIsFloat);
    emitVecSelectValLane(arg.rhs, opVt, lane, opWidth, opIsFloat);

    indent();
    std::string opStr;
    if (opIsFloat) {
      std::string prefix = (opWidth <= 32 ? "f32." : "f64.");
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
      std::string prefix = (opWidth <= 32 ? "i32." : "i64.");
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
    // vector lane cmp returns i1; sign-extend bit 0 so true is -1.
    emitSignExtend(1, (targetWidth > 32 ? 64 : 32));
  }

  void WasmBackend::emitVecCoefLane(
      const Coef &coef, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
      bool isFloat
  ) {
    uint32_t wasmWidth = (targetWidth <= 32 ? 32 : 64);
    std::visit(
        [this, &vt, lane, targetWidth, wasmWidth, isFloat](auto &&arg) {
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
            indent();
            out_ << "i32.const 0\n";
          } else {
            std::visit(
                [this, &vt, lane, targetWidth](auto &&id) {
                  using ID = std::decay_t<decltype(id)>;
                  if constexpr (std::is_same_v<ID, SymId>) {
                    indent();
                    out_ << "call " << mangleName(getMangledSymbolName(curFuncName_, id.name))
                         << "__" << lane << "\n";
                    std::uint32_t srcWidth = getIntWidth(vt.elem);
                    if (srcWidth <= 32 && targetWidth > 32) {
                      indent();
                      out_ << "i64.extend_i32_s\n";
                    } else if (srcWidth > 32 && targetWidth <= 32) {
                      indent();
                      out_ << "i32.wrap_i64\n";
                    }
                  } else {
                    emitVecLValueLane({id, {}, id.span}, vt, lane, targetWidth, false);
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

  void WasmBackend::emitVecLValueLane(
      const LValue &lv, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
      bool isFloat
  ) {
    (void) isFloat;
    if (!locals_.count(lv.base.name))
      return;
    const auto &info = locals_.at(lv.base.name);

    // Register-strategy storage: the strategy owns lane reads of
    // plain vector locals and params.
    if (lv.accesses.empty() && std::holds_alternative<VecType>(info.refractirType->v) &&
        !vecLowering_->usesFrameMemory()) {
      const auto &pvt = std::get<VecType>(info.refractirType->v);
      vecLowering_->emitLaneRead(*this, lv.base.name, pvt, lane);
      emitVecLaneConvert(pvt.elem, targetWidth);
      return;
    }

    // Whole-vector read of a vector *param*: the param local
    // holds the caller's spill address (the by-address boundary ABI), so
    // a lane read is a load at pointer + lane*elemSize.
    if (info.isParam && std::holds_alternative<VecType>(info.refractirType->v) &&
        lv.accesses.empty()) {
      const auto &pvt = std::get<VecType>(info.refractirType->v);
      std::uint32_t elemSize = getTypeSize(pvt.elem);
      indent();
      out_ << "local.get " << mangleName(lv.base.name) << "\n";
      if (lane > 0) {
        indent();
        out_ << "i32.const " << (lane * elemSize) << "\n";
        indent();
        out_ << "i32.add\n";
      }
      std::uint32_t width = getIntWidth(pvt.elem);
      bool valIsFloat = std::holds_alternative<FloatType>(pvt.elem->v);
      indent();
      if (valIsFloat) {
        out_ << (width == 32 ? "f32.load\n" : "f64.load\n");
      } else {
        out_
            << (width <= 8 ? "i32.load8_s\n"
                           : (width <= 16 ? "i32.load16_s\n"
                                          : (width <= 32 ? "i32.load\n" : "i64.load\n")));
      }
      emitVecLaneConvert(pvt.elem, targetWidth);
      return;
    }

    if (!info.isAggregate || !lv.accesses.empty()) {
      emitLValue(lv, false);
      TypePtr ty = getLValueType(lv);
      if (ty) {
        bool valIsFloat = std::holds_alternative<FloatType>(ty->v);
        std::uint32_t width = 32;
        if (valIsFloat) {
          width = (std::get<FloatType>(ty->v).kind == FloatType::Kind::F32) ? 32 : 64;
        } else if (auto it = std::get_if<IntType>(&ty->v)) {
          width = it->bits.value_or(32);
        }
        if (!valIsFloat) {
          if (width <= 32 && targetWidth > 32) {
            indent();
            out_ << "i64.extend_i32_s\n";
          } else if (width > 32 && targetWidth <= 32) {
            indent();
            out_ << "i32.wrap_i64\n";
          }
        } else {
          if (width == 32 && targetWidth == 64) {
            indent();
            out_ << "f64.promote_f32\n";
          } else if (width == 64 && targetWidth == 32) {
            indent();
            out_ << "f32.demote_f64\n";
          }
        }
      }
      return;
    }

    TypePtr elemTy = vt.elem;
    if (auto localVt = std::get_if<VecType>(&info.refractirType->v)) {
      elemTy = localVt->elem;
    }
    std::uint32_t elemSize = getTypeSize(elemTy);
    indent();
    out_ << "local.get $__old_sp\n";
    indent();
    out_ << "i32.const " << (info.offset - lane * elemSize) << "\n";
    indent();
    out_ << "i32.sub\n";

    std::uint32_t width = getIntWidth(elemTy);
    bool valIsFloat = std::holds_alternative<FloatType>(elemTy->v);

    indent();
    if (valIsFloat) {
      out_ << (width == 32 ? "f32.load\n" : "f64.load\n");
    } else {
      out_
          << (width <= 8
                  ? "i32.load8_s\n"
                  : (width <= 16 ? "i32.load16_s\n" : (width <= 32 ? "i32.load\n" : "i64.load\n")));
    }

    emitVecLaneConvert(elemTy, targetWidth);
  }

  // Initializer for a register-strategy vector local. Mirrors
  // the frame-memory emitInitVal semantics per lane (literals broadcast,
  // aggregate elements land per-lane, syms call the per-lane provider,
  // Local copies from the source's storage), but routes every write
  // through the strategy instead of address+store. Undef lanes are left
  // at the WASM local default (reading undef is UB; no sentinel needed).
  void
  WasmBackend::emitVecLocalInit(const std::string &name, const VecType &vt, const InitVal &iv) {
    std::uint32_t width = getIntWidth(vt.elem);
    std::uint32_t wasmW = (width <= 32) ? 32 : 64;
    bool isF = std::holds_alternative<FloatType>(vt.elem->v);

    auto pushElem = [&](const InitVal &el) {
      switch (el.kind) {
        case InitVal::Kind::Int:
          indent();
          if (isF) {
            out_ << (width <= 32 ? "f32.const " : "f64.const ") << std::get<IntLit>(el.value).value
                 << ".0\n";
          } else {
            out_ << (wasmW == 32 ? "i32.const " : "i64.const ") << std::get<IntLit>(el.value).value
                 << "\n";
            emitSignExtend(width, wasmW);
          }
          break;
        case InitVal::Kind::Float:
          indent();
          out_ << (width <= 32 ? "f32.const " : "f64.const ")
               << formatFloatLit(std::get<FloatLit>(el.value).value) << "\n";
          break;
        case InitVal::Kind::Atom:
          emitAtom(*std::get<AtomPtr>(el.value), width, isF);
          break;
        default:
          // Undef element: any value is fine (reading it back is UB).
          indent();
          out_ << (isF ? (width <= 32 ? "f32.const 0.0" : "f64.const 0.0")
                       : (wasmW == 32 ? "i32.const 0" : "i64.const 0"))
               << "\n";
          break;
      }
    };

    switch (iv.kind) {
      case InitVal::Kind::Undef:
        break;
      case InitVal::Kind::Aggregate: {
        const auto &elements = std::get<std::vector<InitValPtr>>(iv.value);
        for (std::uint64_t k = 0; k < vt.size && k < elements.size(); ++k) {
          if (elements[k]->kind == InitVal::Kind::Undef)
            continue;
          vecLowering_->emitLaneWrite(*this, name, vt, k, [&] { pushElem(*elements[k]); });
        }
        break;
      }
      case InitVal::Kind::Sym: {
        const auto &sid = std::get<SymId>(iv.value);
        for (std::uint64_t k = 0; k < vt.size; ++k) {
          vecLowering_->emitLaneWrite(*this, name, vt, k, [&] {
            indent();
            out_ << "call " << mangleName(getMangledSymbolName(curFuncName_, sid.name)) << "__" << k
                 << "\n";
            if (!isF)
              emitSignExtend(width, wasmW);
          });
        }
        break;
      }
      case InitVal::Kind::Local: {
        const auto &lid = std::get<LocalId>(iv.value);
        for (std::uint64_t k = 0; k < vt.size; ++k) {
          vecLowering_->emitLaneWrite(*this, name, vt, k, [&] {
            emitVecLValueLane({lid, {}, iv.span}, vt, k, width, isF);
          });
        }
        break;
      }
      default:
        // Int / Float / Atom broadcast into every lane.
        for (std::uint64_t k = 0; k < vt.size; ++k) {
          vecLowering_->emitLaneWrite(*this, name, vt, k, [&] { pushElem(iv); });
        }
        break;
    }
  }

  void WasmBackend::emitVecSelectValLane(
      const SelectVal &sv, const VecType &vt, std::uint64_t lane, std::uint32_t targetWidth,
      bool isFloat
  ) {
    if (std::holds_alternative<RValue>(sv)) {
      emitVecLValueLane(std::get<RValue>(sv), vt, lane, targetWidth, isFloat);
    } else {
      emitVecCoefLane(std::get<Coef>(sv), vt, lane, targetWidth, isFloat);
    }
  }
} // namespace refractir
