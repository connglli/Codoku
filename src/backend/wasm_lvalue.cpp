#include <sstream>
#include "analysis/type_utils.hpp"
#include "backend/wasm_backend.hpp"
#include "wasm_internal.hpp"

namespace refractir {

  void WasmBackend::emitAddress(const LValue &lv) {
    if (!locals_.count(lv.base.name))
      return;
    const auto &info = locals_.at(lv.base.name);

    if (info.isParam) {
      indent();
      out_ << "local.get " << mangleName(lv.base.name) << "\n";
    } else {
      indent();
      out_ << "local.get $__old_sp\n";
      indent();
      out_ << "i32.const " << info.offset << "\n";
      indent();
      out_ << "i32.sub\n";
    }

    TypePtr curType = info.refractirType;
    for (const auto &acc: lv.accesses) {
      if (auto ai = std::get_if<AccessIndex>(&acc)) {
        if (auto at = std::get_if<ArrayType>(&curType->v)) {
          std::uint32_t elemSize = getTypeSize(at->elem);
          emitIndex(ai->index);
          indent();
          out_ << "i32.const " << elemSize << "\n";
          indent();
          out_ << "i32.mul\n";
          indent();
          out_ << "i32.add\n";
          curType = at->elem;
        } else if (auto vt = std::get_if<VecType>(&curType->v)) {
          std::uint32_t elemSize = getTypeSize(vt->elem);
          emitIndex(ai->index);
          indent();
          out_ << "i32.const " << elemSize << "\n";
          indent();
          out_ << "i32.mul\n";
          indent();
          out_ << "i32.add\n";
          curType = vt->elem;
        }
      } else if (auto af = std::get_if<AccessField>(&acc)) {
        if (auto st = std::get_if<StructType>(&curType->v)) {
          if (structLayouts_.count(st->name.name)) {
            const auto &sinfo = structLayouts_.at(st->name.name);
            if (sinfo.fields.count(af->field)) {
              const auto &finfo = sinfo.fields.at(af->field);
              indent();
              out_ << "i32.const " << finfo.offset << "\n";
              indent();
              out_ << "i32.add\n";
              curType = finfo.type;
            }
          }
        }
      }
    }
  }

  void WasmBackend::emitLValue(const LValue &lv, bool isStore) {
    if (!locals_.count(lv.base.name))
      return;
    const auto &info = locals_.at(lv.base.name);
    // `%v[idx]` on a register-strategy vector local/param: the
    // strategy owns the lane read. Stores never reach here — lane writes
    // are intercepted at AssignInstr.
    if (!isStore && !lv.accesses.empty() &&
        std::holds_alternative<VecType>(info.refractirType->v) &&
        !vecLowering_->usesFrameMemory()) {
      const auto &vt = std::get<VecType>(info.refractirType->v);
      if (auto ai = std::get_if<AccessIndex>(&lv.accesses[0])) {
        if (auto lit = std::get_if<IntLit>(&ai->index)) {
          vecLowering_->emitLaneRead(*this, lv.base.name, vt, lit->value);
        } else {
          vecLowering_->emitLaneReadDyn(*this, lv.base.name, vt, [&] { emitIndex(ai->index); });
        }
        return;
      }
    }
    if (info.isAggregate || !lv.accesses.empty()) {
      if (!isStore) {
        emitAddress(lv);
        indent();
        TypePtr curType = info.refractirType;
        for (const auto &acc: lv.accesses) {
          if (std::get_if<AccessIndex>(&acc)) {
            if (auto at = std::get_if<ArrayType>(&curType->v))
              curType = at->elem;
            else if (auto vt = std::get_if<VecType>(&curType->v))
              curType = vt->elem;
          } else if (auto af = std::get_if<AccessField>(&acc)) {
            if (auto st = std::get_if<StructType>(&curType->v)) {
              auto &fld = af->field;
              if (structLayouts_.count(st->name.name) &&
                  structLayouts_.at(st->name.name).fields.count(fld))
                curType = structLayouts_.at(st->name.name).fields.at(fld).type;
            }
          }
        }
        std::uint32_t width = 0;
        bool valIsFloat = false;
        if (auto bits = TypeUtils::getIntBitWidth(curType)) {
          width = *bits;
        } else if (curType && std::holds_alternative<FloatType>(curType->v)) {
          valIsFloat = true;
          width = (std::get<FloatType>(curType->v).kind == FloatType::Kind::F32) ? 32 : 64;
        } else if (curType && std::holds_alternative<PtrType>(curType->v)) {
          width = 32;
        }

        if (valIsFloat) {
          out_ << (width == 32 ? "f32.load" : "f64.load") << "\n";
        } else {
          out_ << (width <= 8
                       ? "i32.load8_s"
                       : (width <= 16 ? "i32.load16_s" : (width <= 32 ? "i32.load" : "i64.load")))
               << "\n";
        }
      }
    } else {
      if (isStore) {
        indent();
        out_ << "local.set " << mangleName(lv.base.name) << "\n";
      } else {
        indent();
        out_ << "local.get " << mangleName(lv.base.name) << "\n";
      }
    }
  }

  void WasmBackend::emitInitVal(const InitVal &iv, const TypePtr &type, std::uint32_t baseOffset) {
    if (auto at = std::get_if<ArrayType>(&type->v)) {
      std::uint32_t elemSize = getTypeSize(at->elem);
      if (iv.kind == InitVal::Kind::Aggregate) {
        const auto &elements = std::get<std::vector<InitValPtr>>(iv.value);
        for (size_t i = 0; i < elements.size() && i < at->size; ++i) {
          emitInitVal(*elements[i], at->elem, baseOffset - i * elemSize);
        }
      } else if (iv.kind == InitVal::Kind::Local) {
        const auto &lid = std::get<LocalId>(iv.value);
        const auto &srcInfo = locals_.at(lid.name);
        emitCopy(type, baseOffset, lid.name, srcInfo.offset);
      } else {
        for (std::uint64_t i = 0; i < at->size; ++i) {
          emitInitVal(iv, at->elem, baseOffset - i * elemSize);
        }
      }
    } else if (auto vt = std::get_if<VecType>(&type->v)) {
      std::uint32_t elemSize = getTypeSize(vt->elem);
      if (iv.kind == InitVal::Kind::Aggregate) {
        const auto &elements = std::get<std::vector<InitValPtr>>(iv.value);
        for (size_t i = 0; i < elements.size() && i < vt->size; ++i) {
          emitInitVal(*elements[i], vt->elem, baseOffset - i * elemSize);
        }
      } else if (iv.kind == InitVal::Kind::Sym) {
        const auto &sid = std::get<SymId>(iv.value);
        for (std::uint64_t i = 0; i < vt->size; ++i) {
          indent();
          out_ << "local.get $__old_sp\n";
          indent();
          out_ << "i32.const " << (baseOffset - i * elemSize) << "\n";
          indent();
          out_ << "i32.sub\n";

          indent();
          out_ << "call " << mangleName(getMangledSymbolName(curFuncName_, sid.name)) << "__" << i
               << "\n";

          std::uint32_t dstWidth = getIntWidth(vt->elem);
          bool valIsFloat = std::holds_alternative<FloatType>(vt->elem->v);

          indent();
          if (valIsFloat) {
            out_ << (dstWidth == 32 ? "f32.store\n" : "f64.store\n");
          } else {
            out_ << (dstWidth <= 8
                         ? "i32.store8"
                         : (dstWidth <= 16 ? "i32.store16"
                                           : (dstWidth <= 32 ? "i32.store" : "i64.store")))
                 << "\n";
          }
        }
      } else if (iv.kind == InitVal::Kind::Local) {
        const auto &lid = std::get<LocalId>(iv.value);
        const auto &srcInfo = locals_.at(lid.name);
        emitCopy(type, baseOffset, lid.name, srcInfo.offset);
      } else {
        for (std::uint64_t i = 0; i < vt->size; ++i) {
          emitInitVal(iv, vt->elem, baseOffset - i * elemSize);
        }
      }
    } else if (auto st = std::get_if<StructType>(&type->v)) {
      if (structLayouts_.count(st->name.name)) {
        const auto &sinfo = structLayouts_.at(st->name.name);
        if (iv.kind == InitVal::Kind::Aggregate) {
          const auto &elements = std::get<std::vector<InitValPtr>>(iv.value);
          for (size_t i = 0; i < elements.size() && i < sinfo.fieldNames.size(); ++i) {
            const auto &fname = sinfo.fieldNames[i];
            const auto &finfo = sinfo.fields.at(fname);
            emitInitVal(*elements[i], finfo.type, baseOffset - finfo.offset);
          }
        } else if (iv.kind == InitVal::Kind::Local) {
          const auto &lid = std::get<LocalId>(iv.value);
          const auto &srcInfo = locals_.at(lid.name);
          emitCopy(type, baseOffset, lid.name, srcInfo.offset);
        } else {
          for (const auto &fname: sinfo.fieldNames) {
            const auto &finfo = sinfo.fields.at(fname);
            emitInitVal(iv, finfo.type, baseOffset - finfo.offset);
          }
        }
      }
    } else {
      // Leaf types (scalar or pointer stack slots)
      indent();
      out_ << "local.get $__old_sp\n";
      indent();
      out_ << "i32.const " << baseOffset << "\n";
      indent();
      out_ << "i32.sub\n";

      if (iv.kind == InitVal::Kind::Int) {
        indent();
        if (std::holds_alternative<FloatType>(type->v)) {
          out_ << (getIntWidth(type) <= 32 ? "f32.const " : "f64.const ")
               << std::get<IntLit>(iv.value).value << ".0\n";
        } else {
          out_ << (getIntWidth(type) <= 32 ? "i32.const " : "i64.const ")
               << std::get<IntLit>(iv.value).value << "\n";
        }
      } else if (iv.kind == InitVal::Kind::Float) {
        indent();
        out_ << (getIntWidth(type) <= 32 ? "f32.const " : "f64.const ")
             << formatFloatLit(std::get<FloatLit>(iv.value).value) << "\n";
      } else if (iv.kind == InitVal::Kind::Sym) {
        const auto &sid = std::get<SymId>(iv.value);
        indent();
        out_ << "call " << mangleName(getMangledSymbolName(curFuncName_, sid.name)) << "\n";
        std::uint32_t srcWidth = 32;
        bool srcIsFloat = false;
        if (syms_.count(sid.name)) {
          srcWidth = getIntWidth(syms_.at(sid.name));
          if (std::holds_alternative<FloatType>(syms_.at(sid.name)->v))
            srcIsFloat = true;
        }
        if (!srcIsFloat) {
          if (srcWidth <= 32 && getIntWidth(type) > 32) {
            indent();
            out_ << "i64.extend_i32_s\n";
          } else if (srcWidth > 32 && getIntWidth(type) <= 32) {
            indent();
            out_ << "i32.wrap_i64\n";
          }
        } else {
          if (srcWidth == 32 && getIntWidth(type) == 64) {
            indent();
            out_ << "f64.promote_f32\n";
          }
        }
      } else if (iv.kind == InitVal::Kind::Null) {
        indent();
        out_ << "i32.const 0\n";
      } else if (iv.kind == InitVal::Kind::Local) {
        const auto &lid = std::get<LocalId>(iv.value);
        const auto &srcInfo = locals_.at(lid.name);
        if (srcInfo.isAggregate) {
          indent();
          out_ << "local.get $__old_sp\n";
          indent();
          out_ << "i32.const " << srcInfo.offset << "\n";
          indent();
          out_ << "i32.sub\n";
          std::uint32_t width = 32;
          bool valIsFloat = false;
          if (auto bits = TypeUtils::getIntBitWidth(type)) {
            width = *bits;
          } else if (type && std::holds_alternative<FloatType>(type->v)) {
            valIsFloat = true;
            width = (std::get<FloatType>(type->v).kind == FloatType::Kind::F32) ? 32 : 64;
          }
          indent();
          if (valIsFloat) {
            out_ << (width == 32 ? "f32.load\n" : "f64.load\n");
          } else {
            out_ << (width <= 8
                         ? "i32.load8_u"
                         : (width <= 16 ? "i32.load16_u" : (width <= 32 ? "i32.load" : "i64.load")))
                 << "\n";
          }
        } else {
          indent();
          out_ << "local.get " << mangleName(lid.name) << "\n";
        }
      } else if (iv.kind == InitVal::Kind::Atom) {
        const auto &atom = std::get<AtomPtr>(iv.value);
        std::uint32_t width = 32;
        bool valIsFloat = false;
        if (auto bits = TypeUtils::getIntBitWidth(type)) {
          width = *bits;
        } else if (type && std::holds_alternative<FloatType>(type->v)) {
          valIsFloat = true;
          width = (std::get<FloatType>(type->v).kind == FloatType::Kind::F32) ? 32 : 64;
        }
        emitAtom(*atom, width, valIsFloat);
      } else {
        // Undef / default
        indent();
        if (std::holds_alternative<FloatType>(type->v)) {
          out_ << (getIntWidth(type) <= 32 ? "f32.const 0.0\n" : "f64.const 0.0\n");
        } else {
          out_ << (getIntWidth(type) <= 32 ? "i32.const 0\n" : "i64.const 0\n");
        }
      }

      std::uint32_t width = 0;
      bool valIsFloat = false;
      if (auto bits = TypeUtils::getIntBitWidth(type)) {
        width = *bits;
      } else if (type && std::holds_alternative<FloatType>(type->v)) {
        valIsFloat = true;
        width = (std::get<FloatType>(type->v).kind == FloatType::Kind::F32) ? 32 : 64;
      } else if (type && std::holds_alternative<PtrType>(type->v)) {
        width = 32;
      }

      indent();
      if (valIsFloat) {
        out_ << (width == 32 ? "f32.store\n" : "f64.store\n");
      } else {
        out_ << (width <= 8
                     ? "i32.store8"
                     : (width <= 16 ? "i32.store16" : (width <= 32 ? "i32.store" : "i64.store")))
             << "\n";
      }
    }
  }

  void WasmBackend::emitPtrExpr(const Expr &expr, const TypePtr &ptrType) {
    // Emit a pointer-valued expression as an i32 WASM address.
    // For ptr ± int, scale the integer offset by the pointee element size.
    const auto &pt = std::get<PtrType>(ptrType->v);
    uint32_t elemSize = getTypeSize(pt.pointee);

    emitAtom(expr.first, 32, false);

    for (const auto &t: expr.rest) {
      emitAtom(t.atom, 32, false);
      // Scale integer offset by element size (pointer arithmetic)
      if (elemSize > 1) {
        indent();
        out_ << "i32.const " << elemSize << "\n";
        indent();
        out_ << "i32.mul\n";
      }
      indent();
      out_ << (t.op == AddOp::Plus ? "i32.add\n" : "i32.sub\n");
    }
  }

  bool WasmBackend::isPtrDiff(const Expr &expr) const {
    // Match `ptr_lvalue - ptr_lvalue` as the entire expression. Per spec §6.8.6
    // this is the only mixed form that yields a non-pointer result (i64).
    if (expr.rest.size() != 1)
      return false;
    if (expr.rest[0].op != AddOp::Minus)
      return false;
    auto firstRv = std::get_if<RValueAtom>(&expr.first.v);
    auto secondRv = std::get_if<RValueAtom>(&expr.rest[0].atom.v);
    if (!firstRv || !secondRv)
      return false;
    auto firstIt = locals_.find(firstRv->rval.base.name);
    auto secondIt = locals_.find(secondRv->rval.base.name);
    if (firstIt == locals_.end() || secondIt == locals_.end())
      return false;
    return std::holds_alternative<PtrType>(firstIt->second.refractirType->v) &&
           std::holds_alternative<PtrType>(secondIt->second.refractirType->v);
  }

  void WasmBackend::emitPtrDiff(const Expr &expr) {
    // (q - p) in bytes, then divide by sizeof(pointee) to yield element distance.
    // Result is left on the stack as i64 (spec §6.8.6: ptr T - ptr T → i64).
    auto firstRv = std::get<RValueAtom>(expr.first.v);
    const auto &firstInfo = locals_.at(firstRv.rval.base.name);
    const auto &pt = std::get<PtrType>(firstInfo.refractirType->v);
    uint32_t elemSize = getTypeSize(pt.pointee);

    emitAtom(expr.first, 32, false);
    emitAtom(expr.rest[0].atom, 32, false);
    indent();
    out_ << "i32.sub\n";
    if (elemSize > 1) {
      indent();
      out_ << "i32.const " << elemSize << "\n";
      indent();
      out_ << "i32.div_s\n";
    }
    indent();
    out_ << "i64.extend_i32_s\n";
  }
} // namespace refractir
