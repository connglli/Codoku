#include <cmath>
#include <sstream>
#include "analysis/type_utils.hpp"
#include "backend/c_backend.hpp"
#include "c_internal.hpp"

namespace refractir {

  void CBackend::emitLValue(const LValue &lv) {
    // Vector lane access through the LValue path goes through the
    // strategy's emitLaneRead so each strategy controls its lane syntax
    // (vecext: `name[k]`; scalars: `name_k`; structarray: `name.lanes[k]`;
    // structscalars: `name.l<k>`).
    if (lv.accesses.size() == 1) {
      if (auto ai = std::get_if<AccessIndex>(&lv.accesses[0])) {
        auto baseTy = getLValueType(LValue{lv.base, {}, lv.span});
        if (baseTy && std::holds_alternative<VecType>(baseTy->v)) {
          auto &vt = std::get<VecType>(baseTy->v);
          // Render the index into a string; reuse emitIndex via a tmp stream.
          std::ostringstream tmp;
          std::ostream &orig = out_;
          std::streambuf *origBuf = orig.rdbuf(tmp.rdbuf());
          emitIndex(ai->index);
          orig.rdbuf(origBuf);
          std::string idxStr = tmp.str();
          // Dynamic lane indices need a runtime bounds check —
          // GCC vec-ext lane access doesn't trap on OOB by itself. For
          // an IntLit index the parser already pinned it, so skip the
          // check (the typechecker may also have rejected it).
          bool isLit = std::holds_alternative<IntLit>(ai->index);
          if (!isLit && !noUbGuards_) {
            // Wrap with a GCC statement-expression: evaluate idx once,
            // trap if out of bounds, then read the lane.
            std::string wrapped = "({ int64_t _vi = (" + idxStr +
                                  "); if ((uint64_t)_vi >= (uint64_t)" + std::to_string(vt.size) +
                                  "ull) __builtin_trap(); _vi; })";
            out_ << vecLowering_->emitLaneRead(mangleName(lv.base.name), vt, wrapped);
          } else {
            out_ << vecLowering_->emitLaneRead(mangleName(lv.base.name), vt, idxStr);
          }
          return;
        }
      }
    }
    out_ << mangleName(lv.base.name);
    for (const auto &acc: lv.accesses) {
      if (auto ai = std::get_if<AccessIndex>(&acc)) {
        out_ << "[";
        emitIndex(ai->index);
        out_ << "]";
      } else if (auto af = std::get_if<AccessField>(&acc)) {
        out_ << "." << af->field;
      }
    }
  }

  void CBackend::emitInitVal(const InitVal &iv, TypePtr expectedType) {
    // When ``expectedType`` is unknown (nullptr), leave the existing context
    // alone — the caller already established it (e.g., from the surrounding
    // let.type). Only override when we know the destination type locally.
    CtxGuard ctx(isDoubleCtx_, expectedType ? isOrContainsF64(expectedType) : isDoubleCtx_);
    switch (iv.kind) {
      case InitVal::Kind::Int:
        out_ << std::get<IntLit>(iv.value).value;
        break;
      case InitVal::Kind::Float:
        out_ << formatFloatLit(std::get<FloatLit>(iv.value).value);
        if (!isDoubleCtx_)
          out_ << "f";
        break;
      case InitVal::Kind::Sym:
        out_ << getMangledSymbolName(curFuncName_, std::get<SymId>(iv.value).name) << "()";
        break;
      case InitVal::Kind::Local:
        out_ << mangleName(std::get<LocalId>(iv.value).name);
        break;
      case InitVal::Kind::Undef:
        out_ << "0";
        break;
      case InitVal::Kind::Null:
        out_ << "NULL";
        break;
      case InitVal::Kind::Aggregate: {
        out_ << "{";
        const auto &elements = std::get<std::vector<InitValPtr>>(iv.value);
        for (size_t i = 0; i < elements.size(); ++i) {
          TypePtr elemType = nullptr;
          if (expectedType) {
            if (auto at = std::get_if<ArrayType>(&expectedType->v))
              elemType = at->elem;
            else if (auto st = std::get_if<StructType>(&expectedType->v))
              elemType = getStructFieldTypeAt(st->name.name, i);
          }
          emitInitVal(*elements[i], elemType);
          if (i + 1 < elements.size())
            out_ << ", ";
        }
        out_ << "}";
        break;
      }
      case InitVal::Kind::Atom: {
        // §3.4.2 atom-form init — emit the atom inline (the
        // typechecker has already verified the atom's type matches the
        // target).
        emitAtom(*std::get<AtomPtr>(iv.value));
        break;
      }
    }
  }
} // namespace refractir
