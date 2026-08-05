#include <sstream>
#include "analysis/type_utils.hpp"
#include "backend/wasm_backend.hpp"
#include "wasm_internal.hpp"

namespace refractir {

  TypePtr WasmBackend::getLValueType(const LValue &lv) {
    if (!locals_.count(lv.base.name))
      return nullptr;
    const auto &info = locals_.at(lv.base.name);
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
    return curType;
  }

  TypePtr WasmBackend::getSelectValType(const SelectVal &sv) {
    if (std::holds_alternative<RValue>(sv)) {
      return getLValueType(std::get<RValue>(sv));
    } else {
      const auto &coef = std::get<Coef>(sv);
      return std::visit(
          [this](auto &&arg) -> TypePtr {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, IntLit>) {
              return std::make_shared<Type>(Type{IntType{IntType::Kind::I32, 32, {}}, {}});
            } else if constexpr (std::is_same_v<T, FloatLit>) {
              return std::make_shared<Type>(Type{FloatType{FloatType::Kind::F64, {}}, {}});
            } else if constexpr (std::is_same_v<T, NullLit>) {
              return std::make_shared<Type>(Type{PtrType{nullptr, {}}, {}});
            } else {
              return std::visit(
                  [this](auto &&id) -> TypePtr {
                    using ID = std::decay_t<decltype(id)>;
                    if constexpr (std::is_same_v<ID, SymId>) {
                      if (syms_.count(id.name))
                        return syms_.at(id.name);
                    } else {
                      if (locals_.count(id.name))
                        return locals_.at(id.name).refractirType;
                    }
                    return nullptr;
                  },
                  arg
              );
            }
          },
          coef
      );
    }
  }

  TypePtr WasmBackend::getCoefType(const Coef &coef) {
    return std::visit(
        [this](auto &&arg) -> TypePtr {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, IntLit>) {
            // Int literals are polymorphic in RefractIR; we return I32/I64
            // based on the literal value for overload resolution.
            auto t = std::make_shared<Type>();
            if (arg.value > 2147483647LL || arg.value < -2147483648LL) {
              t->v = IntType{IntType::Kind::I64, {}, {}};
            } else {
              t->v = IntType{IntType::Kind::I32, {}, {}};
            }
            return t;
          } else if constexpr (std::is_same_v<T, FloatLit>) {
            auto t = std::make_shared<Type>();
            t->v = FloatType{FloatType::Kind::F64, {}};
            return t;
          } else if constexpr (std::is_same_v<T, NullLit>) {
            auto t = std::make_shared<Type>();
            t->v = PtrType{nullptr, {}};
            return t;
          } else {
            return std::visit(
                [this](auto &&id) -> TypePtr {
                  using ID = std::decay_t<decltype(id)>;
                  if constexpr (std::is_same_v<ID, SymId>) {
                    if (syms_.count(id.name))
                      return syms_.at(id.name);
                  } else {
                    if (locals_.count(id.name))
                      return locals_.at(id.name).refractirType;
                  }
                  return nullptr;
                },
                arg
            );
          }
        },
        coef
    );
  }

  TypePtr WasmBackend::getAtomType(const Atom &atom) {
    return std::visit(
        [this](auto &&arg) -> TypePtr {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, OpAtom>) {
            return getLValueType(arg.rval);
          } else if constexpr (std::is_same_v<T, UnaryAtom>) {
            return getLValueType(arg.rval);
          } else if constexpr (std::is_same_v<T, SelectAtom>) {
            if (std::holds_alternative<RValue>(arg.vtrue)) {
              return getLValueType(std::get<RValue>(arg.vtrue));
            } else {
              return getCoefType(std::get<Coef>(arg.vtrue));
            }
          } else if constexpr (std::is_same_v<T, CoefAtom>) {
            return getCoefType(arg.coef);
          } else if constexpr (std::is_same_v<T, RValueAtom>) {
            return getLValueType(arg.rval);
          } else if constexpr (std::is_same_v<T, CastAtom>) {
            return arg.dstType;
          } else if constexpr (std::is_same_v<T, AddrAtom>) {
            auto t = std::make_shared<Type>();
            t->v = PtrType{getLValueType(arg.lv), {}};
            return t;
          } else if constexpr (std::is_same_v<T, LoadAtom>) {
            auto pt = getLValueType(arg.rval);
            if (pt) {
              if (auto ptr = std::get_if<PtrType>(&pt->v)) {
                return ptr->pointee;
              }
            }
            return nullptr;
          } else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
            auto pt = getLValueType(arg.rval);
            if (pt) {
              if (auto ptr = std::get_if<PtrType>(&pt->v)) {
                if (auto at = std::get_if<ArrayType>(&ptr->pointee->v)) {
                  auto resPtr = std::make_shared<Type>();
                  resPtr->v = PtrType{at->elem, {}};
                  return resPtr;
                }
              }
            }
            return nullptr;
          } else if constexpr (std::is_same_v<T, PtrFieldAtom>) {
            auto pt = getLValueType(arg.rval);
            if (pt) {
              if (auto ptr = std::get_if<PtrType>(&pt->v)) {
                if (auto st = std::get_if<StructType>(&ptr->pointee->v)) {
                  if (structLayouts_.count(st->name.name)) {
                    const auto &layout = structLayouts_.at(st->name.name);
                    if (layout.fields.count(arg.field)) {
                      auto resPtr = std::make_shared<Type>();
                      resPtr->v = PtrType{layout.fields.at(arg.field).type, {}};
                      return resPtr;
                    }
                  }
                }
              }
            }
            return nullptr;
          } else if constexpr (std::is_same_v<T, CallAtom>) {
            // Honour the overload pinned by the type checker;
            // see CallAtom::resolvedIntrinsic.
            if (arg.resolvedIntrinsic)
              return arg.resolvedIntrinsic->retType;
            if (prog_) {
              for (const auto &f: prog_->funs) {
                if (f.name.name == arg.callee.name)
                  return f.retType;
              }
              for (const auto &d: prog_->extDecls) {
                if (d.name.name == arg.callee.name)
                  return d.retType;
              }
              const IntrinsicDecl *intr = nullptr;
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
              if (intr)
                return intr->retType;
            }
            return nullptr;
          }
          return nullptr;
        },
        atom.v
    );
  }

  TypePtr WasmBackend::getExprType(const Expr &expr) { return getAtomType(expr.first); }
} // namespace refractir
