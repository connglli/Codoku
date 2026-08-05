#include <cassert>
#include <cmath>
#include <sstream>
#include "analysis/type_utils.hpp"
#include "backend/c_backend.hpp"
#include "c_internal.hpp"

namespace refractir {

  TypePtr CBackend::getLValueType(const LValue &lv) {
    auto it = varTypes_.find(lv.base.name);
    // Every let-local, param, and sym is recorded in ``recordVar`` at the
    // top of each function; reaching this assert means a code path emitted
    // an lvalue whose base was never declared — that would silently mis-flag
    // the float-evaluation context and produce a wrong result.
    assert(it != varTypes_.end() && "lvalue base not in varTypes_");
    if (it == varTypes_.end())
      return nullptr;
    TypePtr cur = it->second;
    for (const auto &acc: lv.accesses) {
      if (!cur)
        return nullptr;
      if (std::holds_alternative<AccessIndex>(acc)) {
        if (auto at = std::get_if<ArrayType>(&cur->v))
          cur = at->elem;
        else if (auto vt = std::get_if<VecType>(&cur->v))
          cur = vt->elem;
        else if (auto pt = std::get_if<PtrType>(&cur->v))
          cur = pt->pointee;
        else
          return nullptr;
      } else if (auto af = std::get_if<AccessField>(&acc)) {
        if (auto st = std::get_if<StructType>(&cur->v))
          cur = findStructFieldType(st->name.name, af->field);
        else
          return nullptr;
      }
    }
    return cur;
  }

  TypePtr
  CBackend::findStructFieldType(const std::string &structName, const std::string &fieldName) const {
    auto it = structFields_.find(structName);
    if (it == structFields_.end())
      return nullptr;
    for (const auto &[name, type]: it->second)
      if (name == fieldName)
        return type;
    return nullptr;
  }

  TypePtr CBackend::getStructFieldTypeAt(const std::string &structName, size_t idx) const {
    auto it = structFields_.find(structName);
    if (it == structFields_.end() || idx >= it->second.size())
      return nullptr;
    return it->second[idx].second;
  }

  TypePtr CBackend::getCoefType(const Coef &coef) {
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
            // Float literals inherit the surrounding ``isDoubleCtx_`` —
            // see the header note: callers must invoke this BEFORE setting
            // their own context, so the answer reflects the outer scope.
            auto t = std::make_shared<Type>();
            t->v = FloatType{isDoubleCtx_ ? FloatType::Kind::F64 : FloatType::Kind::F32, {}};
            return t;
          } else if constexpr (std::is_same_v<T, NullLit>) {
            auto t = std::make_shared<Type>();
            t->v = PtrType{nullptr, {}};
            return t;
          } else {
            return std::visit(
                [this](auto &&id) -> TypePtr {
                  auto it = varTypes_.find(id.name);
                  if (it != varTypes_.end()) {
                    return it->second;
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

  TypePtr CBackend::getAtomType(const Atom &atom) {
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
            // ptrindex p, i : ptr [N] T → ptr T
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
            // ptrfield p, f : ptr @S → ptr FieldType
            auto pt = getLValueType(arg.rval);
            if (pt) {
              if (auto ptr = std::get_if<PtrType>(&pt->v)) {
                if (auto st = std::get_if<StructType>(&ptr->pointee->v)) {
                  auto sit = structFields_.find(st->name.name);
                  if (sit != structFields_.end()) {
                    for (const auto &[name, ty]: sit->second) {
                      if (name == arg.field) {
                        auto resPtr = std::make_shared<Type>();
                        resPtr->v = PtrType{ty, {}};
                        return resPtr;
                      }
                    }
                  }
                }
              }
            }
            return nullptr;
          } else if constexpr (std::is_same_v<T, CallAtom>) {
            // Use the overload the type checker pinned onto
            // the AST node. Fall back to the local heuristic only for
            // un-typechecked input.
            if (arg.resolvedIntrinsic)
              return arg.resolvedIntrinsic->retType;
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
            return nullptr;
          }
          return nullptr;
        },
        atom.v
    );
  }

  // RefractIR requires every atom in an Expr to share a single type, so the
  // first atom's type is the whole expression's type.
  TypePtr CBackend::getExprType(const Expr &expr) { return getAtomType(expr.first); }

  TypePtr CBackend::getInitValType(const InitVal &iv) {
    switch (iv.kind) {
      case InitVal::Kind::Int: {
        auto t = std::make_shared<Type>();
        t->v = IntType{IntType::Kind::I64, {}, {}};
        return t;
      }
      case InitVal::Kind::Float: {
        auto t = std::make_shared<Type>();
        t->v = FloatType{isDoubleCtx_ ? FloatType::Kind::F64 : FloatType::Kind::F32, {}};
        return t;
      }
      case InitVal::Kind::Sym: {
        auto it = varTypes_.find(std::get<SymId>(iv.value).name);
        return (it != varTypes_.end()) ? it->second : nullptr;
      }
      case InitVal::Kind::Local: {
        auto it = varTypes_.find(std::get<LocalId>(iv.value).name);
        return (it != varTypes_.end()) ? it->second : nullptr;
      }
      case InitVal::Kind::Atom: {
        return getAtomType(*std::get<AtomPtr>(iv.value));
      }
      default:
        return nullptr;
    }
  }
} // namespace refractir
