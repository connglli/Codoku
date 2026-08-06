#include "reify/twin_mini.hpp"
#include "ast/build.hpp"

namespace refractir::reify {

  TypePtr leafType(const StateValue &v) {
    if (v.kind == StateValue::Kind::Float) {
      FloatType ft;
      ft.kind = v.bits == 32 ? FloatType::Kind::F32 : FloatType::Kind::F64;
      return std::make_shared<Type>(Type{ft, {}});
    }
    if (v.bits == 32)
      return buildI32();
    if (v.bits == 64)
      return buildI64();
    return std::make_shared<Type>(
        Type{IntType{IntType::Kind::ICustom, static_cast<int>(v.bits), {}}, {}}
    );
  }

  std::string leafKey(const std::string &root, const std::vector<Access> &path) {
    std::string k = root;
    for (const auto &acc: path) {
      if (auto af = std::get_if<AccessField>(&acc))
        k += "." + af->field;
      else
        k += "[" + std::to_string(std::get<IntLit>(std::get<AccessIndex>(acc).index).value) + "]";
    }
    return k;
  }

  std::optional<std::string> leafKey(const LValue &lv) {
    std::string k = lv.base.name;
    for (const auto &acc: lv.accesses) {
      if (auto af = std::get_if<AccessField>(&acc)) {
        k += "." + af->field;
        continue;
      }
      const auto &idx = std::get<AccessIndex>(acc).index;
      auto il = std::get_if<IntLit>(&idx);
      if (!il)
        return std::nullopt;
      k += "[" + std::to_string(il->value) + "]";
    }
    return k;
  }

  Expr ptrFixExpr(const std::optional<LValue> &target) {
    if (target)
      return buildAddrExpr(*target);
    return buildNullExpr();
  }

  InitVal stateToInit(const StateValue &v, const TypePtr &ty, const StructMap &structs) {
    InitVal iv;
    switch (v.kind) {
      case StateValue::Kind::Int:
        iv.kind = InitVal::Kind::Int;
        iv.value = IntLit{v.intVal, {}};
        break;
      case StateValue::Kind::Float:
        iv.kind = InitVal::Kind::Float;
        iv.value = FloatLit{v.floatVal, {}};
        break;
      case StateValue::Kind::Array:
      case StateValue::Kind::Vec: {
        TypePtr elemT;
        if (auto at = std::get_if<ArrayType>(&ty->v))
          elemT = at->elem;
        else
          elemT = std::get<VecType>(ty->v).elem;
        std::vector<InitValPtr> elems;
        elems.reserve(v.elems.size());
        for (const auto &e: v.elems)
          elems.push_back(std::make_shared<InitVal>(stateToInit(e, elemT, structs)));
        iv.kind = InitVal::Kind::Aggregate;
        iv.value = std::move(elems);
        break;
      }
      case StateValue::Kind::Struct: {
        const auto &st = std::get<StructType>(ty->v);
        const StructDecl *sd = structs.at(st.name.name);
        std::vector<InitValPtr> elems;
        elems.reserve(sd->fields.size());
        for (const auto &f: sd->fields) {
          const StateValue *fv = nullptr;
          for (const auto &[nm, val]: v.fields)
            if (nm == f.name) {
              fv = &val;
              break;
            }
          elems.push_back(
              std::make_shared<InitVal>(
                  fv ? stateToInit(*fv, f.type, structs)
                     : InitVal{InitVal::Kind::Undef, IntLit{}, {}}
              )
          );
        }
        iv.kind = InitVal::Kind::Aggregate;
        iv.value = std::move(elems);
        break;
      }
      case StateValue::Kind::Ptr:
        // Declared null; a fixup assign before the body sets the real
        // provenance (aggregate inits admit null but not addr atoms).
        iv.kind = InitVal::Kind::Null;
        iv.value = IntLit{};
        break;
      default:
        iv.kind = InitVal::Kind::Undef;
        iv.value = IntLit{};
        break;
    }
    return iv;
  }

  void declareRoots(
      const std::vector<MiniRoot> &roots, const StructMap &structs, std::vector<LetDecl> &lets
  ) {
    for (const auto &r: roots) {
      LetDecl d;
      d.isMutable = !r.isParam;
      d.name = LocalId{r.name, {}};
      d.type = r.type;
      d.init = stateToInit(r.init, r.type, structs);
      lets.push_back(std::move(d));
    }
  }

  std::vector<Instr> ptrInitInstrs(const std::vector<MiniRoot> &roots) {
    std::vector<Instr> out;
    for (const auto &r: roots)
      for (const auto &fx: r.ptrFixes)
        if (fx.initTarget)
          out.push_back(
              Instr{AssignInstr{buildLValue(r.name, fx.path), ptrFixExpr(fx.initTarget), {}}}
          );
    return out;
  }

  std::vector<Instr> ptrFinalInstrs(const std::vector<MiniRoot> &roots) {
    std::vector<Instr> out;
    for (const auto &r: roots)
      for (const auto &fx: r.ptrFixes)
        out.push_back(
            Instr{AssignInstr{buildLValue(r.name, fx.path), ptrFixExpr(fx.finalTarget), {}}}
        );
    return out;
  }

} // namespace refractir::reify
