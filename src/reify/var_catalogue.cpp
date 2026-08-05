#include "reify/var_catalogue.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include "reify/hyperparameters.hpp"

#include "reify/ast_builder.hpp"

namespace refractir::reify {

  // ---------------------------------------------------------------------------
  // VarCatalogue accessors
  // ---------------------------------------------------------------------------

  std::vector<const VarEntry *> VarCatalogue::scalarsOf(const TypePtr &t) const {
    std::vector<const VarEntry *> result;
    for (const auto &v: vars)
      if (isScalarType(v.type) && typeEquals(v.type, t))
        result.push_back(&v);
    return result;
  }

  std::vector<const VarEntry *> VarCatalogue::allScalars() const {
    std::vector<const VarEntry *> result;
    for (const auto &v: vars)
      if (isScalarType(v.type))
        result.push_back(&v);
    return result;
  }

  std::vector<const VarEntry *> VarCatalogue::vecsOf(const TypePtr &t) const {
    std::vector<const VarEntry *> result;
    for (const auto &v: vars)
      if (isVecType(v.type) && typeEquals(v.type, t))
        result.push_back(&v);
    return result;
  }

  std::vector<const VarEntry *> VarCatalogue::vecsWithElem(const TypePtr &elemT) const {
    std::vector<const VarEntry *> result;
    for (const auto &v: vars)
      if (isVecType(v.type) && typeEquals(std::get<VecType>(v.type->v).elem, elemT))
        result.push_back(&v);
    return result;
  }

  std::vector<const VarEntry *> VarCatalogue::ptrsOf(const TypePtr &pointeeT) const {
    std::vector<const VarEntry *> result;
    for (const auto &v: vars)
      if (isPtrType(v.type) && typeEquals(pointeeType(v.type), pointeeT))
        result.push_back(&v);
    return result;
  }

  std::vector<const VarEntry *> VarCatalogue::ptrsToPtr() const {
    std::vector<const VarEntry *> result;
    for (const auto &v: vars)
      if (isPtrType(v.type) && isPtrType(pointeeType(v.type)))
        result.push_back(&v);
    return result;
  }

  std::vector<const VarEntry *> VarCatalogue::addressable() const {
    std::vector<const VarEntry *> result;
    // Per spec §3.5.2 `addr` requires a `let mut` root, so
    // parameters (immutable) are NOT addressable.
    for (const auto &v: vars)
      if (!v.isParam && !isPtrType(v.type) && !isAggType(v.type))
        result.push_back(&v);
    return result;
  }

  std::vector<const VarEntry *> VarCatalogue::allAddressable() const {
    std::vector<const VarEntry *> result;
    // Per spec §3.5.2 `addr` only requires a `let mut` root, so every
    // non-param, non-vec local qualifies: scalars, pointers and whole
    // aggregates alike. The aggregates have to be offered here — a
    // `ptr [N] T` / `ptr @S` LHS reassign in `genPtrAtom` needs the
    // whole-aggregate target that pointer-arithmetic sources
    // (`ptr [N] T_scalar`) are built on, and the sub-lvalue walk alone
    // never produces one. Vectors stay excluded: spec §3 forbids `addr`
    // on vector locals.
    for (const auto &v: vars)
      if (!v.isParam && !isVecType(v.type))
        result.push_back(&v);
    return result;
  }

  const VarEntry *VarCatalogue::findAny(const TypePtr &t, std::mt19937 &rng) const {
    std::vector<const VarEntry *> candidates;
    for (const auto &v: vars)
      if (typeEquals(v.type, t))
        candidates.push_back(&v);
    if (candidates.empty())
      return nullptr;
    std::uniform_int_distribution<int> d(0, (int) candidates.size() - 1);
    return candidates[d(rng)];
  }

  const VarEntry *VarCatalogue::findAddressableOfType(const TypePtr &t, std::mt19937 &rng) const {
    std::vector<const VarEntry *> candidates;
    for (const auto &v: vars)
      if (!v.isParam && !isAggType(v.type) && typeEquals(v.type, t))
        candidates.push_back(&v);
    if (candidates.empty())
      return nullptr;
    std::uniform_int_distribution<int> d(0, (int) candidates.size() - 1);
    return candidates[d(rng)];
  }

  // ---------------------------------------------------------------------------
  // genVarCatalogue
  // ---------------------------------------------------------------------------

  namespace {

    // Builds a VarCatalogue one phase at a time. A pointer can only target a
    // variable that already exists, so the phases run in dependency order:
    // non-pointer vars, then `ptr T`, then `ptr ptr T`, then aggregate pointers
    // and the locals that stage a navigation through them, and finally the
    // function parameters. The naming counters and the struct-declaration
    // helper are shared across all of them, which is what makes this a builder
    // rather than a chain of free functions.
    struct CatalogueBuilder {
      CatalogueBuilder(std::mt19937 &r, const VarGenConfig &c) :
          rng(r), cfg(c), tcfg(c.typeConfig) {}

      std::mt19937 &rng;
      const VarGenConfig &cfg;
      const TypeGenConfig &tcfg;

      VarCatalogue cat;
      int structIdx = 0;
      int scalarIdx = 0;
      int arrayIdx = 0;
      int vecIdx = 0;
      int ptrIdx = 0;
      int ptrPtrIdx = 0;
      std::uniform_real_distribution<double> prob{0.0, 1.0};

      // Split cfg.nVars between the phases — see rysmith::hp::kFracNonPtrVars /
      // kFracPtr1Vars.
      int nNonPtr = std::max(1, (int) (cfg.nVars * rysmith::hp::kFracNonPtrVars));
      int nPtr1 = std::max(0, (int) (cfg.nVars * rysmith::hp::kFracPtr1Vars));
      int nPtr2 = std::max(0, cfg.nVars - nNonPtr - nPtr1);

      // Helper: create a struct declaration. Fields are independently drawn —
      // ~30% chance of an array field ([N] scalar) to produce struct-of-arrays.
      std::string makeStructDecl(const TypePtr &) {
        // Namespace struct names by `<id>_<funcIdx>_<j>`. The
        // funcIdx slot is needed because sibling funcs in the same
        // rysmith run share `cfg.genId`: without it two siblings each
        // declare `@struct_<id>_0` with different contents and rylink's
        // bundle stage trips on the collision when merging them.
        std::string sname = "@struct_" + (cfg.genId.empty() ? std::string("RY") : cfg.genId) + "_" +
                            std::to_string(cfg.funcIdx) + "_" + std::to_string(structIdx++);
        std::uniform_int_distribution<int> nfd(1, std::max(1, tcfg.maxAggElems));
        int nf = nfd(rng);
        StructDecl sd;
        sd.name = GlobalId{sname, {}};
        for (int fi = 0; fi < nf; fi++) {
          FieldDecl fd;
          fd.name = "f" + std::to_string(fi);
          // Struct field is an array with probability rysmith::hp::kPStructFieldIsArray
          if (prob(rng) < rysmith::hp::kPStructFieldIsArray && tcfg.maxAggElems >= 1) {
            std::uniform_int_distribution<int> szd(1, std::max(1, tcfg.maxAggElems));
            uint64_t sz = (uint64_t) szd(rng);
            ArrayType at;
            at.size = sz;
            at.elem = genScalarType(rng, tcfg.enableFp);
            fd.type = std::make_shared<Type>(Type{at, {}});
          } else {
            fd.type = genScalarType(rng, tcfg.enableFp);
          }
          sd.fields.push_back(std::move(fd));
        }
        cat.structDecls.push_back(std::move(sd));
        return sname;
      }

      void addNonPointerVars() {
        for (int i = 0; i < nNonPtr; i++) {
          // Draw a type that is not ptr
          TypePtr t = genRandomType(rng, tcfg, 0);

          // If we drew a ptr type at depth 0, convert to scalar fallback
          // (ptr handling is reserved for phases 2/3). Same for vec if disabled.
          if (isPtrType(t)) {
            t = genScalarType(rng, tcfg.enableFp);
          }
          if (isVecType(t) && !tcfg.enableVec) {
            t = genScalarType(rng, tcfg.enableFp);
          }

          VarEntry v;
          v.type = t;

          if (isScalarType(t)) {
            v.name = "%v" + std::to_string(scalarIdx++);
          } else if (isVecType(t)) {
            // Vec var — named %vec0, %vec1, etc.
            v.name = "%vec" + std::to_string(vecIdx++);
          } else if (std::holds_alternative<ArrayType>(t->v)) {
            auto at = std::get<ArrayType>(t->v);
            // If element is a struct placeholder, resolve it to a real struct decl
            // (array-of-struct) rather than replacing with a scalar.
            if (std::holds_alternative<StructType>(at.elem->v)) {
              std::string elemSname = makeStructDecl(at.elem);
              StructType st;
              st.name = GlobalId{elemSname, {}};
              at.elem = std::make_shared<Type>(Type{st, {}});
              v.type = std::make_shared<Type>(Type{at, {}});
              // Store elem struct name for expr/checksum lookups
              v.structTypeName = elemSname;
            }
            v.name = "%a" + std::to_string(arrayIdx++);
          } else if (std::holds_alternative<StructType>(t->v)) {
            // Assign a real struct name
            std::string sname = makeStructDecl(t);
            v.structTypeName = sname;
            // Update the type to reference the correct struct name
            StructType st;
            st.name = GlobalId{sname, {}};
            v.type = std::make_shared<Type>(Type{st, {}});
            v.name = "%t" + std::to_string(structIdx - 1);
          }
          cat.vars.push_back(std::move(v));
        }

        // Guarantee at least one i32 scalar (needed as fallback RValue)
        auto i32scalars = cat.scalarsOf(makeI32());
        if (i32scalars.empty()) {
          VarEntry v;
          v.name = "%v" + std::to_string(scalarIdx++);
          v.type = makeI32();
          cat.vars.push_back(std::move(v));
        }
      }

      void addPointerVars() {
        // Each pointer must target an existing non-ptr, non-agg var.
        // We snapshot names+types BEFORE pushing to cat.vars to avoid invalidating
        // pointers when the vector reallocates.
        if (tcfg.maxPtrDepth >= 1) {
          struct TargetInfo {
            std::string name;
            TypePtr type;
          };

          std::vector<TargetInfo> addressableTargets;
          for (const auto &v: cat.vars)
            if (!isPtrType(v.type) && !isAggType(v.type) && !isVecType(v.type))
              addressableTargets.push_back({v.name, v.type});

          for (int i = 0; i < nPtr1 && !addressableTargets.empty(); i++) {
            std::uniform_int_distribution<int> pd(0, (int) addressableTargets.size() - 1);
            const TargetInfo &target = addressableTargets[pd(rng)];

            VarEntry pv;
            pv.name = "%p" + std::to_string(ptrIdx++);
            pv.ptrTarget = target.name;

            PtrType pt;
            pt.pointee = target.type;
            pv.type = std::make_shared<Type>(Type{pt, {}});

            cat.vars.push_back(std::move(pv));
          }
        }
      }

      void addPointerPointerVars() {
        // Each `ptr ptr T` must target an existing pointer var.
        // Snapshot before the loop to avoid dangling pointers on vector reallocation.
        if (tcfg.maxPtrDepth >= 2 && nPtr2 > 0) {
          struct TargetInfo {
            std::string name;
            TypePtr type;
          };

          std::vector<TargetInfo> ptrTargets;
          for (const auto &v: cat.vars)
            if (isPtrType(v.type))
              ptrTargets.push_back({v.name, v.type});

          for (int i = 0; i < nPtr2 && !ptrTargets.empty(); i++) {
            std::uniform_int_distribution<int> pd(0, (int) ptrTargets.size() - 1);
            const TargetInfo &target = ptrTargets[pd(rng)];

            VarEntry ppv;
            ppv.name = "%pp" + std::to_string(ptrPtrIdx++);
            ppv.ptrTarget = target.name;

            PtrType pt;
            pt.pointee = target.type;
            ppv.type = std::make_shared<Type>(Type{pt, {}});

            cat.vars.push_back(std::move(ppv));
          }
        }
      }

      void addAggregatePointers() {
        // Aggregate pointers: `ptr [N] T` and `ptr @S`.
        // Each points to an existing array or struct var.
        if (tcfg.enableAggPtr && tcfg.maxPtrDepth >= 1) {
          struct AggTarget {
            std::string name;
            TypePtr type; // the array or struct type
          };

          std::vector<AggTarget> aggTargets;
          for (const auto &v: cat.vars) {
            if (std::holds_alternative<ArrayType>(v.type->v))
              aggTargets.push_back({v.name, v.type});
            else if (std::holds_alternative<StructType>(v.type->v))
              aggTargets.push_back({v.name, v.type});
          }
          int nAggPtr = std::max(0, (int) (cfg.nVars * rysmith::hp::kFracAggPtrVars));
          for (int i = 0; i < nAggPtr && !aggTargets.empty(); i++) {
            std::uniform_int_distribution<int> ad(0, (int) aggTargets.size() - 1);
            const auto &tgt = aggTargets[ad(rng)];
            VarEntry apv;
            apv.name = "%ap" + std::to_string(i);
            apv.ptrTarget = tgt.name;
            PtrType pt;
            pt.pointee = tgt.type;
            apv.type = std::make_shared<Type>(Type{pt, {}});
            cat.vars.push_back(std::move(apv));
          }
        }
      }

      void addNavigationStaging() {
        // Navigation-staging pointers. Chained
        // ptrindex/ptrfield navigation of an aggregate pointer
        // (`ptr [N][M] F -> ptr [M] F -> ptr F`, `ptr [N] @S -> ptr @S -> ...`)
        // stages each intermediate through a `let mut` local of the sub-aggregate
        // pointer type. Those staging types essentially never coexist with their
        // source by chance, so inject ONE shared local per distinct sub-aggregate
        // pointer type reachable from any aggregate pointer (deduped against
        // existing vars). Target-less (start `undef`, written by the generated
        // chain before any read). See tryEmitNavChain.
        {
          std::vector<TypePtr> subAggs;
          std::function<void(const TypePtr &)> collect = [&](const TypePtr &a) {
            if (auto *at = std::get_if<ArrayType>(&a->v)) {
              if (isAggType(at->elem)) {
                subAggs.push_back(at->elem);
                collect(at->elem);
              }
            } else if (auto *st = std::get_if<StructType>(&a->v)) {
              for (const auto &sd: cat.structDecls) {
                if (sd.name.name != st->name.name)
                  continue;
                for (const auto &f: sd.fields)
                  if (isAggType(f.type)) {
                    subAggs.push_back(f.type);
                    collect(f.type);
                  }
                break;
              }
            }
          };
          // Snapshot of current pointer pointees: don't grow `cat.vars` mid-scan.
          std::vector<TypePtr> aggPointees;
          for (const auto &v: cat.vars)
            if (isPtrType(v.type)) {
              auto p = pointeeType(v.type);
              if (p && isAggType(p))
                aggPointees.push_back(p);
            }
          for (const auto &p: aggPointees)
            collect(p);

          int navIdx = 0;
          std::vector<TypePtr> injected;
          auto present = [&](const TypePtr &ptee) {
            for (const auto &v: cat.vars)
              if (isPtrType(v.type) && typeEquals(pointeeType(v.type), ptee))
                return true;
            for (const auto &t: injected)
              if (typeEquals(t, ptee))
                return true;
            return false;
          };
          for (const auto &sub: subAggs) {
            if (present(sub))
              continue;
            injected.push_back(sub);
            VarEntry nv;
            nv.name = "%nav" + std::to_string(navIdx++);
            PtrType pt;
            pt.pointee = sub;
            nv.type = std::make_shared<Type>(Type{pt, {}});
            cat.vars.push_back(std::move(nv));
          }
        }
      }

      void addParameters() {
        // Scalar function parameters. Generated last so
        // they sit at known offsets but participate in every later helper's
        // RValue lookup (allScalars / scalarsOf / findAny respect them).
        // Per spec §3.5.2 they're immutable — never targets of `addr` and
        // never on the LHS of an assignment; the addressable() /
        // allAddressable() / findAny() / findAddressableOfType() filters
        // below already exclude them via the `isParam` flag.
        for (int i = 0; i < cfg.nParams; i++) {
          VarEntry pv;
          // `%pa<i>` for parameters; `%p<i>` is taken by ptr vars,
          // `%pp<i>` by ptr-ptr, `%ppp<i>` (if ever generated) by ptr-ptr-ptr.
          pv.name = "%pa" + std::to_string(i);
          pv.type = genScalarType(rng, tcfg.enableFp);
          pv.isParam = true;
          cat.vars.push_back(std::move(pv));
        }
      }
    };

  } // namespace

  VarCatalogue genVarCatalogue(std::mt19937 &rng, const VarGenConfig &cfg) {
    CatalogueBuilder b(rng, cfg);
    b.addNonPointerVars();
    b.addPointerVars();
    b.addPointerPointerVars();
    b.addAggregatePointers();
    b.addNavigationStaging();
    b.addParameters();
    return std::move(b.cat);
  }

} // namespace refractir::reify
