#include <atomic>
#include <optional>
#include <random>
#include <stdexcept>
#include "analysis/cfg.hpp"
#include "analysis/type_utils.hpp"
#include "internal.hpp"
#include "solver/solver.hpp"

namespace refractir {

  // §9.6.1 — symbolic interprocedural call. Phase 6: the
  // callee is restricted to straight-line CFG (single entry block whose
  // terminator is ret). Multi-block callees with conditional branches
  // would require a sub-path specification (deferred).
  //
  // `callerFun` / `callerStore` are non-null when invoked from a
  // call-site inside a symbolic execution frame. Callee `store`s that
  // land on a caller-side `let mut` (via a pointer parameter or any
  // pointer derived from one) are reflected into `callerStore` so the
  // caller observes the side effect — SPEC §9.6.1 step 4 mandates that
  // `Mem[T]` reflect callee stores.
  SymbolicValue SymbolicExecutor::callFunction(
      const FunDecl &callee, std::vector<SymbolicValue> args, smt::ISolver &solver,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub, const FunDecl *callerFun,
      SymbolicStore *callerStore
  ) {
    // Save caller frame state.
    const FunDecl *prevFun = currentFun_;
    auto savedProv = prov_.take();
    const FunDecl *prevOuterFun = outerFun_;
    SymbolicStore *prevOuterStore = outerStore_;
    const std::uint64_t prevFrame = currentFrame_;
    const std::uint64_t prevOuterFrame = outerFrame_;

    currentFun_ = &callee;
    // Expose the caller frame so LoadAtom / StoreInstr handlers can
    // route pointer-parameter accesses back to the caller's storage.
    outerFun_ = callerFun;
    outerStore_ = callerStore;
    // A fresh activation id, never reused within this solve. The callee's
    // locals are tagged under it, so once the frame is restored below its
    // pointers name storage no live cell carries.
    outerFrame_ = currentFrame_;
    currentFrame_ = nextFrame_++;

    SymbolicStore calleeStore;

    auto restoreFrame = [&]() {
      currentFun_ = prevFun;
      prov_.restore(std::move(savedProv));
      outerFun_ = prevOuterFun;
      outerStore_ = prevOuterStore;
      currentFrame_ = prevFrame;
      outerFrame_ = prevOuterFrame;
    };

    try {
      // Bind parameters.
      if (args.size() != callee.params.size())
        throw std::runtime_error(
            "Solver: arity mismatch calling " + callee.name.name + ": expected " +
            std::to_string(callee.params.size()) + ", got " + std::to_string(args.size())
        );
      for (size_t i = 0; i < callee.params.size(); ++i) {
        calleeStore[callee.params[i].name.name] = args[i];
      }

      // Bind syms via the shared per-FunDecl cache (one hole, one value).
      auto &symCache = calleeSyms_[&callee];
      for (const auto &s: callee.syms) {
        auto it = symCache.find(s.name.name);
        if (it == symCache.end()) {
          auto sv = createSymbolicValue(s.type, callee.name.name + "$" + s.name.name, solver, &pc);
          symCache[s.name.name] = sv;
          calleeStore[s.name.name] = sv;
        } else {
          calleeStore[s.name.name] = it->second;
        }
      }

      // Init lets.
      for (const auto &l: callee.lets) {
        if (l.init) {
          calleeStore[l.name.name] = evalInit(*l.init, l.type, solver, calleeStore, pc, ub);
        } else {
          calleeStore[l.name.name] = makeUndef(l.type, solver);
        }
      }

      // Build CFG and walk straight-line. Start at the first block; the
      // terminator must be either an unconditional br (chase it) or a
      // ret (capture the value). Conditional br on a non-constant cond
      // makes the call non-resolvable in Phase 6.
      DiagBag diags;
      CFG cfg = CFG::build(callee, diags);
      if (diags.hasErrors())
        throw std::runtime_error("CFG build failed for callee " + callee.name.name);

      std::size_t pcIdx = cfg.entry;
      // Cap each block at a small number of visits so the
      // random-sampling walker terminates even on callees with CFG
      // back-edges. SPEC §13 reserves user-supplied sub-path syntax to
      // make the choice exact -- this bound is the interim.
      constexpr int kMaxBlockVisits = 4;
      std::unordered_map<std::size_t, int> visitCount;
      SymbolicValue retVal(SymbolicValue::Kind::Undef);
      bool returned = false;

      while (!returned) {
        if (++visitCount[pcIdx] > kMaxBlockVisits)
          throw std::runtime_error(
              "Solver: callee " + callee.name.name + " exceeded " +
              std::to_string(kMaxBlockVisits) +
              " visits per block (loop unrolling cap reached -- use a sub-path "
              "spec when SPEC §13 lands)"
          );
        const Block &block = callee.blocks[pcIdx];

        for (const auto &ins: block.instrs) {
          std::visit(
              [&](auto &&arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, AssignInstr>) {
                  auto rhs = evalExpr(arg.rhs, solver, calleeStore, pc, ub);
                  setLValue(arg.lhs, rhs, solver, calleeStore, pc, ub);
                } else if constexpr (std::is_same_v<T, AssumeInstr>) {
                  pc.push_back(evalCond(arg.cond, solver, calleeStore, pc, ub));
                } else if constexpr (std::is_same_v<T, RequireInstr>) {
                  if (currentReq_)
                    currentReq_->push_back(evalCond(arg.cond, solver, calleeStore, pc, ub));
                } else if constexpr (std::is_same_v<T, StoreInstr>) {
                  // §9.6.1 step 4 — callee `store`s must update
                  // every reachable `let mut`, including caller-side
                  // ones whose addresses are exposed through pointer
                  // parameters.  Mirror the top-level handler's
                  // mux-update enumeration; run it on the callee frame
                  // (callee-local pointers) and the caller frame (so
                  // pointers passed in as parameters land their stores
                  // on the caller's view of memory).
                  SymbolicValue ptrVal = evalExpr(arg.ptr, solver, calleeStore, pc, ub);
                  smt::Term ptrTerm = ptrVal.term;

                  auto bv64Store = solver.make_bv_sort(kPtrBits);
                  auto nullStore = solver.make_bv_value_int64(bv64Store, 0);
                  ub.push_back(solver.make_term(smt::Kind::DISTINCT, {ptrTerm, nullStore}));
                  if (ptrVal.prov_base.internal && ptrVal.prov_size.internal) {
                    auto zero = solver.make_bv_value_int64(bv64Store, 0);
                    auto hasProv = solver.make_term(smt::Kind::DISTINCT, {ptrVal.prov_base, zero});
                    auto inLow = solver.make_term(smt::Kind::BV_ULE, {ptrVal.prov_base, ptrTerm});
                    auto endAddr =
                        solver.make_term(smt::Kind::BV_ADD, {ptrVal.prov_base, ptrVal.prov_size});
                    auto inHi = solver.make_term(smt::Kind::BV_ULT, {ptrTerm, endAddr});
                    auto cond = solver.make_term(
                        smt::Kind::IMPLIES,
                        {hasProv, solver.make_term(smt::Kind::AND, {inLow, inHi})}
                    );
                    ub.push_back(cond);
                  }

                  TypePtr pointeeType;
                  if (auto *rv = std::get_if<RValueAtom>(&arg.ptr.first.v)) {
                    TypePtr rvalType = resolveLValueType(rv->rval);
                    if (rvalType) {
                      if (auto pt = std::get_if<PtrType>(&rvalType->v))
                        pointeeType = pt->pointee;
                    }
                  }
                  if (!pointeeType)
                    throw std::runtime_error(
                        "store: cannot derive pointee type (only `store %p, ...` "
                        "with a ptr-typed local or parameter %p is currently supported)"
                    );

                  auto pointeeSort = getSort(pointeeType, solver);
                  SymbolicValue valVal =
                      evalExpr(arg.val, solver, calleeStore, pc, ub, std::optional(pointeeSort));
                  smt::Term valTerm = valVal.term;

                  auto bv64 = solver.make_bv_sort(kPtrBits);
                  std::vector<smt::Term> storeMatchConds;
                  std::function<
                      void(const TypePtr &, SymbolicValue &, std::uint64_t, std::uint64_t)>
                      enumStoreFrame;
                  enumStoreFrame = [&](const TypePtr &ty, SymbolicValue &sv, std::uint64_t baseTag,
                                       std::uint64_t off) {
                    if (!ty)
                      return;
                    if (typeMatch(ty, pointeeType)) {
                      auto tagTerm =
                          solver.make_bv_value_int64(bv64, static_cast<int64_t>(baseTag + off));
                      auto cond = solver.make_term(smt::Kind::EQUAL, {ptrTerm, tagTerm});
                      storeMatchConds.push_back(cond);
                      sv.term = solver.make_term(smt::Kind::ITE, {cond, valTerm, sv.term});
                      // Pointer pointee: provenance follows the value
                      // under `cond`, same as the top-level handler (see there).
                      if (std::get_if<PtrType>(&pointeeType->v)) {
                        auto zeroP = solver.make_bv_value_int64(bv64, 0);
                        auto curBase = sv.prov_base.internal ? sv.prov_base : zeroP;
                        auto curSize = sv.prov_size.internal ? sv.prov_size : zeroP;
                        auto newBase = valVal.prov_base.internal ? valVal.prov_base : zeroP;
                        auto newSize = valVal.prov_size.internal ? valVal.prov_size : zeroP;
                        sv.prov_base = solver.make_term(smt::Kind::ITE, {cond, newBase, curBase});
                        sv.prov_size = solver.make_term(smt::Kind::ITE, {cond, newSize, curSize});
                      }
                      // Writing makes the cell as defined as the stored value
                      // (see the top-level handler).
                      auto curDef = sv.is_defined.internal ? sv.is_defined : solver.make_true();
                      auto valDef =
                          valVal.is_defined.internal ? valVal.is_defined : solver.make_true();
                      sv.is_defined = solver.make_term(smt::Kind::ITE, {cond, valDef, curDef});
                      return;
                    }
                    if (auto at = std::get_if<ArrayType>(&ty->v)) {
                      std::uint64_t stride = sizeofTagUnits(at->elem, structs_);
                      for (std::uint64_t k = 0; k < at->size && k < sv.arrayVal.size(); ++k)
                        enumStoreFrame(at->elem, sv.arrayVal[k], baseTag, off + k * stride);
                      return;
                    }
                    if (auto st = std::get_if<StructType>(&ty->v)) {
                      auto sIt = structs_.find(st->name.name);
                      if (sIt == structs_.end())
                        return;
                      std::uint64_t fOff = 0;
                      for (const auto &f: sIt->second->fields) {
                        auto fIt = sv.structVal.find(f.name);
                        if (fIt != sv.structVal.end())
                          enumStoreFrame(f.type, fIt->second, baseTag, off + fOff);
                        fOff += sizeofTagUnits(f.type, structs_);
                      }
                      return;
                    }
                  };

                  // Callee frame: stores rooted in callee-local addresses.
                  for (const auto &l: callee.lets) {
                    auto it = calleeStore.find(l.name.name);
                    if (it == calleeStore.end())
                      continue;
                    enumStoreFrame(l.type, it->second, tagOfLocal(currentFrame_, l.name.name), 0);
                  }
                  // Caller frame: stores rooted in caller-visible
                  // addresses exposed via pointer parameters.  The mux
                  // condition `ptrTerm == tag_of(caller_local) + off`
                  // is automatically false when the call site passed a
                  // pointer to a different local, so the wrong frame
                  // never receives a write.
                  if (callerFun && callerStore) {
                    for (const auto &l: callerFun->lets) {
                      auto it = callerStore->find(l.name.name);
                      if (it == callerStore->end())
                        continue;
                      enumStoreFrame(l.type, it->second, tagOfLocal(outerFrame_, l.name.name), 0);
                    }
                  }
                  if (!storeMatchConds.empty()) {
                    smt::Term anyMatch = storeMatchConds[0];
                    for (size_t j = 1; j < storeMatchConds.size(); ++j)
                      anyMatch = solver.make_term(smt::Kind::OR, {anyMatch, storeMatchConds[j]});
                    ub.push_back(anyMatch);
                  }
                }
              },
              ins
          );
        }

        std::visit(
            [&](auto &&t) {
              using T = std::decay_t<decltype(t)>;
              if constexpr (std::is_same_v<T, RetTerm>) {
                if (t.value) {
                  retVal = evalExpr(*t.value, solver, calleeStore, pc, ub);
                }
                returned = true;
              } else if constexpr (std::is_same_v<T, BrTerm>) {
                if (!t.isConditional) {
                  pcIdx = cfg.indexOf.at(t.dest.name);
                } else {
                  // Conditional branch in callee: sample one
                  // path randomly. The chosen branch's cond is
                  // conjoined to PC so the SMT search stays consistent.
                  // SPEC §13 lists user-supplied sub-paths as planned
                  // future work; random sampling is the interim.
                  smt::Term condTerm = evalCond(*t.cond, solver, calleeStore, pc, ub);
                  bool takeThen = std::uniform_int_distribution<int>(0, 1)(calleeRng_) == 0;
                  if (takeThen) {
                    pc.push_back(condTerm);
                    pcIdx = cfg.indexOf.at(t.thenLabel.name);
                  } else {
                    pc.push_back(solver.make_term(smt::Kind::NOT, {condTerm}));
                    pcIdx = cfg.indexOf.at(t.elseLabel.name);
                  }
                }
              } else if constexpr (std::is_same_v<T, UnreachableTerm>) {
                ub.push_back(solver.make_false());
                returned = true;
              }
            },
            block.term
        );
      }

      restoreFrame();
      return retVal;
    } catch (...) {
      restoreFrame();
      throw;
    }
  }

  // §9.6.2 contract-form `decl` expansion.
  SymbolicValue SymbolicExecutor::callContract(
      const ExtDecl &decl, const std::vector<std::shared_ptr<Expr>> &argExprs,
      std::vector<SymbolicValue> args, smt::ISolver &solver, SymbolicStore &callerStore,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {
    if (args.size() != decl.params.size())
      throw std::runtime_error(
          "Solver: arity mismatch calling " + decl.name.name + ": expected " +
          std::to_string(decl.params.size()) + ", got " + std::to_string(args.size())
      );
    if (!decl.contract)
      throw std::runtime_error("Solver: callContract on non-contract decl");

    // §9.6.2 step 4: havoc the storage backing every
    // pointer argument's provenance object before the post-state is
    // assumed. The contract may write through any pointer parameter,
    // so we replace the source local's symbolic value with a fresh
    // constant. Resolution walks the call-site argument expression to
    // find the root local; we handle the two most common forms:
    //   - `addr %x` -> root is %x directly.
    //   - plain `%p` (or `%p` with no accesses) where %p has a known
    //     provenance entry -> reverse the FNV-1a tag against caller
    //     locals/params to find the source.
    // Other forms (load-derived pointers, ptr arithmetic with
    // unknown provenance, ptrindex/ptrfield) fall back to "no havoc";
    // callers that need them today can constrain post-state via the
    // contract clauses directly.
    auto findRootLocal = [&](const Expr &e) -> std::string {
      if (!e.rest.empty())
        return {};
      auto &a = e.first;
      if (auto addr = std::get_if<AddrAtom>(&a.v)) {
        return addr->lv.base.name;
      }
      if (auto rv = std::get_if<RValueAtom>(&a.v)) {
        if (!rv->rval.accesses.empty())
          return {};
        const std::string &name = rv->rval.base.name;
        auto pe = prov_.lookup(name);
        if (!pe)
          return {};
        uint64_t targetTag = pe->baseTag;
        if (!currentFun_)
          return {};
        for (const auto &l: currentFun_->lets)
          if (tagOfLocal(currentFrame_, l.name.name) == targetTag)
            return l.name.name;
        for (const auto &p: currentFun_->params)
          if (tagOfLocal(currentFrame_, p.name.name) == targetTag)
            return p.name.name;
        return {};
      }
      return {};
    };

    auto havocLocal = [&](const std::string &name) {
      auto it = callerStore.find(name);
      if (it == callerStore.end())
        return;
      // Find the local's declared type.
      TypePtr ty;
      if (currentFun_) {
        for (const auto &l: currentFun_->lets)
          if (l.name.name == name) {
            ty = l.type;
            break;
          }
        if (!ty)
          for (const auto &p: currentFun_->params)
            if (p.name.name == name) {
              ty = p.type;
              break;
            }
      }
      if (!ty)
        return;
      static std::atomic<uint64_t> havocCounter{0};
      uint64_t n = havocCounter.fetch_add(1, std::memory_order_relaxed);
      std::string freshName = decl.name.name + "$havoc$" + name + "$" + std::to_string(n);
      it->second = createSymbolicValue(ty, freshName, solver, &pc);
    };

    for (size_t i = 0; i < decl.params.size(); ++i) {
      const auto &paramTy = decl.params[i].type;
      if (!paramTy || !std::holds_alternative<PtrType>(paramTy->v))
        continue;
      std::string rootName = findRootLocal(*argExprs[i]);
      if (!rootName.empty())
        havocLocal(rootName);
    }

    // Build the contract's eval store: start with the caller's store
    // (so load through pointer parameters sees the caller's locals --
    // including the freshly-havoc'd ones), then layer in the
    // contract's parameter bindings.
    SymbolicStore contractStore = callerStore;
    for (size_t i = 0; i < decl.params.size(); ++i) {
      contractStore[decl.params[i].name.name] = args[i];
    }

    // Save caller fun context. Build a synthetic FunDecl that mixes
    // the caller's locals (so load-enumeration inside the contract can
    // see the caller's memory) with the decl's params and a `ret`
    // local (so resolveLValueType picks up the contract's identifiers).
    const FunDecl *prevFun = currentFun_;
    auto synth = std::make_unique<FunDecl>();
    synth->name = decl.name;
    if (prevFun) {
      synth->params = prevFun->params;
      synth->lets.reserve(prevFun->lets.size() + 1);
      for (const auto &l: prevFun->lets) {
        LetDecl copy;
        copy.isMutable = l.isMutable;
        copy.name = l.name;
        copy.type = l.type;
        synth->lets.push_back(std::move(copy));
      }
    }
    // Append the contract's params (won't collide with caller's name
    // space in practice; param names are local to the contract).
    for (const auto &p: decl.params)
      synth->params.push_back(p);
    synth->retType = decl.retType;
    LetDecl retLet;
    retLet.isMutable = false;
    retLet.name = LocalId{"ret", {}};
    retLet.type = decl.retType;
    synth->lets.push_back(std::move(retLet));
    currentFun_ = synth.get();
    auto restore = [&]() { currentFun_ = prevFun; };

    try {
      // Step 1 (§9.6.2.1): evaluate each pre clause; conjoin to PC.
      // Violation prunes the caller's path (rule 23).
      for (const auto &pre: decl.contract->pres) {
        auto cond = evalCond(pre.cond, solver, contractStore, pc, ub);
        pc.push_back(cond);
      }

      // Step 2 (§9.6.2.2): fresh symbolic ret_sym.
      SymbolicValue retSym =
          createSymbolicValue(decl.retType, decl.name.name + "$ret", solver, &pc);

      // Step 3 (§9.6.2.3): assume each post clause with `ret` bound.
      contractStore["ret"] = retSym;
      for (const auto &post: decl.contract->posts) {
        auto cond = evalCond(post.cond, solver, contractStore, pc, ub);
        pc.push_back(cond);
      }

      restore();
      return retSym;
    } catch (...) {
      restore();
      throw;
    }
  }
} // namespace refractir
