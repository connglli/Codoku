#include "reify/checksum.hpp"
#include "reify/ast_builder.hpp"

#include "analysis/type_utils.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fstream>
#include <sstream>

#include "analysis/definite_init.hpp"
#include "analysis/pass_manager.hpp"
#include "analysis/reachability.hpp"
#include "analysis/unused_name.hpp"
#include "ast/sir_printer.hpp"
#include "backend/c_backend.hpp"
#include "backend/c_vec_lowering.hpp"
#include "backend/wasm_backend.hpp"
#include "frontend/diagnostics.hpp"
#include "frontend/lexer.hpp"
#include "frontend/parser.hpp"
#include "frontend/semchecker.hpp"
#include "frontend/typechecker.hpp"
#include "interp/interpreter.hpp"
#include "reify/type_gen.hpp"

namespace fs = std::filesystem;
using namespace refractir;

namespace refractir::reify {


  // ---------------------------------------------------------------------------
  // Exit-block CRC32 rewriter — see include/reify/checksum.hpp for usage
  // ---------------------------------------------------------------------------

  namespace {

    // Build a single-atom Expr — used to wrap the state/val operands as
    // Expr arguments to the @crc32_update call.
    Expr atomToExpr(Atom a) { return Expr{std::move(a), {}, {}}; }

    // Build an Expr that reads `%_chk`. Both the state arg of @crc32_update
    // and the LHS-side `%_chk = ...` reference the same accumulator local.
    Expr chkReadExpr() {
      LValue lv = LValue{LocalId{"%_chk", {}}, {}, {}};
      Atom a = Atom{RValueAtom{std::move(lv), {}}, {}};
      return atomToExpr(std::move(a));
    }

    // True when `a` is `RValueAtom(%_chk)` with no accesses — the marker we
    // use to find the running-state operand in a buildSumChecksum addition.
    bool isChkRead(const Atom &a) {
      auto *rv = std::get_if<RValueAtom>(&a.v);
      if (!rv)
        return false;
      if (rv->rval.base.name != "%_chk")
        return false;
      if (!rv->rval.accesses.empty())
        return false;
      return true;
    }

    // True when `instr` is the `%_chk = 0;` init that buildSumChecksum emits
    // before any accumulation. Matches a single-atom RHS of integer-literal
    // value 0; the init is left untouched by the rewrite.
    bool isChkInitZero(const AssignInstr &ai) {
      if (ai.lhs.base.name != "%_chk")
        return false;
      if (!ai.lhs.accesses.empty())
        return false;
      if (!ai.rhs.rest.empty())
        return false;
      auto *ca = std::get_if<CoefAtom>(&ai.rhs.first.v);
      if (!ca)
        return false;
      auto *il = std::get_if<IntLit>(&ca->coef);
      return il && il->value == 0;
    }

    // Ensure `@crc32_update(state: i32, val: i32) : i32` is declared at the
    // program level. Idempotent: only appends a new IntrinsicDecl when no
    // matching signature is already present.
    void ensureCrc32UpdateDecl(Program &prog) {
      ensureIntrinsicDecl(
          prog, "@crc32_update", makeI32(), {{"%state", makeI32()}, {"%val", makeI32()}}
      );
    }

    static const StructDecl *
    lookupStruct(const TypeUtils::StructTable &structs, const std::string &nm) {
      auto it = structs.find(nm);
      return it == structs.end() ? nullptr : it->second;
    }

    // Packed size of `t`, in the byte scale the solver reports offsets in.
    static std::uint64_t packedSize(const TypeUtils::StructTable &structs, const TypePtr &t) {
      return TypeUtils::packedSizeof(t, structs);
    }

    // Type-equality probe restricted to the type families a RefractIR pointer
    // pointee can name. It decides where the offset walk stops: a `ptr T` must
    // land on a sub-value of type T, not on a leaf inside T.
    static bool sameLeafType(const TypePtr &a, const TypePtr &b) {
      if (!a || !b)
        return a == b;
      if (a->v.index() != b->v.index())
        return false;
      if (auto pa = std::get_if<IntType>(&a->v)) {
        auto pb = std::get_if<IntType>(&b->v);
        if (pa->kind != pb->kind)
          return false;
        if (pa->kind == IntType::Kind::ICustom)
          return pa->bits.value_or(32) == pb->bits.value_or(32);
        return true;
      }
      if (auto pa = std::get_if<FloatType>(&a->v))
        return pa->kind == std::get<FloatType>(b->v).kind;
      if (auto pa = std::get_if<PtrType>(&a->v))
        return sameLeafType(pa->pointee, std::get<PtrType>(b->v).pointee);
      if (auto pa = std::get_if<ArrayType>(&a->v)) {
        auto pb = std::get_if<ArrayType>(&b->v);
        return pa->size == pb->size && sameLeafType(pa->elem, pb->elem);
      }
      if (auto pa = std::get_if<VecType>(&a->v)) {
        auto pb = std::get_if<VecType>(&b->v);
        return pa->size == pb->size && sameLeafType(pa->elem, pb->elem);
      }
      if (auto pa = std::get_if<StructType>(&a->v))
        return pa->name.name == std::get<StructType>(b->v).name.name;
      return false;
    }

    // Walk type `t` for a contiguous in-object offset (packed bytes — the scale
    // the solver reports LetExitValue::targetOffset in), appending the field and
    // index accesses that reach a sub-value whose type matches `target`. Stops at
    // the first matching (type, remOff == 0) pair, so a `ptr [N] T` lands on the
    // whole array rather than its first scalar leaf. Returns true and narrows `t`
    // to the matched sub-type; false means the offset lands on no sub-value of the
    // requested type, which happens when the pointer pointee disagrees with the
    // base local's leaf layout.
    //
    // The size model here must be the one the solver measured with, or the walk
    // reaches the wrong leaf and this oracle silently disagrees with the program
    // it is meant to check.
    static bool resolveOffsetPath(
        TypePtr &t, std::uint64_t &remOff, std::vector<Access> &acc, const TypePtr &target,
        const TypeUtils::StructTable &structs
    ) {
      if (!t)
        return false;
      if (remOff == 0 && sameLeafType(t, target))
        return true;
      if (auto at = std::get_if<ArrayType>(&t->v)) {
        uint64_t stride = packedSize(structs, at->elem);
        if (stride == 0)
          return false;
        uint64_t idx = remOff / stride;
        if (idx >= at->size)
          return false;
        acc.push_back(AccessIndex{Index{IntLit{(int64_t) idx, {}}}, {}});
        remOff -= idx * stride;
        t = at->elem;
        return resolveOffsetPath(t, remOff, acc, target, structs);
      }
      if (auto vt = std::get_if<VecType>(&t->v)) {
        uint64_t stride = packedSize(structs, vt->elem);
        if (stride == 0)
          return false;
        uint64_t idx = remOff / stride;
        if (idx >= vt->size)
          return false;
        acc.push_back(AccessIndex{Index{IntLit{(int64_t) idx, {}}}, {}});
        remOff -= idx * stride;
        t = vt->elem;
        return resolveOffsetPath(t, remOff, acc, target, structs);
      }
      if (auto st = std::get_if<StructType>(&t->v)) {
        const StructDecl *sd = lookupStruct(structs, st->name.name);
        if (!sd)
          return false;
        for (const auto &f: sd->fields) {
          uint64_t fsz = packedSize(structs, f.type);
          if (remOff < fsz) {
            acc.push_back(AccessField{f.name, {}});
            t = f.type;
            return resolveOffsetPath(t, remOff, acc, target, structs);
          }
          remOff -= fsz;
        }
        return false;
      }
      // Scalar / ptr already exhausted aggregate walks; only a
      // typeEq match at remOff==0 (handled at the top) succeeds.
      return false;
    }

  } // namespace

  size_t rewriteExitToCrc32Checksum(
      refractir::Program &prog, const std::string &funcName,
      const std::unordered_map<std::string, refractir::SymbolicExecutor::LetExitValue>
          &letExitValues
  ) {
    // A pointer let may be checksummed only if the solver resolved its
    // exit-time target to a live object (Ptr kind, non-empty targetLocal).
    // An unresolved target (undef pointer, cross-object arithmetic) means
    // `load %p` would dereference an address the solver never constrained —
    // UB the strict interpreter traps on but the SUM never exercised.
    auto pointerLoadable = [&](const std::string &name) {
      auto it = letExitValues.find(name);
      return it != letExitValues.end() &&
             it->second.kind == refractir::SymbolicExecutor::LetExitValue::Kind::Ptr &&
             !it->second.targetLocal.empty();
    };
    // 1. Find the entry function.
    FunDecl *entry = nullptr;
    std::string canonical = funcName.empty() || funcName[0] == '@' ? funcName : "@" + funcName;
    for (auto &f: prog.funs) {
      if (f.name.name == canonical) {
        entry = &f;
        break;
      }
    }
    if (!entry)
      return 0;

    // 2. Find its exit block (terminator is RetTerm). buildSumChecksum always
    // emits its chain there; if no RetTerm block exists, the function
    // wasn't shaped by genFunction and there's nothing for us to rewrite.
    Block *exit = nullptr;
    for (auto &b: entry->blocks) {
      if (std::holds_alternative<RetTerm>(b.term)) {
        exit = &b;
        break;
      }
    }
    if (!exit)
      return 0;

    // Idempotency guard: a `@crc32_update` call in the exit block can only
    // come from a prior run of this rewrite (buildSumChecksum emits a plain
    // `+` sum). Steps 1-3 already no-op on a second pass, but step 4 would
    // re-append the pointer-load chain and push duplicate `%_pld_*` scratch
    // lets. Bail so the whole rewrite is idempotent.
    for (const auto &instr: exit->instrs) {
      const auto *ai = std::get_if<AssignInstr>(&instr);
      if (!ai)
        continue;
      const auto *call = std::get_if<CallAtom>(&ai->rhs.first.v);
      if (call && call->callee.name == "@crc32_update")
        return 0;
    }

    // 3. Walk instructions. Skip the `%_chk = 0;` init. For each
    // `%_chk = %_chk + <atom>` (or `<atom> + %_chk` for cast-first form),
    // extract the addend and replace the RHS with
    // `call @crc32_update(%_chk, <addend>)`. Any other shape (a regular
    // assignment, a require, etc.) is left as-is — we never silently
    // drop instructions in case future versions of buildSumChecksum add
    // pre/post-amble work alongside the accumulator.
    size_t updates = 0;
    for (auto &instr: exit->instrs) {
      auto *ai = std::get_if<AssignInstr>(&instr);
      if (!ai)
        continue;
      if (ai->lhs.base.name != "%_chk" || !ai->lhs.accesses.empty())
        continue;
      if (isChkInitZero(*ai))
        continue;
      // buildSumChecksum always emits exactly one tail `+ atom`. Defensively
      // bail on any other RHS shape so we don't corrupt unrelated code.
      if (ai->rhs.rest.size() != 1)
        continue;
      if (ai->rhs.rest[0].op != AddOp::Plus)
        continue;
      Atom *addendPtr = nullptr;
      if (isChkRead(ai->rhs.first)) {
        addendPtr = &ai->rhs.rest[0].atom;
      } else if (isChkRead(ai->rhs.rest[0].atom)) {
        addendPtr = &ai->rhs.first;
      } else {
        continue;
      }

      CallAtom ca;
      ca.callee = GlobalId{"@crc32_update", {}};
      ca.args.push_back(std::make_shared<Expr>(chkReadExpr()));
      ca.args.push_back(std::make_shared<Expr>(atomToExpr(std::move(*addendPtr))));
      ai->rhs = atomToExpr(Atom{std::move(ca), {}});
      ++updates;
    }

    // 4. Append `load %p` → CRC32 steps for every pointer let with a
    // scalar non-pointer pointee. These were intentionally left out
    // of the SUM (see buildSumChecksum's comment) so the solver
    // never had to encode load-dispatch in the checksum; here we
    // splice them back in for runtime opacity. The added steps go
    // BEFORE the existing terminator (still a `ret %_chk;`) and the
    // accumulator is the same `%_chk` the chain above leaves at the
    // end. Scratch lets `%_pld_<n>` are added to the function's
    // let list for non-i32 pointees (CastAtom cannot wrap a
    // LoadAtom, so the load needs a named scratch to cast from).
    int scratchCounter = 0;

    // Collect snapshot of ptr lets up-front; we'll push_back new
    // `%_pld_*` lets and walking the live vector while mutating it
    // would mis-iterate.
    struct PtrLetInfo {
      std::string name;
      TypePtr pointee;
    };

    std::vector<PtrLetInfo> ptrLets;
    for (const auto &let: entry->lets) {
      if (!isPtrType(let.type))
        continue;
      auto ptee = pointeeType(let.type);
      if (!ptee)
        continue;
      if (!isScalarType(ptee) || isPtrType(ptee))
        continue; // ptr-to-ptr / ptr-to-aggregate: skip
      if (!pointerLoadable(let.name.name))
        continue; // exit-time target unresolved (undef / cross-object): unsafe to load
      ptrLets.push_back({let.name.name, ptee});
    }
    for (const auto &pl: ptrLets) {
      LValue pLV{LocalId{pl.name, {}}, {}, {}};
      // Build the val operand of `crc32_update`.
      Atom valAtom;
      if (isIntType(pl.pointee) && intBitWidth(pl.pointee) == 32) {
        // Direct load — already i32, no scratch needed.
        LoadAtom la;
        la.rval = std::move(pLV);
        valAtom = Atom{std::move(la), {}};
      } else {
        // Allocate a scratch let `%_pld_<n>` of the pointee type,
        // emit `%_pld_<n> = load %p;`, then pass `(%_pld_<n> as i32)`
        // as the val.
        std::string slotName = "%_pld_" + std::to_string(scratchCounter++);
        LetDecl slot;
        slot.isMutable = true;
        slot.name = LocalId{slotName, {}};
        slot.type = pl.pointee;
        slot.init = InitVal{InitVal::Kind::Undef, LocalId{}, {}};
        entry->lets.push_back(std::move(slot));
        // %_pld_n = load %p;
        {
          LoadAtom la;
          la.rval = std::move(pLV);
          AssignInstr load;
          load.lhs = LValue{LocalId{slotName, {}}, {}, {}};
          load.rhs = atomToExpr(Atom{std::move(la), {}});
          exit->instrs.push_back(Instr{std::move(load)});
        }
        // valAtom = (%_pld_n as i32)
        CastAtom cast;
        cast.src = LValue{LocalId{slotName, {}}, {}, {}};
        cast.dstType = makeI32();
        valAtom = Atom{std::move(cast), {}};
      }
      // %_chk = call @crc32_update(%_chk, valAtom);
      CallAtom call;
      call.callee = GlobalId{"@crc32_update", {}};
      call.args.push_back(std::make_shared<Expr>(chkReadExpr()));
      call.args.push_back(std::make_shared<Expr>(atomToExpr(std::move(valAtom))));
      AssignInstr chkUpd;
      chkUpd.lhs = localLV("%_chk");
      chkUpd.rhs = atomToExpr(Atom{std::move(call), {}});
      exit->instrs.push_back(Instr{std::move(chkUpd)});
      ++updates;
    }

    if (updates > 0)
      ensureCrc32UpdateDecl(prog);
    return updates;
  }

  // ---------------------------------------------------------------------------
  // Minimal CRC32 oracle — see include/reify/checksum.hpp for usage
  // ---------------------------------------------------------------------------

  namespace {

    // Convert one solver-extracted LetExitValue into an AST InitVal of
    // the matching shape. `declType` is the let's declared type — used
    // both to pick scalar vs aggregate kind and to walk struct fields
    // in declaration order. `structs` is searched for struct field
    // ordering. Pointer / Undef entries fall back to InitVal::Undef so
    // the addr-replay in the entry block still has somewhere to write.
    InitVal letExitValueToInitVal(
        const refractir::SymbolicExecutor::LetExitValue &lev, const TypePtr &declType,
        const std::vector<StructDecl> &structs
    ) {
      using LEV = refractir::SymbolicExecutor::LetExitValue;
      InitVal iv;
      switch (lev.kind) {
        case LEV::Kind::Int: {
          iv.kind = InitVal::Kind::Int;
          int64_t v = 0;
          if (auto pi = std::get_if<int64_t>(&lev.scalar))
            v = *pi;
          else if (auto pd = std::get_if<double>(&lev.scalar))
            v = static_cast<int64_t>(*pd);
          iv.value = IntLit{v, {}};
          return iv;
        }
        case LEV::Kind::Float: {
          iv.kind = InitVal::Kind::Float;
          double v = 0.0;
          if (auto pd = std::get_if<double>(&lev.scalar))
            v = *pd;
          else if (auto pi = std::get_if<int64_t>(&lev.scalar))
            v = static_cast<double>(*pi);
          iv.value = FloatLit{v, {}};
          return iv;
        }
        case LEV::Kind::Array:
        case LEV::Kind::Vec: {
          TypePtr elemTy;
          if (declType && std::holds_alternative<ArrayType>(declType->v))
            elemTy = std::get<ArrayType>(declType->v).elem;
          else if (declType && std::holds_alternative<VecType>(declType->v))
            elemTy = std::get<VecType>(declType->v).elem;
          std::vector<InitValPtr> children;
          children.reserve(lev.elems.size());
          for (const auto &c: lev.elems)
            children.push_back(
                std::make_shared<InitVal>(letExitValueToInitVal(c, elemTy, structs))
            );
          iv.kind = InitVal::Kind::Aggregate;
          iv.value = std::move(children);
          return iv;
        }
        case LEV::Kind::Struct: {
          const StructDecl *sd = nullptr;
          if (declType && std::holds_alternative<StructType>(declType->v)) {
            const auto &sn = std::get<StructType>(declType->v).name.name;
            for (const auto &s: structs)
              if (s.name.name == sn) {
                sd = &s;
                break;
              }
          }
          std::vector<InitValPtr> children;
          if (sd) {
            children.reserve(sd->fields.size());
            for (const auto &f: sd->fields) {
              auto it = lev.fields.find(f.name);
              if (it != lev.fields.end()) {
                children.push_back(
                    std::make_shared<InitVal>(letExitValueToInitVal(it->second, f.type, structs))
                );
              } else {
                InitVal u;
                u.kind = InitVal::Kind::Undef;
                children.push_back(std::make_shared<InitVal>(std::move(u)));
              }
            }
          }
          iv.kind = InitVal::Kind::Aggregate;
          iv.value = std::move(children);
          return iv;
        }
        case LEV::Kind::Ptr:
        case LEV::Kind::Undef:
        default:
          iv.kind = InitVal::Kind::Undef;
          return iv;
      }
    }

    // Atom-graph deep clone restricted to the variants the checksum
    // chain and the load preamble produce. Block / Atom are not
    // generically copyable (SelectAtom holds unique_ptrs), so we
    // rebuild rather than copy. Throws on any unexpected variant —
    // the caller is the exit-block-only path emitted by
    // buildSumChecksum + rewriter, which never emits SelectAtom /
    // CmpAtom / aggregates inside the accumulator chain or scratch
    // loads.
    Atom cloneChecksumAtom(const Atom &a);

    Expr cloneChecksumExpr(const Expr &e) {
      Expr ne;
      ne.first = cloneChecksumAtom(e.first);
      ne.span = e.span;
      for (const auto &tail: e.rest)
        ne.rest.push_back({tail.op, cloneChecksumAtom(tail.atom), tail.span});
      return ne;
    }

    Atom cloneChecksumAtom(const Atom &a) {
      return std::visit(
          [&a](const auto &v) -> Atom {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, CoefAtom>) {
              return Atom{CoefAtom{v.coef, v.span}, a.span};
            } else if constexpr (std::is_same_v<T, RValueAtom>) {
              return Atom{RValueAtom{v.rval, v.span}, a.span};
            } else if constexpr (std::is_same_v<T, CastAtom>) {
              return Atom{CastAtom{v.src, v.dstType, v.span}, a.span};
            } else if constexpr (std::is_same_v<T, LoadAtom>) {
              return Atom{LoadAtom{v.rval, v.span}, a.span};
            } else if constexpr (std::is_same_v<T, OpAtom>) {
              return Atom{OpAtom{v.op, v.coef, v.rval, v.span}, a.span};
            } else if constexpr (std::is_same_v<T, UnaryAtom>) {
              return Atom{UnaryAtom{v.op, v.rval, v.span}, a.span};
            } else if constexpr (std::is_same_v<T, AddrAtom>) {
              return Atom{AddrAtom{v.lv, v.span}, a.span};
            } else if constexpr (std::is_same_v<T, CallAtom>) {
              // CallAtom arg list holds shared_ptr<Expr>; sharing is
              // safe because consumers (printer, interpreter, C
              // backend) treat the AST as immutable.
              CallAtom ca;
              ca.callee = v.callee;
              ca.args = v.args;
              ca.span = v.span;
              return Atom{std::move(ca), a.span};
            } else {
              throw std::runtime_error(
                  "buildMiniCrc32Prog: unsupported atom variant in checksum chain"
              );
            }
          },
          a.v
      );
    }

  } // namespace

  refractir::Program buildMiniCrc32Prog(
      const refractir::Program &full, const std::string &funcName,
      const std::unordered_map<std::string, refractir::SymbolicExecutor::LetExitValue>
          &letExitValues
  ) {
    Program minimal;
    minimal.structs = full.structs;

    std::string canonical = funcName.empty() || funcName[0] == '@' ? funcName : "@" + funcName;
    const FunDecl *entry = nullptr;
    for (const auto &f: full.funs) {
      if (f.name.name == canonical) {
        entry = &f;
        break;
      }
    }
    if (!entry)
      return minimal;

    const Block *fullExit = nullptr;
    for (const auto &b: entry->blocks) {
      if (std::holds_alternative<RetTerm>(b.term)) {
        fullExit = &b;
        break;
      }
    }
    if (!fullExit)
      return minimal;

    // Carry over only intrinsic decls referenced by the exit block.
    // The minimal program's body uses only @crc32_update; importing
    // everything else (especially randomly-generated intrinsics)
    // would demand argument lists we don't reproduce.
    for (const auto &id: full.intrinsics) {
      if (id.name.name == "@crc32_update")
        minimal.intrinsics.push_back(id);
    }

    FunDecl mini;
    mini.name = GlobalId{"@minimal_" + funcName, {}};
    mini.retType = makeI32();
    mini.params = entry->params;

    // Lets: same declarations, but scalar / aggregate inits rewritten
    // to the solver-known exit-time values. Pointer lets keep their
    // declared `undef` init — the entry block below assigns them
    // explicitly so the exit-time target replay is plain to read.
    // `%_chk` and `%_pld_*` scratch slots are left at their original
    // (already-undef / already-zero) inits — they're overwritten in
    // the exit block before being read.
    for (const auto &letd: entry->lets) {
      LetDecl newLet;
      newLet.isMutable = letd.isMutable;
      newLet.name = letd.name;
      newLet.type = letd.type;
      auto it = letExitValues.find(letd.name.name);
      if (isPtrType(letd.type) || letd.name.name == "%_chk" ||
          letd.name.name.rfind("%_pld_", 0) == 0) {
        newLet.init = letd.init;
      } else if (it != letExitValues.end() &&
                 it->second.kind != SymbolicExecutor::LetExitValue::Kind::Undef &&
                 it->second.kind != SymbolicExecutor::LetExitValue::Kind::Ptr) {
        newLet.init = letExitValueToInitVal(it->second, letd.type, full.structs);
      } else {
        newLet.init = letd.init;
      }
      mini.lets.push_back(std::move(newLet));
    }

    // Walk type `t` for a contiguous in-object offset (packed bytes —
    // the scale the solver reports LetExitValue::targetOffset in) and
    // append the field / index accesses needed to reach the scalar leaf
    // at that offset. Returns true on success and narrows `t` to the
    // leaf type. False means the offset doesn't land on a clean leaf —
    // happens when the target type's leaves disagree with the pointer's
    // pointee width.
    //
    // The size model here MUST be the one the solver measured with, or
    // the walk lands on the wrong leaf and this oracle silently disagrees
    // with the program it is meant to check.
    TypeUtils::StructTable structTable;
    for (const auto &s: full.structs)
      structTable[s.name.name] = &s;

    // Entry block: one `%p = addr <exit-leaf>;` per pointer let,
    // where `<exit-leaf>` is the scalar leaf reached by walking the
    // SOLVER's exit-time (targetLocal, targetOffset) into the
    // target's declared type. This is what makes the oracle agree
    // with the full program when the body retargets pointers via
    // load-through-pp / ptrfield / ptrindex / pointer arithmetic.
    // Falls back to the original entry-block addr-init when the
    // offset doesn't resolve to a clean leaf — better an
    // observable mismatch than silently dropping a real divergence.
    Block entryBlk;
    entryBlk.label = BlockLabel{"^entry", {}};
    auto findLetType = [&](const std::string &nm) -> TypePtr {
      for (const auto &l: entry->lets)
        if (l.name.name == nm)
          return l.type;
      for (const auto &p: entry->params)
        if (p.name.name == nm)
          return p.type;
      return {};
    };
    for (const auto &letd: entry->lets) {
      if (!isPtrType(letd.type))
        continue;
      std::string targetLocal;
      uint64_t targetOffset = 0;
      auto it = letExitValues.find(letd.name.name);
      if (it != letExitValues.end() &&
          it->second.kind == SymbolicExecutor::LetExitValue::Kind::Ptr &&
          !it->second.targetLocal.empty()) {
        targetLocal = it->second.targetLocal;
        targetOffset = it->second.targetOffset;
      }
      LValue addrSrc;
      bool resolved = false;
      if (!targetLocal.empty()) {
        TypePtr targetTy = findLetType(targetLocal);
        TypePtr pointeeTy = pointeeType(letd.type);
        std::vector<Access> acc;
        uint64_t remOff = targetOffset;
        TypePtr cur = targetTy;
        if (targetTy && pointeeTy && resolveOffsetPath(cur, remOff, acc, pointeeTy, structTable)) {
          addrSrc.base = LocalId{targetLocal, {}};
          addrSrc.accesses = std::move(acc);
          resolved = true;
        }
      }
      if (!resolved) {
        // Fallback: replay the original entry-block `%p = addr X;`.
        // Drops cleanly when the body never resolved %p to a single
        // object — the pointer stays at its entry-init target.
        for (const auto &b: entry->blocks) {
          if (b.label.name != "^entry")
            continue;
          for (const auto &instr: b.instrs) {
            auto *ai = std::get_if<AssignInstr>(&instr);
            if (!ai)
              continue;
            if (ai->lhs.base.name != letd.name.name || !ai->lhs.accesses.empty())
              continue;
            if (!ai->rhs.rest.empty())
              continue;
            if (auto *aa = std::get_if<AddrAtom>(&ai->rhs.first.v)) {
              addrSrc = aa->lv;
              resolved = true;
              break;
            }
          }
          break;
        }
      }
      if (!resolved)
        continue; // truly unresolvable — leave %p undef
      AssignInstr ai;
      ai.lhs = LValue{LocalId{letd.name.name, {}}, {}, {}};
      ai.rhs = Expr{Atom{AddrAtom{std::move(addrSrc), {}}, {}}, {}, {}};
      entryBlk.instrs.push_back(Instr{std::move(ai)});
    }
    BrTerm br;
    br.dest = BlockLabel{"^exit", {}};
    br.thenLabel = br.dest;
    br.elseLabel = br.dest;
    br.isConditional = false;
    entryBlk.term = Terminator{std::move(br)};
    mini.blocks.push_back(std::move(entryBlk));

    // Exit block: verbatim clone of the full's exit (the load
    // preamble, the `%_chk = 0;` init, the @crc32_update chain, and
    // `ret %_chk;`). The chain reads exactly the same lets / pointer
    // loads the full program reads, so the only thing that drives
    // its value is the lets' exit-time inits + the pointer targets
    // we just replayed.
    Block exitBlk;
    exitBlk.label = fullExit->label;
    for (const auto &instr: fullExit->instrs) {
      auto *ai = std::get_if<AssignInstr>(&instr);
      if (!ai)
        continue;
      AssignInstr cloned;
      cloned.lhs = ai->lhs;
      cloned.rhs = cloneChecksumExpr(ai->rhs);
      cloned.span = ai->span;
      exitBlk.instrs.push_back(Instr{std::move(cloned)});
    }
    if (auto *rt = std::get_if<RetTerm>(&fullExit->term)) {
      RetTerm nrt;
      if (rt->value)
        nrt.value = cloneChecksumExpr(*rt->value);
      nrt.span = rt->span;
      exitBlk.term = Terminator{std::move(nrt)};
    }
    exitBlk.span = fullExit->span;
    mini.blocks.push_back(std::move(exitBlk));

    minimal.funs.push_back(std::move(mini));
    return minimal;
  }

} // namespace refractir::reify
