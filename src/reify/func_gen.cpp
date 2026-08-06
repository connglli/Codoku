#include "reify/func_gen.hpp"
#include "analysis/type_utils.hpp"
#include "ast/build.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "analysis/intrinsics.hpp"
#include "reify/hyperparameters.hpp"
#include "reify/intrinsic_whitelist.hpp"

namespace refractir::reify {

  // ---------------------------------------------------------------------------
  // let-init value sampling
  // ---------------------------------------------------------------------------

  // Weighted-mixture integer init literal — see hp::kInitInt_* for the
  // three-bucket rationale (small / mid / boundary).
  static int64_t pickInitIntLit(std::mt19937 &rng, const TypePtr &t) {
    auto [lo, hi] = signedIntRange(intBitWidth(t));
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    double u = coin(rng);
    if (u < rysmith::hp::kInitInt_pSmall) {
      int64_t mag = std::min<int64_t>(rysmith::hp::kInitInt_SmallMag, hi);
      std::uniform_int_distribution<int64_t> d(std::max<int64_t>(-mag, lo), mag);
      return d(rng);
    }
    if (u < rysmith::hp::kInitInt_pSmall + rysmith::hp::kInitInt_pMid) {
      std::uniform_int_distribution<int64_t> d(lo, hi);
      return d(rng);
    }
    const int64_t cand[] = {0, -1, 1, lo, hi, lo + 1, hi - 1};
    std::uniform_int_distribution<std::size_t> d(0, sizeof(cand) / sizeof(cand[0]) - 1);
    return cand[d(rng)];
  }

  // Float init literal: a dyadic, bit-exact, f32-safe value from the shared
  // pool (uniform). Same pool the expression generator uses, so the FP
  // serialization / differential-testing invariant is preserved.
  static double pickInitFloatLit(std::mt19937 &rng) {
    std::uniform_int_distribution<std::size_t> d(0, rysmith::hp::kFloatLitPoolSize - 1);
    return rysmith::hp::kFloatLitPool[d(rng)];
  }

  // ---------------------------------------------------------------------------
  // Unified recursive initializer
  // ---------------------------------------------------------------------------

  static InitVal
  makeInitVal(std::mt19937 &rng, const TypePtr &t, const std::vector<StructDecl> &structDecls) {
    if (TypeUtils::isInt(t)) {
      InitVal iv;
      iv.kind = InitVal::Kind::Int;
      iv.value = IntLit{pickInitIntLit(rng, t), {}};
      return iv;
    }
    if (TypeUtils::isFloat(t)) {
      InitVal iv;
      iv.kind = InitVal::Kind::Float;
      iv.value = FloatLit{pickInitFloatLit(rng), {}};
      return iv;
    }
    if (TypeUtils::isPtr(t)) {
      InitVal iv;
      iv.kind = InitVal::Kind::Undef;
      return iv;
    }
    if (std::holds_alternative<ArrayType>(t->v)) {
      const auto &at = std::get<ArrayType>(t->v);
      std::vector<InitValPtr> children;
      for (uint64_t i = 0; i < at.size; i++) {
        children.push_back(std::make_shared<InitVal>(makeInitVal(rng, at.elem, structDecls)));
      }
      InitVal iv;
      iv.kind = InitVal::Kind::Aggregate;
      iv.value = std::move(children);
      return iv;
    }
    if (std::holds_alternative<StructType>(t->v)) {
      const std::string &sname = std::get<StructType>(t->v).name.name;
      const StructDecl *sd = nullptr;
      for (const auto &decl: structDecls)
        if (decl.name.name == sname) {
          sd = &decl;
          break;
        }
      if (sd) {
        std::vector<InitValPtr> children;
        for (const auto &field: sd->fields) {
          children.push_back(std::make_shared<InitVal>(makeInitVal(rng, field.type, structDecls)));
        }
        InitVal iv;
        iv.kind = InitVal::Kind::Aggregate;
        iv.value = std::move(children);
        return iv;
      }
    }
    // VecType: brace-init with N scalar elements.
    if (std::holds_alternative<VecType>(t->v)) {
      const auto &vt = std::get<VecType>(t->v);
      std::vector<InitValPtr> children;
      for (uint64_t i = 0; i < vt.size; i++) {
        children.push_back(std::make_shared<InitVal>(makeInitVal(rng, vt.elem, structDecls)));
      }
      InitVal iv;
      iv.kind = InitVal::Kind::Aggregate;
      iv.value = std::move(children);
      return iv;
    }
    InitVal iv;
    iv.kind = InitVal::Kind::Undef;
    return iv;
  }

  // ---------------------------------------------------------------------------
  // Checksum builder
  // ---------------------------------------------------------------------------

  // Build the *sum-form* checksum the solver sees: `%_chk = 0;` then
  // one `%_chk = %_chk + atom` per scalar leaf of every let-init local
  // and every parameter. Pointer leaves are deliberately NOT
  // dereferenced here — the underlying object is already hashed via
  // direct access (every pointer in the generated program points to an
  // existing local, set up in the entry block by addr-init), so a
  // `load %p` in the sum would only force the solver to encode a
  // load-dispatch chain for no coverage gain. The CRC32 rewriter
  // (`rewriteExitToCrc32Checksum`) re-adds `load %p` calls post-solve,
  // where the load serves as runtime opacity against the C optimizer
  // rather than an SMT obligation.
  static std::vector<Instr> buildSumChecksum(const VarCatalogue &vars) {
    std::vector<Instr> instrs;
    auto i32 = buildI32();

    // %_chk = 0;
    {
      Expr zero = buildExpr(buildCoefAtom(IntLit{0, {}}));
      instrs.push_back(Instr{AssignInstr{buildLValue("%_chk"), std::move(zero), {}}});
    }

    // Emit `%_chk = cast_atom + %_chk` (cast first) or `%_chk = %_chk + rval`.
    // CastAtom must be the FIRST atom — it cannot appear in tail position.
    auto emitChkAccum = [&](Atom valueAtom, bool isCast) {
      Expr rhs;
      if (isCast) {
        rhs.first = std::move(valueAtom);
        rhs.rest.push_back({AddOp::Plus, buildRValAtom(buildLValue("%_chk")), {}});
      } else {
        rhs.first = buildRValAtom(buildLValue("%_chk"));
        rhs.rest.push_back({AddOp::Plus, std::move(valueAtom), {}});
      }
      instrs.push_back(Instr{AssignInstr{buildLValue("%_chk"), std::move(rhs), {}}});
    };

    // Emit a scalar LValue into the checksum, casting to i32 when needed.
    auto emitScalarLV = [&](LValue lv, const TypePtr &t) {
      if (TypeUtils::isInt(t) && intBitWidth(t) == 32) {
        emitChkAccum(buildRValAtom(std::move(lv)), /*isCast=*/false);
      } else if (TypeUtils::isScalar(t)) {
        CastAtom ca;
        ca.src = std::move(lv);
        ca.dstType = i32;
        emitChkAccum(Atom{std::move(ca), {}}, /*isCast=*/true);
      }
    };

    // Look up a struct decl by name; returns nullptr if absent.
    auto findStruct = [&](const std::string &sname) -> const StructDecl * {
      for (const auto &d: vars.structDecls) {
        if (d.name.name == sname)
          return &d;
      }
      return nullptr;
    };

    // Recursively walk every NON-POINTER scalar leaf of `(lv, t)` and
    // fold each into `%_chk`. Aggregate types (Array, Vec, Struct) —
    // and arbitrary nestings thereof — are unrolled at compile time.
    // Pointer leaves are skipped: the rewriter appends `load %p` steps
    // post-solve so the sum stays cheap for the SMT solver while the
    // final CRC32 still folds every byte the pointer chain can reach.
    std::function<void(LValue, const TypePtr &)> emitTypeHash;
    emitTypeHash = [&](LValue lv, const TypePtr &t) {
      if (!t)
        return;
      if (TypeUtils::isPtr(t))
        return; // deferred to rewriteExitToCrc32Checksum
      if (TypeUtils::isScalar(t)) {
        emitScalarLV(std::move(lv), t);
        return;
      }
      if (std::holds_alternative<ArrayType>(t->v)) {
        const auto &at = std::get<ArrayType>(t->v);
        for (uint64_t i = 0; i < at.size; i++) {
          LValue inner = lv;
          inner.accesses.push_back(AccessIndex{Index{IntLit{(int64_t) i, {}}}, {}});
          emitTypeHash(std::move(inner), at.elem);
        }
        return;
      }
      if (std::holds_alternative<VecType>(t->v)) {
        const auto &vt = std::get<VecType>(t->v);
        for (uint64_t i = 0; i < vt.size; i++) {
          LValue inner = lv;
          inner.accesses.push_back(AccessIndex{Index{IntLit{(int64_t) i, {}}}, {}});
          emitTypeHash(std::move(inner), vt.elem);
        }
        return;
      }
      if (std::holds_alternative<StructType>(t->v)) {
        const std::string &sname = std::get<StructType>(t->v).name.name;
        if (const StructDecl *sd = findStruct(sname)) {
          for (const auto &f: sd->fields) {
            LValue inner = lv;
            inner.accesses.push_back(AccessField{f.name, {}});
            emitTypeHash(std::move(inner), f.type);
          }
        }
        return;
      }
    };

    // Every var — let-init locals AND function parameters — feeds the
    // checksum. vars.vars contains both (params have `isParam == true`
    // and the generator places them last but they take the same code
    // path here).
    for (const auto &v: vars.vars) {
      emitTypeHash(buildLValue(v.name), v.type);
    }

    return instrs;
  }

  // ---------------------------------------------------------------------------
  // genFunction
  // ---------------------------------------------------------------------------

  FuncGenResult genFunction(
      const RyCFG &cfg, const std::vector<std::string> &path, const VarCatalogue &vars,
      const FuncGenConfig &fcfg
  ) {
    std::mt19937 rng(fcfg.seed);

    SymCounter sym;
    sym.coefLo = fcfg.coefLo;
    sym.coefHi = fcfg.coefHi;
    sym.valueLo = fcfg.valueLo;
    sym.valueHi = fcfg.valueHi;
    sym.indexLo = fcfg.indexLo;
    sym.indexHi = fcfg.indexHi;

    // Track which (intrinsic, bitwidth) pairs are used during
    // expression generation so we can emit IntrinsicDecl entries into the
    // Program.  Make a mutable copy of fcfg.exprCfg so we can attach the
    // tracking set without mutating the caller's config.
    std::set<IntrinsicUseKey> usedIntrinsics;
    ExprGenConfig exprCfg = fcfg.exprCfg;
    if (fcfg.enableIntrinsics) {
      exprCfg.usedIntrinsics = &usedIntrinsics;
    } else {
      exprCfg.enableIntrinsics = false;
      exprCfg.usedIntrinsics = nullptr;
    }

    // Off-path volume scaling (see FuncGenConfig::offPathMultiplier): the
    // CLI volume knobs describe on-path blocks; off-path blocks — which the
    // solver never visits — get them scaled, widening the compiler-facing
    // surface at zero solver cost.
    auto scaleOff = [&](int v) { return (int) std::lround(v * fcfg.offPathMultiplier); };
    int offNStmts = std::max(0, scaleOff(fcfg.nStmts));
    ExprGenConfig offExprCfg = exprCfg;
    offExprCfg.minAtoms = std::max(1, scaleOff(exprCfg.minAtoms));
    offExprCfg.maxAtoms = std::max(offExprCfg.minAtoms, scaleOff(exprCfg.maxAtoms));

    // Generate a single input sym (i32) used for interest-init requires
    std::string inputSym = sym.nextValue();

    std::unordered_set<std::string> pathSet(path.begin(), path.end());
    auto cfgLabels = cfg.labels();

    std::vector<Block> blocks;
    blocks.reserve(cfgLabels.size());

    for (const auto &blkLabel: cfgLabels) {
      const auto *blk = cfg.get(blkLabel);
      bool onPath = pathSet.count(blkLabel) > 0;
      bool isExit = (blkLabel == cfg.exitLabel);

      Block block;
      block.label = BlockLabel{"^" + blkLabel, {}};

      int coefsBefore = sym.countOfKind(SymKind::Coef);

      if (isExit) {
        // Exit block: compute checksum and return. Pointer loads are
        // deferred to rewriteExitToCrc32Checksum so the SUM the solver
        // sees stays free of `load %p` constraints.
        auto chkInstrs = buildSumChecksum(vars);
        for (auto &ci: chkInstrs)
          block.instrs.push_back(std::move(ci));

        Expr retVal = buildExpr(buildRValAtom(buildLValue("%_chk")));
        block.term = Terminator{RetTerm{std::move(retVal), {}}};

      } else {
        // Non-exit block: generate statements

        // Special entry block setup: assign ptr vars and interest-init require
        if (blkLabel == cfg.entry) {
          // Assign all ptr vars (addr or addr-of-ptr)
          for (const auto &v: vars.vars) {
            if (!TypeUtils::isPtr(v.type))
              continue;
            if (!v.ptrTarget)
              continue;

            // %p = addr %target
            LValue lhs = buildLValue(v.name);
            Expr rhs = buildExpr(buildAddrAtom(buildLValue(*v.ptrTarget)));
            block.instrs.push_back(Instr{AssignInstr{std::move(lhs), std::move(rhs), {}}});
          }

          // Interest-init: require inputSym != 0
          if (fcfg.enableInterestInits) {
            RequireInstr req;
            req.cond.lhs = buildExpr(buildCoefAtom(LocalOrSymId{SymId{inputSym, {}}}));
            req.cond.op = RelOp::NE;
            req.cond.rhs = buildExpr(buildCoefAtom(IntLit{0, {}}));
            req.message = "nonzero input";
            block.instrs.push_back(Instr{std::move(req)});
          }
        }

        // Generate statements for on-path non-exit blocks
        if (onPath) {
          auto stmts = genBlockStmts(rng, &sym, vars, fcfg.nStmts, true, exprCfg);
          for (auto &s: stmts)
            block.instrs.push_back(std::move(s));
        } else {
          // Off-path: concrete-only stmts at the scaled volume
          auto stmts = genBlockStmts(rng, nullptr, vars, offNStmts, false, offExprCfg);
          for (auto &s: stmts)
            block.instrs.push_back(std::move(s));
        }

        // Interest coef requires (on-path only, before terminator)
        if (fcfg.enableInterestCoefs && onPath) {
          auto reqs =
              interestCoefRequires(rng, sym, coefsBefore, fcfg.pLargeCoef, fcfg.largeCoefThreshold);
          for (auto &r: reqs)
            block.instrs.push_back(std::move(r));
        }

        // Terminator
        if (blk->isBranch()) {
          Cond cond =
              genCond(rng, onPath ? &sym : nullptr, vars, onPath, onPath ? exprCfg : offExprCfg);
          BrTerm br;
          br.cond = std::move(cond);
          br.thenLabel = BlockLabel{"^" + blk->succs[0], {}};
          br.elseLabel = BlockLabel{"^" + blk->succs[1], {}};
          br.dest = br.thenLabel;
          br.isConditional = true;
          block.term = Terminator{std::move(br)};
        } else if (blk->isGoto()) {
          BrTerm br;
          br.dest = BlockLabel{"^" + blk->succs[0], {}};
          br.thenLabel = br.dest;
          br.elseLabel = br.dest;
          br.isConditional = false;
          block.term = Terminator{std::move(br)};
        } else {
          // Dead-end non-exit (shouldn't happen in well-formed CFG)
          Expr zero = buildExpr(buildCoefAtom(IntLit{0, {}}));
          block.term = Terminator{RetTerm{std::move(zero), {}}};
        }
      }

      blocks.push_back(std::move(block));
    }

    // ---------------------------------------------------------------------------
    // Assemble Program
    // ---------------------------------------------------------------------------

    Program prog;
    prog.structs = vars.structDecls;

    // Emit IntrinsicDecl entries for each (kind, bitwidth) pair
    // used during expression generation. The semchecker now supports
    // overloaded intrinsics with different signatures, so the same intrinsic
    // may appear at multiple bitwidths (e.g. @popcount i32 / i64).
    if (fcfg.enableIntrinsics)
      appendUsedIntrinsicDecls(usedIntrinsics, prog.intrinsics);

    FunDecl fun;
    fun.name = GlobalId{"@" + fcfg.funcName, {}};
    fun.retType = buildI32();
    fun.syms = sym.makeDecls();

    // Split VarCatalogue entries: isParam → FunDecl.params,
    // others → FunDecl.lets. Parameters carry no initializer (the
    // solver synthesises their values and the SOLVED header records
    // them; symiri replays them as positional CLI args).
    for (const auto &v: vars.vars) {
      if (v.isParam) {
        ParamDecl pd;
        pd.name = LocalId{v.name, {}};
        pd.type = v.type;
        fun.params.push_back(std::move(pd));
        continue;
      }
      LetDecl let;
      let.isMutable = true;
      let.name = LocalId{v.name, {}};
      let.type = v.type;

      if (TypeUtils::isPtr(v.type)) {
        // Ptr vars assigned in entry block via addr; declare as undef.
        InitVal iv;
        iv.kind = InitVal::Kind::Undef;
        let.init = std::move(iv);
      } else {
        let.init = makeInitVal(rng, v.type, vars.structDecls);
      }

      fun.lets.push_back(std::move(let));
    }

    // %_chk: let mut %_chk: i32 = 0;
    {
      LetDecl let;
      let.isMutable = true;
      let.name = LocalId{"%_chk", {}};
      let.type = buildI32();
      InitVal iv;
      iv.kind = InitVal::Kind::Int;
      iv.value = IntLit{0, {}};
      let.init = std::move(iv);
      fun.lets.push_back(std::move(let));
    }

    fun.blocks = std::move(blocks);
    prog.funs.push_back(std::move(fun));

    // Build path labels with ^ prefix
    std::vector<std::string> pathLabels;
    pathLabels.reserve(path.size());
    for (const auto &lbl: path)
      pathLabels.push_back("^" + lbl);

    return FuncGenResult{std::move(prog), std::move(pathLabels)};
  }

  // ---------------------------------------------------------------------------
  // spliceNontermCorrections
  // ---------------------------------------------------------------------------

  // Locals spliced in to drive a period-k orbit (see the counter below).
  // Named like the checksum accumulator `%_chk`: generator scaffolding, not
  // part of the random variable catalogue.
  static constexpr const char *kNontermPeriodVar = "%_ntper";
  static constexpr const char *kNontermModulusVar = "%_ntmod";

  void spliceNontermCorrections(
      Program &prog, const std::string &funcName, const std::vector<std::string> &cycleLabels,
      const std::string &latchLabel, int period
  ) {
    FunDecl *fun = nullptr;
    for (auto &f: prog.funs)
      if (f.name.name == "@" + funcName) {
        fun = &f;
        break;
      }
    if (!fun)
      return;

    // Declared type of each mutable let (the checksum accumulator is
    // generation scaffolding, never real state — exclude it).
    std::unordered_map<std::string, TypePtr> letTypes;
    for (const auto &l: fun->lets)
      if (l.name.name != "%_chk")
        letTypes[l.name.name] = l.type;

    std::unordered_set<std::string> cycleSet(cycleLabels.begin(), cycleLabels.end());

    // Mutable lets the cycle *touches* (assigns via any access path), in
    // first-touch order. Every scalar-int leaf of a touched let is corrected
    // below, so a dynamic-index write — whose target leaf is not statically
    // known — is still closed (all leaves of that let get a correction). A let
    // the cycle never writes is invariant across the lap and needs none.
    std::vector<std::string> touched;
    std::unordered_set<std::string> seen;
    for (auto &blk: fun->blocks) {
      if (!cycleSet.count(blk.label.name))
        continue;
      for (auto &ins: blk.instrs)
        if (auto *a = std::get_if<AssignInstr>(&ins)) {
          const std::string &nm = a->lhs.base.name;
          if (letTypes.count(nm) && seen.insert(nm).second)
            touched.push_back(nm);
        }
    }
    if (touched.empty())
      return;

    Block *latch = nullptr;
    for (auto &blk: fun->blocks)
      if (blk.label.name == latchLabel) {
        latch = &blk;
        break;
      }
    if (!latch)
      return;

    auto findStruct = [&](const std::string &sname) -> const StructDecl * {
      for (const auto &d: prog.structs)
        if (d.name.name == sname)
          return &d;
      return nullptr;
    };

    // Append `<leaf> = <leaf> + %?ntK;` to the latch for every *scalar* leaf of
    // a touched let, recursing through array / vector / struct nesting. The
    // fresh correction symbol gives the solver the freedom to restore that leaf
    // to its header-entry value, so the fixed point is always solvable.
    //
    // `+` is the one closer for every scalar domain, because in each it is the
    // group operation the solver can invert:
    //   iN   — wrapping add is a bijection, so some %?nt always closes the gap;
    //   fN   — add reaches the header value whenever the difference is
    //          representable, and the asserted finite-result guard keeps the
    //          lap UB-free (an unreachable target just yields UNSAT, and
    //          rysmith resamples);
    //   ptr  — pointer arithmetic shifts the offset within the pointee's
    //          object, so a pointer that merely *moves* over the lap is
    //          closeable; one that changes object is not (cross-object
    //          arithmetic is itself UB), which again surfaces as UNSAT.
    // The first int-or-float leaf is remembered so an @observe beacon can be
    // planted on it below; a pointer cannot carry the beacon (@observe's
    // domain is int-or-fp).
    int k = 0;
    bool haveObsLeaf = false;
    LValue obsLv;
    TypePtr obsTy;
    std::function<void(LValue, const TypePtr &)> correctLeaves;
    correctLeaves = [&](LValue lv, const TypePtr &t) {
      if (!t)
        return;
      const bool ptrLeaf = TypeUtils::isPtr(t);
      if (TypeUtils::isInt(t) || TypeUtils::isFloat(t) || ptrLeaf) {
        if (!haveObsLeaf && !ptrLeaf) {
          haveObsLeaf = true;
          obsLv = lv;
          obsTy = t;
        }
        // A pointer's correction shifts its offset, so the symbol is the
        // integer element-distance type — there is no `sym` of pointer type.
        TypePtr symTy =
            ptrLeaf ? std::make_shared<Type>(Type{IntType{IntType::Kind::I64, {}, {}}, {}}) : t;
        std::string symName = "%?nt" + std::to_string(k++);
        SymDecl d;
        d.name = SymId{symName, {}};
        d.kind = SymKind::Value;
        d.type = std::move(symTy);
        fun->syms.push_back(std::move(d));

        Expr rhs;
        rhs.first = buildRValAtom(lv); // read the leaf
        rhs.rest.push_back({AddOp::Plus, buildCoefAtom(LocalOrSymId{SymId{symName, {}}}), {}});
        latch->instrs.push_back(Instr{AssignInstr{std::move(lv), std::move(rhs), {}}});
        return;
      }
      if (std::holds_alternative<ArrayType>(t->v)) {
        const auto &at = std::get<ArrayType>(t->v);
        for (uint64_t i = 0; i < at.size; i++) {
          LValue inner = lv;
          inner.accesses.push_back(AccessIndex{Index{IntLit{(int64_t) i, {}}}, {}});
          correctLeaves(std::move(inner), at.elem);
        }
        return;
      }
      if (std::holds_alternative<VecType>(t->v)) {
        const auto &vt = std::get<VecType>(t->v);
        for (uint64_t i = 0; i < vt.size; i++) {
          LValue inner = lv;
          inner.accesses.push_back(AccessIndex{Index{IntLit{(int64_t) i, {}}}, {}});
          correctLeaves(std::move(inner), vt.elem);
        }
        return;
      }
      if (std::holds_alternative<StructType>(t->v)) {
        const std::string &sname = std::get<StructType>(t->v).name.name;
        if (const StructDecl *sd = findStruct(sname))
          for (const auto &f: sd->fields) {
            LValue inner = lv;
            inner.accesses.push_back(AccessField{f.name, {}});
            correctLeaves(std::move(inner), f.type);
          }
        return;
      }
    };

    for (const auto &nm: touched)
      correctLeaves(buildLValue(nm), letTypes[nm]);

    // For a period-k orbit (k > 1), plant a modular counter. The additive
    // corrections above can only ever close a period-1 orbit: closing after k
    // laps with a single correction value would need the leaf to wrap around
    // (k·c ≡ 0 with no intermediate hitting 0), and RefractIR's `+` traps on
    // signed overflow, so the solver is forced to c = 0 — which recurs after
    // one lap, not k. A counter supplies the k-cycle directly instead:
    //
    //   %__ntper = %__ntper + 1;        // 0,1,…,k-1,k  — never overflows
    //   %__ntper = %__ntper % %__ntmod; // k wraps to 0
    //
    // so %__ntper is 0 only at every k-th arrival. That is all the period
    // constraint needs: the intermediate-arrival requirement is that the
    // *state* differs, i.e. at least one leaf differs, so every other leaf is
    // free to stay put (correction 0) and still satisfy it. The modulus lives
    // in its own never-written local because an atom is `coef OP rval`, which
    // cannot spell `%v % 3` with the literal on the right; being invariant, it
    // recurs trivially.
    if (period > 1) {
      auto i32Ty = std::make_shared<Type>(Type{IntType{IntType::Kind::I32, {}, {}}, {}});
      auto addLet = [&](const std::string &name, std::int64_t initVal) {
        LetDecl d;
        d.isMutable = true;
        d.name = LocalId{name, {}};
        d.type = i32Ty;
        InitVal iv;
        iv.kind = InitVal::Kind::Int;
        iv.value = IntLit{initVal, {}};
        d.init = std::move(iv);
        fun->lets.push_back(std::move(d));
      };
      addLet(kNontermPeriodVar, 0);
      addLet(kNontermModulusVar, period);

      Expr inc;
      inc.first = buildRValAtom(buildLValue(kNontermPeriodVar));
      inc.rest.push_back({AddOp::Plus, buildCoefAtom(IntLit{1, {}}), {}});
      latch->instrs.push_back(
          Instr{AssignInstr{buildLValue(kNontermPeriodVar), std::move(inc), {}}}
      );

      OpAtom mod;
      mod.op = AtomOpKind::Mod;
      mod.coef = LocalOrSymId{LocalId{kNontermPeriodVar, {}}};
      mod.rval = RValue{buildLValue(kNontermModulusVar)};
      latch->instrs.push_back(
          Instr{
              AssignInstr{buildLValue(kNontermPeriodVar), buildExpr(Atom{std::move(mod), {}}), {}}
          }
      );
    }

    // Plant one observability beacon in the cycle: `%leaf = call @observe(%leaf);`.
    // @observe is the identity, so the header fixed point is unaffected, but its
    // C lowering performs an observable volatile write — a side effect that
    // stops the optimizer from treating the infinite loop as side-effect-free
    // and deleting it under the C11 §6.8.5p6 / mustprogress termination
    // assumption (which fires non-deterministically at -O2 otherwise).
    if (haveObsLeaf) {
      bool declared = false;
      for (const auto &id: prog.intrinsics)
        if (id.name.name == "@observe") {
          declared = true;
          break;
        }
      if (!declared) {
        IntrinsicDecl id;
        id.name = GlobalId{"@observe", {}};
        id.retType = obsTy;
        ParamDecl p;
        p.name = LocalId{"%x", {}};
        p.type = obsTy;
        id.params.push_back(std::move(p));
        prog.intrinsics.push_back(std::move(id));
      }
      CallAtom ca;
      ca.callee = GlobalId{"@observe", {}};
      ca.args.push_back(std::make_shared<Expr>(buildExpr(buildRValAtom(obsLv))));
      Expr obsRhs = buildExpr(Atom{std::move(ca), {}});
      latch->instrs.push_back(Instr{AssignInstr{obsLv, std::move(obsRhs), {}}});
    }
  }

} // namespace refractir::reify
