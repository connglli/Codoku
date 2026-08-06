#include "reify/expr_gen.hpp"
#include "analysis/type_utils.hpp"
#include "reify/ast_builder.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <optional>
#include "reify/hyperparameters.hpp"
#include "reify/intrinsic_whitelist.hpp"

namespace refractir::reify {

  // ---------------------------------------------------------------------------
  // Internal helpers — AST factories
  // ---------------------------------------------------------------------------

  static LValue arrayLV(const std::string &name, int64_t idx) {
    LValue lv;
    lv.base = LocalId{name, {}};
    lv.accesses.push_back(AccessIndex{Index{IntLit{idx, {}}}, {}});
    return lv;
  }

  static LValue structLV(const std::string &name, const std::string &field) {
    LValue lv;
    lv.base = LocalId{name, {}};
    lv.accesses.push_back(AccessField{field, {}});
    return lv;
  }

  static Coef symCoef(const std::string &sname) { return LocalOrSymId{SymId{sname, {}}}; }

  static Coef intCoef(int64_t v) { return IntLit{v, {}}; }

  static Coef floatCoef(double v) { return FloatLit{v, {}}; }

  static Atom opAtom(AtomOpKind op, Coef c, RValue rv) {
    return Atom{OpAtom{op, std::move(c), std::move(rv), {}}, {}};
  }

  static Atom unaryAtom(RValue rv) {
    return Atom{UnaryAtom{UnaryOpKind::Not, std::move(rv), {}}, {}};
  }

  // Pick a random element from a non-empty vector
  template<typename T>
  static const T &pickOne(std::mt19937 &rng, const std::vector<T> &v) {
    assert(!v.empty());
    std::uniform_int_distribution<int> d(0, static_cast<int>(v.size()) - 1);
    return v[d(rng)];
  }

  // Filter a vector of var pointers to exclude any entry whose name matches
  // `excludeName`. When `excludeName` is unset the input is returned
  // unchanged. Applied at every variable-pool pick in the body of an RHS
  // expression to make `%x = %x;`, `%x = -4 * %x;`, etc. impossible.
  //
  // Takes its pool by value: every caller hands over a freshly built vector
  // from the catalogue, so the argument moves in, is filtered in place, and
  // moves out without allocating.
  static std::vector<const VarEntry *>
  excluding(std::vector<const VarEntry *> vec, const std::optional<std::string> &excludeName) {
    if (excludeName)
      std::erase_if(vec, [&](const VarEntry *v) { return v->name == *excludeName; });
    return vec;
  }

  // A Coef carries no runtime dependency unless it names a LocalId (a
  // SymId resolves to a solver-chosen literal post-solve, a bare literal
  // is trivially constant).
  static bool isTriviallyConstantCoef(const Coef &c) {
    if (auto *lsi = std::get_if<LocalOrSymId>(&c))
      return std::holds_alternative<SymId>(*lsi); // LocalId reads the var → not trivial
    return true;                                  // IntLit, FloatLit, NullLit
  }

  static bool isTriviallyConstantAtom(const Atom &a); // mutually recursive with Expr

  // An Expr is trivially constant when every atom in it is.
  static bool isTriviallyConstantExpr(const Expr &e) {
    if (!isTriviallyConstantAtom(e.first))
      return false;
    for (const auto &t: e.rest)
      if (!isTriviallyConstantAtom(t.atom))
        return false;
    return true;
  }

  // A select arm (RValue | Coef): an RValue reads a local; a Coef follows
  // the Coef rule above.
  static bool isTriviallyConstantSelectVal(const SelectVal &sv) {
    if (std::holds_alternative<RValue>(sv))
      return false; // RValue == LValue → reads a local
    return isTriviallyConstantCoef(std::get<Coef>(sv));
  }

  // An atom is "trivially constant" when it carries no runtime LValue
  // dependency: a bare literal / SymId CoefAtom (the solver picks a literal
  // post-solve, indistinguishable from a hardcoded literal), a CastAtom
  // whose source is not an LValue, or a SelectAtom whose condition/mask and
  // both arms are themselves trivially constant (e.g. `select 0 == 0, 1, 2`
  // — which the compiler folds flat). Every other atom kind (RValueAtom,
  // OpAtom, UnaryAtom, CastAtom-from-LValue, AddrAtom, LoadAtom, PtrIndex,
  // PtrField, CallAtom, or a select reading any local) contributes a real
  // data dependency. Used by the post-check that rejects all-trivial RHSs.
  static bool isTriviallyConstantAtom(const Atom &a) {
    if (auto *ca = std::get_if<CoefAtom>(&a.v))
      return isTriviallyConstantCoef(ca->coef);
    if (auto *cast = std::get_if<CastAtom>(&a.v))
      return !std::holds_alternative<LValue>(cast->src);
    if (auto *sa = std::get_if<SelectAtom>(&a.v)) {
      if (!isTriviallyConstantSelectVal(sa->vtrue) || !isTriviallyConstantSelectVal(sa->vfalse))
        return false;
      if (sa->cond)
        return isTriviallyConstantExpr(sa->cond->lhs) && isTriviallyConstantExpr(sa->cond->rhs);
      if (sa->maskExpr)
        return isTriviallyConstantExpr(*sa->maskExpr);
      return true;
    }
    return false;
  }

  // ---------------------------------------------------------------------------
  // SymCounter implementation
  // ---------------------------------------------------------------------------

  std::string SymCounter::next(SymKind kind, const TypePtr &type) {
    auto name = "%?s" + std::to_string(n++);
    int64_t lo, hi;
    if (kind == SymKind::Coef) {
      lo = coefLo;
      hi = coefHi;
    } else if (kind == SymKind::Value) {
      lo = valueLo;
      hi = valueHi;
    } else {
      lo = indexLo;
      hi = indexHi;
    }
    entries.push_back({name, kind, type, lo, hi});
    return name;
  }

  std::string SymCounter::nextCoef(const TypePtr &type) { return next(SymKind::Coef, type); }

  std::string SymCounter::nextValue() { return next(SymKind::Value, makeI32()); }

  std::string SymCounter::nextIndex() { return next(SymKind::Index, makeI32()); }

  int SymCounter::countOfKind(SymKind kind) const {
    int c = 0;
    for (const auto &e: entries)
      if (e.kind == kind)
        c++;
    return c;
  }

  std::vector<std::string> SymCounter::namesOfKindSince(SymKind kind, int since) const {
    std::vector<std::string> result;
    int c = 0;
    for (const auto &e: entries) {
      if (e.kind == kind) {
        if (c >= since)
          result.push_back(e.name);
        c++;
      }
    }
    return result;
  }

  std::vector<SymDecl> SymCounter::makeDecls() const {
    std::vector<SymDecl> decls;
    for (const auto &e: entries) {
      SymDecl d;
      d.name = SymId{e.name, {}};
      d.kind = e.kind;
      d.type = e.type;
      d.domain = Domain{DomainInterval{e.lo, e.hi, {}}};
      decls.push_back(std::move(d));
    }
    return decls;
  }

  // ---------------------------------------------------------------------------
  // Interest coef requires
  // ---------------------------------------------------------------------------

  std::vector<Instr> interestCoefRequires(
      std::mt19937 &rng, const SymCounter &sym, int coefCountBefore, double pLargeCoef,
      int64_t largeCoefThreshold
  ) {
    std::vector<Instr> instrs;
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::uniform_int_distribution<int> side(0, 1);
    // Walk new coef entries (post-coefCountBefore) directly so each
    // require's threshold can be sized to that coef's actual bitwidth —
    // a require like `c > 2^20` is malformed for an i8/i16 coef because
    // the literal doesn't fit the typechecker's per-type range.
    int idx = 0;
    for (const auto &e: sym.entries) {
      if (e.kind != SymKind::Coef)
        continue;
      const int thisIdx = idx++;
      if (thisIdx < coefCountBefore)
        continue;
      if (coin(rng) >= pLargeCoef)
        continue;
      // [bugfix] Vec lane operations create coef syms whose type is the
      // vec element type (expr_gen.cpp `sym->nextCoef(vt.elem)`), which
      // can be f32 / f64. intBitWidth asserts int-ness and would
      // crash. The "large coef" notion only applies to integer
      // bit-vectors, so skip non-int coefs entirely.
      if (!TypeUtils::isInt(e.type))
        continue;
      // Effective feasible range for this coef = its declared --coef-domain
      // [e.lo, e.hi] intersected with the type's representable range. The
      // solver applies the same clamp when emitting the domain constraint
      // (see solver.cpp), so the require literal must land inside this range
      // — otherwise it is either rejected by the typechecker (literal too
      // wide for the coef type) or unsatisfiable against the domain.
      uint32_t bits = intBitWidth(e.type);
      if (bits == 0) // unknown — skip
        continue;
      int64_t typeLo, typeHi;
      if (bits >= 64) {
        typeLo = INT64_MIN;
        typeHi = INT64_MAX;
      } else {
        typeHi = ((int64_t) 1 << (bits - 1)) - 1;
        typeLo = -typeHi - 1;
      }
      int64_t dlo = std::max(e.lo, typeLo);
      int64_t dhi = std::min(e.hi, typeHi);
      if (dlo > dhi)
        continue; // empty domain — nothing to constrain

      // Clamp the requested magnitude threshold into the feasible range,
      // independently per side: `c > posT` needs a value in (posT, dhi];
      // `c < negT` needs one in [dlo, negT). When the domain is roomier than
      // the threshold the require keeps its requested magnitude; when it is
      // tighter the require degrades to the largest in-domain magnitude
      // (forcing c toward dhi / dlo) instead of going UNSAT.
      const int64_t T = largeCoefThreshold;
      int64_t posT = std::min(T, dhi - 1);  // c in (posT, dhi]
      int64_t negT = std::max(-T, dlo + 1); // c in [dlo, negT)
      bool posViable = dhi >= 1 && posT >= 0 && posT < dhi;
      bool negViable = dlo <= -1 && negT <= 0 && negT > dlo;
      if (!posViable && !negViable)
        continue; // domain too tight to force a nonzero magnitude

      // Encode |c| > T as a single relop by picking a viable sign.
      // RequireInstr.cond is one Cond, so we can't OR two predicates in a
      // single require; the disjunction `c > posT ∨ c < negT` is encoded
      // per-coef as one of the two sides. When both are viable the side is
      // random so the magnitude distribution covers both signs.
      bool positive = (posViable && negViable) ? (side(rng) == 0) : posViable;
      RequireInstr req;
      req.cond.lhs = simpleExpr(coefAtom(symCoef(e.name)));
      req.cond.op = positive ? RelOp::GT : RelOp::LT;
      req.cond.rhs = simpleExpr(coefAtom(intCoef(positive ? posT : negT)));
      req.message = positive ? "coef large positive" : "coef large negative";
      instrs.push_back(Instr{std::move(req)});
    }
    return instrs;
  }

  // ---------------------------------------------------------------------------
  // Core expression builders
  // ---------------------------------------------------------------------------

  // Return the [lo, hi] inclusive range used for concrete integer literals
  // of the given target type. Centralised so off-path coef draws and bare
  // literal atoms share a single source of truth.
  static std::pair<int64_t, int64_t> concreteIntRange(const TypePtr &targetType) {
    uint32_t bits = intBitWidth(targetType);
    int64_t lo = rysmith::hp::kConcreteInt_Default_Lo, hi = rysmith::hp::kConcreteInt_Default_Hi;
    if (bits == 8) {
      lo = rysmith::hp::kConcreteInt_I8_Lo;
      hi = rysmith::hp::kConcreteInt_I8_Hi;
    } else if (bits == 16) {
      lo = rysmith::hp::kConcreteInt_I16_Lo;
      hi = rysmith::hp::kConcreteInt_I16_Hi;
    } else if (bits == 32) {
      lo = rysmith::hp::kConcreteInt_I32_Lo;
      hi = rysmith::hp::kConcreteInt_I32_Hi;
    } else if (bits == 64) {
      lo = rysmith::hp::kConcreteInt_I64_Lo;
      hi = rysmith::hp::kConcreteInt_I64_Hi;
    } else if (bits == 1) {
      // i1 holds exactly {0, -1}: true is all-ones (spec §6.4), and any other
      // literal is a hard type error rather than a value that wraps.
      lo = -1;
      hi = 0;
    } else if (bits >= 2 && bits < 64) {
      // Custom iN widths span the full signed range, mirroring the
      // standard widths above. The typechecker's strict literal range
      // check makes anything wider a hard error.
      hi = (int64_t(1) << (bits - 1)) - 1;
      lo = -hi - 1;
    }
    return {lo, hi};
  }

  static int64_t pickConcreteIntLit(std::mt19937 &rng, const TypePtr &targetType) {
    auto [lo, hi] = concreteIntRange(targetType);
    std::uniform_int_distribution<int64_t> d(lo, hi);
    return d(rng);
  }

  // Pick a nonzero concrete literal from the per-width pool. Used as
  // the dividend in off-path `lit / %v` and `lit % %v` atoms — div-by-zero
  // on the runtime variable %v is a separate, pre-existing concern (off-
  // path code is sampled around the solver-chosen execution path).
  static int64_t pickConcreteIntLitNonzero(std::mt19937 &rng, const TypePtr &targetType) {
    int64_t v = pickConcreteIntLit(rng, targetType);
    if (v == 0)
      v = 1;
    return v;
  }

  // Generate a random concrete integer literal in [lo, hi] of the given bitwidth.
  static Atom genConcreteIntAtom(std::mt19937 &rng, const TypePtr &targetType) {
    return coefAtom(intCoef(pickConcreteIntLit(rng, targetType)));
  }

  // Generate a concrete float atom
  static Atom genConcreteFloatAtom(std::mt19937 &rng) {
    std::uniform_int_distribution<std::size_t> d(0, rysmith::hp::kFloatLitPoolSize - 1);
    return coefAtom(floatCoef(rysmith::hp::kFloatLitPool[d(rng)]));
  }

  // Build a SelectVal of targetType: either a same-typed scalar local or a
  // literal/sym Coef. For off-path (sym == nullptr), only locals and literals.
  static SelectVal pickSelectVal(
      std::mt19937 &rng, SymCounter *sym, const VarCatalogue &vars, const TypePtr &targetType,
      const std::optional<std::string> &excludeName = std::nullopt
  ) {
    auto scalars = excluding(vars.scalarsOf(targetType), excludeName);
    // `s in [0, 99]` walks the same percent-slot pattern as the atom
    // dispatch tables: [0, kSelectArm_LocalEnd) → local, then sym,
    // then literal. Default split is uniform thirds.
    std::uniform_int_distribution<int> slot(0, 99);
    int s = slot(rng);
    if (s < rysmith::hp::kSelectArm_LocalEnd && !scalars.empty()) {
      auto *v = pickOne(rng, scalars);
      return SelectVal{LValue{LocalId{v->name, {}}, {}, {}}};
    }
    if (s < rysmith::hp::kSelectArm_SymEnd && sym != nullptr && TypeUtils::isInt(targetType)) {
      // [bugfix] Sym slot must produce a sym whose declared type matches
      // `targetType`. SymCounter::nextValue() hardcoded i32, so a select
      // returning i64 ended up with arms of mismatched widths (i32 sym
      // vs i64 literal/local) and the typechecker rejected the program.
      // Skip the sym slot for FP target types (rysmith does not currently
      // mint FP value-syms in this code path); fall through to the
      // float-literal pool below.
      return SelectVal{symCoef(sym->next(SymKind::Value, targetType))};
    }
    // Literal slot draws from the same per-width / FP pool as bare
    // concrete atoms instead of the hardcoded `1` / `1.0`. Every
    // select fallback arm was exactly `1`, which collapses any
    // `select cond, 1, 1` (both literal arms) to a constant — the
    // compiler removes the select entirely and the cond evaluation with
    // it.
    if (TypeUtils::isFloat(targetType)) {
      std::uniform_int_distribution<std::size_t> d(0, rysmith::hp::kFloatLitPoolSize - 1);
      return SelectVal{floatCoef(rysmith::hp::kFloatLitPool[d(rng)])};
    }
    return SelectVal{intCoef(pickConcreteIntLit(rng, targetType))};
  }

  // Generate a SelectAtom of the given target type. Cond compares a scalar
  // local to a literal (defensive: same-type) so the typechecker is happy.
  static Atom genSelectAtom(
      std::mt19937 &rng, SymCounter *sym, const VarCatalogue &vars, const TypePtr &targetType,
      const std::optional<std::string> &excludeName = std::nullopt
  ) {
    Cond cond;
    auto allScalars = excluding(vars.allScalars(), excludeName);
    if (!allScalars.empty()) {
      auto *vL = pickOne(rng, allScalars);
      cond.lhs = simpleExpr(rvalAtom(localLV(vL->name)));
      // RHS literal of the lhs var's type. On-path (sym != null) it stays 0
      // so the path constraint is a simple sign test; off-path it is
      // drawn from the per-width pool — the threshold value is opaque to
      // the compiler either way, but a fixed 0 needlessly homogenises the
      // never-executed arms.
      if (TypeUtils::isFloat(vL->type)) {
        double f = 0.0;
        if (!sym) {
          std::uniform_int_distribution<std::size_t> d(0, rysmith::hp::kFloatLitPoolSize - 1);
          f = rysmith::hp::kFloatLitPool[d(rng)];
        }
        cond.rhs = simpleExpr(coefAtom(floatCoef(f)));
      } else {
        int64_t i = sym ? 0 : pickConcreteIntLit(rng, vL->type);
        cond.rhs = simpleExpr(coefAtom(intCoef(i)));
      }
    } else {
      cond.lhs = simpleExpr(coefAtom(intCoef(0)));
      cond.rhs = simpleExpr(coefAtom(intCoef(0)));
    }
    static const RelOp relops[] = {RelOp::EQ, RelOp::NE, RelOp::LT,
                                   RelOp::LE, RelOp::GT, RelOp::GE};
    std::uniform_int_distribution<int> ro(0, 5);
    cond.op = relops[ro(rng)];

    SelectAtom sa;
    sa.cond = std::make_unique<Cond>(std::move(cond));
    sa.vtrue = pickSelectVal(rng, sym, vars, targetType, excludeName);
    sa.vfalse = pickSelectVal(rng, sym, vars, targetType, excludeName);
    return Atom{std::move(sa), {}};
  }

  // Forward declaration — defined later in this file.
  static std::pair<Expr, std::vector<Instr>> genExprWithRequires(
      std::mt19937 &rng, SymCounter *sym, const VarCatalogue &vars, const TypePtr &targetType,
      bool onPath, const ExprGenConfig &cfg,
      const std::optional<std::string> &excludeName = std::nullopt
  );

  // Generate an intrinsic-call atom producing `targetType`. Every intrinsic
  // callable for this target competes in one uniform pool: the scalar
  // whitelist (args are scalar `targetType`) and the horizontal reductions
  // (arg is a `<N> targetType` vector). A reduction is callable exactly when
  // a matching vector operand is in scope — that availability is its only
  // gate. The two shapes differ solely in how the argument list is built;
  struct ExprGenContext {
    std::mt19937 &rng;
    SymCounter *sym = nullptr;
    const VarCatalogue &vars;
    const ExprGenConfig &cfg;
    bool onPath = true;
    std::vector<Instr> &extraRequires;
    const std::optional<std::string> &excludeName;
  };

  // selection, use-recording and declaration are identical. Scalar-argument
  // generation may add div-by-zero guards to extraRequires; the reduction's
  // vector operand needs none (the solver prunes partial-sum overflow and
  // non-finite intermediates like any other intrinsic UB). Records the used
  // (kind, element type, lanes) in cfg.usedIntrinsics if non-null.
  static Atom genIntrinsicCallAtom(const ExprGenContext &ctx, const TypePtr &targetType) {
    const bool isFloat = TypeUtils::isFloat(targetType);
    const uint32_t elemBits =
        isFloat ? (std::get<FloatType>(targetType->v).kind == FloatType::Kind::F32 ? 32u : 64u)
                : intBitWidth(targetType);

    // One candidate = one callable intrinsic. `reduceVec` is null for a
    // scalar intrinsic and the `<N> T` operand var for a reduction. Every
    // property comes from the canonical signature table.
    struct Candidate {
      IntrinsicKind kind;
      int paramCount;            // scalar arg count (unused for reductions)
      const VarEntry *reduceVec; // non-null ⇒ reduction over this vector var
    };

    std::vector<Candidate> cands;

    // One pass over the generatable set. An intrinsic is a candidate when
    // its element domain fits the target (FP targets need a float-admitting
    // intrinsic) and, for reductions, when a `<N> targetType` vector operand
    // is in scope — one candidate per available operand.
    auto vecs = excluding(ctx.vars.vecsWithElem(targetType), ctx.excludeName);
    for (IntrinsicKind kind: getGeneratableIntrinsics()) {
      // Skip i1-returning intrinsics — i1 is not a common target type.
      if (intrinsicReturnsI1(kind))
        continue;
      if (isFloat && !intrinsicAllowsFloat(kind))
        continue;
      int arity = static_cast<int>(intrinsicArity(kind));
      if (isReductionIntrinsic(kind)) {
        for (auto *vv: vecs)
          cands.push_back({kind, arity, vv});
      } else {
        // @bswap is width-restricted: the semchecker rejects widths
        // that are not a multiple of 8.
        if (intrinsicInfo(kind).widthMultipleOf8 && elemBits % 8 != 0)
          continue;
        cands.push_back({kind, arity, nullptr});
      }
    }

    if (cands.empty())
      return isFloat ? genConcreteFloatAtom(ctx.rng) : coefAtom(intCoef(0));

    const auto &c =
        cands[std::uniform_int_distribution<int>(0, static_cast<int>(cands.size()) - 1)(ctx.rng)];

    CallAtom ca;
    ca.callee = GlobalId{std::string(intrinsicName(c.kind)), {}};
    if (c.reduceVec) {
      const TypePtr vecTy = c.reduceVec->type;
      ca.args.push_back(
          std::make_shared<Expr>(
              genExpr(ctx.rng, ctx.sym, ctx.vars, vecTy, ctx.onPath, ctx.cfg, ctx.excludeName)
          )
      );
      if (ctx.cfg.usedIntrinsics)
        ctx.cfg.usedIntrinsics->insert(
            {c.kind, elemBits, isFloat,
             static_cast<std::uint32_t>(std::get<VecType>(vecTy->v).size)}
        );
    } else {
      for (int pi = 0; pi < c.paramCount; pi++) {
        if (ctx.onPath && ctx.sym) {
          auto [ae, areqs] = genExprWithRequires(
              ctx.rng, ctx.sym, ctx.vars, targetType, ctx.onPath, ctx.cfg, ctx.excludeName
          );
          for (auto &r: areqs)
            ctx.extraRequires.push_back(std::move(r));
          ca.args.push_back(std::make_shared<Expr>(std::move(ae)));
        } else {
          ca.args.push_back(
              std::make_shared<Expr>(genExpr(
                  ctx.rng, ctx.sym, ctx.vars, targetType, ctx.onPath, ctx.cfg, ctx.excludeName
              ))
          );
        }
      }
      if (ctx.cfg.usedIntrinsics)
        ctx.cfg.usedIntrinsics->insert({c.kind, elemBits, false, 0});
    }
    return Atom{std::move(ca), {}};
  }

  // Generate a single Atom of the given integer type (on-path, uses sym)
  static Atom genIntAtomOnPath(const ExprGenContext &ctx, const TypePtr &targetType) {
    assert(ctx.sym != nullptr);
    // Collect scalars of the target type for use as RValues
    auto scalarsOfT = excluding(ctx.vars.scalarsOf(targetType), ctx.excludeName);
    bool hasRval = !scalarsOfT.empty();

    // Also collect scalars of ANY int type for casts
    auto allScalars = excluding(ctx.vars.allScalars(), ctx.excludeName);
    std::vector<const VarEntry *> otherIntScalars;
    for (auto *v: allScalars)
      if (TypeUtils::isInt(v->type) && !TypeUtils::areTypesEqual(v->type, targetType))
        otherIntScalars.push_back(v);

    std::uniform_int_distribution<int> slot(0, 99);
    int s = slot(ctx.rng);

    auto pickRval = [&]() -> RValue {
      assert(hasRval);
      auto *v = pickOne(ctx.rng, scalarsOfT);
      return localLV(v->name);
    };

    if (s < rysmith::hp::kIntOnPath_CoefBareEnd) {
      // Standalone coef sym
      return coefAtom(symCoef(ctx.sym->nextCoef(targetType)));
    }
    if (s < rysmith::hp::kIntOnPath_MulEnd && hasRval) {
      // Linear: sym * rval
      return opAtom(AtomOpKind::Mul, symCoef(ctx.sym->nextCoef(targetType)), pickRval());
    }
    if (s < rysmith::hp::kIntOnPath_BitwiseEnd && ctx.cfg.enableAllOps && hasRval) {
      // Bitwise
      static const AtomOpKind bops[] = {AtomOpKind::And, AtomOpKind::Or, AtomOpKind::Xor};
      std::uniform_int_distribution<int> opPick(0, 2);
      return opAtom(bops[opPick(ctx.rng)], symCoef(ctx.sym->nextCoef(targetType)), pickRval());
    }
    if (s < rysmith::hp::kIntOnPath_ShiftEnd && ctx.cfg.enableAllOps && hasRval) {
      // Shift: OpAtom{Shl/Shr/LShr, coef, rval} = coef SHIFT rval.
      // coef must match targetType; rval (shift amount) must be i32.
      // For i32 targets we use an index sym as the shifted value and
      // pick an i32 var for the amount; for non-i32 targets we fall
      // through to a standalone coef sym (no cross-width rval available).
      auto i32scalars = excluding(ctx.vars.scalarsOf(makeI32()), ctx.excludeName);
      if (!i32scalars.empty() || TypeUtils::isInt(targetType)) {
        static const AtomOpKind sops[] = {AtomOpKind::Shl, AtomOpKind::Shr, AtomOpKind::LShr};
        std::uniform_int_distribution<int> opPick(0, 2);
        auto idxSym = ctx.sym->nextIndex(); // always i32
        if (intBitWidth(targetType) == 32) {
          if (!i32scalars.empty()) {
            auto *shiftAmt = pickOne(ctx.rng, i32scalars);
            return opAtom(sops[opPick(ctx.rng)], symCoef(idxSym), localLV(shiftAmt->name));
          }
        } else {
          (void) idxSym; // index sym was consumed, drop it
          return coefAtom(symCoef(ctx.sym->nextCoef(targetType)));
        }
      }
      return coefAtom(symCoef(ctx.sym->nextCoef(targetType)));
    }
    if (s < rysmith::hp::kIntOnPath_UnaryNotEnd && hasRval) {
      // Unary NOT
      return unaryAtom(pickRval());
    }
    if (s < rysmith::hp::kIntOnPath_CastEnd && !otherIntScalars.empty()) {
      // CastAtom from another int width
      auto *srcVar = pickOne(ctx.rng, otherIntScalars);
      CastAtom ca;
      ca.src = LValue{LocalId{srcVar->name, {}}, {}, {}};
      ca.dstType = targetType;
      return Atom{std::move(ca), {}};
    }
    if (s < rysmith::hp::kIntOnPath_DivModEnd && ctx.cfg.enableDiv && hasRval) {
      // Div/Mod: OpAtom{Div/Mod, sym_coef, rval}  = sym / rval
      // We need a require(rval != 0) guard on-path
      std::uniform_int_distribution<int> dm(0, 1);
      AtomOpKind op = dm(ctx.rng) ? AtomOpKind::Mod : AtomOpKind::Div;
      auto *rv = pickOne(ctx.rng, scalarsOfT);
      std::string symName = ctx.sym->nextCoef(targetType);
      // Add require: rval != 0
      RequireInstr req;
      req.cond.lhs = simpleExpr(rvalAtom(localLV(rv->name)));
      req.cond.op = RelOp::NE;
      req.cond.rhs = simpleExpr(coefAtom(intCoef(0)));
      req.message = "div nonzero";
      ctx.extraRequires.push_back(Instr{std::move(req)});
      return opAtom(op, symCoef(symName), localLV(rv->name));
    }
    if (s < rysmith::hp::kIntOnPath_LoadEnd) {
      // Load from a ptr T var if any exist
      auto ptrs = excluding(ctx.vars.ptrsOf(targetType), ctx.excludeName);
      if (!ptrs.empty()) {
        auto *pv = pickOne(ctx.rng, ptrs);
        return Atom{LoadAtom{localLV(pv->name), {}}, {}};
      }
    }
    if (s < rysmith::hp::kIntOnPath_SelectEnd && ctx.cfg.enableSelect) {
      return genSelectAtom(ctx.rng, ctx.sym, ctx.vars, targetType, ctx.excludeName);
    }
    if (s < rysmith::hp::kIntOnPath_IntrinsicEnd && ctx.cfg.enableIntrinsics) {
      return genIntrinsicCallAtom(ctx, targetType);
    }
    // Fallback: standalone sym
    return coefAtom(symCoef(ctx.sym->nextCoef(targetType)));
  }

  // Off-path OpAtom coefficient: a same-type LocalId with probability
  // kPOffPathVarCoef (`%a * %b` — every operator admits a LocalId coef per
  // the grammar, both sides are runtime values the compiler can't fold,
  // and the solver never visits off-path blocks), else a per-width
  // literal. `nonzeroLit` keeps the literal branch's dividend nonzero for
  // div/mod (a 0 dividend folds the whole term).
  static Coef pickOffPathIntCoef(
      std::mt19937 &rng, const VarCatalogue &vars, const TypePtr &targetType,
      const std::optional<std::string> &excludeName, bool nonzeroLit
  ) {
    auto pool = excluding(vars.scalarsOf(targetType), excludeName);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    if (!pool.empty() && coin(rng) < rysmith::hp::kPOffPathVarCoef)
      return LocalOrSymId{LocalId{pickOne(rng, pool)->name, {}}};
    int64_t v = nonzeroLit ? pickConcreteIntLitNonzero(rng, targetType)
                           : pickConcreteIntLit(rng, targetType);
    return IntLit{v, {}};
  }

  // FP analogue. The literal branch draws from kFloatMulCoefPool (dyadic,
  // 0-free) so the term never folds to a constant.
  static Coef pickOffPathFpCoef(
      std::mt19937 &rng, const VarCatalogue &vars, const TypePtr &targetType,
      const std::optional<std::string> &excludeName
  ) {
    auto pool = excluding(vars.scalarsOf(targetType), excludeName);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    if (!pool.empty() && coin(rng) < rysmith::hp::kPOffPathVarCoef)
      return LocalOrSymId{LocalId{pickOne(rng, pool)->name, {}}};
    std::uniform_int_distribution<std::size_t> d(0, rysmith::hp::kFloatMulCoefPoolSize - 1);
    return FloatLit{rysmith::hp::kFloatMulCoefPool[d(rng)], {}};
  }

  // Generate a single Atom of the given integer type (off-path; never
  // executed at the solved inputs, so the only constraint is that it
  // typechecks and compiles)
  static Atom genIntAtomOffPath(
      std::mt19937 &rng, const VarCatalogue &vars, const TypePtr &targetType,
      const ExprGenConfig &cfg, const std::optional<std::string> &excludeName = std::nullopt
  ) {
    auto scalarsOfT = excluding(vars.scalarsOf(targetType), excludeName);
    bool hasRval = !scalarsOfT.empty();

    auto allScalars = excluding(vars.allScalars(), excludeName);
    std::vector<const VarEntry *> otherIntScalars;
    for (auto *v: allScalars)
      if (TypeUtils::isInt(v->type) && !TypeUtils::areTypesEqual(v->type, targetType))
        otherIntScalars.push_back(v);

    std::uniform_int_distribution<int> slot(0, 99);
    int s = slot(rng);

    if (s < rysmith::hp::kIntOffPath_ConcreteEnd || !hasRval) {
      return genConcreteIntAtom(rng, targetType);
    }
    if (s < rysmith::hp::kIntOffPath_MulEnd) {
      auto *v = pickOne(rng, scalarsOfT);
      return opAtom(
          AtomOpKind::Mul, pickOffPathIntCoef(rng, vars, targetType, excludeName, false),
          localLV(v->name)
      );
    }
    if (s < rysmith::hp::kIntOffPath_BitwiseEnd && cfg.enableAllOps) {
      auto *v = pickOne(rng, scalarsOfT);
      static const AtomOpKind bops[] = {AtomOpKind::And, AtomOpKind::Or, AtomOpKind::Xor};
      std::uniform_int_distribution<int> op(0, 2);
      return opAtom(
          bops[op(rng)], pickOffPathIntCoef(rng, vars, targetType, excludeName, false),
          localLV(v->name)
      );
    }
    if (s < rysmith::hp::kIntOffPath_ShiftEnd && cfg.enableAllOps) {
      // All three shift ops at any width. The shift amount is a
      // runtime var and may be negative / oversized at runtime — UB the
      // block never reaches; it only has to typecheck (coef width ==
      // rval width) and compile.
      auto *v = pickOne(rng, scalarsOfT);
      static const AtomOpKind sops[] = {AtomOpKind::Shl, AtomOpKind::Shr, AtomOpKind::LShr};
      std::uniform_int_distribution<int> op(0, 2);
      return opAtom(
          sops[op(rng)], pickOffPathIntCoef(rng, vars, targetType, excludeName, false),
          localLV(v->name)
      );
    }
    if (s < rysmith::hp::kIntOffPath_CastEnd && !otherIntScalars.empty()) {
      auto *v = pickOne(rng, otherIntScalars);
      CastAtom ca;
      ca.src = LValue{LocalId{v->name, {}}, {}, {}};
      ca.dstType = targetType;
      return Atom{std::move(ca), {}};
    }
    if (s < rysmith::hp::kIntOffPath_DivModEnd && cfg.enableDiv && hasRval) {
      // The divisor (the runtime LValue %v) carries a div-by-zero risk
      // this slot has always had — irrelevant, the block never executes.
      auto *v = pickOne(rng, scalarsOfT);
      std::uniform_int_distribution<int> dm(0, 1);
      AtomOpKind op = dm(rng) ? AtomOpKind::Mod : AtomOpKind::Div;
      return opAtom(
          op, pickOffPathIntCoef(rng, vars, targetType, excludeName, true), localLV(v->name)
      );
    }
    if (s < rysmith::hp::kIntOffPath_PlainRvalEnd) {
      auto *v = pickOne(rng, scalarsOfT);
      return rvalAtom(localLV(v->name));
    }
    if (s < rysmith::hp::kIntOffPath_LoadEnd) {
      // Load from a ptr T var if available
      auto ptrs = excluding(vars.ptrsOf(targetType), excludeName);
      if (!ptrs.empty()) {
        auto *pv = pickOne(rng, ptrs);
        return Atom{LoadAtom{localLV(pv->name), {}}, {}};
      }
    }
    if (s < rysmith::hp::kIntOffPath_SelectEnd && cfg.enableSelect) {
      return genSelectAtom(rng, /*sym=*/nullptr, vars, targetType, excludeName);
    }
    if (s < rysmith::hp::kIntOffPath_IntrinsicEnd && cfg.enableIntrinsics) {
      std::vector<Instr> dummyReqs; // off-path args are concrete-only, no requires expected
      ExprGenContext ctx{rng, nullptr, vars, cfg, false, dummyReqs, excludeName};
      return genIntrinsicCallAtom(ctx, targetType);
    }
    return genConcreteIntAtom(rng, targetType);
  }

  // Generate a float atom (on-path: cast from i32 sym or concrete float)
  static Atom genFloatAtomOnPath(
      std::mt19937 &rng, SymCounter &sym, const VarCatalogue &vars, const TypePtr &targetType,
      const ExprGenConfig &cfg, const std::optional<std::string> &excludeName = std::nullopt
  ) {
    auto fpVars = excluding(vars.scalarsOf(targetType), excludeName);
    auto i32scalars = excluding(vars.scalarsOf(makeI32()), excludeName);

    std::uniform_int_distribution<int> slot(0, 99);
    int s = slot(rng);

    if (s < rysmith::hp::kFloatOnPath_CastFromI32SymEnd) {
      // Cast from i32 sym to float (keeps SMT in BV theory)
      auto symName = sym.nextValue();
      CastAtom ca;
      ca.src = SymId{symName, {}};
      ca.dstType = targetType;
      return Atom{std::move(ca), {}};
    }
    if (s < rysmith::hp::kFloatOnPath_MulLitEnd && !fpVars.empty()) {
      // Multiply by concrete float literal
      auto *v = pickOne(rng, fpVars);
      std::uniform_int_distribution<std::size_t> ld(0, rysmith::hp::kFloatMulCoefPoolSize - 1);
      return opAtom(
          AtomOpKind::Mul, floatCoef(rysmith::hp::kFloatMulCoefPool[ld(rng)]), localLV(v->name)
      );
    }
    if (s < rysmith::hp::kFloatOnPath_CastFromVarEnd && !i32scalars.empty()) {
      // CastAtom from i32 var
      auto *v = pickOne(rng, i32scalars);
      CastAtom ca;
      ca.src = LValue{LocalId{v->name, {}}, {}, {}};
      ca.dstType = targetType;
      return Atom{std::move(ca), {}};
    }
    if (s < rysmith::hp::kFloatOnPath_SelectEnd && cfg.enableSelect) {
      return genSelectAtom(rng, &sym, vars, targetType, excludeName);
    }
    if (s < rysmith::hp::kFloatOnPath_IntrinsicEnd && cfg.enableIntrinsics) {
      // The only FP-typed intrinsics are the reductions; genIntrinsicCallAtom
      // falls back to a concrete literal when no FP vector operand is in
      // scope. FP args carry no div guards, so a discarded requires sink is
      // safe here.
      std::vector<Instr> reqSink;
      ExprGenContext ctx{rng, &sym, vars, cfg, true, reqSink, excludeName};
      return genIntrinsicCallAtom(ctx, targetType);
    }
    // Concrete float literal
    return genConcreteFloatAtom(rng);
  }

  // Off-path FP atom: about half var-ref, half concrete literal. Returning a
  // literal unconditionally would leave every off-path FP expression trivially
  // constant, with no runtime LValue reference for the trivial-shape post-check
  // to build on, so a same-typed FP var is used whenever one is in scope.
  static Atom genFloatAtomOffPath(
      std::mt19937 &rng, const VarCatalogue &vars, const TypePtr &targetType,
      const ExprGenConfig &cfg, const std::optional<std::string> &excludeName = std::nullopt
  ) {
    auto fpVars = excluding(vars.scalarsOf(targetType), excludeName);
    std::uniform_int_distribution<int> slot(0, 99);
    int s = slot(rng);
    if (fpVars.empty()) {
      // No FP scalar var to read/scale — an FP reduction is still a valid
      // same-typed source when a vector operand is in scope (else the call
      // helper itself falls back to a concrete literal).
      if (s < rysmith::hp::kFloatOffPath_IntrinsicEnd && cfg.enableIntrinsics) {
        std::vector<Instr> reqSink;
        ExprGenContext ctx{rng, nullptr, vars, cfg, false, reqSink, excludeName};
        return genIntrinsicCallAtom(ctx, targetType);
      }
      return genConcreteFloatAtom(rng);
    }
    if (s < rysmith::hp::kFloatOffPath_ReadEnd) {
      auto *v = pickOne(rng, fpVars);
      return rvalAtom(localLV(v->name));
    }
    // Floats admit Mul / Div / Mod (fmod) per the typechecker; the
    // div-by-zero / overflow UB these can hit at runtime never fires in an
    // off-path block, and bit-exactness is moot for code that never
    // executes — it only has to compile.
    auto pickVar = [&]() { return localLV(pickOne(rng, fpVars)->name); };
    if (s < rysmith::hp::kFloatOffPath_MulEnd)
      return opAtom(
          AtomOpKind::Mul, pickOffPathFpCoef(rng, vars, targetType, excludeName), pickVar()
      );
    if (s < rysmith::hp::kFloatOffPath_DivEnd)
      return opAtom(
          AtomOpKind::Div, pickOffPathFpCoef(rng, vars, targetType, excludeName), pickVar()
      );
    if (s < rysmith::hp::kFloatOffPath_ModEnd)
      return opAtom(
          AtomOpKind::Mod, pickOffPathFpCoef(rng, vars, targetType, excludeName), pickVar()
      );
    if (s < rysmith::hp::kFloatOffPath_IntrinsicEnd && cfg.enableIntrinsics) {
      std::vector<Instr> reqSink;
      ExprGenContext ctx{rng, nullptr, vars, cfg, false, reqSink, excludeName};
      return genIntrinsicCallAtom(ctx, targetType);
    }
    return genConcreteFloatAtom(rng);
  }

  // Generate a ptr-type atom. Collects all candidate atoms uniformly so each
  // option (addr, copy, load-from-ptr-ptr) appears with equal probability.
  static Atom genPtrAtom(
      std::mt19937 &rng, const VarCatalogue &vars, const TypePtr &targetType,
      const std::optional<std::string> &excludeName = std::nullopt
  ) {
    assert(TypeUtils::isPtr(targetType));
    TypePtr ptee = TypeUtils::pointee(targetType);

    std::vector<Atom> options;

    auto nameExcluded = [&](const std::string &n) { return excludeName && n == *excludeName; };

    // addr of any addressable var whose type equals the pointee
    for (auto *v: vars.allAddressable()) {
      if (nameExcluded(v->name))
        continue;
      if (TypeUtils::areTypesEqual(v->type, ptee)) {
        options.push_back(Atom{AddrAtom{localLV(v->name), {}}, {}});
      }
    }

    // addr of a SUB-lvalue rooted at a let-mut aggregate local
    // (`addr %t.f0`, `addr %a[1]`, and nested paths). Per SPEC §3.4.2 any
    // sub-lvalue of a `let mut` local is addressable; field/element
    // provenance is exactly the alias-analysis surface (-O3 SROA / TBAA)
    // that whole-var addr never reaches. Params (immutable) are excluded.
    std::function<void(const LValue &, const TypePtr &)> collectSub = [&](const LValue &lv,
                                                                          const TypePtr &t) {
      if (TypeUtils::areTypesEqual(t, ptee))
        options.push_back(Atom{AddrAtom{lv, {}}, {}});
      if (auto *at = std::get_if<ArrayType>(&t->v)) {
        for (uint64_t i = 0; i < at->size; i++) {
          LValue sub = lv;
          sub.accesses.push_back(AccessIndex{Index{IntLit{(int64_t) i, {}}}, {}});
          collectSub(sub, at->elem);
        }
      } else if (auto *st = std::get_if<StructType>(&t->v)) {
        for (const auto &sd: vars.structDecls) {
          if (sd.name.name != st->name.name)
            continue;
          for (const auto &f: sd.fields) {
            LValue sub = lv;
            sub.accesses.push_back(AccessField{f.name, {}});
            collectSub(sub, f.type);
          }
        }
      }
    };
    for (const auto &v: vars.vars) {
      if (v.isParam || nameExcluded(v.name) || !TypeUtils::isAggregate(v.type))
        continue;
      collectSub(LValue{LocalId{v.name, {}}, {}, {}}, v.type);
    }

    // copy from an existing ptr T var (same type)
    for (auto *pv: vars.ptrsOf(ptee)) {
      if (nameExcluded(pv->name))
        continue;
      options.push_back(rvalAtom(localLV(pv->name)));
    }

    // load from a ptr ptr T var to materialise a ptr T value
    for (auto *ppv: vars.ptrsOf(targetType)) {
      if (nameExcluded(ppv->name))
        continue;
      options.push_back(Atom{LoadAtom{localLV(ppv->name), {}}, {}});
    }

    // PtrIndexAtom: if ptee is scalar and there exists a
    // ptr [N] ptee var, we can ptrindex it at a safe index.
    if (TypeUtils::isScalar(ptee)) {
      for (const auto &v: vars.vars) {
        if (!TypeUtils::isPtr(v.type))
          continue;
        if (nameExcluded(v.name))
          continue;
        auto innerPtee = TypeUtils::pointee(v.type);
        if (!innerPtee || !std::holds_alternative<ArrayType>(innerPtee->v))
          continue;
        const auto &at = std::get<ArrayType>(innerPtee->v);
        if (TypeUtils::areTypesEqual(at.elem, ptee)) {
          std::uniform_int_distribution<int64_t> idxd(0, (int64_t) at.size - 1);
          PtrIndexAtom pi;
          pi.rval = localLV(v.name);
          pi.index = Index{IntLit{idxd(rng), {}}};
          options.push_back(Atom{std::move(pi), {}});
        }
      }
    }

    // PtrFieldAtom: if there exists a ptr @S var where @S has
    // a field of type ptee, we can ptrfield it.
    for (const auto &v: vars.vars) {
      if (!TypeUtils::isPtr(v.type))
        continue;
      if (nameExcluded(v.name))
        continue;
      auto innerPtee = TypeUtils::pointee(v.type);
      if (!innerPtee || !std::holds_alternative<StructType>(innerPtee->v))
        continue;
      const auto &st = std::get<StructType>(innerPtee->v);
      for (const auto &sd: vars.structDecls) {
        if (sd.name.name != st.name.name)
          continue;
        for (const auto &f: sd.fields) {
          if (TypeUtils::areTypesEqual(f.type, ptee)) {
            PtrFieldAtom pf;
            pf.rval = localLV(v.name);
            pf.field = f.name;
            options.push_back(Atom{std::move(pf), {}});
          }
        }
      }
    }

    if (!options.empty()) {
      std::uniform_int_distribution<int> d(0, (int) options.size() - 1);
      return std::move(options[d(rng)]);
    }
    return coefAtom(NullLit{{}});
  }

  // ---------------------------------------------------------------------------
  // genPtrArithRhs
  //
  // Build an in-bounds pointer-arithmetic RHS for reassigning a `ptr F`
  // (F scalar) local, or return nullopt if no provably-safe source is in
  // scope. Two shapes, drawn uniformly across every candidate:
  //
  //   (array)  `ptrindex %ap, b ± d`   off `%ap : ptr [N] F`
  //            b and b±d both in [0, N-1] — real elements (§7.5 rule 16),
  //            so the result stays dereferenceable on any path.
  //
  //   (struct) `ptrfield %sp, f_p ± d`  off `%sp : ptr @S`, where f_p and
  //            f_{p±d} belong to a maximal run of CONSECUTIVE same-type
  //            (== F) fields. A `ptrfield` result's provenance is the whole
  //            struct (§7.5 rule 15), so arithmetic may roam its sizeof(@S)
  //            bytes; the packed layout (no padding) makes the stride
  //            d·sizeof(F) land exactly on f_{p±d}, an F-typed cell, so the
  //            eventual load satisfies the typed-access check (rule 15b).
  //
  // The offset d is non-zero (genuine arithmetic, including the `- iN` form
  // a plain navigate never produces) and a literal, and provenance never
  // leaves the originating aggregate (rule 10 / 15), so no path-dependent UB
  // can arise — the result is safe to `load` even though we cannot know here
  // whether a later block will.
  // ---------------------------------------------------------------------------
  static std::optional<Expr> genPtrArithRhs(
      std::mt19937 &rng, const VarCatalogue &vars, const TypePtr &lhsPtee,
      const std::string &lhsName
  ) {
    struct Cand {
      bool isStruct;
      std::string src;                    // source pointer var name
      uint64_t arrayN = 0;                // array shape: element count N
      std::vector<std::string> runFields; // struct shape: same-type field run
    };

    std::vector<Cand> cands;

    for (const auto &v: vars.vars) {
      if (v.name == lhsName || !TypeUtils::isPtr(v.type))
        continue;
      auto ptee = TypeUtils::pointee(v.type);
      if (!ptee)
        continue;

      // Array source: `%ap : ptr [N] F` with N >= 2 (so a distinct base and
      // result index exist).
      if (auto *at = std::get_if<ArrayType>(&ptee->v)) {
        if (at->size >= 2 && TypeUtils::areTypesEqual(at->elem, lhsPtee))
          cands.push_back({false, v.name, at->size, {}});
        continue;
      }

      // Struct source: `%sp : ptr @S` with a run of >= 2 consecutive
      // same-type fields whose type matches F. Emit one candidate per run.
      if (auto *st = std::get_if<StructType>(&ptee->v)) {
        const StructDecl *sd = nullptr;
        for (const auto &d: vars.structDecls)
          if (d.name.name == st->name.name) {
            sd = &d;
            break;
          }
        if (!sd)
          continue;
        std::size_t i = 0;
        while (i < sd->fields.size()) {
          std::size_t j = i;
          while (j < sd->fields.size() &&
                 TypeUtils::areTypesEqual(sd->fields[j].type, sd->fields[i].type))
            j++;
          // [i, j) is a maximal run of identically-typed fields.
          if ((j - i) >= 2 && TypeUtils::areTypesEqual(sd->fields[i].type, lhsPtee)) {
            std::vector<std::string> run;
            for (std::size_t k = i; k < j; k++)
              run.push_back(sd->fields[k].name);
            cands.push_back({true, v.name, 0, std::move(run)});
          }
          i = j;
        }
      }
    }

    if (cands.empty())
      return std::nullopt;

    const Cand &c = cands[std::uniform_int_distribution<std::size_t>(0, cands.size() - 1)(rng)];

    // Navigable extent M (array elements / run fields), then a base position
    // b and a DISTINCT result position r within it. Drawing r over [0, M-2]
    // and bumping it past b yields a uniform pick from [0, M-1] \ {b}, so the
    // offset d = r - b is always non-zero.
    int64_t M = c.isStruct ? (int64_t) c.runFields.size() : (int64_t) c.arrayN;
    int64_t b = std::uniform_int_distribution<int64_t>(0, M - 1)(rng);
    int64_t r = std::uniform_int_distribution<int64_t>(0, M - 2)(rng);
    if (r >= b)
      r++;
    int64_t d = r - b;

    Atom nav;
    if (c.isStruct) {
      PtrFieldAtom pf;
      pf.rval = localLV(c.src);
      pf.field = c.runFields[(std::size_t) b];
      nav = Atom{std::move(pf), {}};
    } else {
      PtrIndexAtom pi;
      pi.rval = localLV(c.src);
      pi.index = Index{IntLit{b, {}}};
      nav = Atom{std::move(pi), {}};
    }

    Expr rhs;
    rhs.first = std::move(nav);
    rhs.rest.push_back({d > 0 ? AddOp::Plus : AddOp::Minus, coefAtom(intCoef(d > 0 ? d : -d)), {}});
    return rhs;
  }

  // ---------------------------------------------------------------------------
  // tryEmitPtrVarArith
  //
  // Emit the DIRECT pointer-variable arithmetic shape for reassigning
  // `%p2 : ptr F` (F scalar):
  //
  //   %p2 = %p1 ± d;             // ptr_var ± iN,  i ± d in [0, N-1]
  //
  // The step's left operand is a pointer READ FROM A VARIABLE, so this is
  // the only shape that exercises the `ptr_var ± iN` lowering path. Rather
  // than synthesising an anchor, we REUSE a pointer `%p1` already live in
  // this block whose current value is a known in-bounds array element — its
  // most-recent assignment is `%p1 = ptrindex %ap, <i>` or
  // `%p1 = addr %a[<i>]` (literal i; `%ap : ptr [N] F` / `%a : [N] F`,
  // F == lhsPtee, N ≥ 2) and it has not been reassigned since. Those anchors
  // are emitted organically by genPtrAtom's ptrindex / sub-lvalue-addr
  // slots, so `%p1` carries genuine provenance and `%p1 ± d` is real,
  // non-foldable pointer arithmetic (not a throwaway def the optimizer folds
  // straight back into a single ptrindex).
  //
  // Safety is local def-use: `%p1`'s provenance is the array it was indexed
  // from (§7.5 rule 15) and its index `i` is a literal recovered from the
  // anchor, so `i ± d ∈ [0, N-1]` keeps the result inside that object and
  // load-safe on any path. The anchor's own validity (`%ap` non-null,
  // in-bounds i) is already established where it was emitted — by the solver
  // on-path, by the [0, N-1] index choice off-path — and reassigning `%ap`
  // later cannot disturb `%p1`, whose pointer value was captured at the
  // ptrindex/addr.
  //
  // Returns true (pushing one step) on success, false (pushing nothing) when
  // no reusable anchor is live, so the caller falls back to the combined
  // ptrindex/ptrfield expression form.
  // ---------------------------------------------------------------------------
  static bool tryEmitPtrVarArith(
      std::vector<Instr> &result, std::mt19937 &rng, const VarCatalogue &vars,
      const TypePtr &lhsPtee, const std::string &lhsName
  ) {
    auto typeOf = [&](const std::string &n) -> TypePtr {
      for (const auto &v: vars.vars)
        if (v.name == n)
          return v.type;
      return nullptr;
    };
    // `[N] F` with F == lhsPtee and N >= 2 → N, else nullopt.
    auto arrayN = [&](const TypePtr &t) -> std::optional<int64_t> {
      if (t)
        if (auto *at = std::get_if<ArrayType>(&t->v))
          if (at->size >= 2 && TypeUtils::areTypesEqual(at->elem, lhsPtee))
            return (int64_t) at->size;
      return std::nullopt;
    };

    struct Anchor {
      std::string ptr;
      int64_t N;
      int64_t i;
    };

    std::vector<Anchor> anchors;
    std::vector<std::string> seen; // pointers whose latest assignment we've passed

    // Walk the block's emitted statements newest-first; the first assignment
    // to a given pointer is its current value (a later one would clobber it).
    for (auto it = result.rbegin(); it != result.rend(); ++it) {
      auto *asg = std::get_if<AssignInstr>(&*it);
      if (!asg || !asg->lhs.accesses.empty())
        continue; // only a plain `%p = ...` defines a pointer's current value
      const std::string &p = asg->lhs.base.name;
      auto pty = typeOf(p);
      if (!pty || !TypeUtils::isPtr(pty))
        continue;
      if (std::find(seen.begin(), seen.end(), p) != seen.end())
        continue; // an older, already-superseded assignment to this pointer
      seen.push_back(p);
      if (!TypeUtils::areTypesEqual(TypeUtils::pointee(pty), lhsPtee) || !asg->rhs.rest.empty())
        continue; // pointee must match %p2; the anchor must be a single atom
      const Atom &a = asg->rhs.first;
      // An anchor needs at least two cells: the offset is drawn as a distinct
      // index in [0, N-1] \ {i}, which is empty for a one-element array.
      if (auto *pix = std::get_if<PtrIndexAtom>(&a.v)) {
        // %p = ptrindex %ap, <IntLit i>
        auto *il = std::get_if<IntLit>(&pix->index);
        if (il && pix->rval.accesses.empty())
          if (auto n = arrayN(TypeUtils::pointee(typeOf(pix->rval.base.name))))
            if (*n >= 2 && il->value >= 0 && il->value < *n)
              anchors.push_back({p, *n, il->value});
      } else if (auto *ad = std::get_if<AddrAtom>(&a.v)) {
        // %p = addr %a[<IntLit i>]
        if (ad->lv.accesses.size() == 1)
          if (auto *aidx = std::get_if<AccessIndex>(&ad->lv.accesses[0]))
            if (auto *il = std::get_if<IntLit>(&aidx->index))
              if (auto n = arrayN(typeOf(ad->lv.base.name)))
                if (*n >= 2 && il->value >= 0 && il->value < *n)
                  anchors.push_back({p, *n, il->value});
      } else if (auto *pf = std::get_if<PtrFieldAtom>(&a.v)) {
        // %p = ptrfield %sp, <field f>. Treat the maximal run of consecutive
        // same-type fields containing f as the "array": f sits at position
        // (idx - lo) within a run of length (hi - lo + 1), and stepping ± d
        // lands on a same-typed field cell (packed layout → exact stride).
        if (pf->rval.accesses.empty())
          if (auto inner = TypeUtils::pointee(typeOf(pf->rval.base.name)))
            if (auto *st = std::get_if<StructType>(&inner->v))
              for (const auto &sd: vars.structDecls) {
                if (sd.name.name != st->name.name)
                  continue;
                int idx = -1;
                for (int k = 0; k < (int) sd.fields.size(); k++)
                  if (sd.fields[k].name == pf->field) {
                    idx = k;
                    break;
                  }
                if (idx >= 0 && TypeUtils::areTypesEqual(sd.fields[idx].type, lhsPtee)) {
                  int lo = idx, hi = idx;
                  while (lo > 0 &&
                         TypeUtils::areTypesEqual(sd.fields[lo - 1].type, sd.fields[idx].type))
                    lo--;
                  while (hi + 1 < (int) sd.fields.size() &&
                         TypeUtils::areTypesEqual(sd.fields[hi + 1].type, sd.fields[idx].type))
                    hi++;
                  if (hi - lo + 1 >= 2)
                    anchors.push_back({p, hi - lo + 1, idx - lo});
                }
                break;
              }
      }
    }

    if (anchors.empty())
      return false;

    const auto &an =
        anchors[std::uniform_int_distribution<std::size_t>(0, anchors.size() - 1)(rng)];
    // Distinct result index r in [0, N-1] \ {i}, so the offset d is non-zero.
    int64_t r = std::uniform_int_distribution<int64_t>(0, an.N - 2)(rng);
    if (r >= an.i)
      r++;
    int64_t d = r - an.i;

    Expr step;
    step.first = rvalAtom(localLV(an.ptr));
    step.rest.push_back(
        {d > 0 ? AddOp::Plus : AddOp::Minus, coefAtom(intCoef(d > 0 ? d : -d)), {}}
    );
    result.push_back(Instr{AssignInstr{localLV(lhsName), std::move(step), {}}});
    return true;
  }

  // ---------------------------------------------------------------------------
  // Uniform aggregate-pointer navigation: tryEmitNavChain
  //
  // The general form of nested navigation, replacing the array-only special
  // case. For a `ptr F` reassign (F loadable), find an in-scope aggregate
  // pointer `%app : ptr A` and a chain of navigation steps that reaches a
  // sub-element of type F, staging each intermediate through a `let mut`
  // local (ptrindex/ptrfield results are Atoms, not lvalues):
  //
  //   array  [N] U  ->  %s = ptrindex %p, i      (U next)
  //   struct @S     ->  %s = ptrfield %p, f      (field type next)
  //
  // applied uniformly and recursively until the leaf type F is reached, e.g.
  // `ptr [N][M] F -> ptr [M] F -> ptr F`, `ptr [N] @S -> ptr @S -> ptr Ffld`,
  // or any mix. Indices are in-bounds literals and field paths are static, so
  // the leaf is dereferenceable on any path. Only chains of >= 2 steps are
  // emitted here; the single-step navigations are genPtrAtom's job.
  // ---------------------------------------------------------------------------
  struct NavStep {
    bool isField;
    std::string field;  // ptrfield field name
    int64_t arraySize;  // ptrindex bound (when !isField)
    TypePtr resultPtee; // pointee type produced by this step
  };

  // Collect navigation paths from aggregate type `A` to a sub-element of type
  // `target`. Depth-bounded; each emitted path's last step lands on `target`.
  static void collectNavPaths(
      const VarCatalogue &vars, const TypePtr &A, const TypePtr &target, std::vector<NavStep> &cur,
      std::vector<std::vector<NavStep>> &out, int maxDepth
  ) {
    if ((int) cur.size() >= maxDepth || out.size() >= 16)
      return;
    if (auto *at = std::get_if<ArrayType>(&A->v)) {
      cur.push_back({false, "", (int64_t) at->size, at->elem});
      if (TypeUtils::areTypesEqual(at->elem, target))
        out.push_back(cur);
      if (TypeUtils::isAggregate(at->elem))
        collectNavPaths(vars, at->elem, target, cur, out, maxDepth);
      cur.pop_back();
    } else if (auto *st = std::get_if<StructType>(&A->v)) {
      for (const auto &sd: vars.structDecls) {
        if (sd.name.name != st->name.name)
          continue;
        for (const auto &f: sd.fields) {
          cur.push_back({true, f.name, 0, f.type});
          if (TypeUtils::areTypesEqual(f.type, target))
            out.push_back(cur);
          if (TypeUtils::isAggregate(f.type))
            collectNavPaths(vars, f.type, target, cur, out, maxDepth);
          cur.pop_back();
        }
        break;
      }
    }
  }

  static bool tryEmitNavChain(
      std::vector<Instr> &result, std::mt19937 &rng, const VarCatalogue &vars,
      const TypePtr &lhsPtee, const std::string &lhsName
  ) {
    if (!TypeUtils::isScalar(lhsPtee) && !TypeUtils::isPtr(lhsPtee))
      return false; // leaf must be loadable

    struct Cand {
      std::string app;
      std::vector<NavStep> path;
    };

    std::vector<Cand> cands;
    for (const auto &v: vars.vars) {
      if (!TypeUtils::isPtr(v.type))
        continue;
      auto A = TypeUtils::pointee(v.type);
      if (!A || !TypeUtils::isAggregate(A))
        continue;
      std::vector<NavStep> cur;
      std::vector<std::vector<NavStep>> paths;
      collectNavPaths(vars, A, lhsPtee, cur, paths, /*maxDepth=*/3);
      for (auto &p: paths)
        if (p.size() >= 2) // single-step navigations are genPtrAtom's job
          cands.push_back({v.name, std::move(p)});
    }
    if (cands.empty())
      return false;
    std::shuffle(cands.begin(), cands.end(), rng);

    // Find an in-scope staging local of type `ptr ptee` (not the LHS).
    auto findStage = [&](const TypePtr &ptee) -> std::optional<std::string> {
      std::vector<std::string> opts;
      for (const auto &v: vars.vars) {
        if (v.isParam || v.name == lhsName || !TypeUtils::isPtr(v.type))
          continue;
        if (TypeUtils::areTypesEqual(TypeUtils::pointee(v.type), ptee))
          opts.push_back(v.name);
      }
      if (opts.empty())
        return std::nullopt;
      return opts[std::uniform_int_distribution<std::size_t>(0, opts.size() - 1)(rng)];
    };

    for (auto &c: cands) {
      // Staging locals for every intermediate step (all but the last).
      std::vector<std::string> stages;
      bool ok = true;
      for (std::size_t i = 0; i + 1 < c.path.size(); i++) {
        auto s = findStage(c.path[i].resultPtee);
        if (!s) {
          ok = false;
          break;
        }
        stages.push_back(*s);
      }
      if (!ok)
        continue;

      std::string cur = c.app;
      for (std::size_t i = 0; i < c.path.size(); i++) {
        const NavStep &st = c.path[i];
        const std::string &dst = (i + 1 < c.path.size()) ? stages[i] : lhsName;
        Atom nav;
        if (st.isField) {
          PtrFieldAtom pf;
          pf.rval = localLV(cur);
          pf.field = st.field;
          nav = Atom{std::move(pf), {}};
        } else {
          PtrIndexAtom pi;
          pi.rval = localLV(cur);
          pi.index =
              Index{IntLit{std::uniform_int_distribution<int64_t>(0, st.arraySize - 1)(rng), {}}};
          nav = Atom{std::move(pi), {}};
        }
        Expr e;
        e.first = std::move(nav);
        result.push_back(Instr{AssignInstr{localLV(dst), std::move(e), {}}});
        cur = dst;
      }
      return true;
    }
    return false;
  }

  // ---------------------------------------------------------------------------
  // Trivial-shape post-check helpers
  //
  // Shared by `genExpr` and `genExprWithRequires`. The two callers used to
  // carry near-identical inline lambdas + post-check blocks; factoring
  // here keeps the trivial-shape contract in one place. Only int / fp
  // target types are supported — ptr / vec / struct are shape-restricted
  // by their own dispatch and shouldn't be reshaped here.
  // ---------------------------------------------------------------------------

  // Thin wrapper around the per-direction × per-class atom dispatch.
  // Threads `extraRequires` so on-path int atoms can publish their
  // div-by-zero guards (the off-path/FP paths don't touch it).
  static Atom genOneAtomOfType(const ExprGenContext &ctx, const TypePtr &targetType) {
    if (TypeUtils::isInt(targetType))
      return (ctx.onPath && ctx.sym)
                 ? genIntAtomOnPath(ctx, targetType)
                 : genIntAtomOffPath(ctx.rng, ctx.vars, targetType, ctx.cfg, ctx.excludeName);
    return (ctx.onPath && ctx.sym)
               ? genFloatAtomOnPath(
                     ctx.rng, *ctx.sym, ctx.vars, targetType, ctx.cfg, ctx.excludeName
                 )
               : genFloatAtomOffPath(ctx.rng, ctx.vars, targetType, ctx.cfg, ctx.excludeName);
  }

  // Build a cheap, solver-linear atom that reads a runtime LValue and
  // is UB-safe. The chain, each step falling through when its pool is
  // empty:
  //   1. concrete-coef multiply or plain read of a same-type scalar — the
  //      coef is a moderate nonzero literal (kCheapMulCoef*) for ints and a
  //      kFloatMulCoefPool dyadic for floats, so the on-path no-overflow
  //      constraint degrades to a loose linear bound instead of a nonlinear
  //      `sym * var` term (`allowBareRead=false` forces the multiply so the
  //      result can replace a plain copy without recreating one);
  //   2. a cast from a differently-typed scalar, restricted to total
  //      conversions — int→int, int→fp, and the widening f32→f64. A
  //      narrowing fp→fp (f64→f32) can overflow to ±inf and fp→int can
  //      be out-of-range, both UB, so they are excluded;
  //   3. a load through a ptr-to-targetType.
  // Returns nullopt when no readable LValue exists besides the excluded
  // LHS — a genuinely input-free pool cannot carry a runtime dependency.
  static std::optional<Atom> genCheapLinearAtom(
      std::mt19937 &rng, const VarCatalogue &vars, const TypePtr &targetType,
      const std::optional<std::string> &excludeName, bool allowBareRead
  ) {
    auto same = excluding(vars.scalarsOf(targetType), excludeName);
    if (!same.empty()) {
      auto *v = pickOne(rng, same);
      std::uniform_int_distribution<int> slot(0, 99);
      bool mul = !allowBareRead || slot(rng) < rysmith::hp::kCheapAtom_MulEnd;
      if (mul && TypeUtils::isInt(targetType)) {
        std::uniform_int_distribution<int64_t> d(
            rysmith::hp::kCheapMulCoefLo, rysmith::hp::kCheapMulCoefHi
        );
        int64_t c = d(rng);
        if (c == 0)
          c = 2; // a 0 coef folds the term and drops the read
        return opAtom(AtomOpKind::Mul, intCoef(c), localLV(v->name));
      }
      if (mul && TypeUtils::isFloat(targetType)) {
        std::uniform_int_distribution<std::size_t> d(0, rysmith::hp::kFloatMulCoefPoolSize - 1);
        return opAtom(
            AtomOpKind::Mul, floatCoef(rysmith::hp::kFloatMulCoefPool[d(rng)]), localLV(v->name)
        );
      }
      return rvalAtom(localLV(v->name));
    }
    auto scalarWidth = [](const TypePtr &t) -> uint32_t {
      if (TypeUtils::isInt(t))
        return intBitWidth(t);
      if (auto *ft = std::get_if<FloatType>(&t->v))
        return ft->kind == FloatType::Kind::F32 ? 32u : 64u;
      return 0;
    };
    auto safeCastSource = [&](const TypePtr &src) -> bool {
      if (TypeUtils::areTypesEqual(src, targetType))
        return false; // same type → covered by the read/mul step
      if (TypeUtils::isInt(targetType))
        return TypeUtils::isInt(src); // int→int is total; fp→int can be range-UB
      if (TypeUtils::isInt(src))
        return true; // int→fp is always finite
      return TypeUtils::isFloat(src) && scalarWidth(src) < scalarWidth(targetType); // widening only
    };
    auto allScalars = excluding(vars.allScalars(), excludeName);
    for (auto *v: allScalars) {
      if (!safeCastSource(v->type))
        continue;
      CastAtom ca;
      ca.src = LValue{LocalId{v->name, {}}, {}, {}};
      ca.dstType = targetType;
      return Atom{std::move(ca), {}};
    }
    auto ptrs = excluding(vars.ptrsOf(targetType), excludeName);
    if (!ptrs.empty()) {
      auto *pv = pickOne(rng, ptrs);
      return Atom{LoadAtom{localLV(pv->name), {}}, {}};
    }
    return std::nullopt;
  }

  // Generate an atom that carries a runtime LValue dependency.
  //
  // On-path the cheap-linear chain is used directly: rerolling the slot
  // table would converge on `sym * var` — a fresh free variable plus a
  // nonlinear BV multiply, the most solver-expensive replacement possible
  // for what was a solver-free all-literal RHS.
  //
  // Off-path (solver never looks) the standard dispatch is rerolled up to
  // `kAtomRerollMaxAttempts` times for shape diversity before the same
  // cheap chain takes over. When even the chain has no readable LValue the
  // last trivial roll is returned — the kPAllowAllLiteral slice absorbs it.
  //
  // `allowBareRead=false` additionally rules out a bare RValueAtom result:
  // a caller about to use the atom as a whole single-atom RHS would
  // otherwise manufacture a plain copy, whose rate must stay solely under
  // kPAllowPlainCopy's control.
  static Atom genNonTrivialAtomOfType(
      const ExprGenContext &ctx, const TypePtr &targetType, bool allowBareRead = true
  ) {
    if (ctx.onPath && !ctx.cfg.condContext) {
      if (auto cheap =
              genCheapLinearAtom(ctx.rng, ctx.vars, targetType, ctx.excludeName, allowBareRead))
        return std::move(*cheap);
      return TypeUtils::isFloat(targetType) ? genConcreteFloatAtom(ctx.rng)
                                            : genConcreteIntAtom(ctx.rng, targetType);
    }
    Atom last = genOneAtomOfType(ctx, targetType);
    auto acceptable = [&](const Atom &a) {
      return !isTriviallyConstantAtom(a) &&
             (allowBareRead || !std::holds_alternative<RValueAtom>(a.v));
    };
    if (acceptable(last))
      return last;
    for (int r = 1; r < rysmith::hp::kAtomRerollMaxAttempts; r++) {
      Atom cand = genOneAtomOfType(ctx, targetType);
      if (acceptable(cand))
        return cand;
      last = std::move(cand);
    }
    if (auto cheap =
            genCheapLinearAtom(ctx.rng, ctx.vars, targetType, ctx.excludeName, allowBareRead))
      return std::move(*cheap);
    return last;
  }

  // Apply the trivial-shape post-check in place. With probability
  // `1 - kPAllowAllLiteral`, if every atom is trivially constant the
  // single-atom case grows to two atoms by appending a non-trivial
  // tail; the multi-atom case has its last atom rewritten. Replacing
  // the last (not the first) atom keeps the typechecker's first-atom
  // width inference intact.
  static void rewriteIfAllTriviallyConstant(
      const ExprGenContext &ctx, const TypePtr &targetType, std::vector<Atom> &atoms
  ) {
    if (!TypeUtils::isInt(targetType) && !TypeUtils::isFloat(targetType))
      return;
    std::uniform_real_distribution<double> allowCoin(0.0, 1.0);
    if (allowCoin(ctx.rng) < rysmith::hp::kPAllowAllLiteral)
      return;
    bool allTrivial = std::all_of(atoms.begin(), atoms.end(), [](const Atom &a) {
      return isTriviallyConstantAtom(a);
    });
    if (!allTrivial)
      return;
    // When the replacement becomes the WHOLE RHS (single atom replaced
    // in place), a bare read would manufacture a plain copy out of an
    // all-literal RHS — copy frequency must stay solely under
    // kPAllowPlainCopy's control, so bare reads are ruled out here. Tail
    // positions (append / last-atom replacement) keep them: a read joined
    // by +/- is not a copy.
    bool inPlaceWholeRhs = atoms.size() == 1 && ctx.cfg.maxAtoms <= 1;
    Atom replacement = genNonTrivialAtomOfType(ctx, targetType, /*allowBareRead=*/!inPlaceWholeRhs);
    if (atoms.size() == 1) {
      if (ctx.cfg.maxAtoms > 1) {
        atoms.push_back(std::move(replacement));
      } else {
        atoms[0] = std::move(replacement);
      }
    } else {
      atoms.back() = std::move(replacement);
    }
  }

  // ---------------------------------------------------------------------------
  // Main genExpr
  // ---------------------------------------------------------------------------

  Expr genExpr(
      std::mt19937 &rng, SymCounter *sym, const VarCatalogue &vars, const TypePtr &targetType,
      bool onPath, const ExprGenConfig &cfg, const std::optional<std::string> &excludeName
  ) {
    // Build a 1-3 atom expression (first + rest).
    // Note: div/mod safety requires are lost in this public API;
    // use genBlockStmts which calls genExprWithRequires internally.
    std::vector<Instr> dummyReqs;
    ExprGenContext ctx{rng, sym, vars, cfg, onPath, dummyReqs, excludeName};
    std::vector<Atom> atoms;

    std::uniform_int_distribution<int> nAtomsDist(cfg.minAtoms, cfg.maxAtoms);
    int nAtoms = nAtomsDist(rng);

    // For ptr types, always single atom
    if (TypeUtils::isPtr(targetType)) {
      nAtoms = 1;
    }
    // For float types, limit to 1-2 atoms
    if (TypeUtils::isFloat(targetType) && nAtoms > 2) {
      nAtoms = 2;
    }
    // Vec types: 1-2 atoms (lane-wise +/- is valid).
    if (TypeUtils::isVec(targetType)) {
      nAtoms = std::min(nAtoms, 2);
    }

    for (int i = 0; i < nAtoms; i++) {
      Atom a;
      if (TypeUtils::isInt(targetType)) {
        if (onPath && sym) {
          a = genIntAtomOnPath(ctx, targetType);
        } else {
          a = genIntAtomOffPath(rng, vars, targetType, cfg, excludeName);
        }
      } else if (TypeUtils::isFloat(targetType)) {
        if (onPath && sym) {
          a = genFloatAtomOnPath(rng, *sym, vars, targetType, cfg, excludeName);
        } else {
          a = genFloatAtomOffPath(rng, vars, targetType, cfg, excludeName);
        }
      } else if (TypeUtils::isPtr(targetType)) {
        a = genPtrAtom(rng, vars, targetType, excludeName);
      } else if (TypeUtils::isVec(targetType)) {
        // Vec atom generation.
        auto vecs = excluding(vars.vecsOf(targetType), excludeName);
        const auto &vt = std::get<VecType>(targetType->v);

        if (i == 0) {
          // First atom MUST be vec-typed (RValueAtom or OpAtom on a vec var).
          // A bare scalar literal as first atom would fail the typechecker.
          if (!vecs.empty()) {
            std::uniform_int_distribution<int> slot(0, 99);
            int s = slot(rng);
            if (s < rysmith::hp::kVecCopyEnd) {
              auto *v = pickOne(rng, vecs);
              a = rvalAtom(localLV(v->name));
            } else if (s < rysmith::hp::kVecSymMulEnd && onPath && sym) {
              auto *v = pickOne(rng, vecs);
              a = opAtom(AtomOpKind::Mul, symCoef(sym->nextCoef(vt.elem)), localLV(v->name));
            } else if (s < rysmith::hp::kVecConcMulEnd) {
              auto *v = pickOne(rng, vecs);
              int64_t c = std::uniform_int_distribution<
                  int64_t>(rysmith::hp::kVecConcMulLo, rysmith::hp::kVecConcMulHi)(rng);
              if (c == 0)
                c = 1;
              a = opAtom(AtomOpKind::Mul, intCoef(c), localLV(v->name));
            } else {
              auto *v = pickOne(rng, vecs);
              a = rvalAtom(localLV(v->name));
            }
          } else {
            // No vec vars — shouldn't happen for whole-vec assign (guarded
            // by genBlockStmts), but safety fallback.
            CoefAtom ca;
            ca.coef = IntLit{0, {}};
            a = Atom{std::move(ca), {}};
          }
        } else {
          // Tail atom (i > 0): can be vec-typed OR a broadcast scalar literal.
          std::uniform_int_distribution<int> tailSlot(0, 99);
          int ts = tailSlot(rng);
          if (ts < rysmith::hp::kVecTailCopyEnd && !vecs.empty()) {
            // Vec copy (lane-wise +/- with another vec var).
            auto *v = pickOne(rng, vecs);
            a = rvalAtom(localLV(v->name));
          } else if (ts < rysmith::hp::kVecTailOpEnd && !vecs.empty()) {
            // OpAtom on vec var.
            auto *v = pickOne(rng, vecs);
            int64_t c = std::uniform_int_distribution<
                int64_t>(rysmith::hp::kVecConcMulLo, rysmith::hp::kVecConcMulHi)(rng);
            if (c == 0)
              c = 1;
            a = opAtom(AtomOpKind::Mul, intCoef(c), localLV(v->name));
          } else {
            // Broadcast scalar literal (valid in +/- tail position). The
            // int draw uses the same per-width concrete pool as bare
            // literal atoms — pre-consolidation this was a hardcoded
            // [-8, 8] uniform, which narrowed the literal magnitude
            // distribution well below what the compiler folds.
            CoefAtom ca;
            if (TypeUtils::isFloat(vt.elem)) {
              std::uniform_int_distribution<std::size_t> fd(0, rysmith::hp::kFloatLitPoolSize - 1);
              ca.coef = FloatLit{rysmith::hp::kFloatLitPool[fd(rng)], {}};
            } else {
              ca.coef = IntLit{pickConcreteIntLit(rng, vt.elem), {}};
            }
            a = Atom{std::move(ca), {}};
          }
        }
      } else {
        // Unknown type (e.g. struct) — fallback to i32 concrete atom
        a = genConcreteIntAtom(rng, makeI32());
      }
      atoms.push_back(std::move(a));
    }

    // Trivial-shape post-check. `genExpr` is also called for intrinsic
    // arguments — `call @foo(3, 5)` is just as foldable as a body
    // `%x = 3 + 5;` and benefits from the same reshape.
    rewriteIfAllTriviallyConstant(ctx, targetType, atoms);

    // Build Expr from atoms
    Expr expr;
    expr.first = std::move(atoms[0]);
    std::uniform_real_distribution<double> addOpCoin(0.0, 1.0);
    for (std::size_t i = 1; i < atoms.size(); i++) {
      // For float, always use Plus (subtraction is fine but let's keep it simple)
      AddOp op =
          TypeUtils::isFloat(targetType)
              ? AddOp::Plus
              : (addOpCoin(rng) < rysmith::hp::kPTailAddOpIsPlus ? AddOp::Plus : AddOp::Minus);
      expr.rest.push_back({op, std::move(atoms[i]), {}});
    }

    (void) dummyReqs; // div/mod requires are handled by genExprWithRequires / genBlockStmts
    return expr;
  }

  // ---------------------------------------------------------------------------
  // genExprWithRequires — internal, returns both expr and any safety requires
  // ---------------------------------------------------------------------------

  // Generate the first atom of an off-path integer expression.
  // MUST produce an atom whose type is unambiguously targetType (not just
  // inferred from context). This is because the TypeChecker uses the first
  // atom's type to set the expected type for tail atoms. If the first atom
  // is a bare IntLit with no context, it defaults to i32 which may mismatch
  // the tail atoms that reference actual vars of targetType.
  static Atom genFirstIntAtomOffPath(
      std::mt19937 &rng, const VarCatalogue &vars, const TypePtr &targetType,
      [[maybe_unused]] const ExprGenConfig &cfg,
      const std::optional<std::string> &excludeName = std::nullopt
  ) {
    auto scalarsOfT = excluding(vars.scalarsOf(targetType), excludeName);
    auto allScalars = excluding(vars.allScalars(), excludeName);
    std::vector<const VarEntry *> otherIntScalars;
    for (auto *v: allScalars)
      if (TypeUtils::isInt(v->type) && !TypeUtils::areTypesEqual(v->type, targetType))
        otherIntScalars.push_back(v);

    if (!scalarsOfT.empty()) {
      // RValueAtom: type is determined by the variable's declared type
      auto *v = pickOne(rng, scalarsOfT);
      return rvalAtom(localLV(v->name));
    }
    if (!otherIntScalars.empty()) {
      // CastAtom: explicitly typed by dstType
      auto *v = pickOne(rng, otherIntScalars);
      CastAtom ca;
      ca.src = LValue{LocalId{v->name, {}}, {}, {}};
      ca.dstType = targetType;
      return Atom{std::move(ca), {}};
    }
    // No variables at all — bare literal (OK for single-atom expressions
    // where expectedBits comes from the assignment context)
    return genConcreteIntAtom(rng, targetType);
  }

  // Generate the first atom of an on-path integer expression.
  // For non-i32 types, must produce an atom whose type is unambiguously
  // targetType so the printed concrete program (after symbol substitution)
  // still type-checks. A bare CoefAtom{SymId} prints as a bare integer after
  // substitution, which the TypeChecker defaults to i32 — causing a bitwidth
  // mismatch when targetType is wider/narrower. We use CastAtom{SymId, dst}
  // which prints as "<value> as <type>" and preserves the type annotation.
  static Atom genFirstIntAtomOnPath(
      std::mt19937 &rng, SymCounter &sym, const VarCatalogue &vars, const TypePtr &targetType,
      const ExprGenConfig &cfg, std::vector<Instr> &extraRequires,
      const std::optional<std::string> &excludeName = std::nullopt
  ) {
    // For i32, a bare CoefAtom is fine (i32 is the default inferred type).
    if (intBitWidth(targetType) == 32) {
      ExprGenContext ctx{rng, &sym, vars, cfg, true, extraRequires, excludeName};
      return genIntAtomOnPath(ctx, targetType);
    }
    // Non-i32: prefer RValueAtom (variable of targetType) since its type is
    // inferred from the declaration. Failing that, wrap a sym coef in a CastAtom.
    auto scalarsOfT = excluding(vars.scalarsOf(targetType), excludeName);
    if (!scalarsOfT.empty()) {
      // 50% chance of RValueAtom vs CastAtom{sym}
      std::uniform_int_distribution<int> coin(0, 1);
      if (coin(rng)) {
        auto *v = pickOne(rng, scalarsOfT);
        return rvalAtom(localLV(v->name));
      }
    }
    // CastAtom{SymId, targetType}: prints as "<solved_value> as <type>"
    CastAtom ca;
    ca.src = SymId{sym.nextCoef(targetType), {}};
    ca.dstType = targetType;
    return Atom{std::move(ca), {}};
  }

  static std::pair<Expr, std::vector<Instr>> genExprWithRequires(
      std::mt19937 &rng, SymCounter *sym, const VarCatalogue &vars, const TypePtr &targetType,
      bool onPath, const ExprGenConfig &cfg, const std::optional<std::string> &excludeName
  ) {
    std::vector<Instr> reqs;
    ExprGenContext ctx{rng, sym, vars, cfg, onPath, reqs, excludeName};
    std::vector<Atom> atoms;

    std::uniform_int_distribution<int> nAtomsDist(cfg.minAtoms, cfg.maxAtoms);
    int nAtoms = nAtomsDist(rng);

    if (TypeUtils::isPtr(targetType))
      nAtoms = 1;
    if (TypeUtils::isFloat(targetType) && nAtoms > 2)
      nAtoms = 2;

    for (int i = 0; i < nAtoms; i++) {
      Atom a;
      if (TypeUtils::isInt(targetType)) {
        if (onPath && sym) {
          // For i==0, use genFirstIntAtomOnPath to ensure non-i32 types get
          // an explicitly-typed first atom. A bare CoefAtom{SymId} would print
          // as a bare integer literal after model substitution, defaulting to
          // i32.
          if (i == 0) {
            a = genFirstIntAtomOnPath(rng, *sym, vars, targetType, cfg, reqs, excludeName);
          } else {
            a = genIntAtomOnPath(ctx, targetType);
          }
        } else {
          // The first atom of an off-path expression must be explicitly typed.
          // The TypeChecker evaluates the condition LHS with expectedBits=nullopt,
          // meaning a bare IntLit first atom defaults to i32 regardless of
          // condType. Using genFirstIntAtomOffPath for i==0 ensures the first
          // atom is always a RValueAtom or CastAtom whose type is unambiguous.
          if (i == 0) {
            a = genFirstIntAtomOffPath(rng, vars, targetType, cfg, excludeName);
          } else {
            a = genIntAtomOffPath(rng, vars, targetType, cfg, excludeName);
          }
        }
      } else if (TypeUtils::isFloat(targetType)) {
        if (onPath && sym) {
          a = genFloatAtomOnPath(rng, *sym, vars, targetType, cfg, excludeName);
        } else {
          a = genFloatAtomOffPath(rng, vars, targetType, cfg, excludeName);
        }
      } else if (TypeUtils::isPtr(targetType)) {
        a = genPtrAtom(rng, vars, targetType, excludeName);
      } else if (TypeUtils::isVec(targetType)) {
        // Vec target in genExprWithRequires — delegate to the same
        // logic as genExpr's vec branch.
        auto vecs = excluding(vars.vecsOf(targetType), excludeName);
        const auto &vt = std::get<VecType>(targetType->v);
        if (i == 0 && !vecs.empty()) {
          auto *v = pickOne(rng, vecs);
          if (onPath && sym) {
            std::uniform_int_distribution<int> slot(0, 1);
            if (slot(rng) == 0)
              a = rvalAtom(localLV(v->name));
            else
              a = opAtom(AtomOpKind::Mul, symCoef(sym->nextCoef(vt.elem)), localLV(v->name));
          } else {
            a = rvalAtom(localLV(v->name));
          }
        } else if (i > 0 && !vecs.empty()) {
          std::uniform_int_distribution<int> ts(0, 1);
          if (ts(rng) == 0) {
            auto *v = pickOne(rng, vecs);
            a = rvalAtom(localLV(v->name));
          } else {
            CoefAtom ca;
            if (TypeUtils::isFloat(vt.elem))
              ca.coef = FloatLit{1.0, {}};
            else
              ca.coef = IntLit{1, {}};
            a = Atom{std::move(ca), {}};
          }
        } else {
          CoefAtom ca;
          ca.coef = IntLit{0, {}};
          a = Atom{std::move(ca), {}};
        }
      } else {
        // Struct type — generate i32 concrete (struct assignments handled specially)
        a = genConcreteIntAtom(rng, makeI32());
      }
      atoms.push_back(std::move(a));
    }

    // A single-atom bare-RValueAtom RHS (`%v = %w;`) is reshaped so it
    // can't bulk-collapse under SCCP — except for a kPAllowPlainCopy
    // slice that survives verbatim: copy propagation is a distinct dataflow
    // shape the optimizer exercises, and forbidding it outright homogenises
    // every def into an arithmetic join. Self-assigns remain impossible
    // (excludeName filters the LHS from every pool). Skip for ptr/vec:
    // ptr is forced single-atom with `excludeName` already filtering
    // the pool; vec already mixes whole-vec copy with lane-wise tails
    // in its own dispatch.
    std::uniform_real_distribution<double> copyCoin(0.0, 1.0);
    if (atoms.size() == 1 && std::holds_alternative<RValueAtom>(atoms[0].v) &&
        (TypeUtils::isInt(targetType) || TypeUtils::isFloat(targetType)) &&
        copyCoin(rng) >= rysmith::hp::kPAllowPlainCopy) {
      if (cfg.maxAtoms > 1) {
        // On-path the appended tail is a cheap linear atom — a slot-
        // table roll would mostly mint a fresh sym (free variable) or a
        // nonlinear `sym * var` for what is purely an anti-copy reshape.
        std::optional<Atom> tail =
            (onPath && !cfg.condContext)
                ? genCheapLinearAtom(rng, vars, targetType, excludeName, true)
                : std::nullopt;
        if (tail) {
          atoms.push_back(std::move(*tail));
        } else {
          atoms.push_back(genOneAtomOfType(ctx, targetType));
        }
      } else {
        // In-place replacement must not be another bare read (that
        // would recreate a plain copy of a different var), so the cheap
        // chain is asked for a multiply / cast / load.
        std::optional<Atom> replacement =
            (onPath && !cfg.condContext)
                ? genCheapLinearAtom(rng, vars, targetType, excludeName, false)
                : std::nullopt;
        for (int attempt = 0; !replacement && attempt < 10; attempt++) {
          Atom cand = genOneAtomOfType(ctx, targetType);
          if (!std::holds_alternative<RValueAtom>(cand.v))
            replacement = std::move(cand);
        }
        if (replacement) {
          atoms[0] = std::move(*replacement);
        } else {
          CoefAtom ca;
          if (onPath && sym) {
            ca.coef = SymId{sym->nextCoef(targetType), {}};
          } else {
            if (TypeUtils::isFloat(targetType)) {
              ca.coef = FloatLit{1.0, {}};
            } else {
              ca.coef = IntLit{1, {}};
            }
          }
          atoms[0] = Atom{std::move(ca), {}};
        }
      }
    }

    rewriteIfAllTriviallyConstant(ctx, targetType, atoms);

    Expr expr;
    expr.first = std::move(atoms[0]);
    std::uniform_real_distribution<double> addOpCoin(0.0, 1.0);
    for (std::size_t i = 1; i < atoms.size(); i++) {
      AddOp op =
          TypeUtils::isFloat(targetType)
              ? AddOp::Plus
              : (addOpCoin(rng) < rysmith::hp::kPTailAddOpIsPlus ? AddOp::Plus : AddOp::Minus);
      expr.rest.push_back({op, std::move(atoms[i]), {}});
    }

    return {std::move(expr), std::move(reqs)};
  }

  // ---------------------------------------------------------------------------
  // genCond
  // ---------------------------------------------------------------------------

  Cond genCond(
      std::mt19937 &rng, SymCounter *sym, const VarCatalogue &vars, bool onPath,
      const ExprGenConfig &cfg
  ) {
    // Pick a random integer scalar type from available vars for the condition
    auto scalars = vars.allScalars();
    TypePtr condType = makeI32(); // default
    if (!scalars.empty()) {
      // Filter to int types only (can't compare floats with relational ops in RefractIR)
      std::vector<const VarEntry *> intScalars;
      for (auto *v: scalars)
        if (TypeUtils::isInt(v->type))
          intScalars.push_back(v);
      if (!intScalars.empty()) {
        auto *v = pickOne(rng, intScalars);
        condType = v->type;
      }
    }

    // Mark the arm expressions as condition context so their trivial-
    // shape replacements keep minting syms — conditions are the solver's
    // handles for driving the path; sym-free arms starve it of freedom and
    // inflate the UNSAT/retry rate.
    ExprGenConfig condCfg = cfg;
    condCfg.condContext = true;

    std::vector<Instr> lhsReqs, rhsReqs;
    auto [lhs, lreqs] = genExprWithRequires(rng, sym, vars, condType, onPath, condCfg);
    auto [rhs, rreqs] = genExprWithRequires(rng, sym, vars, condType, onPath, condCfg);
    // Note: condition requires are discarded here (condition can't have inline requires easily)
    // This is acceptable since the requires from div are an optional safety feature
    (void) lreqs;
    (void) rreqs;

    static const RelOp relOps[] = {RelOp::LT, RelOp::GT, RelOp::LE,
                                   RelOp::GE, RelOp::EQ, RelOp::NE};
    std::uniform_int_distribution<int> opPick(0, 5);

    return Cond{std::move(lhs), relOps[opPick(rng)], std::move(rhs), {}};
  }

  // ---------------------------------------------------------------------------
  // genBlockStmts
  // ---------------------------------------------------------------------------

  // Where one generated assignment writes, and the type its RHS must have.
  struct AssignTarget {
    LValue lhs;
    TypePtr type;
  };

  // Choose an assignment target inside `lhsVar`. Returns nullopt when the
  // variable offers no usable slot — a nested-array element, a struct whose
  // fields are all pointers, an unknown type — which the caller retries with a
  // fresh LHS pick. Pointer LHSs never come here: they build and push their own
  // assignment through emitPtrReassign.
  static std::optional<AssignTarget>
  pickAssignTarget(std::mt19937 &rng, const VarCatalogue &vars, const VarEntry &lhsVar) {
    if (TypeUtils::isScalar(lhsVar.type))
      return AssignTarget{localLV(lhsVar.name), lhsVar.type};

    if (std::holds_alternative<ArrayType>(lhsVar.type->v)) {
      const auto &at = std::get<ArrayType>(lhsVar.type->v);
      std::uniform_int_distribution<int64_t> idxd(0, (int64_t) at.size - 1);
      int64_t idx = idxd(rng);
      if (TypeUtils::isScalar(at.elem))
        return AssignTarget{arrayLV(lhsVar.name, idx), at.elem};
      if (std::holds_alternative<StructType>(at.elem->v)) {
        // Array-of-struct: pick a scalar field, generate %a[i].f = expr
        const std::string &ename = std::get<StructType>(at.elem->v).name.name;
        const StructDecl *sd = nullptr;
        for (const auto &decl: vars.structDecls)
          if (decl.name.name == ename) {
            sd = &decl;
            break;
          }
        if (!sd || sd->fields.empty())
          return std::nullopt;
        std::vector<const FieldDecl *> scalarFields;
        for (const auto &f: sd->fields)
          if (TypeUtils::isScalar(f.type))
            scalarFields.push_back(&f);
        if (scalarFields.empty())
          return std::nullopt;
        const FieldDecl *f = pickOne(rng, scalarFields);
        LValue lhs = arrayLV(lhsVar.name, idx);
        lhs.accesses.push_back(AccessField{f->name, {}});
        return AssignTarget{std::move(lhs), f->type};
      }
      return std::nullopt; // nested array or other, skip
    }

    if (std::holds_alternative<StructType>(lhsVar.type->v)) {
      const std::string &sname = lhsVar.structTypeName;
      const StructDecl *sd = nullptr;
      for (const auto &decl: vars.structDecls)
        if (decl.name.name == sname) {
          sd = &decl;
          break;
        }
      if (!sd || sd->fields.empty()) {
        // The struct's declaration is not in scope; fall back to a scalar.
        auto scalars = vars.allScalars();
        if (scalars.empty())
          return std::nullopt;
        auto *sv = pickOne(rng, scalars);
        return AssignTarget{localLV(sv->name), sv->type};
      }
      std::uniform_int_distribution<int> fpick(0, (int) sd->fields.size() - 1);
      const auto &f = sd->fields[fpick(rng)];
      if (TypeUtils::isScalar(f.type))
        return AssignTarget{structLV(lhsVar.name, f.name), f.type};
      if (std::holds_alternative<ArrayType>(f.type->v)) {
        // Struct-of-array field: pick an element, generate %t.f[i] = expr
        const auto &fat = std::get<ArrayType>(f.type->v);
        if (!TypeUtils::isScalar(fat.elem))
          return std::nullopt;
        std::uniform_int_distribution<int64_t> idxd2(0, (int64_t) fat.size - 1);
        LValue lhs = structLV(lhsVar.name, f.name);
        lhs.accesses.push_back(AccessIndex{Index{IntLit{idxd2(rng), {}}}, {}});
        return AssignTarget{std::move(lhs), fat.elem};
      }
      return std::nullopt; // ptr field or other, skip
    }

    if (TypeUtils::isVec(lhsVar.type)) {
      // Whole-vec only when another vec var of the same type can supply the
      // RHS — the typechecker rejects a scalar one. Excluding the LHS from
      // that pool keeps `%vec = %vec;` impossible, and when it leaves the pool
      // empty the lane write below takes over, whose RHS lives in scalar
      // territory.
      const auto &vt = std::get<VecType>(lhsVar.type->v);
      auto sameVecs = excluding(vars.vecsOf(lhsVar.type), lhsVar.name);
      bool canWholeVec = !sameVecs.empty();
      std::uniform_int_distribution<int> vslot(0, 99);
      if (canWholeVec && vslot(rng) >= rysmith::hp::kVecLaneWriteProb)
        return AssignTarget{localLV(lhsVar.name), lhsVar.type};
      // Lane write: %vec[i] = scalar_expr
      std::uniform_int_distribution<int64_t> ld(0, (int64_t) vt.size - 1);
      LValue lhs = localLV(lhsVar.name);
      lhs.accesses.push_back(AccessIndex{Index{IntLit{ld(rng), {}}}, {}});
      return AssignTarget{std::move(lhs), vt.elem};
    }

    return std::nullopt; // unknown type, skip
  }

  // Emit one pointer reassignment for `lhsVar` into `out`. A pointer LHS never
  // shares the generic RHS path: each shape here — an in-bounds arithmetic
  // step, an aggregate navigation chain, or a plain redirect — builds and
  // pushes its own assignment, so exactly one lands per call.
  static void emitPtrReassign(
      std::vector<Instr> &out, std::mt19937 &rng, const VarCatalogue &vars, const VarEntry &lhsVar,
      bool onPath, const ExprGenConfig &cfg
  ) {
    LValue lhs = localLV(lhsVar.name);

    // First try the pointer-arithmetic slot: an in-bounds element step off an
    // aggregate the LHS pointee lives in. genPtrArithRhs covers both
    // `ptrindex %ap, b ± d` (array element) and `ptrfield %sp, f ± d`
    // (consecutive same-type struct fields), each provably load-safe on any
    // path. It falls through (nullopt) when no such source is in scope.
    TypePtr lhsPtee = TypeUtils::pointee(lhsVar.type);
    std::uniform_real_distribution<double> ptrArithCoin(0.0, 1.0);
    bool emitted = false;
    // Pointer arithmetic is offered for any loadable pointee — scalar or
    // pointer (`ptr ptr T`). Aggregate/vector pointees are excluded: they
    // aren't loadable, so an element step has nothing valid to dereference.
    // The element-type matching downstream is by type equality, so pointer
    // pointees flow through unchanged.
    if (cfg.enablePtrArith && lhsPtee &&
        (TypeUtils::isScalar(lhsPtee) || TypeUtils::isPtr(lhsPtee)) &&
        ptrArithCoin(rng) < rysmith::hp::kPPtrArith) {
      if (!onPath) {
        // Off-path (unexecuted) blocks are never run and never symbolically
        // constrained, so pointer arithmetic here can be arbitrary — any
        // pointer plus any offset, no in-bounds reasoning, since the result is
        // never dereferenced or UB-checked. Reuse genPtrAtom for the base (same
        // definite-init safety as the default redirect) and append an unbounded
        // stride; this stresses pointer / alias codegen the in-bounds on-path
        // forms can't. Skip the stride on a `null` base, where `null ± k` would
        // be a pointless (and parse-noisy) shape.
        Expr rhs;
        rhs.first = genPtrAtom(rng, vars, lhsVar.type, lhsVar.name);
        const auto *ca = std::get_if<CoefAtom>(&rhs.first.v);
        if (!(ca && std::holds_alternative<NullLit>(ca->coef))) {
          std::uniform_int_distribution<int64_t> mag(1, rysmith::hp::kOffPathPtrStrideMax);
          bool minus = std::uniform_int_distribution<int>(0, 1)(rng) == 1;
          rhs.rest.push_back({minus ? AddOp::Minus : AddOp::Plus, coefAtom(intCoef(mag(rng))), {}});
        }
        out.push_back(Instr{AssignInstr{std::move(lhs), std::move(rhs), {}}});
        emitted = true;
      } else if (ptrArithCoin(rng) < rysmith::hp::kPPtrArithVarShare &&
                 tryEmitPtrVarArith(out, rng, vars, lhsPtee, lhsVar.name)) {
        // On-path: prefer the direct ptr-var arithmetic shape (`%p2 = %p1 ± d`
        // on a pointer already anchored to an array element in this block).
        // tryEmitPtrVarArith builds and pushes its own assignment, so its
        // branch doesn't consume `lhs`.
        emitted = true;
      } else if (auto rhs = genPtrArithRhs(rng, vars, lhsPtee, lhsVar.name)) {
        // On-path fallback: the combined ptrindex/ptrfield expression.
        out.push_back(Instr{AssignInstr{std::move(lhs), std::move(*rhs), {}}});
        emitted = true;
      }
    }

    // Aggregate-pointer navigation: when the arithmetic slot declined, a
    // `ptr F` reassign may instead navigate an aggregate to its loadable leaf
    // via a chain of staged ptrindex/ptrfield steps. Plain navigation (no
    // arithmetic), so it runs regardless of --no-ptrarith, and is rolled after
    // the arith slot to leave that stream intact.
    if (!emitted && lhsPtee && ptrArithCoin(rng) < rysmith::hp::kPNavChain &&
        tryEmitNavChain(out, rng, vars, lhsPtee, lhsVar.name)) {
      emitted = true;
    }

    if (!emitted) {
      // Default: single-atom ptr redirect via `genPtrAtom`. Exclude the LHS ptr
      // from its own RHS pool — `%p = %p;` is a no-op that SCCP folds, and the
      // ptr-copy slot inside `genPtrAtom` would otherwise include `%p`. Bare
      // `%p = %q;` (q != p) is intentionally permitted: aliasing through a
      // plain ptr copy is a useful, semantically distinct shape, and adding
      // `+ 1` to an arbitrary scalar-ptr source would step past its
      // single-element `addr %scalar` object and make any later `load %p` UB.
      Expr rhs = simpleExpr(genPtrAtom(rng, vars, lhsVar.type, lhsVar.name));
      out.push_back(Instr{AssignInstr{std::move(lhs), std::move(rhs), {}}});
    }
  }

  std::vector<Instr> genBlockStmts(
      std::mt19937 &rng, SymCounter *sym, const VarCatalogue &vars, int nStmts, bool onPath,
      const ExprGenConfig &cfg
  ) {
    std::vector<Instr> result;
    std::uniform_real_distribution<double> prob(0.0, 1.0);

    // Collect mutable (non-ptr) vars for assignment targets
    // We assign to scalars and array/struct elements.
    // Function parameters are immutable per spec §3.5.2 — exclude.
    std::vector<const VarEntry *> allVars;
    for (const auto &v: vars.vars)
      if (!v.isParam)
        allVars.push_back(&v);

    // Collect ptr vars for store operations
    std::vector<const VarEntry *> ptrVars;
    for (const auto &v: vars.vars)
      if (TypeUtils::isPtr(v.type))
        ptrVars.push_back(&v);

    // Splice a single StoreInstr (plus its safety requires) into `result`.
    // Returns false when no usable ptr exists or the chosen pointee type
    // is one we don't store through (aggregate pointees aren't supported
    // by the RefractIR `store` form). The Bernoulli chain below stops rolling
    // on the first false to avoid burning the rng on no-ops.
    auto tryEmitStore = [&]() -> bool {
      if (ptrVars.empty())
        return false;
      auto *pv = pickOne(rng, ptrVars);
      TypePtr ptee = TypeUtils::pointee(pv->type);
      Expr ptrExpr = simpleExpr(rvalAtom(localLV(pv->name)));
      Expr valExpr;
      if (TypeUtils::isInt(ptee) || TypeUtils::isFloat(ptee)) {
        auto [ve, reqs] = genExprWithRequires(rng, sym, vars, ptee, onPath, cfg);
        if (onPath)
          for (auto &req: reqs)
            result.push_back(std::move(req));
        valExpr = std::move(ve);
      } else if (TypeUtils::isPtr(ptee)) {
        // Store a pointer value through a `ptr ptr T` slot. genPtrAtom
        // yields a single valid pointer atom (addr / ptr-copy / load /
        // ptrindex / ptrfield); the solver models the stored provenance, so
        // a later load through the slot reads the right object on-path.
        valExpr = simpleExpr(genPtrAtom(rng, vars, ptee));
      } else {
        // Aggregate pointee — no whole-aggregate store (SPEC: navigate to a
        // scalar/pointer leaf first).
        return false;
      }
      StoreInstr st;
      st.ptr = std::move(ptrExpr);
      st.val = std::move(valExpr);
      result.push_back(Instr{std::move(st)});
      return true;
    };

    for (int s = 0; s < nStmts; s++) {
      // Bernoulli chain of StoreInstrs spliced before this AssignInstr.
      // Each successful roll emits one store and rolls again; the chain
      // stops on the first failed roll. Stores do not consume the
      // `nStmts` budget — it tracks AssignInstrs only, so `--n-stmts N`
      // names exactly what it produces. With the default
      // `kPStoreBeforeAssign = 0.25` the expected chain length is
      // p / (1 - p) ≈ 0.33 stores per assignment.
      while (prob(rng) < rysmith::hp::kPStoreBeforeAssign) {
        if (!tryEmitStore())
          break;
      }

      // Try up to `kAssignTargetMaxAttempts` LHS picks before giving up.
      // Aggregate LHSs occasionally land on a slot that can't be built
      // (array-of-nested-array, struct with only ptr fields, etc.); each
      // such pick is a `continue` to the next attempt rather than a
      // `continue` to the next `nStmts` slot, so `--n-stmts N` actually
      // produces N AssignInstrs per block.
      bool assignEmitted = false;
      for (int attempt = 0; attempt < rysmith::hp::kAssignTargetMaxAttempts && !assignEmitted;
           attempt++) {
        if (allVars.empty())
          break;
        auto *lhsVar = pickOne(rng, allVars);

        // A pointer LHS emits its own assignment and never reaches the shared
        // RHS path below.
        if (TypeUtils::isPtr(lhsVar->type)) {
          emitPtrReassign(result, rng, vars, *lhsVar, onPath, cfg);
          assignEmitted = true;
          continue;
        }

        auto target = pickAssignTarget(rng, vars, *lhsVar);
        if (!target)
          continue; // this var offers no usable slot; retry with another LHS

        // The LHS root name (e.g. `%v0`, `%a` in `%a[i].f`, `%vec` in
        // `%vec` whole-vec or `%vec[i]` lane write) is forbidden as an
        // RValue anywhere in the RHS. For aggregate LHS (`%a[i]`, `%t.f`),
        // excluding the root has no practical effect (the root is not a
        // scalar and scalar pickers never see it), but it costs nothing.
        auto [rhs, reqs] =
            genExprWithRequires(rng, sym, vars, target->type, onPath, cfg, lhsVar->name);

        // Insert safety requires before assignment (on-path only — off-path
        // blocks are never executed at the solved inputs).
        if (onPath) {
          for (auto &req: reqs)
            result.push_back(std::move(req));
        }

        result.push_back(Instr{AssignInstr{std::move(target->lhs), std::move(rhs), {}}});
        assignEmitted = true;
      } // end of LHS-pick retry loop
    }

    return result;
  }

} // namespace refractir::reify
