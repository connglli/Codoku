#include "reify/twin_gen.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "ast/sir_printer.hpp"
#include "frontend/lexer.hpp"
#include "frontend/parser.hpp"
#include "reify/common.hpp"
#include "reify/expr_gen.hpp"
#include "reify/intrinsic_whitelist.hpp"
#include "reify/twin_mini.hpp"
#include "reify/var_catalogue.hpp"

namespace refractir::reify {

  namespace {

    constexpr const char *kMiniFun = "@__twin_mini";

    TypePtr makeI32() {
      return std::make_shared<Type>(Type{IntType{IntType::Kind::I32, {}, {}}, {}});
    }

    // `<leaf> + %?sym` — the always-solvable additive correction.
    Expr correctionExpr(LValue leaf, const std::string &symName) {
      Expr e = rvalExpr(std::move(leaf));
      Expr::Tail t;
      t.op = AddOp::Plus;
      t.atom = Atom{CoefAtom{Coef{LocalOrSymId{SymId{symName, {}}}}, {}}, {}};
      e.rest.push_back(std::move(t));
      return e;
    }

    // One generation attempt. Throws on internal errors (parse/analysis);
    // returns nullopt on UNSAT / verification mismatch.
    std::optional<TwinGenResult> attempt(
        const Program &prog, const std::vector<MiniRoot> &roots, std::mt19937 &rng,
        const SymbolicExecutor::SolverFactory &solverFactory, const TwinGenConfig &cfg
    ) {
      const StructMap structs = structMap(prog);

      // -- 1. Mini-program skeleton: every root becomes a let initialized
      // to its entry value, plus the catalogue the statement generator
      // draws its operands from.
      Program mini;
      mini.structs = prog.structs;
      FunDecl f;
      f.name = GlobalId{kMiniFun, {}};
      f.retType = makeI32();

      declareRoots(roots, structs, f.lets);

      VarCatalogue cat;
      cat.structDecls = prog.structs;
      for (const auto &r: roots) {
        VarEntry ve;
        ve.name = r.name;
        ve.type = r.type;
        ve.isParam = r.isParam; // generator treats it as read-only
        if (auto st = std::get_if<StructType>(&r.type->v))
          ve.structTypeName = st->name.name;
        cat.vars.push_back(std::move(ve));
      }

      // -- 2. Random body.
      std::set<IntrinsicUseKey> used;
      ExprGenConfig ecfg;
      ecfg.usedIntrinsics = &used;
      SymCounter sym;
      Block entry;
      entry.label = BlockLabel{"^entry", {}};
      // Pointer cells are declared null; set their real entry provenance
      // before any generated statement can read them.
      entry.instrs = ptrInitInstrs(roots);
      auto body = genBlockStmts(rng, &sym, cat, cfg.nStmts, /*onPath=*/true, ecfg);
      const bool bodyStores = std::any_of(body.begin(), body.end(), [](const Instr &i) {
        return std::holds_alternative<StoreInstr>(i);
      });
      entry.instrs.insert(
          entry.instrs.end(), std::make_move_iterator(body.begin()),
          std::make_move_iterator(body.end())
      );

      // Roots the random body wrote: their leaves need corrections too,
      // else the equality requires would demand the random RHS hit the
      // target exactly (almost always UNSAT).
      std::unordered_set<std::string> written;
      for (const auto &ins: entry.instrs)
        if (auto ai = std::get_if<AssignInstr>(&ins))
          written.insert(ai->lhs.base.name);

      // -- 3. Additive corrections. One fresh unconstrained symbol per
      // leaf that changed (s' != s) or belongs to a written root: BV
      // addition is invertible, so an integer correction always reaches
      // the target; float corrections may miss under rounding and are
      // retried.
      f.syms = sym.makeDecls();
      int nCorr = 0;
      for (const auto &r: roots) {
        std::vector<StateLeaf> initLeaves, targetLeaves;
        bool hp = false, hu = false;
        enumStateLeaves(r.init, initLeaves, hp, hu);
        enumStateLeaves(r.target, targetLeaves, hp, hu);
        if (hu || initLeaves.size() != targetLeaves.size())
          return std::nullopt; // shape drift — cannot model this root
        // A store in the generated body can write through any pointer, so
        // every mutable root counts as written.
        bool wholeRoot = written.count(r.name) > 0 || (bodyStores && !r.isParam);
        for (std::size_t i = 0; i < initLeaves.size(); ++i) {
          if (initLeaves[i].val.kind == StateValue::Kind::Ptr)
            continue; // pointer cells are reproduced by the final fixups
          bool diff = !bitExactEq(initLeaves[i].val, targetLeaves[i].val);
          if (!diff && !wholeRoot)
            continue;
          if (r.isParam)
            return std::nullopt; // an immutable root cannot change value
          std::string sn = "%?twc" + std::to_string(nCorr++);
          SymDecl sd;
          sd.name = SymId{sn, {}};
          sd.kind = SymKind::Value;
          sd.type = leafType(targetLeaves[i].val);
          f.syms.push_back(std::move(sd));
          LValue lv = leafLV(r.name, targetLeaves[i].path);
          entry.instrs.push_back(Instr{AssignInstr{lv, correctionExpr(lv, sn), {}}});
        }
      }

      // -- 3b. Reproduce every pointer cell's exit provenance. Whatever
      // the generated body did to a pointer (reassigned it, left it), the
      // final fixup lands it on the captured s' target.
      {
        auto fin = ptrFinalInstrs(roots);
        entry.instrs.insert(
            entry.instrs.end(), std::make_move_iterator(fin.begin()),
            std::make_move_iterator(fin.end())
        );
      }

      // -- 4. Equality requires pinning the final state to s', one per
      // leaf, against same-typed immutable lets (bare literals in a Cond
      // default to 32 bits). These are solver scaffolding: recorded and
      // stripped from the extracted body.
      int nEq = 0;
      for (const auto &r: roots) {
        std::vector<StateLeaf> targetLeaves;
        bool hp = false, hu = false;
        enumStateLeaves(r.target, targetLeaves, hp, hu);
        for (const auto &leaf: targetLeaves) {
          if (leaf.val.kind == StateValue::Kind::Ptr)
            continue;
          std::string tn = "%__t" + std::to_string(nEq++);
          LetDecl d;
          d.isMutable = false;
          d.name = LocalId{tn, {}};
          d.type = leafType(leaf.val);
          InitVal iv;
          if (leaf.val.kind == StateValue::Kind::Float) {
            iv.kind = InitVal::Kind::Float;
            iv.value = FloatLit{leaf.val.floatVal, {}};
          } else {
            iv.kind = InitVal::Kind::Int;
            iv.value = IntLit{leaf.val.intVal, {}};
          }
          d.init = iv;
          f.lets.push_back(std::move(d));

          Cond c;
          c.lhs = rvalExpr(leafLV(r.name, leaf.path));
          c.op = RelOp::EQ;
          c.rhs = rvalExpr(leafLV(tn, {}));
          entry.instrs.push_back(Instr{RequireInstr{std::move(c), {}, {}}});
        }
      }

      entry.term =
          Terminator{RetTerm{Expr{Atom{CoefAtom{Coef{IntLit{0, {}}}, {}}, {}}, {}, {}}, {}}};
      f.blocks.push_back(std::move(entry));
      mini.funs.push_back(std::move(f));
      appendUsedIntrinsicDecls(used, mini.intrinsics);

      if (!runAnalysisPasses(mini, /*verbose=*/false))
        return std::nullopt; // a generated shape the checkers reject

      // -- 5. Solve the single-block path.
      SymbolicExecutor::Config scfg;
      scfg.timeout_ms = cfg.timeoutMs;
      scfg.seed = rng();
      SymbolicExecutor ex(mini, scfg, solverFactory);
      auto res = ex.solve(kMiniFun, {"^entry"});
      if (!res.sat)
        return std::nullopt;

      // -- 6. Concretize through the canonical printer (model substitution
      // + bit-exact float formatting) and re-parse.
      std::ostringstream oss;
      SIRPrinter printer(oss, res.model, res.vecModel);
      printer.print(mini);
      std::string txt = oss.str();
      Lexer lx(txt);
      Parser ps(lx.lexAll());
      Program conc = ps.parseProgram();
      if (!runAnalysisPasses(conc, /*verbose=*/false))
        return std::nullopt;

      // -- 7. Bit-exact verification: run the concrete mini-program and
      // compare the final state against s'. The solver's FP equality is
      // IEEE (== conflates +0.0 / -0.0); this re-run is the arbiter.
      StateProfile pf = profileProgram(conc, kMiniFun, {}, StateGranularity::Ppp);
      if (pf.trace.empty())
        return std::nullopt;
      const StatePoint &fin = pf.trace.back();
      for (const auto &r: roots) {
        const StateValue *got = nullptr;
        for (const auto &[nm, val]: fin.vars)
          if (nm == r.name) {
            got = &val;
            break;
          }
        if (!got || !bitExactEq(*got, r.target))
          return std::nullopt;
      }

      // -- 8. Extract the body, stripping the trailing equality requires.
      for (auto &fn: conc.funs)
        if (fn.name.name == kMiniFun) {
          auto instrs = std::move(fn.blocks.front().instrs);
          instrs.resize(instrs.size() - nEq);
          TwinGenResult out;
          out.instrs = std::move(instrs);
          out.intrinsics = std::move(conc.intrinsics);
          return out;
        }
      return std::nullopt;
    }

  } // namespace

  std::optional<TwinGenResult> generateTwin(
      const Program &prog, const std::vector<MiniRoot> &roots, std::mt19937 &rng,
      const SymbolicExecutor::SolverFactory &solverFactory, const TwinGenConfig &cfg
  ) {
    if (roots.empty())
      return std::nullopt;
    for (int i = 0; i < cfg.retries; ++i) {
      try {
        if (auto r = attempt(prog, roots, rng, solverFactory, cfg))
          return r;
      } catch (const std::exception &) {
        // A UB trap in verification or a parse/analysis hiccup: this
        // attempt is void, the next one draws fresh statements.
      }
    }
    return std::nullopt;
  }

} // namespace refractir::reify
