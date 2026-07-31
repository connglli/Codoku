#include "reify/twin_probe.hpp"

#include <sstream>
#include <unordered_set>

#include "analysis/type_utils.hpp"
#include "ast/sir_printer.hpp"
#include "frontend/lexer.hpp"
#include "frontend/parser.hpp"
#include "reify/common.hpp"
#include "reify/type_gen.hpp"

namespace refractir::reify {

  namespace {

    constexpr const char *kProbeFun = "@__twin_probe";
    constexpr const char *kProbeEntry = "^__probe";
    constexpr const char *kExitPrefix = "^__x_";

    // FNV-1a over the executed block labels. Only ever compared against
    // another pathId, so any stable, well-mixing hash would do.
    std::uint64_t hashStep(std::uint64_t h, const std::string &s) {
      for (unsigned char c: s) {
        h ^= c;
        h *= 1099511628211ull;
      }
      return h ^ '\n';
    }

    // The AST is move-only (SelectAtom holds a unique_ptr), so a region's
    // blocks cannot be copied out of the host program — and the host must
    // survive intact. Round-tripping through the canonical printer and
    // parser yields a duplicate we are free to dismantle. Once per region,
    // not once per probe.
    Program duplicate(const Program &prog) {
      std::ostringstream oss;
      SIRPrinter(oss).print(prog);
      std::string txt = oss.str();
      Lexer lx(txt);
      Parser ps(lx.lexAll());
      return ps.parseProgram();
    }

    // The `ret` value of a landing block. Never observed — the effect is
    // read from the state profile — but it must typecheck against the
    // harness's return type.
    Expr zeroOf(const TypePtr &ty) {
      if (isPtrType(ty))
        return Expr{Atom{CoefAtom{Coef{NullLit{}}, {}}, {}}, {}, {}};
      if (TypeUtils::getFloatBitWidth(ty))
        return Expr{Atom{CoefAtom{Coef{FloatLit{0.0, {}}}, {}}, {}}, {}, {}};
      return Expr{Atom{CoefAtom{Coef{IntLit{0, {}}}, {}}, {}}, {}, {}};
    }

    // Point every out-of-region target at its landing block, recording the
    // targets seen. In-region targets and `ret` are left alone: a region
    // may legitimately end by returning.
    void redirectTerm(
        Terminator &t, const std::unordered_set<std::string> &inRegion,
        std::unordered_set<std::string> &exits
    ) {
      auto br = std::get_if<BrTerm>(&t);
      if (!br)
        return;
      // `dest` carries the target of an unconditional branch and is empty
      // on a conditional one, where then/else carry them instead.
      auto fix = [&](BlockLabel &l) {
        if (l.name.empty() || inRegion.count(l.name))
          return;
        exits.insert(l.name);
        l.name = kExitPrefix + l.name.substr(1); // drop the original '^'
      };
      fix(br->dest);
      fix(br->thenLabel);
      fix(br->elseLabel);
    }

  } // namespace

  RegionProbe::RegionProbe(
      const Program &host, const std::string &funcName,
      const std::vector<std::string> &regionLabels, const std::vector<MiniRoot> &roots
  ) : nRoots_(roots.size()) {
    if (regionLabels.empty() || roots.empty())
      return;

    Program dup;
    try {
      dup = duplicate(host);
    } catch (const std::exception &) {
      return;
    }
    FunDecl *src = nullptr;
    for (auto &f: dup.funs)
      if (f.name.name == funcName) {
        src = &f;
        break;
      }
    if (!src)
      return;

    std::unordered_set<std::string> inRegion(regionLabels.begin(), regionLabels.end());

    harness_.structs = host.structs;
    harness_.intrinsics = host.intrinsics;

    FunDecl f;
    f.name = GlobalId{kProbeFun, {}};
    f.retType = src->retType;
    declareRoots(roots, structMap(host), f.lets);

    // Locals the region touches but the entry state does not cover (still
    // `undef` when the region starts) have no root, yet the region's
    // instructions name them. Redeclare them exactly as the home function
    // did, uninitialized — so reading one before it is written is UB here
    // for the same reason it is UB there.
    {
      std::unordered_set<std::string> declared;
      for (const auto &r: roots)
        declared.insert(r.name);
      auto addUndef = [&](const std::string &nm, const TypePtr &ty, bool mut) {
        if (declared.count(nm))
          return;
        LetDecl d;
        d.isMutable = mut;
        d.name = LocalId{nm, {}};
        d.type = ty;
        d.init = InitVal{InitVal::Kind::Undef, IntLit{}, {}};
        f.lets.push_back(std::move(d));
      };
      for (const auto &p: src->params)
        addUndef(p.name.name, p.type, /*mut=*/false);
      for (const auto &l: src->lets)
        addUndef(l.name.name, l.type, l.isMutable);
    }

    Block entry;
    entry.label = BlockLabel{kProbeEntry, {}};
    entry.instrs = ptrInitInstrs(roots);
    BrTerm toRegion;
    toRegion.dest = BlockLabel{regionLabels.front(), {}};
    toRegion.thenLabel = toRegion.dest;
    toRegion.elseLabel = toRegion.dest;
    toRegion.isConditional = false;
    entry.term = Terminator{std::move(toRegion)};
    f.blocks.push_back(std::move(entry));

    std::unordered_set<std::string> exits;
    for (const auto &lbl: regionLabels) {
      Block *b = nullptr;
      for (auto &cand: src->blocks)
        if (cand.label.name == lbl) {
          b = &cand;
          break;
        }
      if (!b)
        return; // region label the function does not have
      redirectTerm(b->term, inRegion, exits);
      f.blocks.push_back(std::move(*b));
    }
    for (const auto &e: exits) {
      Block land;
      land.label = BlockLabel{kExitPrefix + e.substr(1), {}};
      land.term = Terminator{RetTerm{zeroOf(f.retType), {}}};
      f.blocks.push_back(std::move(land));
    }
    harness_.funs.push_back(std::move(f));

    if (!runAnalysisPasses(harness_, /*verbose=*/false))
      return; // a region shape the checkers reject in isolation

    structs_ = structMap(harness_);
    valid_ = true;
  }

  ProbeResult RegionProbe::run(const std::vector<MiniRoot> &roots) {
    ProbeResult out;
    if (!valid_ || roots.size() != nRoots_)
      return out;

    // Re-seed: only the root initializers differ between probes, and
    // declareRoots laid them down first, in root order.
    auto &lets = harness_.funs.front().lets;
    for (std::size_t i = 0; i < nRoots_; ++i)
      lets[i].init = stateToInit(roots[i].init, roots[i].type, structs_);

    // Any UB, or a loop the probed state drives past the cap, leaves `ok`
    // false: either way the state is not usable.
    StateProfile pf;
    try {
      pf =
          profileProgram(harness_, kProbeFun, {}, StateGranularity::Ppp, rytwin::hp::kProbeStepCap);
    } catch (const std::exception &) {
      return out;
    }
    if (pf.trace.empty())
      return out;

    // Block entries (instr == -1) give the path; the last point gives the
    // exit state. The exit is hashed in too: one block that branches to
    // two different outside targets runs the same block sequence either
    // way, but those are not the same path.
    std::uint64_t h = 1469598103934665603ull;
    for (const auto &p: pf.trace) {
      if (p.instr != -1 || p.block == kProbeEntry)
        continue;
      if (p.block.rfind(kExitPrefix, 0) == 0) {
        out.exitLabel = "^" + p.block.substr(std::string(kExitPrefix).size());
        break;
      }
      h = hashStep(h, p.block);
    }
    out.pathId = hashStep(h, out.exitLabel);
    out.effect = pf.trace.back().vars;
    out.ok = true;
    return out;
  }

} // namespace refractir::reify
