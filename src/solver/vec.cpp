#include <optional>
#include <stdexcept>
#include "analysis/type_utils.hpp"
#include "internal.hpp"
#include "solver/solver.hpp"

namespace refractir {

  // Wrap per-lane symbolic values into a Vec-kind SymbolicValue. File-local so
  // both evalVecExpr / evalVecExprAtom and the per-kind helpers can share it.
  static SymbolicValue buildVec(std::vector<SymbolicValue> lanes) {
    SymbolicValue r;
    r.kind = SymbolicValue::Kind::Vec;
    r.arrayVal = std::move(lanes);
    return r;
  }

  // evalVecExpr — evaluate an Expr whose value is a vector,
  // returning a SymbolicValue with Kind::Vec. Single-atom Expr only
  // (chains involving vectors aren't exercised by our v0.2.1 tests; the
  // typechecker already permits them, so this is the natural follow-up
  // when those patterns appear in user code).
  SymbolicValue SymbolicExecutor::evalVecExpr(
      const Expr &e, const VecType &vt, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {

    auto laneSort = getSort(vt.elem, solver);

    // Fast path: chained +/- on vector lanes. We resolve each atom to its
    // N per-lane terms by lowering as a single-atom Expr — the dispatcher
    // below handles the supported atom kinds.
    if (!e.rest.empty()) {
      bool fpLane = solver.is_fp_sort(laneSort);
      // Atom isn't copy-assignable (SelectAtom holds unique_ptr), so we
      // call into the single-atom dispatcher in place via a tiny helper
      // that reuses the existing branch by visiting the Atom variant.
      auto evalAtomToLanes = [&](const Atom &a, auto &dispatch) -> std::vector<smt::Term> {
        SymbolicValue v = dispatch(a);
        std::vector<smt::Term> out(vt.size);
        if (v.kind == SymbolicValue::Kind::Vec) {
          for (std::uint64_t k = 0; k < vt.size; ++k)
            out[k] = v.arrayVal[k].term;
        } else {
          for (std::uint64_t k = 0; k < vt.size; ++k)
            out[k] = v.term;
        }
        return out;
      };
      // Recurse into evalVecExprAtom (declared below) for each atom in the
      // chain. We dispatch by constructing a one-atom view through the
      // overloaded helper.
      auto dispatchAtom = [&](const Atom &a) -> SymbolicValue {
        return evalVecExprAtom(a, vt, solver, store, pc, ub);
      };
      auto rmRNE = fpLane ? solver.make_rm_value(smt::RoundingMode::RNE) : smt::Term();
      std::vector<smt::Term> acc = evalAtomToLanes(e.first, dispatchAtom);
      for (const auto &term: e.rest) {
        auto next = evalAtomToLanes(term.atom, dispatchAtom);
        for (std::uint64_t k = 0; k < vt.size; ++k) {
          smt::Kind opK;
          if (fpLane)
            opK = term.op == AddOp::Plus ? smt::Kind::FP_ADD : smt::Kind::FP_SUB;
          else {
            // Per-lane signed BV add/sub overflow is UB
            // (rule 1 / 2 lifted to lanes).
            auto ovK =
                term.op == AddOp::Plus ? smt::Kind::BV_SADD_OVERFLOW : smt::Kind::BV_SSUB_OVERFLOW;
            auto ov = solver.make_term(ovK, {acc[k], next[k]});
            ub.push_back(solver.make_term(smt::Kind::NOT, {ov}));
            opK = term.op == AddOp::Plus ? smt::Kind::BV_ADD : smt::Kind::BV_SUB;
          }
          acc[k] = fpLane ? solver.make_term(opK, {rmRNE, acc[k], next[k]})
                          : solver.make_term(opK, {acc[k], next[k]});
          // Per-lane FP result must be finite (rule 6/7 lifted).
          if (fpLane)
            assertFPFinite(acc[k], solver, ub);
        }
      }
      std::vector<SymbolicValue> lanes;
      lanes.reserve(vt.size);
      for (std::uint64_t k = 0; k < vt.size; ++k)
        lanes.emplace_back(SymbolicValue::Kind::Int, acc[k], solver.make_true());
      return buildVec(std::move(lanes));
    }

    return evalVecExprAtom(e.first, vt, solver, store, pc, ub);
  }

  SymbolicValue SymbolicExecutor::evalVecExprAtom(
      const Atom &a, const VecType &vt, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {
    return std::visit(
        [&](auto &&arg) -> SymbolicValue {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, CoefAtom>) {
            // Vector sym or local — look up the store entry.
            if (auto id = std::get_if<LocalOrSymId>(&arg.coef)) {
              std::string nm = std::visit([](auto &&x) { return x.name; }, *id);
              auto it = store.find(nm);
              if (it != store.end() && it->second.kind == SymbolicValue::Kind::Vec)
                return it->second;
            }
            // Literal broadcast (rare; vector init handled elsewhere).
            smt::Term bv = evalCoef(arg.coef, solver, store, getSort(vt.elem, solver));
            std::vector<SymbolicValue> lanes;
            lanes.reserve(vt.size);
            for (std::uint64_t k = 0; k < vt.size; ++k)
              lanes.emplace_back(SymbolicValue::Kind::Int, bv, solver.make_true());
            return buildVec(std::move(lanes));
          } else if constexpr (std::is_same_v<T, RValueAtom>) {
            return evalLValue(arg.rval, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, CmpAtom>) {
            return evalVecCmpAtom(arg, vt, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, SelectAtom>) {
            return evalVecSelectAtom(arg, vt, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, OpAtom>) {
            return evalVecOpAtom(arg, vt, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, CastAtom>) {
            return evalVecCastAtom(arg, vt, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, UnaryAtom>) {
            return evalVecUnaryAtom(arg, vt, solver, store, pc, ub);
          } else {
            throw std::runtime_error("Solver: this vector RHS atom kind isn't yet lowered");
          }
        },
        a.v
    );
  }

  SymbolicValue SymbolicExecutor::evalVecCmpAtom(
      const CmpAtom &arg, const VecType &vt, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {

    // Per-lane comparison → <N> i1.
    // Infer operand lane sort from whichever side is a vec RValue.
    auto getOperandLaneSort = [&]() -> std::optional<smt::Sort> {
      auto getValSort = [&](const SelectVal &sv) -> std::optional<smt::Sort> {
        if (auto rv = std::get_if<RValue>(&sv)) {
          auto val = evalLValue(*rv, solver, store, pc, ub);
          if (val.kind == SymbolicValue::Kind::Vec && !val.arrayVal.empty())
            return solver.get_sort(val.arrayVal[0].term);
        }
        if (auto cf = std::get_if<Coef>(&sv)) {
          if (auto id = std::get_if<LocalOrSymId>(cf)) {
            std::string nm = std::visit([](auto &&x) { return x.name; }, *id);
            auto it = store.find(nm);
            if (it != store.end() && it->second.kind == SymbolicValue::Kind::Vec &&
                !it->second.arrayVal.empty())
              return solver.get_sort(it->second.arrayVal[0].term);
          }
        }
        return std::nullopt;
      };
      if (auto s = getValSort(arg.lhs))
        return s;
      if (auto s = getValSort(arg.rhs))
        return s;
      return std::nullopt;
    };
    auto opLaneSort = getOperandLaneSort();
    auto getVecOperand = [&](const SelectVal &sv) -> SymbolicValue {
      if (auto rv = std::get_if<RValue>(&sv))
        return evalLValue(*rv, solver, store, pc, ub);
      if (auto cf = std::get_if<Coef>(&sv)) {
        if (auto id = std::get_if<LocalOrSymId>(cf)) {
          std::string nm = std::visit([](auto &&x) { return x.name; }, *id);
          auto it = store.find(nm);
          if (it != store.end() && it->second.kind == SymbolicValue::Kind::Vec)
            return it->second;
        }
        smt::Term bv = evalCoef(*cf, solver, store, opLaneSort);
        std::vector<SymbolicValue> lanes;
        lanes.reserve(vt.size);
        for (std::uint64_t k = 0; k < vt.size; ++k)
          lanes.emplace_back(SymbolicValue::Kind::Int, bv, solver.make_true());
        return buildVec(std::move(lanes));
      }
      throw std::runtime_error("Vec cmp: unsupported SelectVal");
    };
    auto lhsV = getVecOperand(arg.lhs);
    auto rhsV = getVecOperand(arg.rhs);
    std::vector<SymbolicValue> lanes;
    lanes.reserve(vt.size);
    smt::Sort i1 = solver.make_bv_sort(1);
    smt::Term one = solver.make_bv_value(i1, "1", 10);
    smt::Term zero = solver.make_bv_value(i1, "0", 10);
    // Detect FP lanes — BV_SLT/etc only work on BV sorts; for FP
    // operands we need FP_LT/FP_LEQ/FP_GT/FP_GEQ. The CmpAtom's
    // result type is i1 (so `vt.elem` is i1 here, an integer),
    // so we have to probe the *operand* sort to pick the right
    // SMT op. EQUAL/DISTINCT are polymorphic across both
    // theories.
    bool fpLane =
        !lhsV.arrayVal.empty() && solver.is_fp_sort(solver.get_sort(lhsV.arrayVal[0].term));
    for (std::uint64_t k = 0; k < vt.size; ++k) {
      const auto &l = lhsV.arrayVal[k].term;
      const auto &r = rhsV.arrayVal[k].term;
      smt::Kind opKind = relOpToSmtKind(arg.op, fpLane);
      smt::Term cond = solver.make_term(opKind, {l, r});
      smt::Term laneBit = solver.make_term(smt::Kind::ITE, {cond, one, zero});
      lanes.emplace_back(SymbolicValue::Kind::Int, laneBit, solver.make_true());
    }
    return buildVec(std::move(lanes));

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalVecSelectAtom(
      const SelectAtom &arg, const VecType &vt, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {

    // Mask form (cond form is scalar — wouldn't yield a vector).
    if (!arg.maskExpr)
      throw std::runtime_error("Vec SelectAtom: expected mask form");
    // Mask is a vector value: either an lvalue (RValueAtom) or a
    // bare sym/local reference (CoefAtom holding LocalOrSymId).
    SymbolicValue maskV;
    if (auto maskRv = std::get_if<RValueAtom>(&arg.maskExpr->first.v)) {
      maskV = evalLValue(maskRv->rval, solver, store, pc, ub);
    } else if (auto maskCf = std::get_if<CoefAtom>(&arg.maskExpr->first.v)) {
      auto id = std::get_if<LocalOrSymId>(&maskCf->coef);
      if (!id)
        throw std::runtime_error("Vec SelectAtom: mask coef must be local/sym identifier");
      std::string nm = std::visit([](auto &&x) { return x.name; }, *id);
      auto it = store.find(nm);
      if (it == store.end() || it->second.kind != SymbolicValue::Kind::Vec)
        throw std::runtime_error("Vec SelectAtom: mask must resolve to a vector");
      maskV = it->second;
    } else {
      throw std::runtime_error("Vec SelectAtom: mask must be a vector lvalue or identifier");
    }
    auto loadArm = [&](const SelectVal &sv) -> SymbolicValue {
      if (auto rv = std::get_if<RValue>(&sv))
        return evalLValue(*rv, solver, store, pc, ub);
      if (auto cf = std::get_if<Coef>(&sv)) {
        if (auto id = std::get_if<LocalOrSymId>(cf)) {
          std::string nm = std::visit([](auto &&x) { return x.name; }, *id);
          auto it = store.find(nm);
          if (it != store.end() && it->second.kind == SymbolicValue::Kind::Vec)
            return it->second;
        }
        smt::Term bv = evalCoef(*cf, solver, store, getSort(vt.elem, solver));
        std::vector<SymbolicValue> lanes;
        lanes.reserve(vt.size);
        for (std::uint64_t k = 0; k < vt.size; ++k)
          lanes.emplace_back(SymbolicValue::Kind::Int, bv, solver.make_true());
        return buildVec(std::move(lanes));
      }
      throw std::runtime_error("Vec select arm: unsupported");
    };
    SymbolicValue vtArm = loadArm(arg.vtrue);
    SymbolicValue vfArm = loadArm(arg.vfalse);
    std::vector<SymbolicValue> lanes;
    lanes.reserve(vt.size);
    smt::Sort i1 = solver.make_bv_sort(1);
    smt::Term oneI1 = solver.make_bv_value(i1, "1", 10);
    for (std::uint64_t k = 0; k < vt.size; ++k) {
      smt::Term cond = solver.make_term(smt::Kind::EQUAL, {maskV.arrayVal[k].term, oneI1});
      smt::Term chosen =
          solver.make_term(smt::Kind::ITE, {cond, vtArm.arrayVal[k].term, vfArm.arrayVal[k].term});
      // Propagate per-lane is_defined: select on an undef
      // lane is still undef. Use the lane's source is_defined if
      // present; default to "defined" otherwise.
      smt::Term chosenDef;
      if (vtArm.arrayVal[k].is_defined.internal && vfArm.arrayVal[k].is_defined.internal) {
        chosenDef = solver.make_term(
            smt::Kind::ITE, {cond, vtArm.arrayVal[k].is_defined, vfArm.arrayVal[k].is_defined}
        );
      } else {
        chosenDef = solver.make_true();
      }
      lanes.emplace_back(SymbolicValue::Kind::Int, chosen, chosenDef);
    }
    return buildVec(std::move(lanes));

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalVecOpAtom(
      const OpAtom &arg, const VecType &vt, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {

    // Per-lane scalar op with coef broadcast or coef-as-vector.
    std::vector<smt::Term> coefLanes;
    auto laneSort = getSort(vt.elem, solver);
    // Resolve coef as a per-lane sequence.
    if (auto i = std::get_if<IntLit>(&arg.coef)) {
      smt::Term bv = solver.make_bv_value(laneSort, std::to_string(i->value), 10);
      for (std::uint64_t k = 0; k < vt.size; ++k)
        coefLanes.push_back(bv);
    } else if (auto fl = std::get_if<FloatLit>(&arg.coef)) {
      if (!solver.is_fp_sort(laneSort))
        throw std::runtime_error("Vec OpAtom: float-literal coef on non-FP vector lane sort");
      smt::Term fp = solver.make_fp_value_from_real(laneSort, fl->value, smt::RoundingMode::RNE);
      for (std::uint64_t k = 0; k < vt.size; ++k)
        coefLanes.push_back(fp);
    } else if (auto id = std::get_if<LocalOrSymId>(&arg.coef)) {
      std::string nm = std::visit([](auto &&x) { return x.name; }, *id);
      auto it = store.find(nm);
      if (it == store.end())
        throw std::runtime_error("Vec OpAtom: coef name not in store: " + nm);
      if (it->second.kind == SymbolicValue::Kind::Vec) {
        for (std::uint64_t k = 0; k < vt.size; ++k)
          coefLanes.push_back(it->second.arrayVal[k].term);
      } else {
        // Scalar local/sym → broadcast.
        for (std::uint64_t k = 0; k < vt.size; ++k)
          coefLanes.push_back(it->second.term);
      }
    } else {
      throw std::runtime_error("Vec OpAtom: unsupported coef shape");
    }
    // Rval is a vector lvalue.
    SymbolicValue rvalV = evalLValue(arg.rval, solver, store, pc, ub);
    std::vector<SymbolicValue> lanes;
    lanes.reserve(vt.size);
    bool fpLane = solver.is_fp_sort(laneSort);
    auto rmRNE = fpLane ? solver.make_rm_value(smt::RoundingMode::RNE) : smt::Term();
    for (std::uint64_t k = 0; k < vt.size; ++k) {
      smt::Term c = coefLanes[k];
      smt::Term r = rvalV.arrayVal[k].term;
      smt::Term out;
      if (fpLane) {
        // Float lanes: SMT-LIB FP theory.
        switch (arg.op) {
          case AtomOpKind::Mul:
            out = solver.make_term(smt::Kind::FP_MUL, {rmRNE, c, r});
            break;
          case AtomOpKind::Div:
            out = solver.make_term(smt::Kind::FP_DIV, {rmRNE, c, r});
            break;
          default:
            throw std::runtime_error("Vec FP OpAtom: only mul/div supported on FP lanes");
        }
        // Per-lane FP rule 6/7: each lane must be finite.
        assertFPFinite(out, solver, ub);
      } else {
        // Per-lane shift UB: amount in [0, width) and Shl
        // requires non-negative operand (rule 4/5).
        auto sortR = solver.get_sort(r);
        auto sortC = solver.get_sort(c);
        auto bvZeroR = solver.make_bv_zero(sortR);
        auto bvZeroC = solver.make_bv_zero(sortC);
        uint32_t laneWidth = solver.get_bv_width(sortC);
        auto widthTerm = solver.make_bv_value(sortR, std::to_string(laneWidth), 10);
        auto addShiftBounds = [&]() {
          ub.push_back(solver.make_term(smt::Kind::BV_SLE, {bvZeroR, r}));
          ub.push_back(solver.make_term(smt::Kind::BV_ULT, {r, widthTerm}));
        };
        switch (arg.op) {
          case AtomOpKind::Mul: {
            auto ov = solver.make_term(smt::Kind::BV_SMUL_OVERFLOW, {c, r});
            ub.push_back(solver.make_term(smt::Kind::NOT, {ov}));
            out = solver.make_term(smt::Kind::BV_MUL, {c, r});
            break;
          }
          case AtomOpKind::Div: {
            ub.push_back(solver.make_term(smt::Kind::DISTINCT, {r, bvZeroR}));
            // Per-lane INT_MIN / -1 overflow is UB
            // (rule 4 lifted to lanes by rule 21).
            auto min_signed = solver.make_bv_min_signed(solver.get_sort(c));
            auto minus_one = solver.make_bv_value_int64(solver.get_sort(r), -1);
            auto is_min = solver.make_term(smt::Kind::EQUAL, {c, min_signed});
            auto is_minus_one = solver.make_term(smt::Kind::EQUAL, {r, minus_one});
            auto div_overflow = solver.make_term(smt::Kind::AND, {is_min, is_minus_one});
            ub.push_back(solver.make_term(smt::Kind::NOT, {div_overflow}));
            out = solver.make_term(smt::Kind::BV_SDIV, {c, r});
            break;
          }
          case AtomOpKind::Mod: {
            ub.push_back(solver.make_term(smt::Kind::DISTINCT, {r, bvZeroR}));
            // Per-lane INT_MIN % -1 overflow is UB
            // (rule 4 lifted to lanes by rule 21).
            auto min_signed_m = solver.make_bv_min_signed(solver.get_sort(c));
            auto minus_one_m = solver.make_bv_value_int64(solver.get_sort(r), -1);
            auto is_min_m = solver.make_term(smt::Kind::EQUAL, {c, min_signed_m});
            auto is_minus_one_m = solver.make_term(smt::Kind::EQUAL, {r, minus_one_m});
            auto mod_overflow = solver.make_term(smt::Kind::AND, {is_min_m, is_minus_one_m});
            ub.push_back(solver.make_term(smt::Kind::NOT, {mod_overflow}));
            out = solver.make_term(smt::Kind::BV_SREM, {c, r});
            break;
          }
          case AtomOpKind::And:
            out = solver.make_term(smt::Kind::BV_AND, {c, r});
            break;
          case AtomOpKind::Or:
            out = solver.make_term(smt::Kind::BV_OR, {c, r});
            break;
          case AtomOpKind::Xor:
            out = solver.make_term(smt::Kind::BV_XOR, {c, r});
            break;
          case AtomOpKind::Shl: {
            addShiftBounds();
            ub.push_back(solver.make_term(smt::Kind::BV_SLE, {bvZeroC, c}));
            auto shifted = solver.make_term(smt::Kind::BV_SHL, {c, r});
            auto unshifted = solver.make_term(smt::Kind::BV_ASHR, {shifted, r});
            ub.push_back(solver.make_term(smt::Kind::EQUAL, {unshifted, c}));
            out = shifted;
            break;
          }
          case AtomOpKind::Shr:
            addShiftBounds();
            out = solver.make_term(smt::Kind::BV_ASHR, {c, r});
            break;
          case AtomOpKind::LShr:
            addShiftBounds();
            out = solver.make_term(smt::Kind::BV_SHR, {c, r});
            break;
        }
      }
      lanes.emplace_back(SymbolicValue::Kind::Int, out, solver.make_true());
    }
    return buildVec(std::move(lanes));

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalVecCastAtom(
      const CastAtom &arg, const VecType &vt, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {

    // Per-lane cast. Src is a vector lvalue or sym; dst is the
    // outer vt (already known to be vector by AssignInstr).
    SymbolicValue srcV;
    std::visit(
        [&](auto &&s) {
          using S = std::decay_t<decltype(s)>;
          if constexpr (std::is_same_v<S, LValue>)
            srcV = evalLValue(s, solver, store, pc, ub);
          else if constexpr (std::is_same_v<S, SymId>) {
            auto it = store.find(s.name);
            if (it != store.end())
              srcV = it->second;
          } else {
            throw std::runtime_error("Vec cast: unsupported src kind");
          }
        },
        arg.src
    );
    if (srcV.kind != SymbolicValue::Kind::Vec)
      throw std::runtime_error("Vec cast: src must be a vector value");
    auto dstSort = getSort(vt.elem, solver);
    bool dstIsFp = solver.is_fp_sort(dstSort);
    std::vector<SymbolicValue> lanes;
    lanes.reserve(vt.size);
    for (std::uint64_t k = 0; k < vt.size; ++k) {
      smt::Term lane = srcV.arrayVal[k].term;
      auto srcSort = solver.get_sort(lane);
      bool srcIsFp = solver.is_fp_sort(srcSort);
      smt::Term out;
      if (srcIsFp && !dstIsFp) { // FP -> BV (RTZ + range check per lane)
        uint32_t width = solver.get_bv_width(dstSort);
        assertFPFinite(lane, solver, ub);
        double lo = -std::ldexp(1.0, static_cast<int>(width) - 1);
        double hi = std::ldexp(1.0, static_cast<int>(width) - 1);
        auto loFp = solver.make_fp_value_from_real(srcSort, lo, smt::RoundingMode::RNE);
        auto hiFp = solver.make_fp_value_from_real(srcSort, hi, smt::RoundingMode::RNE);
        ub.push_back(solver.make_term(smt::Kind::FP_GEQ, {lane, loFp}));
        ub.push_back(solver.make_term(smt::Kind::FP_LT, {lane, hiFp}));
        auto rmRTZ = solver.make_rm_value(smt::RoundingMode::RTZ);
        out = solver.make_term(smt::Kind::FP_TO_SBV, {rmRTZ, lane}, {width});
      } else if (!srcIsFp && dstIsFp) {
        auto [exp, sig] = solver.get_fp_dims(dstSort);
        auto rmRNE = solver.make_rm_value(smt::RoundingMode::RNE);
        out = solver.make_term(smt::Kind::FP_TO_FP_FROM_SBV, {rmRNE, lane}, {exp, sig});
        // Per-lane finite (rule 6/7 lifted).
        assertFPFinite(out, solver, ub);
      } else if (srcIsFp && dstIsFp) {
        auto [exp, sig] = solver.get_fp_dims(dstSort);
        auto rmRNE = solver.make_rm_value(smt::RoundingMode::RNE);
        out = solver.make_term(smt::Kind::FP_TO_FP_FROM_FP, {rmRNE, lane}, {exp, sig});
        // Per-lane finite (rule 6/7 lifted) — catches
        // f64→f32 overflow producing ±inf.
        assertFPFinite(out, solver, ub);
      } else { // BV -> BV
        uint32_t sw = solver.get_bv_width(srcSort);
        uint32_t dw = solver.get_bv_width(dstSort);
        if (sw == dw)
          out = lane;
        else if (sw < dw)
          out = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {lane}, {dw - sw});
        else
          out = solver.make_term(smt::Kind::BV_EXTRACT, {lane}, {dw - 1, 0});
      }
      lanes.emplace_back(SymbolicValue::Kind::Int, out, solver.make_true());
    }
    return buildVec(std::move(lanes));

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalVecUnaryAtom(
      const UnaryAtom &arg, const VecType &vt, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {

    // Per-lane bitwise NOT for vector operand.
    SymbolicValue srcV = evalLValue(arg.rval, solver, store, pc, ub);
    if (srcV.kind != SymbolicValue::Kind::Vec)
      throw std::runtime_error("Vec UnaryAtom: src must be vector");
    std::vector<SymbolicValue> lanes;
    lanes.reserve(vt.size);
    for (std::uint64_t k = 0; k < vt.size; ++k) {
      smt::Term lane = srcV.arrayVal[k].term;
      smt::Term out = solver.make_term(smt::Kind::BV_NOT, {lane});
      lanes.emplace_back(SymbolicValue::Kind::Int, out, solver.make_true());
    }
    return buildVec(std::move(lanes));

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

} // namespace refractir
