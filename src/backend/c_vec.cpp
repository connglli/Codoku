#include <cmath>
#include <sstream>
#include "analysis/type_utils.hpp"
#include "backend/c_backend.hpp"
#include "c_internal.hpp"

namespace refractir {

  // cmp on vector operands lowers to a lane-wise loop. We need a
  // C-expression string for each side; SelectVal's parts can be either an
  // RValue (local name) or a Coef (literal / local / sym).
  // Per-lane C expression for a SelectVal. Delegates lane access
  // to the active VecLowering so each strategy picks its lane syntax.
  std::string
  CBackend::sirSelectValLane(const SelectVal &sv, const VecType &vt, const std::string &kExpr) {
    if (auto rv = std::get_if<RValue>(&sv)) {
      return vecLowering_->emitLaneRead(mangleName(rv->base.name), vt, kExpr);
    }
    if (auto cf = std::get_if<Coef>(&sv)) {
      if (auto i = std::get_if<IntLit>(cf))
        return std::to_string(i->value);
      if (auto f = std::get_if<FloatLit>(cf)) {
        std::ostringstream os;
        os.precision(17);
        os << f->value;
        return os.str();
      }
      if (auto id = std::get_if<LocalOrSymId>(cf)) {
        std::string nm = std::visit([](auto &&x) { return x.name; }, *id);
        return mangleName(nm);
      }
    }
    return "/*?*/";
  }

  void CBackend::emitVecCmpAssign(const LValue &lhs, const CmpAtom &c, const VecType &vt) {
    std::string dst = mangleName(lhs.base.name);
    const char *op = nullptr;
    switch (c.op) {
      case RelOp::EQ:
        op = "==";
        break;
      case RelOp::NE:
        op = "!=";
        break;
      case RelOp::LT:
        op = "<";
        break;
      case RelOp::LE:
        op = "<=";
        break;
      case RelOp::GT:
        op = ">";
        break;
      case RelOp::GE:
        op = ">=";
        break;
    }
    VecType operandVt = vt;
    if (auto rv = std::get_if<RValue>(&c.lhs)) {
      auto t = getLValueType(*rv);
      if (t && std::holds_alternative<VecType>(t->v))
        operandVt = std::get<VecType>(t->v);
    }
    for (std::uint64_t k = 0; k < vt.size; ++k) {
      std::string kS = std::to_string(k);
      std::string dstLane = vecLowering_->emitLaneRead(dst, vt, kS);
      std::string l = sirSelectValLane(c.lhs, operandVt, kS);
      std::string r = sirSelectValLane(c.rhs, operandVt, kS);
      if (k) {
        out_ << ";\n";
        indent();
      }
      out_ << dstLane << " = ((" << l << ") " << op << " (" << r << ")) ? 1 : 0";
    }
    out_ << ";\n";
  }

  void
  CBackend::emitVecMaskSelectAssign(const LValue &lhs, const SelectAtom &s, const VecType &vt) {
    std::string dst = mangleName(lhs.base.name);
    if (!s.maskExpr->rest.empty())
      throw std::runtime_error("Mask-form select: complex mask expressions not yet lowered to C");

    VecType armVt = vt;
    if (auto rv = std::get_if<RValue>(&s.vtrue)) {
      auto t = getLValueType(*rv);
      if (t && std::holds_alternative<VecType>(t->v))
        armVt = std::get<VecType>(t->v);
    }

    for (std::uint64_t k = 0; k < vt.size; ++k) {
      std::string kS = std::to_string(k);
      std::string dstLane = vecLowering_->emitLaneRead(dst, vt, kS);

      std::string ml;
      if (auto maskRv = std::get_if<RValueAtom>(&s.maskExpr->first.v)) {
        auto &maskVt = std::get<VecType>(getLValueType(maskRv->rval)->v);
        ml = vecLowering_->emitLaneRead(mangleName(maskRv->rval.base.name), maskVt, kS);
      } else if (auto maskCmp = std::get_if<CmpAtom>(&s.maskExpr->first.v)) {
        VecType cmpVt = vt;
        if (auto rv = std::get_if<RValue>(&maskCmp->lhs)) {
          if (auto t = getLValueType(*rv))
            if (std::holds_alternative<VecType>(t->v))
              cmpVt = std::get<VecType>(t->v);
        }
        std::string l = sirSelectValLane(maskCmp->lhs, cmpVt, kS);
        std::string r = sirSelectValLane(maskCmp->rhs, cmpVt, kS);
        const char *op = (maskCmp->op == RelOp::EQ)   ? "=="
                         : (maskCmp->op == RelOp::NE) ? "!="
                         : (maskCmp->op == RelOp::LT) ? "<"
                         : (maskCmp->op == RelOp::LE) ? "<="
                         : (maskCmp->op == RelOp::GT) ? ">"
                                                      : ">=";
        ml = "((" + l + ") " + op + " (" + r + "))";
      } else {
        throw std::runtime_error("Unsupported mask atom in select");
      }

      std::string vtLane = sirSelectValLane(s.vtrue, armVt, kS);
      std::string vfLane = sirSelectValLane(s.vfalse, armVt, kS);
      if (k) {
        out_ << ";\n";
        indent();
      }
      out_ << dstLane << " = (" << ml << ") ? (" << vtLane << ") : (" << vfLane << ")";
    }
    out_ << ";\n";
  }

  // emitVecAtomLane: return a C expression for lane k of an Atom
  // that yields a vector value. Used by the lane-unroll path when the
  // active strategy can't lower vector ops as native C operators.
  std::string CBackend::emitVecAtomLane(const Atom &a, const VecType &vt, std::uint64_t k) {
    std::string kS = std::to_string(k);
    return std::visit(
        [&](auto &&arg) -> std::string {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, RValueAtom>) {
            // Vector lvalue → lane k via strategy.
            std::string base = mangleName(arg.rval.base.name);
            if (arg.rval.accesses.empty()) {
              return vecLowering_->emitLaneRead(base, vt, kS);
            }
            // Path with accesses (struct fields, etc.) — fall back to
            // emitLValue + lane subscript. Not exercised by our tests.
            return base + "/* nested */" + "[" + kS + "]";
          } else if constexpr (std::is_same_v<T, CoefAtom>) {
            return emitVecCoefAtomLane(arg, vt, k);
          } else if constexpr (std::is_same_v<T, OpAtom>) {
            return emitVecOpAtomLane(arg, vt, k);
          } else if constexpr (std::is_same_v<T, UnaryAtom>) {
            std::string rvalLane =
                vecLowering_->emitLaneRead(mangleName(arg.rval.base.name), vt, kS);
            return "(~(" + rvalLane + "))";
          } else if constexpr (std::is_same_v<T, CastAtom>) {
            return emitVecCastAtomLane(arg, vt, k);
          } else {
            // Other atoms (cmp / mask-select / load / addr) aren't handled
            // by the lane-unroll path — they're special-cased at
            // AssignInstr level above.
            return "/*?atom*/";
          }
        },
        a.v
    );
  }

  std::string
  CBackend::emitVecCoefAtomLane(const CoefAtom &arg, const VecType & /*vt*/, std::uint64_t k) {
    std::string kS = std::to_string(k);

    // Literal broadcasts.
    if (auto i = std::get_if<IntLit>(&arg.coef))
      return std::to_string(i->value);
    if (auto f = std::get_if<FloatLit>(&arg.coef)) {
      std::ostringstream os;
      os.precision(17);
      os << f->value;
      return os.str();
    }
    if (auto id = std::get_if<LocalOrSymId>(&arg.coef)) {
      std::string nm = std::visit([](auto &&x) { return x.name; }, *id);
      // If it names a vector local/sym, take lane k; else broadcast.
      auto vinfo = varTypes_.find(nm);
      if (vinfo != varTypes_.end() && std::holds_alternative<VecType>(vinfo->second->v)) {
        auto &vvt = std::get<VecType>(vinfo->second->v);
        return vecLowering_->emitLaneRead(mangleName(nm), vvt, kS);
      }
      return mangleName(nm);
    }
    return "/*?coef*/";
  }

  std::string CBackend::emitVecOpAtomLane(const OpAtom &arg, const VecType &vt, std::uint64_t k) {
    std::string kS = std::to_string(k);

    // coef <op> rval, lane-wise. coef may be a literal (broadcast)
    // or a vector lvalue.
    std::string coefLane;
    if (auto i = std::get_if<IntLit>(&arg.coef))
      coefLane = std::to_string(i->value);
    else if (auto f = std::get_if<FloatLit>(&arg.coef)) {
      std::ostringstream os;
      os.precision(17);
      os << f->value;
      coefLane = os.str();
    } else if (auto id = std::get_if<LocalOrSymId>(&arg.coef)) {
      std::string nm = std::visit([](auto &&x) { return x.name; }, *id);
      auto vinfo = varTypes_.find(nm);
      if (vinfo != varTypes_.end() && std::holds_alternative<VecType>(vinfo->second->v)) {
        auto &vvt = std::get<VecType>(vinfo->second->v);
        coefLane = vecLowering_->emitLaneRead(mangleName(nm), vvt, kS);
      } else {
        coefLane = mangleName(nm);
      }
    } else {
      coefLane = "/*?coef*/";
    }
    std::string rvalLane = vecLowering_->emitLaneRead(mangleName(arg.rval.base.name), vt, kS);
    const char *op = nullptr;
    switch (arg.op) {
      case AtomOpKind::Mul:
        op = "*";
        break;
      case AtomOpKind::Div:
        op = "/";
        break;
      case AtomOpKind::Mod:
        op = "%";
        break;
      case AtomOpKind::And:
        op = "&";
        break;
      case AtomOpKind::Or:
        op = "|";
        break;
      case AtomOpKind::Xor:
        op = "^";
        break;
      case AtomOpKind::Shl:
        op = "<<";
        break;
      case AtomOpKind::Shr:
        op = ">>";
        break;
      case AtomOpKind::LShr:
        op = ">>";
        break;
    }
    // Float % needs fmod / fmodf, with the §2.9 intermediate-overflow
    // UB check on x/y (see scalar `%` emission for the rationale).
    bool isFloat = vt.elem && std::holds_alternative<FloatType>(vt.elem->v);
    if (arg.op == AtomOpKind::Mod && isFloat) {
      bool isF32Lane = std::get<FloatType>(vt.elem->v).kind == FloatType::Kind::F32;
      const char *ty = isF32Lane ? "float" : "double";
      const char *fn = isF32Lane ? "fmodf" : "fmod";
      std::string guard =
          noUbGuards_ ? "" : "if (!__builtin_isfinite(_mx / _my)) __builtin_trap(); ";
      return std::string("({ ") + ty + " _mx = (" + coefLane + "), _my = (" + rvalLane + "); " +
             guard + fn + "(_mx, _my); })";
    }
    if (arg.op == AtomOpKind::LShr) {
      // Logical (unsigned) right-shift: cast lane to unsigned at
      // the lane's actual bit width so we don't accidentally
      // sign-extend through a wider unsigned type.
      const char *u = "uint32_t";
      if (auto it = std::get_if<IntType>(&vt.elem->v)) {
        int bits = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
        if (bits <= 8)
          u = "uint8_t";
        else if (bits <= 16)
          u = "uint16_t";
        else if (bits <= 32)
          u = "uint32_t";
        else
          u = "uint64_t";
      }
      return std::string("((") + u + ")(" + coefLane + ") >> (" + rvalLane + "))";
    }
    return "((" + coefLane + ") " + op + " (" + rvalLane + "))";
  }

  std::string
  CBackend::emitVecCastAtomLane(const CastAtom &arg, const VecType &vt, std::uint64_t k) {
    std::string kS = std::to_string(k);

    // src is LValue (or literal/sym, but those aren't vectors).
    auto lv = std::get_if<LValue>(&arg.src);
    if (!lv)
      return "/*?cast*/";
    auto srcTy = getLValueType(*lv);
    if (!srcTy || !std::holds_alternative<VecType>(srcTy->v))
      return "/*?cast2*/";
    auto &srcVt = std::get<VecType>(srcTy->v);
    std::string srcLane = vecLowering_->emitLaneRead(mangleName(lv->base.name), srcVt, kS);
    // C cast on scalar lane.
    std::ostringstream os;
    os << "((";
    // Element C type from vt:
    if (auto it = std::get_if<IntType>(&vt.elem->v)) {
      int bits = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
      if (bits <= 8)
        os << "int8_t";
      else if (bits <= 16)
        os << "int16_t";
      else if (bits <= 32)
        os << "int32_t";
      else
        os << "int64_t";
    } else if (auto ft = std::get_if<FloatType>(&vt.elem->v)) {
      os << (ft->kind == FloatType::Kind::F32 ? "float" : "double");
    }
    os << ")(" << srcLane << "))";
    return os.str();
  }

  std::string CBackend::emitVecExprLane(const Expr &e, const VecType &vt, std::uint64_t k) {
    std::string result = emitVecAtomLane(e.first, vt, k);
    for (const auto &tail: e.rest) {
      std::string rhs = emitVecAtomLane(tail.atom, vt, k);
      const char *op = (tail.op == AddOp::Plus) ? "+" : "-";
      result = "(" + result + " " + op + " " + rhs + ")";
    }
    return result;
  }

  void CBackend::emitVecAssign(const LValue &lhs, const Expr &rhs, const VecType &vt) {
    std::string dst = mangleName(lhs.base.name);
    // Whole-vector copy fast path: RHS is a single bare RValueAtom of
    // the same vector type — let the strategy emit one copy statement.
    if (rhs.rest.empty()) {
      if (auto rv = std::get_if<RValueAtom>(&rhs.first.v)) {
        if (rv->rval.accesses.empty()) {
          auto srcTy = getLValueType(rv->rval);
          if (srcTy && std::holds_alternative<VecType>(srcTy->v)) {
            vecLowering_->emitWholeCopy(out_, dst, mangleName(rv->rval.base.name), vt);
            out_ << ";\n";
            return;
          }
        }
      }
    }
    // Unroll: emit one statement per lane.
    for (std::uint64_t k = 0; k < vt.size; ++k) {
      if (k) {
        indent();
      }
      std::string dstLane = vecLowering_->emitLaneRead(dst, vt, std::to_string(k));
      out_ << dstLane << " = " << emitVecExprLane(rhs, vt, k) << ";\n";
    }
  }
} // namespace refractir
