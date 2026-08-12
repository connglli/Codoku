#include <cassert>
#include <functional>
#include <stdexcept>
#include "backend/py_backend.hpp"
#include "py_internal.hpp"

namespace refractir {

  namespace {

    // Renders `base + scale*idx`-style offset terms, folding literal
    // parts so the common static paths stay readable.
    struct OffsetExpr {
      std::uint64_t constPart = 0;
      std::vector<std::string> dynTerms;

      void addConst(std::uint64_t c) { constPart += c; }

      void addDyn(const std::string &t) { dynTerms.push_back(t); }

      std::string str() const {
        if (dynTerms.empty())
          return std::to_string(constPart);
        std::string s;
        for (const auto &t: dynTerms) {
          if (!s.empty())
            s += " + ";
          s += t;
        }
        if (constPart != 0)
          s += " + " + std::to_string(constPart);
        return s;
      }

      std::string plusConst(std::uint64_t c) const {
        OffsetExpr tmp = *this;
        tmp.addConst(c);
        return tmp.str();
      }
    };

  } // namespace

  PyBackend::PathInfo PyBackend::resolvePath(const LValue &lv) {
    auto it = varTypes_.find(lv.base.name);
    assert(it != varTypes_.end() && "lvalue base not recorded");
    TypePtr cur = it->second;

    PathInfo info;
    info.boxed = boxedRoots_.count(lv.base.name) > 0;
    info.buf = pyLocal(lv.base.name);
    if (!info.boxed) {
      if (!lv.accesses.empty())
        throw std::runtime_error(
            "python target: element access on unboxed local not yet supported"
        );
      info.type = cur;
      return info;
    }

    OffsetExpr off;
    // Extent of the innermost enclosing object: narrows at each field
    // access and widens to the whole array at each index access.
    std::string lo = "0";
    std::string hi = std::to_string(byteSize(cur));
    for (const auto &acc: lv.accesses) {
      if (auto ai = std::get_if<AccessIndex>(&acc)) {
        std::uint64_t size;
        TypePtr elem;
        if (auto at = std::get_if<ArrayType>(&cur->v)) {
          size = at->size;
          elem = at->elem;
        } else if (auto vt = std::get_if<VecType>(&cur->v)) {
          size = vt->size; // lane subscript
          elem = vt->elem;
        } else {
          throw std::runtime_error("python target: subscript on non-array not yet supported");
        }
        const std::uint64_t elemLeaves = byteSize(elem);
        lo = off.str();
        hi = off.plusConst(size * elemLeaves);
        if (auto lit = std::get_if<IntLit>(&ai->index)) {
          // Literal indices are validated statically by the checker.
          off.addConst(static_cast<std::uint64_t>(lit->value) * elemLeaves);
        } else {
          std::string idx = "_idx(" + indexStr(ai->index) + ", " + std::to_string(size) + ")";
          off.addDyn(elemLeaves == 1 ? idx : idx + " * " + std::to_string(elemLeaves));
        }
        cur = elem;
      } else {
        const auto &af = std::get<AccessField>(acc);
        auto st = std::get_if<StructType>(&cur->v);
        if (!st)
          throw std::runtime_error("python target: field access on non-struct");
        TypePtr fieldTy;
        std::uint64_t foff = fieldByteOffset(st->name.name, af.field, &fieldTy);
        // SPEC §7.5 rule 15: a field pointer's provenance is the
        // immediate containing STRUCT — arithmetic may roam across
        // sibling fields.
        lo = off.str();
        hi = off.plusConst(byteSize(cur));
        off.addConst(foff);
        cur = fieldTy;
      }
    }
    info.off = off.str();
    info.lo = lo;
    info.hi = hi;
    info.type = cur;
    return info;
  }

  std::string PyBackend::indexStr(const Index &idx) {
    return std::visit(
        [this](auto &&arg) -> std::string {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, IntLit>) {
            return std::to_string(arg.value);
          } else {
            return std::visit(
                [this](auto &&id) -> std::string {
                  if constexpr (std::is_same_v<std::decay_t<decltype(id)>, SymId>) {
                    return symCall(id.name);
                  } else {
                    if (boxedRoots_.count(id.name)) {
                      return "_rd(" + pyLocal(id.name) + ", 0)";
                    }
                    return pyLocal(id.name);
                  }
                },
                arg
            );
          }
        },
        idx
    );
  }

  std::string PyBackend::laneIdxExpr(const Index &idx, std::uint64_t n) {
    if (auto lit = std::get_if<IntLit>(&idx))
      return std::to_string(lit->value);
    return "_idx(" + indexStr(idx) + ", " + std::to_string(n) + ")";
  }

  std::string PyBackend::lvalueStr(const LValue &lv) {
    // Vector locals/params are stored per the vec-lowering strategy;
    // vectors nested in aggregates stay flat slots of the boxed root.
    auto vit = varTypes_.find(lv.base.name);
    if (vit != varTypes_.end() && vit->second) {
      if (auto *vt = std::get_if<VecType>(&vit->second->v)) {
        const std::string name = pyLocal(lv.base.name);
        if (lv.accesses.empty()) {
          // Whole-vector value: a fresh, undef-checked lane list
          // (vectors are value types; consumers compute with lanes).
          return vecLowering_->readListExpr(name, *vt);
        }
        const auto &ai = std::get<AccessIndex>(lv.accesses.front());
        return vecLowering_->laneRead(name, *vt, laneIdxExpr(ai.index, vt->size));
      }
    }
    PathInfo p = resolvePath(lv);
    if (!p.boxed)
      return p.buf;
    if (p.type && std::holds_alternative<VecType>(p.type->v)) {
      // Whole vector inside an aggregate: a fresh, undef-checked lane
      // list read straight from the root's slots.
      // Lanes sit one element-width apart in the byte-indexed buffer, so
      // the read strides rather than walking consecutive slots.
      const auto &vt = std::get<VecType>(p.type->v);
      return "_vrd(" + p.buf + ", " + p.off + ", " + std::to_string(vt.size) + ", " +
             std::to_string(byteSize(vt.elem)) + ")";
    }
    const bool aggregate = p.type && (std::holds_alternative<ArrayType>(p.type->v) ||
                                      std::holds_alternative<StructType>(p.type->v));
    if (aggregate) {
      // Whole-aggregate value (e.g. an array argument): pass the root
      // list itself, or a slice copy for a sub-aggregate. Callees never
      // mutate parameters, so aliasing the root is safe.
      if (p.off == "0" && p.hi == std::to_string(byteSize(p.type)) && p.lo == "0")
        return p.buf;
      return p.buf + "[" + p.off + ":" + p.off + " + " + std::to_string(byteSize(p.type)) + "]";
    }
    return "_rd(" + p.buf + ", " + p.off + ")";
  }

  std::string PyBackend::addrAtomStr(const AddrAtom &arg) {
    PathInfo p = resolvePath(arg.lv);
    if (!p.boxed)
      throw std::runtime_error("python target: addr of a non-boxed local (backend bug)");
    // `addr` names a `let mut` of the executing function (spec §3.5.2), so the
    // storage is always this activation's and the frame cell always in scope.
    return "_Ptr(" + p.buf + ", " + p.off + ", " + std::to_string(byteSize(p.type)) + ", " + p.lo +
           ", " + p.hi + ", " + kFrameCell + ")";
  }

  std::string PyBackend::loadAtomStr(const LoadAtom &arg) {
    return "_load(" + lvalueStr(arg.rval) + ")";
  }

  std::string PyBackend::ptrIndexAtomStr(const PtrIndexAtom &arg) {
    TypePtr pt = getLValueType(arg.rval);
    const PtrType *ptr = pt ? std::get_if<PtrType>(&pt->v) : nullptr;
    const ArrayType *at = ptr && ptr->pointee ? std::get_if<ArrayType>(&ptr->pointee->v) : nullptr;
    if (!at)
      throw std::runtime_error("python target: ptrindex on a non-array pointer");
    return "_pidx(" + lvalueStr(arg.rval) + ", " + indexStr(arg.index) + ", " +
           std::to_string(at->size) + ", " + std::to_string(byteSize(at->elem)) + ")";
  }

  std::string PyBackend::ptrFieldAtomStr(const PtrFieldAtom &arg) {
    TypePtr pt = getLValueType(arg.rval);
    const PtrType *ptr = pt ? std::get_if<PtrType>(&pt->v) : nullptr;
    const StructType *st =
        ptr && ptr->pointee ? std::get_if<StructType>(&ptr->pointee->v) : nullptr;
    if (!st)
      throw std::runtime_error("python target: ptrfield on a non-struct pointer");
    TypePtr fieldTy;
    std::uint64_t foff = fieldByteOffset(st->name.name, arg.field, &fieldTy);
    // Provenance of the field pointer is the whole struct (§7.5 rule
    // 15), so pass the struct's leaf size as the bound.
    auto sit = structFields_.find(st->name.name);
    std::uint64_t slen = 0;
    for (const auto &[_, fty]: sit->second)
      slen += byteSize(fty);
    return "_pfield(" + lvalueStr(arg.rval) + ", " + std::to_string(foff) + ", " +
           std::to_string(byteSize(fieldTy)) + ", " + std::to_string(slen) + ")";
  }

  void PyBackend::emitAssign(const AssignInstr &ins) {
    // Vector locals/params: storage per the vec-lowering strategy.
    auto vit = varTypes_.find(ins.lhs.base.name);
    if (vit != varTypes_.end() && vit->second) {
      if (auto *vt = std::get_if<VecType>(&vit->second->v)) {
        const std::string name = pyLocal(ins.lhs.base.name);
        F32Guard ctx(f32Ctx_, !containsF64(vt->elem));
        if (!ins.lhs.accesses.empty()) {
          // Lane write.
          const auto &ai = std::get<AccessIndex>(ins.lhs.accesses.front());
          line(
              vecLowering_->laneWrite(name, *vt, laneIdxExpr(ai.index, vt->size), exprStr(ins.rhs))
          );
          return;
        }
        // Whole-vector copy from another vector lvalue transfers raw
        // slots (undef lanes stay undef — copying is not a read).
        if (ins.rhs.rest.empty()) {
          if (auto rv = std::get_if<RValueAtom>(&ins.rhs.first.v)) {
            const LValue &src = rv->rval;
            auto sit = varTypes_.find(src.base.name);
            if (sit != varTypes_.end() && sit->second &&
                std::holds_alternative<VecType>(sit->second->v) && src.accesses.empty()) {
              line(vecLowering_->wholeCopy(name, pyLocal(src.base.name), *vt));
              return;
            }
            // Source vector lives inside an aggregate: raw slot slice.
            PathInfo s = resolvePath(src);
            const std::string n = std::to_string(vt->size);
            std::string rhs = s.buf + "[" + s.off + ":" + s.off + " + " + n + "]";
            if (s.off == "0")
              rhs = "list(" + s.buf + ")";
            line(vecLowering_->assignFromList(name, *vt, rhs));
            return;
          }
        }
        // Lane comprehension: already a fresh list.
        line(vecLowering_->assignFromList(name, *vt, exprStr(ins.rhs)));
        return;
      }
    }
    PathInfo p = resolvePath(ins.lhs);
    F32Guard ctx(f32Ctx_, !containsF64(p.type));
    if (p.type && std::holds_alternative<VecType>(p.type->v)) {
      const std::string n = std::to_string(byteSize(p.type));
      // Destination vector lives inside an aggregate: raw slot slice
      // write into the boxed root (source handling as above).
      std::string rhs;
      if (ins.rhs.rest.empty()) {
        if (auto rv = std::get_if<RValueAtom>(&ins.rhs.first.v)) {
          const LValue &src = rv->rval;
          auto sit = varTypes_.find(src.base.name);
          if (sit != varTypes_.end() && sit->second &&
              std::holds_alternative<VecType>(sit->second->v) && src.accesses.empty()) {
            rhs = vecLowering_->rawListExpr(
                pyLocal(src.base.name), *std::get_if<VecType>(&sit->second->v)
            );
          } else {
            PathInfo s = resolvePath(src);
            rhs = s.buf + "[" + s.off + ":" + s.off + " + " + n + "]";
            if (s.off == "0")
              rhs = "list(" + s.buf + ")";
          }
        }
      }
      if (rhs.empty())
        rhs = exprStr(ins.rhs); // lane comprehension: already a fresh list
      if (p.off == "0" && !p.buf.empty() && p.lo == "0")
        line(p.buf + " = " + rhs);
      else
        line(p.buf + "[" + p.off + ":" + p.off + " + " + n + "] = " + rhs);
      return;
    }
    if (p.type && (std::holds_alternative<ArrayType>(p.type->v) ||
                   std::holds_alternative<StructType>(p.type->v)))
      throw std::runtime_error("python target: aggregate assignment not supported");
    std::string rhs = exprStr(ins.rhs);
    if (p.boxed)
      line(p.buf + "[" + p.off + "] = " + rhs);
    else
      line(p.buf + " = " + rhs);
  }

  void PyBackend::emitStore(const StoreInstr &ins) {
    TypePtr pointee;
    if (auto pt = getExprType(ins.ptr))
      if (auto ptr = std::get_if<PtrType>(&pt->v))
        pointee = ptr->pointee;
    F32Guard ctx(f32Ctx_, !containsF64(pointee));
    line("_store(" + exprStr(ins.ptr) + ", " + exprStr(ins.val) + ")");
  }

  std::string PyBackend::scalarInit(const InitVal &iv, const TypePtr &elemType) {
    const bool toF32 = floatWidth(elemType) == 32;
    switch (iv.kind) {
      case InitVal::Kind::Int:
        return std::to_string(std::get<IntLit>(iv.value).value);
      case InitVal::Kind::Float: {
        std::string s = formatFloatLit(std::get<FloatLit>(iv.value).value);
        return toF32 ? "_f32(" + s + ")" : s;
      }
      case InitVal::Kind::Sym:
        return symCall(std::get<SymId>(iv.value).name);
      case InitVal::Kind::Local: {
        // Scalar copy from another local (aggregate copies are
        // rejected by the checker); read through the box if needed.
        const std::string &src = std::get<LocalId>(iv.value).name;
        if (boxedRoots_.count(src))
          return "_rd(" + pyLocal(src) + ", 0)";
        return pyLocal(src);
      }
      case InitVal::Kind::Null:
        return "_NULL";
      case InitVal::Kind::Undef:
        return "_UNDEF";
      case InitVal::Kind::Atom: {
        std::string s = atomStr(*std::get<AtomPtr>(iv.value));
        return toF32 ? "_f32(" + s + ")" : s;
      }
      case InitVal::Kind::Aggregate:
        break;
    }
    throw std::runtime_error("python target: unexpected aggregate initializer element");
  }

  std::string PyBackend::flattenInit(const InitVal &iv, const TypePtr &type) {
    // A vector local's storage is a dense lane list, not a byte-indexed
    // buffer, so its lanes must stay adjacent; every other root is an
    // object whose leaves sit at their byte offsets.
    const bool densePacking = type && std::holds_alternative<VecType>(type->v);
    // Adjacent leaf items merge into one bracketed list; broadcast /
    // undef sub-inits contribute `[v] * N` pieces concatenated with +.
    std::vector<std::string> pieces;
    std::vector<std::string> run;
    auto flushRun = [&] {
      if (run.empty())
        return;
      std::string s = "[";
      for (std::size_t i = 0; i < run.size(); ++i) {
        if (i)
          s += ", ";
        s += run[i];
      }
      pieces.push_back(s + "]");
      run.clear();
    };

    std::function<void(const InitVal &, const TypePtr &)> flatten = [&](const InitVal &v,
                                                                        const TypePtr &t) {
      const bool aggregateTy =
          t && (std::holds_alternative<ArrayType>(t->v) ||
                std::holds_alternative<StructType>(t->v) || std::holds_alternative<VecType>(t->v));
      if (v.kind != InitVal::Kind::Aggregate) {
        if (!aggregateTy) {
          // The buffer is byte-indexed, so a leaf owns its first byte and
          // pads the rest of its width. `_PAD` traps if read, which is what
          // an interior (misaligned) access amounts to.
          run.push_back(v.kind == InitVal::Kind::Undef ? "_UNDEF" : scalarInit(v, t));
          if (!densePacking)
            for (std::uint64_t k = 1; k < byteSize(t); ++k)
              run.push_back("_PAD");
          return;
        }
        // Broadcast / undef over every leaf of the sub-aggregate.
        // Broadcasting reaches heterogeneous leaves (e.g. f64 fields
        // broadcast with 0); python numerics tolerate that.
        flushRun();
        std::string elem = v.kind == InitVal::Kind::Undef ? "_UNDEF" : scalarInit(v, nullptr);
        pieces.push_back("[" + elem + "] * " + std::to_string(byteSize(t)));
        return;
      }
      const auto &elems = std::get<std::vector<InitValPtr>>(v.value);
      if (auto at = std::get_if<ArrayType>(&t->v)) {
        for (const auto &e: elems)
          flatten(*e, at->elem);
      } else if (auto vt = std::get_if<VecType>(&t->v)) {
        for (const auto &e: elems)
          flatten(*e, vt->elem);
      } else if (auto st = std::get_if<StructType>(&t->v)) {
        auto sit = structFields_.find(st->name.name);
        if (sit == structFields_.end())
          throw std::runtime_error("python target: unknown struct in initializer");
        for (std::size_t i = 0; i < elems.size() && i < sit->second.size(); ++i)
          flatten(*elems[i], sit->second[i].second);
      } else {
        throw std::runtime_error("python target: brace initializer on a non-aggregate");
      }
    };

    flatten(iv, type);
    flushRun();
    if (pieces.empty())
      return "[]";
    std::string out;
    for (const auto &p: pieces) {
      if (!out.empty())
        out += " + ";
      out += p;
    }
    return out;
  }

  void PyBackend::collectBoxedRoots(const FunDecl &f) {
    boxedRoots_.clear();
    takesAddress_ = false;
    // Top-level vector locals/params are managed by the
    // vec-lowering strategy, not flattened into a boxed leaf-slot
    // list — only array/struct aggregates are boxed roots. (Vectors
    // *nested* in an array/struct are part of the enclosing root and
    // stay flat; that root is boxed by name here.)
    auto isBoxedAggregate = [](const TypePtr &t) {
      return t &&
             (std::holds_alternative<ArrayType>(t->v) || std::holds_alternative<StructType>(t->v));
    };
    for (const auto &p: f.params)
      if (isBoxedAggregate(p.type))
        boxedRoots_.insert(p.name.name);
    for (const auto &l: f.lets)
      if (isBoxedAggregate(l.type))
        boxedRoots_.insert(l.name.name);

    // Scalars become boxed when their address is taken anywhere.
    std::function<void(const Atom &)> walkAtom;
    std::function<void(const Expr &)> walkExpr = [&](const Expr &e) {
      walkAtom(e.first);
      for (const auto &t: e.rest)
        walkAtom(t.atom);
    };
    auto walkCond = [&](const Cond &c) {
      walkExpr(c.lhs);
      walkExpr(c.rhs);
    };
    walkAtom = [&](const Atom &a) {
      std::visit(
          [&](const auto &arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, AddrAtom>) {
              boxedRoots_.insert(arg.lv.base.name);
              takesAddress_ = true;
            } else if constexpr (std::is_same_v<T, SelectAtom>) {
              if (arg.cond)
                walkCond(*arg.cond);
              if (arg.maskExpr)
                walkExpr(*arg.maskExpr);
            } else if constexpr (std::is_same_v<T, CallAtom>) {
              for (const auto &e: arg.args)
                walkExpr(*e);
            }
          },
          a.v
      );
    };
    for (const auto &l: f.lets)
      if (l.init && l.init->kind == InitVal::Kind::Atom)
        walkAtom(*std::get<AtomPtr>(l.init->value));
    for (const auto &b: f.blocks) {
      for (const auto &ins: b.instrs) {
        std::visit(
            [&](const auto &arg) {
              using T = std::decay_t<decltype(arg)>;
              if constexpr (std::is_same_v<T, AssignInstr>) {
                walkExpr(arg.rhs);
              } else if constexpr (std::is_same_v<T, AssumeInstr>) {
                walkCond(arg.cond);
              } else if constexpr (std::is_same_v<T, RequireInstr>) {
                walkCond(arg.cond);
              } else {
                walkExpr(arg.ptr);
                walkExpr(arg.val);
              }
            },
            ins
        );
      }
      std::visit(
          [&](const auto &t) {
            using T = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<T, BrTerm>) {
              if (t.isConditional && t.cond)
                walkCond(*t.cond);
            } else if constexpr (std::is_same_v<T, RetTerm>) {
              if (t.value)
                walkExpr(*t.value);
            }
          },
          b.term
      );
    }
  }

} // namespace refractir
