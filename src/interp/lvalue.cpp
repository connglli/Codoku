#include <functional>
#include <stdexcept>
#include "analysis/type_utils.hpp"
#include "error.hpp"
#include "internal.hpp"
#include "interp/interpreter.hpp"

namespace refractir {

  RuntimeValue Interpreter::evalLValue(const LValue &lv, const Store &store) {
    const RuntimeValue *cur = &store.at(lv.base.name);

    for (const auto &acc: lv.accesses) {
      if (auto ai = std::get_if<AccessIndex>(&acc)) {
        if (cur->kind == RuntimeValue::Kind::Undef)
          throw UndefinedBehaviorError("UB: Reading field of undef");
        if (cur->kind != RuntimeValue::Kind::Array && cur->kind != RuntimeValue::Kind::Vec)
          throw std::runtime_error("Indexing non-array");

        // Eval index
        RuntimeValue idxVal;
        const auto &idx = ai->index;
        if (std::holds_alternative<IntLit>(idx)) {
          idxVal.kind = RuntimeValue::Kind::Int;
          idxVal.intVal = std::get<IntLit>(idx).value;
        } else {
          const auto &id = std::get<LocalOrSymId>(idx);
          if (auto lid = std::get_if<LocalId>(&id))
            idxVal = store.at(lid->name);
          else { // SymId
            auto sid = std::get_if<SymId>(&id);
            idxVal = store.at(sid->name);
          }
        }
        if (idxVal.kind == RuntimeValue::Kind::Undef)
          throw UndefinedBehaviorError("UB: Undef index");

        if (idxVal.intVal < 0 || (size_t) idxVal.intVal >= cur->arrayVal.size())
          throw UndefinedBehaviorError(
              cur->kind == RuntimeValue::Kind::Vec ? "UB: Vector lane index out of bounds"
                                                   : "UB: Array index out of bounds"
          );

        cur = &cur->arrayVal[idxVal.intVal];
      } else if (auto af = std::get_if<AccessField>(&acc)) {
        if (cur->kind == RuntimeValue::Kind::Undef)
          throw UndefinedBehaviorError("UB: Reading field of undef");
        if (cur->kind != RuntimeValue::Kind::Struct)
          throw std::runtime_error("Accessing field of non-struct");
        auto it = cur->structVal.find(af->field);
        if (it == cur->structVal.end())
          throw UndefinedBehaviorError("UB: Uninitialized field read");
        cur = &it->second;
      }
    }
    if (cur->kind == RuntimeValue::Kind::Undef)
      throw UndefinedBehaviorError("UB: Reading undef value");
    return *cur;
  }

  void Interpreter::setLValue(const LValue &lv, RuntimeValue val, Store &store) {
    RuntimeValue *cur = &store.at(lv.base.name);

    for (const auto &acc: lv.accesses) {
      if (auto ai = std::get_if<AccessIndex>(&acc)) {
        if (cur->kind != RuntimeValue::Kind::Array && cur->kind != RuntimeValue::Kind::Vec)
          throw std::runtime_error("Indexing non-array");
        RuntimeValue idxVal;
        const auto &idx = ai->index;
        if (std::holds_alternative<IntLit>(idx)) {
          idxVal.kind = RuntimeValue::Kind::Int;
          idxVal.intVal = std::get<IntLit>(idx).value;
        } else {
          const auto &id = std::get<LocalOrSymId>(idx);
          if (auto lid = std::get_if<LocalId>(&id))
            idxVal = store.at(lid->name);
          else {
            auto sid = std::get_if<SymId>(&id);
            idxVal = store.at(sid->name);
          }
        }
        if (idxVal.intVal < 0 || (size_t) idxVal.intVal >= cur->arrayVal.size())
          throw UndefinedBehaviorError(
              cur->kind == RuntimeValue::Kind::Vec ? "UB: Vector lane index out of bounds"
                                                   : "UB: Array index out of bounds"
          );
        cur = &cur->arrayVal[idxVal.intVal];
      } else if (auto af = std::get_if<AccessField>(&acc)) {
        if (cur->kind != RuntimeValue::Kind::Struct)
          throw std::runtime_error("Accessing field of non-struct");
        cur = &cur->structVal[af->field];
      }
    }

    // Enforce destination precision.
    //   Int:   canonicalize to bit width.
    //   Float: round to declared precision (f32 destinations must store the
    //          f32 image, not the wider double the RHS evaluator may have
    //          produced — evalCoef tags FloatLit with bits=64 by default).
    if (val.kind == RuntimeValue::Kind::Int) {
      val.bits = cur->bits;
      val.intVal = canonicalize(val.intVal, val.bits);
    } else if (val.kind == RuntimeValue::Kind::Float) {
      val.bits = cur->bits;
      if (val.bits == 32)
        val.floatVal = static_cast<double>(static_cast<float>(val.floatVal));
    }
    *cur = val;

    // Spec §9.4.7 / interp Store↔Heap consistency: when an addr-taken local
    // is updated via a direct assignment, the heap-side mirror must reflect
    // the new value so subsequent loads through `load <ptr>` see it.
    auto ait = memory_.addrMap().find(lv.base.name);
    if (ait == memory_.addrMap().end())
      return;
    uint64_t base = ait->second;
    const RuntimeValue &top = store.at(lv.base.name);

    // For nested accesses, recompute the offset relative to base.
    uint64_t addr = base;
    const RuntimeValue *walk = &top;
    auto typeIt = typeMap_.find(lv.base.name);
    TypePtr curType = (typeIt != typeMap_.end()) ? typeIt->second : nullptr;
    for (const auto &acc: lv.accesses) {
      if (auto ai = std::get_if<AccessIndex>(&acc)) {
        if (walk->kind != RuntimeValue::Kind::Array)
          return;
        int64_t idx = 0;
        if (std::holds_alternative<IntLit>(ai->index))
          idx = std::get<IntLit>(ai->index).value;
        else {
          const auto &id = std::get<LocalOrSymId>(ai->index);
          idx = std::visit([&](auto &&v) { return store.at(v.name).intVal; }, id);
        }
        if (idx < 0 || (size_t) idx >= walk->arrayVal.size())
          return;
        uint64_t elemSize = 0;
        if (curType) {
          if (auto at = std::get_if<ArrayType>(&curType->v)) {
            elemSize = typeLayout_.sizeofType(at->elem);
            curType = at->elem;
          }
        }
        if (elemSize == 0)
          return; // no type info — give up syncing nested case
        addr += idx * elemSize;
        walk = &walk->arrayVal[idx];
      } else if (auto af = std::get_if<AccessField>(&acc)) {
        if (walk->kind != RuntimeValue::Kind::Struct)
          return;
        if (!curType)
          return;
        auto st = std::get_if<StructType>(&curType->v);
        if (!st)
          return;
        auto sit = typeLayout_.structs().find(st->name.name);
        if (sit == typeLayout_.structs().end())
          return;
        addr += typeLayout_.fieldOffset(*sit->second, af->field);
        for (const auto &f: sit->second->fields)
          if (f.name == af->field) {
            curType = f.type;
            break;
          }
        auto sfit = walk->structVal.find(af->field);
        if (sfit == walk->structVal.end())
          return;
        walk = &sfit->second;
      }
    }

    std::function<void(uint64_t, const RuntimeValue &, const TypePtr &)> flattenToHeap =
        [&](uint64_t targetAddr, const RuntimeValue &rv, const TypePtr &ty) {
          if (rv.kind == RuntimeValue::Kind::Array) {
            auto innerAt = ty ? std::get_if<ArrayType>(&ty->v) : nullptr;
            auto innerElem = innerAt ? innerAt->elem : TypePtr{};
            uint64_t innerElemSz = innerElem ? typeLayout_.sizeofType(innerElem) : 4;
            for (std::size_t j = 0; j < rv.arrayVal.size(); ++j)
              flattenToHeap(targetAddr + j * innerElemSz, rv.arrayVal[j], innerElem);
          } else if (rv.kind == RuntimeValue::Kind::Struct) {
            auto innerSt = ty ? std::get_if<StructType>(&ty->v) : nullptr;
            if (innerSt) {
              auto sd = typeLayout_.structs().find(innerSt->name.name);
              if (sd != typeLayout_.structs().end()) {
                uint64_t off = 0;
                for (const auto &f: sd->second->fields) {
                  auto fit = rv.structVal.find(f.name);
                  if (fit != rv.structVal.end()) {
                    flattenToHeap(targetAddr + off, fit->second, f.type);
                  }
                  off += typeLayout_.sizeofType(f.type);
                }
              }
            }
          } else {
            memory_.heap()[targetAddr] = rv;
          }
        };

    flattenToHeap(addr, *walk, curType);
  }

  bool Interpreter::evalCond(const Cond &c, const Store &store) {
    RuntimeValue l = evalExpr(c.lhs, store);
    RuntimeValue r = evalExpr(c.rhs, store);

    if (l.kind == RuntimeValue::Kind::Undef || r.kind == RuntimeValue::Kind::Undef)
      throw UndefinedBehaviorError("UB: Reading undef in condition");

    // Promote Int to Float if needed (Literal inference support)
    if (l.kind == RuntimeValue::Kind::Float && r.kind == RuntimeValue::Kind::Int) {
      r.floatVal = static_cast<double>(r.intVal);
      r.kind = RuntimeValue::Kind::Float;
    } else if (l.kind == RuntimeValue::Kind::Int && r.kind == RuntimeValue::Kind::Float) {
      l.floatVal = static_cast<double>(l.intVal);
      l.kind = RuntimeValue::Kind::Float;
    }

    if (l.kind == RuntimeValue::Kind::Int && r.kind == RuntimeValue::Kind::Int) {
      // Spec §6.12 + §6.4: all iN are signed; the typechecker
      // range-checks integer literals against the target type's signed
      // range [-2^(N-1), 2^(N-1) - 1], so by the time we reach this
      // comparison both sides are already canonical signed int64 values
      // in their declared widths.  No bit-masking shim required.
      int64_t li = l.intVal, ri = r.intVal;
      switch (c.op) {
        case RelOp::EQ:
          return li == ri;
        case RelOp::NE:
          return li != ri;
        case RelOp::LT:
          return li < ri;
        case RelOp::LE:
          return li <= ri;
        case RelOp::GT:
          return li > ri;
        case RelOp::GE:
          return li >= ri;
      }
    } else if (l.kind == RuntimeValue::Kind::Float && r.kind == RuntimeValue::Kind::Float) {
      switch (c.op) {
        case RelOp::EQ:
          return l.floatVal == r.floatVal;
        case RelOp::NE:
          return l.floatVal != r.floatVal;
        case RelOp::LT:
          return l.floatVal < r.floatVal;
        case RelOp::LE:
          return l.floatVal <= r.floatVal;
        case RelOp::GT:
          return l.floatVal > r.floatVal;
        case RelOp::GE:
          return l.floatVal >= r.floatVal;
      }
    }
    if (l.kind == RuntimeValue::Kind::Ptr && r.kind == RuntimeValue::Kind::Ptr) {
      switch (c.op) {
        case RelOp::EQ:
          return l.ptrVal == r.ptrVal;
        case RelOp::NE:
          return l.ptrVal != r.ptrVal;
        case RelOp::LT:
        case RelOp::LE:
        case RelOp::GT:
        case RelOp::GE: {
          // Relational comparison requires same provenance object (spec §7.5 rule 14).
          // ptrBase==0 implies null or invalid pointer — always UB for relational ops.
          if (l.ptrBase == 0 || r.ptrBase == 0 || l.ptrBase != r.ptrBase)
            throw UndefinedBehaviorError(
                "UB: Relational pointer comparison across different objects"
            );
          if (c.op == RelOp::LT)
            return l.ptrVal < r.ptrVal;
          if (c.op == RelOp::LE)
            return l.ptrVal <= r.ptrVal;
          if (c.op == RelOp::GT)
            return l.ptrVal > r.ptrVal;
          return l.ptrVal >= r.ptrVal;
        }
      }
    }
    throw std::runtime_error("Cond operands must be same scalar kind");
  }
} // namespace refractir
