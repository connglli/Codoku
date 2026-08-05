#include "interp/memory.hpp"
#include <memory>
#include "analysis/type_utils.hpp"

namespace refractir {

  void Memory::reset() {
    heap_.clear();
    objects_.clear();
    addrMap_.clear();
    nextAddr_ = 4096; // leave address 0 for null
    nextProvId_ = 1;
  }

  std::uint64_t Memory::bumpAlloc(std::uint64_t bytes) {
    std::uint64_t base = nextAddr_;
    nextAddr_ += (bytes + 7) & ~7ULL; // 8-byte alignment
    return base;
  }

  // Allocate (or return existing) base address for varName with type t.
  // Syncs the current store value into the heap.
  std::uint64_t
  Memory::allocObject(const std::string &varName, const TypePtr &t, const Store &store) {
    auto it = addrMap_.find(varName);
    if (it != addrMap_.end())
      return it->second;

    uint64_t totalSize = layout_.sizeofType(t);
    uint64_t base = bumpAlloc(totalSize);

    uint64_t elemSize =
        layout_.sizeofType(std::get_if<ArrayType>(&t->v) ? std::get<ArrayType>(t->v).elem : t);
    uint64_t count = std::get_if<ArrayType>(&t->v) ? std::get<ArrayType>(t->v).size : 1;
    addObject(
        ObjectInfo{
            varName, "", base, base + totalSize, elemSize, count, static_cast<std::uint64_t>(-1), 0,
            t
        }
    );
    addrMap_[varName] = base;

    // Sync current store value into heap. For arrays-of-structs the
    // element storage isn't itself a flat scalar — we recurse so that
    // every leaf field has its own per-byte heap entry and ObjectInfo.
    // This lets `ptrfield %p, f` for `%p = addr %arr[k]` resolve to a
    // valid object-bound load address even when %p has been advanced
    // via pointer arithmetic.
    auto sit = store.find(varName);
    if (sit != store.end()) {
      const RuntimeValue &sv = sit->second;
      if (sv.kind == RuntimeValue::Kind::Array) {
        auto at = std::get_if<ArrayType>(&t->v);
        auto elemTy = at ? at->elem : t;
        bool elemIsStruct = elemTy && std::holds_alternative<StructType>(elemTy->v);
        const StructDecl *sd = nullptr;
        if (elemIsStruct) {
          auto sIt = layout_.structs().find(std::get<StructType>(elemTy->v).name.name);
          if (sIt != layout_.structs().end())
            sd = sIt->second;
        }
        for (std::size_t i = 0; i < sv.arrayVal.size(); ++i) {
          uint64_t elemBase = base + i * elemSize;
          if (sd) {
            // Per-element struct: create a whole-element ObjectInfo (for
            // ptrfield provenance = the struct) plus per-field ObjectInfos.
            uint64_t off = 0;
            uint64_t minFS = elemSize;
            for (const auto &f: sd->fields) {
              uint64_t fs = layout_.sizeofType(f.type);
              if (fs < minFS)
                minFS = fs;
            }
            addObject(
                ObjectInfo{
                    varName, "", elemBase, elemBase + elemSize, minFS, elemSize / minFS, i, 0,
                    elemTy
                }
            );
            for (const auto &f: sd->fields) {
              uint64_t fSize = layout_.sizeofType(f.type);
              addObject(
                  ObjectInfo{
                      varName, f.name, elemBase + off, elemBase + off + fSize, fSize, 1, i, 0,
                      f.type
                  }
              );
              if (sv.arrayVal[i].kind == RuntimeValue::Kind::Struct) {
                auto fit = sv.arrayVal[i].structVal.find(f.name);
                if (fit != sv.arrayVal[i].structVal.end())
                  // Flatten aggregate fields (e.g. `f: [2] i32`) to per-leaf
                  // heap entries rather than storing the whole field value.
                  flattenValueToHeap(elemBase + off, fit->second, f.type);
              }
              off += fSize;
            }
          } else if (sv.arrayVal[i].kind == RuntimeValue::Kind::Array) {
            // Nested array element (e.g., [3] i32 inside
            // [2][3] i32): create a sub-array ObjectInfo and recursively
            // flatten leaf elements into per-address heap entries so that
            // ptrindex can navigate into sub-arrays and load individual
            // elements.
            auto subAt = elemTy ? std::get_if<ArrayType>(&elemTy->v) : nullptr;
            uint64_t subElemSize = subAt ? layout_.sizeofType(subAt->elem) : 4;
            uint64_t subCount = subAt ? subAt->size : sv.arrayVal[i].arrayVal.size();
            addObject(
                ObjectInfo{
                    varName, "", elemBase, elemBase + elemSize, subElemSize, subCount,
                    static_cast<std::uint64_t>(-1), 0, elemTy
                }
            );
            // Recursively flatten leaves into heap (handles array-of-array
            // and array-of-struct element nesting).
            flattenValueToHeap(elemBase, sv.arrayVal[i], elemTy);
          } else {
            heap_[elemBase] = sv.arrayVal[i];
          }
        }
      } else {
        heap_[base] = sv;
      }
    }
    return base;
  }

  const ObjectInfo *Memory::findObject(std::uint64_t addr) const {
    const ObjectInfo *best = nullptr;
    for (const auto &o: objects_) {
      if (addr >= o.base && addr < o.end) {
        if (!best) {
          best = &o;
        } else {
          uint64_t oSize = o.end - o.base;
          uint64_t bestSize = best->end - best->base;
          if (oSize < bestSize) {
            best = &o;
          } else if (oSize == bestSize && best->fieldName.empty() && !o.fieldName.empty()) {
            best = &o;
          }
        }
      }
    }
    return best;
  }

  // Like findObject, but also matches the one-past-the-end address (valid for arithmetic).
  // Two-pass: interior membership wins over one-past-the-end so that consecutive
  // allocations (where src.end == dst.base) are unambiguous.
  const ObjectInfo *Memory::findObjectForArith(std::uint64_t addr) const {
    for (const auto &o: objects_)
      if (addr >= o.base && addr < o.end)
        return &o;
    for (const auto &o: objects_)
      if (addr == o.end)
        return &o;
    return nullptr;
  }

  ObjectInfo &Memory::addObject(ObjectInfo obj) {
    obj.provId = nextProvId_++;
    objects_.push_back(std::move(obj));
    return objects_.back();
  }

  const ObjectInfo *Memory::findObjectByProvId(std::uint64_t provId) const {
    for (const auto &o: objects_)
      if (o.provId == provId)
        return &o;
    return nullptr;
  }

  const ObjectInfo *Memory::findObjectByBaseAddress(std::uint64_t base) const {
    for (const auto &o: objects_)
      if (o.base == base)
        return &o;
    return nullptr;
  }

  const ObjectInfo *Memory::findFieldOrStructObject(std::uint64_t addr, const TypePtr &type) const {
    uint64_t size = layout_.sizeofType(type);
    // Base + size alone is ambiguous for a single-element outer array: a
    // `[1][3] i16` and its sole `[3] i16` element share the same base and
    // the same 6-byte span, yet differ in element count (1 vs 3). Prefer an
    // object whose declared type matches `type` exactly so the navigated
    // element's count is correct; fall back to the first size match (the
    // historical behaviour) when no type is recorded.
    const ObjectInfo *sizeMatch = nullptr;
    for (const auto &o: objects_) {
      if (o.base == addr && (o.end - o.base) == size) {
        if (o.type && type && TypeUtils::areTypesEqual(o.type, type))
          return &o;
        if (!sizeMatch)
          sizeMatch = &o;
      }
    }
    if (sizeMatch)
      return sizeMatch;
    return findObject(addr);
  }

  void Memory::flattenValueToHeap(std::uint64_t addr, const RuntimeValue &v, const TypePtr &ty) {
    if (ty) {
      if (auto at = std::get_if<ArrayType>(&ty->v)) {
        std::uint64_t es = layout_.sizeofType(at->elem);
        for (std::size_t i = 0; i < v.arrayVal.size(); ++i)
          flattenValueToHeap(addr + i * es, v.arrayVal[i], at->elem);
        return;
      }
      if (auto vt = std::get_if<VecType>(&ty->v)) {
        std::uint64_t es = layout_.sizeofType(vt->elem);
        for (std::size_t i = 0; i < v.arrayVal.size(); ++i)
          flattenValueToHeap(addr + i * es, v.arrayVal[i], vt->elem);
        return;
      }
      if (auto st = std::get_if<StructType>(&ty->v)) {
        auto sit = layout_.structs().find(st->name.name);
        if (sit != layout_.structs().end() && v.kind == RuntimeValue::Kind::Struct) {
          std::uint64_t off = 0;
          for (const auto &f: sit->second->fields) {
            auto fit = v.structVal.find(f.name);
            if (fit != v.structVal.end())
              flattenValueToHeap(addr + off, fit->second, f.type);
            off += layout_.sizeofType(f.type);
          }
          return;
        }
      }
    }
    heap_[addr] = v; // scalar / pointer leaf
  }

  // Materialize a struct variable into the heap: allocate one ObjectInfo per field
  // with provenance [fieldBase, fieldBase+sizeof(T)).  Idempotent (no-op if already done).
  std::uint64_t
  Memory::materializeStruct(const std::string &varName, const StructDecl &s, const Store &store) {
    auto it = addrMap_.find(varName);
    if (it != addrMap_.end())
      return it->second;

    // Compute total size (sequential field layout, no padding).
    uint64_t totalSize = 0;
    for (const auto &f: s.fields)
      totalSize += layout_.sizeofType(f.type);

    uint64_t base = bumpAlloc(totalSize);
    addrMap_[varName] = base;

    // Create a whole-struct ObjectInfo so that ptrfield-derived
    // pointers (whose provenance = the struct per rule 15) can roam over
    // the entire struct range. The elemSize is set to the smallest field
    // size so ptr arith steps by the right granularity.
    uint64_t minFieldSize = totalSize;
    for (const auto &f: s.fields) {
      uint64_t fs = layout_.sizeofType(f.type);
      if (fs < minFieldSize)
        minFieldSize = fs;
    }
    addObject(
        ObjectInfo{
            varName, "", base, base + totalSize, minFieldSize, totalSize / minFieldSize,
            static_cast<std::uint64_t>(-1), 0, std::make_shared<Type>(StructType{s.name, s.span})
        }
    );

    // Create one ObjectInfo per field and sync its value into the heap.
    // Recurse into nested struct fields so that Rule 15b
    // typed-access mismatch checks can resolve nested field ObjectInfos.
    uint64_t offset = 0;
    auto sv = store.find(varName);
    for (const auto &f: s.fields) {
      uint64_t fBase = base + offset;
      uint64_t fElemSize;
      uint64_t fCount;
      bool fieldIsStruct = false;
      const StructDecl *fieldSD = nullptr;
      if (auto at = std::get_if<ArrayType>(&f.type->v)) {
        fCount = at->size;
        fElemSize = layout_.sizeofType(at->elem);
      } else if (auto st = std::get_if<StructType>(&f.type->v)) {
        // Nested struct field: create per-sub-field ObjectInfos.
        fCount = 1;
        fElemSize = layout_.sizeofType(f.type);
        auto sit = layout_.structs().find(st->name.name);
        if (sit != layout_.structs().end()) {
          fieldIsStruct = true;
          fieldSD = sit->second;
        }
      } else {
        fCount = 1;
        fElemSize = layout_.sizeofType(f.type);
      }
      uint64_t fSize = fCount * fElemSize;
      addObject(
          ObjectInfo{
              varName, f.name, fBase, fBase + fSize, fElemSize, fCount,
              static_cast<std::uint64_t>(-1), 0, f.type
          }
      );

      // For nested struct fields, create sub-field ObjectInfos
      // so that Rule 15b checks can identify typed-access mismatches inside
      // nested structs (e.g., ptr i32 accessing an i64 field).
      if (fieldIsStruct && fieldSD) {
        uint64_t subOff = 0;
        for (const auto &sf: fieldSD->fields) {
          uint64_t sfSize = layout_.sizeofType(sf.type);
          addObject(
              ObjectInfo{
                  varName, sf.name, fBase + subOff, fBase + subOff + sfSize, sfSize, 1,
                  static_cast<std::uint64_t>(-1), 0, sf.type
              }
          );
          subOff += sfSize;
        }
      }
      // Sync the field value into the heap, flattening any aggregate (nested
      // struct, array field, array-of-struct, …) down to per-leaf cells so a
      // `load` through a pointer to a deep cell reads a scalar there.
      if (sv != store.end() && sv->second.kind == RuntimeValue::Kind::Struct) {
        auto fit = sv->second.structVal.find(f.name);
        if (fit != sv->second.structVal.end())
          flattenValueToHeap(fBase, fit->second, f.type);
      }
      offset += fSize;
    }
    return base;
  }
} // namespace refractir
