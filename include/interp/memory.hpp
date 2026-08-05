#pragma once

#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include "ast/ast.hpp"
#include "interp/type_layout.hpp"
#include "interp/value.hpp"

namespace refractir {

  /// Per-object provenance: tracks base address, size, element size.
  /// Namespace-scoped so both Memory and its Interpreter consumers can name it.
  struct ObjectInfo {
    std::string varName;    // originating local variable name
    std::string fieldName;  // non-empty for struct-field objects (addr lv.f)
    std::uint64_t base;     // base address (never 0)
    std::uint64_t end;      // base + totalSize (exclusive)
    std::uint64_t elemSize; // sizeof(element type) in bytes
    std::uint64_t count;    // number of elements
    // For array-of-struct field cells: the element index of
    // the containing struct (i.e. `%arr[k].fld`'s k). -1 / SIZE_MAX
    // when not array-nested. Used by StoreInstr to mirror the heap
    // write back into the right `store["%arr"].arrayVal[k]` cell.
    std::uint64_t arrayIdx = static_cast<std::uint64_t>(-1);
    std::uint64_t provId = 0; // unique provenance object ID
    TypePtr type = nullptr;   // The static type of the object/field

    ObjectInfo() = default;

    ObjectInfo(
        std::string vn, std::string fn, std::uint64_t b, std::uint64_t e, std::uint64_t es,
        std::uint64_t c, std::uint64_t ai = -1, std::uint64_t pi = 0, TypePtr t = nullptr
    ) :
        varName(vn), fieldName(fn), base(b), end(e), elemSize(es), count(c), arrayIdx(ai),
        provId(pi), type(t) {}
  };

  /**
   * The interpreter's memory model for pointer operations.
   *
   * Owns the flat heap (address -> RuntimeValue cell), the provenance object
   * table, the per-local address map, and the bump-allocator / provenance-id
   * counters. Provides the allocate / materialize / flatten / lookup
   * operations the interpreter performs through pointers. Holds a reference to
   * a TypeLayout for size queries; does not depend on the Interpreter.
   *
   * State that several callers touch in bespoke ways (heap, objects, addrMap)
   * is exposed through accessors, mirroring TypeLayout::structs().
   */
  class Memory {
  public:
    explicit Memory(const TypeLayout &layout) : layout_(layout) {}

    // Allocate (or return the existing) base address for `varName` of type `t`,
    // syncing its current store value into the heap.
    std::uint64_t allocObject(const std::string &varName, const TypePtr &t, const Store &store);
    // Materialize a struct local: one ObjectInfo per field plus a whole-struct
    // object; idempotent (no-op if already materialized).
    std::uint64_t
    materializeStruct(const std::string &varName, const StructDecl &s, const Store &store);
    // Write every scalar/pointer leaf of `v` (of static type `ty`) into the
    // flat heap at its byte offset from `addr`, recursing through arrays,
    // vectors, and structs, so a `load` through a pointer to a deep cell
    // (e.g. `%a[k].f[i]`) finds a scalar there rather than a sub-aggregate.
    void flattenValueToHeap(std::uint64_t addr, const RuntimeValue &v, const TypePtr &ty);

    ObjectInfo &addObject(ObjectInfo obj);
    const ObjectInfo *findObject(std::uint64_t addr) const;
    const ObjectInfo *findObjectForArith(std::uint64_t addr) const;
    const ObjectInfo *findObjectByProvId(std::uint64_t provId) const;
    const ObjectInfo *findObjectByBaseAddress(std::uint64_t base) const;
    const ObjectInfo *findFieldOrStructObject(std::uint64_t addr, const TypePtr &type) const;

    /// Clear all per-function state (heap, objects, addresses) and reset the
    /// bump allocator (null stays at 0) and provenance counter.
    void reset();
    /// 8-byte-aligned bump allocation of `bytes`; returns the new base address.
    std::uint64_t bumpAlloc(std::uint64_t bytes);

    using AddrMap = std::unordered_map<std::string, std::uint64_t>;

    std::unordered_map<std::uint64_t, RuntimeValue> &heap() { return heap_; }

    const std::unordered_map<std::uint64_t, RuntimeValue> &heap() const { return heap_; }

    const std::list<ObjectInfo> &objects() const { return objects_; }

    AddrMap &addrMap() { return addrMap_; }

    const AddrMap &addrMap() const { return addrMap_; }

    const TypeLayout &layout() const { return layout_; }

  private:
    const TypeLayout &layout_;
    // heap_: flat address -> RuntimeValue (one slot per element)
    std::unordered_map<std::uint64_t, RuntimeValue> heap_;
    // objects_: per-function allocation tracking
    std::list<ObjectInfo> objects_;
    // addrMap_: varName -> base address (assigned lazily on first addr)
    AddrMap addrMap_;
    // nextAddr_: allocator counter (starts at 4096 to leave null = 0 at bottom)
    std::uint64_t nextAddr_ = 4096;
    // nextProvId_: unique provenance ID counter
    std::uint64_t nextProvId_ = 1;
  };

} // namespace refractir
