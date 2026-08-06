#include "interp/type_layout.hpp"
#include <cstdint>
#include <stdexcept>
#include "analysis/type_utils.hpp"

namespace refractir {

  TypeLayout::TypeLayout(const Program &prog) : structs_(TypeUtils::buildStructTable(prog)) {}

  const StructDecl *TypeLayout::lookupStruct(const std::string &name) const {
    auto it = structs_.find(name);
    return it == structs_.end() ? nullptr : it->second;
  }

  std::uint64_t TypeLayout::sizeofType(const TypePtr &t) const {
    return TypeUtils::packedSizeof(t, structs_);
  }

  TypePtr TypeLayout::getCellTypeAtOffset(TypePtr t, std::uint64_t offset) const {
    if (!t)
      return nullptr;
    if (auto at = std::get_if<ArrayType>(&t->v)) {
      uint64_t elemSz = sizeofType(at->elem);
      if (elemSz == 0)
        return at->elem;
      uint64_t subOff = offset % elemSz;
      return getCellTypeAtOffset(at->elem, subOff);
    }
    if (auto st = std::get_if<StructType>(&t->v)) {
      auto sit = structs_.find(st->name.name);
      if (sit != structs_.end()) {
        uint64_t off = 0;
        for (const auto &f: sit->second->fields) {
          uint64_t fSize = sizeofType(f.type);
          if (offset >= off && offset < off + fSize) {
            return getCellTypeAtOffset(f.type, offset - off);
          }
          off += fSize;
        }
      }
    }
    return t;
  }

  // Byte offset of named field within struct s (sequential layout, no padding).
  std::uint64_t TypeLayout::fieldOffset(const StructDecl &s, const std::string &fieldName) const {
    uint64_t offset = 0;
    for (const auto &f: s.fields) {
      if (f.name == fieldName)
        return offset;
      offset += sizeofType(f.type);
    }
    throw std::runtime_error("Internal: field '" + fieldName + "' not found in struct");
  }

} // namespace refractir
