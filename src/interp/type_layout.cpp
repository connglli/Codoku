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

  std::optional<std::vector<Access>> TypeLayout::accessPathAtOffset(
      const TypePtr &root, std::uint64_t offset, const TypePtr &target
  ) const {
    TypePtr t = root;
    std::vector<Access> path;
    while (true) {
      if (offset == 0 && TypeUtils::areTypesEqual(t, target))
        return path;
      if (const ArrayType *at = TypeUtils::asArray(t)) {
        std::uint64_t elemSize = sizeofType(at->elem);
        if (elemSize == 0)
          return std::nullopt;
        std::uint64_t idx = offset / elemSize;
        if (idx >= at->size)
          return std::nullopt; // out of bounds, or one past the end
        path.push_back(AccessIndex{Index{IntLit{static_cast<std::int64_t>(idx), {}}}, {}});
        offset -= idx * elemSize;
        t = at->elem;
        continue;
      }
      if (const StructType *st = TypeUtils::asStruct(t)) {
        const StructDecl *sd = lookupStruct(st->name.name);
        if (!sd)
          return std::nullopt;
        std::uint64_t fieldOfs = 0;
        const FieldDecl *hit = nullptr;
        for (const auto &f: sd->fields) {
          std::uint64_t size = sizeofType(f.type);
          if (offset < fieldOfs + size) {
            hit = &f;
            break;
          }
          fieldOfs += size;
        }
        if (!hit)
          return std::nullopt;
        path.push_back(AccessField{hit->name, {}});
        offset -= fieldOfs;
        t = hit->type;
        continue;
      }
      return std::nullopt; // a leaf, but the offset or the type disagrees
    }
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
