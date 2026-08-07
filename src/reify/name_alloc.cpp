#include "reify/name_alloc.hpp"

#include <string>
#include <variant>

#include "analysis/type_utils.hpp"

namespace refractir::reify {

  namespace {

    std::string typeKey(const TypePtr &t) {
      auto bits = TypeUtils::getIntBitWidth(t);
      return bits ? "i" + std::to_string(*bits) : "?";
    }

    LetDecl makeLet(const std::string &name, const TypePtr &type, std::int64_t init, bool mut) {
      LetDecl d;
      d.isMutable = mut;
      d.name = LocalId{name, {}};
      d.type = type;
      // A declaration's initializer has to suit its type: a scratch cell a
      // rule routes a value through is a pointer, and `let mut %p: ptr i32 =
      // 0;` is not a program the checker accepts.
      if (type && std::holds_alternative<PtrType>(type->v))
        d.init = InitVal{InitVal::Kind::Null, IntLit{0, {}}, {}};
      else if (TypeUtils::getFloatBitWidth(type))
        d.init = InitVal{InitVal::Kind::Float, FloatLit{(double) init, {}}, {}};
      else
        d.init = InitVal{InitVal::Kind::Int, IntLit{init, {}}, {}};
      return d;
    }

    bool nameTaken(const std::vector<LetDecl> &lets, const std::string &nm) {
      for (const auto &l: lets)
        if (l.name.name == nm)
          return true;
      return false;
    }

  } // namespace

  std::string NameAllocator::fresh(const TypePtr &type, std::vector<LetDecl> &lets) {
    // A function may hold several rewritten bodies, each with its own
    // allocator, and they all declare into the same list — so a name is only
    // fresh once nothing else has claimed it.
    std::string nm = prefix_ + std::to_string(next_++);
    while (nameTaken(lets, nm) || (taken_ && nameTaken(*taken_, nm)))
      nm = prefix_ + std::to_string(next_++);
    lets.push_back(makeLet(nm, type, 0, /*mut=*/true));
    return nm;
  }

  std::string
  NameAllocator::literal(std::int64_t value, const TypePtr &type, std::vector<LetDecl> &lets) {
    const std::string key = typeKey(type);
    for (const auto &[v, k, nm]: pool_)
      if (v == value && k == key)
        return nm;
    std::string nm = prefix_ + "k" + std::to_string(next_++);
    while (nameTaken(lets, nm) || (taken_ && nameTaken(*taken_, nm)))
      nm = prefix_ + "k" + std::to_string(next_++);
    lets.push_back(makeLet(nm, type, value, /*mut=*/false));
    pool_.emplace_back(value, key, nm);
    return nm;
  }

} // namespace refractir::reify
