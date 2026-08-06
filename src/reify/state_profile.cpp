#include "reify/state_profile.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <ostream>
#include <sstream>

#include "analysis/type_utils.hpp"
#include "interp/interpreter.hpp"
#include "interp/memory.hpp"
#include "interp/type_layout.hpp"

namespace refractir::reify {

  namespace {
    // Fill a Ptr StateValue's provenance from the live memory model. Any
    // resolution miss (foreign frame, freed object, inconsistent base)
    // leaves the leaf opaque — consumers treat opaque like undef.
    void resolvePtr(StateValue &sv, const RuntimeValue &rv, const Memory &memory) {
      if (rv.ptrVal == 0) {
        sv.ptrNull = true;
        return;
      }
      // rv.ptrBase carries the provenance object's ID, not an address.
      if (auto prov = memory.resolveProvenance(rv.ptrVal, rv.ptrBase)) {
        sv.ptrRoot = prov->root;
        sv.ptrOfs = prov->offset;
      }
    }
  } // namespace

  StateValue toStateValue(const RuntimeValue &rv, const Memory &memory) {
    StateValue sv = toStateValue(rv);
    if (rv.kind == RuntimeValue::Kind::Ptr) {
      resolvePtr(sv, rv, memory);
    } else if (rv.kind == RuntimeValue::Kind::Array || rv.kind == RuntimeValue::Kind::Vec) {
      for (std::size_t i = 0; i < rv.arrayVal.size(); ++i)
        sv.elems[i] = toStateValue(rv.arrayVal[i], memory);
    } else if (rv.kind == RuntimeValue::Kind::Struct) {
      for (auto &[name, val]: sv.fields)
        val = toStateValue(rv.structVal.at(name), memory);
    }
    return sv;
  }

  StateValue toStateValue(const RuntimeValue &rv) {
    StateValue sv;
    switch (rv.kind) {
      case RuntimeValue::Kind::Int:
        sv.kind = StateValue::Kind::Int;
        sv.intVal = rv.intVal;
        sv.bits = rv.bits;
        break;
      case RuntimeValue::Kind::Float:
        sv.kind = StateValue::Kind::Float;
        sv.floatVal = rv.floatVal;
        sv.bits = rv.bits;
        break;
      case RuntimeValue::Kind::Array:
        sv.kind = StateValue::Kind::Array;
        sv.elems.reserve(rv.arrayVal.size());
        for (const auto &e: rv.arrayVal)
          sv.elems.push_back(toStateValue(e));
        break;
      case RuntimeValue::Kind::Vec:
        sv.kind = StateValue::Kind::Vec;
        sv.elems.reserve(rv.arrayVal.size());
        for (const auto &e: rv.arrayVal)
          sv.elems.push_back(toStateValue(e));
        break;
      case RuntimeValue::Kind::Struct:
        sv.kind = StateValue::Kind::Struct;
        // structVal is an unordered_map; sort fields by name for a stable
        // serialization. Consumers that need declaration order re-derive it
        // from the struct decl, so name order here is purely for determinism.
        {
          std::vector<std::string> names;
          names.reserve(rv.structVal.size());
          for (const auto &[k, _]: rv.structVal)
            names.push_back(k);
          std::sort(names.begin(), names.end());
          for (const auto &n: names)
            sv.fields.emplace_back(n, toStateValue(rv.structVal.at(n)));
        }
        break;
      case RuntimeValue::Kind::Ptr:
        sv.kind = StateValue::Kind::Ptr;
        break;
      case RuntimeValue::Kind::Undef:
      default:
        sv.kind = StateValue::Kind::Undef;
        break;
    }
    return sv;
  }

  namespace {
    void enumStateLeavesRec(
        const StateValue &v, std::vector<Access> &path, std::vector<StateLeaf> &out, bool &hasPtr,
        bool &hasUndef
    ) {
      switch (v.kind) {
        case StateValue::Kind::Int:
        case StateValue::Kind::Float:
          out.push_back({path, v});
          break;
        case StateValue::Kind::Array:
        case StateValue::Kind::Vec:
          for (std::size_t i = 0; i < v.elems.size(); ++i) {
            path.push_back(AccessIndex{Index{IntLit{(std::int64_t) i, {}}}, {}});
            enumStateLeavesRec(v.elems[i], path, out, hasPtr, hasUndef);
            path.pop_back();
          }
          break;
        case StateValue::Kind::Struct:
          for (const auto &f: v.fields) {
            path.push_back(AccessField{f.first, {}});
            enumStateLeavesRec(f.second, path, out, hasPtr, hasUndef);
            path.pop_back();
          }
          break;
        case StateValue::Kind::Ptr:
          // Pointer leaves are emitted with their provenance in `val`;
          // the flag lets consumers that cannot handle them reject fast.
          out.push_back({path, v});
          hasPtr = true;
          break;
        case StateValue::Kind::Undef:
        default:
          hasUndef = true;
          break;
      }
    }
  } // namespace

  void
  enumStateLeaves(const StateValue &v, std::vector<StateLeaf> &out, bool &hasPtr, bool &hasUndef) {
    std::vector<Access> path;
    enumStateLeavesRec(v, path, out, hasPtr, hasUndef);
  }

  bool bitExactEq(const StateValue &a, const StateValue &b) {
    if (a.kind != b.kind)
      return false;
    switch (a.kind) {
      case StateValue::Kind::Int:
        return a.intVal == b.intVal;
      case StateValue::Kind::Float: {
        std::uint64_t ba, bb;
        static_assert(sizeof(ba) == sizeof(a.floatVal));
        std::memcpy(&ba, &a.floatVal, sizeof(ba));
        std::memcpy(&bb, &b.floatVal, sizeof(bb));
        return ba == bb;
      }
      case StateValue::Kind::Array:
      case StateValue::Kind::Vec: {
        if (a.elems.size() != b.elems.size())
          return false;
        for (std::size_t i = 0; i < a.elems.size(); ++i)
          if (!bitExactEq(a.elems[i], b.elems[i]))
            return false;
        return true;
      }
      case StateValue::Kind::Struct: {
        if (a.fields.size() != b.fields.size())
          return false;
        for (std::size_t i = 0; i < a.fields.size(); ++i)
          if (a.fields[i].first != b.fields[i].first ||
              !bitExactEq(a.fields[i].second, b.fields[i].second))
            return false;
        return true;
      }
      case StateValue::Kind::Ptr:
        return a.ptrNull == b.ptrNull && a.ptrRoot == b.ptrRoot && a.ptrOfs == b.ptrOfs;
      default:
        return true; // Undef: kind-only comparison
    }
  }

  void attachStateProfile(Interpreter &interp, StateProfile &out, StateGranularity gran) {
    interp.setStateHook(
        [&out](
            const std::string &funcName, std::uint64_t frameId, const std::string &blockLabel,
            int instrIdx, const Store &store, const Memory &memory
        ) {
          StatePoint pt;
          pt.func = funcName;
          pt.frame = frameId;
          pt.block = blockLabel;
          pt.instr = instrIdx;
          // Collect names first so the record is deterministically ordered.
          std::vector<std::string> names;
          names.reserve(store.size());
          for (const auto &[name, rv]: store) {
            // Skip uninitialized top-level locals — they carry no state.
            if (rv.kind == RuntimeValue::Kind::Undef)
              continue;
            names.push_back(name);
          }
          std::sort(names.begin(), names.end());
          for (const auto &n: names)
            pt.vars.emplace_back(n, toStateValue(store.at(n), memory));
          out.trace.push_back(std::move(pt));
        },
        /*perInstr=*/gran == StateGranularity::Ppp
    );
  }

  StateProfile profileProgram(
      const Program &prog, const std::string &func, const std::vector<std::string> &paramArgs,
      StateGranularity granularity, std::uint64_t maxBlockSteps
  ) {
    StateProfile profile;
    profile.func = func.empty() || func[0] == '@' ? func : "@" + func;
    profile.granularity = granularity;

    // Suppress the interpreter's "Result:" line — we only want the captured
    // states, and touching the process-global std::cout would be unsafe
    // under concurrency.
    std::ostringstream sink;
    Interpreter interp(prog, sink);
    attachStateProfile(interp, profile, granularity);
    interp.setMaxBlockSteps(maxBlockSteps);
    interp.run(profile.func, {}, paramArgs, /*dumpExec=*/false);
    return profile;
  }

  namespace {

    void writeString(std::ostream &os, const std::string &s) {
      os << '"';
      for (char c: s) {
        if (c == '"' || c == '\\')
          os << '\\' << c;
        else
          os << c;
      }
      os << '"';
    }

    // Emit one StateValue as a tagged JSON object. Integers and floats are
    // carried as strings so large i64 values and bit-exact floats survive
    // the JSON boundary without precision loss.
    void writeValue(std::ostream &os, const StateValue &v) {
      switch (v.kind) {
        case StateValue::Kind::Int:
          os << "{\"k\":\"i\",\"w\":" << v.bits << ",\"v\":\"" << v.intVal << "\"}";
          break;
        case StateValue::Kind::Float:
          os << "{\"k\":\"f\",\"w\":" << v.bits << ",\"v\":\"" << formatDouble(v.floatVal) << "\"}";
          break;
        case StateValue::Kind::Array:
        case StateValue::Kind::Vec: {
          os << "{\"k\":\"" << (v.kind == StateValue::Kind::Array ? "arr" : "vec") << "\",\"e\":[";
          for (std::size_t i = 0; i < v.elems.size(); ++i) {
            if (i)
              os << ',';
            writeValue(os, v.elems[i]);
          }
          os << "]}";
          break;
        }
        case StateValue::Kind::Struct: {
          os << "{\"k\":\"st\",\"f\":{";
          for (std::size_t i = 0; i < v.fields.size(); ++i) {
            if (i)
              os << ',';
            writeString(os, v.fields[i].first);
            os << ':';
            writeValue(os, v.fields[i].second);
          }
          os << "}}";
          break;
        }
        case StateValue::Kind::Ptr:
          if (v.ptrNull)
            os << "{\"k\":\"ptr\",\"null\":1}";
          else if (!v.ptrRoot.empty()) {
            os << "{\"k\":\"ptr\",\"root\":";
            writeString(os, v.ptrRoot);
            os << ",\"ofs\":" << v.ptrOfs << "}";
          } else
            os << "{\"k\":\"ptr\"}";
          break;
        case StateValue::Kind::Undef:
        default:
          os << "{\"k\":\"undef\"}";
          break;
      }
    }

  } // namespace

  namespace {

    // Focused recursive-descent reader for the regular JSON that
    // writeStateProfileJson emits. Not a general JSON parser — it assumes
    // the object-key order and value shapes this file produces, and only
    // unescapes the `"` / `\` pairs writeString introduces.
    struct JParser {
      const std::string &s;
      std::size_t i = 0;
      bool ok = true;

      explicit JParser(const std::string &str) : s(str) {}

      void skip() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
          ++i;
      }

      bool peek(char c) {
        skip();
        return i < s.size() && s[i] == c;
      }

      void eat(char c) {
        if (peek(c))
          ++i;
        else
          ok = false;
      }

      void comma() {
        if (peek(','))
          ++i;
      }

      std::string str() {
        skip();
        std::string out;
        if (i >= s.size() || s[i] != '"') {
          ok = false;
          return out;
        }
        ++i;
        while (i < s.size() && s[i] != '"') {
          char c = s[i++];
          if (c == '\\' && i < s.size())
            out.push_back(s[i++]); // writeString only escapes the quote and backslash chars
          else
            out.push_back(c);
        }
        if (i < s.size() && s[i] == '"')
          ++i;
        else
          ok = false;
        return out;
      }

      long num() {
        skip();
        std::size_t st = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+'))
          ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
          ++i;
        if (st == i) {
          ok = false;
          return 0;
        }
        return std::stol(s.substr(st, i - st));
      }
    };

    StateValue parseValue(JParser &p) {
      StateValue v;
      std::string kind, raw;
      p.eat('{');
      while (p.ok && !p.peek('}')) {
        std::string key = p.str();
        p.eat(':');
        if (key == "k") {
          kind = p.str();
        } else if (key == "w") {
          v.bits = static_cast<std::uint32_t>(p.num());
        } else if (key == "v") {
          raw = p.str();
        } else if (key == "root") {
          v.ptrRoot = p.str();
        } else if (key == "ofs") {
          v.ptrOfs = static_cast<std::uint64_t>(p.num());
        } else if (key == "null") {
          v.ptrNull = p.num() != 0;
        } else if (key == "e") {
          p.eat('[');
          while (p.ok && !p.peek(']')) {
            v.elems.push_back(parseValue(p));
            p.comma();
          }
          p.eat(']');
        } else if (key == "f") {
          p.eat('{');
          while (p.ok && !p.peek('}')) {
            std::string fn = p.str();
            p.eat(':');
            v.fields.emplace_back(fn, parseValue(p));
            p.comma();
          }
          p.eat('}');
        }
        p.comma();
      }
      p.eat('}');
      if (kind == "i") {
        v.kind = StateValue::Kind::Int;
        v.intVal = raw.empty() ? 0 : std::stoll(raw);
      } else if (kind == "f") {
        v.kind = StateValue::Kind::Float;
        v.floatVal = raw.empty() ? 0.0 : parseFloatLiteral(raw);
      } else if (kind == "arr") {
        v.kind = StateValue::Kind::Array;
      } else if (kind == "vec") {
        v.kind = StateValue::Kind::Vec;
      } else if (kind == "st") {
        v.kind = StateValue::Kind::Struct;
      } else if (kind == "ptr") {
        v.kind = StateValue::Kind::Ptr;
      } else {
        v.kind = StateValue::Kind::Undef;
      }
      return v;
    }

  } // namespace

  std::optional<StateProfile> readStateProfileJson(const std::string &json) {
    JParser p(json);
    StateProfile prof;
    p.eat('{');
    while (p.ok && !p.peek('}')) {
      std::string key = p.str();
      p.eat(':');
      if (key == "func") {
        prof.func = p.str();
      } else if (key == "granularity") {
        prof.granularity = p.str() == "ppp" ? StateGranularity::Ppp : StateGranularity::Pbb;
      } else if (key == "trace") {
        p.eat('[');
        while (p.ok && !p.peek(']')) {
          StatePoint pt;
          p.eat('{');
          while (p.ok && !p.peek('}')) {
            std::string k2 = p.str();
            p.eat(':');
            if (k2 == "block") {
              pt.block = p.str();
            } else if (k2 == "fn") {
              pt.func = p.str();
            } else if (k2 == "frame") {
              pt.frame = static_cast<std::uint64_t>(p.num());
            } else if (k2 == "instr") {
              pt.instr = static_cast<int>(p.num());
            } else if (k2 == "vars") {
              p.eat('{');
              while (p.ok && !p.peek('}')) {
                std::string vn = p.str();
                p.eat(':');
                pt.vars.emplace_back(vn, parseValue(p));
                p.comma();
              }
              p.eat('}');
            }
            p.comma();
          }
          p.eat('}');
          prof.trace.push_back(std::move(pt));
          p.comma();
        }
        p.eat(']');
      }
      p.comma();
    }
    p.eat('}');
    if (!p.ok)
      return std::nullopt;
    return prof;
  }

  void writeStateProfileJson(std::ostream &os, const StateProfile &profile) {
    os << "{\n  \"func\": ";
    writeString(os, profile.func);
    os << ",\n  \"granularity\": \""
       << (profile.granularity == StateGranularity::Ppp ? "ppp" : "pbb") << "\",\n";
    os << "  \"trace\": [\n";
    for (std::size_t i = 0; i < profile.trace.size(); ++i) {
      const StatePoint &pt = profile.trace[i];
      os << "    {\"fn\": ";
      writeString(os, pt.func);
      os << ", \"frame\": " << pt.frame << ", \"block\": ";
      writeString(os, pt.block);
      os << ", \"instr\": " << pt.instr << ", \"vars\": {";
      for (std::size_t j = 0; j < pt.vars.size(); ++j) {
        if (j)
          os << ", ";
        writeString(os, pt.vars[j].first);
        os << ": ";
        writeValue(os, pt.vars[j].second);
      }
      os << "}}";
      if (i + 1 < profile.trace.size())
        os << ',';
      os << '\n';
    }
    os << "  ]\n}\n";
  }

} // namespace refractir::reify
