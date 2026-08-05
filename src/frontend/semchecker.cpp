#include "frontend/semchecker.hpp"
#include <algorithm>
#include "analysis/intrinsics.hpp"
#include "analysis/type_utils.hpp"

namespace refractir {

  refractir::PassResult SemChecker::run(Program &prog, DiagBag &diags) {
    std::unordered_set<std::string> globalNames;

    for (const auto &s: prog.structs) {
      if (globalNames.count(s.name.name)) {
        diags.error("Duplicate global name (struct): " + s.name.name, s.span);
      }
      globalNames.insert(s.name.name);
      checkStruct(s, diags);
    }

    for (const auto &f: prog.funs) {
      if (globalNames.count(f.name.name)) {
        diags.error("Duplicate global name (function): " + f.name.name, f.span);
      }
      globalNames.insert(f.name.name);
      checkFunction(f, diags);
    }

    // External declarations: each `decl @name` is a global name and
    // must not collide with any struct/fun/intrinsic. A contract-form `decl`
    // and a `fun` with the same name within the same file is rejected (the
    // cross-file body+contract conflict is enforced by the link resolver).
    for (const auto &d: prog.extDecls) {
      if (globalNames.count(d.name.name)) {
        diags.error("Duplicate global name (decl): " + d.name.name, d.span);
      }
      globalNames.insert(d.name.name);
      checkExtDecl(d, diags);
    }

    // Intrinsic declarations. Two intrinsics with the same name
    // but different parameter-type signatures are distinct functions and
    // may coexist (overloading). Same name + same param types = duplicate.
    // Intrinsic names are tracked separately so that same-name intrinsics
    // with different signatures don't collide with each other, but still
    // conflict with non-intrinsic globals (struct, fun, decl).
    std::unordered_set<std::string> intrinsicNames;
    std::unordered_set<std::string> intrinsicSigs;
    for (const auto &d: prog.intrinsics) {
      if (globalNames.count(d.name.name)) {
        diags.error("Duplicate global name (intrinsic): " + d.name.name, d.span);
      }
      std::string sig = d.name.name;
      sig += "(";
      for (size_t i = 0; i < d.params.size(); ++i) {
        if (i > 0)
          sig += ",";
        if (auto bits = TypeUtils::getIntBitWidth(d.params[i].type))
          sig += "i" + std::to_string(*bits);
        else if (auto ft =
                     d.params[i].type ? std::get_if<FloatType>(&d.params[i].type->v) : nullptr)
          // FP overloads must not collide on the same arity:
          // @to_bits(f32) and @to_bits(f64) are distinct intrinsics with
          // distinct lowerings.  Use the FP precision in the sig string so
          // both can be declared in the same program.
          sig += "f" + std::string(ft->kind == FloatType::Kind::F32 ? "32" : "64");
        else if (auto vt =
                     d.params[i].type ? std::get_if<VecType>(&d.params[i].type->v) : nullptr) {
          // Reduction overloads differ by vector shape:
          // @reduce_add(<4> i32) and @reduce_add(<8> i32) are distinct
          // declarations.  Encode both the lane count and the element type
          // so they don't collide on the same arity.
          sig += "<" + std::to_string(vt->size) + ">";
          if (auto ebits = TypeUtils::getIntBitWidth(vt->elem))
            sig += "i" + std::to_string(*ebits);
          else if (auto eft = vt->elem ? std::get_if<FloatType>(&vt->elem->v) : nullptr)
            sig += "f" + std::string(eft->kind == FloatType::Kind::F32 ? "32" : "64");
          else
            sig += "?";
        } else
          sig += "?";
      }
      sig += ")";
      if (intrinsicSigs.count(sig)) {
        diags.error("Duplicate intrinsic signature: " + d.name.name, d.span);
      }
      intrinsicNames.insert(d.name.name);
      intrinsicSigs.insert(sig);
      checkIntrinsicDecl(d, diags);
    }
    for (const auto &name: intrinsicNames)
      globalNames.insert(name);
    return diags.hasErrors() ? refractir::PassResult::Error : refractir::PassResult::Success;
  }

  void SemChecker::checkStruct(const StructDecl &s, DiagBag &diags) {
    std::unordered_set<std::string> fields;
    for (const auto &f: s.fields) {
      if (fields.count(f.name)) {
        diags.error("Duplicate field name: " + f.name, f.span);
      }
      fields.insert(f.name);
    }
  }

  void SemChecker::checkFunction(const FunDecl &f, DiagBag &diags) {
    if (f.blocks.empty()) {
      diags.error("Function must have at least one basic block", f.span);
    }

    checkSigils(f, diags);
    checkDuplicates(f, diags);

    // Check domains
    for (const auto &s: f.syms) {
      if (s.domain) {
        if (auto interval = std::get_if<DomainInterval>(&(*s.domain))) {
          if (interval->lo > interval->hi) {
            diags.error("Invalid symbol domain: lower bound > upper bound", interval->span);
          }
        }
      }
    }
  }

  void SemChecker::checkSigils(const FunDecl &f, DiagBag &diags) {
    // Inside a function, symbols must be local (%?) not global (@?)
    for (const auto &s: f.syms) {
      if (s.name.name.rfind("@?", 0) == 0) {
        diags.error(
            "Global symbol '" + s.name.name +
                "' declared in local scope. Use '%?' for local symbols.",
            s.name.span
        );
      }
    }
  }

  void SemChecker::checkDuplicates(const FunDecl &f, DiagBag &diags) {
    std::unordered_set<std::string> locals;
    std::unordered_set<std::string> labels;

    for (const auto &p: f.params) {
      if (locals.count(p.name.name)) {
        diags.error("Duplicate parameter name: " + p.name.name, p.span);
      }
      locals.insert(p.name.name);
    }

    for (const auto &s: f.syms) {
      if (locals.count(s.name.name)) {
        diags.error("Duplicate name (symbol): " + s.name.name, s.span);
      }
      locals.insert(s.name.name);
    }

    for (const auto &l: f.lets) {
      if (locals.count(l.name.name)) {
        diags.error("Duplicate name (local): " + l.name.name, l.span);
      }
      locals.insert(l.name.name);
    }

    for (const auto &b: f.blocks) {
      if (labels.count(b.label.name)) {
        diags.error("Duplicate block label: " + b.label.name, b.label.span);
      }
      labels.insert(b.label.name);
    }
  }

  // §3.4: a contract must have at least one `post` clause. `pre`
  // clauses are optional. Parameter names must be unique.
  void SemChecker::checkExtDecl(const ExtDecl &d, DiagBag &diags) {
    std::unordered_set<std::string> params;
    for (const auto &p: d.params) {
      if (params.count(p.name.name)) {
        diags.error("Duplicate parameter name: " + p.name.name, p.span);
      }
      params.insert(p.name.name);
    }
    if (d.contract) {
      if (d.contract->posts.empty()) {
        diags.error(
            "Contract on '" + d.name.name + "' must contain at least one `post` clause", d.span
        );
      }
    }
  }

  void SemChecker::checkIntrinsicDecl(const IntrinsicDecl &d, DiagBag &diags) {
    std::unordered_set<std::string> params;
    for (const auto &p: d.params) {
      if (params.count(p.name.name)) {
        diags.error("Duplicate parameter name: " + p.name.name, p.span);
      }
      params.insert(p.name.name);
    }

    // Per-intrinsic signature validation.
    // The interpreter/solver/codegen rely on these invariants; rejecting
    // mis-shaped declarations at check time is cheaper than diagnosing
    // them mid-execution.
    auto kind = getIntrinsicKind(d.name.name);
    if (!kind) {
      diags.error("Unknown intrinsic name: " + d.name.name, d.span);
      return;
    }

    // Validate the declaration against the intrinsic's canonical signature
    // (analysis/intrinsics.hpp). A signature is expressed over one type
    // parameter T: we infer T from the declaration, check its class against
    // the signature's domain, then verify each parameter and the return
    // against their slot forms. Diagnostics are intentionally generic — the
    // contract is simply "the declaration matches the canonical signature".
    const IntrinsicInfo &info = intrinsicInfo(*kind);

    if (d.params.size() != info.params.size()) {
      diags.error(
          "Intrinsic " + d.name.name + " expects " + std::to_string(info.params.size()) +
              " parameter(s), got " + std::to_string(d.params.size()),
          d.span
      );
      return;
    }

    // FP width (32/64) of a type, or nullopt if it is not floating-point.
    auto fpBits = [](const TypePtr &t) -> std::optional<std::uint32_t> {
      if (t)
        if (auto fp = std::get_if<FloatType>(&t->v))
          return fp->kind == FloatType::Kind::F32 ? 32u : 64u;
      return std::nullopt;
    };
    auto vecElem = [](const TypePtr &t) -> TypePtr {
      if (t)
        if (auto vt = std::get_if<VecType>(&t->v))
          return vt->elem;
      return nullptr;
    };

    // Infer the type parameter T from the first slot that carries it — a `T`
    // slot's declared type, or a `VecOfT` slot's element. Scan parameters
    // then the return. Signatures with no such slot (e.g. @check_chksum)
    // leave T null; their slots are all concrete and need no T.
    TypePtr T;
    auto inferT = [&](IntrinsicSigType form, const TypePtr &dt) {
      if (T)
        return;
      if (form == IntrinsicSigType::T)
        T = dt;
      else if (form == IntrinsicSigType::VecOfT)
        T = vecElem(dt);
    };
    for (std::size_t i = 0; i < info.params.size(); ++i)
      inferT(info.params[i], d.params[i].type);
    inferT(info.ret, d.retType);

    const bool tIsInt = T && TypeUtils::getIntBitWidth(T).has_value();
    const bool tIsFp = T && fpBits(T).has_value();
    auto tWidth = [&]() -> std::optional<std::uint32_t> {
      if (tIsInt)
        return TypeUtils::getIntBitWidth(T);
      if (tIsFp)
        return fpBits(T);
      return std::nullopt;
    };

    // The type parameter's class must lie within the signature's domain.
    if (T) {
      bool classOk = (info.domain == IntrinsicDomain::Int && tIsInt) ||
                     (info.domain == IntrinsicDomain::Fp && tIsFp) ||
                     (info.domain == IntrinsicDomain::IntOrFp && (tIsInt || tIsFp));
      if (!classOk)
        diags.error(
            "Intrinsic " + d.name.name + ": element type does not match the intrinsic's domain",
            d.span
        );
    }

    // Verify one slot's declared type against its form.
    auto checkSlot = [&](IntrinsicSigType form, const TypePtr &dt, const SourceSpan &span,
                         const std::string &what) {
      switch (form) {
        case IntrinsicSigType::T:
          if (!T || !TypeUtils::areTypesEqual(dt, T))
            diags.error(
                "Intrinsic " + d.name.name + ": " + what + " must be the element type T", span
            );
          break;
        case IntrinsicSigType::VecOfT: {
          TypePtr e = vecElem(dt);
          if (!e || !T || !TypeUtils::areTypesEqual(e, T))
            diags.error(
                "Intrinsic " + d.name.name + ": " + what + " must be a vector `<N> T`", span
            );
          break;
        }
        case IntrinsicSigType::I1: {
          auto ib = TypeUtils::getIntBitWidth(dt);
          if (!ib || *ib != 1)
            diags.error("Intrinsic " + d.name.name + ": " + what + " must be i1", span);
          break;
        }
        case IntrinsicSigType::IntWidthOfT: {
          auto ib = TypeUtils::getIntBitWidth(dt);
          auto tw = tWidth();
          if (!ib || !tw || *ib != *tw)
            diags.error(
                "Intrinsic " + d.name.name + ": " + what + " must be the iN of the same width as T",
                span
            );
          break;
        }
        case IntrinsicSigType::I32: {
          auto ib = TypeUtils::getIntBitWidth(dt);
          if (!ib || *ib != 32)
            diags.error("Intrinsic " + d.name.name + ": " + what + " must be i32", span);
          break;
        }
      }
    };
    checkSlot(info.ret, d.retType, d.span, "return type");
    for (std::size_t i = 0; i < info.params.size(); ++i)
      checkSlot(
          info.params[i], d.params[i].type, d.params[i].span, "parameter " + std::to_string(i)
      );

    // @bswap: the byte-swapped width must be a whole number of bytes.
    if (info.widthMultipleOf8)
      if (auto tw = tWidth(); tw && (*tw % 8) != 0)
        diags.error(
            "Intrinsic " + d.name.name + " requires a width that is a multiple of 8, got i" +
                std::to_string(*tw),
            d.span
        );
  }

} // namespace refractir
