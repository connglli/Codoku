#pragma once

// -I include-path resolution shared by symiri / symirc / symirsolve.
// Inline because the implementation is small and trivial to inline.

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ast/ast.hpp"
#include "frontend/lexer.hpp"
#include "frontend/parser.hpp"

namespace refractir {

  /**
   * Load every `.sir` file under each `-I` directory (recursively) and
   * return the parsed Programs. The caller must keep the returned vector
   * alive for as long as references into it (e.g. via resolveLinkDecls)
   * are used.
   *
   * -I paths are canonicalised and deduped so the same
   * directory passed twice doesn't double-load (which would otherwise
   * produce false ambiguity diagnostics).
   */
  inline std::vector<Program> loadIncludeDirs(const std::vector<std::string> &dirs) {
    std::vector<Program> libs;
    std::unordered_set<std::string> seen;
    for (const auto &dir: dirs) {
      if (!std::filesystem::exists(dir))
        throw std::runtime_error("-I path does not exist: " + dir);
      std::string canonical;
      try {
        canonical = std::filesystem::canonical(dir).string();
      } catch (const std::exception &) {
        canonical = dir; // fall back to the raw spelling
      }
      if (!seen.insert(canonical).second)
        continue;
      // §11.1 Non-recursive scan: a sibling subdirectory of an
      // `-I` path is invisible. Tests that need a specific subdir as a
      // library root must name it explicitly with its own `-I` flag.
      // Matches the literal spec wording and prevents fixture
      // collections in `test/tmp/` from bleeding into tests that pass
      // `-I test/tmp` for a different sub-fixture.
      for (const auto &entry: std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file())
          continue;
        if (entry.path().extension() != ".sir")
          continue;
        std::ifstream ifs(entry.path());
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string libSrc = ss.str();
        try {
          Lexer lx(libSrc);
          auto toks = lx.lexAll();
          Parser ps(std::move(toks));
          Program lib = ps.parseProgram();
          // Tag every fun with its source-file stem so the
          // C backend's --split-by-source mode knows where each fun
          // came from once they're merged into main.
          std::string stem = entry.path().stem().string();
          for (auto &f: lib.funs)
            if (f.sourceStem.empty())
              f.sourceStem = stem;
          libs.push_back(std::move(lib));
        } catch (const LexError &e) {
          std::cerr << "Error in -I file " << entry.path() << ": " << e.what() << "\n";
          throw;
        } catch (const ParseError &e) {
          std::cerr << "Error in -I file " << entry.path() << ": " << e.what() << "\n";
          throw;
        }
      }
    }
    return libs;
  }

  /**
   * Resolve every link-form `decl @name` in `main` against the loaded
   * library Programs.
   *
   * §3.3 two-pass loading and §11.1 -I resolution:
   *   1. Count every global-name binding (`fun`, link-form `decl`,
   *      contract-form `decl`, `intrinsic`) across primary + all libs.
   *   2. Diagnose duplicates that are not legal under the spec:
   *        - two `fun` bodies with the same name (ambiguity);
   *        - `fun` body + contract-form `decl` (mutually exclusive);
   *        - `intrinsic` + any other kind of binding (collision).
   *   3. Pull every lib `fun` body into `main.funs` so that lib bodies
   *      that call sibling lib funs see them during semantic / type
   *      checking. Also pull in lib `struct` and `intrinsic` decls.
   *      Link-form `decl`s in main are dropped once a matching body is
   *      present; unresolved link-form decls are kept and surface later
   *      as "undeclared function" errors when called.
   */
  inline void resolveLinkDecls(Program &main, std::vector<Program> &libs) {
    // ---- Pass 1: signature scan ----
    enum class Kind { Fun, DeclLink, DeclContract, Intrinsic };

    struct Binding {
      Kind kind;
      std::size_t libIdx; // -1 for main
      std::string sourceLabel;
    };

    std::unordered_map<std::string, std::vector<Binding>> sig;

    auto addMainBinding = [&](const std::string &name, Kind k) {
      sig[name].push_back({k, static_cast<std::size_t>(-1), "primary"});
    };
    for (const auto &f: main.funs)
      addMainBinding(f.name.name, Kind::Fun);
    for (const auto &d: main.extDecls)
      addMainBinding(d.name.name, d.contract ? Kind::DeclContract : Kind::DeclLink);
    for (const auto &i: main.intrinsics)
      addMainBinding(i.name.name, Kind::Intrinsic);

    for (std::size_t li = 0; li < libs.size(); ++li) {
      const auto &lib = libs[li];
      auto label = "-I lib #" + std::to_string(li);
      for (const auto &f: lib.funs)
        sig[f.name.name].push_back({Kind::Fun, li, label});
      for (const auto &d: lib.extDecls)
        sig[d.name.name].push_back({d.contract ? Kind::DeclContract : Kind::DeclLink, li, label});
      for (const auto &i: lib.intrinsics)
        sig[i.name.name].push_back({Kind::Intrinsic, li, label});
    }

    // ---- Pass 1b: diagnose disallowed duplicates ----
    auto locStr = [](const std::vector<Binding> &bs) {
      std::string s;
      for (std::size_t i = 0; i < bs.size(); ++i) {
        if (i)
          s += ", ";
        s += bs[i].sourceLabel;
      }
      return s;
    };
    // Multiple link-form `decl`s or multiple `intrinsic` declarations
    // for the same name are OK across files (each file independently
    // re-declares the signature so that it can stand alone). Disallowed:
    //   - multiple `fun` bodies (ambiguous body),
    //   - multiple contract-form `decl`s (ambiguous contract),
    //   - `fun` body + contract-form `decl` (§3.3 mutually exclusive),
    //   - `intrinsic` + `fun` body or contract-form `decl` (binding
    //     collision: an intrinsic is the body).
    for (const auto &[name, bs]: sig) {
      if (bs.size() < 2)
        continue;
      std::size_t funCnt = 0, contractCnt = 0, intrCnt = 0;
      for (const auto &b: bs) {
        if (b.kind == Kind::Fun)
          ++funCnt;
        else if (b.kind == Kind::DeclContract)
          ++contractCnt;
        else if (b.kind == Kind::Intrinsic)
          ++intrCnt;
      }
      if (funCnt > 1)
        throw std::runtime_error(
            "Ambiguous body for `" + name + "`: multiple `fun` definitions found (" + locStr(bs) +
            ")"
        );
      if (contractCnt > 1)
        throw std::runtime_error(
            "Ambiguous contract for `" + name + "`: multiple contract-form `decl`s found (" +
            locStr(bs) + ")"
        );
      if (contractCnt > 0 && funCnt > 0)
        throw std::runtime_error(
            "Conflict for `" + name +
            "`: contract-form `decl` and `fun` body are mutually exclusive (§3.3) (" + locStr(bs) +
            ")"
        );
      if (intrCnt > 0 && (funCnt > 0 || contractCnt > 0))
        throw std::runtime_error(
            "Conflict for `" + name + "`: `intrinsic` collides with another declaration (" +
            locStr(bs) + ")"
        );
    }

    // ---- Pass 2: move lib bodies into main ----
    std::unordered_set<std::string> mainFunNames;
    for (const auto &f: main.funs)
      mainFunNames.insert(f.name.name);
    for (auto &lib: libs) {
      for (auto it = lib.funs.begin(); it != lib.funs.end();) {
        if (mainFunNames.insert(it->name.name).second) {
          main.funs.push_back(std::move(*it));
          it = lib.funs.erase(it);
        } else {
          ++it;
        }
      }
    }

    // Drop link-form decls whose body is now present in main (either
    // as a `fun` or as an `intrinsic`). Unresolved link decls are kept
    // so calls to them surface a clear diagnostic from semcheck or the
    // interpreter — declaring a link-form `decl` that is never called
    // is allowed (no body required).
    std::unordered_set<std::string> mainIntrSeen;
    for (const auto &i: main.intrinsics)
      mainIntrSeen.insert(i.name.name);
    for (const auto &lib: libs)
      for (const auto &i: lib.intrinsics)
        mainIntrSeen.insert(i.name.name);
    std::vector<ExtDecl> keep;
    for (auto &d: main.extDecls) {
      if (d.contract) {
        keep.push_back(std::move(d));
        continue;
      }
      if (mainFunNames.count(d.name.name))
        continue;
      if (mainIntrSeen.count(d.name.name))
        continue;
      keep.push_back(std::move(d));
    }
    main.extDecls = std::move(keep);

    // Pull in lib-side link-form `decl`s for any name that:
    //   - is not already declared in main, AND
    //   - has no fun body or intrinsic in main yet.
    // These signatures keep lib bodies type-checkable when they call
    // siblings that were resolved into main from a different lib.
    std::unordered_set<std::string> mainExtNames;
    for (const auto &d: main.extDecls)
      mainExtNames.insert(d.name.name);
    for (auto &lib: libs) {
      for (auto it = lib.extDecls.begin(); it != lib.extDecls.end();) {
        if (it->contract) {
          ++it;
          continue;
        }
        if (mainFunNames.count(it->name.name) || mainIntrSeen.count(it->name.name) ||
            mainExtNames.count(it->name.name)) {
          it = lib.extDecls.erase(it);
          continue;
        }
        mainExtNames.insert(it->name.name);
        main.extDecls.push_back(std::move(*it));
        it = lib.extDecls.erase(it);
      }
    }

    // Pull in structs from libs that aren't already in main.
    std::unordered_set<std::string> mainStructNames;
    for (const auto &s: main.structs)
      mainStructNames.insert(s.name.name);
    for (auto &lib: libs) {
      for (auto it = lib.structs.begin(); it != lib.structs.end();) {
        if (!mainStructNames.count(it->name.name)) {
          main.structs.push_back(std::move(*it));
          mainStructNames.insert(main.structs.back().name.name);
          it = lib.structs.erase(it);
        } else {
          ++it;
        }
      }
    }

    // Pull in intrinsic decls from libs (lib funs may call them).
    std::unordered_set<std::string> mainIntrNames;
    for (const auto &i: main.intrinsics)
      mainIntrNames.insert(i.name.name);
    for (auto &lib: libs) {
      for (auto it = lib.intrinsics.begin(); it != lib.intrinsics.end();) {
        if (!mainIntrNames.count(it->name.name)) {
          main.intrinsics.push_back(*it);
          mainIntrNames.insert(it->name.name);
          it = lib.intrinsics.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

} // namespace refractir
