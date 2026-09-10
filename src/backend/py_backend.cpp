#include "backend/py_backend.hpp"
#include <cassert>
#include <stdexcept>
#include <unordered_set>
#include "analysis/dominators.hpp"
#include "analysis/loop_info.hpp"
#include "analysis/reducibility.hpp"
#include "analysis/structured_lowering.hpp"
#include "backend/vec_shapes.hpp"
#include "py_internal.hpp"

namespace refractir {

  namespace {

    // Semantics helpers emitted once per module. Python ints are
    // unbounded and its // and % floor, so every iN operation with UB
    // or wrapping rules goes through a checked helper; f32 arithmetic
    // round-trips through struct to round each operation to single
    // precision (RNE per IEEE 754).
    const char *kPreamble = R"PY(import math
import struct


class RefractIRTrap(Exception):
    pass


def _trap(msg):
    raise RefractIRTrap(msg)


def _ichk(v, n):
    if v < -(1 << (n - 1)) or v >= 1 << (n - 1):
        _trap("signed overflow (i%d)" % n)
    return v


def _iadd(a, b, n):
    return _ichk(a + b, n)


def _isub(a, b, n):
    return _ichk(a - b, n)


def _imul(a, b, n):
    return _ichk(a * b, n)


def _sdiv(a, b, n):
    if b == 0:
        _trap("division by zero")
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return _ichk(q, n)


def _srem(a, b, n):
    if b == 0:
        _trap("remainder by zero")
    if a == -(1 << (n - 1)) and b == -1:
        _trap("signed remainder overflow (INT_MIN % -1)")
    r = abs(a) % abs(b)
    return -r if a < 0 else r


def _shl(a, k, n):
    if k < 0 or k >= n:
        _trap("shift amount out of range")
    if a < 0:
        _trap("left shift of negative value")
    return _ichk(a << k, n)


def _ashr(a, k, n):
    if k < 0 or k >= n:
        _trap("shift amount out of range")
    return a >> k


def _lshr(a, k, n):
    if k < 0 or k >= n:
        _trap("shift amount out of range")
    r = (a & ((1 << n) - 1)) >> k
    return r - (1 << n) if r >= 1 << (n - 1) else r


def _cast_int(v, m):
    v &= (1 << m) - 1
    return v - (1 << m) if v >= 1 << (m - 1) else v


def _fin(x):
    if not math.isfinite(x):
        _trap("non-finite float result")
    return x


def _f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        _trap("f32 overflow")


def _fadd(a, b, n):
    r = a + b
    if n == 32:
        r = _f32(r)
    return _fin(r)


def _fsub(a, b, n):
    r = a - b
    if n == 32:
        r = _f32(r)
    return _fin(r)


def _fmul(a, b, n):
    r = a * b
    if n == 32:
        r = _f32(r)
    return _fin(r)


def _fdiv(a, b, n):
    if b == 0.0:
        _trap("float division by zero")
    r = a / b
    if n == 32:
        r = _f32(r)
    return _fin(r)


def _ffmod(a, b, n):
    if b == 0.0:
        _trap("float remainder by zero")
    q = a / b
    if n == 32:
        q = _f32(q)
    if not math.isfinite(q):
        _trap("float remainder intermediate overflow")
    r = math.fmod(a, b)
    if n == 32:
        r = _f32(r)
    return _fin(r)


def _f2i(x, n):
    if not math.isfinite(x):
        _trap("float-to-int of non-finite value")
    v = int(x)
    if v < -(1 << (n - 1)) or v >= 1 << (n - 1):
        _trap("float-to-int out of range")
    return v


class _Ptr:
    # A provenance-tracked pointer into a flat leaf-slot list: `off` is
    # the current leaf offset, `stride` the pointee's leaf count, and
    # [lo, hi) the extent of the innermost enclosing object (hi itself
    # is the legal one-past-end position). `frame` is the liveness cell of
    # the activation owning the storage, or None for a pointer that owns
    # none (null).
    __slots__ = ("buf", "off", "stride", "lo", "hi", "frame")

    def __init__(self, buf, off, stride, lo, hi, frame=None):
        self.buf = buf
        self.off = off
        self.stride = stride
        self.lo = lo
        self.hi = hi
        self.frame = frame


_NULL = _Ptr(None, 0, 0, 0, 0)
_UNDEF = ["undef"]  # unique identity sentinel
_PAD = ["pad"]  # interior byte of a wider leaf; never a valid access


def _live(p):
    # SPEC 7.5 rule 27. A returned activation's locals are dead, but the
    # list holding them outlives the call as long as a pointer references
    # it, so liveness is asked of the frame rather than observed of the
    # storage.
    if p.frame is not None and not p.frame[0]:
        _trap("access through a pointer to a returned activation")


def _rd(buf, off):
    v = buf[off]
    if v is _UNDEF:
        _trap("read of undef value")
    if v is _PAD:
        _trap("access to the interior of a value")
    return v


def _idx(i, n):
    if i < 0 or i >= n:
        _trap("array index out of bounds")
    return i


def _vrd(buf, off, n, stride):
    return [_rd(buf, off + k * stride) for k in range(n)]


def _padd(p, n):
    if p.buf is None:
        _trap("pointer arithmetic on null")
    off = p.off + n * p.stride
    if off < p.lo or off > p.hi:
        _trap("pointer arithmetic out of object bounds")
    return _Ptr(p.buf, off, p.stride, p.lo, p.hi, p.frame)


def _pdiff(p, q):
    if p.buf is None or q.buf is None or p.buf is not q.buf:
        _trap("cross-object pointer subtraction")
    return (p.off - q.off) // p.stride


def _peq(p, q):
    return p.buf is q.buf and p.off == q.off


def _prel(p, q):
    if p.buf is None or q.buf is None or p.buf is not q.buf:
        _trap("relational compare of cross-object pointers")
    return p.off - q.off


def _load(p):
    if p.buf is None:
        _trap("null pointer dereference")
    _live(p)
    if p.off < p.lo or p.off + p.stride > p.hi:
        _trap("pointer dereference out of bounds")
    return _rd(p.buf, p.off)


def _store(p, v):
    if p.buf is None:
        _trap("null pointer store")
    _live(p)
    if p.off < p.lo or p.off + p.stride > p.hi:
        _trap("pointer store out of bounds")
    p.buf[p.off] = v


def _pidx(p, i, n, estride):
    if p.buf is None:
        _trap("ptrindex on null pointer")
    _live(p)
    if p.off >= p.hi:
        _trap("ptrindex on one-past-end pointer")
    if i < 0 or i > n:
        _trap("ptrindex index out of range")
    return _Ptr(p.buf, p.off + i * estride, estride, p.off, p.off + n * estride, p.frame)


def _pfield(p, foff, flen, slen):
    # Provenance of a field pointer is the whole containing struct
    # (SPEC 7.5 rule 15): arithmetic may roam across sibling fields.
    if p.buf is None:
        _trap("ptrfield on null pointer")
    _live(p)
    if p.off >= p.hi:
        _trap("ptrfield on one-past-end pointer")
    return _Ptr(p.buf, p.off + foff, flen, p.off, p.off + slen, p.frame)
)PY";

    // --no-ub-guards preamble: the same value semantics as
    // kPreamble with every UB trap removed. Sound only for known-UB-free
    // programs (the dropped guards never fire on one). The arithmetic
    // operators are inlined at their call sites (see PyBackend::binInt
    // etc.), so no _iadd/_sdiv/_shl/… helpers are needed here; only the
    // normalization primitives (_f32, _cast_int) and the pointer/memory
    // model survive, guard-free. `_trap` is kept as a no-op so any
    // UB-precondition check (emitted unchanged) is inert. Non-UB failures
    // (@check_chksum mismatch, `require` violations, calls to
    // contract-form declarations) raise RefractIRTrap directly instead
    // of going through `_trap`, so they still fire in this mode.
    const char *kPreambleNoGuards = R"PY(import math
import struct


class RefractIRTrap(Exception):
    pass


def _trap(msg):
    pass


def _cast_int(v, m):
    v &= (1 << m) - 1
    return v - (1 << m) if v >= 1 << (m - 1) else v


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class _Ptr:
    # Same shape as the guarded build's, `frame` included, so a pointer
    # means the same thing either way; nothing here consults it.
    __slots__ = ("buf", "off", "stride", "lo", "hi", "frame")

    def __init__(self, buf, off, stride, lo, hi, frame=None):
        self.buf = buf
        self.off = off
        self.stride = stride
        self.lo = lo
        self.hi = hi
        self.frame = frame


_NULL = _Ptr(None, 0, 0, 0, 0)
_UNDEF = ["undef"]  # unique identity sentinel
_PAD = ["pad"]  # interior byte of a wider leaf; never a valid access


def _rd(buf, off):
    return buf[off]


def _idx(i, n):
    return i


def _vrd(buf, off, n, stride):
    return [buf[off + k * stride] for k in range(n)]


def _padd(p, n):
    return _Ptr(p.buf, p.off + n * p.stride, p.stride, p.lo, p.hi, p.frame)


def _pdiff(p, q):
    return (p.off - q.off) // p.stride


def _peq(p, q):
    return p.buf is q.buf and p.off == q.off


def _prel(p, q):
    return p.off - q.off


def _load(p):
    return p.buf[p.off]


def _store(p, v):
    p.buf[p.off] = v


def _pidx(p, i, n, estride):
    return _Ptr(p.buf, p.off + i * estride, estride, p.off, p.off + n * estride, p.frame)


def _pfield(p, foff, flen, slen):
    return _Ptr(p.buf, p.off + foff, flen, p.off, p.off + slen, p.frame)
)PY";

    const std::unordered_set<std::string> &pyKeywords() {
      static const std::unordered_set<std::string> kw = {
          "False",   "None",     "True",     "and",    "as",   "assert", "async",  "await",
          "break",   "class",    "continue", "def",    "del",  "elif",   "else",   "except",
          "finally", "for",      "from",     "global", "if",   "import", "in",     "is",
          "lambda",  "nonlocal", "not",      "or",     "pass", "raise",  "return", "try",
          "while",   "with",     "yield",    "match",  "case",
      };
      return kw;
    }

    std::string escapePyString(const std::string &s) {
      std::string out;
      for (char c: s) {
        if (c == '\\' || c == '"')
          out += '\\';
        out += c;
      }
      return out;
    }

  } // namespace

  std::string PyBackend::stripSigil(const std::string &name) {
    if (name.empty())
      return name;
    std::size_t start = 0;
    if (name[0] == '@' || name[0] == '%' || name[0] == '^') {
      start = 1;
      if (name.size() > 1 && name[1] == '?')
        start = 2;
    }
    return name.substr(start);
  }

  std::string PyBackend::mangleFun(const std::string &name) const {
    if (noMainMangle_ && name == "@main")
      return "main";
    return "refractir_" + stripSigil(name);
  }

  std::string PyBackend::symCall(const std::string &symName) const {
    // Provider convention shared with the C backend's extern symbols:
    // <fun>__<sym>, sigils stripped. The embedding defines these in
    // the module globals before calling the entry function.
    return stripSigil(curFuncName_) + "__" + stripSigil(symName) + "()";
  }

  std::string PyBackend::pyLocal(const std::string &sigiled) {
    auto it = pyNames_.find(sigiled);
    assert(it != pyNames_.end() && "local not in the per-function name map");
    return it->second;
  }

  void PyBackend::buildNameMap(const FunDecl &f, const ControlTree &tree) {
    pyNames_.clear();
    takenNames_.clear();
    // Guard flags own their names verbatim; user locals starting with
    // '_' are renamed below, so the namespaces cannot collide.
    for (const auto &flag: tree.flagNames)
      takenNames_.insert(flag);
    auto add = [&](const std::string &sigiled) {
      std::string base = stripSigil(sigiled);
      if (base.empty() || base[0] == '_' || pyKeywords().count(base))
        base = "v_" + base;
      while (takenNames_.count(base))
        base += "_";
      takenNames_.insert(base);
      pyNames_.emplace(sigiled, base);
    };
    for (const auto &p: f.params)
      add(p.name.name);
    for (const auto &l: f.lets)
      add(l.name.name);
  }

  void PyBackend::line(const std::string &s) {
    for (int i = 0; i < indent_; ++i)
      out_ << "    ";
    out_ << s << "\n";
    ++stmtCount_;
  }

  void PyBackend::comment(const std::string &s) {
    for (int i = 0; i < indent_; ++i)
      out_ << "    ";
    out_ << "# " << s << "\n";
    // Comments don't count as suite statements ("pass" still needed).
  }

  void PyBackend::emit(const Program &prog) {
    prog_ = &prog;
    structFields_.clear();
    for (const auto &s: prog.structs) {
      auto &fields = structFields_[s.name.name];
      for (const auto &f: s.fields)
        fields.emplace_back(f.name, f.type);
    }
    // Vector-lowering strategy: storage form of vector
    // locals. Default to "array" (plain lane lists — the historical
    // representation); struct strategies emit per-shape classes after
    // the preamble. The stamp mirrors the C backend's
    // `// vec-lowering:` traceability comment.
    if (!vecLowering_)
      vecLowering_ = makePyVecLowering("array");
    out_ << "# Generated by symirc --target python\n";
    out_ << "# vec-lowering: " << vecLowering_->name() << "\n"
         << (noUbGuards_ ? kPreambleNoGuards : kPreamble);
    vecLowering_->emitPreamble(out_, collectVecShapes(prog));
    emitIntrinsicHelpers(prog);
    // External declarations: a link-form decl resolved via -I carries
    // its body, which emits like a normal function (the python module
    // is self-contained — no separate link step). A contract-form decl
    // has no body by design; it emits a stub that traps if reached at
    // runtime (the C analogue is an unresolved extern at link time).
    for (const auto &d: prog.extDecls) {
      if (d.resolvedBody) {
        out_ << "\n\n";
        emitFunction(*d.resolvedBody);
      } else {
        out_ << "\n\ndef " << mangleFun(d.name.name) << "(";
        for (std::size_t i = 0; i < d.params.size(); ++i)
          out_ << (i ? ", " : "") << "_p" << i;
        out_ << "):\n    raise RefractIRTrap(\"call to external declaration " << d.name.name
             << "\")\n";
      }
    }
    for (const auto &f: prog.funs) {
      out_ << "\n\n";
      emitFunction(f);
    }

    // Module driver: --emit-main output is directly
    // executable (`python3 prog.py`), mirroring the C target where
    // main() is the process entry point. main()'s i32 becomes the
    // exit status (truncated mod 256 by the OS, exactly like a C
    // binary). Embedders stay inert: importing the module (or the
    // test driver's exec with bare globals) never trips the guard.
    // A parameterized @main gets no driver — nothing could supply
    // its arguments.
    if (noMainMangle_) {
      for (const auto &f: prog.funs) {
        if (f.name.name == "@main" && f.params.empty()) {
          out_ << "\n\nif __name__ == \"__main__\":\n"
               << "    import sys\n\n"
               << "    sys.exit(main())\n";
          break;
        }
      }
    }
  }

  void PyBackend::emitFunction(const FunDecl &f) {
    curFuncName_ = f.name.name;
    varTypes_.clear();
    for (const auto &p: f.params) {
      requireSupportedType(p.type, "parameter");
      varTypes_[p.name.name] = p.type;
    }
    for (const auto &s: f.syms) {
      requireSupportedType(s.type, "symbol");
      varTypes_[s.name.name] = s.type;
    }
    for (const auto &l: f.lets) {
      requireSupportedType(l.type, "local");
      varTypes_[l.name.name] = l.type;
    }

    collectBoxedRoots(f);

    // The pipeline (ReducibilityCheck included for this target) has
    // validated the function, so the structuring below is total.
    DiagBag diags;
    CFG cfg = CFG::build(f, diags);
    DomTree dt = DomTree::build(cfg);
    LoopInfo li = LoopInfo::build(cfg, dt);
    ControlTree tree = StructuredLowering::run(Structurizer::build(f, cfg, dt, li), f, cfg);

    buildNameMap(f, tree);

    std::string sig = "def " + mangleFun(f.name.name) + "(";
    for (std::size_t i = 0; i < f.params.size(); ++i) {
      if (i)
        sig += ", ";
      sig += pyLocal(f.params[i].name.name);
    }
    sig += "):";
    line(sig);
    stmtCount_ = 0;
    suite([&] {
      // Vector params arrive as lane lists (the function-boundary
      // ABI); convert to the strategy's storage form on entry.
      for (const auto &p: f.params) {
        if (auto *vt = p.type ? std::get_if<VecType>(&p.type->v) : nullptr) {
          std::string unpack = vecLowering_->unpackParam(pyLocal(p.name.name), *vt);
          if (!unpack.empty())
            line(unpack);
        }
      }
      const auto emitBody = [&] {
        for (const auto &flag: tree.flagNames)
          line(flag + " = False");
        for (const auto &l: f.lets)
          emitLet(l);
        if (tree.root)
          emitNode(tree, *tree.root, f, cfg);
      };
      if (!takesAddress_) {
        emitBody();
        return;
      }
      // The storage behind a pointer this function hands out is a Python
      // list, which an escaped pointer keeps alive well past the return.
      // The frame says what the object graph cannot: that the activation
      // owning it is gone (SPEC §7.5 rule 27). `finally` so a trap unwinding
      // through here closes the frame too, and so the return expression is
      // evaluated while the frame is still open.
      line(std::string(kFrameCell) + " = [True]");
      line("try:");
      suite(emitBody);
      line("finally:");
      suite([&] { line(std::string(kFrameCell) + "[0] = False"); });
    });
    stmtCount_ = 0;
    indent_ = 0;
  }

  void PyBackend::emitLet(const LetDecl &l) {
    const std::string name = pyLocal(l.name.name);
    const bool boxed = boxedRoots_.count(l.name.name) > 0;
    const bool aggregate = l.type && (std::holds_alternative<ArrayType>(l.type->v) ||
                                      std::holds_alternative<StructType>(l.type->v) ||
                                      std::holds_alternative<VecType>(l.type->v));
    if (aggregate) {
      // Vector locals go through the vec-lowering strategy; arrays and
      // structs are always flat leaf-slot lists (the memory model).
      if (auto *vt = std::get_if<VecType>(&l.type->v)) {
        if (!l.init || l.init->kind == InitVal::Kind::Undef) {
          line(vecLowering_->declUndef(name, *vt));
        } else if (l.init->kind == InitVal::Kind::Local) {
          // Whole-vector copy init (vectors are value types).
          line(vecLowering_->wholeCopy(name, pyLocal(std::get<LocalId>(l.init->value).name), *vt));
        } else if (l.init->kind == InitVal::Kind::Sym) {
          // Provider call: already a fresh lane list.
          line(vecLowering_->assignFromList(
              name, *vt, "list(" + symCall(std::get<SymId>(l.init->value).name) + ")"
          ));
        } else {
          line(vecLowering_->assignFromList(name, *vt, flattenInit(*l.init, l.type)));
        }
        return;
      }
      if (!l.init || l.init->kind == InitVal::Kind::Undef) {
        line(name + " = [_UNDEF] * " + std::to_string(byteSize(l.type)));
      } else {
        line(name + " = " + flattenInit(*l.init, l.type));
      }
      return;
    }
    if (boxed) {
      // Address-taken scalar: a one-slot box.
      if (!l.init || l.init->kind == InitVal::Kind::Undef)
        line(name + " = [_UNDEF]");
      else
        line(name + " = [" + scalarInit(*l.init, l.type) + "]");
      return;
    }
    if (!l.init || l.init->kind == InitVal::Kind::Undef) {
      // Definite-init guarantees no read before the first assignment.
      return;
    }
    line(name + " = " + scalarInit(*l.init, l.type));
  }

  std::string PyBackend::condText(const FunDecl &f, std::size_t block, bool negate) {
    const auto *br = std::get_if<BrTerm>(&f.blocks[block].term);
    assert(br && br->isConditional && br->cond && "structured condition on a non-branch block");
    std::string s = condStr(*br->cond);
    return negate ? "not (" + s + ")" : s;
  }

  void PyBackend::emitBlockInstrs(const Block &b) {
    if (!b.instrs.empty())
      comment(b.label.name);
    for (const auto &ins: b.instrs) {
      std::visit(
          [this](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, AssignInstr>) {
              emitAssign(arg);
            } else if constexpr (std::is_same_v<T, AssumeInstr>) {
              comment("assume " + condStr(arg.cond));
            } else if constexpr (std::is_same_v<T, RequireInstr>) {
              if (!noRequire_) {
                std::string msg = "require violation";
                if (arg.message)
                  msg += ": " + escapePyString(*arg.message);
                // Not a UB guard (`--no-require` governs this separately):
                // raise directly so the violation still fires with
                // --no-ub-guards, matching the C (assert) and WASM
                // (unconditional unreachable) lowerings.
                line("if not (" + condStr(arg.cond) + "): raise RefractIRTrap(\"" + msg + "\")");
              }
            } else {
              static_assert(std::is_same_v<T, StoreInstr>);
              emitStore(arg);
            }
          },
          ins
      );
    }
  }

  void PyBackend::emitNode(
      const ControlTree &tree, const ControlTree::Node &n, const FunDecl &f, const CFG &cfg
  ) {
    std::visit(
        [&](const auto &node) {
          using T = std::decay_t<decltype(node)>;
          if constexpr (std::is_same_v<T, ControlTree::Seq>) {
            for (const auto &item: node.items)
              if (item)
                emitNode(tree, *item, f, cfg);
          } else if constexpr (std::is_same_v<T, ControlTree::BlockStmts>) {
            emitBlockInstrs(f.blocks[node.block]);
          } else if constexpr (std::is_same_v<T, ControlTree::If>) {
            line("if " + condText(f, node.block, node.negate) + ":");
            suite([&] {
              if (node.thenBr)
                emitNode(tree, *node.thenBr, f, cfg);
            });
            if (node.elseBr) {
              line("else:");
              suite([&] { emitNode(tree, *node.elseBr, f, cfg); });
            }
          } else if constexpr (std::is_same_v<T, ControlTree::Loop>) {
            line("while True:");
            suite([&] { emitNode(tree, *node.body, f, cfg); });
          } else if constexpr (std::is_same_v<T, ControlTree::CondLoop>) {
            line("while " + condText(f, node.header, node.negate) + ":");
            suite([&] { emitNode(tree, *node.body, f, cfg); });
          } else if constexpr (std::is_same_v<T, ControlTree::DoWhile>) {
            // Python has no do-while; re-expand to the exact
            // pre-peephole form (`while True:` + tail `if c: break`).
            line("while True:");
            suite([&] {
              if (node.body)
                emitNode(tree, *node.body, f, cfg);
              line("if " + condText(f, node.latch, !node.negate) + ":");
              suite([&] { line("break"); });
            });
          } else if constexpr (std::is_same_v<T, ControlTree::Break>) {
            assert(node.levels == 1 && "multi-level break survived lowering");
            line("break");
          } else if constexpr (std::is_same_v<T, ControlTree::Continue>) {
            assert(node.levels == 0 && "multi-level continue survived lowering");
            line("continue");
          } else if constexpr (std::is_same_v<T, ControlTree::SetFlag>) {
            line(tree.flagNames[node.flag] + " = True");
          } else if constexpr (std::is_same_v<T, ControlTree::FlagBreak>) {
            const std::string &flag = tree.flagNames[node.flag];
            if (node.isFinal) {
              line("if " + flag + ":");
              suite([&] {
                line(flag + " = False");
                line("break");
              });
            } else {
              line("if " + flag + ": break");
            }
          } else if constexpr (std::is_same_v<T, ControlTree::FlagContinue>) {
            const std::string &flag = tree.flagNames[node.flag];
            line("if " + flag + ":");
            suite([&] {
              line(flag + " = False");
              line("continue");
            });
          } else if constexpr (std::is_same_v<T, ControlTree::Guarded>) {
            std::string cond;
            for (int fl: node.flags) {
              if (!cond.empty())
                cond += " and ";
              cond += "not " + tree.flagNames[fl];
            }
            line("if " + cond + ":");
            suite([&] { emitNode(tree, *node.body, f, cfg); });
          } else if constexpr (std::is_same_v<T, ControlTree::ResetFlag>) {
            line(tree.flagNames[node.flag] + " = False");
          } else if constexpr (std::is_same_v<T, ControlTree::Return>) {
            const auto *ret = std::get_if<RetTerm>(&f.blocks[node.block].term);
            assert(ret && "Return node on a block without ret");
            if (ret->value) {
              F32Guard ctx(f32Ctx_, !containsF64(f.retType));
              line("return " + exprStr(*ret->value));
            } else {
              line("return");
            }
          } else if constexpr (std::is_same_v<T, ControlTree::Trap>) {
            line("_trap(\"unreachable executed\")");
          } else {
            // FallThrough / JumpJoin are eliminated by StructuredLowering.
            assert(false && "unlowered transfer node reached the python emitter");
          }
        },
        n.v
    );
  }

} // namespace refractir
