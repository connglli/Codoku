#include "backend/wasm_backend.hpp"
#include <algorithm>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include "analysis/type_utils.hpp"
#include "backend/wasm_vec_lowering.hpp"
#include "wasm_internal.hpp"

namespace refractir {

  void WasmBackend::setVecLowering(std::unique_ptr<WasmVecLowering> vl) {
    vecLowering_ = std::move(vl);
  }

  void WasmBackend::indent() {
    for (int i = 0; i < indent_level_; ++i)
      out_ << "  ";
  }

  std::string WasmBackend::mangleName(const std::string &name) {
    if (name.empty())
      return name;
    size_t start = 0;
    if (name[0] == '@' || name[0] == '%' || name[0] == '^') {
      start = 1;
      if (name.size() > 1 && name[1] == '?')
        start = 2;
    }
    return "$" + name.substr(start);
  }

  std::string WasmBackend::stripSigil(const std::string &name) {
    if (name.empty())
      return name;
    size_t start = 0;
    if (name[0] == '@' || name[0] == '%' || name[0] == '^') {
      start = 1;
      if (name.size() > 1 && name[1] == '?')
        start = 2;
    }
    return name.substr(start);
  }

  std::string
  WasmBackend::getMangledSymbolName(const std::string &funcName, const std::string &symName) {
    return stripSigil(funcName) + "__" + stripSigil(symName);
  }

  std::string WasmBackend::getWasmType(const TypePtr &type) {
    if (!type)
      return "";
    return std::visit(
        [](auto &&arg) -> std::string {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, IntType>) {
            if (arg.kind == IntType::Kind::I64 || (arg.bits.has_value() && *arg.bits > 32))
              return "i64";
            return "i32";
          } else if constexpr (std::is_same_v<T, FloatType>) {
            return arg.kind == FloatType::Kind::F32 ? "f32" : "f64";
          } else {
            return "i32";
          }
        },
        type->v
    );
  }

  std::uint32_t WasmBackend::getIntWidth(const TypePtr &type) {
    if (!type)
      return 32;
    if (auto it = std::get_if<IntType>(&type->v)) {
      if (it->kind == IntType::Kind::I32)
        return 32;
      if (it->kind == IntType::Kind::I64)
        return 64;
      return it->bits.value_or(32);
    }
    if (auto ft = std::get_if<FloatType>(&type->v)) {
      return (ft->kind == FloatType::Kind::F32) ? 32 : 64;
    }
    return 32;
  }

  std::uint32_t WasmBackend::getTypeSize(const TypePtr &type) {
    if (!type)
      return 0;
    return std::visit(
        [this](auto &&arg) -> std::uint32_t {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, IntType>) {
            std::uint32_t bits = 32;
            if (arg.kind == IntType::Kind::I32)
              bits = 32;
            else if (arg.kind == IntType::Kind::I64)
              bits = 64;
            else
              bits = arg.bits.value_or(32);
            if (bits <= 8)
              return 1;
            if (bits <= 16)
              return 2;
            if (bits <= 32)
              return 4;
            return 8;
          } else if constexpr (std::is_same_v<T, FloatType>) {
            return arg.kind == FloatType::Kind::F32 ? 4 : 8;
          } else if constexpr (std::is_same_v<T, StructType>) {
            if (structLayouts_.count(arg.name.name))
              return structLayouts_.at(arg.name.name).totalSize;
            return 0;
          } else if constexpr (std::is_same_v<T, ArrayType>) {
            return arg.size * getTypeSize(arg.elem);
          } else if constexpr (std::is_same_v<T, PtrType>) {
            return 4; // WASM pointers are 32-bit (4 bytes)
          } else if constexpr (std::is_same_v<T, VecType>) {
            return arg.size * getTypeSize(arg.elem);
          }
          return 0;
        },
        type->v
    );
  }

  void WasmBackend::computeLayouts(const Program &prog) {
    for (const auto &s: prog.structs) {
      StructInfo info;
      std::uint32_t offset = 0;
      for (const auto &f: s.fields) {
        std::uint32_t size = getTypeSize(f.type);
        if (size >= 8 && offset % 8 != 0)
          offset += 8 - (offset % 8);
        else if (size >= 4 && offset % 4 != 0)
          offset += 4 - (offset % 4);

        info.fields[f.name] = {offset, size, f.type};
        info.fieldNames.push_back(f.name);
        offset += size;
      }
      if (offset % 8 != 0)
        offset += 8 - (offset % 8);
      info.totalSize = offset;
      structLayouts_[s.name.name] = info;
    }
  }

  void WasmBackend::emitMask(std::uint32_t bitwidth, std::uint32_t wasmWidth) {
    if (bitwidth >= wasmWidth)
      return;
    if (wasmWidth == 32) {
      uint32_t mask = (1ULL << bitwidth) - 1;
      indent();
      out_ << "i32.const " << mask << "\n";
      indent();
      out_ << "i32.and\n";
    } else {
      uint64_t mask = (1ULL << bitwidth) - 1;
      indent();
      out_ << "i64.const " << mask << "\n";
      indent();
      out_ << "i64.and\n";
    }
  }

  void WasmBackend::emitSignExtend(std::uint32_t fromWidth, std::uint32_t toWidth) {
    if (fromWidth == toWidth)
      return;
    if (toWidth <= 32) {
      indent();
      out_ << "i32.const " << (32 - fromWidth) << "\n";
      indent();
      out_ << "i32.shl\n";
      indent();
      out_ << "i32.const " << (32 - fromWidth) << "\n";
      indent();
      out_ << "i32.shr_s\n";
    } else {
      if (fromWidth < 32) {
        emitSignExtend(fromWidth, 32);
        indent();
        out_ << "i64.extend_i32_s\n";
      } else if (fromWidth == 32) {
        indent();
        out_ << "i64.extend_i32_s\n";
      } else {
        indent();
        out_ << "i64.const " << (64 - fromWidth) << "\n";
        indent();
        out_ << "i64.shl\n";
        indent();
        out_ << "i64.const " << (64 - fromWidth) << "\n";
        indent();
        out_ << "i64.shr_s\n";
      }
    }
  }

  void WasmBackend::emitCopy(
      const TypePtr &type, std::uint32_t dstOffset, const std::string &srcName,
      std::uint32_t srcOffset
  ) {
    if (auto at = std::get_if<ArrayType>(&type->v)) {
      std::uint32_t elemSize = getTypeSize(at->elem);
      for (std::uint64_t i = 0; i < at->size; ++i) {
        emitCopy(at->elem, dstOffset - i * elemSize, srcName, srcOffset - i * elemSize);
      }
    } else if (auto vt = std::get_if<VecType>(&type->v)) {
      std::uint32_t elemSize = getTypeSize(vt->elem);
      for (std::uint64_t i = 0; i < vt->size; ++i) {
        emitCopy(vt->elem, dstOffset - i * elemSize, srcName, srcOffset - i * elemSize);
      }
    } else if (auto st = std::get_if<StructType>(&type->v)) {
      if (structLayouts_.count(st->name.name)) {
        const auto &sinfo = structLayouts_.at(st->name.name);
        for (const auto &fname: sinfo.fieldNames) {
          const auto &finfo = sinfo.fields.at(fname);
          emitCopy(finfo.type, dstOffset - finfo.offset, srcName, srcOffset - finfo.offset);
        }
      }
    } else {
      // Leaf copy
      indent();
      out_ << "local.get $__old_sp\n";
      indent();
      out_ << "i32.const " << dstOffset << "\n";
      indent();
      out_ << "i32.sub\n";

      // Load source
      const auto &srcInfo = locals_.at(srcName);
      if (srcInfo.isAggregate) {
        indent();
        out_ << "local.get $__old_sp\n";
        indent();
        out_ << "i32.const " << srcOffset << "\n";
        indent();
        out_ << "i32.sub\n";

        std::uint32_t width = 0;
        bool valIsFloat = false;
        if (auto bits = TypeUtils::getIntBitWidth(type)) {
          width = *bits;
        } else if (type && std::holds_alternative<FloatType>(type->v)) {
          valIsFloat = true;
          width = (std::get<FloatType>(type->v).kind == FloatType::Kind::F32) ? 32 : 64;
        } else if (type && std::holds_alternative<PtrType>(type->v)) {
          width = 32;
        }

        indent();
        if (valIsFloat) {
          out_ << (width == 32 ? "f32.load\n" : "f64.load\n");
        } else {
          out_ << (width <= 8
                       ? "i32.load8_u"
                       : (width <= 16 ? "i32.load16_u" : (width <= 32 ? "i32.load" : "i64.load")))
               << "\n";
        }
      } else {
        indent();
        out_ << "local.get " << mangleName(srcName) << "\n";
      }

      // Store destination
      std::uint32_t width = 0;
      bool valIsFloat = false;
      if (auto bits = TypeUtils::getIntBitWidth(type)) {
        width = *bits;
      } else if (type && std::holds_alternative<FloatType>(type->v)) {
        valIsFloat = true;
        width = (std::get<FloatType>(type->v).kind == FloatType::Kind::F32) ? 32 : 64;
      } else if (type && std::holds_alternative<PtrType>(type->v)) {
        width = 32;
      }

      indent();
      if (valIsFloat) {
        out_ << (width == 32 ? "f32.store\n" : "f64.store\n");
      } else {
        out_ << (width <= 8
                     ? "i32.store8"
                     : (width <= 16 ? "i32.store16" : (width <= 32 ? "i32.store" : "i64.store")))
             << "\n";
      }
    }
  }

  // intrinsicHelperName and emitIntrinsicHelper are implemented in
  // src/backend/wasm_intrinsics.cpp — the single source of truth for
  // WASM-side intrinsic code generation. See that file to add new intrinsics.


  void WasmBackend::emit(const Program &prog) {
    prog_ = &prog;
    computeLayouts(prog);
    if (!vecLowering_)
      vecLowering_ = makeWasmVecLowering("vecext");
    if (!noModuleTags_) {
      out_ << "(module\n";
      indent_level_++;
    }
    indent();
    out_ << ";; vec-lowering: " << vecLowering_->name() << "\n";

    for (const auto &f: prog.funs) {
      for (const auto &s: f.syms) {
        if (auto vt = std::get_if<VecType>(&s.type->v)) {
          for (std::uint64_t i = 0; i < vt->size; ++i) {
            indent();
            out_ << "(import \"" << stripSigil(f.name.name) << "\" \"" << stripSigil(s.name.name)
                 << "__" << i << "\" (func "
                 << mangleName(getMangledSymbolName(f.name.name, s.name.name)) << "__" << i
                 << " (result " << getWasmType(vt->elem) << ")))\n";
          }
        } else {
          indent();
          out_ << "(import \"" << stripSigil(f.name.name) << "\" \"" << stripSigil(s.name.name)
               << "\" (func " << mangleName(getMangledSymbolName(f.name.name, s.name.name))
               << " (result " << getWasmType(s.type) << ")))\n";
        }
      }
    }

    // Import link-form decls FIRST — WAT requires all (import ...)
    // declarations to precede globals and function definitions in the module.
    for (const auto &d: prog.extDecls) {
      indent();
      out_ << "(import \"\" \"" << stripSigil(d.name.name) << "\" (func "
           << mangleName(d.name.name);
      for (const auto &p: d.params) {
        out_ << " (param " << getWasmType(p.type) << ")";
        (void) p;
      }
      out_ << " (result " << getWasmType(d.retType) << ")))\n";
    }

    indent();
    out_ << "(memory 16)\n"; // 1MB
    indent();
    out_ << "(global $__stack_pointer (mut i32) (i32.const 1048576))\n";

    // Emit intrinsic helper functions (these are func defs, not imports).
    for (const auto &intr: prog.intrinsics) {
      emitIntrinsicHelper(intr);
    }

    for (const auto &f: prog.funs) {
      curFuncName_ = f.name.name;
      locals_.clear();
      syms_.clear();
      callVecScratch_.clear();
      stackSize_ = 0;

      // Pre-scan: collect variables whose address is taken (must be spilled to shadow stack)
      std::unordered_set<std::string> addrTaken;
      std::function<void(const Expr &)> scanExpr;
      std::function<void(const Atom &)> scanAtom;

      scanAtom = [&](const Atom &a) {
        std::visit(
            [&](auto &&arg) {
              using T = std::decay_t<decltype(arg)>;
              if constexpr (std::is_same_v<T, AddrAtom>) {
                addrTaken.insert(arg.lv.base.name);
              } else if constexpr (std::is_same_v<T, SelectAtom>) {
                if (arg.cond) {
                  scanExpr(arg.cond->lhs);
                  scanExpr(arg.cond->rhs);
                }
                if (arg.maskExpr) {
                  scanExpr(*arg.maskExpr);
                }
              } else if constexpr (std::is_same_v<T, CallAtom>) {
                // §2.12 A call argument may itself contain
                // `addr <lv>`, which must spill <lv> onto the shadow
                // stack. Recurse into every argument expression.
                for (const auto &ap: arg.args) {
                  scanExpr(*ap);
                }
                // Size this call site's scratch block for
                // vectors crossing the boundary (spilled args + sret).
                VecCallSlots slots = vecCallSlots(arg);
                if (slots.totalBytes > 0) {
                  if (stackSize_ % 8 != 0)
                    stackSize_ += 8 - (stackSize_ % 8);
                  stackSize_ += slots.totalBytes;
                  callVecScratch_[&arg] = stackSize_;
                }
              }
            },
            a.v
        );
      };

      scanExpr = [&](const Expr &e) {
        scanAtom(e.first);
        for (const auto &t: e.rest) {
          scanAtom(t.atom);
        }
      };

      auto scanInitVal = [&](auto &self, const InitVal &iv) -> void {
        if (iv.kind == InitVal::Kind::Atom) {
          scanAtom(*std::get<AtomPtr>(iv.value));
        } else if (iv.kind == InitVal::Kind::Aggregate) {
          const auto &elements = std::get<std::vector<InitValPtr>>(iv.value);
          for (const auto &el: elements) {
            self(self, *el);
          }
        }
      };

      for (const auto &l: f.lets) {
        if (l.init) {
          scanInitVal(scanInitVal, *l.init);
        }
      }

      for (const auto &b: f.blocks) {
        for (const auto &ins: b.instrs) {
          std::visit(
              [&](auto &&instr) {
                using IT = std::decay_t<decltype(instr)>;
                if constexpr (std::is_same_v<IT, AssignInstr>) {
                  scanExpr(instr.rhs);
                } else if constexpr (std::is_same_v<IT, StoreInstr>) {
                  scanExpr(instr.ptr);
                  scanExpr(instr.val);
                } else if constexpr (std::is_same_v<IT, AssumeInstr>) {
                  // assume cond may contain `call @f(addr %x)`
                  scanExpr(instr.cond.lhs);
                  scanExpr(instr.cond.rhs);
                } else if constexpr (std::is_same_v<IT, RequireInstr>) {
                  scanExpr(instr.cond.lhs);
                  scanExpr(instr.cond.rhs);
                }
              },
              ins
          );
        }
        if (auto rt = std::get_if<RetTerm>(&b.term)) {
          if (rt->value) {
            scanExpr(*rt->value);
          }
        } else if (auto bt = std::get_if<BrTerm>(&b.term)) {
          // conditional br's predicate may contain `call`.
          if (bt->isConditional && bt->cond) {
            scanExpr(bt->cond->lhs);
            scanExpr(bt->cond->rhs);
          }
        }
      }

      for (const auto &s: f.syms) {
        syms_[s.name.name] = s.type;
      }

      for (const auto &p: f.params) {
        locals_[p.name.name] = {getWasmType(p.type), true, getIntWidth(p.type), false, 0, p.type};
      }
      for (const auto &l: f.lets) {
        // Mark as aggregate if it's a struct/array, a vector under a
        // frame-memory vec-lowering strategy, OR if its address is taken
        // (needs a memory slot)
        bool isAgg =
            std::holds_alternative<StructType>(l.type->v) ||
            std::holds_alternative<ArrayType>(l.type->v) ||
            (std::holds_alternative<VecType>(l.type->v) && vecLowering_->usesFrameMemory()) ||
            addrTaken.count(l.name.name);
        if (isAgg) {
          std::uint32_t size = getTypeSize(l.type);
          if (stackSize_ % 8 != 0)
            stackSize_ += 8 - (stackSize_ % 8);
          stackSize_ += size;
          locals_[l.name.name] = {"i32", false, getIntWidth(l.type), true, stackSize_, l.type};
        } else {
          locals_[l.name.name] = {
              getWasmType(l.type), false, getIntWidth(l.type), false, 0, l.type
          };
        }
      }

      indent();
      out_ << "(func " << mangleName(f.name.name);
      for (const auto &p: f.params) {
        out_ << " (param " << mangleName(p.name.name) << " " << getWasmType(p.type) << ")";
      }
      // Vector returns go through a hidden trailing sret address
      // param instead of a WASM result — the callee writes the lanes into
      // caller-owned frame memory.
      bool retVec = f.retType && std::holds_alternative<VecType>(f.retType->v);
      if (retVec) {
        out_ << " (param $__sret i32)";
      } else if (f.retType) {
        out_ << " (result " << getWasmType(f.retType) << ")";
      }
      out_ << "\n";
      indent_level_++;

      indent();
      out_ << "(local $__pc i32)\n";
      indent();
      out_ << "(local $__old_sp i32)\n";
      indent();
      out_ << "(local $__ptr_temp i32)\n"; // scratch register for null-checked ptr ops
      indent();
      out_ << "(local $__idx_temp i32)\n"; // scratch register for index bounds checks
      indent();
      // Scratch FP scalars used by the inline fmod expansion to
      // hold the intermediate quotient `x/y` while we trap on non-finite
      // results (spec §2.9 + §7.4 rule 6).  Always emitted alongside the
      // other __* scratch slots so every function has them — the WASM
      // tooling tolerates unused locals.
      out_ << "(local $__fmod_q_f32 f32)\n";
      indent();
      out_ << "(local $__fmod_q_f64 f64)\n";
      vecLowering_->declareFuncScratch(*this);
      for (const auto &l: f.lets) {
        if (!locals_[l.name.name].isAggregate) {
          if (std::holds_alternative<VecType>(l.type->v)) {
            // Register vec-lowering strategy: the strategy owns
            // the backing locals for this vector.
            vecLowering_->declareLocals(*this, l.name.name, std::get<VecType>(l.type->v));
          } else {
            indent();
            out_ << "(local " << mangleName(l.name.name) << " " << locals_[l.name.name].wasmType
                 << ")\n";
          }
        }
      }
      if (!vecLowering_->usesFrameMemory()) {
        // Backing locals for vector params (unpacked from the pointer
        // ABI below, once the shadow-stack prologue has run).
        for (const auto &p: f.params)
          if (std::holds_alternative<VecType>(p.type->v))
            vecLowering_->declareLocals(*this, p.name.name, std::get<VecType>(p.type->v));
      }

      if (stackSize_ > 0) {
        indent();
        out_ << "global.get $__stack_pointer\n";
        indent();
        out_ << "local.set $__old_sp\n";
        indent();
        out_ << "global.get $__stack_pointer\n";
        indent();
        out_ << "i32.const " << stackSize_ << "\n";
        indent();
        out_ << "i32.sub\n";
        indent();
        out_ << "global.set $__stack_pointer\n";
      }

      if (!vecLowering_->usesFrameMemory()) {
        // Unpack pointer-ABI vector params into strategy storage.
        for (const auto &p: f.params)
          if (std::holds_alternative<VecType>(p.type->v))
            vecLowering_->unpackParam(*this, p.name.name, std::get<VecType>(p.type->v));
      }

      for (const auto &l: f.lets) {
        if (l.init) {
          if (locals_[l.name.name].isAggregate) {
            emitInitVal(*l.init, l.type, locals_[l.name.name].offset);
          } else if (std::holds_alternative<VecType>(l.type->v)) {
            // Register-strategy vector local.
            emitVecLocalInit(l.name.name, std::get<VecType>(l.type->v), *l.init);
          } else if (l.init->kind == InitVal::Kind::Int) {
            indent();
            bool isTargetFloat = std::holds_alternative<FloatType>(l.type->v);
            if (isTargetFloat) {
              out_ << (locals_[l.name.name].bitwidth <= 32 ? "f32.const " : "f64.const ")
                   << std::get<IntLit>(l.init->value).value << ".0\n";
            } else {
              out_ << (locals_[l.name.name].bitwidth <= 32 ? "i32.const " : "i64.const ")
                   << std::get<IntLit>(l.init->value).value << "\n";
              emitSignExtend(
                  getIntWidth(l.type), (locals_[l.name.name].wasmType == "i32" ? 32 : 64)
              );
            }
            indent();
            out_ << "local.set " << mangleName(l.name.name) << "\n";
          } else if (l.init->kind == InitVal::Kind::Float) {
            indent();
            out_ << (locals_[l.name.name].wasmType == "f32" ? "f32.const " : "f64.const ")
                 << formatFloatLit(std::get<FloatLit>(l.init->value).value) << "\n";
            indent();
            out_ << "local.set " << mangleName(l.name.name) << "\n";
          } else if (l.init->kind == InitVal::Kind::Null) {
            // null pointer = i32 0 in WASM
            indent();
            out_ << "i32.const 0\n";
            indent();
            out_ << "local.set " << mangleName(l.name.name) << "\n";
          } else if (l.init->kind == InitVal::Kind::Local) {
            emitLValue({std::get<LocalId>(l.init->value), {}, l.init->span}, false);
            indent();
            out_ << "local.set " << mangleName(l.name.name) << "\n";
          } else if (l.init->kind == InitVal::Kind::Sym) {
            const auto &sid = std::get<SymId>(l.init->value);
            indent();
            out_ << "call " << mangleName(getMangledSymbolName(curFuncName_, sid.name)) << "\n";
            // Handle int extension if needed
            std::uint32_t srcWidth = 32;
            bool srcIsFloat = false;
            if (syms_.count(sid.name)) {
              srcWidth = getIntWidth(syms_.at(sid.name));
              if (std::holds_alternative<FloatType>(syms_.at(sid.name)->v))
                srcIsFloat = true;
            }
            if (!srcIsFloat) {
              if (srcWidth <= 32 && getIntWidth(l.type) > 32) {
                indent();
                out_ << "i64.extend_i32_s\n";
              } else if (srcWidth > 32 && getIntWidth(l.type) <= 32) {
                indent();
                out_ << "i32.wrap_i64\n";
              }
            } else {
              // Handle float promotion if needed
              if (srcWidth == 32 && getIntWidth(l.type) == 64) {
                indent();
                out_ << "f64.promote_f32\n";
              }
            }
            indent();
            out_ << "local.set " << mangleName(l.name.name) << "\n";
          } else if (l.init->kind == InitVal::Kind::Atom) {
            const auto &atom = std::get<AtomPtr>(l.init->value);
            bool isFloat = std::holds_alternative<FloatType>(l.type->v);
            emitAtom(*atom, locals_[l.name.name].bitwidth, isFloat);
            indent();
            out_ << "local.set " << mangleName(l.name.name) << "\n";
          }
        }
      }

      // Structured emission (--structured-lowering) rebuilds
      // genuine block/loop/if from the reducible CFG; the default path
      // below is the $__pc + br_table dispatch loop, which also handles
      // irreducible control flow.
      if (structuredLowering_) {
        emitStructuredBody(f);
      } else {
        indent();
        out_ << "i32.const 0\n";
        indent();
        out_ << "local.set $__pc\n";

        indent();
        out_ << "(loop $__refractir_dispatch_loop\n";
        indent_level_++;

        for (size_t i = 0; i < f.blocks.size(); ++i) {
          indent();
          out_ << "(block " << mangleName(f.blocks[i].label.name) << "\n";
          indent_level_++;
        }

        indent();
        out_ << "local.get $__pc\n";
        indent();
        out_ << "br_table";
        for (int i = f.blocks.size() - 1; i >= 0; --i) {
          out_ << " " << i;
        }
        out_ << " 0\n";

        for (int i = f.blocks.size() - 1; i >= 0; --i) {
          indent_level_--;
          indent();
          out_ << ") ;; " << f.blocks[i].label.name << "\n";

          const auto &b = f.blocks[i];
          for (const auto &ins: b.instrs)
            emitInstr(ins);

          std::visit(
              [this, &f, &f_blocks = f.blocks](auto &&arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, BrTerm>) {
                  if (arg.isConditional) {
                    emitCond(*arg.cond);
                    indent();
                    out_ << "if\n";
                    indent_level_++;
                    int thenIdx = -1;
                    for (size_t j = 0; j < f_blocks.size(); ++j)
                      if (f_blocks[j].label.name == arg.thenLabel.name)
                        thenIdx = j;
                    indent();
                    out_ << "i32.const " << thenIdx << "\n";
                    indent();
                    out_ << "local.set $__pc\n";
                    indent_level_--;
                    indent();
                    out_ << "else\n";
                    indent_level_++;
                    int elseIdx = -1;
                    for (size_t j = 0; j < f_blocks.size(); ++j)
                      if (f_blocks[j].label.name == arg.elseLabel.name)
                        elseIdx = j;
                    indent();
                    out_ << "i32.const " << elseIdx << "\n";
                    indent();
                    out_ << "local.set $__pc\n";
                    indent_level_--;
                    indent();
                    out_ << "end\n";
                    indent();
                    out_ << "br $__refractir_dispatch_loop\n";
                  } else {
                    int destIdx = -1;
                    for (size_t j = 0; j < f_blocks.size(); ++j)
                      if (f_blocks[j].label.name == arg.dest.name)
                        destIdx = j;
                    indent();
                    out_ << "i32.const " << destIdx << "\n";
                    indent();
                    out_ << "local.set $__pc\n";
                    indent();
                    out_ << "br $__refractir_dispatch_loop\n";
                  }
                } else if constexpr (std::is_same_v<T, RetTerm>) {
                  emitReturn(arg, f);
                } else if constexpr (std::is_same_v<T, UnreachableTerm>) {
                  indent();
                  out_ << "unreachable\n";
                }
              },
              b.term
          );
        }

        indent_level_--;
        indent();
        out_ << ") ;; dispatch loop\n";

        if (f.retType && !retVec && !f.blocks.empty()) {
          indent();
          bool isFloat = std::holds_alternative<FloatType>(f.retType->v);
          if (isFloat) {
            out_ << (getIntWidth(f.retType) <= 32 ? "f32.const 0.0\n" : "f64.const 0.0\n");
          } else {
            out_ << (getIntWidth(f.retType) <= 32 ? "i32.const 0\n" : "i64.const 0\n");
          }
        }
      } // end dispatch-loop path

      indent_level_--;
      indent();
      out_ << ")\n\n";
      indent();
      std::string exportedName = stripSigil(f.name.name);
      if (exportedName == "main") {
        if (noMainMangle_) {
          out_ << "(export \"main\" (func " << mangleName(f.name.name) << "))\n";
        } else {
          out_ << "(export \"refractir_main\" (func " << mangleName(f.name.name) << "))\n";
        }
      } else {
        out_ << "(export \"" << exportedName << "\" (func " << mangleName(f.name.name) << "))\n";
      }
    }

    if (!noModuleTags_) {
      indent_level_--;
      out_ << ")\n";
    }
  }

  // Statement-level emission of one instruction. Shared by the $__pc
  // dispatch-loop body in emit() and the structured control-tree
  // emitter (src/backend/wasm_structured.cpp).
  void WasmBackend::emitInstr(const Instr &ins) {
    std::visit(
        [this](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, AssignInstr>) {
            if (locals_.count(arg.lhs.base.name)) {
              const auto &info = locals_.at(arg.lhs.base.name);
              TypePtr lhsTy = getLValueType(arg.lhs);
              if (lhsTy && std::holds_alternative<VecType>(lhsTy->v) && arg.lhs.accesses.empty()) {
                auto &vt = std::get<VecType>(lhsTy->v);
                std::uint32_t elemSize = getTypeSize(vt.elem);
                bool valIsFloat = std::holds_alternative<FloatType>(vt.elem->v);
                uint32_t width = getIntWidth(vt.elem);
                // `%v = call @f(...)` with a frame-memory
                // lhs — write through sret straight into the lhs
                // storage, no lane loop. Register-strategy lhs
                // falls through: the call materializes once into
                // its scratch and the loop below copies lanes out.
                if (arg.rhs.rest.empty() && vecLowering_->usesFrameMemory()) {
                  if (auto ca = std::get_if<CallAtom>(&arg.rhs.first.v)) {
                    auto [ptypes, rtype] = calleeSignature(*ca);
                    (void) ptypes;
                    if (rtype && std::holds_alternative<VecType>(rtype->v)) {
                      emitCallAtom(*ca, info.offset);
                      return;
                    }
                  }
                }
                // Nested vector-returning calls evaluate exactly
                // once, before the per-lane rhs walk.
                materializeVecCalls(arg.rhs);
                for (uint64_t i = 0; i < vt.size; ++i) {
                  if (!vecLowering_->usesFrameMemory()) {
                    vecLowering_->emitLaneWrite(*this, arg.lhs.base.name, vt, i, [&] {
                      emitVecExprLane(arg.rhs, vt, i, width, valIsFloat);
                    });
                    continue;
                  }
                  indent();
                  out_ << "local.get $__old_sp\n";
                  indent();
                  out_ << "i32.const " << (info.offset - i * elemSize) << "\n";
                  indent();
                  out_ << "i32.sub\n";

                  emitVecExprLane(arg.rhs, vt, i, width, valIsFloat);

                  indent();
                  if (valIsFloat) {
                    out_ << (width == 32 ? "f32.store\n" : "f64.store\n");
                  } else {
                    out_ << (width <= 8 ? "i32.store8"
                                        : (width <= 16 ? "i32.store16"
                                                       : (width <= 32 ? "i32.store" : "i64.store")))
                         << "\n";
                  }
                }
              } else if (!arg.lhs.accesses.empty() &&
                         std::holds_alternative<VecType>(info.refractirType->v) &&
                         !vecLowering_->usesFrameMemory()) {
                // `%v[idx] = expr` on a register-strategy
                // vector local: the strategy owns the lane write.
                const auto &vt = std::get<VecType>(info.refractirType->v);
                std::uint32_t width = getIntWidth(vt.elem);
                bool valIsFloat = std::holds_alternative<FloatType>(vt.elem->v);
                auto emitValue = [&] { emitExpr(arg.rhs, width, valIsFloat); };
                if (auto ai = std::get_if<AccessIndex>(&arg.lhs.accesses[0])) {
                  if (auto lit = std::get_if<IntLit>(&ai->index)) {
                    vecLowering_->emitLaneWrite(
                        *this, arg.lhs.base.name, vt, lit->value, emitValue
                    );
                  } else {
                    vecLowering_->emitLaneWriteDyn(
                        *this, arg.lhs.base.name, vt, [&] { emitIndex(ai->index); }, emitValue
                    );
                  }
                }
              } else if (info.isAggregate || !arg.lhs.accesses.empty()) {
                emitAddress(arg.lhs);
                TypePtr curType = info.refractirType;
                for (const auto &acc: arg.lhs.accesses) {
                  if (std::holds_alternative<AccessIndex>(acc)) {
                    if (auto at = std::get_if<ArrayType>(&curType->v))
                      curType = at->elem;
                    else if (auto vt = std::get_if<VecType>(&curType->v))
                      curType = vt->elem;
                  } else if (auto af = std::get_if<AccessField>(&acc)) {
                    if (auto st = std::get_if<StructType>(&curType->v)) {
                      auto &fld = af->field;
                      if (structLayouts_.count(st->name.name) &&
                          structLayouts_.at(st->name.name).fields.count(fld))
                        curType = structLayouts_.at(st->name.name).fields.at(fld).type;
                    }
                  }
                }
                std::uint32_t width = 0;
                bool valIsFloat = false;
                if (auto bits = TypeUtils::getIntBitWidth(curType)) {
                  width = *bits;
                } else if (curType && std::holds_alternative<FloatType>(curType->v)) {
                  valIsFloat = true;
                  width = (std::get<FloatType>(curType->v).kind == FloatType::Kind::F32) ? 32 : 64;
                } else if (curType && std::holds_alternative<PtrType>(curType->v)) {
                  // WASM pointers are 32-bit (one i32 cell).
                  width = 32;
                }

                emitExpr(arg.rhs, width, valIsFloat);
                indent();
                if (valIsFloat) {
                  out_ << (width == 32 ? "f32.store\n" : "f64.store\n");
                } else {
                  if (width <= 8)
                    out_ << "i32.store8\n";
                  else if (width <= 16)
                    out_ << "i32.store16\n";
                  else if (width <= 32)
                    out_ << "i32.store\n";
                  else
                    out_ << "i64.store\n";
                }
              } else {
                bool isFloat = std::holds_alternative<FloatType>(info.refractirType->v);
                bool isPtr = std::holds_alternative<PtrType>(info.refractirType->v);
                if (isPtr) {
                  emitPtrExpr(arg.rhs, info.refractirType);
                } else if (isPtrDiff(arg.rhs)) {
                  // ptr - ptr → i64 element distance. Emit byte diff, then /sizeof.
                  emitPtrDiff(arg.rhs);
                } else {
                  emitExpr(arg.rhs, info.bitwidth, isFloat);
                }
                indent();
                out_ << "local.set " << mangleName(arg.lhs.base.name) << "\n";
              }
            }
          } else if constexpr (std::is_same_v<T, RequireInstr>) {
            if (!noRequire_) {
              emitCond(arg.cond);
              indent();
              out_ << "i32.eqz\n";
              indent();
              out_ << "if\n";
              indent_level_++;
              indent();
              out_ << "unreachable\n";
              indent_level_--;
              indent();
              out_ << "end\n";
            }
          } else if constexpr (std::is_same_v<T, StoreInstr>) {
            // *ptr = val — with null-pointer trap
            // Determine pointee type from the pointer expression
            uint32_t storeWidth = 32;
            bool storeIsFloat = false;
            if (auto rva = std::get_if<RValueAtom>(&arg.ptr.first.v)) {
              if (locals_.count(rva->rval.base.name)) {
                const auto &pinfo = locals_.at(rva->rval.base.name);
                if (auto pt = std::get_if<PtrType>(&pinfo.refractirType->v)) {
                  if (auto bits = TypeUtils::getIntBitWidth(pt->pointee)) {
                    storeWidth = *bits;
                  } else if (pt->pointee && std::holds_alternative<FloatType>(pt->pointee->v)) {
                    storeIsFloat = true;
                    storeWidth = (std::get<FloatType>(pt->pointee->v).kind == FloatType::Kind::F32)
                                     ? 32
                                     : 64;
                  }
                }
              }
            }
            // Emit ptr expr; with guards on, save to $__ptr_temp,
            // null-check, and restore. Under --no-ub-guards the
            // pointer is simply left on the stack for the store.
            emitExpr(arg.ptr, 32, false);
            if (!noUbGuards_) {
              indent();
              out_ << "local.tee $__ptr_temp\n";
              indent();
              out_ << "i32.eqz\n";
              indent();
              out_ << "if\n";
              indent_level_++;
              indent();
              out_ << "unreachable\n";
              indent_level_--;
              indent();
              out_ << "end\n";
              indent();
              out_ << "local.get $__ptr_temp\n";
            }
            emitExpr(arg.val, storeWidth, storeIsFloat);
            indent();
            if (storeIsFloat) {
              out_ << (storeWidth <= 32 ? "f32.store\n" : "f64.store\n");
            } else {
              out_
                  << (storeWidth <= 8    ? "i32.store8\n"
                      : storeWidth <= 16 ? "i32.store16\n"
                      : storeWidth <= 32 ? "i32.store\n"
                                         : "i64.store\n");
            }
          }
        },
        ins
    );
  }

  // `ret` terminator emission: a vector return through the hidden
  // `$__sret` slot or a scalar result, the shadow-stack teardown, then
  // `return`. Shared by both emitters likewise.
  void WasmBackend::emitReturn(const RetTerm &arg, const FunDecl &f) {
    if (arg.value && f.retType && std::holds_alternative<VecType>(f.retType->v)) {
      // Vector return: copy the lanes into the
      // caller's sret slot; the function has no WASM result.
      const auto &vt = std::get<VecType>(f.retType->v);
      std::uint32_t elemSize = getTypeSize(vt.elem);
      std::uint32_t width = getIntWidth(vt.elem);
      bool valIsFloat = std::holds_alternative<FloatType>(vt.elem->v);
      materializeVecCalls(*arg.value);
      for (uint64_t i = 0; i < vt.size; ++i) {
        indent();
        out_ << "local.get $__sret\n";
        if (i > 0) {
          indent();
          out_ << "i32.const " << (i * elemSize) << "\n";
          indent();
          out_ << "i32.add\n";
        }
        emitVecExprLane(*arg.value, vt, i, width, valIsFloat);
        indent();
        if (valIsFloat) {
          out_ << (width == 32 ? "f32.store\n" : "f64.store\n");
        } else {
          out_ << (width <= 8
                       ? "i32.store8"
                       : (width <= 16 ? "i32.store16" : (width <= 32 ? "i32.store" : "i64.store")))
               << "\n";
        }
      }
    } else if (arg.value) {
      bool isFloat = std::holds_alternative<FloatType>(f.retType->v);
      emitExpr(*arg.value, getIntWidth(f.retType), isFloat);
    }
    if (stackSize_ > 0) {
      indent();
      out_ << "local.get $__old_sp\n";
      indent();
      out_ << "global.set $__stack_pointer\n";
    }
    indent();
    out_ << "return\n";
  }

} // namespace refractir
