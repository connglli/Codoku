// "vecext" WASM vec-lowering: a vector `<N> T` local lives in
// native SIMD-128 registers — ceil(N*sizeof(T)/16) `v128` locals
// `$v__0, $v__1, ...`, lane k at sublane k%L of register k/L where
// L = 16/sizeof(T). This is the WASM analogue of the C backend's
// "vecext" (GCC vector_size) strategy; SPEC v0.2.1 §10.16 explicitly
// permits splitting shapes wider than 16 bytes across registers.
//
// Compute stays the backend's per-lane unrolled stack code (per the
// WasmVecLowering contract); this strategy governs storage, so lanes
// move through extract_lane/replace_lane. Integer lanes are stored at
// their byte width and recovered sign-extended via extract_lane_s —
// the same round trip the frame-memory strategy gets from a
// truncating store + loadN_s. Dynamic lane indices have no native
// v128 operation and lower to a per-lane `select` chain over scratch
// locals (index and value are materialized first so their effects
// happen exactly once), guarded by an OOB trap: an out-of-range index
// is UB, and both the "array" strategy (running off the top of linear
// memory) and the C backend's vecext (sanitizers) trap on it — a
// silent select chain would diverge from every sibling lowering.

#include "backend/wasm_backend.hpp"
#include "backend/wasm_vec_lowering.hpp"

namespace refractir {

  namespace {

    class WasmVecextLowering final : public WasmVecLowering {
    public:
      std::string name() const override { return "vecext"; }

      bool usesFrameMemory() const override { return false; }

      void declareFuncScratch(WasmBackend &b) override {
        // Scratch for the dynamic-index select chains: the materialized
        // index plus one accumulator per scalar WASM type. Unused locals
        // are tolerated by the WASM tooling.
        for (const char *decl:
             {"(local $__vlane_idx i32)", "(local $__vlane_i32 i32)", "(local $__vlane_i64 i64)",
              "(local $__vlane_f32 f32)", "(local $__vlane_f64 f64)"}) {
          indent(b);
          out(b) << decl << "\n";
        }
      }

      void declareLocals(WasmBackend &b, const std::string &name, const VecType &vt) override {
        for (std::uint64_t r = 0; r < regCount(b, vt); ++r) {
          indent(b);
          out(b) << "(local " << laneLocal(b, name, r) << " v128)\n";
        }
      }

      void unpackParam(WasmBackend &b, const std::string &name, const VecType &vt) override {
        // Full 16-byte chunks load as whole registers (v128.load allows
        // unaligned access); a part-filled tail register loads lane by
        // lane so no byte outside the caller's spill slot is touched.
        std::uint32_t elemSize = typeSizeOf(b, vt.elem);
        std::uint32_t lpr = 16 / elemSize;
        std::uint64_t totalBytes = vt.size * elemSize;
        for (std::uint64_t r = 0; r < regCount(b, vt); ++r) {
          if ((r + 1) * 16 <= totalBytes) {
            indent(b);
            out(b) << "local.get " << paramLocal(b, name) << "\n";
            if (r > 0) {
              indent(b);
              out(b) << "i32.const " << (r * 16) << "\n";
              indent(b);
              out(b) << "i32.add\n";
            }
            indent(b);
            out(b) << "v128.load\n";
            indent(b);
            out(b) << "local.set " << laneLocal(b, name, r) << "\n";
            continue;
          }
          for (std::uint64_t k = r * lpr; k < vt.size; ++k) {
            indent(b);
            out(b) << "local.get " << laneLocal(b, name, r) << "\n";
            indent(b);
            out(b) << "local.get " << paramLocal(b, name) << "\n";
            if (k > 0) {
              indent(b);
              out(b) << "i32.const " << (k * elemSize) << "\n";
              indent(b);
              out(b) << "i32.add\n";
            }
            indent(b);
            out(b) << scalarLoad(b, vt) << "\n";
            indent(b);
            out(b) << shape(b, vt) << ".replace_lane " << (k % lpr) << "\n";
            indent(b);
            out(b) << "local.set " << laneLocal(b, name, r) << "\n";
          }
        }
      }

      void emitLaneRead(
          WasmBackend &b, const std::string &name, const VecType &vt, std::uint64_t lane
      ) override {
        std::uint32_t lpr = 16 / typeSizeOf(b, vt.elem);
        indent(b);
        out(b) << "local.get " << laneLocal(b, name, lane / lpr) << "\n";
        indent(b);
        out(b) << shape(b, vt) << ".extract_lane" << extractSuffix(b, vt) << " " << (lane % lpr)
               << "\n";
      }

      void emitLaneReadDyn(
          WasmBackend &b, const std::string &name, const VecType &vt,
          const std::function<void()> &emitIdx
      ) override {
        // acc = lane0; for k in 1..N-1: acc = idx==k ? lane_k : acc
        std::string acc = scratchAcc(b, vt);
        emitIdx();
        indent(b);
        out(b) << "local.set $__vlane_idx\n";
        emitIdxBoundsTrap(b, vt);
        emitLaneRead(b, name, vt, 0);
        indent(b);
        out(b) << "local.set " << acc << "\n";
        for (std::uint64_t k = 1; k < vt.size; ++k) {
          emitLaneRead(b, name, vt, k);
          indent(b);
          out(b) << "local.get " << acc << "\n";
          indent(b);
          out(b) << "local.get $__vlane_idx\n";
          indent(b);
          out(b) << "i32.const " << k << "\n";
          indent(b);
          out(b) << "i32.eq\n";
          indent(b);
          out(b) << "select\n";
          indent(b);
          out(b) << "local.set " << acc << "\n";
        }
        indent(b);
        out(b) << "local.get " << acc << "\n";
      }

      void emitLaneWrite(
          WasmBackend &b, const std::string &name, const VecType &vt, std::uint64_t lane,
          const std::function<void()> &emitValue
      ) override {
        std::uint32_t lpr = 16 / typeSizeOf(b, vt.elem);
        std::string reg = laneLocal(b, name, lane / lpr);
        indent(b);
        out(b) << "local.get " << reg << "\n";
        emitValue();
        indent(b);
        out(b) << shape(b, vt) << ".replace_lane " << (lane % lpr) << "\n";
        indent(b);
        out(b) << "local.set " << reg << "\n";
      }

      void emitLaneWriteDyn(
          WasmBackend &b, const std::string &name, const VecType &vt,
          const std::function<void()> &emitIdx, const std::function<void()> &emitValue
      ) override {
        // For every lane k: reg = idx==k ? replace_lane_k(reg, val) : reg
        std::uint32_t lpr = 16 / typeSizeOf(b, vt.elem);
        std::string acc = scratchAcc(b, vt);
        emitIdx();
        indent(b);
        out(b) << "local.set $__vlane_idx\n";
        emitIdxBoundsTrap(b, vt);
        emitValue();
        indent(b);
        out(b) << "local.set " << acc << "\n";
        for (std::uint64_t k = 0; k < vt.size; ++k) {
          std::string reg = laneLocal(b, name, k / lpr);
          indent(b);
          out(b) << "local.get " << reg << "\n";
          indent(b);
          out(b) << "local.get " << acc << "\n";
          indent(b);
          out(b) << shape(b, vt) << ".replace_lane " << (k % lpr) << "\n";
          indent(b);
          out(b) << "local.get " << reg << "\n";
          indent(b);
          out(b) << "local.get $__vlane_idx\n";
          indent(b);
          out(b) << "i32.const " << k << "\n";
          indent(b);
          out(b) << "i32.eq\n";
          indent(b);
          out(b) << "select (result v128)\n";
          indent(b);
          out(b) << "local.set " << reg << "\n";
        }
      }

    private:
      /// Trap when the materialized index is outside [0, N) — ge_u also
      /// catches negative indices as huge unsigned values.
      static void emitIdxBoundsTrap(WasmBackend &b, const VecType &vt) {
        indent(b);
        out(b) << "local.get $__vlane_idx\n";
        indent(b);
        out(b) << "i32.const " << vt.size << "\n";
        indent(b);
        out(b) << "i32.ge_u\n";
        indent(b);
        out(b) << "if\n";
        indent(b);
        out(b) << "  unreachable\n";
        indent(b);
        out(b) << "end\n";
      }

      static std::uint64_t regCount(WasmBackend &b, const VecType &vt) {
        std::uint64_t bytes = vt.size * typeSizeOf(b, vt.elem);
        return (bytes + 15) / 16;
      }

      /// The v128 lane-shape prefix for this element type.
      static std::string shape(WasmBackend &b, const VecType &vt) {
        if (std::holds_alternative<FloatType>(vt.elem->v))
          return typeSizeOf(b, vt.elem) == 4 ? "f32x4" : "f64x2";
        switch (typeSizeOf(b, vt.elem)) {
          case 1:
            return "i8x16";
          case 2:
            return "i16x8";
          case 4:
            return "i32x4";
          default:
            return "i64x2";
        }
      }

      /// Sub-word integer lanes extract sign-extended.
      static std::string extractSuffix(WasmBackend &b, const VecType &vt) {
        if (std::holds_alternative<FloatType>(vt.elem->v))
          return "";
        return typeSizeOf(b, vt.elem) <= 2 ? "_s" : "";
      }

      static std::string scalarLoad(WasmBackend &b, const VecType &vt) {
        std::uint32_t width = intWidthOf(b, vt.elem);
        if (std::holds_alternative<FloatType>(vt.elem->v))
          return width == 32 ? "f32.load" : "f64.load";
        return width <= 8
                   ? "i32.load8_s"
                   : (width <= 16 ? "i32.load16_s" : (width <= 32 ? "i32.load" : "i64.load"));
      }

      /// Scratch accumulator local matching the element's WASM type.
      static std::string scratchAcc(WasmBackend &b, const VecType &vt) {
        return "$__vlane_" + wasmTypeOf(b, vt.elem);
      }
    };

  } // namespace

  // Registered by makeWasmVecLowering (src/backend/wasm_vec_lowering_array.cpp).
  std::unique_ptr<WasmVecLowering> makeWasmVecextLowering() {
    return std::make_unique<WasmVecextLowering>();
  }

} // namespace refractir
