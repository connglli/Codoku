// "scalars" WASM vec-lowering: a vector `<N> T` local is
// stored as N separate WASM locals `$v__0 .. $v__{N-1}` (elem-typed,
// integer lanes kept sign-extended at their WASM width). Mirrors the C
// and python "scalars" strategies — including the refusal of dynamic
// lane indices, since a runtime index cannot select a distinct local.
// Pointer-ABI vector params are unpacked into the lane locals in the
// prologue; the boundary itself stays the shared memory ABI.

#include <stdexcept>
#include "backend/wasm_backend.hpp"
#include "backend/wasm_vec_lowering.hpp"

namespace refractir {

  namespace {

    class WasmScalarsLowering final : public WasmVecLowering {
    public:
      std::string name() const override { return "scalars"; }

      bool usesFrameMemory() const override { return false; }

      void declareLocals(WasmBackend &b, const std::string &name, const VecType &vt) override {
        for (std::uint64_t k = 0; k < vt.size; ++k) {
          indent(b);
          out(b) << "(local " << laneLocal(b, name, k) << " " << wasmTypeOf(b, vt.elem) << ")\n";
        }
      }

      void unpackParam(WasmBackend &b, const std::string &name, const VecType &vt) override {
        std::uint32_t elemSize = typeSizeOf(b, vt.elem);
        std::uint32_t width = intWidthOf(b, vt.elem);
        bool isFloat = std::holds_alternative<FloatType>(vt.elem->v);
        for (std::uint64_t k = 0; k < vt.size; ++k) {
          indent(b);
          out(b) << "local.get " << paramLocal(b, name) << "\n";
          if (k > 0) {
            indent(b);
            out(b) << "i32.const " << (k * elemSize) << "\n";
            indent(b);
            out(b) << "i32.add\n";
          }
          indent(b);
          if (isFloat) {
            out(b) << (width == 32 ? "f32.load\n" : "f64.load\n");
          } else {
            out(b)
                << (width <= 8 ? "i32.load8_s\n"
                               : (width <= 16 ? "i32.load16_s\n"
                                              : (width <= 32 ? "i32.load\n" : "i64.load\n")));
          }
          indent(b);
          out(b) << "local.set " << laneLocal(b, name, k) << "\n";
        }
      }

      void emitLaneRead(
          WasmBackend &b, const std::string &name, const VecType &, std::uint64_t lane
      ) override {
        indent(b);
        out(b) << "local.get " << laneLocal(b, name, lane) << "\n";
      }

      void emitLaneReadDyn(
          WasmBackend &, const std::string &name, const VecType &, const std::function<void()> &
      ) override {
        rejectDyn(name);
      }

      void emitLaneWrite(
          WasmBackend &b, const std::string &name, const VecType &, std::uint64_t lane,
          const std::function<void()> &emitValue
      ) override {
        emitValue();
        indent(b);
        out(b) << "local.set " << laneLocal(b, name, lane) << "\n";
      }

      void emitLaneWriteDyn(
          WasmBackend &, const std::string &name, const VecType &, const std::function<void()> &,
          const std::function<void()> &
      ) override {
        rejectDyn(name);
      }

    private:
      [[noreturn]] static void rejectDyn(const std::string &name) {
        throw std::runtime_error(
            "vec-lowering 'scalars' does not support dynamic lane index on vector '" + name +
            "' (a runtime index cannot select a distinct WASM local)"
        );
      }
    };

  } // namespace

  // Registered by makeWasmVecLowering (src/backend/wasm_vec_lowering_array.cpp).
  std::unique_ptr<WasmVecLowering> makeWasmScalarsLowering() {
    return std::make_unique<WasmScalarsLowering>();
  }

} // namespace refractir
