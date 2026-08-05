// "array" WASM vec-lowering: a vector `<N> T` local is a
// shadow-stack aggregate (the historical storage). The strategy object
// only *classifies* — every access keeps the backend's frame-memory
// address+load/store paths, so the lane hooks must never be reached.

#include <stdexcept>
#include "backend/wasm_backend.hpp"
#include "backend/wasm_vec_lowering.hpp"

namespace refractir {

  std::ostream &WasmVecLowering::out(WasmBackend &b) { return b.out_; }

  void WasmVecLowering::indent(WasmBackend &b) { b.indent(); }

  std::string
  WasmVecLowering::laneLocal(WasmBackend &b, const std::string &name, std::uint64_t lane) {
    return b.mangleName(name) + "__" + std::to_string(lane);
  }

  std::string WasmVecLowering::paramLocal(WasmBackend &b, const std::string &name) {
    return b.mangleName(name);
  }

  std::string WasmVecLowering::wasmTypeOf(WasmBackend &b, const TypePtr &t) {
    return b.getWasmType(t);
  }

  std::uint32_t WasmVecLowering::intWidthOf(WasmBackend &b, const TypePtr &t) {
    return b.getIntWidth(t);
  }

  std::uint32_t WasmVecLowering::typeSizeOf(WasmBackend &b, const TypePtr &t) {
    return b.getTypeSize(t);
  }

  void WasmVecLowering::signExtend(WasmBackend &b, std::uint32_t from, std::uint32_t to) {
    b.emitSignExtend(from, to);
  }

  namespace {

    class WasmArrayLowering final : public WasmVecLowering {
    public:
      std::string name() const override { return "array"; }

      bool usesFrameMemory() const override { return true; }

      [[noreturn]] static void unreached(const char *op) {
        throw std::logic_error(
            std::string("wasm vec-lowering 'array': ") + op +
            " hook reached — frame-memory accesses must use the backend's aggregate paths"
        );
      }

      void declareLocals(WasmBackend &, const std::string &, const VecType &) override {
        unreached("declareLocals");
      }

      void unpackParam(WasmBackend &, const std::string &, const VecType &) override {
        unreached("unpackParam");
      }

      void
      emitLaneRead(WasmBackend &, const std::string &, const VecType &, std::uint64_t) override {
        unreached("emitLaneRead");
      }

      void emitLaneReadDyn(
          WasmBackend &, const std::string &, const VecType &, const std::function<void()> &
      ) override {
        unreached("emitLaneReadDyn");
      }

      void emitLaneWrite(
          WasmBackend &, const std::string &, const VecType &, std::uint64_t,
          const std::function<void()> &
      ) override {
        unreached("emitLaneWrite");
      }

      void emitLaneWriteDyn(
          WasmBackend &, const std::string &, const VecType &, const std::function<void()> &,
          const std::function<void()> &
      ) override {
        unreached("emitLaneWriteDyn");
      }
    };

  } // namespace

  // Per-strategy makers defined in their own TUs.
  std::unique_ptr<WasmVecLowering> makeWasmScalarsLowering();
  std::unique_ptr<WasmVecLowering> makeWasmVecextLowering();

  std::unique_ptr<WasmVecLowering> makeWasmVecLowering(const std::string &name) {
    if (name == "array")
      return std::make_unique<WasmArrayLowering>();
    if (name == "scalars")
      return makeWasmScalarsLowering();
    if (name == "vecext")
      return makeWasmVecextLowering();
    return nullptr;
  }

} // namespace refractir
