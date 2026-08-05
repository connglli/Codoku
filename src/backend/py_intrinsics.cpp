// Python-side built-in intrinsic emission.
//
// Mirrors the interpreter's semantics (src/interp/intrinsics.cpp — the
// single source of truth) with one generic, width-parameterized python
// helper per IntrinsicKind. Helpers are emitted on demand: only kinds
// declared by the program land in the module. Call sites pass the
// widths pinned by the resolved IntrinsicDecl, so a program using
// @abs at i8 and i32 shares one `_in_abs(x, n)` helper.
//
// To add a new intrinsic: add its IntrinsicKind
// (include/analysis/intrinsics.hpp), a helper snippet in
// `helperSource`, and a call-shape case in `call`.


#include <set>
#include <stdexcept>
#include "analysis/intrinsics.hpp"
#include "backend/py_backend.hpp"

namespace refractir {

  namespace {

    // Python source of the generic helper for one kind. Result
    // wrapping mirrors the interpreter's makeInt (mask + sign-extend
    // == _cast_int); explicit traps mirror its UndefinedBehaviorError
    // checks. Predicates return the canonical i1 {0,-1}.
    const char *helperSource(IntrinsicKind k) {
      switch (k) {
        case IntrinsicKind::Abs:
          return R"PY(
def _in_abs(x, n):
    if x == -(1 << (n - 1)):
        _trap("@abs of INT_MIN is not representable")
    return -x if x < 0 else x
)PY";
        case IntrinsicKind::Min:
          return R"PY(
def _in_min(a, b):
    return a if a < b else b
)PY";
        case IntrinsicKind::Max:
          return R"PY(
def _in_max(a, b):
    return a if a > b else b
)PY";
        case IntrinsicKind::Popcount:
          return R"PY(
def _in_popcount(x, pn, n):
    c = bin(x & ((1 << pn) - 1)).count("1")
    if c > (1 << (n - 1)) - 1:
        _trap("@popcount result not representable")
    return c
)PY";
        case IntrinsicKind::Clz:
          return R"PY(
def _in_clz(x, n):
    u = x & ((1 << n) - 1)
    if u == 0:
        _trap("@clz requires non-zero input")
    return _cast_int(n - u.bit_length(), n)
)PY";
        case IntrinsicKind::Ctz:
          return R"PY(
def _in_ctz(x, n):
    u = x & ((1 << n) - 1)
    if u == 0:
        _trap("@ctz requires non-zero input")
    return _cast_int((u & -u).bit_length() - 1, n)
)PY";
        case IntrinsicKind::AbsDiff:
          return R"PY(
def _in_abs_diff(a, b, n):
    r = a - b
    r = -r if r < 0 else r
    if r > (1 << (n - 1)) - 1:
        _trap("@abs_diff result not representable")
    return r
)PY";
        case IntrinsicKind::Signum:
          return R"PY(
def _in_signum(x):
    return (1 if x > 0 else 0) - (1 if x < 0 else 0)
)PY";
        case IntrinsicKind::Clamp:
          return R"PY(
def _in_clamp(v, lo, hi):
    if lo > hi:
        _trap("@clamp requires lo <= hi")
    return lo if v < lo else (hi if v > hi else v)
)PY";
        case IntrinsicKind::Midpoint:
          return R"PY(
def _in_midpoint(a, b):
    s = a + b
    return -((-s) // 2) if s < 0 else s // 2
)PY";
        case IntrinsicKind::Parity:
          return R"PY(
def _in_parity(x, pn):
    return -(bin(x & ((1 << pn) - 1)).count("1") & 1)
)PY";
        case IntrinsicKind::Bswap:
          return R"PY(
def _in_bswap(x, n):
    u = x & ((1 << n) - 1)
    r = 0
    for i in range(n // 8):
        r |= ((u >> (i * 8)) & 0xFF) << ((n // 8 - 1 - i) * 8)
    return _cast_int(r, n)
)PY";
        case IntrinsicKind::Bitreverse:
          return R"PY(
def _in_bitreverse(x, n):
    u = x & ((1 << n) - 1)
    r = 0
    for i in range(n):
        if (u >> i) & 1:
            r |= 1 << (n - 1 - i)
    return _cast_int(r, n)
)PY";
        case IntrinsicKind::Rotl:
          return R"PY(
def _in_rotl(x, k, n):
    if k < 0 or k >= n:
        _trap("@rotl requires 0 <= n < N")
    u = x & ((1 << n) - 1)
    if k:
        u = ((u << k) | (u >> (n - k))) & ((1 << n) - 1)
    return _cast_int(u, n)
)PY";
        case IntrinsicKind::Rotr:
          return R"PY(
def _in_rotr(x, k, n):
    if k < 0 or k >= n:
        _trap("@rotr requires 0 <= n < N")
    u = x & ((1 << n) - 1)
    if k:
        u = ((u >> k) | (u << (n - k))) & ((1 << n) - 1)
    return _cast_int(u, n)
)PY";
        case IntrinsicKind::IsPow2:
          return R"PY(
def _in_is_pow2(x):
    return -1 if x > 0 and (x & (x - 1)) == 0 else 0
)PY";
        case IntrinsicKind::Ilog2:
          return R"PY(
def _in_ilog2(x, n):
    if x <= 0:
        _trap("@ilog2 requires x > 0")
    return _cast_int(x.bit_length() - 1, n)
)PY";
        case IntrinsicKind::WrappingAdd:
          return R"PY(
def _in_wrapping_add(a, b, n):
    return _cast_int(a + b, n)
)PY";
        case IntrinsicKind::WrappingSub:
          return R"PY(
def _in_wrapping_sub(a, b, n):
    return _cast_int(a - b, n)
)PY";
        case IntrinsicKind::WrappingMul:
          return R"PY(
def _in_wrapping_mul(a, b, n):
    return _cast_int(a * b, n)
)PY";
        case IntrinsicKind::WrappingNeg:
          return R"PY(
def _in_wrapping_neg(x, n):
    return _cast_int(-x, n)
)PY";
        case IntrinsicKind::WrappingShl:
          return R"PY(
def _in_wrapping_shl(x, k, n):
    if k < 0 or k >= n:
        _trap("@wrapping_shl requires 0 <= n < N")
    return _cast_int(x << k, n)
)PY";
        case IntrinsicKind::WrappingShr:
          return R"PY(
def _in_wrapping_shr(x, k, n):
    if k < 0 or k >= n:
        _trap("@wrapping_shr requires 0 <= n < N")
    return x >> k
)PY";
        case IntrinsicKind::SaturatingAdd:
          return R"PY(
def _in_saturating_add(a, b, n):
    s = a + b
    lo, hi = -(1 << (n - 1)), (1 << (n - 1)) - 1
    return lo if s < lo else (hi if s > hi else s)
)PY";
        case IntrinsicKind::SaturatingSub:
          return R"PY(
def _in_saturating_sub(a, b, n):
    s = a - b
    lo, hi = -(1 << (n - 1)), (1 << (n - 1)) - 1
    return lo if s < lo else (hi if s > hi else s)
)PY";
        case IntrinsicKind::SaturatingMul:
          return R"PY(
def _in_saturating_mul(a, b, n):
    s = a * b
    lo, hi = -(1 << (n - 1)), (1 << (n - 1)) - 1
    return lo if s < lo else (hi if s > hi else s)
)PY";
        case IntrinsicKind::SaturatingNeg:
          return R"PY(
def _in_saturating_neg(x, n):
    return (1 << (n - 1)) - 1 if x == -(1 << (n - 1)) else -x
)PY";
        case IntrinsicKind::DivEuclid:
          return R"PY(
def _in_div_euclid(a, b, n):
    if b == 0:
        _trap("@div_euclid divisor is zero")
    if a == -(1 << (n - 1)) and b == -1:
        _trap("@div_euclid INT_MIN / -1 not representable")
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    if a - q * b < 0:
        q -= 1 if b > 0 else -1
    return q
)PY";
        case IntrinsicKind::RemEuclid:
          return R"PY(
def _in_rem_euclid(a, b, n):
    if b == 0:
        _trap("@rem_euclid divisor is zero")
    if a == -(1 << (n - 1)) and b == -1:
        _trap("@rem_euclid INT_MIN rem -1 not representable")
    return a % abs(b)
)PY";
        case IntrinsicKind::Fabs:
          return R"PY(
def _in_fabs(x):
    return math.fabs(x)
)PY";
        case IntrinsicKind::Fneg:
          return R"PY(
def _in_fneg(x):
    return -x
)PY";
        case IntrinsicKind::Copysign:
          return R"PY(
def _in_copysign(x, y):
    return math.copysign(x, y)
)PY";
        case IntrinsicKind::Signbit:
          return R"PY(
def _in_signbit(x):
    return -1 if math.copysign(1.0, x) < 0 else 0
)PY";
        case IntrinsicKind::ToBits:
          return R"PY(
def _in_to_bits(x, fb):
    if fb == 32:
        return struct.unpack("<i", struct.pack("<f", x))[0]
    return struct.unpack("<q", struct.pack("<d", x))[0]
)PY";
        case IntrinsicKind::FromBits:
          return R"PY(
def _in_from_bits(x, fb):
    if fb == 32:
        r = struct.unpack("<f", struct.pack("<i", x))[0]
    else:
        r = struct.unpack("<d", struct.pack("<q", x))[0]
    if not math.isfinite(r):
        _trap("@from_bits result is non-finite")
    return r
)PY";
        case IntrinsicKind::IsNormal:
          return R"PY(
def _in_is_normal(x, fb):
    m = 1.1754943508222875e-38 if fb == 32 else 2.2250738585072014e-308
    return -1 if abs(x) >= m else 0
)PY";
        case IntrinsicKind::IsSubnormal:
          return R"PY(
def _in_is_subnormal(x, fb):
    m = 1.1754943508222875e-38 if fb == 32 else 2.2250738585072014e-308
    a = abs(x)
    return -1 if 0.0 < a < m else 0
)PY";
        case IntrinsicKind::Fmin:
          return R"PY(
def _in_fmin(x, y):
    if x < y:
        return x
    if y < x:
        return y
    return x if math.copysign(1.0, x) < 0 else y
)PY";
        case IntrinsicKind::Fmax:
          return R"PY(
def _in_fmax(x, y):
    if x > y:
        return x
    if y > x:
        return y
    return y if math.copysign(1.0, x) < 0 else x
)PY";
        case IntrinsicKind::Sqrt:
          return R"PY(
def _in_sqrt(x, fb):
    if x < 0.0:
        _trap("@sqrt of a negative value is NaN")
    r = math.sqrt(x)
    return _f32(r) if fb == 32 else r
)PY";
        case IntrinsicKind::Floor:
          return R"PY(
def _in_floor(x):
    r = float(math.floor(x))
    return math.copysign(0.0, x) if r == 0.0 else r
)PY";
        case IntrinsicKind::Ceil:
          return R"PY(
def _in_ceil(x):
    r = float(math.ceil(x))
    return math.copysign(0.0, x) if r == 0.0 else r
)PY";
        case IntrinsicKind::Trunc:
          return R"PY(
def _in_trunc(x):
    r = float(math.trunc(x))
    return math.copysign(0.0, x) if r == 0.0 else r
)PY";
        case IntrinsicKind::Fract:
          return R"PY(
def _in_fract(x, fb):
    r = x - float(math.trunc(x))
    return _f32(r) if fb == 32 else r
)PY";
        case IntrinsicKind::Recip:
          return R"PY(
def _in_recip(x, fb):
    if x == 0.0:
        _trap("@recip of zero is non-finite")
    r = 1.0 / x
    if fb == 32:
        r = _f32(r)
    if not math.isfinite(r):
        _trap("@recip result is non-finite")
    return r
)PY";
        case IntrinsicKind::Crc32Update:
          return R"PY(
_CRC32_TAB = []


def _in_crc32_update(state, val, vbits):
    if not _CRC32_TAB:
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ 0xEDB88320 if c & 1 else c >> 1
            _CRC32_TAB.append(c)
    s = state & 0xFFFFFFFF
    u = val & ((1 << vbits) - 1)
    for _ in range((vbits + 7) // 8):
        s = (s >> 8) ^ _CRC32_TAB[(s ^ (u & 0xFF)) & 0xFF]
        u >>= 8
    return _cast_int(s, 32)
)PY";
        case IntrinsicKind::CheckChksum:
          return R"PY(
def _in_check_chksum(expected, actual):
    if expected != actual:
        _trap("@check_chksum mismatch")
    return actual
)PY";
        // Observability beacon — identity. Python has no forward-progress
        // assumption, so a diverging loop is preserved without an observable
        // side effect; the helper just returns its argument.
        case IntrinsicKind::Observe:
          return R"PY(
def _in_observe(x):
    return x
)PY";
        // Horizontal vector reductions (§12.4) never reach here:
        // emitIntrinsicHelpers routes them to emitReductionHelperDefs, whose
        // body varies with the guard mode and the element width and so
        // cannot come from a static source string. Reaching this case means
        // that routing was bypassed.
        case IntrinsicKind::ReduceAdd:
        case IntrinsicKind::ReduceMin:
        case IntrinsicKind::ReduceMax:
        case IntrinsicKind::ReduceAnd:
        case IntrinsicKind::ReduceOr:
        case IntrinsicKind::ReduceXor:
          throw std::runtime_error(
              "python target: horizontal vector reductions (@reduce_*) are emitted by "
              "emitReductionHelperDefs, not helperSource"
          );
      }
      return "";
    }

    std::uint32_t intBitsOf(const TypePtr &t) {
      if (!t)
        return 32;
      if (auto it = std::get_if<IntType>(&t->v)) {
        switch (it->kind) {
          case IntType::Kind::I32:
            return 32;
          case IntType::Kind::I64:
            return 64;
          case IntType::Kind::ICustom:
            return static_cast<std::uint32_t>(it->bits.value_or(32));
        }
      }
      return 32;
    }

    std::uint32_t fpBitsOf(const TypePtr &t) {
      auto fp = t ? std::get_if<FloatType>(&t->v) : nullptr;
      return (fp && fp->kind == FloatType::Kind::F32) ? 32 : 64;
    }

  } // namespace

  void PyBackend::emitIntrinsicHelpers(const Program &prog) {
    std::set<IntrinsicKind> kinds;
    for (const auto &intr: prog.intrinsics) {
      auto k = getIntrinsicKind(intr.name.name);
      if (!k)
        throw std::runtime_error("python target: unknown intrinsic " + intr.name.name);
      kinds.insert(*k);
    }
    for (IntrinsicKind k: kinds) {
      // Reductions are generated dynamically (mode- and
      // width-aware) rather than from a static source string.
      if (isReductionIntrinsic(k))
        emitReductionHelperDefs(k);
      else
        out_ << "\n" << helperSource(k);
    }
  }

  // Horizontal vector reductions (§12.4). Each helper takes the
  // vector as a lane list (the Python function-boundary ABI) and folds it
  // left-to-right, reading every lane through `_rd` (undef → trap). The
  // element domain splits int (`_i`) from float (`_f`); @reduce_add carries
  // per-step UB — an out-of-range int partial sum or a non-finite fp
  // intermediate — emitted only under UB guards. min/max fold with the
  // IEEE signed-zero tie-break for floats (matching @fmin/@fmax); the
  // bitwise members never trap. Python's unbounded ints make the int fold
  // exact; f32 steps round through `_f32`.
  void PyBackend::emitReductionHelperDefs(IntrinsicKind k) {
    const bool g = !noUbGuards_;
    out_ << "\n";
    switch (k) {
      case IntrinsicKind::ReduceAdd:
        out_ << "def _in_reduce_add_i(v, n):\n";
        if (g)
          out_ << "    lo = -(1 << (n - 1)); hi = (1 << (n - 1)) - 1\n";
        out_ << "    acc = _rd(v, 0)\n"
                "    for k in range(1, len(v)):\n"
                "        acc = acc + _rd(v, k)\n";
        if (g)
          out_ << "        if acc < lo or acc > hi:\n"
                  "            _trap(\"@reduce_add overflow\")\n";
        out_ << "    return acc\n\n";
        out_ << "def _in_reduce_add_f(v, w):\n"
                "    acc = _rd(v, 0)\n"
                "    for k in range(1, len(v)):\n"
                "        acc = acc + _rd(v, k)\n"
                "        if w == 32:\n"
                "            acc = _f32(acc)\n";
        if (g)
          out_ << "        if not math.isfinite(acc):\n"
                  "            _trap(\"@reduce_add non-finite\")\n";
        out_ << "    return acc\n";
        break;
      case IntrinsicKind::ReduceMin:
      case IntrinsicKind::ReduceMax: {
        const bool isMin = (k == IntrinsicKind::ReduceMin);
        const char *cmp = isMin ? "<" : ">";
        const char *nm = isMin ? "min" : "max";
        out_ << "def _in_reduce_" << nm
             << "_i(v):\n"
                "    acc = _rd(v, 0)\n"
                "    for k in range(1, len(v)):\n"
                "        x = _rd(v, k)\n"
                "        if x "
             << cmp
             << " acc:\n"
                "            acc = x\n"
                "    return acc\n\n";
        out_ << "def _in_reduce_" << nm
             << "_f(v):\n"
                "    acc = _rd(v, 0)\n"
                "    for k in range(1, len(v)):\n"
                "        x = _rd(v, k)\n"
                "        if x "
             << cmp
             << " acc:\n"
                "            acc = x\n"
                "        elif not (acc "
             << cmp << " x):\n"
             << (isMin ? "            acc = acc if math.copysign(1.0, acc) < 0 else x\n"
                       : "            acc = x if math.copysign(1.0, acc) < 0 else acc\n")
             << "    return acc\n";
        break;
      }
      case IntrinsicKind::ReduceAnd:
      case IntrinsicKind::ReduceOr:
      case IntrinsicKind::ReduceXor: {
        const char *op = (k == IntrinsicKind::ReduceAnd)  ? "&"
                         : (k == IntrinsicKind::ReduceOr) ? "|"
                                                          : "^";
        const char *nm = (k == IntrinsicKind::ReduceAnd)  ? "and"
                         : (k == IntrinsicKind::ReduceOr) ? "or"
                                                          : "xor";
        out_ << "def _in_reduce_" << nm
             << "(v):\n"
                "    acc = _rd(v, 0)\n"
                "    for k in range(1, len(v)):\n"
                "        acc = acc "
             << op
             << " _rd(v, k)\n"
                "    return acc\n";
        break;
      }
      default:
        throw std::runtime_error("python reduction: unexpected kind");
    }
  }

  std::string
  PyBackend::callIntrinsic(const IntrinsicDecl &intr, const std::vector<std::string> &args) {
    auto kindOpt = getIntrinsicKind(intr.name.name);
    if (!kindOpt)
      throw std::runtime_error("python target: unknown intrinsic " + intr.name.name);
    const IntrinsicKind k = *kindOpt;

    auto join = [&](std::initializer_list<std::string> extra) {
      std::string s;
      for (const auto &a: args) {
        if (!s.empty())
          s += ", ";
        s += a;
      }
      for (const auto &e: extra) {
        if (!s.empty())
          s += ", ";
        s += e;
      }
      return s;
    };
    const std::string retN = std::to_string(intBitsOf(intr.retType));
    auto paramN = [&](std::size_t i) { return std::to_string(intBitsOf(intr.params[i].type)); };

    switch (k) {
      case IntrinsicKind::Abs:
        return "_in_abs(" + join({retN}) + ")";
      case IntrinsicKind::Min:
        return "_in_min(" + join({}) + ")";
      case IntrinsicKind::Max:
        return "_in_max(" + join({}) + ")";
      case IntrinsicKind::Popcount:
        return "_in_popcount(" + join({paramN(0), retN}) + ")";
      case IntrinsicKind::Clz:
        return "_in_clz(" + join({retN}) + ")";
      case IntrinsicKind::Ctz:
        return "_in_ctz(" + join({retN}) + ")";
      case IntrinsicKind::AbsDiff:
        return "_in_abs_diff(" + join({retN}) + ")";
      case IntrinsicKind::Signum:
        return "_in_signum(" + join({}) + ")";
      case IntrinsicKind::Clamp:
        return "_in_clamp(" + join({}) + ")";
      case IntrinsicKind::Midpoint:
        return "_in_midpoint(" + join({}) + ")";
      case IntrinsicKind::Parity:
        return "_in_parity(" + join({paramN(0)}) + ")";
      case IntrinsicKind::Bswap:
        return "_in_bswap(" + join({retN}) + ")";
      case IntrinsicKind::Bitreverse:
        return "_in_bitreverse(" + join({retN}) + ")";
      case IntrinsicKind::Rotl:
        return "_in_rotl(" + join({retN}) + ")";
      case IntrinsicKind::Rotr:
        return "_in_rotr(" + join({retN}) + ")";
      case IntrinsicKind::IsPow2:
        return "_in_is_pow2(" + join({}) + ")";
      case IntrinsicKind::Ilog2:
        return "_in_ilog2(" + join({retN}) + ")";
      case IntrinsicKind::WrappingAdd:
        return "_in_wrapping_add(" + join({retN}) + ")";
      case IntrinsicKind::WrappingSub:
        return "_in_wrapping_sub(" + join({retN}) + ")";
      case IntrinsicKind::WrappingMul:
        return "_in_wrapping_mul(" + join({retN}) + ")";
      case IntrinsicKind::WrappingNeg:
        return "_in_wrapping_neg(" + join({retN}) + ")";
      case IntrinsicKind::WrappingShl:
        return "_in_wrapping_shl(" + join({retN}) + ")";
      case IntrinsicKind::WrappingShr:
        return "_in_wrapping_shr(" + join({retN}) + ")";
      case IntrinsicKind::SaturatingAdd:
        return "_in_saturating_add(" + join({retN}) + ")";
      case IntrinsicKind::SaturatingSub:
        return "_in_saturating_sub(" + join({retN}) + ")";
      case IntrinsicKind::SaturatingMul:
        return "_in_saturating_mul(" + join({retN}) + ")";
      case IntrinsicKind::SaturatingNeg:
        return "_in_saturating_neg(" + join({retN}) + ")";
      case IntrinsicKind::DivEuclid:
        return "_in_div_euclid(" + join({retN}) + ")";
      case IntrinsicKind::RemEuclid:
        return "_in_rem_euclid(" + join({retN}) + ")";
      case IntrinsicKind::Fabs:
        return "_in_fabs(" + join({}) + ")";
      case IntrinsicKind::Fneg:
        return "_in_fneg(" + join({}) + ")";
      case IntrinsicKind::Copysign:
        return "_in_copysign(" + join({}) + ")";
      case IntrinsicKind::Signbit:
        return "_in_signbit(" + join({}) + ")";
      case IntrinsicKind::ToBits:
        return "_in_to_bits(" + join({std::to_string(fpBitsOf(intr.params[0].type))}) + ")";
      case IntrinsicKind::FromBits:
        return "_in_from_bits(" + join({std::to_string(fpBitsOf(intr.retType))}) + ")";
      case IntrinsicKind::IsNormal:
        return "_in_is_normal(" + join({std::to_string(fpBitsOf(intr.params[0].type))}) + ")";
      case IntrinsicKind::IsSubnormal:
        return "_in_is_subnormal(" + join({std::to_string(fpBitsOf(intr.params[0].type))}) + ")";
      case IntrinsicKind::Fmin:
        return "_in_fmin(" + join({}) + ")";
      case IntrinsicKind::Fmax:
        return "_in_fmax(" + join({}) + ")";
      case IntrinsicKind::Sqrt:
        return "_in_sqrt(" + join({std::to_string(fpBitsOf(intr.retType))}) + ")";
      case IntrinsicKind::Floor:
        return "_in_floor(" + join({}) + ")";
      case IntrinsicKind::Ceil:
        return "_in_ceil(" + join({}) + ")";
      case IntrinsicKind::Trunc:
        return "_in_trunc(" + join({}) + ")";
      case IntrinsicKind::Fract:
        return "_in_fract(" + join({std::to_string(fpBitsOf(intr.retType))}) + ")";
      case IntrinsicKind::Recip:
        return "_in_recip(" + join({std::to_string(fpBitsOf(intr.retType))}) + ")";
      case IntrinsicKind::Crc32Update:
        return "_in_crc32_update(" + join({paramN(1)}) + ")";
      case IntrinsicKind::CheckChksum:
        return "_in_check_chksum(" + join({}) + ")";
      case IntrinsicKind::Observe:
        return "_in_observe(" + join({}) + ")";
      // Reductions: fold the vector-list argument. Add carries a
      // width (int overflow / f32 rounding); min/max/bitwise do not.
      case IntrinsicKind::ReduceAdd: {
        const VecType &vt = std::get<VecType>(intr.params[0].type->v);
        if (auto fw = floatWidth(vt.elem))
          return "_in_reduce_add_f(" + join({std::to_string(fw)}) + ")";
        return "_in_reduce_add_i(" + join({std::to_string(intWidth(vt.elem))}) + ")";
      }
      case IntrinsicKind::ReduceMin:
      case IntrinsicKind::ReduceMax: {
        const VecType &vt = std::get<VecType>(intr.params[0].type->v);
        const char *nm = (k == IntrinsicKind::ReduceMin) ? "min" : "max";
        const char *suf = floatWidth(vt.elem) ? "_f" : "_i";
        return std::string("_in_reduce_") + nm + suf + "(" + join({}) + ")";
      }
      case IntrinsicKind::ReduceAnd:
        return "_in_reduce_and(" + join({}) + ")";
      case IntrinsicKind::ReduceOr:
        return "_in_reduce_or(" + join({}) + ")";
      case IntrinsicKind::ReduceXor:
        return "_in_reduce_xor(" + join({}) + ")";
    }
    return "";
  }

} // namespace refractir
