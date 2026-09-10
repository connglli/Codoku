# RefractIR standard intrinsics

Normative reference: §12 of [SPEC_v0.2.3.md](./SPEC_v0.2.3.md). Five implementation files back every intrinsic, one per tool, and a new intrinsic lands in all five at once:

| Tool | File |
|---|---|
| Interpreter | [src/interp/intrinsics.cpp](../src/interp/intrinsics.cpp) |
| Solver | [src/solver/intrinsics.cpp](../src/solver/intrinsics.cpp) |
| Compiler, C | [src/backend/c_intrinsics.cpp](../src/backend/c_intrinsics.cpp) |
| Compiler, WASM | [src/backend/wasm_intrinsics.cpp](../src/backend/wasm_intrinsics.cpp) |
| Compiler, Python | [src/backend/py_intrinsics.cpp](../src/backend/py_intrinsics.cpp) |

## Overview

An intrinsic is a built-in function whose semantics the RefractIR toolchain owns outright rather than delegating to the target language. Each one has a hard-coded implementation in the interpreter, a fixed SMT encoding in the solver, and a lowering rule in each backend.

The per-intrinsic sections below spell out the C and WASM lowerings, which need width- and grammar-specific rules. The Python backend emits each intrinsic as a helper in the module preamble, computing directly in Python with unbounded integers and the `math` module, and trapping on the same UB preconditions as the interpreter.

Intrinsics are declared in source with the `intrinsic` keyword before use:

```text
intrinsic @abs(%x: i32) : i32;
```

The declaration says nothing about the body; the toolchain provides it. Every intrinsic is declared once per concrete bit-width the program uses (e.g. `i32` for 32-bit `@abs`). The same generic encoding applies regardless of the width `N`.

### Declaration syntax

```text
IntrinsicDecl := "intrinsic" GlobalId "(" ParamList? ")" ":" Type ";" ;
```

Most intrinsics are typed over `iN` for any integer width `N >= 1`, input and return width agreeing, and take the uniform widening-and-mask lowering. Two exceptions live in §12.4: `@parity` and `@is_pow2` return `i1` whatever the input width, because they are pure predicates. Their widening-and-mask rule treats the return type as a fixed `i1`, lowered to `int8_t` in C, `i32` in WASM and `bv(1)` in SMT, unrelated to the input width `N`.

Every `iN` type is a signed two's-complement integer, `i1` included, so the two representable `i1` values are `{0, -1}` rather than `{0, 1}` (spec §6.4). Every predicate intrinsic here (`@parity`, `@is_pow2`, `@signbit`, `@is_normal`, `@is_subnormal`) therefore produces `-1` for true and `0` for false. The interpreter sign-extends bit 0 in `argSint` and `makeInt`, so the convention is uniform across argument read-back, result construction and `iN as iM` widening, and the backends sign-extend at the producing site, after a `cmp` or a predicate body, so an `i1` local in storage always holds `0` or `-1`.

Floating-point intrinsics are supported: the P0 floating-point basic IEEE family, batch D, covers `@fabs`, `@fneg`, `@copysign`, `@signbit`, `@to_bits`, `@from_bits`, `@is_normal`, `@is_subnormal`, `@fmin`, `@fmax`, `@sqrt`, `@floor`, `@ceil`, `@trunc`, `@fract` and `@recip`, specified in §12.6. Throughout this document `fN` denotes any concrete floating-point type, `f32` or `f64`. *Priority tiers* below covers what is deferred.

### Widening-and-mask lowering (§11.5)

The compiler maps `iN` to the smallest machine width `W` that fits:

| `N` | `W` (C / WASM) |
|-----|----------------|
| 1-8 | 8 |
| 9-16 | 16 |
| 17-32 | 32 |
| 33-64 | 64 |

Each helper widens operands to `W`, performs the operation at `W`-bit width, then sign-extends / masks the result back to `N` bits.

## 12.1 Arithmetic intrinsics

### `@abs`: signed absolute value

```text
intrinsic @abs(%x: iN) : iN;
```

Returns the absolute value of `%x`.

UB conditions:
- `%x == INT_MIN_N`: the result would be `−INT_MIN_N`, which is not representable in a signed `iN`. This is consistent with rule 4 in §7.1 (signed integer overflow is UB).

Semantics per tool:

| Tool | Behaviour |
|------|-----------|
| Interpreter | Assert `x != INT_MIN_N` (aborts path if violated); return `x < 0 ? -x : x`. |
| Solver | Add `x ≠ INT_MIN_N` to path condition; return `ite(bvsge(x, 0), x, bvneg(x))`. |
| C codegen | `if (a0 == INT_MIN_N) __builtin_trap(); return a0 < 0 ? -a0 : a0;` (widened to `intW_t`). |
| WASM codegen | `if (a0 == INT_MIN_N) unreachable;` then `select` between `0 - a0` and `a0` based on `lt_s`. |

Example:

```text
intrinsic @abs(%x: i32) : i32;

fun @main() : i32 {
  let %n: i32 = -42;
  let mut %r: i32 = 0;
^entry:
  %r = call @abs(%n);   // %r == 42
  ret %r;
}
```

Synthesis example: find `%?x` such that `@abs(%?x) == 5`:

```text
intrinsic @abs(%x: i32) : i32;
fun @main() : i32 {
  sym %?x: value i32 in [-100, 100];
  let mut %r: i32 = 0;
^entry:
  %r = call @abs(%?x);
  require %r == 5, "abs(?x) must equal 5";
  br ^exit;
^exit:
  ret %?x;  // solver returns 5 or -5
}
```

### `@min`, `@max`: signed minimum / maximum

```text
intrinsic @min(%a: iN, %b: iN) : iN;
intrinsic @max(%a: iN, %b: iN) : iN;
```

Return the signed minimum or maximum of two `iN` values. No UB conditions (the result is always representable when the inputs are).

Semantics per tool:

| Tool | `@min` | `@max` |
|------|--------|--------|
| Interpreter | `a < b ? a : b` | `a > b ? a : b` |
| Solver | `ite(bvsle(a, b), a, b)` | `ite(bvsge(a, b), a, b)` |
| C codegen | `a0 < a1 ? a0 : a1` (widened to `intW_t`) | `a0 > a1 ? a0 : a1` |
| WASM codegen | `select` with `lt_s` | `select` with `gt_s` |

Example:

```text
intrinsic @min(%a: i32, %b: i32) : i32;
intrinsic @max(%a: i32, %b: i32) : i32;

fun @clamp(%v: i32, %lo: i32, %hi: i32) : i32 {
  let mut %r: i32 = 0;
^entry:
  %r = call @max(%v, %lo);
  %r = call @min(%r, %hi);
  ret %r;
}
```

## 12.2 Bit-counting intrinsics

### `@clz`: count leading zeros

```text
intrinsic @clz(%x: iN) : iN;
```

Returns the number of leading zero bits in `%x`, counting from bit `N−1` (most significant) down to bit 0.

UB conditions:
- `%x == 0`: following C/C++ `__builtin_clz` semantics, zero input is UB. Callers must ensure `%x != 0` on the chosen execution path.

Result range: `[0, N−1]` (since `%x != 0`, at least one bit is set).

Semantics per tool:

| Tool | Behaviour |
|------|-----------|
| Interpreter | Loop from bit `N−1` down; count zeros before the first `1`. Software fallback for any `N`. |
| Solver | Add `x ≠ 0` to PC; build an ITE chain: iterate `i = 0 … N−1`, emit `ite(bit(N−1−i) == 1, i, result)` from the back. |
| C codegen | `u = maskU(a0); if (u==0) __builtin_trap(); __builtin_clz[ll](u) − (W−N)` (widened to `uintW_t`). |
| WASM codegen | Mask → `local.tee`; eqz guard → `unreachable`; then `iW.clz − (W−N)`. |

The `W−N` subtraction excludes the high `W−N` bits of the widened word, which are always zero after masking, from the builtin's count.

Example:

```text
intrinsic @clz(%x: i32) : i32;

fun @main() : i32 {
  let %n: i32 = 1;     // binary: 000...001
  let mut %r: i32 = 0;
^entry:
  %r = call @clz(%n);  // %r == 31 (for i32)
  ret %r;
}
```

### `@ctz`: count trailing zeros

```text
intrinsic @ctz(%x: iN) : iN;
```

Returns the number of trailing zero bits in `%x`, counting from bit 0 (least significant) up to bit `N−1`.

UB conditions:
- `%x == 0`: following C/C++ `__builtin_ctz` semantics, zero input is UB.

Result range: `[0, N−1]` (since `%x != 0`).

Semantics per tool:

| Tool | Behaviour |
|------|-----------|
| Interpreter | Loop from bit 0 up; count zeros before the first `1`. |
| Solver | Add `x ≠ 0` to PC; build an ITE chain: iterate `i = N−1 … 0`, emit `ite(bit(i) == 1, i, result)` from the back. |
| C codegen | `u = maskU(a0); if (u==0) __builtin_trap(); __builtin_ctz[ll](u)` (no bias needed; low bits are unaffected by widening). |
| WASM codegen | Mask → `local.tee`; eqz guard → `unreachable`; then `iW.ctz`. |

Example:

```text
intrinsic @ctz(%x: i32) : i32;

fun @main() : i32 {
  let %n: i32 = 12;    // binary: 000...01100
  let mut %r: i32 = 0;
^entry:
  %r = call @ctz(%n);  // %r == 2
  ret %r;
}
```

### `@popcount`: population count

```text
intrinsic @popcount(%x: iN) : iN;
```

Returns the number of 1 bits in `%x` (also called Hamming weight).

UB conditions:
- Result overflow at very narrow widths: for a signed `iN`, the result must fit in `iN`. E.g. for `i2` the signed maximum is 1, so `@popcount(3)` (result 2) is UB because 2 > `INT_MAX_i2 = 1`. This is rule 25 in §7.7. For `N ≥ 7` this UB is unreachable because `popcount ≤ N ≤ INT_MAX_N`.

Result range: `[0, N]` (or `[0, INT_MAX_N]` for narrow widths).

Semantics per tool:

| Tool | Behaviour |
|------|-----------|
| Interpreter | `__builtin_popcountll(maskU(x))`; UB-check that result ≤ `INT_MAX_N`. |
| Solver | Build `Σ zext(bit_k(x), N)` via `bvadd` over `N` one-bit extracts. No explicit UB guard — the bit-vector sum is always within `[0, N]`. |
| C codegen | `u = maskU(a0); __builtin_popcount[ll](u)` (widened to `uintW_t`). |
| WASM codegen | Mask input then `iW.popcnt`. |

Example:

```text
intrinsic @popcount(%x: i32) : i32;

fun @main() : i32 {
  let %n: i32 = 7;       // binary: 111
  let mut %r: i32 = 0;
^entry:
  %r = call @popcount(%n); // %r == 3
  ret %r;
}
```

## 12.3 Integer extras

These broaden the arithmetic family with the solver-trivial integer operations programs reach for most often. Every entry is trivial or easy for the solver, and trivial on both compiled backends.

### `@abs_diff`: absolute difference

```text
intrinsic @abs_diff(%a: iN, %b: iN) : iN;
```

Returns `|a − b|` interpreted as a non-negative signed `iN`.

UB conditions:
- `|a − b| > INT_MAX_N`: the absolute difference is not representable in signed `iN` (rule 25 of §7.7). Equivalently: the signed subtraction underlying the result would overflow.

Result range: `[0, INT_MAX_N]`.

| Tool | Behaviour |
|---|---|
| Interpreter | Compute `int64_t s = (int64_t)a − (int64_t)b` (always non-overflowing for `N ≤ 32`; uses `__int128` for `N == 64`). Trap if `\|s\| > INT_MAX_N`. Return `\|s\|`. |
| Solver | `r = ite(bvsge(a, b), bvsub(a, b), bvsub(b, a))`. Conjoin `bvsge(r, 0)` to `PC` (signed-comparison-only UB check). |
| C codegen | Widen both operands to `int64_t` (`__int128` for `N == 64`), subtract, take absolute value, trap if it exceeds `INT_MAX_N`, narrow back. No unsigned cast. |
| WASM codegen | Both differences computed via `iW.sub` (WASM signed/unsigned-agnostic bit-level subtraction), `select` on `iW.lt_s`, UB-check by sign-extending and comparing `lt_s 0`. |

### `@signum`: sign of a signed integer

```text
intrinsic @signum(%x: iN) : iN;
```

Returns `−1` if `x < 0`, `0` if `x == 0`, `+1` if `x > 0`. No UB.

Result range: `{−1, 0, +1}`.

| Tool | Behaviour |
|---|---|
| Interpreter | `(x > 0) − (x < 0)`. |
| Solver | `ite(bvslt(x, 0), bv(−1, N), ite(EQUAL(x, 0), bv(0, N), bv(1, N)))`. |
| C codegen | Same ternary, widened. |
| WASM codegen | Two nested `select` over `iW.lt_s` against `0`. |

### `@clamp`: signed clamp

```text
intrinsic @clamp(%v: iN, %lo: iN, %hi: iN) : iN;
```

Returns `v` clipped to `[lo, hi]` under signed ordering: `max(lo, min(v, hi))`.

UB conditions:
- `lo > hi` (signed). The clamp range must be non-empty (matches Rust's `iN::clamp`, which panics on inverted bounds).

Result range: `[lo, hi]`.

| Tool | Behaviour |
|---|---|
| Interpreter | Trap if `lo > hi`; otherwise `v < lo ? lo : (v > hi ? hi : v)`. |
| Solver | Conjoin `bvsle(lo, hi)` to `PC`; emit `ite(bvslt(v, lo), lo, ite(bvsgt(v, hi), hi, v))`. |
| C codegen | UB-trap on inverted bounds, then nested ternaries. |
| WASM codegen | UB-check (`lt_s`, `if; unreachable; end`) then two `select` on `lt_s` / `gt_s`. |

### `@midpoint`: signed midpoint (truncation toward zero)

```text
intrinsic @midpoint(%a: iN, %b: iN) : iN;
```

Returns `(a + b) / 2` with division truncating toward zero (consistent with §2.5: `midpoint(3, 0) == 1`, `midpoint(−3, 0) == −1`). The mathematical midpoint of any two `iN` values is always representable in `iN`, so this operation has no UB.

| Tool | Behaviour |
|---|---|
| Interpreter | Compute `(int64_t)a + (int64_t)b` to avoid `iN` overflow, divide by 2 with C-style truncation toward 0, sign-mask to `N` bits. |
| Solver | Sign-extend both operands to `bv(N+1)`, `bvadd`, `bvsdiv` by `2` (signed BV division truncates toward zero), extract low `N` bits. |
| C codegen | `int64_t s = (int64_t)a0 + (int64_t)a1; intW_t r = (intW_t)(s / 2);`; the sum always fits in `i64` for `N ≤ 64`. |
| WASM codegen | `i64.extend_iW_s` both, `i64.add`, `i64.const 2`, `i64.div_s`, `iW.wrap_i64`, sign-mask. (For `N = 64`, use `i128` emulation via two-step: split into high/low, midpoint of halves; but `N ≤ 63` is the common case — handle `N = 64` via a separate path that adds with a carry detection.) |

## 12.4 Bit-manipulation

Every entry is trivial or easy for the solver. WASM has native shift ops, which `@rotl` and `@rotr` use directly, while `@bswap` and `@bitreverse` need small compositions there.

### `@parity`: bit parity

```text
intrinsic @parity(%x: iN) : i1;
```

Returns `-1` (the `i1` true value) if `x` has an odd number of one-bits, else `0`. The return type is `i1`, not `iN`: `@parity` and `@is_pow2` are the first intrinsics whose return type is not parameterised by the input width.

No UB. Result range: `{0, -1}` (see the i1 convention note above).

| Tool | Behaviour |
|---|---|
| Interpreter | `__builtin_parityll(maskU(x)) & 1`, sign-extended to `i1` by `makeInt`. |
| Solver | XOR all `N` single-bit extracts of `x` together; the resulting `bv(1)` is the `i1` result. |
| C codegen | Widen, mask, `__builtin_parity[ll](u)`, sign-extend bit 0 so true is stored as `-1` in the `int8_t` cell. |
| WASM codegen | Mask, `iW.popcnt`, `iW.const 1`, `iW.and`, then sextend bit 0 (`i32.shl 31; i32.shr_s 31`) so the `i32` result is `0` / `-1`. |

### `@bswap`: byte swap

```text
intrinsic @bswap(%x: iN) : iN;
```

Reverses the byte order of `x`.

Declaration restriction: `@bswap(%x: iN) : iN` is only well-formed when `N % 8 == 0`. The semantic checker rejects declarations with `N` not a multiple of 8. For `N == 8` the operation is the identity (included for uniformity).

No UB.

| Tool | Behaviour |
|---|---|
| Interpreter | Byte loop: extract bytes from the masked value and reassemble in reverse order. |
| Solver | `BV_CONCAT` of `N/8` byte-sized `BV_EXTRACT` chunks in reverse order. |
| C codegen | Widen + mask, emit `__builtin_bswap{16,32,64}` (for `N == 8` the helper returns `a0` unchanged). For `N` that doesn't match a native bswap (e.g. `i24` is excluded by the declaration rule), no case needed. |
| WASM codegen | No native byte-swap. Emit a sequence of `iW.shr_u` / `iW.and` / `iW.shl` / `iW.or` operations — one per byte, positioning each byte at its reversed offset. |

### `@bitreverse`: bit reversal

```text
intrinsic @bitreverse(%x: iN) : iN;
```

Reverses all `N` bits of `x` (bit `i` ↔ bit `N − 1 − i`). Defined for all `N ≥ 1`. No UB.

| Tool | Behaviour |
|---|---|
| Interpreter | Bit loop: write the result bit-by-bit by extracting bit `i` of input and ORing it into position `N − 1 − i` of output. |
| Solver | `BV_CONCAT` of all `N` 1-bit `BV_EXTRACT` slices in reversed order. |
| C codegen | Software bit-loop (GCC has no `__builtin_bitreverse`). Clang's `__builtin_bitreverseN` is unused — the loop is short enough that the optimiser handles common widths. |
| WASM codegen | Software bit-loop via a `loop`/`br_if` block. |

### `@rotl`, `@rotr`: bitwise rotation

```text
intrinsic @rotl(%x: iN, %n: iN) : iN;
intrinsic @rotr(%x: iN, %n: iN) : iN;
```

Rotate `x` by `n` bit positions. `@rotl` shifts toward higher-order bits; `@rotr` toward lower-order bits.

UB conditions (consistent with the overshift rule §7.1 rule 5):
- `n < 0`: negative rotation amount.
- `n >= N`: the amount must be in `[0, N)`.

Callers wanting Rust/WASM-style mod-`N` behaviour must mask `n` themselves (`n & (N − 1)` for power-of-two `N`).

| Tool | Behaviour |
|---|---|
| Interpreter | Trap on out-of-range `n`; otherwise bit-level shift-and-or — extract the low `N` bits of `x`, shift left by `n` and right by `N − n` (bit-level, zero-fill), OR the results, mask back to `N` bits. |
| Solver | Conjoin `bvsge(n, 0) ∧ bvslt(n, N)` to `PC`; emit the rotation as `bvor(bvshl(x, n), bv_shr_logical(x, N − n))`. Logical shifts are used because rotation is a bit-permutation, not a signed-arithmetic shift. |
| C codegen | UB-trap, then a shift-and-or composition on the iN bit pattern using the backend's bit-level helpers (the same widening-and-mask vehicle that hosts iN values). |
| WASM codegen | UB-check (`iW.lt_s 0` or `iW.ge_s N`, `if; unreachable; end`); compose `iW.shl` and `iW.shr_u` (logical shift right is the bit-level primitive used by rotation, not a numeric reinterpretation). |

### `@is_pow2`: power-of-two predicate

```text
intrinsic @is_pow2(%x: iN) : i1;
```

Returns `-1` (the `i1` true value) if `x` is a positive power of two (`x > 0` and `popcount(x) == 1`), else `0`. Like `@parity`, the return type is `i1`.

No UB. Result range: `{0, -1}`.

| Tool | Behaviour |
|---|---|
| Interpreter | `(x > 0) && ((x & (x − 1)) == 0)`, sign-extended to `i1` by `makeInt`. |
| Solver | `ite(bvslt(0, x) ∧ EQUAL(bvand(x, bvsub(x, 1)), bv(0, N)), bv(1, 1), bv(0, 1))`. |
| C codegen | Same boolean, widened, written with `&&`, sign-extend bit 0 so true is `-1`. |
| WASM codegen | Compose `iW.gt_s 0` with `iW.sub`, `iW.and`, `iW.eqz`; AND the two via `iW.and`, then sextend bit 0 to yield `0` / `-1`. |

### `@ilog2`: floor log base 2

```text
intrinsic @ilog2(%x: iN) : iN;
```

Returns `floor(log2(x))` for strictly positive `x` (the position of the most significant set bit).

UB conditions:
- `x <= 0` (signed). `@ilog2` is only defined for strictly positive `x`.

Result range: `[0, N − 2]` (the sign bit must be zero, so the highest possible MSB position is `N − 2`).

| Tool | Behaviour |
|---|---|
| Interpreter | Trap on `x <= 0`; otherwise `(N − 1) − __builtin_clzll(maskU(x))`. |
| Solver | Conjoin `bvslt(bv(0, N), x)` to `PC`; build the `@clz` ITE chain and return `(N − 1) − clz`. |
| C codegen | UB-trap; `(N − 1) − __builtin_clz[ll](u)` widened. |
| WASM codegen | UB-check (`iW.const 0; iW.le_s; if; unreachable; end`); mask, `iW.clz`, `iW.const (W − 1); iW.sub`. |

## 12.5 Integer overflow-aware family

Wrapping arithmetic computes `(op) mod 2^N` with no UB on overflow. Saturating arithmetic clamps at `[INT_MIN_N, INT_MAX_N]`. Euclidean division differs from the C-style truncating division in §2.5 only when the dividend's sign forces a negative remainder. All members are declared at one common `iN` width; input and result share the same type.

The tuple-returning members of this family (`@checked_*`, `@overflowing_*`) and the cross-width `@widening_mul` are slated for a follow-up batch that introduces a multi-value return ABI; the scalar-result subset shipped here exercises the same arithmetic primitives.

### `@wrapping_add`, `@wrapping_sub`, `@wrapping_mul`, `@wrapping_neg`: modular arithmetic

```text
intrinsic @wrapping_add(%a: iN, %b: iN) : iN;
intrinsic @wrapping_sub(%a: iN, %b: iN) : iN;
intrinsic @wrapping_mul(%a: iN, %b: iN) : iN;
intrinsic @wrapping_neg(%x: iN) : iN;
```

Compute the operation modulo `2^N` and sign-extend the low `N` bits back into the declared `iN`. None of these intrinsics raises UB on overflow; `@wrapping_neg(INT_MIN_N) == INT_MIN_N` is the canonical fixed point.

| Tool | Behaviour |
|---|---|
| Interpreter | Compute on `uint64_t`, narrow via the iN mask. |
| Solver | Direct `bvadd / bvsub / bvmul / bvneg` (modular by construction). |
| C codegen | `(uty)(ua op ub)` then `sextN`. |
| WASM codegen | `iW.{add,sub,mul,sub}`, then `sextN`. |

### `@wrapping_shl`, `@wrapping_shr`: modular shifts

```text
intrinsic @wrapping_shl(%x: iN, %n: iN) : iN;
intrinsic @wrapping_shr(%x: iN, %n: iN) : iN;
```

`@wrapping_shl` is a left shift mod `2^N`; `@wrapping_shr` is the arithmetic right shift (preserves the sign bit). Both require the shift count to be in `[0, N)`, matching the shift rule of spec §7.1 rule 5.

UB conditions: `n < 0` or `n >= N`.

| Tool | Behaviour |
|---|---|
| Interpreter | `argSint` for the shift; `x << n` / `x >> n` on int64, then narrow. |
| Solver | Push `0 ≤ n < N` to `PC`; `bvshl` / `bvashr`. |
| C codegen | Trap on UB; shift the widened unsigned (shl) or sign-extended signed (shr) value. |
| WASM codegen | UB-check via two `lt_s` / `ge_s` traps; `iW.{shl,shr_s}`. |

### `@saturating_add`, `@saturating_sub`, `@saturating_mul`, `@saturating_neg`: clamp at iN bounds

```text
intrinsic @saturating_add(%a: iN, %b: iN) : iN;
intrinsic @saturating_sub(%a: iN, %b: iN) : iN;
intrinsic @saturating_mul(%a: iN, %b: iN) : iN;
intrinsic @saturating_neg(%x: iN) : iN;
```

Compute the true result over the integers and clamp to `[INT_MIN_N, INT_MAX_N]`. `@saturating_neg(INT_MIN_N) == INT_MAX_N` is the only overflow case for the unary form.

| Tool | Behaviour |
|---|---|
| Interpreter | Widen to `int64` (or `__int128` for N == 64); compute true result; clamp. |
| Solver | `BV_S{ADD,SUB,MUL}_OVERFLOW` predicate plus `BV_NEG`; ITE-select the saturated bound from the sign of `a` (or `a XOR b` for mul). |
| C codegen | Widen to `int64` (N ≤ 32) / `__int128` (N == 64), clamp; narrow back. |
| WASM codegen | i64-widen + clamp for N ≤ 32; sign-bit identity overflow detection for N == 64. |

### `@div_euclid`, `@rem_euclid`: Euclidean division

```text
intrinsic @div_euclid(%a: iN, %b: iN) : iN;
intrinsic @rem_euclid(%a: iN, %b: iN) : iN;
```

The Euclidean quotient rounds toward `−∞` instead of toward 0; the Euclidean remainder is always non-negative and strictly less than `|b|`. Equivalent to Rust's `i*::div_euclid` / `i*::rem_euclid`.

UB conditions:
- `b == 0` for either.
- `a == INT_MIN_N && b == -1`: the true quotient `−INT_MIN_N` overflows the iN range (identical to the C-style div / mod UB, §7.1).

| Tool | Behaviour |
|---|---|
| Interpreter | Trunc div; if the trunc remainder is negative, step the quotient toward `−∞`. |
| Solver | Push `b ≠ 0` and `¬(a = INT_MIN ∧ b = −1)` to `PC`; combine `bvsdiv` / `bvsrem` with an ITE adjustment. |
| C codegen | Trap on UB; `q = a / b`, `r = a − q·b`, adjust if `r < 0`. |
| WASM codegen | Two unreachable-guarded checks; `iW.div_s` + `iW.mul / iW.sub`; conditional adjustment via `if/end`. |

### Example: overflow-aware accumulator

```text
intrinsic @saturating_add(%a: i16, %b: i16) : i16;
intrinsic @wrapping_mul(%a: i16, %b: i16) : i16;
intrinsic @rem_euclid(%a: i32, %b: i32) : i32;

fun @reduce(%xs: [4] i16, %m: i32) : i32 {
  let mut %acc: i16 = 0;
  let mut %i:   i32 = 0;
  let mut %wide: i32 = 0;
  let mut %r:   i32 = 0;
^loop:
  %acc  = call @saturating_add(%acc, %xs[%i]);
  %acc  = call @wrapping_mul(%acc, 3 as i16);
  %i    = %i + 1;
  br %i < 4, ^loop, ^done;
^done:
  %wide = %acc as i32;
  %r    = call @rem_euclid(%wide, %m);
  ret %r;
}
```

### Example: bit-manipulation family

```text
intrinsic @bitreverse(%x: i32) : i32;
intrinsic @rotl(%x: i32, %n: i32) : i32;
intrinsic @is_pow2(%x: i32) : i1;

fun @demo(%x: i32) : i32 {
  let mut %r: i32 = 0;
  let mut %p: i1 = 0;
  let %four: i32 = 4;
^entry:
  %r = call @rotl(%x, %four);
  %r = call @bitreverse(%r);
  %p = call @is_pow2(%r);
  br %p, ^pow2, ^other;
^pow2:
  ret %r;
^other:
  ret %r - %r;     // returns 0
}
```

## Quick-reference table

| Intrinsic | Signature | UB conditions | Result range |
|-----------|-----------|---------------|--------------|
| `@abs` | `(iN) → iN` | `x == INT_MIN_N` | `[0, INT_MAX_N]` |
| `@min` | `(iN, iN) → iN` | — | `[INT_MIN_N, INT_MAX_N]` |
| `@max` | `(iN, iN) → iN` | — | `[INT_MIN_N, INT_MAX_N]` |
| `@clz` | `(iN) → iN` | `x == 0` | `[0, N−1]` |
| `@ctz` | `(iN) → iN` | `x == 0` | `[0, N−1]` |
| `@popcount` | `(iN) → iN` | result > `INT_MAX_N` (narrow `N` only) | `[0, N]` |
| `@abs_diff` | `(iN, iN) → iN` | `\|a − b\| > INT_MAX_N` | `[0, INT_MAX_N]` |
| `@signum` | `(iN) → iN` | — | `{−1, 0, +1}` |
| `@clamp` | `(iN, iN, iN) → iN` | `lo > hi` (signed) | `[lo, hi]` |
| `@midpoint` | `(iN, iN) → iN` | — | `[INT_MIN_N, INT_MAX_N]` |
| `@parity` | `(iN) → i1` | — | `{0, -1}` |
| `@bswap` | `(iN) → iN`, `N % 8 == 0` | — (declaration rejected if `N` is not a multiple of 8) | full `iN` |
| `@bitreverse` | `(iN) → iN` | — | full `iN` |
| `@rotl` | `(iN, iN) → iN` | `n < 0` or `n >= N` | full `iN` |
| `@rotr` | `(iN, iN) → iN` | `n < 0` or `n >= N` | full `iN` |
| `@is_pow2` | `(iN) → i1` | — | `{0, -1}` |
| `@ilog2` | `(iN) → iN` | `x <= 0` (signed) | `[0, N − 2]` |
| `@wrapping_add` | `(iN, iN) → iN` | — | full `iN` |
| `@wrapping_sub` | `(iN, iN) → iN` | — | full `iN` |
| `@wrapping_mul` | `(iN, iN) → iN` | — | full `iN` |
| `@wrapping_neg` | `(iN) → iN` | — | full `iN` |
| `@wrapping_shl` | `(iN, iN) → iN` | `n < 0` or `n >= N` | full `iN` |
| `@wrapping_shr` | `(iN, iN) → iN` | `n < 0` or `n >= N` | full `iN` |
| `@saturating_add` | `(iN, iN) → iN` | — | `[INT_MIN_N, INT_MAX_N]` |
| `@saturating_sub` | `(iN, iN) → iN` | — | `[INT_MIN_N, INT_MAX_N]` |
| `@saturating_mul` | `(iN, iN) → iN` | — | `[INT_MIN_N, INT_MAX_N]` |
| `@saturating_neg` | `(iN) → iN` | — | `[INT_MIN_N, INT_MAX_N]` |
| `@div_euclid` | `(iN, iN) → iN` | `b == 0` or `(a == INT_MIN_N ∧ b == -1)` | `[INT_MIN_N, INT_MAX_N]` |
| `@rem_euclid` | `(iN, iN) → iN` | `b == 0` or `(a == INT_MIN_N ∧ b == -1)` | `[0, \|b\| − 1]` |
| `@fabs` | `(fN) → fN` | — | `[+0.0, +max]` |
| `@fneg` | `(fN) → fN` | — | full finite `fN` |
| `@copysign` | `(fN, fN) → fN` | — | full finite `fN` |
| `@signbit` | `(fN) → i1` | — | `{0, -1}` |
| `@to_bits` | `(f32) → i32`, `(f64) → i64` | — | finite-float bit patterns (never an `±∞`/NaN encoding) |
| `@from_bits` | `(i32) → f32`, `(i64) → f64` | pattern decodes to `±∞` or NaN | finite `fN` |
| `@is_normal` | `(fN) → i1` | — | `{0, -1}` |
| `@is_subnormal` | `(fN) → i1` | — | `{0, -1}` |
| `@fmin` | `(fN, fN) → fN` | — | full finite `fN` |
| `@fmax` | `(fN, fN) → fN` | — | full finite `fN` |
| `@sqrt` | `(fN) → fN` | `x < 0` (NaN result) | `[0, +max]` |
| `@floor` | `(fN) → fN` | — | integral `fN` |
| `@ceil` | `(fN) → fN` | — | integral `fN` |
| `@trunc` | `(fN) → fN` | — | integral `fN` |
| `@fract` | `(fN) → fN` | — | `(-1, 1)` |
| `@recip` | `(fN) → fN` | result non-finite (`x = ±0.0` or overflow) | finite `fN` |
| `@crc32_update` | `(i32, iN) → i32`, `N ∈ {8, 16, 24, 32, 40, 48, 56, 64}` | — | full `i32` |
| `@check_chksum` | `(i32, i32) → i32` | `expected != actual` (mandatory trap on every target, kept under `--no-ub-guards`; UB in symiri) | `= actual` on match |
| `@observe` | `(T) → T` | — | `= x` (identity, any `iN`/`fN`; observable `volatile` write in C) |
| `@reduce_add` | `(<N> T) → T`, `T ∈ iN, fN` | int: partial sum out of `iN` range; fp: non-finite intermediate; any `undef` lane | scalar `T` |
| `@reduce_min` | `(<N> T) → T`, `T ∈ iN, fN` | any `undef` lane | scalar `T` |
| `@reduce_max` | `(<N> T) → T`, `T ∈ iN, fN` | any `undef` lane | scalar `T` |
| `@reduce_and` | `(<N> iN) → iN` | any `undef` lane | full `iN` |
| `@reduce_or` | `(<N> iN) → iN` | any `undef` lane | full `iN` |
| `@reduce_xor` | `(<N> iN) → iN` | any `undef` lane | full `iN` |

## Adding a new intrinsic

Every piece below has to be in place before a new intrinsic merges. Delegating the semantics to the target language is not an option: RefractIR owns them, and a target that disagrees is a bug in the lowering.

1. Declare the signature in the program, `intrinsic @name(...) : T;`.
2. Interpreter: a branch in `Interpreter::callIntrinsic`, throwing `UndefinedBehaviorError` on every UB precondition.
3. Solver: a branch in `SymbolicExecutor::callBuiltinIntrinsicSMT`, pushing the UB guards onto `pc`.
4. C: a branch in `CBackend::emitIntrinsicHelper`, with `__builtin_trap()` on the UB paths.
5. WASM: a branch in `WasmBackend::emitIntrinsicHelper`, with `unreachable` on the UB paths.
6. Python: a helper in the module preamble, raising `RefractIRTrap` on the UB paths.
7. Documentation: the intrinsic in spec §12 and in this file.
8. Tests: `test/interp/`, `test/compile/`, `test/solver/` and `test/xval/`.

The file for each tool is in the table at the top of this document.

Specify six things: the signature, the UB preconditions (which inputs make the result undefined), the result range, the SMT encoding as an exact BV formula or ITE chain, the interpreter algorithm, and the lowering pattern per backend.

## Priority tiers and rejection policy

The shipped intrinsics above are only the seed of the design space. The target surface spans the full C `<math.h>` / WASM numeric / Rust `iN`/`fN` API, and RefractIR deliberately does not intend to implement all of it: many functions (transcendentals, recursive number theory) have no efficient SMT-BV / QF_FP encoding, and shipping them anyway would let users write programs that hang the solver. Each candidate is therefore classified by how cleanly the solver and the C backend can handle it, since those two drive prioritization; WASM comes second to last, and anything the solver cannot handle precisely comes last.

### Rejection layers

Each intrinsic carries per-backend support flags. Calling an unsupported intrinsic produces an error at the earliest layer that can detect it:

| Layer | Rejection means | Used for |
|---|---|---|
| Frontend (semantic checker) | `intrinsic @x` declaration is refused. Program will not parse. | Truly nonsensical or stateful intrinsics: `@rand`, `@time`, anything whose *value* is non-deterministic or state-dependent, anything that produces non-finite FP. (`@observe` is admitted: its value is the deterministic identity; only its C *lowering* has an observable write, see §12.7.) |
| Solver (`symirsolve`) | Declaration and program are accepted; reaching a `call @x` on a *symbolic* path makes that path infeasible (the same effect as UB pruning). Concrete-only paths still solve. | Transcendentals, recursive number theory, anything without a precise SMT encoding. |
| WASM backend | Compile-time error from `symirc --target wasm`: "`@x` not lowerable to WASM target". | No shipped intrinsic uses this layer. It is the designated home for a P3 libm-backed intrinsic whose WASM polyfill would diverge from the C target, as `wasi-libc` and `glibc` do in the last ULP. |
| Interpreter | Runtime error, distinct from UB: "intrinsic not implemented in this build". | Nothing. The interpreter is the reference oracle, and this layer stays empty. |

The consistency rule behind all of this: the interpreter agrees with every other backend on every value. When a libm-based intrinsic is supported by both the C backend and the interpreter, both link against the same libm, so cross-validation is byte-equal by construction.

### Tier summary

| Tier | Solver | C | WASM | Interp | Status |
|---|---|---|---|---|---|
| P0 | trivial to easy | trivial | trivial to easy | trivial | Shipped, in four batches (below) |
| P1 | trivial to easy | trivial | easy, as composed lowerings; `@remainder` hard | trivial | Unblocked; ships on demand (below) |
| P2 | medium: a bounded encoding that may time out | trivial | medium | trivial | Behind a feature flag |
| P3 | rejects the path | trivial via libm | rejected, or libm with ULP drift | trivial via libm | Deferred |
| P4 | — | — | — | — | Rejected at the frontend, permanently |

### P0

P0 splits into four groups by domain, all four shipped:

* Batch A, integer extras (§12.3): `@abs_diff`, `@signum`, `@clamp`, `@midpoint`.
* Batch B, bit-manipulation (§12.4): `@parity`, `@bswap`, `@bitreverse`, `@rotl`, `@rotr`, `@is_pow2`, `@ilog2`.
* Batch C, the integer overflow-aware family, scalar-result subset (§12.5): `@wrapping_{add,sub,mul,neg,shl,shr}`, `@saturating_{add,sub,mul,neg}`, `@div_euclid`, `@rem_euclid`.
* Batch D, the floating-point basic IEEE family (§12.6), in five sub-batches: D.1 sign and bit ops (`@fabs`, `@fneg`, `@copysign`, `@signbit`, `@to_bits`, `@from_bits`), D.2 classification predicates (`@is_normal`, `@is_subnormal`), D.3 min and max (`@fmin`, `@fmax`), D.4 correctly-rounded math (`@sqrt`, `@floor`, `@ceil`, `@trunc`), and D.5 compositions (`@fract`, `@recip`). Every entry maps to a QF_FP operation and a direct WASM `fN.*` opcode or a short composition of them, lowering at the native FP precision with no widening-and-mask.

Two groups of P0 candidates are outstanding. The tuple-returning overflow operations `@checked_*` and `@overflowing_*`, and the cross-width `@widening_mul` (`iN x iN -> i2N`), wait on a multi-value return ABI. And `@fma`, `@rint`, `@to_degrees` and `@to_radians` were held out of batch D to keep the initial set trivially bit-exact across backends; each needs a care point settled first. `@fma` must lower to `fma()` or `fmaf()`, never to `x*y+z`, or the single rounding is lost. `@rint` needs a rounding-mode-independent lowering, `__builtin_roundeven*` or SSE4.1 `roundsd` with the RNE immediate baked in, since RefractIR grants no fenv access. `@to_degrees` and `@to_radians` need their conversion constant pinned to a bit pattern in the spec.

Exponent manipulation is not an intrinsic target at all. `@ldexp`, `@scalbn`, `@ilogb` and `@logb` were considered for batch D and dropped, because their symbolic story is not solver-friendly: scaling by `2^exp` for a symbolic integer `exp` has no clean QF_FP encoding, as constructing `2^exp` overflows independently of the finite product, and exponent extraction needs a normal-versus-subnormal bit split. Neither fits the design goal of predictable BV and QF_FP constraints with minimal nonlinearity, so they stay out of the language rather than shipping behind a partial encoding.

### §12.6 D.1 per-intrinsic spec

#### `@fabs`

```text
intrinsic @fabs(%x: fN) : fN;
```

Returns the magnitude of `%x` (clears the IEEE 754 sign bit). Never UB on the RefractIR finite-only domain.

SMT encoding: `fp.abs(x)`.
Interpreter: `std::fabs(x)`.
C: `__builtin_fabsf` / `__builtin_fabs`.
WASM: `f32.abs` / `f64.abs`.

#### `@fneg`

```text
intrinsic @fneg(%x: fN) : fN;
```

Flips the IEEE 754 sign bit. `@fneg(+0.0) == -0.0` and vice-versa. Distinct from the surface `-x` (which is the arithmetic `0 - x`).

SMT encoding: `fp.neg(x)`.
Interpreter: `-x`.
C: unary `-`.
WASM: `f32.neg` / `f64.neg`.

#### `@copysign`

```text
intrinsic @copysign(%x: fN, %y: fN) : fN;
```

Returns `%x` with the sign of `%y`. Both operands must be the same FP width.

SMT encoding: `ite(fp.isNegative(x) = fp.isNegative(y), x, fp.neg(x))`.
Interpreter: `std::copysign(x, y)`.
C: `__builtin_copysignf` / `__builtin_copysign`.
WASM: `f32.copysign` / `f64.copysign`.

#### `@signbit`

```text
intrinsic @signbit(%x: fN) : i1;
```

Returns `-1` (the `i1` true value) if `%x` has its IEEE 754 sign bit set (negative finite or `-0.0`), else `0`.

SMT encoding: `ite(fp.isNegative(x), bv1#1, bv1#0)`.
Interpreter: `std::signbit(x) ? 1 : 0`, sign-extended to `i1` by `makeInt`.
C: `__builtin_signbit(a) ? 1 : 0`, sign-extended to the `int8_t` i1 cell.
WASM: reinterpret as BV and test the high bit (`iN.reinterpret_fN` then `iN.lt_s` against zero), then sextend bit 0 so the result is `0` / `-1`.

#### `@to_bits`

```text
intrinsic @to_bits(%x: f32) : i32;
intrinsic @to_bits(%x: f64) : i64;
```

Returns the raw IEEE 754 bit pattern of `%x` reinterpreted as a signed integer of equal width. Width-matched: f32 ↔ i32, f64 ↔ i64.

SMT encoding: introduce a fresh BV `b` of width `N`, conjoin `fp.eq(x, ((_ to_fp eb sb) b))` to `PC`, return `b`. The encoding relies on RefractIR's finite-only domain (§2.9) for the bit pattern of `x` to be uniquely determined.
Interpreter: `memcpy` reinterpret.
C: `__builtin_memcpy(&r, &a, sizeof(r))`.
WASM: `i32.reinterpret_f32` / `i64.reinterpret_f64`.

#### `@from_bits`

```text
intrinsic @from_bits(%x: i32) : f32;
intrinsic @from_bits(%x: i64) : f64;
```

Returns the floating-point value whose IEEE 754 bit pattern equals `%x`. Width-matched. UB if the resulting value is not finite (that is, if the bit pattern decodes to `±∞` or NaN), consistent with the §2.9 finite-only domain.

SMT encoding: `r = ((_ to_fp eb sb) x)`, with UB-preconditions `not(fp.isInfinite(r))` and `not(fp.isNaN(r))` conjoined to `PC`.
Interpreter: `memcpy` reinterpret, then `std::isfinite` check.
C: `__builtin_memcpy(&r, &a, sizeof(r))` then `__builtin_isfinite` trap.
WASM: `fN.reinterpret_iN` then explicit `|r| < +inf` check (NaN makes the comparison false), trap via `unreachable`.

### §12.6 D.2 per-intrinsic spec

Both are unary predicates over `fN`, returning `i1`. Under RefractIR's finite-only domain (§2.9) the input is always finite, so the classification reduces to three categories: *normal*, *subnormal*, or *zero* (±0).

#### `@is_normal`

```text
intrinsic @is_normal(%x: fN) : i1;
```

Returns `-1` (the `i1` true value) iff `%x` is a normal IEEE 754 value (biased exponent in `[1, max-1]`). Returns `0` for ±0, subnormals, and (out-of-domain but defensively false) ±∞ and NaN.

SMT encoding: `ite(fp.isNormal(x), bv1#1, bv1#0)`.
Interpreter: `std::isnormal(x)`, sign-extended to `i1` by `makeInt`; for f32 inputs stored as `double`, narrow to `float` first so the f32-precision exponent range is what gets classified.
C: `__builtin_isnormal(a0)`, sign-extended to the `int8_t` i1 cell; the generic-macro dispatch picks the right precision from the helper's parameter type.
WASM: no native op. Compose by extracting the biased exponent from `iN.reinterpret_fN(x)` and testing `(exp != 0) && (exp != max)`, then sextend bit 0 so the result is `0` / `-1`.

#### `@is_subnormal`

```text
intrinsic @is_subnormal(%x: fN) : i1;
```

Returns `-1` (the `i1` true value) iff `%x` is a subnormal IEEE 754 value (biased exponent == 0 and mantissa != 0). `±0` is not subnormal; for it the predicate returns `0`.

SMT encoding: `ite(fp.isSubnormal(x), bv1#1, bv1#0)`.
Interpreter: `std::fpclassify(x) == FP_SUBNORMAL`, sign-extended to `i1` by `makeInt`; same f32 narrowing as `@is_normal`.
C: compose from `__builtin_isnormal` and `!= 0.0`: `__builtin_isfinite(a0) && !__builtin_isnormal(a0) && a0 != 0.0`, sign-extended to the `int8_t` i1 cell.
WASM: no native op. Compose by reinterpreting and testing `(exp_bits == 0) && (mantissa != 0)`, then sextend bit 0 so the result is `0` / `-1`.

### §12.6 D.3 per-intrinsic spec

`@fmin` / `@fmax` are binary FP intrinsics declared at any `fN` width:

```text
intrinsic @fmin(%x: fN, %y: fN) : fN;
intrinsic @fmax(%x: fN, %y: fN) : fN;
```

They follow IEEE 754-2008 `minNum` / `maxNum`: under RefractIR's finite-only domain (§2.9) the result is the smaller / larger operand, and on the signed-zero pair the result is `-0` for `fmin` and `+0` for `fmax`. No UB on the RefractIR domain.

The signed-zero tie-break is *implementation-defined* in C's `fmin` / `fmax` and in SMT-LIB's `fp.min` / `fp.max`, so every backend except WASM emits an explicit tie to stay bit-exact with WASM's `fN.min` / `fN.max` (which already follow IEEE 754-2008).

#### `@fmin`

Returns the smaller of `%x` and `%y`; for the signed-zero pair returns `-0`.

SMT encoding:
```
ite(fp.lt(x, y), x,
ite(fp.lt(y, x), y,
ite(fp.isNeg(x), x, y)))   ; tie: pick -0 (fp.isNeg is true for -0)
```
Interpreter: `x < y ? x : y < x ? y : (signbit(x) ? x : y)`.
C: `(a0 < a1) ? a0 : (a1 < a0) ? a1 : __builtin_signbit(a0) ? a0 : a1`.
WASM: native `f32.min` / `f64.min`.

#### `@fmax`

Returns the larger of `%x` and `%y`; for the signed-zero pair returns `+0`.

SMT encoding:
```
ite(fp.gt(x, y), x,
ite(fp.gt(y, x), y,
ite(fp.isNeg(x), y, x)))   ; tie: pick +0 (use the other operand if x is -0)
```
Interpreter: `x > y ? x : y > x ? y : (signbit(x) ? y : x)`.
C: `(a0 > a1) ? a0 : (a1 > a0) ? a1 : __builtin_signbit(a0) ? a1 : a0`.
WASM: native `f32.max` / `f64.max`.

### §12.6 D.4 per-intrinsic spec

The four correctly-rounded math intrinsics are unary `fN → fN`. Each maps to a single IEEE 754 correctly-rounded operation; the lowering uses the operand's native precision directly (no widening-and-mask). Computing at the declared width is what keeps the four backends bit-exact: a double `sqrt` narrowed to `float` afterward would double-round, so the f32 helpers use the single-precision builtin / opcode (`sqrtf`, `f32.sqrt`).

```text
intrinsic @sqrt(%x: fN) : fN;
intrinsic @floor(%x: fN) : fN;
intrinsic @ceil(%x: fN) : fN;
intrinsic @trunc(%x: fN) : fN;
```

#### `@sqrt`

Correctly-rounded square root (RNE).

UB conditions:
- `%x < 0` (strictly negative): the IEEE result is NaN, which is outside the §2.9 finite-only domain. `@sqrt(-0.0) == -0.0` is *not* UB (the result is a finite signed zero). No overflow is possible; `sqrt` never increases magnitude beyond `+∞`.

All four backends enforce the same result-finiteness guard (the finite-only domain of `float.md` §1, the same guard `@from_bits` carries): the square root is computed, then the path is UB if the result is `±∞` or NaN. For sqrt only NaN is reachable (the operand was strictly negative), but the guard is written as full finiteness so every backend's rule is structurally identical.

SMT encoding: `fp.sqrt(RNE, x)`, with `not(fp.isInfinite(r))` and `not(fp.isNaN(r))` conjoined to `PC`.
Interpreter: `sqrtf` (f32) / `sqrt` (f64); trap if `!std::isfinite(r)`.
C: `__builtin_sqrtf` / `__builtin_sqrt`, then `__builtin_trap()` if `!__builtin_isfinite(r)`.
WASM: `fN.sqrt`, then the `(|r| < +inf)` finiteness check: `i32.eqz; if; unreachable; end`.

#### `@floor`, `@ceil`, `@trunc`

Round to an integral value toward `−∞` (`@floor`), toward `+∞` (`@ceil`), or toward zero (`@trunc`). Never UB: the result of rounding a finite value to an integer is always a representable finite value of the same type.

| Intrinsic | SMT (`fp.roundToIntegral`) | Interp | C | WASM |
|---|---|---|---|---|
| `@floor` | RTN | `std::floor` | `__builtin_floorf/floor` | `fN.floor` |
| `@ceil`  | RTP | `std::ceil`  | `__builtin_ceilf/ceil`   | `fN.ceil`  |
| `@trunc` | RTZ | `std::trunc` | `__builtin_truncf/trunc` | `fN.trunc` |

### §12.6 D.5 per-intrinsic spec

Two unary `fN → fN` intrinsics, each a short composition of the correctly-rounded primitives. Native FP precision (no widening); the f32 helpers use single-precision division so the result is bit-exact with the C and WASM backends.

```text
intrinsic @fract(%x: fN) : fN;
intrinsic @recip(%x: fN) : fN;
```

#### `@fract`

Fractional part: `@fract(x) = x - trunc(x)`. The sign follows `x` (`fract(-2.5) = -0.5`), matching Rust's `fN::fract`. The result magnitude is always `< 1`, so it never leaves the finite domain: no UB (like `@trunc`).

SMT encoding: `fp.sub(x, fp.roundToIntegral(RTZ, x))`, with no UB guard.
Interpreter: `x - std::trunc(x)` at the operand precision.
C: `a0 - __builtin_truncf(a0)` / `a0 - __builtin_trunc(a0)`.
WASM: `fN.trunc` then `fN.sub`.

#### `@recip`

Reciprocal: `@recip(x) = 1/x`, RNE. This reuses the FP `/` operator, so it carries the same UB rule.

UB conditions:
- The result is non-finite (`±∞`). This covers `x = ±0.0` (`1/±0.0 = ±∞`) and a tiny `|x|` whose reciprocal overflows (e.g. `recip` of a subnormal). `recip(±0.0)` is UB; unlike `@sqrt(-0.0)`, signed zero is not a defined input here.

SMT encoding: `fp.div(RNE, 1.0, x)`, with `not(fp.isInfinite(r))` and `not(fp.isNaN(r))` conjoined to `PC` (the same finiteness guard the solver applies to every division).
Interpreter: `1.0f / x` (f32) / `1.0 / x` (f64); trap if `!std::isfinite(r)`.
C: `1.0 / a0`, then `__builtin_trap()` if `!__builtin_isfinite(r)` (the explicit check is required, since `-fsanitize=undefined` does not catch overflow-to-∞; this matches the `/` operator's lowering).
WASM: `fN.div`, then the `(|r| < +inf)` finiteness guard via a scratch local (same shape as `@sqrt`).

## 12.7 Checksum primitives

Two intrinsics support the reify pipeline's opaque return-value oracle (see [reify.md](./reify.md)). They are excluded from the random intrinsic whitelist (`include/reify/intrinsic_whitelist.hpp`) so rysmith / rylink never synthesise them in body code; they are only emitted by the post-solve checksum rewriter and by the `@main` wrapper.

The pair is also unique within §12 for being non-mathematical: they exist to defeat compiler folding of the return value, not to model a target operation. The C lowering carries a function-local `static` lookup table and a `static __attribute__((noinline))` qualifier so the optimizer cannot clone the body or propagate the table contents back through CCP / SCCP; both pieces are essential to the "compiler-opaque" property the rewriter relies on.

### `@crc32_update`: table-driven CRC32 update step

```text
intrinsic @crc32_update(%state: i32, %val: iN) : i32;
```

Folds the iN value `%val` byte-wise into the running CRC32 state `%state` using the reflected polynomial `0xEDB88320`, LSB-first byte order, no initial XOR and no final XOR. Overloaded over `N ∈ {8, 16, 24, 32, 40, 48, 56, 64}`: any integer width is accepted and rounded up to whole bytes via `nBytes = (N + 7) / 8`; widths that are not a multiple of 8 zero-pad the high bits of the last byte. Pure function: no UB, no side effects, deterministic for any `(state, val)` pair.

Recurrence (per byte):
```
state' = (state >> 8) ^ tab[(state ^ byte) & 0xFF]
byte_i  = (val >> (8 * i)) & 0xFF
```
with the lookup table built once from the polynomial:
```
tab[i] = repeat 8 times: c = (c & 1) ? ((c >> 1) ^ 0xEDB88320) : (c >> 1)
         starting from c = i
```

Interpreter (`src/interp/intrinsics.cpp::Crc32UpdateIntrinsic`): function-local `static` table built lazily on first call; the recurrence is run directly on a `uint32_t state` and the final value is sign-extended to i32 via `makeInt(32, …)`.

C (`src/backend/c_intrinsics.cpp::Crc32UpdateIntrinsic`): same lazy-init pattern, marked `static __attribute__((noinline))` (via the `CIntrinsic::linkageQualifier()` hook) so the body is not cloned into callers. Each `(width)` overload gets its own helper name and its own copy of the 1 KB lookup table; for the typical i8 / i16 / i32 / i64 quartet that's ~4 KB of `.bss` per .c file, the explicit cost of keeping the helpers free of shared globals.

Python (`src/backend/py_intrinsics.cpp::_in_crc32_update`): same table-driven recurrence over unbounded Python integers, masked to uint32 per step; the table is built lazily at module level.

WASM (`src/backend/wasm_intrinsics.cpp::Crc32UpdateIntrinsic`): table-free. WASM has no cheap module-level mutable table storage, so the helper runs the defining LFSR recurrence directly: per byte, `s ^= byte` then eight rounds of `s = (s >>u 1) ^ ((s & 1) * 0xEDB88320)`. The round function is linear over GF(2), so eight rounds on the full register equal the table form `(s >> 8) ^ tab[(s ^ byte) & 0xFF]`; the values are bit-exact with the interpreter and C lowerings (checked by the self-verifying `test/sbackend/intrinsics_crc32_*` programs, which trap through `@check_chksum` on any mismatch). No imports, no linear-memory footprint. Opacity to the optimizer is not a concern here: the backend emits final WAT with no optimizer behind it.

Solver: not encoded. `@crc32_update` is never reached on a *symbolic* path: the rewriter applies it post-solve to the already-concrete exit block, and the reify pipeline keeps the solver on the original sum-based contract. A symbolic call to it on a solver-visited path makes the path infeasible (same effect as a UB prune).

### `@check_chksum`: equality predicate with abort-on-mismatch

```text
intrinsic @check_chksum(%expected: i32, %actual: i32) : i32;
```

Signature is fixed at i32 for every slot: the running CRC state is always i32, so accepting other widths would only invite implicit-truncation bugs. Returns `%actual` verbatim on equality; diverges on mismatch (UB in symiri, `abort()` in C). The fail-side side effect is what defeats observation folding in the compiled program: the C optimizer cannot fold the call away because `fprintf(stderr, …)` and `abort()` are both externally visible.

Interpreter (`src/interp/intrinsics.cpp::CheckChksumIntrinsic`): throws `UndefinedBehaviorError` on mismatch with the expected / actual values in the message; returns `actual` on equality.

C (`src/backend/c_intrinsics.cpp::CheckChksumIntrinsic`): emits
```c
if (a0 != a1) {
  fprintf(stderr, "@check_chksum mismatch: expected=%d actual=%d\n",
          (int)a0, (int)a1);
  abort();
}
return a1;
```
The C backend's preamble already `#include`s `<stdio.h>` and `<stdlib.h>` unconditionally so the helper itself needs no further options. Note that the program as a whole should be linked with `-lm`: generated code may call libm functions (`fmod` / `fmodf` from floating-point `%`, and the FP intrinsic helpers), which live in a separate library on toolchains older than glibc 2.35 and on musl.

Python (`src/backend/py_intrinsics.cpp::_in_check_chksum`): raises `RefractIRTrap("@check_chksum mismatch")` on mismatch (nonzero exit), returns `actual` on equality.

WASM (`src/backend/wasm_intrinsics.cpp::CheckChksumIntrinsic`): `if (expected != actual) unreachable`, then returns `actual`. The trap is the WASM-native analogue of the C lowering's `fprintf` + `abort`: externally visible divergence without any host imports, consistent with the backend's no-host-stdio model.

Solver: not encoded. Only the rylink-generated `@main` wrapper calls it, and that wrapper is the end of execution; no SMT path ever needs to reason about its post-state.

### `@observe`: observability beacon

```text
intrinsic @observe(%x: T) : T;      // T is any iN or fN
```

Value semantics is the identity: `@observe(v)` returns `v` unchanged, at any integer or floating-point width. The float domain matters because the leaf a beacon is planted on may be of any scalar type: a diverging loop whose state is entirely floating-point still needs an anchor. It is a deterministic, oracle-consistent pure function like every other intrinsic; the interpreter, solver, WASM, and Python all lower it to the identity. Its purpose is a lowering property: the C backend emits an observable `volatile` write of `v`, a side effect the optimizer must preserve. This anchors a computation the compiler cannot prove dead, most usefully the body of a deliberately non-terminating loop (`rysmith --require-nonterm`), keeping the loop alive even under a forward-progress-assuming toolchain.

Because its *value* is the identity, `@observe` is not in the "impure / stateful" category the frontend rejects (see *Rejection layers*): `@rand` / `@time` return non-deterministic or state-dependent values and would break cross-validation, whereas `@observe` returns the same value on every call and its observable write produces no value that enters the oracle comparison.

Interpreter (`ObserveIntrinsic`): returns `args[0]`. Solver (`ObserveIntrinsic`): returns `argVals[0]`, with no UB and no effect on the symbolic state. C (`ObserveCIntrinsic` / `ObserveCFpIntrinsic`): a `static __attribute__((noinline))` helper with `static volatile <T> __rir_observe_sink; __rir_observe_sink = a0; return a0;`; the volatile store is unremovable and noinline stops a caller from folding through the identity return. The float variant is identical but typed on `float` / `double`.
WASM / Python: plain identity; neither target has a forward-progress assumption, so a diverging loop is preserved without an observable side effect.

## 12.8 Horizontal vector reductions

Reductions fold every lane of a vector `<N> T` into one scalar `T`. They are the only intrinsic family whose parameter is a vector rather than a scalar.

```text
intrinsic @reduce_add(%v: <N> T) : T;    // T ∈ iN, fN
intrinsic @reduce_min(%v: <N> T) : T;    // T ∈ iN, fN
intrinsic @reduce_max(%v: <N> T) : T;    // T ∈ iN, fN
intrinsic @reduce_and(%v: <N> T) : T;    // T ∈ iN only
intrinsic @reduce_or (%v: <N> T) : T;    // T ∈ iN only
intrinsic @reduce_xor(%v: <N> T) : T;    // T ∈ iN only
```

Declaration well-formedness (`src/frontend/semchecker.cpp::checkIntrinsicDecl`): the sole parameter must be a vector `<N> T`; the return type must be exactly the element type `T`; and `T` must be in the intrinsic's family: any `iN` or `fN` for `@reduce_add`/`@reduce_min`/`@reduce_max`, integer-only for the bitwise `@reduce_and`/`@reduce_or`/`@reduce_xor`. The typechecker (`src/frontend/typechecker.cpp`) admits the vector parameter (its coarse "iN/fN only" gate is relaxed to "iN, fN, or a vector of these") and disambiguates overloads by the argument's *exact* vector type, so `@reduce_add(<4> i32)` and `@reduce_add(<8> i32)` coexist as distinct declarations. `@reduce_mul` is not provided: an `N−1`-deep chain of symbolic `bvmul` / `fp.mul` is nonlinear and solver-hostile (spec §12.4, §13).

The fold is the normative left-to-right order `((v[0] ⊕ v[1]) ⊕ v[2]) ⊕ …`. It reads every lane, so an `undef` lane is UB (rule 22: the solver conjoins each lane's definedness, the interpreter raises on the first undef lane). Each step is an ordinary scalar operation carrying the ordinary scalar UB. Solver impls live in `src/solver/intrinsics.cpp` (`ReduceIntSolverIntrinsic` / `ReduceFpSolverIntrinsic`), the interpreter in `src/interp/intrinsics.cpp` (`ReduceIntrinsic`); the fp/int split is routed by the scalar return type, so add/min/max register in both the integer and floating-point solver dispatch.

### `@reduce_add`: sequential lane sum

Folds lanes with `+`. Order-dependent: intermediate overflow (int) and FP non-associativity are observable, so no reassociation.

UB conditions:
- Integer: any partial sum outside the signed `iN` range (rule 4).
- Float: a `±∞` or NaN intermediate (rules 6-7).

SMT encoding: `acc = v[0]`; then per lane `k`, integer `acc = bvadd(acc, v[k])` with a per-step `not bvsaddo(acc, v[k])` UB guard; float `acc = fp.add(RNE, acc, v[k])` with a per-step `not (fp.isInfinite ∨ fp.isNaN)` guard (same encoding as the scalar `+` operator).
Interpreter: int fold in `__int128` with a `[INT_MIN_N, INT_MAX_N]` range check per step; float fold via `checkFPResult` (narrow to `fN`, reject non-finite) per step.

### `@reduce_min`, `@reduce_max`: horizontal minimum / maximum

Fold lanes keeping the smaller / larger. Order-independent, so a backend may use a pairwise or hardware reduction. No UB (finite-domain FP makes min/max total). Signed integers use signed comparison. Floating point uses the IEEE 754 minNum / maxNum signed-zero tie-break, identical to `@fmin` / `@fmax` (§12.6 D.3): `@reduce_min` yields `-0.0` when any lane is `-0.0`, `@reduce_max` yields `+0.0` when any lane is `+0.0`, regardless of lane order (a strict-comparison fold would make `±0.0` order-observable).

SMT encoding:
- Integer: `acc = ite(bvslt(v[k], acc), v[k], acc)` for min, `bvsgt` for max.
- Float: the three-level `@fmin` / `@fmax` signed-zero `ite` per step (`fpMinFold` / `fpMaxFold`, sharing `FminSolverIntrinsic` / `FmaxSolverIntrinsic`'s tie-break).

Interpreter: int `acc = v[k] < acc ? v[k] : acc` (resp. `>`); float uses the `signbit`-based `@fmin` / `@fmax` tie-break.

### `@reduce_and`, `@reduce_or`, `@reduce_xor`: bitwise fold

Fold lanes with `&` / `|` / `^`. Integer element type only. Order-independent; no UB.

SMT encoding: `acc = bvand / bvor / bvxor (acc, v[k])`.
Interpreter: `acc &= v[k]` / `|=` / `^=` on the sign-extended lane values, re-masked to `iN` by `makeInt`.

### Backend lowering

Each target lowers a reduction to a helper function that folds the lanes, named `<prefix>_reduce_<op>_v<N>_<elem>` so distinct vector shapes get distinct helpers. Helper-name mangling lives in `intrinsicHelperName` (C / WASM); the call site emits an ordinary call.

- C (`CBackend::emitReductionHelper`): `static inline T H(T a0, …, T a{N-1}) { … }`; the helper takes the `N` lanes as scalar parameters, and the call site emits them per-lane via `emitVecExprLane`. Because no vector crosses the C boundary, this is strategy-independent: it works under every `--vec-lowering` strategy, including `scalars` and `array`. `@reduce_add` accumulates in a wider type (`int64`/`__int128`) with a per-step range `__builtin_trap`; FP add traps on `!__builtin_isfinite`; FP min/max use the `__builtin_signbit` tie-break. Guards route through the `--no-ub-guards` sink.
- WASM (`WasmBackend::emitReductionHelper`): the vector arrives by address (the frame-memory spill ABI is packed under every strategy), so lanes are read with `<elem>.load offset=k*size` (the same layout `scalars::unpackParam` uses), independent of `--vec-lowering`. Folds with native ops (`i32.add`, `i32.and`, native `fN.min`/`fN.max` for the FP min/max tie-break, compare+`select` for integer min/max). Consistent with the backend's stance that arithmetic UB is not sanitized on WASM (`docs/symirc.md`), the fold emits no overflow / finiteness guards; a program that relies on `@reduce_add` trapping on overflow is `// SKIP: WASM`.
- Python (`PyBackend::emitReductionHelperDefs`): the vector arrives as a lane list, folded by a generated `_in_reduce_<op>_<i|f>(v[, n])` reading each lane through `_rd` (undef → trap). Being the strictest target, it keeps the UB guards: `@reduce_add` traps on an out-of-range int partial sum or a non-finite fp intermediate; f32 steps round through `_f32`. Under `--no-ub-guards` the trap lines are dropped (value semantics preserved).

Only the order-insensitive members (`min`/`max`/`and`/`or`/`xor`) may use hardware reductions; `@reduce_add`'s sequential fold is reproduced bit-exactly (the C target cross-validates against the interpreter).

### Example

```text
intrinsic @reduce_add(%v: <4> i32) : i32;

fun @main() : i32 {
  let %v: <4> i32 = {1, 2, 3, 4};
  let mut %r: i32 = 0;
^entry:
  %r = call @reduce_add(%v);   // %r == 10
  ret %r;
}
```

Synthesis example: pick per-lane values in `[0,10]` summing to 25:

```text
intrinsic @reduce_add(%v: <4> i32) : i32;

fun @main() : i32 {
  sym %?v: value <4> i32 in [0, 10];
  let mut %t: <4> i32 = 0;
^entry:
  %t = %?v;
  require call @reduce_add(%t) == 25, "lanes sum to 25";
  ret 0;
}
```

### P1: solver-easy, composed WASM lowerings

Solver and C lowerings are trivial for every member of this tier, and each needs a small composition on WASM rather than a direct opcode. That composition machinery exists on every backend, from multi-step integer helpers up to the loop-based `@crc32_update`, FP compositions like `@fract` and `@recip`, and native v128 vector storage, so WASM difficulty does not decide the tier. What gates each member individually:

Ready, awaiting demand: `@ffs` (`ctz+1` with `0` mapping to `0`), `@next_pow2`, `@fdim`, `@fpclassify`, `@total_cmp`, `@nextafter` (a `@to_bits` and `@from_bits` composition), `@round` (away from zero, distinct from the deferred ties-to-even `@rint`, and bit-exact as `trunc(x)` plus an exact half-way comparison, where the naive `trunc(x + copysign(0.5, x))` double-rounds), `@fmod` (the truncated remainder, not `@remainder`, and a reuse of the `%` operator's inline expansion that already ships everywhere), and saturating float-to-int conversions on the native `iN.trunc_sat_fM_s` opcodes.

Blocked on the multi-value return ABI: `@modf` and `@frexp`, the same blocker as P0's tuple-returning family, and they ship together with it.

Genuinely WASM-hard: `@remainder`, IEEE `fp.rem`. SMT and libm are trivial, but a bit-exact WASM composition needs an iterative reduction, since `x - rint(x/y)*y` double-rounds and the quotient can exceed integer precision.

### P2: solver-feasible but expensive

Encodable in SMT but at quadratic-or-worse cost. Gate behind `--enable-experimental-intrinsics` and enforce a per-call solver timeout with a clear "intrinsic @X is too expensive on this path" diagnostic.

`@isqrt` (bit-by-bit synthesis), `@pow(b, e)` with symbolic `e` (bounded unroll up to `--max-pow-iters`), `@ilog10` (decimal-power ITE chain), `@hypot` with both args symbolic (nonlinear FP).

### P3: solver rejects the path, libm on host targets

The C backend and the interpreter link the host libm; calling these is fine on concrete inputs and on symbolic paths the user does not intend to solve. Reaching one on a path passed to `symirsolve` prunes the path, identically to UB.

The WASM backend rejects these at compile time. `wasi-libc`'s libm diverges from `glibc` at the last ULP for transcendentals, which would silently break xval; the C-target-only contract is simpler than tolerating per-target ULP drift in test harnesses.

- Transcendentals: `@exp`, `@exp2`, `@expm1`, `@log`, `@log2`, `@log10`, `@log1p`, `@pow`/`@powf` with symbolic exponent, `@cbrt`, `@sin`, `@cos`, `@tan`, `@asin`, `@acos`, `@atan`, `@atan2`, `@sin_cos`, `@sinh`, `@cosh`, `@tanh`, `@asinh`, `@acosh`, `@atanh`, `@erf`, `@erfc`, `@tgamma`, `@lgamma`. Bessel (`@j0`/`@y0`/`@jn`/`@yn`) only if there is concrete demand.
- Number theory: `@gcd`, `@lcm`, `@ilog(base)` with symbolic base.

### P4: rejected at the frontend

A declaration of any of these is refused outright, since there is no point letting a user write a program RefractIR cannot reason about.

Stateful and impure: `@rand`, `@srand`, `@time`, `@clock`, `@getpid`, environment access. Each breaks determinism, and re-execution has to give the same result.

I/O: `@printf`, `@scanf`, `@fopen`, and file or network handles.

Non-finite FP producers: `@nan` and `@inf`. The finite-only FP domain of spec §2.9 is a hard invariant.

Byte-level memory: `@memcpy` and `@memset`, deferred in spec §13. They need byte-addressable memory in the solver, which the BV-tag pointer encoding does not provide ([symirsolve.md](./symirsolve.md), pointer encoding). A solver that gained byte-addressable memory would move them to a higher tier.
