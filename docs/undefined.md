# RefractIR strict undefined behaviour

This document is a per-rule companion to spec §7. Each rule states what makes an operation undefined and how `symiri`, `symirc` and `symirsolve` each enforce it. The rule numbers are the spec's.

RefractIR uses strict UB: if any operation on the executed path triggers UB, the whole path is infeasible. `symiri` aborts on UB and exits non-zero. C emitted by `symirc` runs under UBSan with `-fno-sanitize-recover=all`, so UB traps the executable. `symirsolve` adds the UB-precluding constraint to the path condition, so a satisfiable model is one that avoids the UB.

`symirc --no-ub-guards` removes the dynamic guards described in the `symirc` notes below, meaning the explicit `__builtin_trap` / `unreachable` / runtime-helper checks, but not the UBSan-level ones, which belong to the C compiler invocation. It is sound only for a program known to be UB-free, where those guards never fire ([symirc.md](./symirc.md#omitting-ub-guards)). The reify tools set it automatically for their UB-free output.

## Scalar arithmetic (§7.1, §7.4)

### Rule 1, integer division or modulo by zero

`a / b` or `a % b` with `b == 0`.

`symiri` checks the divisor before the operation, `symirc` leaves it to UBSan's `integer-divide-by-zero`, and `symirsolve` adds `(distinct b 0)` per division site.

### Rule 2, out-of-bounds array access

`a[i]` where `a : [N] T` and `i < 0` or `i >= N`.

`symiri` bounds-checks every lvalue index, `symirc` relies on UBSan's `array-bounds` for fixed-size arrays, and `symirsolve` adds `(bvult i N)`, whose unsigned comparison handles a negative index as a large positive one.

### Rule 3, reading `undef`

Reading any leaf whose stored value is `undef`. This subsumes uninitialised locals, uninitialised pointer values and uninitialised vector lanes.

`symiri` tracks `undef` per leaf and throws on read. `symirc` initialises every scalar leaf at declaration and lowers `undef` to a literal `0` plus a path-pruning `assume(false)` token, so the compiled program either zero-reads or traps under UBSan. `symirsolve` carries an `is_defined` flag on every symbolic value and conjoins it at each read.

`DefiniteInitAnalysis` also catches this statically, conservatively and at warning level.

### Rule 4, signed integer overflow

`+`, `-`, `*` or `<<` whose result falls outside the signed range of the target width, and `INT_MIN / -1`. RefractIR treats `<<` as signed arithmetic rather than a bit-vector wrap, so `x << n` is UB when `x * 2^n` does not fit, and also when `x < 0`.

`symiri` applies a per-operation signed-overflow predicate, `symirc` relies on UBSan's `signed-integer-overflow,shift`, and `symirsolve` uses `bvsaddo` / `bvssubo` / `bvsmulo`, with an explicit reconstruct-and-compare for `<<`.

### Rule 5, overshift

`x << n`, `x >> n` or `x >>> n` with `n < 0` or `n >= width(x)`. This rule is about the amount; rule 4 covers the shifted result.

`symiri` range-checks the amount, `symirc` relies on UBSan's `shift`, and `symirsolve` adds `(bvult n width)`.

### Rule 6, FP overflow

Any `+`, `-`, `*` or `/` whose RNE-rounded result would be `±∞`. This covers finite-operand overflow and `x / ±0.0` for non-zero `x`.

`symiri` runs `std::isinf` after every FP operation. `symirc` relies on UBSan's `float-divide-by-zero` plus an explicit `isinf` check the C backend emits, since UBSan does not catch overflow to infinity by itself. `symirsolve` adds `(not (fp.isInfinite result))`.

### Rule 7, FP invalid

Any FP operation producing NaN, which covers `±0.0 / ±0.0` and `x % ±0.0`.

`symiri` runs `std::isnan` after every FP operation, `symirc` emits an `isnan` check after each one, and `symirsolve` adds `(not (fp.isNaN result))`.

### Rule 8, float-to-integer out of range

`fN as iM` where the value truncated toward zero falls outside `iM`.

`symiri` bounds-checks before the cast, `symirc` relies on UBSan's `float-cast-overflow`, and `symirsolve` adds `(bvsle min_iM rounded)` and `(bvsle rounded max_iM)`.

## Pointer UB (§7.5)

### Rule 9, null pointer dereference

`load %p` or `store %p, v` with `%p == null`.

`symiri` checks the pointer before any dereference, the emitted load or store traps under UBSan's null sanitiser, and `symirsolve` adds `(distinct %p (_ bv0 64))` at every dereference.

### Rule 10, out-of-bounds pointer arithmetic

Every pointer carries a provenance object (rule 15). `%p ± n` is UB when the resulting address falls outside `[base, base + size]` of that object. The one-past-the-end address is valid for arithmetic and equality, and UB to dereference (rule 11).

`symiri` tags every `ptr` runtime value with its `ObjectInfo`, a base address and a size in bytes, updates the offset on arithmetic, and aborts when the new offset leaves `[0, size]`. `symirc` lowers `addr` of an aggregate, `ptrindex` and `ptrfield` to arithmetic instrumented with `__builtin_object_size`, catching the wrap through UBSan's `pointer-overflow` and the in-bounds path through an emitted range check. `symirsolve` adds `(bvule new_off size)`, one-past-end included, and records each `addr` operand's provenance base and size for downstream propagation.

### Rule 11, out-of-bounds load or store

`load %p` or `store %p, v` where `%p` lies outside `[base, base + size)`. One-past-the-end counts as outside for the purpose of dereference.

`symiri` combines rule 10's `ObjectInfo` with a strict `< size` check at the dereference, `symirc` uses UBSan plus the emitted bound check, and `symirsolve` adds `(bvult offset size)`.

### Rule 12, cross-object pointer arithmetic

Forming a pointer by arithmetic that crosses out of one local's storage into another's.

`symiri` never produces such a value, since the offset would leave the known allocation and trigger rule 10 first. `symirc` catches the boundary crossing through UBSan's `pointer-overflow`. In `symirsolve` the non-overlap axioms of spec §9.4 make it infeasible by construction.

### Rule 13, uninitialised pointer dereference

`load %p` or `store %p, v` where `%p == undef`. This is rule 3 specialised to pointer values, kept separate for clarity.

`symiri` checks the per-leaf `undef` flag before the dereference. `symirc` initialises pointer leaves to `null`, so the dereference falls to rule 9. `symirsolve` conjoins `is_defined(%p)` at the dereference.

### Rule 14, cross-object pointer comparison

`<`, `<=`, `>` or `>=` between pointers into different originating objects. `==` and `!=` are always defined, since distinct objects have distinct addresses.

`symiri` checks that both operands share one `ObjectInfo`. `symirc` emits an explicit object-id check before the comparison and aborts on a cross-object compare, which UBSan does not cover. `symirsolve` requires both operands to share a base in the path condition.

### Rule 15, aggregate-derived pointer provenance

Every pointer derivation carries a provenance object, uniformly for arrays and structs, decided by the final access:

* `addr %lv` on a top-level local: provenance is `%lv`, the whole local.
* `addr lv.f` or `ptrfield <ptr>, f`: provenance is the immediate containing struct of `f`.
* `addr lv[i]` or `ptrindex <ptr>, i`: provenance is the immediate containing array.
* `<ptr> ± n`: provenance is unchanged.

Arithmetic that walks outside the provenance object is UB under rule 10. Arithmetic *within* a struct or array is allowed, and type discipline is preserved instead at the dereference, by rule 15b.

`symiri` gives a derived pointer the `ObjectInfo` of the immediate containing aggregate and checks arithmetic against that aggregate's bounds. `symirc` uses the aggregate's `sizeof` for the range check, with UBSan's `pointer-overflow` doing the boundary trap. `symirsolve` records a base address constant and a static byte size at each `addr` / `ptrindex` / `ptrfield`, which rule 10's path condition then uses.

### Rule 15b, typed-access mismatch

`load %p` or `store %p, v` through `%p : ptr T` is UB when the runtime address is not the start of a `T`-typed cell within the provenance object:

* In a `[N] U` object, every address `base + k*sizeof(U)` for `0 <= k < N` is a valid `U` cell. The pointer's type necessarily matches the element type, so in-bounds and element-aligned implies valid.
* In an `@S` object, only the offsets of fields whose declared type is `T` are valid cells. A `ptr i32` landing on the offset of an `i64` field is UB, even though the arithmetic stayed inside `@S`.
* Mid-cell, on a cell of another type, or straddling two cells: UB.

This is what makes rule 15 type-safe. Arithmetic is permissive, and the dereference respects the field types.

`symiri` walks the provenance object's layout at every `load` and `store`, finds the cell start nearest the runtime offset, and aborts when that cell's declared type does not match the pointer's static type or the offset is not a cell start. `symirc` precedes the load or store with a layout check generated from the struct's field table, a constant-folded disjunction over the valid offsets, with UBSan catching the trap path. `symirsolve` turns the path condition into a disjunction over the valid `T`-typed cell offsets, so a symbolic offset is constrained to one of them and any other offset forces the path condition false.

### Rule 16, `ptrindex` out of bounds

`ptrindex <ptr>, <i>` with `<ptr> : ptr [N] T` is UB when `i < 0` or `i > N`. `i == N` gives a valid non-dereferenceable address, good for arithmetic and equality.

`symiri` range-checks the index, `symirc` uses UBSan plus an emitted range check, and `symirsolve` adds `(bvsle 0 i)` and `(bvsle i N)`.

### Rule 17, navigation through `null`

`ptrindex <ptr>, <i>` or `ptrfield <ptr>, <f>` where `<ptr>` is `null`. This is caught at the navigation rather than at a later dereference, so the path prunes immediately.

`symiri` null-checks at the navigation site, `symirc` emits a pre-navigation null guard trapped by UBSan, and `symirsolve` adds `(distinct <ptr> (_ bv0 64))` at every navigation.

### Rule 18, navigation through `undef`

`ptrindex` and `ptrfield` read their pointer operand, so an `undef` operand is UB at the navigation site. This follows from rule 3.

`symiri` checks `is_defined` on the operand before computing the offset. `symirc` initialises pointer values to `null` upstream, so rule 17's guard catches it. `symirsolve` conjoins `is_defined(<ptr>)` at the navigation.

### Rule 19, navigation from a one-past-the-end pointer

`ptrindex <ptr>, <i>` or `ptrfield <ptr>, <f>` where `<ptr>` is exactly the one-past-the-end address of its provenance object. That address is valid for arithmetic and equality, but there is no element there to navigate into.

`symiri` checks `offset != size` before each navigation, `symirc` emits the guard alongside the rule-17 null check, and `symirsolve` adds `(distinct offset size)` at every navigation.

## Vector UB (§7.6)

### Rule 20, out-of-bounds vector lane access

`lv[i]`, read or write, where `lv : <N> T` and `i < 0` or `i >= N`.

`symiri` bounds-checks the lane index. `symirc` lowers `v[i]` to a subscript on the vector-extension type with an explicit pre-check, trapped by UBSan. `symirsolve` adds `(bvult i N)`, and for a symbolic `i` the `ite` chain of spec §9.5.4 defines only the in-range lanes, so an out-of-range index forces the path condition false.

### Rule 21, lane-wise scalar UB

Rules 1 to 8 apply per lane to vector operations, and UB in any one lane prunes the whole path. For `%a, %b : <4> i32`, `%a / %b` is UB when any lane of `%b` is zero and `%a + %b` is UB when any lane overflows; `%v as <4> f32` for `%v : <4> f64` is UB when any lane would overflow to `±∞`.

`symiri` iterates the lanes and applies each scalar rule per lane. `symirc` precedes the SIMD operation with per-lane checks, each gated by UBSan. `symirsolve` gives each lane its own conjunct, so the path is feasible only when every lane's check passes.

### Rule 22, reading an `undef` vector lane

Reading a lane whose value is `undef`, whether from a vector initialised with `undef` or from a lane not yet written by a lane-write or a whole-vector copy.

`symiri` tracks an `undef` flag per lane alongside the lane values. `symirc` initialises emitted vector locals lane by lane to `0` where the source says `undef`, so later reads hit defined storage. `symirsolve` carries an `is_defined` flag per lane and conjoins it at each lane read.

## Function call UB (§7.7)

### Rule 23, contract precondition violation

`call @f(...)` where `@f` is a contract-form `decl` and any `pre` clause is false at the call site. The path becomes infeasible.

`symiri` never reaches this rule, because it rejects a contract-form `decl` call before execution begins. `symirc` emits a check of each `pre` clause before the call, and a failed precondition calls `abort()` or `unreachable`. `symirsolve` evaluates each clause with the arguments bound to the parameters, and a false clause conjoins false to the path condition.

### Rule 24, callee UB propagation

UB inside a `fun` callee makes the caller's path infeasible. Call boundaries do not sandbox UB: any statement, condition or nested `call` in the callee that triggers any other rule prunes the calling path.

`symiri` propagates it as an exception that unwinds the interpreter's call stack to the top level. `symirc` monomorphises the callee into a single function, so the usual UBSan instrumentation catches it with no cross-function handling. `symirsolve` conjoins the callee's path condition to the caller's.

### Rule 25, intrinsic UB preconditions

An intrinsic whose declared semantics carries a precondition treats a violation as UB. Two patterns account for all of them: a result the declared return type cannot represent, as in `@abs(INT_MIN_N)`, and an operand outside the intrinsic's domain, as in `@clz(0)` or `@div_euclid(a, 0)`.

Each intrinsic's preconditions are stated with the intrinsic itself, in [intrinsics.md](./intrinsics.md) §12, alongside its per-tool behaviour. An intrinsic that carries no precondition is defined for every input in its signature.

Adding an intrinsic means declaring its preconditions in [intrinsics.md](./intrinsics.md), raising a UB exception in [src/interp/intrinsics.cpp](../src/interp/intrinsics.cpp), emitting a guard in each of [src/backend/c_intrinsics.cpp](../src/backend/c_intrinsics.cpp), [src/backend/wasm_intrinsics.cpp](../src/backend/wasm_intrinsics.cpp) and [src/backend/py_intrinsics.cpp](../src/backend/py_intrinsics.cpp), and conjoining the precondition to the path condition in [src/solver/intrinsics.cpp](../src/solver/intrinsics.cpp).

## What is not UB

A few things RefractIR deliberately defines where other languages do not.

Equality across objects. `==` and `!=` between pointers into different objects are always well-defined, and always false and true respectively. Only relational comparison is UB (rule 14).

The one-past-the-end address. Valid for arithmetic and equality, UB only when dereferenced (rule 11) or navigated through (rule 19).

Whole-vector copy. `%v = %w` for `%v, %w : <N> T` is a lane-by-lane copy with no overflow or aliasing concern.

`fmod` semantics for FP `%`. Aligned with integer `%`, truncating toward zero, rather than IEEE `fp.rem`. It adds no UB case beyond rule 7.

Signed `<<` of a non-negative `x` whose result fits. Well-defined arithmetic shift; only `x < 0` or overflow is UB (rule 4).

Static call-site errors. A call to an undeclared function, an argument count or type mismatch, a recursion cycle in the call graph, and a contract-form `decl` call under `symiri` are all semantic errors caught before execution, so they never reach the UB machinery.

Argument evaluation. UB in an argument expression, as in `call @f(load %null_ptr)`, fires during left-to-right argument evaluation, before the call transfers control. The existing scalar, pointer and vector rules cover it, and no call-specific rule is needed.

Literal range checks. Every integer literal, decimal, hex, octal or binary, is checked at type-check time against the signed two's-complement range of its inferred type, `[-2^(N-1), 2^(N-1)-1]`, and an out-of-range literal is a static error rather than a silent narrowing (spec §6.4, §6.12). Both `let %x: i8 = 200;` and `let %y: i32 = 0x80000000;` are rejected, and an author who meant the bit pattern with the high bit set writes `-128` or `-0x80000000`. The check applies to every `iN` including `i1`, whose representable values are `{0, -1}`, so the literal `1` in `i1` context is rejected. All three tools share the typechecker pass, so all three reject before any execution begins.

## Rule index

| # | Name | Spec |
|---|---|---|
| 1 | Integer division or modulo by zero | §7.1 |
| 2 | Out-of-bounds array access | §7.1 |
| 3 | Reading `undef` | §7.1 |
| 4 | Signed overflow | §7.1 |
| 5 | Overshift | §7.1 |
| 6 | FP overflow | §7.4 |
| 7 | FP invalid (NaN) | §7.4 |
| 8 | Float-to-int out of range | §7.4 |
| 9 | Null pointer dereference | §7.5 |
| 10 | Out-of-bounds pointer arithmetic | §7.5 |
| 11 | Out-of-bounds load or store | §7.5 |
| 12 | Cross-object pointer arithmetic | §7.5 |
| 13 | Uninitialised pointer dereference | §7.5 |
| 14 | Cross-object pointer comparison | §7.5 |
| 15 | Aggregate-derived pointer provenance | §7.5 |
| 15b | Typed-access mismatch | §7.5 |
| 16 | `ptrindex` out of bounds | §7.5 |
| 17 | Navigation through `null` | §7.5 |
| 18 | Navigation through `undef` | §7.5 |
| 19 | Navigation from one-past-the-end | §7.5 |
| 20 | Out-of-bounds vector lane access | §7.6 |
| 21 | Lane-wise scalar UB | §7.6 |
| 22 | Reading an `undef` vector lane | §7.6 |
| 23 | Contract precondition violation | §7.7 |
| 24 | Callee UB propagation | §7.7 |
| 25 | Intrinsic UB preconditions | §7.7 |
