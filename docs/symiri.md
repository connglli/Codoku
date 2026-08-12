# symiri

`symiri` is the reference interpreter for RefractIR. It executes a `.sir` program directly, and its behaviour is the definition of correct execution: the compiler backends and the solver are validated against it.

## Usage

```bash
symiri <input.sir> [args...] [--main <function>] [--sym name=value ...] [options]
```

Entry defaults to `@main`. Positional arguments after the input file bind the entry function's parameters in declaration order. The full option list is `symiri --help`, declared in [src/symiri.cpp](../src/symiri.cpp).

Interpret a concrete program, then a specific entry function:

```bash
symiri prog.sir
symiri prog.sir --main @f0
```

Interpret a symbolic program, binding every symbol it declares:

```bash
symiri prog.sir --main @f0 --sym @?c4=3 --sym %?k=10
```

## Symbol binding

A program that declares symbols runs only when every symbol is bound; a missing or unrecognised binding is an error. Bound values are immutable for the whole run, exactly as a solver-chosen value would be.

A value is parsed as a decimal literal against the symbol's declared type: integers against the declared bit width, floats through `refractir::parseFloatLiteral` ([float.md](./float.md) §9).

A vector symbol takes one value per lane, `--sym '%?v=1,2,3,4'`, and the count must equal the lane count. A single value splats across all lanes.

## Execution

Expressions evaluate left to right, `select` evaluates only the arm it takes, and division and modulo round toward zero (spec §2.4, §2.7, §8).

Undefined behaviour aborts the run with a diagnostic naming the rule and the source span, and the process exits non-zero. [undefined.md](./undefined.md) states each rule and how the interpreter enforces it.

`--check` stops after the frontend, reporting type and semantic errors without executing. `--dump-trace` prints each executed block and every variable update. `--max-bbl-steps <n>` aborts after entering `n` blocks, which bounds a program that may not terminate. `--dump-call` prints each function calls.

## Result reporting

A terminating run prints the entry function's return value on a `Result:` line: a decimal integer, a hex float, `ptr(0x…)`, or `void`. Hex-float form (`printf("%a", …)`) is bit-exact by construction, so cross-validation compares the line against the C backend's output byte for byte ([float.md](./float.md) §11.4).
