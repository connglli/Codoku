# RefractIR Puzzles — `rypuzmk` & `rypuzchk`

A **RefractIR puzzle** is a masked, [rysmith](./reify.md)-generated function: a
concrete leaf function whose body statements have been replaced by `FILL_XXX`
placeholders, packaged with an instruction banner and a single self-checking
test case. The challenge is to fill the blanks so that the function reproduces
the expected output along a prescribed execution path.

Two tools implement the game:

| Tool | Role |
| :--- | :--- |
| `rypuzmk` | **Puzzle maker** — runs `rysmith`, masks the generated leaf, and emits a puzzle (`.sir`) plus an optional ground-truth solution. |
| `rypuzchk` | **Puzzle checker** — validates a candidate solution against a puzzle and prints `[PASS]` / `[FAIL]`. |

Both are thin drivers over the shared frontend; the masking model, the
constant-budget collector and the helper utilities live in
[`puzzle/puzzle_common.hpp`](../puzzle/puzzle_common.hpp).

---

## 1. Anatomy of a puzzle

A generated puzzle file has two parts:

1. An **instruction banner** (a block of `//` comments): the CFG, the task, the
   execution path, the validation command, reference material, the general
   rules, and the `FILL_CONST` budget.
2. The **masked program**: the leaf function with its body blocks masked, an
   unmasked `@main` wrapper that calls the leaf and checks the result, and the
   intrinsic declarations.

Only the leaf function's **interior** is masked. The first block (`^entry`) and
the last block (`^exit`) are printed verbatim — `^entry` seeds the initial
state and `^exit` computes the checksum that defines correctness. Everything
strictly between them is replaced with `FILL_XXX` marks.

### The mask alphabet

Masking is a lossy, token-level rewrite. Each mark stands for a *kind* of
syntactic element, not a specific one:

| Mark | Replaces |
| :--- | :--- |
| `FILL_VAR` | an lvalue (a `%local`, optionally with `[index]` / `.field` accesses) **or** a variable used as a coefficient — the whole lvalue collapses to one mark |
| `FILL_CONST` | an integer / float / `null` literal in an operand position |
| `FILL_OP` | **any** operator or operator-like keyword: expression `+`/`-`; atom `* / % & | ^ << >> >>>`; the `as` cast keyword; unary `~`; `addr`, `load`, `store`, `select`, `cmp`, `ptrindex`, `ptrfield`; a relational operator |
| `FILL_TYPE` | a type (only emitted for the destination type of an `as` cast) |
| `FILL_LABEL` | a branch target (`^label`) — **unstructured mode only** |
| `FILL_CTRL` | a structured control-flow keyword (`break` or `continue`) — **structured mode only** |
| `FILL_FUNC` | a callee name in `call FILL_FUNC(...)` |
| `FILL_FIELD` | a struct field name in `ptrfield` |

`FILL_LABEL` and `FILL_CTRL` are mutually exclusive: unstructured C uses
goto-based control flow and produces `FILL_LABEL` for branch targets, while
structured C uses `break`/`continue` statements and produces `FILL_CTRL` for
those keywords.  A `FILL_CTRL` mark is validated indirectly through path
execution (FAIL_PATH) and CFG topology (FAIL_CFG), not by a dedicated check.

Because the rewrite is lossy, **many** concrete statements mask to the same
skeleton. That ambiguity is intentional — it is what makes a puzzle a puzzle —
but it also bounds what the checker can enforce (see §4).

### `let` initialisers

Local declarations keep their *types* (those are given), but their
*initialisers* are masked the same way, with two carve-outs:

- The sentinels `0` and `1` are left visible (they are structural — array
  fills, counters, identity elements — and not interesting to guess).
- `_`-prefixed scratch locals (e.g. the checksum accumulator `%_chk`) keep
  their initialisers verbatim.

### Machine-readable markers

The banner embeds two marker families that `rypuzchk` parses. They are the
authoritative, stable interface between the two tools — the surrounding prose
can change freely:

```
//@ EXEC_PATH: entry -> b0 -> b1 -> ... -> exit
//@ CFG_EDGE: A -> B
//@ FILL_CONST: <value> <count>
```

`//@ EXEC_PATH:` is the prescribed leaf execution path. Each `//@ CFG_EDGE:` line
declares a directed edge in the control flow graph. Each `//@ FILL_CONST:` line
is one entry of the constant budget (see §3).

### `require` / `assume` are dropped

Pure assertions — `require` and `assume` — are removed from **both** the puzzle
and the ground truth. They are path conditions / properties, not part of the
observable computation, and the embedded checksum is the sole correctness
oracle. Dropping them keeps the skeleton focused on code the solver must
actually reconstruct.

---

## 2. The correctness contract

The `@main` wrapper is fixed and looks like:

```
fun @main() : i32 {
  let mut %r: i32 = undef;
^entry:
  %r = call @<leaf>(<concrete args>);
  %r = call @check_chksum(<expected>, %r);
  ret 0;
}
```

The leaf's `^exit` block folds the **final values of every observable variable**
into a CRC via `@crc32_update`, and `@check_chksum` **aborts** (non-zero exit)
when the result does not match the embedded expected value. Therefore:

> A solution is *correct* iff it passes all checker validation stages in order.

This is the operational definition the checker enforces. It is **not** "recover
the exact original source": several different fills can satisfy all conditions
(see §4).

---

## 3. What `rypuzchk` checks

Given `rypuzchk <puzzle.sir> <solution.sir>`, the validator performs checks in order, from easiest to hardest:

1. **Basics (FAIL_BASICS)**: load the puzzle and solution, verify banner requirements (`//@ EXEC_PATH` and `//@ CFG_EDGE`), and verify that the solution contains no unfilled `FILL_XXX` marks.
2. **Re-masking (FAIL_REMASKING)**: re-mask the solution and require it to match the puzzle skeleton byte-for-byte (modulo comments and whitespace). This is the anti-cheating check.
3. **Parse/Compile (FAIL_PARSE)**: parse the solution (to AST) and/or compile it (with gcc) to verify that it is a valid, compilable program.
4. **CFG (FAIL_CFG)**: build the solution's CFG and verify that its edge set matches the declared `//@ CFG_EDGE:` markers exactly. This ensures branch targets in `FILL_LABEL` are topologically correct before execution.
5. **Path (FAIL_PATH)**: execute the solution and require the trace to match the expected path exactly.
6. **Output (FAIL_OUTPUT)**: check that the exit code is 0 (check_chksum and correctness check passed).
7. **`FILL_CONST` budget (FAIL_FILL_CONST)**: verify that the constant budget matches the budget declared in the banner exactly.
8. **Intrinsics**: every declared intrinsic must be called somewhere (validated under structural integrity / re-masking).

### The `FILL_CONST` budget, precisely

The budget is the multiset of constants that masking *hides* — counted **only**
at masked positions:

- every constant inside a masked body block (no magnitude threshold: `0`, `17`,
  `-0.625` and `1177419489738997077` are all budgeted), and
- every masked `let`-initialiser constant (excluding the visible `0`/`1`
  sentinels and `_`-prefixed locals).

Constants in the fixed `^entry`/`^exit` code are **not** in the budget — they
are already on screen and the solver must not re-introduce them. Constants
absorbed into a `FILL_VAR` (an lvalue base or its subscript/field) are likewise
not counted, because the printer collapses the whole lvalue to one mark.

The producer (`rypuzmk`'s banner) and the consumer (`rypuzchk`'s check) share a
single implementation, `MaskedConstantCollector`, so they cannot drift apart.

---

## 4. Intended degrees of freedom

Because masking is lossy, three classes of variation are accepted by design —
any two solutions that agree on (skeleton, path, constant multiset, observable
final state) are indistinguishable, and all are "correct":

- **Dead state.** A masked statement whose effect is never observed by the
  `^exit` checksum (e.g. a pointer that is never loaded, or a value overwritten
  before exit) may be filled with a different variable or value.
- **Operator kind.** `FILL_OP` does not pin the operator. Any operator that
  yields an observably-equivalent result is accepted (`x - 0` ≡ `x + 0`).
- **Callee identity.** `FILL_FUNC` pins only the call *arity*. Two intrinsics
  with a compatible signature are interchangeable where the result is
  observably equal.

What is **not** free: anything the checksum, the path, the constant budget, or
the skeleton can see — i.e. the actual computed result, the branch structure
and loop counts, the exact constants used, and the statement/atom/block shape.

---

## 5. Usage

### Make a puzzle

```bash
# Generate a puzzle (and its ground-truth solution) from a fresh rysmith run.
./rypuzmk -o puzzle.sir --keep-ground-truth

# Reproducible, with difficulty knobs forwarded to rysmith.
./rypuzmk --seed 42 --min-loop-iter 2 --n-bbls 5 --n-stmts 3 -o puzzle.sir

# Mask an existing concrete .sir instead of generating one.
./rypuzmk existing.sir -o puzzle.sir
```

Key options: `-o/--output`, `--keep-ground-truth` (writes `<output>.gt.sir`),
`-s/--seed`, `-L/--min-loop-iter`, `-B/--n-bbls`, `-S/--n-stmts`,
`--rysmith <path>`, `--pkg-res` (see below).

Pass `--structured` (C puzzle variant only) to emit structured control flow
(`for`/`while`/`do-while` loops with `break`/`continue`) instead of
`goto`-based flat blocks.  This replaces all `FILL_LABEL` marks with
`FILL_CTRL` marks and requires the solver to fill in `break` or `continue`
at each control-flow position.  The CFG topology check and path trace work
unchanged; the mask alphabet is the only difference.

`rypuzmk` validates the generated leaf (≥ 3 blocks, `^entry` first, `^exit`
last) and **self-checks** that the ground truth re-masks to the puzzle it
emits, so a structurally broken (unsolvable) puzzle is never shipped.

### Package a self-contained sandbox

The banner refers to its toolchain and reference material by relative path
(`./symiri`, `./rypuzchk`, `./references/SPEC.md`, `./references/examples/`,
`./references/interp/`, etc.). [`puzzle/pkgres.sh`](../puzzle/pkgres.sh)
populates a directory with exactly those resources:

```bash
./puzzle/pkgres.sh <dir>          # symlink the resources (default)
./puzzle/pkgres.sh <dir> --copy   # copy them (portable / self-contained)
```

It links/copies the RefractIR tools (`symiri`, `symirc`, `symirsolve`,
`rypuzchk`), the SMT solvers found on `PATH` (`z3`, `cvc5`, `bitwuzla`), the
latest `docs/SPEC_*.md` as `references/SPEC.md`, `examples/` as
`references/examples/`, and `test/interp/` as `references/interp/` (the
EXPECT-tagged good-and-bad examples).

`rypuzmk --pkg-res` runs this automatically against the puzzle's parent
directory, so a single command yields a ready-to-solve sandbox:

```bash
mkdir task && ./rypuzmk --seed 42 -o task/puzzle.sir --keep-ground-truth --pkg-res
cd task && ./rypuzchk puzzle.sir my_solution.sir
```

`--pkg-res` requires `--output`. `rypuzmk` locates `pkgres.sh` next to its own
binary; override with `--pkgres-script <path>` if needed.

### Check a solution

```bash
./rypuzchk puzzle.sir solution.sir          # uses ./symiri by default
./rypuzchk puzzle.sir solution.sir --symiri /path/to/symiri
```

Exit code `0` and `[PASS]` mean the solution satisfies every condition in §3.

---

## 6. Tests

End-to-end coverage lives in
[`test/unit/run_puzzle_tests.py`](../test/unit/run_puzzle_tests.py) and runs as
part of `make test-unit`:

```bash
python3 -m test.unit.run_puzzle_tests ./rypuzmk ./rypuzchk ./rysmith ./symiri
```

It asserts the round-trip invariant (ground truth round-trips across seeds), the
banner markers, that `require`/`assume` are dropped, and that the structural and
constant-budget checks reject the obvious cheats (unfilled marks,
deleted/duplicated statements, off-budget constants).
