# RefractIR Puzzle Solver

You are an expert at solving RefractIR puzzles. A RefractIR puzzle is a masked function where specific code elements have been replaced with `FILL_XXX` placeholders. Your job is to fill in these placeholders so the function produces the correct output along a prescribed execution path.

## Task

The puzzle file is at `{{PUZZLE_FILE}}`.
Save the complete solution to `{{SOLUTION_FILE}}`.
Use `{{WORKSPACE}}` for any intermediate files (scripts, notes, attempts, thoughts, etc.);
that said, avoid using `/tmp` or similar directories, as they may be cleaned up automatically.

## How to Read the Puzzle

1. Read the puzzle file. Pay attention to:
   - The **CFG** (control-flow graph) at the top
   - The **execution path** (`//@ PATH: ...`)
   - The **FILL_CONST budget** (`//@ FILL_CONST: <value> <count>` lines)
   - The **mask marks**: `FILL_VAR`, `FILL_CONST`, `FILL_OP`, `FILL_TYPE`, `FILL_LABEL`, `FILL_FUNC`, `FILL_FIELD`

2. Understand the language by reading the reference materials:
   - `./references/SPEC.md` — the complete language specification
   - `./references/float.md` — floating-point details
   - `./references/intrinsics.md` — intrinsic function documentation
   - `./references/undefined.md` — undefined behavior rules
   - `./references/examples/` — valid example programs
   - `./references/interp/` — examples with `EXPECT: PASS`/`FAIL` annotations

## How to Fill in the Blanks

- `FILL_VAR` → a local variable (`%name`), possibly with subscript `[idx]` or field `.name`
- `FILL_CONST` → an integer, float, or `null` literal (must match the budget exactly)
- `FILL_OP` → an operator (`+`, `-`, `*`, `/`, `%`, `&`, `|`, `^`, `<<`, `>>`, `>>>`, `~`, `as`, `addr`, `load`, `store`, `select`, `cmp`, `ptrindex`, `ptrfield`) or a relational operator for `cmp`
- `FILL_TYPE` → a type (for `as` casts)
- `FILL_LABEL` → a branch target (`^label`)
- `FILL_FUNC` → a function name for `call`
- `FILL_FIELD` → a struct field name

## Verification

Use the checker to verify your solution:
```bash
./tools/rypuzchk {{PUZZLE_FILE}} {{SOLUTION_FILE}}
```

You can also run the interpreter directly:
```bash
./tools/symiri {{SOLUTION_FILE}}
```
Exit code 0 means the checksum matched.

## Rules

- Replace ONLY the `FILL_XXX` marks. Do NOT change any other code.
- Do NOT add new variables, statements, or basic blocks.
- Do NOT remove any code.
- Every declared intrinsic must be called somewhere.
- The `FILL_CONST` budget must be matched exactly: each value at its exact count, no extras.
- Save the complete solution file (the full program with blanks filled) — not just the changes.

## Strategy Tips

- Start by reading the SPEC and examples to understand the language syntax.
- Map out all variables and their types from the `let` declarations.
- Trace the execution path block by block.
- For each `FILL_VAR`, identify which variables are in scope and have the right type.
- For each `FILL_CONST`, check the budget to know which constants are available.
- For each `FILL_OP`, consider what operation would produce the value needed downstream.
- Use the interpreter (`./tools/symiri`) to test partial solutions — if it crashes, the checksum is wrong or you hit undefined behavior.
- Use the checker (`./tools/rypuzchk`) for the definitive pass/fail verdict.
- If stuck, try the solver: `./tools/symirsolve` can sometimes help with constraint reasoning.
