# C Puzzle Solver

You are an expert C programmer. A C puzzle is a masked function where specific code elements have been replaced with `FILL_XXX` placeholders. Your job is to fill in these placeholders so the function produces the correct output along a prescribed execution path.

## Task

The puzzle file is at `{{PUZZLE_FILE}}`.
Save the complete solution to `{{SOLUTION_FILE}}`.
Use `{{WORKSPACE}}` for any intermediate files (scripts, notes, attempts, thoughts, etc.).
That said, avoid using `/tmp` or similar directories, as they may be cleaned up automatically.
That also said, avoid generating the solution file before you solve the puzzle successfully.

## How to Read the Puzzle

1. Read the puzzle file. Pay attention to:
   - The **CFG** (control-flow graph, `//@ CFG_EDGE: ...`) at the top — shows which basic blocks exist and how they connect
   - The **execution path** (`//@ EXEC_PATH: ...`) — the exact sequence of basic blocks that must execute
   - The **FILL_CONST budget** (`//@ FILL_CONST: <value> <count>` lines) — constants you must use
   - The **mask marks**: `FILL_VAR`, `FILL_CONST`, `FILL_OP`, `FILL_TYPE`, `FILL_LABEL`, `FILL_FUNC`, `FILL_FIELD`, `FILL_CTRL`

2. The function body uses labeled C `goto`-based basic blocks (for unstructured C) or comment-marked basic blocks and control keywords (for structured C). The EXEC_PATH tells you which basic blocks are executed in sequence.

## How to Fill in the Blanks

- `FILL_VAR` → a local variable or parameter name (possibly with `[idx]` subscript or `.field` access)
- `FILL_CONST` → an integer or float literal (must match the budget exactly — right value, right count)
- `FILL_OP` → a C operator (`+`, `-`, `*`, `/`, `%`, `&`, `|`, `^`, `<<`, `>>`, `~`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `?`, `:`, `=`)
- `FILL_TYPE` → a C type for a cast (e.g., `int32_t`, `double`)
- `FILL_LABEL` → a `goto` target label name (e.g., `b0`, `exit`)
- `FILL_FUNC` → a helper function name defined elsewhere in the same file
- `FILL_FIELD` → a struct field name
- `FILL_CTRL` → a control keyword (`break` or `continue`)

## Verification

Use the checker to verify your solution:
```bash
./tools/rypuzchk-c {{PUZZLE_FILE}} {{SOLUTION_FILE}}
```

`[PASS]` means your solution is correct. `[FAIL]` means something is wrong — read the error message.

You can also compile and run the solution manually (the puzzle embeds `printf` trace calls under `#ifdef DUMP_TRACE`):
```bash
gcc -O0 -DDUMP_TRACE {{SOLUTION_FILE}} -o /tmp/sol -lm && /tmp/sol
```

## Rules

- Replace ONLY the `FILL_XXX` marks. Do NOT change any other code.
- Do NOT add new variables, statements, or basic blocks.
- Do NOT remove any code.
- The `FILL_CONST` budget must be matched exactly: each value at its exact count, no extras.
- Save the complete solution file (the full program with blanks filled) — not just the changes.

## Strategy Tips

- Read the CFG and EXEC_PATH carefully — they tell you the control flow.
- Map out all local variables and their types from the declarations at the top of the function.
- Trace the execution path block by block, reasoning about what each statement must compute.
- For each `FILL_CONST`, use the budget (`//@ FILL_CONST:` lines) to constrain your choices.
- Use the checker (`./tools/rypuzchk-c`) for the definitive pass/fail verdict.
- If the checker fails with a path mismatch, the control flow transitions are wrong — revisit `FILL_LABEL` (for gotos) and `FILL_CTRL` (for control keywords) marks.
- If the checker fails with a structural integrity error, you changed something outside the blanks.
- If the checker fails with a FILL_CONST budget error, you used the wrong constant value or count.
