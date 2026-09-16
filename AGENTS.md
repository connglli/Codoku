# Codoku: Agent Guideline

+ Knowledge Background: Program Analysis, Symbolic Execution, Optimizing Compilers
+ Implementation Language: Python 3 (`./codokus`)
+ Primary Source Directory: `./codokus`

## Project Overview

Codoku (short for *code sudoku*) offers renewable challenges for coding agents. A Codoku puzzle masks a generated Python function with **typed cells** (`<FILL_*>` marks): identifiers, function names, numeric constants, operators, control keywords, and CFG labels. A solution fills every cell subject to global semantic constraints:

1. **Static**: the completion parses and compiles, matches the declared control-flow graph, and draws each constant from the constant table at exactly its listed live and dead counts (`2` and `2.0` count as distinct).
2. **Dynamic**: on the given input it follows the prescribed execution path block-for-block and returns the expected output.

Each choice can ripple through later statements, branches, loops, or the output, so locally valid fills may still invalidate the whole solution, while valid solutions are sparse in a large search space.

The key design goals are:

- Any completion satisfying the constraints is *correct*: the checker accepts more than the witness, and the shipped witness guarantees one valid filling exists.
- Fresh puzzles mint from scratch by semantic reification, so a sample corpus renews after model training instead of going stale.
- Every validation stage is a property of the solution's text or behavior alone: the checker needs nothing but Python 3.

## Scope: the Codoku Center and the RefractIR Dependency

Like [README.md](./README.md), this project centers on Codoku. The RefractIR part is a dependency and must not be edited:

+ `src/`, `include/`, and `docs/` are RefractIR. They supply the intermediate representation, the shared frontend pipeline, and the backends. Understand them; do not patch them, because change requests belong upstream.
+ [docs/](./docs) holds the dependency's documentation and serves as reference material: [docs/SPEC_v0.2.3.md](./docs/SPEC_v0.2.3.md) defines the language, [docs/puzzle.md](./docs/puzzle.md) states the puzzle contract codoku mirrors, and [docs/AGENTS.md](./docs/AGENTS.md) owns the writing standard.
+ Repo-side edits live in `codokus/`, the Codoku suite in [test/unit/run_codoku_tests.py](./test/unit/run_codoku_tests.py), `codokus/Dockerfile`, `README.md`, and this file.

## Puzzle at a Glance

+ A puzzle is a generated leaf: a random Python function from `rysmith --target python`, wrapped in a fixed `@main` harness that checks the checksum, and carry the guarded memory model from [codokus/codoku_preamble.py](./codokus/codoku_preamble.py).
+ Cells: masking is a lossy, token-level rewrite; each cell stands for a *kind* of element, not a specific one:

| Cell | What it hides |
| :--- | :--- |
| `<FILL_VAR>` | a local variable or parameter name (possibly with `[idx]`) |
| `<FILL_CONST>` | a numeric literal (budgeted from the constant table, or free); the sentinels `0`, `1`, `0.0`, and `1.0` stay visible |
| `<FILL_OP>` | an operator or operator-like keyword |
| `<FILL_CTRL>` | a control keyword (`break`, `continue`) |
| `<FILL_LABEL>` | the destination of a general goto flag (`_go_<FILL_LABEL>`) |
| `_<FILL_CTRL>_<FILL_LABEL>` | a break/continue flag kind and destination together |
| `<FILL_FUNC>` | a non-internal callee defined in the same file |
| `<FILL_TYPE>`, `<FILL_FIELD>` | unused by the Python target |

+ Machine-readable markers are the authoritative interface between the producer (creation) and the consumer (checking); the surrounding prose may change freely:

```text
//@ EXEC_PATH: entry -> b0 -> ... -> exit
//@ CFG_EDGE: A -> B
//@ <FILL_CONST>: <value> <live> <dead>
//@ DISABLED_MASKS: <FILL_OP> <FILL_FUNC>
```

+ `<live>` counts the value's slots in blocks on the execution path (plus the pre-entry declarations, which always run); `<dead>` its slots in blocks off it. A constant parked in the wrong region fails even when the totals add up.
+ `DISABLED_MASKS` lists the kinds a profile disabled; their constructs stay visible in the puzzle. `DISABLEABLE_MASKS` in [codokus/codoku_complexity.py](./codokus/codoku_complexity.py) carries the disableable kinds; goto-flag compound tokens and control keywords cannot be disabled, so every flag target remains hidden.

## Toolchain Overview

| Tool | Role |
|------|------|
| `codoku` | CLI over [codokus/codoku.py](./codokus/codoku.py): `create`, `check`, `analyze` |
| [codokus/codoku_creator.py](./codokus/codoku_creator.py) | puzzle creation: profiles, rysmith invocation, preamble swap, masking, acceptance, installation |
| [codokus/codoku_checker.py](./codokus/codoku_checker.py) | solution validation: ordered stages from basics to the constant budget |
| [codokus/codoku_common.py](./codokus/codoku_common.py) | masking locators, maskable-statement scan, CFG extraction |
| [codokus/codoku_complexity.py](./codokus/codoku_complexity.py) | realized `PuzzleMetrics`, the disableable-kinds vocabulary, the solution-space and complexity estimate |
| [codokus/codoku_preamble.py](./codokus/codoku_preamble.py) | guarded memory model spliced into every puzzle |
| `codokus/Dockerfile` | self-contained image with the generator and the modules |

## Development Pipeline

### Generation (per accepted candidate)

```text
master seed
  ↓
sample profile config (ranges over rysmith knobs, disabled mask kinds)
  ↓
rysmith --target python
  ↓
swap_preamble (guarded memory model) + strip refractir_ prefix
  ↓
randomize_checksum (per-step operator sample + expected-value recalibration)
  ↓
mask_puzzle: cells ← per-element rolls (p_mask_ops, p_mask_lhs_vars, p_mask_rhs_vars, p_mask_funcs, p_mask_consts; ctrl and goto tokens always) → filter_disabled_masks
  ↓
self_check: ground truth re-masks byte-for-byte to the puzzle
  ↓
analyze_puzzle → profile_accepts (inclusive metric bounds)
  ↓
install: puzzle.py + INSTRUCTION.md + oracle/
```

### Validation (strict order, from easiest to hardest to reason about)

```text
basics (markers, unfilled cells) → parse → re-mask skeleton match → compile
  → CFG topology vs declared edges → timed run → path trace → checksum output
  → constant budget live/dead split
```

+ The checker infers the masked cells from the puzzle itself and re-masks the solution with the vocabulary it parses from the banner, so producer and consumer cannot drift (`filter_disabled_masks` is the shared mechanism).

## Testing: TDD Approach (MANDATORY)

ALWAYS follow a strict Test-Driven Development discipline.

### Required workflow

1. Write **five failing tests** that expose the bug or demonstrate the desired behavior
2. Run the tests **one by one** to confirm they fail
3. Implement the fix or feature
4. Run the tests **one by one** to confirm they pass
5. Add the **smallest additional test** that covers edge cases, in the correct test directory

### Never

1. Disable failing tests
2. Modify tests to avoid triggering bugs
3. Add workarounds that bypass the real issue
4. Implement features without a test demonstrating them first

### How to Run Tests

The Codoku suite is its own target:

+ Codoku directly: `python3 -m test.unit.run_codoku_tests codokus/codoku.py build/bin/rysmith` (generation needs the `rysmith` binary from the dependency build).
+ `make test-codoku` builds the generator first and runs the suite.

Every other `make test-*` target is RefractIR's tests: they exercise the dependency surface, so we never run them and assume they always pass. A Codoku change is gated by the Codoku suite alone.

## Dependency Management

+ Codoku modules are standard-library-only: every import in `codokus/` is Python's own, so puzzles, the checker, and any embedding carry no third-party dependency.
+ Generation additionally needs the `rysmith` binary (C++20 + Bitwuzla); build it from the dependency sources or pull the Docker image.

### C++ (dependency side)
+ Dependencies are managed **manually**
+ Prefer header-only or standard-library-only solutions
+ When introducing a new dependency:
  + Update `README.md`
  + Clearly document installation steps and versions

### Python (if used for tooling)
+ Virtual environment: `./venv`
+ Activate with:
  ```bash
  source venv/bin/activate
  ````

+ Dependencies:
  + `requirements.txt`: runtime
  + `requirements.dev.txt`: development
+ Always pin exact versions

## Principles and Best Practices

Always follow good practices:

1. Use git frequently and meaningfully
2. Follow **Conventional Commits**
3. Keep `README.md` and this file up to date with shipped behavior
4. Fix **all compiler warnings**
5. Keep a clean, layered project structure
6. Write high-quality comments that explain *why*, not *what*
7. Comments describe the current state, not the change history, unless it is a bugfix or a workaround for a critical known issue
8. Keep functions small, shallow, and focused on a single responsibility
9. Keep CHANGELOG concise (multiple related entries can be summarized in one line)
10. Follow [./docs/AGENTS.md](./docs/AGENTS.md) when writing documents, header comments, or anything else durable in prose

Always check whether a design/implementation is *elegant*:

(1) It retains a minimalist core and a clean conceptual model.
(2) It is simple enough that an experienced developer can understand it within five minutes without any explanation.

Always keep in mind the following principles to make it elegant before designing any new feature or changing existing behavior. Consider these principles, think twice, and then design:

1. KISS: Keep It Simple, Stupid. Is this the simplest design that works?
2. SINE: Simplicity Is Not Enough. Is this design analyzable, testable, and solver-friendly?
3. DRY: Don't Repeat Yourself. Are there existing abstractions/implementations that can be reused?
4. YAGNI: You Ain't Gonna Need It. Do we really need this feature now, or is it speculative?
5. SOLID: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion. Does this design adhere to these principles?

## Before Starting Work

1. Review recent history:

   ```bash
   git log [--oneline] [--stat] [--name-only] # Show brief/extended history
   git show [--summary] [--stat] [--name-only] <commit> # Show brief/extended history of a commit
   git diff <commit> <commit> # Compare two different commits
   git checkout <commit> # Checkout and inspect all the details of a commit
   ```
2. Understand existing design decisions before changing behavior
3. For large tasks, commit incrementally with clear messages

## Before Saving Changes

ALWAYS:

1. Clear all compiler warnings
2. Format code (the pre-commit hooks run `ruff check` / `ruff format` for Python and `clang-format` for C++)
3. Ensure all tests pass (timeouts excepted)
4. Check changes with `git status`
5. Split work into small, reviewable commits
6. Ask the user to review your changes before committing
7. Use Conventional Commit messages:

```text
<type>[optional scope]: <title>

<body>

[optional footer]
```

* Title ≤ 50 characters
* Body explains intent and design impact

**Remember:**
Codoku prioritizes *renewable, semantically-grounded challenges* over surface-level convenience.
Preserve these properties in every change.
