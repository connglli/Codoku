# Codoku

Codoku (short for *code sudoku*) offers renewable challenges for coding agents. Like a sudoku puzzle, a codoku puzzle asks a solver to fill a set of cells whose choices are linked by global constraints, except the constraints arise from static and dynamic program semantics rather than rows, columns, and blocks.

**What Is a Codoku?**
A puzzle masks a generated Python function with **typed cells** (`<FILL_*>` marks): identifiers, function names, numeric constants, operators, control keywords, and CFG labels. A solution fills every cell subject to global semantic constraints:

- **Static**: the completion parses and compiles, matches the declared control-flow graph, and draws each constant from the constant table at exactly its listed multiplicity (`1` and `1.0` count as distinct).
- **Dynamic**: running it on the given input follows the prescribed execution path block-for-block and returns the expected output.

Each choice can ripple through later statements, branches, loops, or the output, so locally valid fills may still invalidate the whole solution, while valid solutions are sparse in a large search space. Every puzzle ships with a witness solution, so at least one valid filling exists; the checker accepts *any* completion satisfying the constraints, not just the witness. New puzzles are generated from scratch by semantic reification, so fresh challenges can always be minted after model training.

## Build

Codokus come from RefractIR and Reify: RefractIR is the symbolic intermediate representation underneath, and Reify (`rysmith`) generates the random Python programs that puzzles are masked from. Solving puzzles needs only **Python 3**. Generating them also needs the `rysmith` binary (C++20 + Bitwuzla), or just use Docker:

```bash
docker build -t codoku:latest -f codokus/Dockerfile .
```

or

```bash
make rysmith codoku
```

## Create Codokus

```bash
codoku create --profile small --seed 42 -o tutorial/
```

Creation samples a profile until a puzzle passes acceptance: `small` straight-line functions with short paths, `medium` loop-driven functions with moderate paths, `large` branching functions with deep loops (`small`/`easy` and `large`/`hard` are aliases).

## Write Codokus

Fill in every cell and save the full program to `solution.py`. Agents may use any tools: running candidates, invoking the checker, writing their own search, but no tool reveals which choice belongs in which cell. What counts is finding a valid solution within budget.

## Check Codokus

```bash
codoku check tutorial/puzzle.py tutorial/solution.py
codoku analyze tutorial/puzzle.py
```

`check` returns `[PASS]`/`[FAIL]` with the violated constraint. `analyze` reports search-space size (the cartesian product of cell domains) and program complexity (Halstead vocabulary, DepDegree data flow, McCabe control flow).

## 📋 License

MIT.
