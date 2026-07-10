# `test/puzzle` — rypuzchk solution-validation fixtures

Each `*.txt` here is one **puzzle + candidate solution** pair that exercises the
puzzle checker (`rypuzchk`). The runner is
[`test/lib/run_puzzle_test.py`](../lib/run_puzzle_test.py); it is wired into
`make test-unit`.

## File format

```
// EXPECT: PASS            (or FAIL)
// DESC: <one-line note on what this case exercises>
<verbatim puzzle>          rypuzmk output, including the //@ EXEC_PATH /
...                        //@ FILL_CONST banner the checker consumes
=>
<verbatim solution>        a full .sir program
...
```

- The leading `// EXPECT:` / `// DESC:` lines are **test metadata**; the runner
  strips them, so the embedded puzzle stays byte-for-byte what `rypuzmk` emitted.
- `=>` is a line that equals `=>` after stripping. It separates the puzzle from
  the solution. Neither RefractIR nor the puzzle banner contains such a line.
- `// EXPECT: PASS` means `rypuzchk` must **accept** the solution (exit 0);
  `// EXPECT: FAIL` means it must **reject** it (non-zero exit). Acceptance is
  judged by exit code alone, because the rejection reasons surface different
  messages (a bare `Error:` for an unparsable solution, a `[FAIL] ...` line for
  a semantic violation).

## Coverage

The fixtures span every way `rypuzchk` can accept or reject a solution.

Accepted (`EXPECT: PASS`):

| File | What it shows |
|------|---------------|
| `pass_ground_truth.txt`         | the exact ground truth is accepted (matches the puzzle) |
| `pass_partial_ground_truth.txt` | a `--p-mask 0.5` puzzle accepts its ground truth |
| `pass_alt_dead_store.txt`       | a *different* valid solution: a masked, dead pointer assignment filled with a different-but-equal pointer — same checksum, path, budget, and skeleton, yet not the ground truth |

Rejected (`EXPECT: FAIL`), one per correctness criterion:

| File | Criterion |
|------|-----------|
| `fail_wrong_value.txt`        | **correctness** — a structurally/budget-valid fill that reads the wrong variable; `check_chksum` rejects the result |
| `fail_unfilled_marks.txt`     | **parse** — a solution that still contains `FILL_XXX` marks |
| `fail_wrong_path.txt`         | **path** — swapped branch labels divert execution off the required `//@ EXEC_PATH` (the divergence also changes the result) |
| `fail_rename_local.txt`       | **structural** — alpha-renaming a local (semantics-preserving) rewrites non-FILL text |
| `fail_insert_stmt.txt`        | **structural** — duplicating an idempotent statement (semantics-preserving) adds a position the puzzle lacks |
| `fail_commute_revealed.txt`   | **structural** — commuting a *revealed* body expression (semantics-preserving) edits code outside the FILL marks |
| `fail_off_budget_const.txt`   | **budget** — a dead initialiser constant changed to an off-budget value executes correctly but violates the FILL_CONST budget |
| `fail_uncalled_intrinsic.txt` | **intrinsic usage** — a declared-but-uncalled intrinsic |

The three structural cases are *semantics-preserving*: they execute correctly
and keep the required path, so the structural re-mask is the only gate that can
legitimately reject them.

## Regenerating

The fixtures embed the output of `rypuzmk --seed 1 -B 3 -S 1` (and its
`--p-mask 0.5` variant). If the generator's output changes, regenerate the
puzzle/ground-truth pair and re-derive the solutions (the mutations are
described in each file's `// DESC:` line).
