#!/usr/bin/env python3
"""codoku — code sudoku puzzle generation & checking.

The current support is for the Python target only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BIN_DIR = Path(__file__).absolute().parent
RYSMITH = BIN_DIR / "rysmith"
RYPUZMK = BIN_DIR / "rypuzmk-tgt"
RYPUZCHK = BIN_DIR / "rypuzchk-tgt"

# Template for the per-puzzle INSTRUCTION.md.
INSTRUCTION_TEMPLATE = """\
# Python Puzzle Solver

You are an expert Python programmer. A Python puzzle is a masked function where specific code elements have been replaced with `FILL_XXX` placeholders. Your job is to fill in these placeholders so the function produces the correct output along a prescribed execution path.

## Task

The puzzle file is at `puzzle.py`.
Save the complete solution to `solution.py`.
Use `./scratch/` for any intermediate files (scripts, notes, attempts, thoughts, etc.).
That said, avoid using `/tmp` or similar directories, as they may be cleaned up automatically.
That also said, avoid generating the solution file before you solve the puzzle successfully.

## How to Read the Puzzle

1. Read the puzzle file. Pay attention to:
   - The **CFG** (control-flow graph, `#//@ CFG_EDGE: ...`) at the top — shows which basic blocks exist and how they connect
   - The **execution path** (`#//@ EXEC_PATH: ...`) — the exact sequence of basic blocks that must execute
   - The **FILL_CONST budget** (`#//@ FILL_CONST: <value> <count>` lines) — constants you must use
   - The **mask marks**: `FILL_VAR`, `FILL_CONST`, `FILL_OP`, `FILL_TYPE`, `FILL_LABEL`, `FILL_FUNC`, `FILL_FIELD`, `FILL_CTRL`

2. The function body uses comment-marked basic blocks and control keywords. The EXEC_PATH tells you which basic blocks are executed in sequence.

## How to Fill in the Blanks

- `FILL_VAR` → a local variable or parameter name (possibly with `[idx]` subscript)
- `FILL_CONST` → an integer, float, boolean, or None literal (must match the budget exactly — right value, right count)
- `FILL_OP` → a Python operator (`+`, `-`, `*`, `/`, `%`, `//`, `&`, `|`, `^`, `<<`, `>>`, `~`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `if`, `else`, `not`, `and`, `or`)
- `FILL_TYPE` → not used (Python is dynamically typed)
- `FILL_LABEL` → not used (Python has no goto-based blocks)
- `FILL_FUNC` → a function call name (e.g., `_cast_int`, `_padd`, `_pdiff`, `_peq`, `_prel`, `_load`, `_store`, `_pidx`, `_pfield`, or any function defined in the file)
- `FILL_FIELD` → not used
- `FILL_CTRL` → a control keyword (`break` or `continue`)

## Verification

Use the checker to verify your solution:
```bash
codoku check puzzle.py solution.py
```

`[PASS]` means your solution is valid. `[FAIL]` means something is wrong — read the error message.

You can also run the solution manually (the puzzle prints trace statements when DUMP_TRACE is set):
```bash
DUMP_TRACE=1 python solution.py
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
- For each `FILL_CONST`, use the budget (`#//@ FILL_CONST:` lines) to constrain your choices.
- Use the checker (`codoku check`) for the definitive pass/fail verdict.
- If the checker fails with a path mismatch, the control flow transitions are wrong — revisit `FILL_CTRL` (for control keywords) marks.
- If the checker fails with a structural integrity error, you changed something outside the blanks.
- If the checker fails with a FILL_CONST budget error, you used the wrong constant value or count.
"""


def run(cmd: list[str], cwd: Path | None = None) -> int:
  return subprocess.call(cmd, cwd=cwd)


def postprocess_puzzle(path: Path) -> None:
  """Rewrite the generated puzzle banner's validation command."""
  text = path.read_text()
  puzzle_name = path.name
  text = text.replace("./tools/rypuzchk", "codoku check")
  text = text.replace("[this_puzzle_file].py", puzzle_name)
  text = text.replace("[your_solution].py", "solution.py")
  path.write_text(text)


def write_instruction(path: Path) -> None:
  """Write the per-puzzle INSTRUCTION.md."""
  path.write_text(INSTRUCTION_TEMPLATE)


def check(puzzle: str, solution: str) -> int:
  return run([RYPUZCHK, puzzle, solution])


def generate(args: argparse.Namespace) -> int:
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  cmd = [
    RYPUZMK,
    "--target",
    "python",
    "--rysmith",
    RYSMITH,
    "--keep-ground-truth",
    "-o",
    "puzzle.py",
  ]
  if args.seed is not None:
    cmd += ["--seed", str(args.seed)]
  rc = run(cmd, cwd=outdir)
  if rc == 0:
    postprocess_puzzle(outdir / "puzzle.py")
    write_instruction(outdir / "INSTRUCTION.md")
    gt = outdir / "puzzle.gt.py"
    if gt.exists():
      oracle_dir = outdir / "oracle"
      oracle_dir.mkdir(parents=True, exist_ok=True)
      gt.replace(oracle_dir / "puzzle.gt.py")
  return rc


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="codoku",
    description="Puzzle generation & checking wrapper for the Python target.",
  )
  sub = parser.add_subparsers(dest="command", metavar="COMMAND")

  create = sub.add_parser(
    "create",
    help="generate one new codoku puzzle (default subcommand)",
    description="Generate one new codoku puzzle (output: <outdir>/puzzle.py).",
  )
  create.add_argument(
    "-o",
    "--outdir",
    default=".",
    help="output directory for the puzzle (default: .)",
  )
  create.add_argument(
    "--seed",
    type=int,
    default=None,
    help="master seed driving generation (default: random)",
  )

  checkp = sub.add_parser(
    "check",
    help="check a solution against a codoku puzzle",
    description="Check a solution against a puzzle.",
  )
  checkp.add_argument("puzzle", nargs="?", default="puzzle.py")
  checkp.add_argument("solution", nargs="?", default="solution.py")

  return parser


def main() -> int:
  argv = sys.argv[1:]
  parser = build_parser()
  if argv and argv[0] == "check":
    args = parser.parse_args(argv)
    return check(args.puzzle, args.solution)
  # Top-level help.
  if argv and argv[0] in ("-h", "--help"):
    parser.parse_args(["-h"])
    return 0
  # `create` (explicit or delegated): all remaining args are create options.
  if argv and argv[0] == "create":
    argv = argv[1:]
  args = parser.parse_args(["create"] + argv)
  return generate(args)


if __name__ == "__main__":
  sys.exit(main())
