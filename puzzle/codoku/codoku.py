#!/usr/bin/env python3
"""codoku — code sudoku puzzle generation & checking.

The current support is for the Python target only.
"""

from __future__ import annotations

import argparse
import random
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

# Difficulty levels: each dimension is a range (seeded uniform draw) and the
# level caps the maximum difficulty score.  The score is
#   B*L + B + B*S*P        (path steps + hidden statements)
#   x2 when the constant budget applies.
# where, B, L, S, and P are options passed into rypuzmk-tgt.
DIFFICULTIES = {
  # A small function with a short path, where part of the code stays visible and constants need no maching.
  "easy": {
    "ranges": {
      "n_bbls": (2, 4),
      "n_stmts": (2, 3),
      "min_loop_iter": (0, 1),
      "p_mask": (0.5, 0.7),
    },
    "lift_consts": True,
    "cap": 25,
  },
  # A loop-driven function where every statement is masked and constants must match a budget.
  "medium": {
    "ranges": {
      "n_bbls": (3, 6),
      "n_stmts": (2, 4),
      "min_loop_iter": (2, 3),
      "p_mask": (0.8, 1.0),
    },
    "lift_consts": False,
    "cap": 100,
  },
  # A large branching function with deep loops, nothing visible but the skeleton, and a tight constant budget.
  "hard": {
    "ranges": {
      "n_bbls": (6, 10),
      "n_stmts": (3, 5),
      "min_loop_iter": (4, 6),
      "p_mask": (1.0, 1.0),
    },
    "lift_consts": False,
    "cap": 250,
  },
}

MAX_SAMPLES = 20


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


def difficulty_score(cfg: dict) -> int:
  score = (
    cfg["n_bbls"] * cfg["min_loop_iter"]
    + cfg["n_bbls"]
    + cfg["n_bbls"] * cfg["n_stmts"] * cfg["p_mask"]
  )
  if not cfg["lift_consts"]:
    score *= 2
  return round(score)


def pick_config(difficulty: str, seed: int | None) -> dict:
  """Sample a config within the level's ranges; keep it under the score cap."""
  spec = DIFFICULTIES[difficulty]
  rng = random.Random(seed)
  best = None
  for _ in range(MAX_SAMPLES):
    cfg = {
      "n_bbls": rng.randint(*spec["ranges"]["n_bbls"]),
      "n_stmts": rng.randint(*spec["ranges"]["n_stmts"]),
      "min_loop_iter": rng.randint(*spec["ranges"]["min_loop_iter"]),
      "p_mask": round(rng.uniform(*spec["ranges"]["p_mask"]), 2),
      "lift_consts": spec["lift_consts"],
    }
    if difficulty_score(cfg) <= spec["cap"]:
      return cfg
    if best is None or difficulty_score(cfg) < difficulty_score(best):
      best = cfg
  return best


def generate(args: argparse.Namespace) -> int:
  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  cfg = pick_config(args.difficulty, args.seed)
  cmd = [
    RYPUZMK,
    "--target",
    "python",
    "--rysmith",
    RYSMITH,
    "--keep-ground-truth",
    "-o",
    "puzzle.py",
    "-B",
    str(cfg["n_bbls"]),
    "-S",
    str(cfg["n_stmts"]),
    "-L",
    str(cfg["min_loop_iter"]),
    "-P",
    str(cfg["p_mask"]),
  ]
  if cfg["lift_consts"]:
    cmd.append("-C")
  if args.seed is not None:
    cmd += ["--seed", str(args.seed)]
  print(
    f"difficulty={args.difficulty} "
    f"bbls={cfg['n_bbls']} stmts={cfg['n_stmts']} "
    f"loops={cfg['min_loop_iter']} p_mask={cfg['p_mask']} "
    f"score={difficulty_score(cfg)}"
  )
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
    "-d",
    "--difficulty",
    choices=sorted(DIFFICULTIES),
    default="medium",
    help="difficulty level; caps the puzzle's difficulty score (default: medium)",
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
