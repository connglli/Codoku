#!/usr/bin/env python3
"""codoku — code sudoku puzzle generation & checking.

The current support is for the Python target only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

BIN_DIR = os.path.dirname(os.path.abspath(__file__))
RYSMITH = os.path.join(BIN_DIR, "rysmith")
RYPUZMK = os.path.join(BIN_DIR, "rypuzmk-tgt")
RYPUZCHK = os.path.join(BIN_DIR, "rypuzchk-tgt")

DEFAULT_OUTPUT = "puzzle.py"


def run(cmd: list[str]) -> int:
  return subprocess.call(cmd)


def postprocess_puzzle(path: str) -> None:
  """Rewrite the generated puzzle banner's validation command."""
  with open(path) as f:
    text = f.read()
  puzzle_name = os.path.basename(path)
  text = text.replace("./tools/rypuzchk", "codoku check")
  text = text.replace("[this_puzzle_file].py", puzzle_name)
  text = text.replace("[your_solution].py", "solution.py")
  with open(path, "w") as f:
    f.write(text)


def check(puzzle: str, solution: str) -> int:
  return run([RYPUZCHK, puzzle, solution])


def generate(args: argparse.Namespace) -> int:
  cmd = [
    RYPUZMK,
    "--target",
    "python",
    "--rysmith",
    RYSMITH,
    "--keep-ground-truth",
    "-o",
    DEFAULT_OUTPUT,
  ]
  if args.seed is not None:
    cmd += ["--seed", str(args.seed)]
  rc = run(cmd)
  if rc == 0:
    postprocess_puzzle(DEFAULT_OUTPUT)
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
    description="Generate one new codoku puzzle (output: puzzle.py).",
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
