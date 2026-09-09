#!/usr/bin/env python3
"""codoku - CLI for the rysmith-based Python codoku puzzle tooling.

Subcommands:
  create  - generate one puzzle (codoku_creator)
  check   - validate a solution (codoku_checker)
  analyze - report a puzzle's realized metrics and complexity estimate

Usage:
  codoku create [--profile small|medium|large] [--seed N] [-o DIR]
  codoku check <puzzle> [<solution>]
  codoku analyze [--json] [--ground-truth GT] [puzzle]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from codoku_checker import main as check_main
from codoku_complexity import main as analyze_main
from codoku_creator import DEFAULT_MAX_ATTEMPTS, PROFILES, generate


def positive_int(value: str) -> int:
  result = int(value)
  if result < 1:
    raise argparse.ArgumentTypeError(f"value must be at least 1, got {value!r}")
  return result


def find_rysmith() -> Path | None:
  # 1. Next to the invoked binary/symlink (e.g. ./codoku -> ./rysmith)
  if sys.argv and sys.argv[0]:
    argv0 = Path(sys.argv[0])
    candidate = argv0.parent.resolve() / "rysmith"
    if candidate.is_file() and os.access(candidate, os.X_OK):
      return candidate
    candidate = argv0.resolve().parent / "rysmith"
    if candidate.is_file() and os.access(candidate, os.X_OK):
      return candidate
  # 2. Next to codoku.py itself
  script_dir = Path(__file__).resolve().parent
  candidate = script_dir / "rysmith"
  if candidate.is_file() and os.access(candidate, os.X_OK):
    return candidate
  # 3. Current working directory
  cwd_candidate = Path.cwd() / "rysmith"
  if cwd_candidate.is_file() and os.access(cwd_candidate, os.X_OK):
    return cwd_candidate.resolve()
  # 4. PATH lookup
  which = shutil.which("rysmith")
  if which:
    return Path(which).resolve()
  return None


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="codoku",
    description="Generate and check Python codoku puzzles (rysiff-based, Python-specific).",
  )
  subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

  create = subparsers.add_parser(
    "create",
    help="generate one codoku puzzle",
    description="Generate one Python codoku puzzle using the selected profile.",
  )
  create.add_argument(
    "-o", "--outdir", default=".", help="output directory for the puzzle (default: .)"
  )
  create.add_argument(
    "-p",
    "--profile",
    "-d",
    "--difficulty",
    dest="profile",
    choices=sorted(PROFILES),
    default="medium",
    help="generation profile, which implies difficulty (default: medium)",
  )
  create.add_argument(
    "--seed",
    type=int,
    default=None,
    help="master seed controlling configuration sampling and generator seeds",
  )
  create.add_argument(
    "--max-attempts",
    type=positive_int,
    default=DEFAULT_MAX_ATTEMPTS,
    help=f"maximum candidate-generation attempts (default: {DEFAULT_MAX_ATTEMPTS})",
  )

  subparsers.add_parser(
    "check",
    help="check a solution against a codoku puzzle",
    description="Check a solution against a generated puzzle.",
  )

  subparsers.add_parser(
    "analyze",
    help="analyze a puzzle against complexity estimate",
    description="Analyze a puzzle file and report its realized metrics "
    "(structure, path, loops, masks, budget, solution space) and the "
    "heuristic complexity estimate.",
  )

  return parser


def main() -> int:
  parser = build_parser()
  argv = sys.argv[1:]
  if argv and argv[0] == "check":
    return check_main(argv[1:])
  if argv and argv[0] == "analyze":
    return analyze_main(argv[1:])
  if argv and argv[0] in ("-h", "--help"):
    parser.parse_args(["-h"])
    return 0
  if argv and argv[0] == "create":
    argv = argv[1:]
  args = parser.parse_args(["create"] + argv)
  rysmith = find_rysmith()
  if rysmith is None:
    print("codoku: error: could not find rysmith binary", file=sys.stderr)
    return 2
  try:
    return generate(args, rysmith_path=rysmith)
  except (FileNotFoundError, RuntimeError, ValueError) as error:
    print(f"codoku: error: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
  sys.exit(main())
