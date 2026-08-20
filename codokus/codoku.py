#!/usr/bin/env python3
"""codoku - CLI for the rysmith-based Python codoku puzzle tooling.

Subcommands:
  create - generate one puzzle (codoku_creator)
  check  - validate a solution (codoku_checker)

Usage:
  codoku create [--profile easy|medium|hard] [--seed N] [-o DIR]
  codoku check <puzzle> [<solution>]
"""

from __future__ import annotations

import argparse
import sys

from codoku_checker import CheckFailure
from codoku_checker import check as run_check
from codoku_creator import DEFAULT_MAX_ATTEMPTS, PROFILES, generate


def positive_int(value: str) -> int:
  result = int(value)
  if result < 1:
    raise argparse.ArgumentTypeError(f"value must be at least 1, got {value!r}")
  return result


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
    help="generation profile; -d/--difficulty are compatibility aliases (default: medium)",
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

  check_parser = subparsers.add_parser(
    "check",
    help="check a solution against a codoku puzzle",
    description="Check a solution against a generated puzzle.",
  )
  check_parser.add_argument("puzzle", nargs="?", default="puzzle.py")
  check_parser.add_argument("solution", nargs="?", default="solution.py")

  return parser


def main() -> int:
  parser = build_parser()
  argv = sys.argv[1:]
  if argv and argv[0] == "check":
    args = parser.parse_args(argv)
    try:
      run_check(args.puzzle, args.solution)
    except CheckFailure as exc:
      print(f"[{exc.result.value}] {exc.message}", file=sys.stderr)
      return 1
    print("[PASS] Solution is valid!")
    return 0
  if argv and argv[0] in ("-h", "--help"):
    parser.parse_args(["-h"])
    return 0
  if argv and argv[0] == "create":
    argv = argv[1:]
  args = parser.parse_args(["create"] + argv)
  try:
    return generate(args)
  except (FileNotFoundError, RuntimeError, ValueError) as error:
    print(f"codoku: error: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
  sys.exit(main())
