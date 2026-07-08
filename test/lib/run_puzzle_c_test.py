"""Solution-validation tests for the puzzle checker (rypuzchk).

Each test is a single ``<name>.txt`` file under ``test/puzzle/`` laid out as a
verbatim puzzle, a ``=>`` separator line, and one candidate solution:

    // EXPECT: PASS              (or FAIL)
    // DESC: <one-line note>
    <verbatim puzzle>           rypuzmk output, incl. the //@ PATH / //@
    ...                         FILL_CONST banner the checker consumes
    =>
    <verbatim solution>         a full C program
    ...

``// EXPECT: PASS`` marks a solution rypuzchk must accept; ``// EXPECT: FAIL``
one it must reject (for any reason -- wrong value, wrong path, structural
tampering, an off-budget constant, ...). Acceptance is judged purely by
rypuzchk's exit code (0 == accepted).

The leading ``// EXPECT:`` / ``// DESC:`` lines are test metadata and are
stripped before the puzzle is handed to rypuzchk. The ``=>`` delimiter is
the separator line.

Run as:

    python3 -m test.lib.run_puzzle_test <test_dir> <rypuzchk>
"""

import os
import sys
import tempfile
import time

from test.lib.framework import TestResult, run_command
from test.lib.style import bold, green, red, yellow


def parse_case(file_path):
  """Parse a ``.txt`` test into (expectation, puzzle_text, solution_text)."""
  with open(file_path, "r") as f:
    lines = f.read().splitlines()

  expectation = None
  i = 0
  while i < len(lines):
    s = lines[i].strip()
    if s.startswith("// EXPECT:"):
      tag = s.split("EXPECT:", 1)[1].strip().upper()
      expectation = "PASS" if tag.startswith("PASS") else "FAIL"
      i += 1
    elif s.startswith("// DESC:"):
      i += 1
    else:
      break

  delim = None
  for j in range(i, len(lines)):
    if lines[j].strip() == "=>":
      delim = j
      break

  if expectation is None or delim is None:
    return None, None, None

  puzzle = "\n".join(lines[i:delim]).strip("\n") + "\n"
  solution = "\n".join(lines[delim + 1 :]).strip("\n") + "\n"
  return expectation, puzzle, solution


def make_test_func(rypuzchk):
  def test_func(file_path):
    expectation, puzzle, solution = parse_case(file_path)
    if expectation is None:
      return TestResult.FAIL, "malformed test: need a '// EXPECT:' tag and a '=>' line"

    with tempfile.TemporaryDirectory(prefix="rypuzchk_") as d:
      puzzle_path = os.path.join(d, "puzzle.c")
      solution_path = os.path.join(d, "solution.c")
      with open(puzzle_path, "w") as f:
        f.write(puzzle)
      with open(solution_path, "w") as f:
        f.write(solution)
      cmd = [rypuzchk, puzzle_path, solution_path]
      result, err = run_command(cmd, timeout=30)

    if err == "TIMEOUT":
      return TestResult.TIMEOUT, "Timeout"

    accepted = result.returncode == 0
    want_accepted = expectation == "PASS"
    if accepted == want_accepted:
      return TestResult.PASS, ""

    verb = "accepted" if accepted else "rejected"
    return (
      TestResult.FAIL,
      f"rypuzchk {verb} a solution marked EXPECT: {expectation} "
      f"(exit {result.returncode})\n"
      f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
    )

  return test_func


def run_suite(test_name, test_dir, test_func):
  test_files = []
  for root, _dirs, files in os.walk(test_dir):
    for file in files:
      if file.endswith(".txt"):
        test_files.append(os.path.join(root, file))
  test_files.sort()

  passed = failed = timeout = 0
  failures = []

  for file_path in test_files:
    print(f"Testing {file_path}...", end=" ", flush=True)
    start = time.time()
    status, message = test_func(file_path)
    ms = int((time.time() - start) * 1000)

    if status == TestResult.PASS:
      passed += 1
      print(f"{green('OK')} ({ms}ms)")
    elif status == TestResult.TIMEOUT:
      timeout += 1
      print(f"{yellow('TIMEOUT')} ({ms}ms)")
      failures.append((file_path, message))
    else:
      failed += 1
      print(f"{red('FAIL')} ({ms}ms)")
      failures.append((file_path, message))

  total = len(test_files)
  print(f"\nSummary ({test_name}): {passed}/{total} passed", end="")
  if timeout > 0:
    print(f", {yellow(str(timeout) + ' timeouts')}", end="")
  if failed > 0:
    print(f", {red(str(failed) + ' failed')}", end="")
  print(".\n")

  if failures:
    print(bold("Failures Details:"))
    for path, msg in failures:
      print(f"--- {red(path)} ---")
      print(msg)
    sys.exit(1)
  sys.exit(0)


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print(
      "Usage: python3 -m test.lib.run_puzzle_test <test_dir> <rypuzchk> [ignored_symiri]"
    )
    sys.exit(1)

  test_dir, rypuzchk = sys.argv[1:3]
  run_suite("puzzle_tests", test_dir, make_test_func(rypuzchk))
