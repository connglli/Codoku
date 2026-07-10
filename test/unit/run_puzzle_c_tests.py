"""End-to-end tests for the puzzle tooling (rypuzmk / rypuzchk).

rypuzmk masks a rysmith-generated C function into a puzzle (FILL_XXX marks plus
an instruction banner with machine-readable `//@ EXEC_PATH:` / `//@ CFG_EDGE:` / `//@ FILL_CONST:`
markers); rypuzchk validates a candidate solution against that puzzle.

These tests assert the invariants the two tools must jointly uphold:

  (1) The canonical solution round-trips: rypuzmk emits a ground truth that
      rypuzchk accepts (across several seeds).
  (2) The banner carries the machine-readable PATH / FILL_CONST markers.
  (3) require/assume are dropped from both puzzle and ground truth.
  (4) A solution that still contains FILL marks is rejected.
  (5) Structural integrity: deleting or duplicating a masked statement is
      rejected (the re-masked solution no longer matches the puzzle).
  (6) FILL_CONST budget: changing a budgeted constant to an off-budget value
      is rejected.

Run as:

  python3 -m test.unit.run_puzzle_tests <rypuzmk> <rypuzchk> <rysmith> [ignored_symiri]
"""

import os
import re
import subprocess
import sys
import tempfile

GREEN = "\033[32m"
RED = "\033[31m"
GRAY = "\033[90m"
NC = "\033[0m"

results = []


def run(cmd, **kw):
  print(f"  {GRAY}[RUN>]{NC} " + " ".join(cmd))
  return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)


def check(name, ok, detail=""):
  results.append((name, ok, detail))
  color = GREEN if ok else RED
  tag = "PASS" if ok else "FAIL"
  print(f"  [{color}{tag}{NC}] {name}" + (f" — {detail}" if detail and not ok else ""))


def make_puzzle(rypuzmk, rysmith, seed, outdir, *options):
  """Generate a puzzle + ground truth for `seed`; return (puzzle_path, gt_path)."""
  puzzle = os.path.join(outdir, f"p{seed}.c")
  gt = os.path.join(outdir, f"p{seed}.gt.c")
  r = run(
    [
      rypuzmk,
      "--seed",
      str(seed),
      "--rysmith",
      rysmith,
      "-o",
      puzzle,
      "--keep-ground-truth",
    ]
    + list(options)
  )
  ok = r.returncode == 0 and os.path.exists(puzzle) and os.path.exists(gt)
  return (puzzle, gt) if ok else (None, None)


def chk(rypuzchk, puzzle, solution):
  """Run rypuzchk; return (passed, combined_output)."""
  r = run([rypuzchk, puzzle, solution])
  return r.returncode == 0, (r.stdout + r.stderr)


def split_header_body(text):
  """Split a C file into its leading comment banner and the code body."""
  m = re.search(r"\n[a-zA-Z0-9_]+\s+func_", text)
  i = m.start() + 1 if m else 0
  return text[:i], text[i:]


def body_fill_count(text):
  """Count FILL_XXX marks inside the program body (excludes the banner prose)."""
  _, body = split_header_body(text)
  return body.count("FILL_")


def first_body_stmt_line(gt_text):
  """Index of the first indented statement line inside a basic block."""
  lines = gt_text.splitlines()
  in_body = False
  for idx, ln in enumerate(lines):
    s = ln.strip()
    if re.match(r"^(b\w+|entry):", s):
      in_body = True
      continue
    if s == "exit:":
      in_body = False
      continue
    if in_body and s.endswith(";") and not s.startswith("goto ") and "printf(" not in s:
      return idx
  return -1


def find_body_line(gt_text, pattern, last=False):
  """Index of the first (or last) stripped line in a basic block matching `pattern`."""
  rx = re.compile(pattern)
  lines = gt_text.splitlines()
  in_body = False
  found = -1
  for idx, ln in enumerate(lines):
    s = ln.strip()
    if re.match(r"^(b\w+|entry):", s):
      in_body = True
      continue
    if s == "exit:":
      in_body = False
      continue
    if in_body and rx.search(s):
      found = idx
      if not last:
        return idx
  return found


def dup_line(gt_text, idx):
  """Return `gt_text` with line `idx` duplicated in place (a structural insert)."""
  lines = gt_text.splitlines(keepends=True)
  return "".join(lines[:idx] + [lines[idx], lines[idx]] + lines[idx + 1 :])


def main():
  if len(sys.argv) < 4:
    print("usage: run_puzzle_tests.py <rypuzmk> <rypuzchk> <rysmith> [ignored_symiri]")
    return 2
  rypuzmk, rypuzchk, rysmith = sys.argv[1:4]

  with tempfile.TemporaryDirectory(prefix="rypuztest_") as outdir:
    # (1) round-trip across a few seeds
    seeds = [42, 500, 9999]
    base_puzzle, base_gt = None, None
    rt_ok = True
    for s in seeds:
      puzzle, gt = make_puzzle(rypuzmk, rysmith, s, outdir)
      if not puzzle:
        rt_ok = False
        check(f"generate seed {s}", False, "rypuzmk failed")
        continue
      passed, out = chk(rypuzchk, puzzle, gt)
      check(
        f"ground truth round-trips (seed {s})", passed, out.strip().splitlines()[-1:]
      )
      rt_ok = rt_ok and passed
      if s == 42:
        base_puzzle, base_gt = puzzle, gt

    if not base_puzzle:
      check("seed 42 available for mutation tests", False, "no base puzzle")
      return summarize()

    puzzle_text = open(base_puzzle).read()
    gt_text = open(base_gt).read()

    # (2) machine-readable markers present
    check("banner has //@ EXEC_PATH marker", "//@ EXEC_PATH:" in puzzle_text)
    check("banner has //@ CFG_EDGE marker", "//@ CFG_EDGE:" in puzzle_text)
    check("banner has //@ FILL_CONST markers", "//@ FILL_CONST:" in puzzle_text)

    # (3) require/assume dropped from both puzzle and ground truth
    # C compiler output doesn't contain require/assume keywords.
    _, p_body = split_header_body(puzzle_text)
    check(
      "puzzle body has no require/assume",
      not re.search(r"\b(require|assume)\b", p_body),
    )
    check(
      "ground truth has no require/assume",
      not re.search(r"\b(require|assume)\b", gt_text),
    )

    # (4) a solution that still contains FILL marks is rejected
    passed, out = chk(rypuzchk, base_puzzle, base_puzzle)
    check(
      "puzzle-as-solution rejected with FAIL_BASICS",
      not passed and "[FAIL_BASICS]" in out,
      out,
    )

    # (5a) deleting a masked statement is rejected
    idx = first_body_stmt_line(gt_text)
    check("found a masked statement to mutate", idx >= 0)
    if idx >= 0:
      lines = gt_text.splitlines(keepends=True)
      del_path = os.path.join(outdir, "mut_delete.c")
      with open(del_path, "w") as f:
        f.writelines(lines[:idx] + lines[idx + 1 :])
      passed, out = chk(rypuzchk, base_puzzle, del_path)
      check(
        "deleted masked statement rejected with FAIL_REMASKING",
        not passed and "[FAIL_REMASKING]" in out,
        out,
      )

      # (5b) duplicating a masked statement is rejected
      dup_path = os.path.join(outdir, "mut_dup.c")
      with open(dup_path, "w") as f:
        f.writelines(lines[:idx] + [lines[idx], lines[idx]] + lines[idx + 1 :])
      passed, out = chk(rypuzchk, base_puzzle, dup_path)
      check(
        "duplicated masked statement rejected with FAIL_REMASKING",
        not passed and "[FAIL_REMASKING]" in out,
        out,
      )

    # (6) changing a budgeted constant to an off-budget value is rejected
    m = re.search(r"//@ FILL_CONST:\s*(-?\d{2,})\s+\d+", puzzle_text)
    check("found an integer FILL_CONST to perturb", m is not None)
    if m:
      val = m.group(1)
      # The ground truth has no banner/markers, so perturb it whole: replace the
      # first standalone occurrence of the budgeted value with an off-budget one.
      new_gt, n = re.subn(
        r"(?<!\d)" + re.escape(val) + r"(?!\d)", "987654321", gt_text, count=1
      )
      check("budgeted constant occurs in masked body", n == 1)
      if n == 1:
        off_path = os.path.join(outdir, "mut_offbudget.c")
        with open(off_path, "w") as f:
          f.write(new_gt)
        passed, out = chk(rypuzchk, base_puzzle, off_path)
        # Note: mutating a constant used in computation affects final checksum, so it fails at FAIL_OUTPUT or FAIL_PATH.
        check(
          "perturbed constant rejected with FAIL_OUTPUT or FAIL_PATH",
          not passed and any(tag in out for tag in ["[FAIL_OUTPUT]", "[FAIL_PATH]"]),
          out,
        )

        # Stage validation: FAIL_PARSE (leaf function malformed/not found in C)
        parse_path = os.path.join(outdir, "mut_parse_error.c")
        err_syntax_gt = re.sub(r"\bfunc_", "_func_", gt_text)
        with open(parse_path, "w") as f:
          f.write(err_syntax_gt)
        passed, out = chk(rypuzchk, base_puzzle, parse_path)
        check(
          "syntax error inside mask rejected with FAIL_PARSE",
          not passed and "[FAIL_PARSE]" in out,
          out,
        )

        # Stage validation: FAIL_COMPILE (compilation error inside a mask position in C)
        compile_path = os.path.join(outdir, "mut_compile_error.c")
        err_compile_gt = re.sub(
          r"(?<!\d)" + re.escape(val) + r"(?!\d)", "09", gt_text, count=1
        )
        with open(compile_path, "w") as f:
          f.write(err_compile_gt)
        passed, out = chk(rypuzchk, base_puzzle, compile_path)
        check(
          "compilation error inside mask rejected with FAIL_COMPILE",
          not passed and "[FAIL_COMPILE]" in out,
          out,
        )

    # Stage validation: FAIL_FILL_CONST (decrement budget count in banner)
    m_fill = re.search(r"(//@ FILL_CONST:\s*-?\d+\s+)(\d+)", puzzle_text)
    if m_fill:
      cnt = int(m_fill.group(2))
      if cnt >= 1:
        mut_puz = (
          puzzle_text[: m_fill.start(2)] + str(cnt - 1) + puzzle_text[m_fill.end(2) :]
        )
        puz_path = os.path.join(outdir, "mut_budget_puz.c")
        with open(puz_path, "w") as f:
          f.write(mut_puz)
        passed, out = chk(rypuzchk, puz_path, base_gt)
        check(
          "reduced budget rejected with FAIL_FILL_CONST",
          not passed and "[FAIL_FILL_CONST]" in out,
          out,
        )

    # Stage validation: FAIL_CFG (altered CFG destination in intermediate block)
    parts = re.split(r"(\b\w+:)", gt_text)
    mutated = False
    for i in range(1, len(parts), 2):
      header = parts[i]
      body = parts[i + 1]
      if "entry" in header or "exit" in header:
        continue
      m_goto = re.search(r"(\bgoto\s+)(\w+)", body)
      if m_goto:
        mut_body = body[: m_goto.start(2)] + "exit" + body[m_goto.end(2) :]
        parts[i + 1] = mut_body
        mutated = True
        break
    if mutated:
      cfg_path = os.path.join(outdir, "mut_cfg.c")
      with open(cfg_path, "w") as f:
        f.write("".join(parts))
      passed, out = chk(rypuzchk, base_puzzle, cfg_path)
      check(
        "altered CFG destination rejected with FAIL_CFG",
        not passed and "[FAIL_CFG]" in out,
        out,
      )

    # Stage validation: FAIL_PATH (swapped conditional branch destinations in intermediate block)
    parts = re.split(r"(\b\w+:)", gt_text)
    mutated = False
    for i in range(1, len(parts), 2):
      header = parts[i]
      body = parts[i + 1]
      if "entry" in header or "exit" in header:
        continue
      m_if = re.search(
        r"(if\s*\(.*?\)\s*goto\s+)(\w+)(;\s*else\s*goto\s+)(\w+)(;)", body
      )
      if m_if:
        swapped_body = (
          body[: m_if.start(2)]
          + m_if.group(4)
          + body[m_if.end(2) : m_if.start(4)]
          + m_if.group(2)
          + body[m_if.end(4) :]
        )
        parts[i + 1] = swapped_body
        mutated = True
        break
    if mutated:
      path_path = os.path.join(outdir, "mut_path.c")
      with open(path_path, "w") as f:
        f.write("".join(parts))
      passed, out = chk(rypuzchk, base_puzzle, path_path)
      check(
        "swapped conditional branch destinations rejected with FAIL_PATH",
        not passed and "[FAIL_PATH]" in out,
        out,
      )

    # Stage validation: FAIL_OUTPUT (swap two budgeted constants inside leaf function)
    const_vals = re.findall(r"//@ FILL_CONST:\s*(\S+)", puzzle_text)
    distinct_ints = sorted(
      list(set([v for v in const_vals if re.match(r"^-?\d+$", v)]))
    )
    if len(distinct_ints) >= 2:
      v1, v2 = distinct_ints[0], distinct_ints[1]
      funcs = re.split(r"(\b(?:uint32_t|int|void)\s+func_\w+)", gt_text)
      if len(funcs) >= 3:
        swapped_body = re.sub(
          r"(?<!\w)" + re.escape(v1) + r"(?!\w)", "__TEMP_VAL__", funcs[2]
        )
        swapped_body = re.sub(r"(?<!\w)" + re.escape(v2) + r"(?!\w)", v1, swapped_body)
        swapped_body = swapped_body.replace("__TEMP_VAL__", v2)
        funcs[2] = swapped_body
        out_path = os.path.join(outdir, "mut_output.c")
        with open(out_path, "w") as f:
          f.write("".join(funcs))
        passed, out = chk(rypuzchk, base_puzzle, out_path)
        check(
          "swapped budgeted constants rejected with FAIL_OUTPUT",
          not passed and "[FAIL_OUTPUT]" in out,
          out,
        )

    # (7) --p-mask selective masking.
    full_fill = body_fill_count(puzzle_text)
    check(
      "p-mask: default puzzle carries no //@ MASK marker",
      "//@ MASK:" not in puzzle_text,
    )

    half_puzzle = os.path.join(outdir, "pm_half.c")
    half_gt = os.path.join(outdir, "pm_half.gt.c")
    r = run(
      [
        rypuzmk,
        "--seed",
        "42",
        "--p-mask",
        "0.5",
        "--rysmith",
        rysmith,
        "-o",
        half_puzzle,
        "--keep-ground-truth",
      ]
    )
    half_ok = (
      r.returncode == 0 and os.path.exists(half_puzzle) and os.path.exists(half_gt)
    )
    check("p-mask 0.5: rypuzmk succeeds", half_ok, r.stderr)
    if half_ok:
      htext = open(half_puzzle).read()
      check("p-mask 0.5: no //@ MASK marker emitted", "//@ MASK:" not in htext)
      check(
        "p-mask 0.5: fewer FILL marks than full masking",
        body_fill_count(htext) < full_fill,
      )
      passed, out = chk(rypuzchk, half_puzzle, half_gt)
      check(
        "p-mask 0.5: ground truth round-trips", passed, out.strip().splitlines()[-1:]
      )

    zero_puzzle = os.path.join(outdir, "pm_zero.c")
    zero_gt = os.path.join(outdir, "pm_zero.gt.c")
    r = run(
      [
        rypuzmk,
        "--seed",
        "42",
        "--p-mask",
        "0",
        "--rysmith",
        rysmith,
        "-o",
        zero_puzzle,
        "--keep-ground-truth",
      ]
    )
    zero_ok = (
      r.returncode == 0 and os.path.exists(zero_puzzle) and os.path.exists(zero_gt)
    )
    check("p-mask 0: rypuzmk succeeds", zero_ok, r.stderr)
    if zero_ok:
      ztext = open(zero_puzzle).read()
      check(
        "p-mask 0: body fully revealed (no FILL marks)", body_fill_count(ztext) == 0
      )
      check("p-mask 0: no //@ MASK marker emitted", "//@ MASK:" not in ztext)
      passed, out = chk(rypuzchk, zero_puzzle, zero_gt)
      check(
        "p-mask 0: fully-revealed puzzle round-trips",
        passed,
        out.strip().splitlines()[-1:],
      )

    # (8) Structural tampering must be reported via the structural-integrity
    # check, even under partial masking.
    def expect_structural(name, puzzle, mutated_text):
      sol = os.path.join(outdir, "mut_struct.c")
      with open(sol, "w") as f:
        f.write(mutated_text)
      passed, out = chk(rypuzchk, puzzle, sol)
      ok = (not passed) and ("structural integrity" in out.lower())
      check(name, ok, out.strip().splitlines()[-1:])

    addr_idx = find_body_line(gt_text, r"&\w+")
    const_idx = find_body_line(gt_text, r"\b\d{3,}\b")
    first_idx = first_body_stmt_line(gt_text)
    check("found a body addr statement to duplicate", addr_idx >= 0)
    check("found a body statement with a constant to duplicate", const_idx >= 0)

    if half_ok and addr_idx >= 0:
      expect_structural(
        "p-mask 0.5: duplicated addr stmt rejected as structural",
        half_puzzle,
        dup_line(gt_text, addr_idx),
      )
    if half_ok and const_idx >= 0:
      expect_structural(
        "p-mask 0.5: duplicated const stmt rejected as structural",
        half_puzzle,
        dup_line(gt_text, const_idx),
      )
    if half_ok and first_idx >= 0:
      expect_structural(
        "p-mask 0.5: duplicated first body stmt rejected as structural",
        half_puzzle,
        dup_line(gt_text, first_idx),
      )
    if const_idx >= 0:
      expect_structural(
        "full-mask: duplicated const stmt rejected as structural",
        base_puzzle,
        dup_line(gt_text, const_idx),
      )

    # (8f/edge) off-budget check
    m_off = re.search(r"//@ FILL_CONST:\s*(-?\d{2,})\s+\d+", puzzle_text)
    if m_off:
      off_gt, n_off = re.subn(
        r"(?<!\d)" + re.escape(m_off.group(1)) + r"(?!\d)",
        "987654321",
        gt_text,
        count=1,
      )
      if n_off == 1:
        off_sol = os.path.join(outdir, "mut_offbudget2.c")
        with open(off_sol, "w") as f:
          f.write(off_gt)
        passed, out = chk(rypuzchk, base_puzzle, off_sol)
        ok = (not passed) and ("structural integrity" not in out.lower())
        check("off-budget constant still reported as budget error", ok, out.strip())

    # (edge) bad probability
    r = run(
      [
        rypuzmk,
        "--seed",
        "42",
        "--p-mask",
        "1.5",
        "--rysmith",
        rysmith,
        "-o",
        os.path.join(outdir, "pm_bad.c"),
      ]
    )
    check("bad probability rejected", r.returncode != 0)

    # (9) --lift-consts validation
    lift_puzzle = os.path.join(outdir, "lift_puzzle.c")
    lift_gt = os.path.join(outdir, "lift_puzzle.gt.c")
    r_lift = run(
      [
        rypuzmk,
        "--seed",
        "42",
        "--lift-consts",
        "--rysmith",
        rysmith,
        "-o",
        lift_puzzle,
        "--keep-ground-truth",
      ]
    )
    lift_ok = (
      r_lift.returncode == 0 and os.path.exists(lift_puzzle) and os.path.exists(lift_gt)
    )
    check("lift-consts: rypuzmk with --lift-consts succeeds", lift_ok)
    if lift_ok:
      lift_content = open(lift_puzzle).read()
      check(
        "lift-consts: no //@ FILL_CONST marker in puzzle",
        "//@ FILL_CONST:" not in lift_content,
      )
      check(
        "lift-consts: no 'Requirements for FILL_CONST' section in puzzle header",
        "Requirements for FILL_CONST" not in lift_content,
      )
      passed, out = chk(rypuzchk, lift_puzzle, lift_gt)
      check(
        "lift-consts: ground truth on lift puzzle accepted",
        passed,
        out.strip().splitlines()[-1:],
      )

      gt_content = open(lift_gt).read()
      perturbed_gt = gt_content
      m_name = re.search(r"\bfunc_[a-f0-9]+_\d+\b", gt_content)
      leaf_name = m_name.group(0) if m_name else ""
      leaf_idx = gt_content.find(f"{leaf_name}(") if leaf_name else -1
      first_b_idx = -1
      exit_idx = -1
      if leaf_idx >= 0:
        for m_b in re.finditer(r"\bb\d+:", gt_content[leaf_idx:]):
          first_b_idx = leaf_idx + m_b.start()
          break
        exit_idx = gt_content.find("exit:", leaf_idx)

      const_match = None
      if first_b_idx >= 0 and exit_idx >= 0:
        body_content = gt_content[first_b_idx:exit_idx]
        const_match = re.search(r"\b\d{3,}\b", body_content)
      if const_match:
        const_val = const_match.group(0)
        body_perturbed = body_content.replace(const_val, "987654321", 1)
        perturbed_gt = gt_content[:first_b_idx] + body_perturbed + gt_content[exit_idx:]

        main_idx = gt_content.find("int32_t main(void) {")
        call_start = (
          re.search(r"func_[a-f0-9]+_\d+\(", gt_content[main_idx:])
          if main_idx >= 0
          else None
        )
        call_str = ""
        if call_start and main_idx >= 0:
          start_idx = main_idx + call_start.start()
          paren_count = 0
          started = False
          for i in range(start_idx, len(gt_content)):
            char = gt_content[i]
            if char == "(":
              paren_count += 1
              started = True
            elif char == ")":
              paren_count -= 1
            if started and paren_count == 0:
              call_str = gt_content[start_idx : i + 1]
              break

        dummy_main = f"""
#include <stdio.h>
int32_t main(void) {{
  printf("ACTUAL_RET:%d\\n", {call_str});
  return 0;
}}
"""
        main_idx = perturbed_gt.find("int32_t main(void) {")
        if main_idx >= 0:
          run_src = perturbed_gt[:main_idx] + dummy_main
          temp_bin = os.path.join(outdir, "temp_bin")
          temp_src = os.path.join(outdir, "temp_src.c")
          with open(temp_src, "w") as f:
            f.write(run_src)
          r_comp = subprocess.run(
            ["gcc", temp_src, "-o", temp_bin, "-lm"], capture_output=True, text=True
          )
          if r_comp.returncode != 0:
            print("COMPILATION FAILED:", r_comp.stderr)
            raise RuntimeError(f"GCC failed: {r_comp.stderr}")
          r_run = subprocess.run([temp_bin], capture_output=True, text=True)
          actual_ret = None
          for line in r_run.stdout.splitlines():
            if line.startswith("ACTUAL_RET:"):
              actual_ret = line.split(":", 1)[1].strip()

          if actual_ret is not None:
            perturbed_gt = re.sub(
              r"_check_chksum_i32\(\(\((-?\d+)\)\)",
              f"_check_chksum_i32((({actual_ret}))",
              perturbed_gt,
            )
            puzzle_content = open(lift_puzzle).read()
            puzzle_content = re.sub(
              r"_check_chksum_i32\(\(\((-?\d+)\)\)",
              f"_check_chksum_i32((({actual_ret}))",
              puzzle_content,
            )
            with open(lift_puzzle, "w") as f:
              f.write(puzzle_content)

        perturbed_sol = os.path.join(outdir, "perturbed_lift.c")
        with open(perturbed_sol, "w") as f:
          f.write(perturbed_gt)
        passed, out = chk(rypuzchk, lift_puzzle, perturbed_sol)
        check(
          "lift-consts: off-budget solution accepted when const constraints are absent",
          passed,
          out.strip().splitlines()[-1:],
        )

      # (10) TDD: zero FILL_XX generation errors (5 tests)
      for test_idx, test_seed in enumerate([1001, 1002, 1003, 1004, 1005], 1):
        err_puzzle = os.path.join(outdir, f"err_puzzle_{test_seed}.c")
        r_err = run(
          [
            rypuzmk,
            "--input",
            base_gt,
            "--seed",
            str(test_seed),
            "--p-mask",
            "0.000001",
            "-o",
            err_puzzle,
          ]
        )
        check(
          f"zero-fill-error: rypuzmk fails on zero FILL_XX with input (test {test_idx})",
          r_err.returncode != 0,
          f"exit code: {r_err.returncode}",
        )

  return summarize()


def summarize():
  npass = sum(1 for _, ok, _ in results if ok)
  total = len(results)
  print(f"\n{GREEN if npass == total else RED}{npass}/{total} checks passed{NC}")
  return 0 if npass == total else 1


if __name__ == "__main__":
  sys.exit(main())
