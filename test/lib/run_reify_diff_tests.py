"""Differential test: rysmith-generated programs (leaves) and rylink-
linked whole programs, interpreted by symiri vs. compiled by symirc +
clang and run under UBSan. Catches bugs in any of reify / interp /
C-backend / solver / linker by cross-checking outputs.

Each batch runs four phases:
  (1) rysmith generation     — leaf functions generated randomly
  (2) rysmith test           — symiri vs C-binary cross-validation
  (3) rylink generation      — whole programs from the same pool
  (4) rylink test            — symiri vs C-binary on the bundled program

Usage:
  python -m test.lib.run_reify_diff_tests \\
      --rysmith ./rysmith --symiri ./symiri --symirc ./symirc \\
      --rylink ./rylink \\
      [--n 100] [--seed 1234] [--out build/test_tmp/reify_diff]
"""

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from test.lib.style import bold, green, red, yellow

# ASan pulls in LeakSanitizer, whose end-of-process scan needs ptrace and
# dies where that is unavailable — taking every otherwise-passing binary
# with it. Leaks are not what these suites test.
os.environ.setdefault("LSAN_OPTIONS", "detect_leaks=0")

# Match either an integer Result (i*) or a hex-float Result (`%a` form
# used by both symiri and the C-side printf when the return is f32/f64).
# Examples: `Result: -42`, `Result: 0x1.5p+10`, `Result: -0x0p+0`,
# `Result: nan`, `Result: inf`. Group 1 is the value verbatim — the
# comparison is byte-equal, which is what we want for bit-exact xval.
_RESULT_RE = re.compile(
  r"Result:\s*(-?(?:0x[0-9a-fA-F.]+p[+-]?[0-9]+|[0-9]+|nan|-?inf))"
)

# Common header / signature regexes shared by leaf and bundled .sir files.
_SOLVED_RE = re.compile(r"^//\s*SOLVED:\s*(.+?)\s*$")  # leaf rysmith header
_PARAMS_RE = re.compile(r"^//\s*PARAMS:(.*)$")  # rylink bundled header
_RETURN_RE = re.compile(r"^//\s*RETURN:\s*(.+?)\s*$")  # rylink bundled header
_ENTRY_RE = re.compile(r"^//\s*ENTRY:\s*@([a-zA-Z0-9_]+)")  # rylink bundled header
_FUN_SIG_RE = re.compile(
  r"fun\s+@([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*:\s*(i[0-9]+|f32|f64)"
)

# C surface types for RefractIR scalars. Same mapping covers param and ret slots.
_CTY = {
  "i8": "int8_t",
  "i16": "int16_t",
  "i32": "int32_t",
  "i64": "int64_t",
  "f32": "float",
  "f64": "double",
}


def _ctype_of(refractir_type):
  """C type for a scalar RefractIR type, or None for non-scalars. Custom iN
  widths map to the same widened storage type the C backend uses
  (N <= 8 -> int8_t, <= 16 -> int16_t, <= 32 -> int32_t, <= 64 -> int64_t)."""
  ct = _CTY.get(refractir_type)
  if ct is not None:
    return ct
  m = re.fullmatch(r"i(\d+)", refractir_type)
  if not m:
    return None
  bits = int(m.group(1))
  if bits < 1 or bits > 64:
    return None
  for cap, name in ((8, "int8_t"), (16, "int16_t"), (32, "int32_t"), (64, "int64_t")):
    if bits <= cap:
      return name
  return None


# Batch size for generation + cross-validation. Large -n runs proceed
# one batch at a time so the user sees periodic progress (and so a
# 10000-program run does not need to materialise 10000 .sir/.c files
# on disk before testing starts). Each batch uses seed = base + batch_idx
# so generated programs do not repeat across batches.
BATCH_SIZE = 100


# ───────────────────────────────────────────────────────────────────────
# Header parsing — one parser handles both leaf and bundled .sir files.
# ───────────────────────────────────────────────────────────────────────


def _parse_program(sir_path, entry_hint=None):
  """Parse a RefractIR file and return
    (entry_name, ret_type, param_types[], param_values[], expected_ret)
  or None if anything is missing.

  Accepts either format:
    - Leaf rysmith .sir: `// SOLVED: %pa0=…, %pa1=…, ret=…`
    - Rylink bundled .sir: `// ENTRY: @f` / `// PARAMS: …` / `// RETURN: …`

  When ``entry_hint`` is given it picks that fun; otherwise it takes the
  first `fun` declaration in the file. The leaf format always has exactly
  one fun; the bundled format relies on the ENTRY comment.
  """
  entry = entry_hint
  param_vals = None
  param_pairs = None  # leaf SOLVED preserves names so we can re-order
  expected = None
  param_types = None
  ret_type = None
  with open(sir_path, "r") as f:
    for line in f:
      if entry is None:
        m = _ENTRY_RE.match(line)
        if m:
          entry = m.group(1)
          continue
      m = _SOLVED_RE.match(line)
      if m:
        # Comma-separated `name=value` tokens; the final `ret=…` is split out.
        tokens = [t.strip() for t in m.group(1).split(",")]
        pairs = []
        for t in tokens:
          if "=" not in t:
            continue
          k, v = t.split("=", 1)
          if k.strip() == "ret":
            expected = v.strip()
          else:
            pairs.append((k.strip(), v.strip()))
        param_pairs = pairs
        continue
      m = _PARAMS_RE.match(line)
      if m:
        body = m.group(1).strip()
        if body == "(none)":
          param_vals = []
        else:
          # Whitespace-separated `name=value` tokens; rylink already
          # emits them in declaration order.
          param_vals = [tok.split("=", 1)[1] for tok in body.split() if "=" in tok]
        continue
      m = _RETURN_RE.match(line)
      if m:
        v = m.group(1).strip()
        expected = None if v == "(none)" else v
        continue
      if ret_type is None:
        m = _FUN_SIG_RE.search(line)
        if m and (entry is None or m.group(1) == entry):
          if entry is None:
            entry = m.group(1)
          ret_type = m.group(3)
          params_str = m.group(2).strip()
          if params_str:
            # `%pa0: i32, %pa1: i8, %pa2: f64` — keep names+types in order.
            ptypes = []
            pnames = []
            for p in params_str.split(","):
              if ":" not in p:
                continue
              n, t = p.split(":", 1)
              pnames.append(n.strip())
              ptypes.append(t.strip())
            param_types = ptypes
            # If the SOLVED header named the params, re-order its values
            # to match the declared param order. Otherwise we trust the
            # bundled-format positional order already in param_vals.
            if param_pairs is not None:
              by_name = {n: v for n, v in param_pairs}
              try:
                param_vals = [by_name[n] for n in pnames]
              except KeyError:
                return None
          else:
            param_types = []
            if param_pairs is not None:
              param_vals = []
  if entry is None or param_types is None or ret_type is None or param_vals is None:
    return None
  if len(param_types) != len(param_vals):
    return None
  return (entry, ret_type, param_types, param_vals, expected)


# ───────────────────────────────────────────────────────────────────────
# main.c synthesis — extern decl + call with the entry's solved args.
# Used for both leaf and bundled programs.
# ───────────────────────────────────────────────────────────────────────


def _write_main_c(path, fname, ret_type, param_types, param_values):
  """Returns True on success, False if the entry has a non-scalar slot
  we can't synthesize from the CLI."""
  cret = _ctype_of(ret_type)
  if cret is None:
    return False
  cparams = []
  for pt in param_types:
    cp = _ctype_of(pt)
    if cp is None:
      return False  # non-scalar param: skip.
    cparams.append(cp)
  lits = []
  for pt, pv in zip(param_types, param_values):
    if pt in ("f32", "f64"):
      # Solver may print a float as `-0` — promote to dotted form so the
      # C lexer parses it as a floating literal rather than an int. This
      # is defensive; the bit-exact formatDouble side always emits an
      # exponent now, but external callers and hand-written cases still
      # rely on us doing the right thing.
      if "." not in pv and "e" not in pv and "E" not in pv:
        pv = pv + ".0"
      lits.append(pv + ("f" if pt == "f32" else ""))
    else:
      lits.append(pv)
  decl_params = ", ".join(cparams) if cparams else "void"
  call_args = ", ".join(lits)
  # FP returns are compared bit-exactly against symiri's `%a` output —
  # decimal printing would silently lose precision at the last digit
  # and produce spurious mismatches on the diff test. The cast through
  # `double` is harmless for f32 (widening preserves bits) and required
  # for f64 to feed printf's promotion rule.
  is_fp_ret = ret_type in ("f32", "f64")
  if is_fp_ret:
    print_stmt = '  printf("Result: %a\\n", (double)r);\n'
  else:
    print_stmt = '  printf("Result: %lld\\n", (long long)r);\n'
  with open(path, "w") as f:
    f.write(
      "#include <stdint.h>\n"
      "#include <stdio.h>\n"
      f"extern {cret} refractir_{fname}({decl_params});\n"
      "int main(void) {\n"
      f"  {cret} r = refractir_{fname}({call_args});\n" + print_stmt + "  return 0;\n"
      "}\n"
    )
  return True


# ───────────────────────────────────────────────────────────────────────
# Cross-validation — symiri vs clang+exe on one (sir, c) pair.
# Handles leaf and bundled uniformly: given the parsed (entry, types,
# values, expected), run symiri and the C binary, classify the outcome.
# ───────────────────────────────────────────────────────────────────────


def _classify(label, sir_path, c_paths, parsed, symiri, clang, main_c, exe, verbose):
  """Returns (bucket, msg). ``label`` is the human-readable program name
  used in failure messages; the bucket vocabulary matches the rest of the
  harness ("passed" / "skipped" / "mismatch" / "ubsan" / "cfail" / "sirfail").

  ``c_paths`` is a list of one or more .c source paths to feed clang.
  Rylink's split-by-source output emits one .c per FunDecl::sourceStem
  plus a (mostly empty) program.c, so the caller hands us all of them
  and we link them together against the synthesised main.c."""
  if not c_paths or not all(os.path.exists(p) for p in c_paths):
    return ("cfail", f"{label}: no .c emitted")
  if parsed is None:
    return ("sirfail", f"{label}: cannot parse program header")
  entry, ret_type, ptypes, pvals, expected = parsed

  argv = [symiri, "--main", f"@{entry}", sir_path]
  if pvals:
    argv += ["--"] + pvals
  try:
    si = subprocess.run(
      argv,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      timeout=15,
    )
  except subprocess.TimeoutExpired:
    return ("sirfail", f"{label}: symiri timeout")
  if si.returncode != 0:
    return ("sirfail", f"{label}: symiri exit {si.returncode}")
  sm = _RESULT_RE.search(si.stdout) or _RESULT_RE.search(si.stderr)
  if sm is None:
    return ("sirfail", f"{label}: symiri no Result line")
  sir_val = sm.group(1)

  # If the .sir header recorded a solver-derived expected value, surface
  # a symiri-vs-expected disagreement before going on to compare to C.
  if expected is not None and expected != sir_val:
    return ("sirfail", f"{label}: symiri={sir_val} expected={expected}")

  if not _write_main_c(main_c, entry, ret_type, ptypes, pvals):
    return ("skipped", None)
  # [v0.2.2] Strict syntax pre-flight. The main clang call below uses
  # `-w` to silence warnings (so the synthesised main.c stays quiet),
  # which also defeats `-Werror=…`. A type-mismatched typedef in the
  # backend-emitted C — e.g. the array-pointer-typedef collision that
  # used to collapse distinct `ptr T` leaves onto a single
  # `_sym_arr_N_x` name — would otherwise be invisible here, hidden
  # behind a clean rc=0 from the warning suppression. Run a
  # `-fsyntax-only` gate over the backend's .c files (excluding the
  # synthesised main.c, which carries no RefractIR-generated typedefs) so
  # any such bug surfaces as a loud `cfail` instead of a silent pass.
  syn = subprocess.run(
    [
      clang,
      "-fsyntax-only",
      *c_paths,
      "-Wincompatible-pointer-types",
      "-Werror=incompatible-pointer-types",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=20,
  )
  if syn.returncode != 0:
    if verbose:
      print(syn.stderr)
    return ("cfail", f"{label}: clang syntax check failed")
  cc = subprocess.run(
    [
      clang,
      "-O0",
      *c_paths,
      main_c,
      "-o",
      exe,
      "-fsanitize=address,undefined",
      "-fno-sanitize-recover=all",
      "-w",
      "-lm",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=20,
  )
  if cc.returncode != 0:
    if verbose:
      print(cc.stderr)
    return ("cfail", f"{label}: clang failed")
  try:
    cr = subprocess.run(
      [exe], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10
    )
  except subprocess.TimeoutExpired:
    return ("cfail", f"{label}: c binary timeout")
  if cr.returncode != 0:
    head = cr.stderr.strip().splitlines()[0] if cr.stderr else "trap"
    return ("ubsan", f"{label}: {head}")
  cm = _RESULT_RE.search(cr.stdout)
  if cm is None:
    return ("cfail", f"{label}: no Result from C binary")
  c_val = cm.group(1)

  if sir_val == c_val:
    return ("passed", None)
  return ("mismatch", f"{label}: symiri={sir_val} c={c_val}")


# ───────────────────────────────────────────────────────────────────────
# Parallel cross-validation
#
# `-j` only parallelises the per-program classify step (symiri vs
# clang+exe).  Generation and batching stay sequential — the user is
# explicit that those phases are fast and ordered, and parallelising
# rysmith/rylink would tangle the on-disk artefact layout the rest of
# this script reads back.
# ───────────────────────────────────────────────────────────────────────


def _classify_with_scratch(
  idx, label, sir_path, c_paths, parsed, symiri, clang, out_dir, verbose
):
  """Per-task wrapper around `_classify`.  Each worker writes its own
  `_main_<idx>.c` and compiles to its own `_test_<idx>` so parallel
  jobs don't trample one another's scratch files.  Returns the same
  `(bucket, msg)` tuple `_classify` returns."""
  main_c = os.path.join(out_dir, f"_main_{idx}.c")
  exe = os.path.join(out_dir, f"_test_{idx}")
  return _classify(
    label, sir_path, c_paths, parsed, symiri, clang, main_c, exe, verbose
  )


_CLASSIFY_KEYS = (
  "idx",
  "label",
  "sir_path",
  "c_paths",
  "parsed",
  "symiri",
  "clang",
  "out_dir",
  "verbose",
)


def _classify_pool(tasks, jobs, fail_early):
  """Run `_classify_with_scratch` over `tasks` (a list of dicts).
  Only the keys in `_CLASSIFY_KEYS` are splatted into the worker;
  callers can stash extra side-data on the task dict (e.g. a
  `_prog_dir` for bug-saving) and read it back when the result
  surfaces.  Yields `(task, bucket, msg)` triples in submission order
  so the caller's bug-saving + reporting stays deterministic.

  When `jobs == 1`, runs in-line (cheaper than spinning a pool and
  preserves the exact subprocess sequencing of the pre-refactor code).
  When `jobs > 1`, dispatches via `ThreadPoolExecutor` — symiri /
  clang / the exe run as their own processes so GIL contention is
  negligible.

  `fail_early` short-circuits result iteration on the first failure
  bucket and cancels any pending futures; already-running tasks keep
  running (Python's ThreadPoolExecutor doesn't interrupt) but their
  results are discarded."""

  def kwargs(task):
    return {k: task[k] for k in _CLASSIFY_KEYS}

  if jobs <= 1 or len(tasks) <= 1:
    for task in tasks:
      bucket, msg = _classify_with_scratch(**kwargs(task))
      yield task, bucket, msg
      if fail_early and bucket not in ("passed", "skipped"):
        return
    return

  with ThreadPoolExecutor(max_workers=jobs) as ex:
    futures = [
      (task, ex.submit(_classify_with_scratch, **kwargs(task))) for task in tasks
    ]
    for task, fut in futures:
      bucket, msg = fut.result()
      yield task, bucket, msg
      if fail_early and bucket not in ("passed", "skipped"):
        for _, pending in futures:
          pending.cancel()
        return


# ───────────────────────────────────────────────────────────────────────
# Per-batch artefact management
# ───────────────────────────────────────────────────────────────────────


def _clear_batch_files(out_dir):
  """Remove per-batch artefacts (top-level .sir/.c/.json + a stale
  rylink/ subdirectory) while preserving the persistent harness files
  (_main.c, _test, bugs/)."""
  for f in os.listdir(out_dir):
    if f in ("_main.c", "_test", "bugs"):
      continue
    p = os.path.join(out_dir, f)
    if f == "rylink" and os.path.isdir(p):
      shutil.rmtree(p)
      continue
    if f.endswith(".sir") or f.endswith(".c") or f.endswith(".json"):
      os.remove(p)


def _save_leaf_bug(bugs_dir, bucket, sir_name, src_dir, batch_num):
  """Copy the failing leaf .sir (and companion .c) into bugs/<bucket>/."""
  dst_dir = os.path.join(bugs_dir, bucket)
  os.makedirs(dst_dir, exist_ok=True)
  stem = sir_name[:-4] if sir_name.endswith(".sir") else sir_name
  prefix = f"b{batch_num}_"
  for suffix in (".sir", ".c"):
    src = os.path.join(src_dir, stem + suffix)
    if os.path.exists(src):
      shutil.copy2(src, os.path.join(dst_dir, prefix + stem + suffix))


def _save_rylink_bug(bugs_dir, bucket, prog_dir, batch_num, flat=False):
  """Copy a failing rylink program into bugs/<bucket>/.

  When ``flat`` is True, ``prog_dir`` is actually the stem path
  (e.g. ``.../rylink/prog_abc_0``) and the artefacts live at
  ``<stem>.sir``, ``<stem>.c`` — mirroring rysmith's flat layout.
  Otherwise the default split-by-source layout is assumed:
  ``<prog_dir>/program.{sir,c}``."""
  dst_dir = os.path.join(bugs_dir, bucket)
  os.makedirs(dst_dir, exist_ok=True)
  if flat:
    prog_name = os.path.basename(prog_dir)
    prefix = f"b{batch_num}_{prog_name}_"
    for suffix in (".sir", ".c"):
      src = prog_dir + suffix
      if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, prefix + prog_name + suffix))
  else:
    prog_name = os.path.basename(prog_dir.rstrip("/"))
    prefix = f"b{batch_num}_{prog_name}_"
    for suffix in (".sir", ".c"):
      src = os.path.join(prog_dir, "program" + suffix)
      if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, prefix + "program" + suffix))


# ───────────────────────────────────────────────────────────────────────
# Reporting
# ───────────────────────────────────────────────────────────────────────


def _print_test_summary(passed, mismatch, ubsan, cfail, sirfail, skipped):
  print(f"  {green('passed')}:        {passed}")
  print(f"  {red('mismatched')}:    {len(mismatch)}")
  print(f"  {red('ubsan traps')}:   {len(ubsan)}")
  print(f"  {red('compile fail')}:  {len(cfail)}")
  print(f"  {red('symiri fail')}:   {len(sirfail)}")
  print(f"  {yellow('skipped')}:       {skipped}")


# ───────────────────────────────────────────────────────────────────────
# Main loop
# ───────────────────────────────────────────────────────────────────────


def run(
  rysmith,
  rylink,
  symiri,
  symirc,
  n,
  seed,
  out_dir,
  clang,
  verbose,
  fail_early=False,
  jobs=1,
):
  os.makedirs(out_dir, exist_ok=True)
  # Wipe stale artefacts from a previous run, including a prior bugs/ tree.
  for f in os.listdir(out_dir):
    p = os.path.join(out_dir, f)
    if os.path.isdir(p):
      shutil.rmtree(p)
    else:
      os.remove(p)
  bugs_dir = os.path.join(out_dir, "bugs")
  os.makedirs(bugs_dir, exist_ok=True)

  passed = 0
  mismatch = []
  ubsan = []
  cfail = []
  sirfail = []
  skipped = 0
  _LISTS = {"mismatch": mismatch, "ubsan": ubsan, "cfail": cfail, "sirfail": sirfail}

  # Scratch `_main_<idx>.c` / `_test_<idx>` paths are constructed
  # inside `_classify_with_scratch` so parallel workers don't collide.
  # The single-shared `_main.c` / `_test` files are no longer used.

  # Aggregate counters surfaced in the verbose end-of-run report.
  leaf_generated_total = 0
  leaf_gen_failed_total = 0
  rylink_generated_total = 0
  rylink_gen_failed_total = 0
  processed_total = 0

  stopped_early = False
  done = 0
  batch_idx = 0
  while done < n and not stopped_early:
    batch_n = min(BATCH_SIZE, n - done)
    batch_seed = seed + batch_idx
    _clear_batch_files(out_dir)

    # Per-batch coins, seeded from `batch_seed` so runs are
    # reproducible. rysmith and rylink flip independent structured
    # coins: a goto rysmith batch feeding a structured rylink batch
    # exercises the descriptor-driven seed filtering.
    batch_rng = random.Random(batch_seed)
    rys_structured = batch_rng.choice([True, False])

    # ── (1) rysmith generation ───────────────────────────────────────
    rys_cmd = [
      rysmith,
      "-n",
      str(batch_n),
      "--target",
      "c",
      "--seed",
      str(batch_seed),
      "-o",
      out_dir,
      # --emit-desc is required by rylink (it consumes the per-fn
      # descriptor JSON sidecar). Cheap when rylink is off, so always on.
      "--emit-desc",
    ]
    if rys_structured:
      rys_cmd += ["--structured-lowering", "random"]
    if verbose:
      print(bold(f"batch #{batch_idx + 1} rysmith generation ({done}/{n} programs):"))
      print(f"  running: {' '.join(rys_cmd)}")
    gen = subprocess.run(
      rys_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    batch_gen_failed = (gen.stderr or "").strip().count("[FAIL]")
    leaf_gen_failed_total += batch_gen_failed

    sirs = sorted(p for p in os.listdir(out_dir) if p.endswith(".sir"))
    leaf_generated_total += len(sirs)
    if verbose:
      print(f"  {green('succeeded')}:     {batch_n - batch_gen_failed}")
      print(f"  {yellow('failed')}:        {batch_gen_failed}")
      print(f"  generated:     {len(sirs)}")

    done += batch_n

    # ── (2) rysmith test ────────────────────────────────────────────────
    if verbose:
      print(bold(f"batch #{batch_idx + 1} rysmith test ({done}/{n} programs):"))
      print(f"  running: symiri vs clang+exe (jobs={jobs})")

    leaf_tasks = []
    for i, sir_name in enumerate(sirs):
      sir_path = os.path.join(out_dir, sir_name)
      c_path = sir_path[:-4] + ".c"
      leaf_tasks.append(
        {
          "idx": i,
          "label": sir_name,
          "sir_path": sir_path,
          "c_paths": [c_path],
          "parsed": _parse_program(sir_path),
          "symiri": symiri,
          "clang": clang,
          "out_dir": out_dir,
          "verbose": verbose,
        }
      )

    for task, bucket, msg in _classify_pool(leaf_tasks, jobs, fail_early):
      processed_total += 1
      if bucket == "passed":
        passed += 1
      elif bucket == "skipped":
        skipped += 1
      else:
        _LISTS[bucket].append(msg)
        _save_leaf_bug(bugs_dir, bucket, task["label"], out_dir, batch_idx + 1)
        if fail_early:
          print(
            f"  [batch {batch_idx + 1}, rysmith {task['label']}] "
            f"{red('first failure')} ({bucket}): {msg}",
            flush=True,
          )
          stopped_early = True
          break

    if not stopped_early:
      if verbose:
        _print_test_summary(passed, mismatch, ubsan, cfail, sirfail, skipped)
      else:
        print(
          f"  [batch #{batch_idx} functions: {done}/{n}] passed={passed} "
          f"mismatch={len(mismatch)} ubsan={len(ubsan)} "
          f"cfail={len(cfail)} sirfail={len(sirfail)} "
          f"skipped={skipped}",
          end="",
          flush=True,
        )
      print()

    # ── (3) rylink generation ────────────────────────────────────────
    ry_progs = []
    # Randomly decide whether to flatten the rylink output for this
    # batch (`batch_rng` was seeded from `batch_seed` at batch start).
    ry_flat = batch_rng.choice([True, False])
    ryl_structured = batch_rng.choice([True, False])
    if not stopped_early:
      ry_dir = os.path.join(out_dir, "rylink")
      # Mirror the rysmith batch size: one rylink program per rysmith
      # function in this batch. The previous version hard-coded
      # BATCH_SIZE (=100) here, so a `--n 5` run still asked rylink
      # for 100 programs and the final count ballooned to ~n + 100.
      ryl_cmd = [
        rylink,
        "-n",
        str(batch_n),
        "--target",
        "c",
        "--seed",
        str(batch_seed),
        "-i",
        out_dir,
        "-o",
        ry_dir,
      ]
      if ry_flat:
        ryl_cmd.append("--no-split-by-source")
      if ryl_structured:
        ryl_cmd += ["--structured-lowering", "random"]
      if verbose:
        print(bold(f"batch #{batch_idx + 1} rylink generation ({done}/{n} programs):"))
        print(f"  running: {' '.join(ryl_cmd)}")
      try:
        ry = subprocess.run(
          ryl_cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          timeout=300,
        )
      except subprocess.TimeoutExpired:
        ry = None
      if os.path.isdir(ry_dir):
        if ry_flat:
          # Flat layout: `<ry_dir>/prog_<id>_<idx>.sir` — collect the
          # stem paths (without extension), mirroring rysmith's layout.
          ry_progs = sorted(
            os.path.join(ry_dir, f[:-4])
            for f in os.listdir(ry_dir)
            if f.startswith("prog_") and f.endswith(".sir")
          )
        else:
          ry_progs = sorted(
            os.path.join(ry_dir, d)
            for d in os.listdir(ry_dir)
            if d.startswith("prog_") and os.path.isdir(os.path.join(ry_dir, d))
          )
      rylink_generated_total += len(ry_progs)
      ry_failed = batch_n - len(ry_progs)
      if ry is None:
        ry_failed = batch_n
      rylink_gen_failed_total += max(0, ry_failed)
      if verbose:
        flat_tag = " (flat)" if ry_flat else ""
        print(f"  {green('succeeded')}:     {len(ry_progs)}{flat_tag}")
        print(f"  {yellow('failed')}:        {max(0, ry_failed)}")
        print(f"  generated:     {len(ry_progs)}")

    # ── (4) rylink test ──────────────────────────────────────────────
    if not stopped_early:
      if verbose:
        print(bold(f"batch #{batch_idx + 1} rylink test ({done}/{n} programs):"))
        print(f"  running: symiri vs clang+exe (jobs={jobs})")

      rylink_tasks = []
      for i, prog_entry in enumerate(ry_progs):
        if ry_flat:
          # `prog_entry` is a stem path, e.g. `<ry_dir>/prog_abc_0`.
          prog_name = os.path.basename(prog_entry)
          sir_path = prog_entry + ".sir"
          c_path = prog_entry + ".c"
          c_paths = [c_path] if os.path.exists(c_path) else []
        else:
          prog_name = os.path.basename(prog_entry.rstrip("/"))
          sir_path = os.path.join(prog_entry, "program.sir")
          # rylink --split-by-source emits one .c per FunDecl::sourceStem
          # plus an (empty) program.c. Hand them all to clang together so
          # cross-stem `call @callee(...)` references resolve at link
          # time. Sorted only for determinism in failure messages.
          c_paths = sorted(
            os.path.join(prog_entry, f)
            for f in os.listdir(prog_entry)
            if f.endswith(".c")
          )
        rylink_tasks.append(
          {
            "idx": i,
            "label": prog_name,
            "sir_path": sir_path,
            "c_paths": c_paths,
            "parsed": _parse_program(sir_path),
            "symiri": symiri,
            "clang": clang,
            "out_dir": out_dir,
            "verbose": verbose,
            # Held on the side for `_save_rylink_bug` after the pool
            # surfaces the result — the worker itself doesn't need it.
            "_prog_dir": prog_entry,
            "_flat": ry_flat,
          }
        )

      for task, bucket, msg in _classify_pool(rylink_tasks, jobs, fail_early):
        processed_total += 1
        if bucket == "passed":
          passed += 1
        elif bucket == "skipped":
          skipped += 1
        else:
          _LISTS[bucket].append(msg)
          _save_rylink_bug(
            bugs_dir,
            bucket,
            task["_prog_dir"],
            batch_idx + 1,
            flat=task.get("_flat", False),
          )
          if fail_early:
            print(
              f"  [batch {batch_idx + 1}, rylink {task['label']}] "
              f"{red('first failure')} ({bucket}): {msg}",
              flush=True,
            )
            stopped_early = True
            break

    if not stopped_early:
      if verbose:
        _print_test_summary(passed, mismatch, ubsan, cfail, sirfail, skipped)
      else:
        print(
          f"  [batch #{batch_idx} programs: {done}/{n}] passed={passed} "
          f"mismatch={len(mismatch)} ubsan={len(ubsan)} "
          f"cfail={len(cfail)} sirfail={len(sirfail)} "
          f"skipped={skipped}",
          end="",
          flush=True,
        )
      print()

    batch_idx += 1

  # End-of-run report.
  print()
  hdr = f"reify differential test (n={processed_total}, seed={seed}"
  if stopped_early:
    hdr += ", stopped early"
  hdr += "):"
  print(bold(hdr))
  if verbose:
    print(f"  rysmith generated:     {leaf_generated_total}")
    print(f"  {red('rysmith gen failed')}:    {leaf_gen_failed_total}")
    print(f"  rylink generated:      {rylink_generated_total}")
    print(f"  {red('rylink gen failed')}:     {rylink_gen_failed_total}")
    print("  ---")
  _print_test_summary(passed, mismatch, ubsan, cfail, sirfail, skipped)

  fail_lines = mismatch + ubsan + cfail + sirfail
  if fail_lines:
    print(bold("\nFailures:"))
    for line in fail_lines[:20]:
      print(f"  {line}")
    if len(fail_lines) > 20:
      print(f"  ... and {len(fail_lines) - 20} more")
    print(
      f"\n  {yellow('saved')} {len(fail_lines)} failing case(s) to "
      f"{bugs_dir}/<bucket>/  (.sir + .c)"
    )

  return len(fail_lines) == 0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--rysmith", default="./rysmith")
  ap.add_argument("--rylink", default="./rylink")
  ap.add_argument("--symiri", default="./symiri")
  ap.add_argument("--symirc", default="./symirc")
  ap.add_argument("--clang", default="clang")
  ap.add_argument(
    "-n",
    "--num",
    type=int,
    default=100,
    help="Total number of programs to generate and test",
  )
  ap.add_argument(
    "--seed",
    type=int,
    default=1234,
    help="Fixed seed so make test is deterministic",
  )
  ap.add_argument("--out", default="build/test_tmp/reify_diff")
  ap.add_argument("--verbose", "-v", action="store_true")
  ap.add_argument(
    "--fail-early",
    action="store_true",
    help="Stop at the first mismatch/ubsan/cfail/sirfail instead of finishing the batch",
  )
  ap.add_argument(
    "-j",
    "--jobs",
    type=int,
    default=1,
    help=(
      "Parallelism for the symiri-vs-C cross-validation step within a "
      "batch.  Batches and generation stay sequential.  Default 1 "
      "(in-line, identical to the historical behaviour)."
    ),
  )
  args = ap.parse_args()
  if args.jobs < 1:
    print(red("error: -j/--jobs must be >= 1"), file=sys.stderr)
    sys.exit(2)

  for tool, path in (
    ("rysmith", args.rysmith),
    ("symiri", args.symiri),
    ("symirc", args.symirc),
  ):
    if not os.path.exists(path):
      print(red(f"error: {tool} not found at {path}"), file=sys.stderr)
      sys.exit(2)
  if args.rylink and not os.path.exists(args.rylink):
    print(red(f"error: rylink not found at {args.rylink}"), file=sys.stderr)
    sys.exit(2)
  if not shutil.which(args.clang):
    print(red(f"error: clang '{args.clang}' not on PATH"), file=sys.stderr)
    sys.exit(2)

  ok = run(
    args.rysmith,
    args.rylink,
    args.symiri,
    args.symirc,
    args.num,
    args.seed,
    args.out,
    args.clang,
    args.verbose,
    fail_early=args.fail_early,
    jobs=args.jobs,
  )
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
