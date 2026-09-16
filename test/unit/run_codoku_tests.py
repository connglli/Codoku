"""End-to-end tests for the rysmith-based codoku (./codokus/).

The new codoku drives rysmith directly (no rypuzmk/rypuzchk binaries) and
uses the vendored puzzle_common.py / checker.py with angle-bracketed
<FILL_XXX> mask tokens.  It must:

  (1) generate a puzzle with `codoku create` (or bare `codoku`), deterministically
      for a given --seed, writing puzzle.py + INSTRUCTION.md + oracle/ into the
      output directory;
  (2) use <FILL_XXX> mask tokens and the `codoku check` validation command;
  (3) validate a solution with `codoku check`, accepting the ground truth;
  (4) reject unknown profiles/flags;
  (5) keep generated masks/budget consistent (re-mask self-check passes).

Run as:

  python3 -m test.unit.run_codoku_tests <codoku.py> <rysmith>
"""

import ast
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

GREEN = "\033[32m"
RED = "\033[31m"
GRAY = "\033[90m"
NC = "\033[0m"

results = []


def run(cmd, cwd=None, **kw):
  print(f"  {GRAY}[RUN>]{NC} " + " ".join(cmd))
  return subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=cwd, **kw)


def run_codoku(bin_path, args, cwd):
  """Run the codoku script with the current interpreter."""
  return run([sys.executable, bin_path] + args, cwd=cwd)


def check(name, ok, detail=""):
  results.append((name, ok, detail))
  color = GREEN if ok else RED
  tag = "PASS" if ok else "FAIL"
  print(f"  [{color}{tag}{NC}] {name}" + (f" ({detail})" if detail and not ok else ""))


def import_codoku(codoku_path):
  """Import codoku_creator as a module (for direct unit tests)."""
  real = os.path.realpath(codoku_path)
  module_dir = os.path.dirname(real)
  if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
  creator = os.path.join(module_dir, "codoku_creator.py")
  spec = importlib.util.spec_from_file_location("_codoku_gen_mod", creator)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def import_codoku_checker(codoku_path):
  """Import codoku_checker as a module (for direct unit tests)."""
  module_dir = os.path.dirname(os.path.realpath(codoku_path))
  if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
  checker = os.path.join(module_dir, "codoku_checker.py")
  spec = importlib.util.spec_from_file_location("_codoku_chk_mod", checker)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def import_codoku_common(codoku_path):
  """Import codoku_common as a module (for direct unit tests)."""
  module_dir = os.path.dirname(os.path.realpath(codoku_path))
  if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
  cfile = os.path.join(module_dir, "codoku_common.py")
  spec = importlib.util.spec_from_file_location("_codoku_common_mod", cfile)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def import_codoku_complexity(codoku_path):
  """Import codoku_complexity as a module (for direct unit tests)."""
  module_dir = os.path.dirname(os.path.realpath(codoku_path))
  if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
  cfile = os.path.join(module_dir, "codoku_complexity.py")
  spec = importlib.util.spec_from_file_location("_codoku_cplx_mod", cfile)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def complexity_unit_tests_pass(cmod) -> bool:
  ok = True

  # Halstead vocabulary on `def f(a, b): c = a + b; return c`:
  # operators {FunctionDef, arguments, Assign, BinOp, Add, Return},
  # operands {a x2, b x2, c x2}.
  leaf = ast.parse("def f(a, b):\n    c = a + b\n    return c\n").body[0]
  h = cmod.compute_halstead(leaf)
  good = (
    (h.operators, h.operands, h.total_operators, h.total_operands) == (6, 3, 6, 6)
    and h.vocabulary == 9
    and h.length == 12
    and h.volume == round(12 * math.log2(9), 2)
    and h.difficulty == 6.0
    and h.effort == round(6.0 * h.volume, 2)
  )
  check("unit: halstead counts (no time/defects)", good, str(h))
  ok = ok and good

  # DepDegree: params -> stmt -> return; the two param uses open one pair
  # edge, plus one edge for c.
  d = cmod.compute_depdegree(leaf)
  good = (
    (d.nodes, d.edges) == (3, 2)
    and d.avg_degree == round(2 * 2 / 3, 2)
    and d.max_degree == 2
    and d.density == round(2 / (3 * 2), 4)
  )
  check("unit: depdegree counts", good, str(d))
  ok = ok and good

  # McCabe: straight line is 1, if/else diamond is 2, empty is 0.
  m_line = cmod.compute_mccabe({"a", "b", "c"}, [("a", "b"), ("b", "c")])
  m_diamond = cmod.compute_mccabe(
    {"entry", "b0", "b1", "exit"},
    [("entry", "b0"), ("entry", "b1"), ("b0", "exit"), ("b1", "exit")],
  )
  m_empty = cmod.compute_mccabe(set(), [])
  good = (
    (m_line.nodes, m_line.edges, m_line.cyclomatic) == (3, 2, 1)
    and (m_diamond.nodes, m_diamond.edges, m_diamond.cyclomatic) == (4, 4, 2)
    and (m_empty.nodes, m_empty.edges, m_empty.cyclomatic) == (0, 0, 0)
    and cmod.McCabeMetrics.zero() == m_empty
  )
  check("unit: mccabe counts", good, f"{m_line} {m_diamond} {m_empty}")
  ok = ok and good

  # Estimated time (E/18) and defects (V/3000) are deliberately absent.
  import dataclasses

  hal_fields = {f.name for f in dataclasses.fields(cmod.HalsteadMetrics)}
  est_fields = {f.name for f in dataclasses.fields(cmod.ComplexityEstimate)}
  good = not any(
    "time" in f or "bug" in f or "defect" in f for f in hal_fields | est_fields
  )
  check("unit: no time/defect heuristics", good, str(sorted(hal_fields | est_fields)))
  ok = ok and good

  # Vocab + dataflow are static terms inside static_struct (no separate axes).
  metrics = cmod.PuzzleMetrics(
    cfg_nodes=2,
    cfg_edges=1,
    cyclomatic=1,
    n_loops=0,
    exec_path_length=2,
    unique_path_blocks=2,
    repeated_block_visits=0,
    max_block_visits=1,
    loop_iterations_total=0,
    loop_iterations_avg=0.0,
    total_masks=0,
    masks_by_kind={},
    sol_space_log10=0.0,
    const_budget_entries=0,
    const_budget_total=0,
    source_lines=5,
    non_comment_source_lines=3,
    hal_operators=6,
    hal_operands=3,
    hal_total_operators=6,
    hal_total_operands=6,
    hal_vocabulary=9,
    hal_length=12,
    hal_volume=38.04,
    hal_difficulty=6.0,
    hal_effort=228.24,
    dep_nodes=3,
    dep_edges=2,
    dep_avg_degree=1.33,
    dep_max_degree=2,
    dep_density=0.3333,
  )
  # Vocab + dataflow fold into static_struct (no separate axes); total is
  # the four-axis sum.
  est = cmod.estimate_complexity(metrics)
  parts = est.static_struct + est.dynamic_trace + est.masking + est.constraints
  good = (
    not hasattr(est, "vocab")
    and not hasattr(est, "dataflow")
    and abs(
      est.static_struct - (2 + 1 + 2 + 0.15 + 0.01 * 38.04 + 0.5 * 6.0 + 2.0 * 1.33 + 2)
    )
    < 0.01
    and abs(est.total - parts) < 1e-9
  )
  check("unit: vocab/dataflow folded into static", good, str(est))
  ok = ok and good

  # Unparsable input measures zero instead of raising.
  tmpdir = tempfile.mkdtemp(prefix="codoku_cplx_edge_")
  try:
    bad = os.path.join(tmpdir, "puzzle.py")
    with open(bad, "w") as f:
      f.write("def broken(:\n  <FILL_VAR> ???\n")
    from pathlib import Path

    edge_metrics = cmod.analyze_puzzle(Path(bad))
    edge_est = cmod.estimate_complexity(edge_metrics)
    edge_parts = (
      edge_est.static_struct
      + edge_est.dynamic_trace
      + edge_est.masking
      + edge_est.constraints
    )
    good = (
      edge_metrics.hal_volume == 0.0
      and edge_metrics.dep_nodes == 0
      and abs(edge_est.total - edge_parts) < 1e-9
    )
    check("unit: unparsable puzzle measures zero", good, str(edge_metrics))
    ok = ok and good
  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

  # Vocab/dataflow come from the ground truth when available: the explicit
  # gt_path, else the oracle sibling.  Markers still come from the puzzle.
  gt_body = "def func_tiny(a, b):\n    c = a + b\n    d = c * 2\n    return d\n"
  masked_body = (
    "#//@ CFG_EDGE: entry -> exit\n"
    "#//@ EXEC_PATH: entry -> exit\n"
    "def func_tiny(a, b):\n"
    "    <FILL_VAR> = a + <FILL_CONST>\n"
    "    <FILL_VAR> = <FILL_VAR> <FILL_OP> <FILL_CONST>\n"
    "    return <FILL_VAR>\n"
  )
  gt_dir = tempfile.mkdtemp(prefix="codoku_cplx_gt_")
  try:
    from pathlib import Path

    outdir = Path(gt_dir) / "out"
    (outdir / "oracle").mkdir(parents=True)
    puzzle_path = outdir / "puzzle.py"
    puzzle_path.write_text(masked_body)
    gt_path = outdir / "oracle" / "puzzle.gt.py"
    gt_path.write_text(gt_body)
    met_gt = cmod.analyze_puzzle(gt_path)
    met_explicit = cmod.analyze_puzzle(puzzle_path, gt_path=gt_path)
    met_found = cmod.analyze_puzzle(puzzle_path)
    good = (
      met_explicit.hal_volume == met_gt.hal_volume
      and (met_explicit.dep_nodes, met_explicit.dep_edges)
      == (met_gt.dep_nodes, met_gt.dep_edges)
      and met_found.hal_volume == met_gt.hal_volume
      and met_found.dep_max_degree == met_gt.dep_max_degree
      and met_found.cfg_nodes == 2
      and met_found.exec_path_length == 2
      and met_found.total_masks > 0
    )
    check("unit: vocab/dataflow measured on GT", good, str(met_found))
    ok = ok and good

    # The vocab source is visible, not silent: text names the GT file (or
    # says fallback), --json carries it, and --ground-truth names it.
    import io
    import json as json_lib
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
      rc_text = cmod.main([str(puzzle_path)])
    text_out = buf.getvalue()
    buf = io.StringIO()
    with redirect_stdout(buf):
      rc_json = cmod.main([str(puzzle_path), "--json"])
    payload = json_lib.loads(buf.getvalue())
    elsewhere = outdir / "elsewhere_gt.py"
    elsewhere.write_text(gt_body)
    buf = io.StringIO()
    with redirect_stdout(buf):
      rc_flag = cmod.main([str(puzzle_path), "--ground-truth", str(elsewhere)])
    # The flag wins even with different content: bare puzzle, no sibling.
    bare_dir = Path(gt_dir) / "bare"
    bare_dir.mkdir()
    lonely = bare_dir / "lonely.py"
    lonely.write_text(masked_body)
    other_gt = bare_dir / "other.py"
    other_gt.write_text(gt_body.replace("c * 2", "c * 2 + a - b"))
    met_flagged = cmod.analyze_puzzle(lonely, gt_path=other_gt)
    met_other = cmod.analyze_puzzle(other_gt)
    good = (
      rc_text == 0
      and str(gt_path) in text_out
      and rc_json == 0
      and payload.get("vocab_source") == str(gt_path)
      and rc_flag == 0
      and cmod.resolve_vocab_source(puzzle_path) == gt_path
      and cmod.resolve_vocab_source(lonely) is None
      and met_flagged.hal_volume == met_other.hal_volume
      and met_flagged.hal_volume != met_gt.hal_volume
    )
    check("unit: vocab source visible + --ground-truth", good, text_out[-200:])
    ok = ok and good

    # Review fixes: trace scaffolding is not puzzle logic, subscript
    # AugAssign redefines the base, and an unusable explicit GT is an error.
    trace_if = (
      '    if __import__("os").environ.get("DUMP_TRACE"):\n        print("^b0:")\n'
    )
    traced_src = "def f(a, b):\n" + trace_if + "    c = a + b\n    return c\n"
    traced_leaf = ast.parse(traced_src).body[0]
    h_traced = cmod.compute_halstead(traced_leaf)
    d_traced = cmod.compute_depdegree(traced_leaf)
    hal_ok = (
      h_traced.operators,
      h_traced.operands,
      h_traced.total_operators,
      h_traced.total_operands,
    ) == (6, 3, 6, 6)
    dep_ok = (d_traced.nodes, d_traced.edges) == (3, 2)
    good = hal_ok and dep_ok
    check("unit: trace scaffolding excluded", good, f"{h_traced} {d_traced}")
    ok = ok and good

    collector = cmod._DepCollector()
    collector.visit(ast.parse("a[i] += v\n").body[0])
    (a_defs, a_uses), *_ = collector.stmts
    good = a_defs == {"a"} and {"i", "v", "a"} <= a_uses
    check("unit: augassign subscript redefines base", good, f"{a_defs} {a_uses}")
    ok = ok and good

    from contextlib import redirect_stderr

    err = io.StringIO()
    with redirect_stderr(err):
      rc_bad = cmod.main([str(puzzle_path), "--ground-truth", str(outdir / "nope.py")])
    good = rc_bad == 2 and "ground-truth" in err.getvalue().lower()
    check("unit: unusable --ground-truth errors", good, err.getvalue()[-200:])
    ok = ok and good
  finally:
    shutil.rmtree(gt_dir, ignore_errors=True)

  return ok


def unit_tests_pass(mod) -> bool:
  ok = True

  # Profiles validate and sample within ranges.
  for name in mod.PROFILES:
    prof = mod.PROFILES[name]
    prof.validate(name)
    rng = __import__("random").Random(1)
    for _ in range(20):
      cfg = mod.sample_config(prof, rng)
      lo, hi = prof.n_bbls.minimum, prof.n_bbls.maximum
      if not lo <= cfg.n_bbls <= hi:
        check(f"unit: {name} bbls within range", False)
        ok = False
        break
    else:
      check(f"unit: {name} samples within ranges", True)

  # easy is the integer-scalar profile (no fp/vec/intrinsics, no pointers);
  # hard is full-featured.
  easy_feats = set(mod.PROFILES["easy"].features)
  easy_ptr = mod.PROFILES["easy"].max_ptr_depth
  if not ({"--no-fp", "--no-vec"} <= easy_feats) or (
    easy_ptr.minimum,
    easy_ptr.maximum,
  ) != (0, 0):
    check(
      "unit: easy uses reduced feature set", False, str(easy_feats) + f" ptr={easy_ptr}"
    )
    ok = False
  else:
    check("unit: easy uses reduced feature set", True)

  # Banner extraction is fail-closed: a .sir missing the trusted
  # comments raises instead of degrading the puzzle banner to [unknown],
  # and a present `// CFG:` header with no edges is legitimate.
  from pathlib import Path

  with tempfile.TemporaryDirectory(prefix="codoku_banner_") as banner_dir:
    bfd = Path(banner_dir)
    cfg_only = bfd / "cfg_only.sir"
    cfg_only.write_text("// CFG:\n//   entry -> exit\n")
    bare = bfd / "bare.sir"
    bare.write_text("def func_t():\n    return 0\n")

    def raises(fn, arg):
      try:
        fn(arg)
      except RuntimeError:
        return True
      return False

    closed = (
      raises(mod.extract_path_from_sir, cfg_only)
      and raises(mod.extract_path_from_sir, bare)
      and raises(mod.extract_cfg_from_sir, bare)
      and mod.extract_cfg_from_sir(cfg_only) == [("entry", "exit")]
    )
    check("unit: banner extraction is fail-closed", closed)

  return ok


def goto_flag_unit_tests_pass(ccommon, chk_mod) -> bool:
  """Structured-goto flags (_go_/_brk_/_cnt_) must all be masked by target-hiding."""
  ok = True

  # (a) Target decoding covers every flag kind and rejects ordinary names.
  cases = {
    "_go_exit": "exit",
    "_go_b6": "b6",
    "_brk_exit": "exit",
    "_brk_b2": "b2",
    "_break_exit": "exit",
    "_cnt_b0": "b0",
    "_continue_b0": "b0",
    "acc": None,
    "_brk": None,
    "_cnt_": "",
  }
  ok_a = True
  for name, want in cases.items():
    got = ccommon.goto_flag_target(name)
    if got != want:
      check(f"unit: goto_flag_target({name})", False, f"got {got!r}, want {want!r}")
      ok_a = False
      ok = False
  if ok_a:
    check("unit: goto_flag_target decodes all flag kinds", True)

  # (b) The maskable-statement predicate fires for every flag spelling.
  mod_src = (
    "def func_test(x):\n"
    "    _brk_exit = False\n"
    "    _cnt_b0 = False\n"
    "    acc = 0\n"
    "    # ^entry\n"
    "    acc = (x + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        acc = (acc + x)\n"
    "        if (acc > 10):\n"
    "            _brk_exit = True\n"
    "            break\n"
    "        _cnt_b0 = True\n"
    "        break\n"
    "    if _brk_exit:\n"
    "        _brk_exit = False\n"
    "        acc = (acc + 1)\n"
    "    if _cnt_b0:\n"
    "        _cnt_b0 = False\n"
    "        acc = (acc + 2)\n"
    "    # ^exit\n"
    "    return acc\n"
  )
  mod_bytes = mod_src.encode("utf-8")
  tree = ast.parse(mod_bytes)
  leaf, _ = ccommon.find_python_leaf_function(tree, mod_bytes)
  if leaf is None:
    check("unit: flag fixture has a leaf function", False, "no leaf found")
    return False
  maskable, entry_line, _ = ccommon.get_python_maskable_statements(leaf, mod_bytes)
  sees_brk = any(ccommon.mentions_go_flag(s) for s in maskable)
  if not sees_brk:
    check("unit: maskable scan sees _brk_/_cnt_ flags", False, "no flagged stmt")
    ok = False
  else:
    check("unit: maskable scan sees _brk_/_cnt_ flags", True)
  plain = ast.parse(b"def func_test(x):\n    y = (x + 1)\n    return y\n")
  if ccommon.mentions_go_flag(plain):
    check("unit: plain code has no goto flags", False, "false positive")
    ok = False
  else:
    check("unit: plain code has no goto flags", True)

  # (c) Masking hides every flag target; _brk_/_cnt_ also hide the kind.
  local_names = ccommon.collect_python_leaf_locals(leaf)
  defined_funcs = set()
  repls: list = []
  for stmt in maskable:
    ccommon.collect_python_replacements(
      stmt, mod_bytes, stmt.lineno > entry_line, repls, {}, local_names, defined_funcs
    )
  masked = ccommon.apply_replacements(mod_bytes, repls).decode("utf-8")
  leaked = sorted(
    set(re.findall(r"_(?:go|brk|cnt)_[A-Za-z0-9]+", masked))
    - {"_go_<FILL_LABEL>", "_<FILL_CTRL>_<FILL_LABEL>"}
  )
  if leaked:
    check("unit: no concrete flag target survives masking", False, str(leaked))
    ok = False
  else:
    check("unit: no concrete flag target survives masking", True)
  if "_<FILL_CTRL>_<FILL_LABEL>" not in masked:
    check("unit: brk/cnt flags hide kind and target", False, "missing compound form")
    ok = False
  else:
    check("unit: brk/cnt flags hide kind and target", True)

  # (c1) Full-word spellings mask identically (the checker accepts both).
  ok_words = True
  for spelled in ("_break_exit", "_continue_b0"):
    node = ast.parse(spelled, mode="eval").body
    src_bytes = spelled.encode("utf-8")
    word_repls: list = []
    ccommon.collect_python_replacements(
      node, src_bytes, True, word_repls, {}, set(), set()
    )
    word_masked = ccommon.apply_replacements(src_bytes, word_repls).decode("utf-8")
    if word_masked != "_<FILL_CTRL>_<FILL_LABEL>":
      check(f"unit: {spelled} masks to compound token", False, word_masked)
      ok_words = False
      ok = False
  if ok_words:
    check("unit: full-word flag spellings mask identically", True)

  # (c2) _go_ keeps its kind prefix (a general goto pairs with no keyword).
  go_src = (
    "def func_g(x):\n"
    "    _go_b1 = False\n"
    "    # ^entry\n"
    "    _go_b1 = True\n"
    "    # ^b0\n"
    "    y = (x + 1)\n"
    "    # ^b1\n"
    "    y = (y + 2)\n"
    "    # ^exit\n"
    "    return y\n"
  ).encode("utf-8")
  go_tree = ast.parse(go_src)
  go_leaf, _ = ccommon.find_python_leaf_function(go_tree, go_src)
  go_maskable, go_entry, _ = ccommon.get_python_maskable_statements(go_leaf, go_src)
  go_locals = ccommon.collect_python_leaf_locals(go_leaf)
  go_repls: list = []
  for stmt in go_maskable:
    ccommon.collect_python_replacements(
      stmt, go_src, stmt.lineno > go_entry, go_repls, {}, go_locals, set()
    )
  go_masked = ccommon.apply_replacements(go_src, go_repls).decode("utf-8")
  if "_go_<FILL_LABEL>" not in go_masked or "_<FILL_CTRL>_<FILL_LABEL>" in go_masked:
    check("unit: go flags keep their kind prefix", False, go_masked)
    ok = False
  else:
    check("unit: go flags keep their kind prefix", True)

  # (c3) Exit-block plumbing: guard test, reset and keyword all maskable.
  exit_src = (
    "def func_e(x):\n"
    "    # ^entry\n"
    "    y = (x + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        y = (y + x)\n"
    "        if (y > 10):\n"
    "            # ^exit\n"
    "            y = (y + 1)\n"
    "            return y\n"
    "        if _cnt_b0:\n"
    "            _cnt_b0 = False\n"
    "            continue\n"
    "    return y\n"
  ).encode("utf-8")
  exit_tree = ast.parse(exit_src)
  exit_leaf, _ = ccommon.find_python_leaf_function(exit_tree, exit_src)
  exit_maskable, exit_entry, _ = ccommon.get_python_maskable_statements(
    exit_leaf, exit_src
  )
  exit_locals = ccommon.collect_python_leaf_locals(exit_leaf)
  exit_repls: list = []
  for stmt in exit_maskable:
    ccommon.collect_python_replacements(
      stmt, exit_src, stmt.lineno > exit_entry, exit_repls, {}, exit_locals, set()
    )
  exit_masked = ccommon.apply_replacements(exit_src, exit_repls).decode("utf-8")
  exit_leaks = sorted(
    set(re.findall(r"_(?:go|brk|cnt)_[A-Za-z0-9]+", exit_masked))
    - {"_go_<FILL_LABEL>", "_<FILL_CTRL>_<FILL_LABEL>"}
  )
  if exit_leaks or exit_masked.count("_<FILL_CTRL>_<FILL_LABEL>") != 2:
    check("unit: exit-guard reset masked with its guard", False, exit_masked)
    ok = False
  else:
    check("unit: exit-guard reset masked with its guard", True)
  if "<FILL_CTRL>" not in exit_masked:
    check("unit: exit-guard keyword masked", False, exit_masked)
    ok = False
  else:
    check("unit: exit-guard keyword masked", True)

  # (c4) Flag setter in the exit region: an `else:` setter outside body
  # blocks must mask too, or it answers the dispatch guard.
  setter_src = (
    "def func_s(x):\n"
    "    # ^entry\n"
    "    y = (x + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        y = (y + x)\n"
    "        if (y > 10):\n"
    "            # ^exit\n"
    "            y = (y + 1)\n"
    "            return y\n"
    "        else:\n"
    "            _cnt_b0 = True\n"
    "            break\n"
    "    return y\n"
  ).encode("utf-8")
  setter_tree = ast.parse(setter_src)
  setter_leaf, _ = ccommon.find_python_leaf_function(setter_tree, setter_src)
  setter_maskable, setter_entry, _ = ccommon.get_python_maskable_statements(
    setter_leaf, setter_src
  )
  setter_locals = ccommon.collect_python_leaf_locals(setter_leaf)
  setter_repls: list = []
  for stmt in setter_maskable:
    ccommon.collect_python_replacements(
      stmt,
      setter_src,
      stmt.lineno > setter_entry,
      setter_repls,
      {},
      setter_locals,
      set(),
    )
  setter_masked = ccommon.apply_replacements(setter_src, setter_repls).decode("utf-8")
  setter_leaks = sorted(
    set(re.findall(r"_(?:go|brk|cnt)_[A-Za-z0-9]+", setter_masked))
    - {"_go_<FILL_LABEL>", "_<FILL_CTRL>_<FILL_LABEL>"}
  )
  if setter_leaks or "_<FILL_CTRL>_<FILL_LABEL>" not in setter_masked:
    check("unit: exit-region setter masked", False, setter_masked)
    ok = False
  else:
    check("unit: exit-region setter masked", True)

  # (e) Dispatch transfer equality: a guard must land where its flag names.
  transfer_src = (
    "def func_d(x):\n"
    "    # ^entry\n"
    "    y = (x + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        y = (y + x)\n"
    "        if _brk_exit:\n"
    "            _brk_exit = False\n"
    "            break\n"
    "        if not _cnt_b0:\n"
    "            y = (y + 2)\n"
    "        else:\n"
    "            _cnt_b0 = False\n"
    "            continue\n"
    "        if _go_exit:\n"
    "            break\n"
    "    # ^exit\n"
    "    return y\n"
  ).encode("utf-8")
  transfer_tree = ast.parse(transfer_src)
  transfer_leaf, _ = ccommon.find_python_leaf_function(transfer_tree, transfer_src)
  dispatches = ccommon.iter_flag_dispatches(transfer_leaf, transfer_src)
  got = sorted((f, e, s) for f, e, s, _ in dispatches)
  want = sorted(
    [
      ("_brk_exit", "exit", "exit"),
      ("_cnt_b0", "b0", "b0"),
      ("_go_exit", "exit", "exit"),
    ]
  )
  if got != want:
    check("unit: dispatch transfers follow the loop model", False, str(got))
    ok = False
  else:
    check("unit: dispatch transfers follow the loop model", True)

  # (e2) End to end: a mistargeted dispatch fails check_cfg with FAIL_CFG
  # while the well-targeted original passes. The fixtures below are real
  # lowered shapes (see MULTI_LEVEL_SRC / SELF_LOOP_SRC), so their flag
  # targets are declared by the SIR-projected edges.
  multi_src = MULTI_LEVEL_SRC.encode("utf-8")
  multi_leaf, _ = ccommon.find_python_leaf_function(ast.parse(multi_src), multi_src)
  actual_edges = sorted(ccommon.build_python_cfg(multi_leaf, multi_src))
  bad_transfer = MULTI_LEVEL_SRC.replace("_brk_exit", "_brk_b0").encode("utf-8")
  bad_leaf, _ = ccommon.find_python_leaf_function(ast.parse(bad_transfer), bad_transfer)
  bad_edges = sorted(ccommon.build_python_cfg(bad_leaf, bad_transfer))
  for src, edges, want_ok in (
    (multi_src, actual_edges, True),
    (bad_transfer, bad_edges, False),
  ):
    t = ast.parse(src)
    node = t.body[0]
    label = "accepts true flag target" if want_ok else "rejects mistargeted flag"
    try:
      chk_mod.check_cfg(node, src, edges)
      passed = want_ok
    except chk_mod.CheckFailure as exc:
      passed = (not want_ok) and exc.result == chk_mod.CheckResult.FAIL_CFG
      if not want_ok and "claims target" not in str(exc.message):
        passed = False
    except Exception as exc:  # noqa: BLE001 - any other error is a failure
      check(f"unit: checker {label}", False, str(exc))
      ok = False
      continue
    check(f"unit: checker {label}", passed)
    ok = ok and passed
  self_src = SELF_LOOP_SRC.encode("utf-8")
  good_src, good_edges = self_src, SELF_LOOP_EDGES
  bad_src = SELF_LOOP_SRC.replace("_brk_exit", "_brk_nope").encode("utf-8")
  for src, edges, want_ok in (
    (good_src, good_edges, True),
    (bad_src, good_edges, False),
  ):
    t = ast.parse(src)
    node = t.body[0]
    label = (
      "accepts declared flag targets" if want_ok else "rejects unknown flag target"
    )
    try:
      chk_mod.check_cfg(node, src, edges)
      passed = want_ok
    except chk_mod.CheckFailure as exc:
      passed = (not want_ok) and exc.result == chk_mod.CheckResult.FAIL_CFG
    except Exception as exc:  # noqa: BLE001 - any other error is a failure
      check(f"unit: checker {label}", False, str(exc))
      ok = False
      continue
    check(f"unit: checker {label}", passed)
    ok = ok and passed

  return ok


MULTI_LEVEL_SRC = """\
def func_493ad2_0(pa0, pa1):
    _brk_exit = False
    _cnt_entry = False
    v0 = 16325
    v1 = 9223372036854775807
    v2 = 2147483646
    v__chk = 0
    while True:
        # ^entry
        v2 = (-1310722 + _cast_int(pa0, 32))
        v2 = ((536608768 - 536915912) + _cast_int(v0, 32))
        if (((_cast_int(536870912, 64) - 1) - (0 if (v0) <= (0) else 1073741829))) < ((_cast_int(1610612715, 64) + (-2147483639 // v1 if (-2147483639 < 0) == (v1 < 0) else -(--2147483639 // v1)))):
            while True:
                # ^b0
                v2 = (-1300407837 if (pa0) == (0) else 1301556872)
                v1 = ((_cast_int(1073747998, 64) - -1073747999) + _cast_int(v0, 64))
                # ^b1
                v2 = ((_cast_int(v0, 32) + _cast_int(pa1, 32)) - _cast_int(v1, 32))
                v1 = (_cast_int(16326, 64) - _cast_int(v0, 64))
                if (((pa0 - (-524286 // pa0 if (-524286 < 0) == (pa0 < 0) else -(--524286 // pa0))) - (0 * pa1))) >= (((_cast_int(524287, 20) - (-1 & pa0)) - (-524287 ^ pa0))):
                    # ^b2
                    v2 = ((_cast_int(pa1, 32) - -551647243) - -264371443)
                    v0 = (((_cast_int(v2, 16) - -15143) - -32566) + -9771)
                    v1 = (_cast_int(pa0, 64) - 4517238977076481430)
                    v2 = (((((_cast_int(pa0, 32) + -2084803915) + 1569701865) + 302539934) - -774380226) + -1630638020)
                    if ((((v2 + (1714810848 << v2)) + -213193905) + (85069559 * v2))) >= (((v2 - v2) - (1277168334 if (pa0) >= (227482) else v2))):
                        # ^b3
                        v0 = (_cast_int(v2, 16) - 19976)
                        v2 = (((((_cast_int(pa0, 32) - -1906059922) - 1312907534) - -1141817475) - 1738823502) + 2011160631)
                        v0 = ((_cast_int(pa0, 16) - -16435) - 261)
                        v1 = ((((_cast_int(v0, 64) + 5885115908715253784) - 5825851685144504661) + 549766622887299018) - 5896690118441534040)
                        if (((((v1 - -8567497505432704657) + 6840167212111924233) + v1) - -7955833527755308626)) != ((((v1 + (-1109064034639631911 ^ v1)) + (-2819723110373104754 & v1)) - (v1 * v1))):
                            # ^b4
                            v1 = (((((_cast_int(pa0, 64) - 4621200110528184748) + 1466042252710198724) + -5731660007434343443) + -3189424937736278487) + 5184697247205444140)
                            v2 = (((_cast_int(pa1, 32) - 2042850073) + 1657856511) + -1301767476)
                            v0 = (_cast_int(pa1, 16) + -31966)
                            v2 = (((_cast_int(v0, 32) - -425826940) - -758129501) - 1849844542)
                            _brk_exit = True
                            break
                        else:
                            _brk_exit = True
                            break
                else:
                    _cnt_entry = True
                    break
            if _brk_exit:
                _brk_exit = False
                break
            if _brk_exit:
                _brk_exit = False
                break
            if _cnt_entry:
                _cnt_entry = False
                continue
        else:
            break
    # ^exit
    v__chk = 0
    v__chk = (_cast_int(v0, 32) + v__chk)
    v__chk = (_cast_int(v1, 32) + v__chk)
    v__chk = (v__chk + v2)
    v__chk = (_cast_int(pa0, 32) + v__chk)
    v__chk = (_cast_int(pa1, 32) + v__chk)
    return v__chk

"""
MULTI_LEVEL_EDGES = [
  ("b0", "b1"),
  ("b1", "b2"),
  ("b1", "entry"),
  ("b2", "b0"),
  ("b2", "b3"),
  ("b3", "b4"),
  ("b3", "exit"),
  ("b4", "exit"),
  ("entry", "b0"),
  ("entry", "exit"),
]

GO_CHAIN_SRC = """\
def func_277737_0(pa0, pa1):
    _go_b4 = False
    v0 = 1390183470
    v1 = -3376710
    a0 = [25, _PAD, _PAD, _PAD, _PAD, _PAD, _PAD, _PAD]
    v__chk = 0
    while True:
        # ^entry
        a0[0] = (_cast_int(2147483647, 64) - _cast_int(v0, 64))
        v0 = _cast_int(pa1, 32)
        if (((_cast_int(-1048575, 24) + 4194304) - (~v1))) >= ((_cast_int(-230978, 24) + (3376709 ^ v1))):
            # ^b0
            v0 = ((-1074790383 + 1083178992) + _cast_int(v1, 32))
            v1 = (_cast_int(4194303, 24) - (7832113 if (pa1) == (0) else -64))
            if (((_cast_int(-1, 16) + 0) - (-1 * pa0))) == (((pa0 - (~pa0)) + 32766)):
                # ^b1
                v0 = ((_cast_int(pa1, 32) - -1000007290) - 180518944)
                a0[0] = (_cast_int(pa0, 64) + 4779544505478012505)
                a0[0] = (((_cast_int(v0, 64) + 5899122092922097481) - -1804417916858868272) + -6012549381708219711)
                a0[0] = (((((_cast_int(pa0, 64) + -4271742685527139969) + -5479604327715065043) - -5494721611445278071) + -8742055666264067646) + -3668418034450071883)
                if not (((((pa0 - 16650) - -10913) - (pa0 * pa0))) >= ((((((pa0 - (32658 if (pa0) < (27043) else 12537)) - (-29464 | pa0)) - -8312) + (30364 * pa0)) - 5588))):
                    _go_b4 = True
            else:
                _go_b4 = True
        if not _go_b4 and not _go_b4:
            # ^b2
            a0[0] = ((_cast_int(-1, 64) + _cast_int(pa1, 64)) + _cast_int(pa1, 64))
            a0[0] = ((_cast_int(0, 64) - -1688389) + _cast_int(v1, 64))
            # ^b3
            v1 = ((_cast_int(-3145728, 24) - _cast_int(pa0, 24)) + -5242880)
            v0 = (8388608 + _cast_int(v1, 32))
        _go_b4 = False
        _go_b4 = False
        # ^b4
        v1 = ((_cast_int(-4194306, 24) + 0) - (-1 if (pa1) >= (0) else -1))
        a0[0] = (_cast_int(4194304, 64) - _cast_int(v0, 64))
        if (((~v0) - (-2 * v0))) < (((-2145386496 | v0) + (2141192193 ^ v0))):
            # ^exit
            v__chk = 0
            v__chk = (v__chk + v0)
            v__chk = (_cast_int(v1, 32) + v__chk)
            v__chk = (_cast_int(_rd(a0, 0), 32) + v__chk)
            v__chk = (_cast_int(pa0, 32) + v__chk)
            v__chk = (_cast_int(pa1, 32) + v__chk)
            return v__chk

"""
GO_CHAIN_EDGES = [
  ("b0", "b1"),
  ("b0", "b4"),
  ("b1", "b2"),
  ("b1", "b4"),
  ("b2", "b3"),
  ("b3", "b4"),
  ("b4", "entry"),
  ("b4", "exit"),
  ("entry", "b0"),
  ("entry", "b2"),
]

ROTATE_CONDLOOP_SRC = """\
def func_493ad2_0(pa0, pa1):
    v0 = -115
    v1 = -6162980551422932163
    v2 = -623431742
    v__chk = 0
    # ^entry
    v2 = (-1572864 + _cast_int(pa0, 32))
    v2 = ((-134217615 - 0) + _cast_int(v0, 32))
    while (((_cast_int(8, 64) - -1073741823) - (2147483645 if (v0) <= (0) else 0))) < ((_cast_int(-1073741813, 64) + (-2147483631 // v1 if (-2147483631 < 0) == (v1 < 0) else -(--2147483631 // v1)))):
        # ^b0
        v2 = (-1300407837 if (pa0) == (0) else 1301556872)
        v1 = ((_cast_int(1073741825, 64) - 1073741708) + _cast_int(v0, 64))
        if ((pa1 + (0 * pa0))) == (((pa1 + 262028) - (-109 ^ pa1))):
            # ^b1
            v1 = (((((_cast_int(v0, 64) - 3618436768347745732) + -1063236922891631481) + -118010701503943815) - -6025136141538511466) + 8406188734100277877)
            v2 = (((_cast_int(v0, 32) - 1756591453) - 1611936888) - -551647243)
            v0 = (((_cast_int(v2, 16) - -15143) - -32566) + -9771)
            v1 = (_cast_int(pa0, 64) - 4517238977076481430)
            # ^b2
            v2 = (((((_cast_int(pa0, 32) + -2084803915) + 1569701865) + 302539934) - -774380226) + -1630638020)
            v1 = (((_cast_int(pa0, 64) + 8822109915019866080) + -8223997732859483313) + -2176851898327607292)
            v1 = (((_cast_int(pa0, 64) - 7796409652611158611) - 9103323340232698618) + -4386925660137850002)
            v0 = (((_cast_int(pa0, 16) - -24711) + -16731) - 635)
            # ^b3
            v1 = (((((_cast_int(v0, 64) + -4309271748649342592) - 530665322524323784) - 7945833385774970504) - 5878441208365549374) - 8623276992307221721)
            v2 = (((_cast_int(pa0, 32) - -1743428950) - -2131161064) + 1356436797)
            v0 = ((((_cast_int(pa1, 16) - -18197) - 20067) + 18539) - 32348)
            v2 = ((((_cast_int(pa1, 32) + 926130707) - -1707575981) - -214359569) + -96858234)
            break
        # ^entry
        v2 = (-1572864 + _cast_int(pa0, 32))
        v2 = ((-134217615 - 0) + _cast_int(v0, 32))
    # ^exit
    v__chk = 0
    v__chk = (_cast_int(v0, 32) + v__chk)
    v__chk = (_cast_int(v1, 32) + v__chk)
    v__chk = (v__chk + v2)
    v__chk = (_cast_int(pa0, 32) + v__chk)
    v__chk = (_cast_int(pa1, 32) + v__chk)
    return v__chk

"""
ROTATE_CONDLOOP_EDGES = [
  ("b0", "b1"),
  ("b0", "entry"),
  ("b1", "b2"),
  ("b2", "b3"),
  ("b3", "exit"),
  ("entry", "b0"),
  ("entry", "exit"),
]


# SIR-projected CFG edges for each lowered shape above: what the fixed
# flag-aware extractor must reconstruct (multi-level continue, jump-join
# chain, rotated/condition loop).

SELF_LOOP_SRC = (
  "def func_t(pa):\n"
  "    _brk_exit = False\n"
  "    _cnt_entry = False\n"
  "    # ^entry\n"
  "    v = (pa + 1)\n"
  "    while True:\n"
  "        # ^b0\n"
  "        v = (v + 2)\n"
  "        if (v > 100):\n"
  "            # ^exit\n"
  "            v = (v + 3)\n"
  "            return v\n"
  "        _cnt_entry = False\n"
)
SELF_LOOP_EDGES = [("entry", "b0"), ("b0", "exit"), ("b0", "b0")]


def sir_extractor_tests_pass(ccommon, chk_mod) -> bool:
  """CFG_EDGE/EXEC_PATH are SIR-projected: the extractor (used at puzzle
  time as an exact-equality gate and by the checker) must reconstruct the
  lowered shapes' true SIR edges - never the mechanical break edges."""
  ok = True

  cases = [
    ("multi-level continue out two levels", MULTI_LEVEL_SRC, MULTI_LEVEL_EDGES),
    ("jump-join chain", GO_CHAIN_SRC, GO_CHAIN_EDGES),
    ("rotated/condition loop", ROTATE_CONDLOOP_SRC, ROTATE_CONDLOOP_EDGES),
    ("self-loop edge kept", SELF_LOOP_SRC, SELF_LOOP_EDGES),
  ]
  for name, src_text, want_edges in cases:
    src = src_text.encode("utf-8")
    tree = ast.parse(src)
    leaf, _ = ccommon.find_python_leaf_function(tree, src)
    got = sorted(ccommon.build_python_cfg(leaf, src))
    good = set(got) == set(want_edges)
    if not good:
      missing = sorted(set(want_edges) - set(got))
      extra = sorted(set(got) - set(want_edges))
      check(
        f"unit: extractor reconstructs {name}",
        False,
        f"missing={missing} extra={extra}",
      )
      ok = False
    else:
      check(f"unit: extractor reconstructs {name}", True)

  # A fill whose flag target renames into a *declared* but wrong block
  # must fail FAIL_CFG: the extracted CFG re-derives every flag's edge
  # from its SetFlag site per solution.
  declared = MULTI_LEVEL_EDGES
  mutated = MULTI_LEVEL_SRC.replace("_cnt_entry", "_cnt_b0")
  go_mutated = GO_CHAIN_SRC.replace("_go_b4", "_go_b3")
  for name, src_text in (
    ("unrenamed", MULTI_LEVEL_SRC),
    ("_cnt_ target renamed into a declared block", mutated),
    ("_go_ target renamed into a declared block", go_mutated),
  ):
    src = src_text.encode("utf-8")
    leaf, _ = ccommon.find_python_leaf_function(ast.parse(src), src)
    try:
      chk_mod.check_cfg(leaf, src, declared)
      check(
        f"unit: check_cfg {name}",
        name == "unrenamed",
        "" if name == "unrenamed" else "no CFG failure raised",
      )
      ok = ok and name == "unrenamed"
    except chk_mod.CheckFailure as exc:
      wrong = name == "unrenamed" or exc.result != chk_mod.CheckResult.FAIL_CFG
      check(f"unit: check_cfg {name}", not wrong, str(exc.result))
      ok = ok and not wrong
    except Exception as exc:  # noqa: BLE001
      check(f"unit: check_cfg {name}", False, str(exc))
      ok = False

  return ok


def checker_unit_tests_pass(chk_mod) -> bool:
  ok = True

  # A solution that hangs must fail with FAIL_TIMEOUT (5s cap).
  hang_dir = tempfile.mkdtemp(prefix="codoku_hang_")
  try:
    hang_path = os.path.join(hang_dir, "hang.py")
    with open(hang_path, "w") as f:
      f.write("while True:\n    pass\n")
    try:
      chk_mod.run_python_solution(hang_path)
      check("unit: hanging solution times out", False, "no timeout raised")
      ok = False
    except chk_mod.CheckFailure as exc:
      check(
        "unit: hanging solution times out",
        exc.result == chk_mod.CheckResult.FAIL_TIMEOUT,
        str(exc.message),
      )
      ok = ok and exc.result == chk_mod.CheckResult.FAIL_TIMEOUT
  finally:
    shutil.rmtree(hang_dir, ignore_errors=True)

  # Flag validation runs even when no CFG edges are declared: a puzzle
  # with no declared edges holds no goto flags, so any flag is unknown.
  flag_src = (
    "def func_t(pa):\n"
    "    _brk_exit = False\n"
    "    # ^entry\n"
    "    _brk_exit = True\n"
    "    return pa\n"
  ).encode("utf-8")
  flag_leaf = ast.parse(flag_src).body[0]
  flag_rejected = False
  flag_cause = None
  try:
    chk_mod.check_cfg(flag_leaf, flag_src, [])
  except chk_mod.CheckFailure as exc:
    flag_rejected = exc.result == chk_mod.CheckResult.FAIL_CFG
    flag_cause = exc.message
  check(
    "unit: flag rejected with empty declared set",
    flag_rejected,
    flag_cause or "no CFG failure raised",
  )
  ok = ok and flag_rejected

  return ok


def disabled_masks_unit_tests(ccommon, mod) -> bool:
  """Test that disabled_masks filters specific mask kinds from puzzle output.

  Goto-flag compound tokens (_go_<FILL_LABEL>, _<FILL_CTRL>_<FILL_LABEL>)
  never match a known mask kind, so disabling any kind preserves them.
  """
  ok = True

  src = (
    "def func_d(x):\n"
    "    # ^entry\n"
    "    y = (x + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        y = helper(y, 2)\n"
    "        if (y > 10):\n"
    "            # ^exit\n"
    "            return y\n"
    "        if _go_exit:\n"
    "            _go_exit = False\n"
    "            break\n"
    "        if _brk_b0:\n"
    "            _brk_b0 = False\n"
    "            break\n"
    "        if _cnt_b0:\n"
    "            _cnt_b0 = False\n"
    "            continue\n"
    "    return y\n"
  ).encode("utf-8")
  tree = ast.parse(src)
  leaf, _ = ccommon.find_python_leaf_function(tree, src)
  if leaf is None:
    check("disabled_masks: fixture has a leaf function", False, "no leaf found")
    return False
  maskable, entry_line, _ = ccommon.get_python_maskable_statements(leaf, src)
  local_names = ccommon.collect_python_leaf_locals(leaf)
  defined_funcs = {"helper"}

  all_repls: list = []
  all_budget: dict = {}
  for stmt in maskable:
    ccommon.collect_python_replacements(
      stmt,
      src,
      stmt.lineno > entry_line,
      all_repls,
      all_budget,
      local_names,
      defined_funcs,
    )
  all_masked = ccommon.apply_replacements(src, all_repls).decode("utf-8")

  baseline_ok = (
    "<FILL_VAR>" in all_masked
    and "<FILL_CONST>" in all_masked
    and "<FILL_OP>" in all_masked
    and "<FILL_FUNC>" in all_masked
    and "<FILL_CTRL>" in all_masked
    and "_<FILL_CTRL>_<FILL_LABEL>" in all_masked
    and "_go_<FILL_LABEL>" in all_masked
  )
  check("disabled_masks: baseline has all mask kinds", baseline_ok, all_masked)
  ok = ok and baseline_ok

  def masked_with(disabled: frozenset[str]) -> str:
    repls: list = []
    for stmt in maskable:
      stmt_repls: list = []
      ccommon.collect_python_replacements(
        stmt,
        src,
        stmt.lineno > entry_line,
        stmt_repls,
        {},
        local_names,
        defined_funcs,
      )
      stmt_repls = mod.filter_disabled_masks(stmt_repls, disabled)
      repls.extend(stmt_repls)
    return ccommon.apply_replacements(src, repls).decode("utf-8")

  # Test 1: Disable <FILL_VAR>.
  m = masked_with(frozenset({"<FILL_VAR>"}))
  t1 = "<FILL_VAR>" not in m and "<FILL_CONST>" in m and "<FILL_OP>" in m
  check("disabled_masks: <FILL_VAR> suppressed", t1, m)
  ok = ok and t1

  # Test 2: Disable <FILL_CONST>.
  m = masked_with(frozenset({"<FILL_CONST>"}))
  t2 = "<FILL_CONST>" not in m and "<FILL_VAR>" in m and "<FILL_OP>" in m
  check("disabled_masks: <FILL_CONST> suppressed", t2, m)
  ok = ok and t2

  # Test 3: Disable <FILL_CTRL> -- standalone suppressed, compound preserved.
  m = masked_with(frozenset({"<FILL_CTRL>"}))
  standalone = m.replace("_<FILL_CTRL>_<FILL_LABEL>", "")
  t3 = (
    "<FILL_CTRL>" not in standalone
    and "_<FILL_CTRL>_<FILL_LABEL>" in m
    and "<FILL_VAR>" in m
  )
  check("disabled_masks: <FILL_CTRL> suppressed, compound preserved", t3, m)
  ok = ok and t3

  # Test 4: Disable <FILL_OP>.
  m = masked_with(frozenset({"<FILL_OP>"}))
  t4 = "<FILL_OP>" not in m and "<FILL_VAR>" in m and "<FILL_CONST>" in m
  check("disabled_masks: <FILL_OP> suppressed", t4, m)
  ok = ok and t4

  # Test 5: Disable all five normal kinds -- only goto-flag compounds survive.
  all_five = frozenset(
    {"<FILL_VAR>", "<FILL_CONST>", "<FILL_OP>", "<FILL_FUNC>", "<FILL_CTRL>"}
  )
  m = masked_with(all_five)
  t5 = (
    "<FILL_VAR>" not in m
    and "<FILL_CONST>" not in m
    and "<FILL_OP>" not in m
    and "<FILL_FUNC>" not in m
    and "<FILL_CTRL>" not in m.replace("_<FILL_CTRL>_<FILL_LABEL>", "")
    and "_<FILL_CTRL>_<FILL_LABEL>" in m
    and "_go_<FILL_LABEL>" in m
  )
  check("disabled_masks: all five suppressed, goto flags survive", t5, m)
  ok = ok and t5

  # Test 6 (edge case): Profile validation rejects unknown and
  # non-disableable mask kinds: kind names are pinned so the vocabulary
  # cannot silently change.
  for kind in (
    "<FILL_BOGUS>",
    "<FILL_TYPE>",
    "<FILL_LABEL>",
    "<FILL_CTRL>",
    "<FILL_FIELD>",
  ):
    try:
      bad = mod.GenerationProfile(
        n_bbls=mod.IntRange(2, 4),
        n_stmts=mod.IntRange(2, 3),
        min_loop_iter=mod.IntRange(0, 1),
        p_mask=mod.FloatRange(0.5, 0.7),
        max_ptr_depth=mod.IntRange(0, 0),
        p_backedge=mod.FloatRange(0.1, 0.3),
        p_branch=mod.FloatRange(0.3, 0.5),
        n_vars=mod.IntRange(6, 10),
        n_params=mod.IntRange(2, 3),
        lift_consts=False,
        features=("--no-fp", "--no-vec", "--no-ptrarith", "--no-intrinsics"),
        disabled_masks=frozenset({kind}),
      )
      bad.validate("bad")
      check(f"disabled_masks: {kind} rejected", False, "no ValueError")
      ok = False
    except ValueError:
      check(f"disabled_masks: {kind} rejected", True)
    except Exception as exc:
      check(f"disabled_masks: {kind} rejected", False, str(exc))
      ok = False

  return ok


def trusted_layout_unit_tests_pass(ccommon) -> bool:
  """_Ptr geometry args (off/stride/lo/hi) and _cast_int widths come from
  the frontend layout and IR types, not the answer space: both stay
  visible and never enter the constant budget. The _Ptr buffer and the
  _cast_int value mask normally."""
  ok = True
  src = (
    "def func_t(v1):\n"
    "    # ^entry\n"
    "    y = (v1 + 1)\n"
    "    while True:\n"
    "        # ^b0\n"
    "        p0 = _Ptr(v1, 0, 2, 0, 2, _frame)\n"
    "        v2 = _cast_int(v1 + 5, 32)\n"
    "        break\n"
    "    # ^exit\n"
    "    return y\n"
  ).encode("utf-8")
  tree = ast.parse(src)
  leaf, _ = ccommon.find_python_leaf_function(tree, src)
  maskable, entry_line, _ = ccommon.get_python_maskable_statements(leaf, src)
  local_names = ccommon.collect_python_leaf_locals(leaf)
  repls: list = []
  budget: dict = {}
  for stmt in maskable:
    ccommon.collect_python_replacements(
      stmt, src, stmt.lineno > entry_line, repls, budget, local_names, set()
    )
  masked = ccommon.apply_replacements(src, repls).decode("utf-8")

  good = "_Ptr(<FILL_VAR>, 0, 2, 0, 2, _frame)" in masked
  check("trusted: _Ptr geometry stays visible", good, masked)
  ok = ok and good

  good = "_cast_int(<FILL_VAR> <FILL_OP> <FILL_CONST>, 32)" in masked
  check("trusted: _cast_int width stays visible", good, masked)
  ok = ok and good

  good = budget == {"5": 1}
  check("trusted: geometry args never enter the budget", good, str(budget))
  ok = ok and good

  return ok


def budget_split_unit_tests_pass(ccommon, chk_mod) -> bool:
  """The budget is a (live, dead) split per value: live slots sit in blocks
  the prescribed run reaches (plus pre-entry declarations, which always
  run), dead slots off it. Totals that add up but park a constant across
  the boundary fail."""
  ok = True
  src = (
    "def func_t(pa):\n"
    "    _brk_exit = False\n"
    "    v1 = 7\n"
    "    while True:\n"
    "        # ^entry\n"
    "        if (pa > 0):\n"
    "            # ^b0\n"
    "            v1 = 5\n"
    "            _brk_exit = True\n"
    "            break\n"
    "        else:\n"
    "            break\n"
    "    # ^exit\n"
    "    return v1\n"
  ).encode("utf-8")
  tree = ast.parse(src)
  leaf, _ = ccommon.find_python_leaf_function(tree, src)
  maskable, entry_line, _ = ccommon.get_python_maskable_statements(leaf, src)
  local_names = ccommon.collect_python_leaf_locals(leaf)
  comments = ccommon.find_python_block_comments(src)
  live_blocks = frozenset({"entry", "exit"})

  repls: list = []
  live_counts: dict = {}
  dead_counts: dict = {}
  for stmt in maskable:
    stmt_budget: dict = {}
    ccommon.collect_python_replacements(
      stmt, src, stmt.lineno > entry_line, repls, stmt_budget, local_names, set()
    )
    slot = (
      live_counts
      if ccommon.stmt_is_on_live_path(comments, stmt.lineno, live_blocks)
      else dead_counts
    )
    for val, cnt in stmt_budget.items():
      slot[val] = slot.get(val, 0) + cnt
  split = ccommon.merge_const_split(live_counts, dead_counts)

  good = (
    split == {"7": (1, 0), "5": (0, 1)}
    and live_counts == {"7": 1}
    and dead_counts == {"5": 1}
  )
  check("split: pre-entry slots live, off-path slots dead", good, str(split))
  ok = ok and good

  banner = (
    "#//@ CFG_EDGE: entry -> b0\n"
    "#//@ CFG_EDGE: b0 -> exit\n"
    "#//@ EXEC_PATH: entry -> exit\n"
    "#//@ <FILL_CONST>: 7 1 0\n"
    "#//@ <FILL_CONST>: 5 0 1\n"
  )
  puzzle = banner + ccommon.apply_replacements(src, repls).decode("utf-8")
  mask_set = chk_mod.infer_mask_set_from_puzzle(leaf, src, puzzle, set())
  good = mask_set is not None
  check("split: mask set infers from the split puzzle", good, str(mask_set))
  ok = ok and good
  if mask_set is None:
    return ok

  actual = chk_mod.check_remasking(
    leaf, src, puzzle, mask_set, set(), frozenset(live_blocks)
  )
  check("split: re-masking counts the true split", actual == split, str(actual))

  req = chk_mod.parse_puzzle_requirements(puzzle)
  truth_ok = True
  detail = ""
  try:
    chk_mod.check_fill_const_budget(actual, req.const_budget)
  except chk_mod.CheckFailure as exc:
    truth_ok = False
    detail = str(exc)
  check("split: the true split passes", truth_ok, detail)
  ok = ok and truth_ok

  parked_ok = False
  try:
    chk_mod.check_fill_const_budget(actual, {"7": (0, 1), "5": (1, 0)})
  except chk_mod.CheckFailure as exc:
    parked_ok = exc.result == chk_mod.CheckResult.FAIL_FILL_CONST
  except Exception as exc:  # noqa: BLE001
    check("split: parked across the boundary fails on equal totals", False, str(exc))
    return ok
  check("split: parked across the boundary fails on equal totals", parked_ok)
  ok = ok and parked_ok

  legacy_ok = False
  try:
    chk_mod.parse_puzzle_requirements("#//@ <FILL_CONST>: 7 1\n")
  except chk_mod.CheckFailure as exc:
    legacy_ok = exc.result == chk_mod.CheckResult.FAIL_PARSE
  check("split: legacy 2-token lines fail closed at parse", legacy_ok)
  ok = ok and legacy_ok

  return ok


def sentinel_unit_tests_pass(ccommon, chk_mod) -> bool:
  """0, 1, 0.0, and 1.0 are structural sentinels (counters, identity
  elements): they stay visible in every zone, never become <FILL_CONST>
  cells, and never enter the budget, so filling a mark with one fails
  re-masking."""
  ok = True
  src = (
    "def func_t(pa):\n"
    "    v1 = 0\n"
    "    v2 = 1\n"
    "    v3 = 7\n"
    "    # ^entry\n"
    "    v4 = pa\n"
    "    # ^b0\n"
    "    v5 = v3 * 0\n"
    "    v6 = v3 + 1\n"
    "    v7 = v3 + 1.0\n"
    "    v8 = v3 * 0.0\n"
    "    v9 = v3 - 2\n"
    "    # ^exit\n"
    "    return v4\n"
  ).encode("utf-8")
  tree = ast.parse(src)
  leaf, _ = ccommon.find_python_leaf_function(tree, src)
  maskable, entry_line, _ = ccommon.get_python_maskable_statements(leaf, src)
  local_names = ccommon.collect_python_leaf_locals(leaf)
  repls: list = []
  budget: dict = {}
  for stmt in maskable:
    ccommon.collect_python_replacements(
      stmt, src, stmt.lineno > entry_line, repls, budget, local_names, set()
    )
  masked = ccommon.apply_replacements(src, repls).decode("utf-8")

  good = "<FILL_OP> 0\n" in masked
  check("sentinel: body 0 stays visible", good, masked)
  ok = ok and good

  good = "<FILL_OP> 1\n" in masked
  check("sentinel: body 1 stays visible", good, masked)
  ok = ok and good

  good = "<FILL_OP> 1.0\n" in masked and "<FILL_OP> 0.0\n" in masked
  check("sentinel: float sentinels stay visible", good, masked)
  ok = ok and good

  good = budget == {"7": 1, "2": 1}
  check("sentinel: 0/1 never enter the budget", good, str(budget))
  ok = ok and good

  banner = (
    "#//@ CFG_EDGE: entry -> b0\n"
    "#//@ CFG_EDGE: b0 -> exit\n"
    "#//@ EXEC_PATH: entry -> b0 -> exit\n"
    "#//@ <FILL_CONST>: 7 1 0\n"
    "#//@ <FILL_CONST>: 2 1 0\n"
  )
  puzzle = banner + masked
  filled_zero = (banner + src.decode("utf-8")).replace("v9 = v3 - 2", "v9 = v3 - 0")
  filled_tree = ast.parse(filled_zero)
  filled_leaf, _ = ccommon.find_python_leaf_function(filled_tree, filled_zero)
  zero_mask_set = chk_mod.infer_mask_set_from_puzzle(
    filled_leaf, filled_zero.encode("utf-8"), puzzle, set()
  )
  good = zero_mask_set is None
  check("sentinel: filling a mark with 0 fails re-masking", good, str(zero_mask_set))
  ok = ok and good

  # Declarations keep their sentinels verbatim even when they sit in the
  # mask set, and the end-to-end re-masked split matches the budget: only
  # non-sentinel values appear, at their true live and dead counts.
  good = "v1 = 0" in masked and "v2 = 1" in masked and "v3 = <FILL_CONST>" in masked
  check("sentinel: declaration sentinels remain visible", good, masked)
  ok = ok and good

  mask_set = chk_mod.infer_mask_set_from_puzzle(leaf, src, puzzle, set())
  split_ok = False
  detail = ""
  if mask_set is not None:
    try:
      split = chk_mod.check_remasking(
        leaf, src, puzzle, mask_set, set(), frozenset({"entry", "b0", "exit"})
      )
      split_ok = split == {"7": (1, 0), "2": (1, 0)}
      detail = str(split)
    except chk_mod.CheckFailure as exc:
      detail = str(exc)
  check("sentinel: end-to-end split excludes 0/1", split_ok, detail)
  ok = ok and split_ok

  return ok


def main():
  if len(sys.argv) < 3:
    print("usage: run_codoku_tests.py <codoku.py> <rysmith>")
    return 2
  codoku_src, rysmith = sys.argv[1:3]

  mod = import_codoku(codoku_src)
  chk_mod = import_codoku_checker(codoku_src)
  unit_tests_pass(mod)
  checker_unit_tests_pass(chk_mod)
  ccommon = import_codoku_common(codoku_src)
  goto_flag_unit_tests_pass(ccommon, chk_mod)
  sir_extractor_tests_pass(ccommon, chk_mod)
  complexity_unit_tests_pass(import_codoku_complexity(codoku_src))
  disabled_masks_unit_tests(ccommon, mod)
  trusted_layout_unit_tests_pass(ccommon)
  budget_split_unit_tests_pass(ccommon, chk_mod)
  sentinel_unit_tests_pass(ccommon, chk_mod)

  with tempfile.TemporaryDirectory(prefix="codoku_gen_") as workdir:
    # Mirror the image layout: codoku + vendored modules + rysmith in one dir.
    src_dir = os.path.dirname(os.path.realpath(codoku_src))
    for f in (
      "codoku.py",
      "codoku_creator.py",
      "codoku_checker.py",
      "codoku_common.py",
      "codoku_preamble.py",
      "codoku_complexity.py",
    ):
      shutil.copy(os.path.join(src_dir, f), os.path.join(workdir, f))
    os.symlink(os.path.abspath(rysmith), os.path.join(workdir, "rysmith"))

    codoku_bin = os.path.join(workdir, "codoku.py")

    # (1) Bare `codoku --seed N` generates the puzzle set.
    r = run_codoku(codoku_bin, ["--seed", "42"], workdir)
    puzzle = os.path.join(workdir, "puzzle.py")
    oracle_gt = os.path.join(workdir, "oracle", "puzzle.gt.py")
    manifest = os.path.join(workdir, "oracle", "metadata.json")
    ok_files = (
      r.returncode == 0
      and os.path.exists(puzzle)
      and os.path.exists(oracle_gt)
      and os.path.exists(os.path.join(workdir, "INSTRUCTION.md"))
      and os.path.exists(manifest)
    )
    check("generate produces puzzle set", ok_files, r.stdout + r.stderr)

    # (2) The puzzle uses <FILL_XXX> tokens and the codoku check command.
    with open(puzzle) as f:
      ptext = f.read()
    check(
      "puzzle uses <FILL_XXX> tokens",
      "<FILL_VAR>" in ptext or "<FILL_CONST>" in ptext or "<FILL_OP>" in ptext,
    )
    check(
      "puzzle banner points at codoku check",
      "codoku check puzzle.py solution.py" in ptext and "./tools/rypuzchk" not in ptext,
    )
    with open(os.path.join(workdir, "INSTRUCTION.md")) as f:
      itext = f.read()
    bare_fill_re = re.compile(r"(?<!<)FILL_[A-Z_]+")
    check(
      "no bare FILL_XXX in puzzle or INSTRUCTION",
      not bare_fill_re.search(ptext) and not bare_fill_re.search(itext),
      bare_fill_re.search(ptext) or bare_fill_re.search(itext),
    )

    # (3) Ground truth passes the vendored checker.
    shutil.copy(oracle_gt, os.path.join(workdir, "solution.py"))
    r = run_codoku(codoku_bin, ["check"], workdir)
    check(
      "ground truth passes check",
      r.returncode == 0 and "[PASS]" in (r.stdout + r.stderr),
      r.stdout + r.stderr,
    )

    # (4) Determinism: same seed reproduces the identical puzzle.
    r2_dir = os.path.join(workdir, "r2")
    os.makedirs(r2_dir)
    for f in (
      "codoku.py",
      "codoku_creator.py",
      "codoku_checker.py",
      "codoku_common.py",
      "codoku_preamble.py",
      "codoku_complexity.py",
    ):
      shutil.copy(os.path.join(workdir, f), os.path.join(r2_dir, f))
    os.symlink(os.path.abspath(rysmith), os.path.join(r2_dir, "rysmith"))
    r2 = run_codoku(
      os.path.join(r2_dir, "codoku.py"), ["create", "--seed", "42"], r2_dir
    )
    with open(puzzle) as f1, open(os.path.join(r2_dir, "puzzle.py")) as f2:
      deterministic = r2.returncode == 0 and f1.read() == f2.read()
    check("same seed is deterministic", deterministic, r2.stdout + r2.stderr)

    # (5) The manifest records realized metrics and complexity estimate.
    with open(manifest) as f:
      meta = json.load(f)
    metrics = meta["realized_metrics"]
    complexity = meta.get("complexity_estimate", {})
    ok_metrics = (
      meta["profile"] == "medium"
      and metrics["exec_path_length"] > 0
      and metrics["cfg_nodes"] > 0
      and metrics["total_masks"] > 0
      and metrics["hal_volume"] > 0
      and metrics["dep_nodes"] > 0
      and "static_struct" in complexity
      and "vocab" not in complexity
      and "dataflow" not in complexity
      and "size" not in complexity
      and "static_structure" not in complexity
    )
    check("manifest records realized metrics", ok_metrics, str(meta))

    # (6) Invalid profile is rejected.
    r = run_codoku(codoku_bin, ["create", "--profile", "bogus"], workdir)
    check("invalid profile rejected", r.returncode == 2, r.stdout + r.stderr)

    # (7) Creation gates: a candidate whose shape makes the extractor
    # diverge from the SIR CFG (or whose ground-truth trace diverges from
    # the SIR PATH) must be rejected, never emitted as a puzzle.
    from pathlib import Path

    cfg = mod.GeneratorConfig(
      n_bbls=5,
      n_stmts=2,
      min_loop_iter=1,
      p_mask=1.0,
      max_ptr_depth=0,
      p_backedge=0.4,
      p_branch=0.5,
      n_vars=6,
      n_params=2,
      lift_consts=False,
      features=("--no-fp", "--no-vec", "--no-ptrarith", "--no-intrinsics"),
    )
    gate_dir = tempfile.mkdtemp(prefix="codoku_gates_")
    try:
      candidate = None
      error = None
      cand_dir = Path(gate_dir) / "cand"
      cand_dir.mkdir(parents=True, exist_ok=True)
      try:
        candidate = mod.generate_candidate(
          cand_dir, cfg, 31, rysmith_path=Path(rysmith).resolve()
        )
      except Exception as exc:  # noqa: BLE001
        error = exc
      if candidate is None:
        check("creation: seed-shape candidate created", False, str(error))
      else:
        check("creation: seed-shape candidate created", True)

        # The banner must reconstruct the trusted companion .sir banner:
        # CFG edges == the .sir CFG, and the path == the `// PATH:` walk.
        sirs = sorted(cand_dir.glob("func_*.sir"))
        sir_cfg = mod.extract_cfg_from_sir(sirs[-1])
        sir_path_str = mod.extract_path_from_sir(sirs[-1])

        ptext = (cand_dir / "puzzle.py").read_text()
        exp_path, _, declared, _ = chk_mod.parse_puzzle_requirements(ptext)
        path_blocks = [b.strip() for b in sir_path_str.split("->") if b.strip()]
        try:
          gt_path = cand_dir / "puzzle.gt.py"
          trace, rc = chk_mod.run_dumps_trace(gt_path, timeout=60)
          trace_ok = rc == 0 and trace == path_blocks
        except RuntimeError:
          trace_ok = False

        good = (
          set(declared) == set(sir_cfg)
          and exp_path == path_blocks
          and all(e in set(sir_cfg) for e in zip(path_blocks, path_blocks[1:]))
          and trace_ok
        )
        check(
          "creation marks carry the SIR CFG + PATH trace",
          good,
          f"cfg_mismatch={set(declared) != set(sir_cfg)} "
          f"path_mismatch={exp_path != path_blocks} "
          f"trace_mismatch={not trace_ok}",
        )
    finally:
      shutil.rmtree(gate_dir, ignore_errors=True)

  n_fail = sum(1 for _, ok, _ in results if not ok)
  print(f"\n{len(results) - n_fail}/{len(results)} codoku tests passed")
  return 1 if n_fail else 0


if __name__ == "__main__":
  sys.exit(main())
