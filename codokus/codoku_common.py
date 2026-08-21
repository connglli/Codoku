"""codoku_common.py - Python-only masking and analysis helpers for codoku.

Vendored from puzzle/target/puzzle_common.py, reduced to the Python
target's needs: mask tokens are the angle-bracketed <FILL_XXX> forms.
"""

import ast

# ---------------------------------------------------------------------------
# Prefix Stripping
# ---------------------------------------------------------------------------


def strip_refractir_prefix(src_bytes: bytes) -> bytes:
  """Strip the ``refractir_`` name prefix from all identifiers.

  rysmith prefixes every generated identifier with ``refractir_`` (or
  ``_refractir_`` for private helpers).  Stripping them makes the puzzle
  source more concise and readable.

  Ordering matters: ``_refractir_`` must be replaced before ``refractir_``
  so that ``_refractir_foo`` becomes ``_foo`` (not ``__foo``).
  """
  src = src_bytes.replace(b"_refractir_", b"_")
  src = src.replace(b"refractir_", b"")
  return src


# ---------------------------------------------------------------------------
# Byte offsets and text replacement
# ---------------------------------------------------------------------------


def get_byte_offsets(
  src_bytes: bytes,
  lineno: int,
  col_offset: int,
  end_lineno: int,
  end_col_offset: int,
) -> tuple[int, int]:
  """Convert 1-indexed lineno and 0-indexed byte col_offset to absolute byte offsets."""
  lines = src_bytes.split(b"\n")
  start = 0
  for i in range(lineno - 1):
    start += len(lines[i]) + 1
  start += col_offset

  end = 0
  for i in range(end_lineno - 1):
    end += len(lines[i]) + 1
  end += end_col_offset
  return start, end


def apply_replacements(src_bytes: bytes, replacements: list) -> bytes:
  """Apply a list of ``(start, end, text)`` replacements to *src_bytes*.

  Replacements are applied in reverse byte-offset order so earlier offsets
  remain valid after each substitution.  Overlapping ranges are silently
  dropped (the outermost wins, consistent with the parent-first walk order).
  """
  sorted_repls = sorted(set(replacements), key=lambda x: (x[0], x[1]), reverse=True)

  last_start = len(src_bytes)
  valid_repls = []
  for start, end, repl in sorted_repls:
    if end <= last_start:
      valid_repls.append((start, end, repl))
      last_start = start

  res = bytearray(src_bytes)
  for start, end, repl in valid_repls:
    res[start:end] = repl.encode("utf-8")
  return bytes(res)


# ---------------------------------------------------------------------------
# Python block comments
# ---------------------------------------------------------------------------


def find_python_block_comments(src: bytes) -> list[tuple[int, int, str, int]]:
  """Find all comments starting with '# ^' in the source.

  Returns a list of tuples: (start_byte, end_byte, label, line_number)
  """
  comments = []
  lines = src.split(b"\n")
  current_offset = 0
  for idx, line in enumerate(lines, 1):
    line_str = line.decode("utf-8", errors="ignore").strip()
    if line_str.startswith("# ^"):
      label = line_str[3:].strip()
      start_byte = current_offset + line.find(b"#")
      end_byte = current_offset + len(line)
      comments.append((start_byte, end_byte, label, idx))
    current_offset += len(line) + 1
  return comments


def get_line_indent(src_bytes: bytes, start_byte: int) -> str:
  """Get the whitespace indentation of the line containing start_byte."""
  line_start = src_bytes.rfind(b"\n", 0, start_byte)
  if line_start == -1:
    line_start = 0
  else:
    line_start += 1
  indent = []
  for i in range(line_start, start_byte):
    char = src_bytes[i : i + 1]
    if char in (b" ", b"\t"):
      indent.append(char)
    else:
      break
  return b"".join(indent).decode("utf-8")


# ---------------------------------------------------------------------------
# Python leaf-function discovery
# ---------------------------------------------------------------------------


def find_python_leaf_function(
  tree, src: bytes
) -> tuple[ast.FunctionDef | None, str | None]:
  """Locate the non-helper, non-main leaf function in the Python AST module.

  Returns (node, name) or (None, None).
  """
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
      # Exclude main and all private helpers (starting with '_').
      if node.name != "main" and not node.name.startswith("_"):
        return node, node.name
  return None, None


def collect_python_leaf_locals(leaf_node: ast.FunctionDef) -> set[str]:
  """Collect all local variable and parameter names in the leaf function."""
  names = set()
  # Parameters
  for arg in leaf_node.args.args:
    names.add(arg.arg)
  # Local assignments
  for stmt in ast.walk(leaf_node):
    if isinstance(stmt, ast.Assign):
      for target in stmt.targets:
        if isinstance(target, ast.Name):
          names.add(target.id)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
          names.add(target.value.id)
  # Exclude scratch variables starting with '_' or 'v__'
  return {
    name for name in names if not name.startswith("_") and not name.startswith("v__")
  }


# ---------------------------------------------------------------------------
# Python masking
# ---------------------------------------------------------------------------


def get_python_maskable_statements(
  leaf_node: ast.FunctionDef, src: bytes
) -> tuple[list[ast.AST], int, int]:
  """Collect maskable statements and the entry/exit line numbers for the Python leaf.

  Returns (maskable_list, entry_line_number, exit_line_number).
  """
  comments = find_python_block_comments(src)
  comments = [c for c in comments if leaf_node.lineno <= c[3] <= leaf_node.end_lineno]
  comment_map = {}
  for _, _, label, idx in comments:
    if label not in comment_map:
      comment_map[label] = idx

  if "entry" not in comment_map or "exit" not in comment_map:
    return [], 0, 0

  entry_line = comment_map["entry"]
  exit_line = comment_map["exit"]

  decls_before_entry = []
  body_statements = []

  # Top-level declarations/assignments before entry block
  for stmt in leaf_node.body:
    if stmt.lineno < entry_line:
      if isinstance(stmt, ast.Assign):
        is_scratch = False
        for target in stmt.targets:
          if isinstance(target, ast.Name) and (
            target.id.startswith("_") or target.id.startswith("v__")
          ):
            is_scratch = True
        if not is_scratch:
          decls_before_entry.append(stmt)

  def get_block_for_line(lineno):
    current_block = None
    for _, _, label, line in comments:
      if line <= lineno:
        current_block = label
      else:
        break
    return current_block

  def get_loop_header(loop_node):
    for _, _, label, line in comments:
      if loop_node.lineno <= line <= loop_node.end_lineno:
        return label
    return "exit"

  def walk(node):
    if not hasattr(node, "lineno"):
      return
    # Skip trace blocks
    if isinstance(node, ast.If) and any(
      isinstance(n, ast.Constant) and n.value == "DUMP_TRACE"
      for n in ast.walk(node.test)
    ):
      return
    block = get_block_for_line(node.lineno)
    if isinstance(node, (ast.While, ast.For)):
      block = get_loop_header(node)

    # Exclude entry and exit blocks
    if block is not None and block != "entry" and block != "exit":
      if isinstance(node, (ast.Assign, ast.Break, ast.Continue, ast.Expr)):
        body_statements.append(node)
        return
      elif isinstance(node, (ast.If, ast.While)):
        # Note: ast.For is intentionally absent here - rysmith does not
        # generate Python for loops, so its maskable-statement traversal is
        # not modeled.
        body_statements.append(node.test)
        for child in node.body:
          walk(child)
        for child in getattr(node, "orelse", []):
          walk(child)
        return

    for child in ast.iter_child_nodes(node):
      walk(child)

  for stmt in leaf_node.body:
    walk(stmt)

  return decls_before_entry + body_statements, entry_line, exit_line


def collect_python_replacements(
  node: ast.AST,
  src_bytes: bytes,
  is_body: bool,
  replacements: list,
  budget_counts: dict,
  local_names: set[str],
  defined_funcs: set[str],
) -> None:
  """Recursively walk node and append (start, end, <FILL_XXX>) replacement tuples.

  Masking rules (mirrors puzzle_common.hpp's SIRMaskedPrinter):
  - lvalues whose base is in *local_names* → ``<FILL_VAR>``
  - number literals → ``<FILL_CONST>`` (counted in *budget_counts*)
  - break/continue → ``<FILL_CTRL>``
  - function calls:
      * Retained preamble helpers (_cast_int, _padd, _pdiff, _peq, _prel,
        _load, _store, _pidx, _pfield) → ``<FILL_OP>``
      * Calls to functions defined in the same file (*defined_funcs*)
        → ``<FILL_FUNC>`` (excluding internal helpers like _trap, _f32, …)
  - binary operators in BinOp → ``<FILL_OP>``
  - comparison operators → ``<FILL_OP>``
  - unary operators → ``<FILL_OP>``
  - ternary if/else in IfExp → ``<FILL_OP>``

  ``is_body`` is True for statements strictly inside the function body (between
  entry/exit).  For let-initialisers (``is_body=False``) the sentinels ``0``
  and ``1`` are left visible.
  """
  if not hasattr(node, "lineno"):
    return

  def get_node_offsets(n):
    return get_byte_offsets(
      src_bytes, n.lineno, n.col_offset, n.end_lineno, n.end_col_offset
    )

  def get_py_leftmost_base(n):
    if isinstance(n, ast.Subscript):
      return get_py_leftmost_base(n.value)
    return n

  leftmost = get_py_leftmost_base(node)

  if not is_body and isinstance(node, ast.Assign):
    collect_python_replacements(
      node.value,
      src_bytes,
      is_body,
      replacements,
      budget_counts,
      local_names,
      defined_funcs,
    )
    return
  if isinstance(leftmost, ast.Name) and leftmost.id in local_names:
    start, end = get_node_offsets(node)
    replacements.append((start, end, "<FILL_VAR>"))
    return

  if isinstance(node, ast.Constant):
    if isinstance(node.value, (int, float, bool)) or node.value is None:
      if isinstance(node.value, bool) or node.value is None:
        return
      val_str = str(node.value)
      if not is_body and node.value in (0, 1, 0.0, 1.0):
        return
      start, end = get_node_offsets(node)
      replacements.append((start, end, "<FILL_CONST>"))
      budget_counts[val_str] = budget_counts.get(val_str, 0) + 1
      return

  if isinstance(node, (ast.Break, ast.Continue)):
    start, end = get_node_offsets(node)
    replacements.append((start, end, "<FILL_CTRL>"))
    return

  if isinstance(node, ast.Call):
    if isinstance(node.func, ast.Name):
      func_name = node.func.id
      if func_name in (
        "_cast_int",
        "_padd",
        "_pdiff",
        "_peq",
        "_prel",
        "_load",
        "_store",
        "_pidx",
        "_pfield",
      ) or (
        func_name in defined_funcs
        and func_name
        not in (
          "_crc32_update_i32",
          "_check_chksum_i32",
          "_in_check_chksum",
          "_trap",
          "_ichk",
          "_fin",
          "_f32",
          "_f64",
          "_Ptr",
          "_rd",
          "_vrd",
          "_idx",
        )
      ):
        start, end = get_node_offsets(node.func)
        replacements.append((start, end, "<FILL_FUNC>"))
    for arg in node.args:
      collect_python_replacements(
        arg,
        src_bytes,
        is_body,
        replacements,
        budget_counts,
        local_names,
        defined_funcs,
      )
    for keyword in node.keywords:
      collect_python_replacements(
        keyword.value,
        src_bytes,
        is_body,
        replacements,
        budget_counts,
        local_names,
        defined_funcs,
      )
    return

  if isinstance(node, ast.BinOp):
    start_left, end_left = get_node_offsets(node.left)
    start_right, end_right = get_node_offsets(node.right)
    op_slice = src_bytes[end_left:start_right]
    for op_sym in (
      b"**",
      b"<<",
      b">>",
      b"//",  # must precede b"/" - floor-div is two chars
      b"+",
      b"-",
      b"*",
      b"/",
      b"%",
      b"&",
      b"|",
      b"^",
    ):
      idx = op_slice.find(op_sym)
      if idx != -1:
        op_start = end_left + idx
        op_end = op_start + len(op_sym)
        replacements.append((op_start, op_end, "<FILL_OP>"))
        break
    collect_python_replacements(
      node.left,
      src_bytes,
      is_body,
      replacements,
      budget_counts,
      local_names,
      defined_funcs,
    )
    collect_python_replacements(
      node.right,
      src_bytes,
      is_body,
      replacements,
      budget_counts,
      local_names,
      defined_funcs,
    )
    return

  if isinstance(node, ast.Compare):
    prev_end = get_node_offsets(node.left)[1]
    for op, comp in zip(node.ops, node.comparators):
      comp_start, comp_end = get_node_offsets(comp)
      op_slice = src_bytes[prev_end:comp_start]
      for op_sym in (
        b"==",
        b"!=",
        b"<=",
        b">=",
        b"<",
        b">",
        b"is not",
        b"is",
        b"not in",
        b"in",
      ):
        idx = op_slice.find(op_sym)
        if idx != -1:
          op_start = prev_end + idx
          op_end = op_start + len(op_sym)
          replacements.append((op_start, op_end, "<FILL_OP>"))
          break
      prev_end = comp_end
      collect_python_replacements(
        comp,
        src_bytes,
        is_body,
        replacements,
        budget_counts,
        local_names,
        defined_funcs,
      )
    collect_python_replacements(
      node.left,
      src_bytes,
      is_body,
      replacements,
      budget_counts,
      local_names,
      defined_funcs,
    )
    return

  if isinstance(node, ast.UnaryOp):
    op_start, op_end = get_node_offsets(node)
    operand_start, operand_end = get_node_offsets(node.operand)
    op_slice = src_bytes[op_start:operand_start]
    for op_sym in (b"not", b"~", b"+", b"-"):
      idx = op_slice.find(op_sym)
      if idx != -1:
        start = op_start + idx
        end = start + len(op_sym)
        replacements.append((start, end, "<FILL_OP>"))
        break
    collect_python_replacements(
      node.operand,
      src_bytes,
      is_body,
      replacements,
      budget_counts,
      local_names,
      defined_funcs,
    )
    return

  if isinstance(node, ast.IfExp):
    body_start, body_end = get_node_offsets(node.body)
    test_start, test_end = get_node_offsets(node.test)
    orelse_start, orelse_end = get_node_offsets(node.orelse)
    if_slice = src_bytes[body_end:test_start]
    idx_if = if_slice.find(b"if")
    if idx_if != -1:
      replacements.append((body_end + idx_if, body_end + idx_if + 2, "<FILL_OP>"))
    else_slice = src_bytes[test_end:orelse_start]
    idx_else = else_slice.find(b"else")
    if idx_else != -1:
      replacements.append((test_end + idx_else, test_end + idx_else + 4, "<FILL_OP>"))
    collect_python_replacements(
      node.body,
      src_bytes,
      is_body,
      replacements,
      budget_counts,
      local_names,
      defined_funcs,
    )
    collect_python_replacements(
      node.test,
      src_bytes,
      is_body,
      replacements,
      budget_counts,
      local_names,
      defined_funcs,
    )
    collect_python_replacements(
      node.orelse,
      src_bytes,
      is_body,
      replacements,
      budget_counts,
      local_names,
      defined_funcs,
    )
    return

  for child in ast.iter_child_nodes(node):
    collect_python_replacements(
      child,
      src_bytes,
      is_body,
      replacements,
      budget_counts,
      local_names,
      defined_funcs,
    )


# ---------------------------------------------------------------------------
# Python CFG extraction
# ---------------------------------------------------------------------------


def build_python_cfg(leaf_node: ast.FunctionDef, src: bytes) -> set[tuple[str, str]]:
  """Extract the CFG edges directly from the Python leaf function's AST."""
  comments = find_python_block_comments(src)
  comments = [c for c in comments if leaf_node.lineno <= c[3] <= leaf_node.end_lineno]

  def get_loop_header(loop_node):
    for _, _, label, line in comments:
      if loop_node.lineno <= line <= loop_node.end_lineno:
        return label
    return "exit"

  def get_exit_label_after(node):
    for _, _, label, line in comments:
      if line > node.end_lineno:
        return label
    return "exit"

  edges = set()
  processed_comments = set()

  def walk(node, pending, loop_stack):
    if not hasattr(node, "lineno"):
      return pending

    # Process any comments that occur before or at this node's line
    for _, _, label, line in comments:
      if line not in processed_comments and line <= node.lineno:
        processed_comments.add(line)
        for p in pending:
          if p != label:
            edges.add((p, label))
        pending = [label]

    if isinstance(node, ast.Break):
      if loop_stack:
        innermost_loop = loop_stack[-1]
        loop_exit = get_exit_label_after(innermost_loop)
        for p in pending:
          edges.add((p, loop_exit))
      return []
    elif isinstance(node, ast.Continue):
      if loop_stack:
        innermost_loop = loop_stack[-1]
        loop_header = get_loop_header(innermost_loop)
        for p in pending:
          edges.add((p, loop_header))
      return []
    elif isinstance(node, ast.Return):
      for p in pending:
        edges.add((p, "exit"))
      return []
    elif isinstance(node, ast.If):
      pending = walk(node.test, pending, loop_stack)

      then_pending = pending
      for child in node.body:
        then_pending = walk(child, then_pending, loop_stack)

      else_pending = pending
      for child in node.orelse:
        else_pending = walk(child, else_pending, loop_stack)

      return then_pending + else_pending
    elif isinstance(node, (ast.While, ast.For)):
      loop_stack.append(node)
      loop_header = get_loop_header(node)
      loop_exit = get_exit_label_after(node)

      for p in pending:
        if p != loop_header:
          edges.add((p, loop_header))

      body_pending = [loop_header]
      for child in node.body:
        body_pending = walk(child, body_pending, loop_stack)

      for bp in body_pending:
        if bp != loop_header:
          edges.add((bp, loop_header))

      loop_stack.pop()

      if isinstance(node, ast.While):
        for p in pending:
          edges.add((p, loop_exit))
        edges.add((loop_header, loop_exit))
        return [loop_header]
      else:
        # rysmith's Python backend never emits for loops (see
        # get_python_maskable_statements), so this branch is kept in
        # lockstep with puzzle_common.py's for(;;) model: no fall-through.
        return []

    cb = pending
    for child in ast.iter_child_nodes(node):
      cb = walk(child, cb, loop_stack)
    return cb

  cb = ["entry"]
  for stmt in leaf_node.body:
    cb = walk(stmt, cb, [])

  filtered_edges = set()
  for f, t in edges:
    if f and t and f != t:
      filtered_edges.add((f, t))
  return filtered_edges
