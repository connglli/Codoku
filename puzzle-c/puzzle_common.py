"""puzzle_common.py — Shared helpers for the C puzzle toolchain.

This module centralises the AST-traversal and text-manipulation routines that
are used by both rypuzmk.py (puzzle maker) and rypuzchk.py (puzzle checker).
Any change to masking logic must be made here and will automatically apply to
both tools.
"""

# ---------------------------------------------------------------------------
# Prefix Stripping
# ---------------------------------------------------------------------------


def strip_refractir_prefix(src_bytes: bytes) -> bytes:
  """Strip the ``refractir_`` name prefix from all identifiers in the C source.

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
# Helper Functions for C AST Processing
# ---------------------------------------------------------------------------


def clean_c_literal(s: str) -> str:
  """Normalize a C integer/float literal string by stripping type suffixes.

  Examples: ``42u`` → ``42``, ``3.14f`` → ``3.14``, ``100LL`` → ``100``.
  The ``-`` sign for negative literals is a separate unary-operator AST node
  and is never part of the literal token, so no sign handling is needed here.
  """
  s = s.strip()
  # Float suffix must be checked before integer suffixes to avoid 'f' clash.
  if s.lower().endswith("f") and not s.lower().startswith("0x"):
    s = s[:-1]
  else:
    for suffix in ("ull", "ll", "ul", "lu", "l", "u"):
      if s.lower().endswith(suffix):
        s = s[: -len(suffix)]
        break
  return s


def sanitize_statement_expressions(src_bytes: bytes) -> bytes:
  """Replace GNU C statement-expression bodies with a harmless ``0;`` stub.

  GNU C statement expressions ``({ ... })`` contain nested curly braces that
  confuse the tree-sitter C parser, which may prematurely close the enclosing
  function body.  We replace the *interior* with spaces (preserving byte
  offsets for all surrounding tokens) and a minimal ``0;`` placeholder so the
  parser sees a syntactically valid statement expression.
  """
  res = bytearray(src_bytes)
  i = 0
  while i < len(res):
    if res[i : i + 2] == b"({":
      brace_count = 1
      j = i + 2
      while j < len(res) - 1:
        if res[j : j + 2] == b"})" and brace_count == 1:
          res[i + 2 : i + 4] = b"0;"
          for k in range(i + 4, j):
            res[k] = ord(" ")
          i = j + 2
          break
        elif res[j] == ord("{"):
          brace_count += 1
          j += 1
        elif res[j] == ord("}"):
          brace_count -= 1
          j += 1
        else:
          j += 1
      else:
        i += 1
    else:
      i += 1
  return bytes(res)


def get_leftmost_base(node):
  """Return the leftmost node inside nested member/subscript/pointer accesses.

  For example, ``a.b[i]`` yields the identifier node for ``a``.
  """
  if node.type in (
    "member_expression",
    "field_expression",
    "subscript_expression",
    "pointer_expression",
  ):
    arg = node.child_by_field_name("argument")
    if arg is None and node.children:
      arg = node.children[0]
    if arg:
      return get_leftmost_base(arg)
  return node


def get_function_name(node, src: bytes) -> str | None:
  """Return the bare identifier string of a function declarator node."""
  if node.type == "identifier":
    return src[node.start_byte : node.end_byte].decode("utf-8")
  for child in node.children:
    name = get_function_name(child, src)
    if name:
      return name
  return None


def _extract_declared_name(node, src: bytes) -> str | None:
  """Recursively extract the base identifier from a declarator node.

  Handles plain identifiers and nested forms like ``*p``, ``a[2]``,
  ``v = init`` (init_declarator).
  """
  if node.type == "identifier":
    return src[node.start_byte : node.end_byte].decode("utf-8")
  # Recurse through pointer/array/init wrappers; skip punctuation tokens.
  for child in node.children:
    if child.type not in ("*", ",", ";", "=", "[", "]", "(", ")"):
      name = _extract_declared_name(child, src)
      if name:
        return name
  return None


def collect_leaf_locals(leaf_node, src_sanitized: bytes) -> set[str]:
  """Collect all locally-scoped identifier names (parameters + locals) in the leaf.

  These are the identifiers that ``collect_replacements`` should mask as
  ``FILL_VAR`` when they appear as lvalue bases in body statements.

  After the ``refractir_`` prefix has been stripped, we can no longer rely on
  the prefix as a discriminator; instead we build an explicit set from the
  function's parameter list and body declarations.
  """
  names: set[str] = set()

  # --- Parameters ---
  declarator = leaf_node.child_by_field_name("declarator")
  if declarator:
    for child in declarator.children:
      if child.type == "parameter_list":
        for param in child.children:
          if param.type == "parameter_declaration":
            decl_node = param.child_by_field_name("declarator")
            if decl_node:
              name = _extract_declared_name(decl_node, src_sanitized)
              if name:
                names.add(name)

  # --- Local declarations in the function body ---
  body = leaf_node.child_by_field_name("body")
  if body:
    for child in body.children:
      if child.type == "declaration":
        decl_node = child.child_by_field_name("declarator")
        if decl_node:
          name = _extract_declared_name(decl_node, src_sanitized)
          if name:
            names.add(name)

  return names


def find_leaf_function(root_node, src: bytes):
  """Locate the non-helper, non-main leaf function node generated by rysmith.

  After prefix stripping, helper functions start with ``_`` (e.g.
  ``_crc32_update_i32``) and the leaf function starts with ``func_``.

  Returns ``(node, name)`` or ``(None, None)`` if not found.
  """
  for child in root_node.children:
    if child.type == "function_definition":
      decl = child.child_by_field_name("declarator")
      if decl:
        name = get_function_name(decl, src)
        # Exclude main and all private helpers (starting with '_').
        if name and name != "main" and not name.startswith("_"):
          return child, name
  return None, None


def get_maskable_statements(func_node, src: bytes):
  """Collect the ordered list of statements eligible for FILL_XXX masking.

  The maskable set is:
  - Variable declarations before the ``entry`` label (excluding
    ``_``-prefixed scratch variables such as ``_chk``).
  - Statements strictly between the ``entry`` label and the ``exit`` label
    (excluding bare basic-block label statements).

  Returns ``(maskable_list, entry_node, exit_node)``.  All three are ``None``
  / empty when the expected labels are absent.
  """
  body = func_node.child_by_field_name("body")
  if not body:
    return [], None, None

  labeled_stmts = [c for c in body.children if c.type == "labeled_statement"]
  # At least an entry, an exit, and an intermediate label are expected
  if len(labeled_stmts) < 3:
    return [], None, None

  entry_node = labeled_stmts[0]
  first_node = labeled_stmts[1]
  exit_node = labeled_stmts[-1]

  decls_before_entry = []
  body_statements = []

  in_body = False
  for child in body.children:
    if child == first_node:
      in_body = True
      continue
    if child == exit_node:
      in_body = False
      continue

    if child.type == "labeled_statement":
      continue

    if not in_body:
      # Declarations and vector assignments before 'entry' are maskable let-initialisers/vector-initialisers.
      if child.start_byte < entry_node.start_byte:
        if child.type == "declaration":
          declarator = child.child_by_field_name("declarator")
          name = None
          if declarator:
            if declarator.type == "init_declarator":
              name_node = declarator.child_by_field_name("declarator")
              if name_node:
                name = src[name_node.start_byte : name_node.end_byte].decode("utf-8")
            else:
              name = src[declarator.start_byte : declarator.end_byte].decode("utf-8")
          if name and not name.startswith("_"):
            decls_before_entry.append(child)
        elif child.type == "expression_statement":
          if child.children:
            assignment = child.children[0]
            if assignment.type == "assignment_expression":
              lhs = assignment.child_by_field_name("left")
              name = None
              if lhs:
                leftmost = get_leftmost_base(lhs)
                if leftmost.type == "identifier":
                  name = src[leftmost.start_byte : leftmost.end_byte].decode("utf-8")
              if name and not name.startswith("_"):
                decls_before_entry.append(child)
    else:
      # These statements are maskable body statements.
      if child.type in (
        "expression_statement",
        "if_statement",
        "goto_statement",
        "selection_statement",
      ):
        body_statements.append(child)

  return decls_before_entry + body_statements, entry_node, exit_node


def collect_defined_functions(root_node, src: bytes) -> set[str]:
  """Return the set of all function names defined (not just declared) in the file.

  These are the only calls eligible for ``FILL_FUNC`` masking.  Compiler
  builtins (``__builtin_isfinite``, ``__builtin_trap``, …), standard-library
  calls, and any other external symbol will never appear here, so they are
  automatically excluded from masking without needing a hardcoded allowlist.
  """
  names: set[str] = set()
  for child in root_node.children:
    if child.type == "function_definition":
      decl = child.child_by_field_name("declarator")
      if decl:
        name = get_function_name(decl, src)
        if name:
          names.add(name)
  return names


def collect_replacements(
  node,
  src: bytes,
  is_body: bool,
  replacements: list,
  budget_counts: dict,
  local_names: set[str],
  defined_funcs: set[str],
) -> None:
  """Recursively walk *node* and append (start, end, FILL_XXX) replacement tuples.

  Masking rules (mirrors puzzle_common.hpp's SIRMaskedPrinter):
  - lvalues whose base is in *local_names* → ``FILL_VAR``
  - struct field names → ``FILL_FIELD``
  - function calls whose callee is in *defined_funcs* → ``FILL_FUNC``
  - cast target types → ``FILL_TYPE``
  - number literals → ``FILL_CONST`` (counted in *budget_counts*)
  - goto targets → ``FILL_LABEL`` (all targets, including ``entry``/``exit``)
  - operators → ``FILL_OP``

  ``is_body`` is True for statements strictly inside the function body (between
  entry/exit).  For let-initialisers (``is_body=False``) the sentinels ``0``
  and ``1`` are left visible, mirroring the C++ ``printMaskedInitVal`` logic.

  *local_names* is the set of parameter and local variable names in the leaf
  function, collected by ``collect_leaf_locals``.

  *defined_funcs* is the set of function names that have a ``function_definition``
  in the translation unit, collected by ``collect_defined_functions``.  Only
  calls to these are masked; compiler builtins (``__builtin_isfinite``, …) and
  other external symbols are never in this set and thus never masked.
  """
  if not is_body and node.type == "declaration":
    declarator = node.child_by_field_name("declarator")
    if declarator and declarator.type == "init_declarator":
      init_val = declarator.child_by_field_name("value")
      if init_val:
        collect_replacements(
          init_val,
          src,
          is_body,
          replacements,
          budget_counts,
          local_names,
          defined_funcs,
        )
    return

  if not is_body and node.type == "expression_statement":
    if node.children:
      assignment = node.children[0]
      if assignment and assignment.type == "assignment_expression":
        rhs = assignment.child_by_field_name("right")
        if rhs:
          collect_replacements(
            rhs,
            src,
            is_body,
            replacements,
            budget_counts,
            local_names,
            defined_funcs,
          )
        return

  # Whole lvalues rooted in a local/parameter variable mask to FILL_VAR.
  leftmost = get_leftmost_base(node)
  if leftmost.type == "identifier":
    base_name = src[leftmost.start_byte : leftmost.end_byte].decode("utf-8")
    if base_name in local_names:
      replacements.append((node.start_byte, node.end_byte, "FILL_VAR"))
      return

  if node.type in ("member_expression", "field_expression"):
    field_node = node.child_by_field_name("field")
    if field_node:
      replacements.append((field_node.start_byte, field_node.end_byte, "FILL_FIELD"))
      arg = node.child_by_field_name("argument")
      if arg:
        collect_replacements(
          arg,
          src,
          is_body,
          replacements,
          budget_counts,
          local_names,
          defined_funcs,
        )
    return

  if node.type == "call_expression":
    function_node = node.child_by_field_name("function")
    if function_node:
      func_name = src[function_node.start_byte : function_node.end_byte].decode("utf-8")
      # Only mask calls to functions that are actually defined in this .c file.
      # This naturally excludes compiler builtins (__builtin_isfinite, etc.),
      # standard-library calls, and any other external symbol.
      # The structural crc32/check_chksum wrappers are defined in the file but
      # never appear inside maskable body statements, so they are skipped in
      # practice; the explicit exclusion below is kept as a safety net.
      if func_name in defined_funcs and func_name not in (
        "_crc32_update_i32",
        "_check_chksum_i32",
      ):
        replacements.append(
          (function_node.start_byte, function_node.end_byte, "FILL_FUNC")
        )

  if node.type == "cast_expression":
    type_node = node.child_by_field_name("type")
    if type_node:
      replacements.append((type_node.start_byte, type_node.end_byte, "FILL_TYPE"))

  if node.type == "number_literal":
    lit_str = src[node.start_byte : node.end_byte].decode("utf-8")
    clean_lit = clean_c_literal(lit_str)
    replacements.append((node.start_byte, node.end_byte, "FILL_CONST"))
    budget_counts[clean_lit] = budget_counts.get(clean_lit, 0) + 1
    return

  if node.type == "goto_statement":
    for child in node.children:
      if child.type in ("identifier", "statement_identifier"):
        replacements.append((child.start_byte, child.end_byte, "FILL_LABEL"))
        return

  if node.type in (
    "+",
    "-",
    "*",
    "/",
    "%",
    "&",
    "|",
    "^",
    "<<",
    ">>",
    "~",
    "=",
    "<",
    ">",
    "<=",
    ">=",
    "==",
    "!=",
    "?",
    ":",
  ):
    replacements.append((node.start_byte, node.end_byte, "FILL_OP"))
    return

  for child in node.children:
    collect_replacements(
      child,
      src,
      is_body,
      replacements,
      budget_counts,
      local_names,
      defined_funcs,
    )


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
