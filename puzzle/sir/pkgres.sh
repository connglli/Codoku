#!/usr/bin/env bash
#
# pkgres.sh — package puzzle resources into a target directory.
#
# A RefractIR puzzle banner refers to its toolchain and reference material by
# relative path (./symiri, ./rypuzchk, ./references/SPEC.md,
# ./references/examples/, ./references/interp/, ...). This script populates a
# directory with exactly those resources so a solver can work in a
# self-contained sandbox next to the puzzle file.
#
# Layout produced in <target_dir>:
#
#   tools/symiri tools/symirc tools/symirsolve tools/rypuzchk  # RefractIR tools
#   tools/z3 tools/cvc5 tools/bitwuzla         # SMT solvers (from PATH, if present)
#   references/SPEC.md                         # latest language spec
#   references/float.md references/intrinsics.md references/undefined.md
#   references/examples/   -> ./examples       # good examples
#   references/interp/     -> ./test/interp    # good & bad examples (EXPECT-tagged)
#
# Usage:
#   pkgres.sh <target_dir> [--copy|--link]
#
#   --link  (default) create absolute symlinks (cheap; valid on this machine)
#   --copy            copy the resources (self-contained; portable)
#
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") <target_dir> [--copy|--link]" >&2
  exit 2
}

TARGET=""
MODE="link"
for arg in "$@"; do
  case "$arg" in
    --copy) MODE="copy" ;;
    --link) MODE="link" ;;
    -h | --help) usage ;;
    -*) echo "unknown option: $arg" >&2; usage ;;
    *)
      if [ -z "$TARGET" ]; then TARGET="$arg"; else echo "extra argument: $arg" >&2; usage; fi
      ;;
  esac
done
[ -n "$TARGET" ] || usage

# Repo root = parent of this script's directory (the script lives in puzzle/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$TARGET" "$TARGET/tools" "$TARGET/references"

# place <src> <dst>: link or copy src to dst, replacing any existing dst.
place() {
  local src="$1" dst="$2"
  if [ ! -e "$src" ]; then
    echo "  skip (missing): $src" >&2
    return 0
  fi
  rm -rf "$dst"
  if [ "$MODE" = copy ]; then
    cp -aL "$src" "$dst"
  else
    ln -s "$src" "$dst"
  fi
  echo "  $MODE: $dst -> $src"
}

# --- RefractIR tools (built at repo root) ---
for t in symiri symirc symirsolve rypuzchk; do
  place "$ROOT/$t" "$TARGET/tools/$t"
done

# --- SMT solvers (external, resolved from PATH) ---
for t in z3 cvc5 bitwuzla; do
  p="$(command -v "$t" 2>/dev/null || true)"
  if [ -n "$p" ]; then
    place "$p" "$TARGET/tools/$t"
  else
    echo "  skip (not on PATH): $t" >&2
  fi
done

# --- References ---
# Grammar: the highest-versioned spec becomes references/SPEC.md.
SPEC="$(ls "$ROOT"/docs/SPEC_*.md 2>/dev/null | sort -V | tail -1)"
[ -n "$SPEC" ] && place "$SPEC" "$TARGET/references/SPEC.md"
# Supplementary references the banner points at.
place "$ROOT/docs/float.md" "$TARGET/references/float.md"
place "$ROOT/docs/intrinsics.md" "$TARGET/references/intrinsics.md"
place "$ROOT/docs/undefined.md" "$TARGET/references/undefined.md"
place "$ROOT/examples" "$TARGET/references/examples"   # good examples
place "$ROOT/test/interp" "$TARGET/references/interp"  # good & bad examples (EXPECT-tagged)

echo "Packaged resources into '$TARGET' (mode: $MODE)."
