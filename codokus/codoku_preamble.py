"""codoku_preamble.py - the guarded emission preamble codoku splices into
rysmith's generated puzzles.

This file is a direct Python module and should be consistent with
``kPreamble`` in py_backend.cpp: the same value normalization
(_cast_int, _f32) and the same guarded pointer/memory model. The
arithmetic helper guards are removed here, because the no-ub-guards
emission inlines every arithmetic operator at its call sites and
references none of them. A drift from kPreamble makes a generated
puzzle disagree with the guarded emission's semantics.

``_trap`` raises, so a puzzle fill that trips a memory-model guard,
executes unreachable code, or violates an intrinsic UB precondition
(an inert no-op in the no-ub-guards emission this preamble replaces)
traps at run time. The generator produces UB-free programs, so none of
them fires on the ground truth.

swap_preamble in codoku_common.py splices this file's definitions into
each generated puzzle before masking, so the filled-in module keeps the
provenance checks a third-party fill otherwise would not see. The
splice starts at the module's first ``import`` and excludes this
docstring, so puzzles carry the preamble code only.
"""

# Keep the emitted preamble's 4-space layout, faithful to kPreamble.
# fmt: off
import math  # noqa: F401
import struct


class RefractIRTrap(Exception):
    pass


def _trap(msg):
    raise RefractIRTrap(msg)


def _cast_int(v, m):
    v &= (1 << m) - 1
    return v - (1 << m) if v >= 1 << (m - 1) else v


def _f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        _trap("f32 overflow")


def _need_int(v, msg):
    # Reject bool and subclasses with overloaded arithmetic/comparisons.
    if type(v) is not int:
        _trap(msg)


def _chk_geom(buf, off, stride, lo, hi):
    # Validate internal consistency, not the authenticity of a root
    # extent. Root-object metadata must come from the trusted frontend.
    for v in (off, stride, lo, hi):
        _need_int(v, "invalid pointer geometry")

    if buf is None:
        if off != 0 or stride != 0 or lo != 0 or hi != 0:
            _trap("invalid null pointer geometry")
        return

    if type(buf) is not list:
        _trap("invalid pointer buffer")
    if stride < 0:
        _trap("invalid pointer stride")
    if lo < 0 or hi < 0 or off < 0:
        _trap("invalid pointer offset")
    if lo > hi:
        _trap("invalid pointer extent")
    if off < lo or off > hi:
        _trap("pointer out of object bounds")

    # Do not require hi <= len(buf). Compact scalar boxes hold one
    # entry even when their logical extent is multiple bytes, e.g.
    # _Ptr([21], 0, 4, 0, 4).
    # Actual list indexing is checked separately at dereference time.


class _Ptr:
    # A provenance-tracked pointer into a flat leaf-slot list: `off` is
    # the current leaf offset, `stride` the pointee's leaf count, and
    # [lo, hi) the extent of the innermost enclosing object (hi itself
    # is the legal one-past-end position). `frame` is the liveness cell of
    # the activation owning the storage, or None for a pointer that owns
    # none (null).
    #
    # Pointer metadata is read-only; storage and frame remain mutable.
    #
    # Construction from buf is retained for frontend-generated roots.
    # It cannot independently authenticate caller-supplied root bounds.
    __slots__ = ("_state",)

    def __init__(self, buf, off, stride, lo, hi, frame=None):
        # Explicitly calling __init__ again must not replace provenance.
        if hasattr(self, "_state"):
            _trap("pointer metadata is immutable")
        _chk_geom(buf, off, stride, lo, hi)
        object.__setattr__(
            self, "_state", (buf, off, stride, lo, hi, frame)
        )

    def __setattr__(self, name, value):
        _trap("pointer metadata is immutable")

    def __delattr__(self, name):
        _trap("pointer metadata is immutable")

    @property
    def buf(self):
        return self._state[0]

    @property
    def off(self):
        return self._state[1]

    @property
    def stride(self):
        return self._state[2]

    @property
    def lo(self):
        return self._state[3]

    @property
    def hi(self):
        return self._state[4]

    @property
    def frame(self):
        return self._state[5]


_NULL = _Ptr(None, 0, 0, 0, 0)
_UNDEF = ["undef"]  # unique identity sentinel
_PAD = ["pad"]  # interior byte of a wider leaf; never a valid access


def _live(p):
    # SPEC 7.5 rule 27. A returned activation's locals are dead, but
    # their storage may remain referenced by a pointer.
    if type(p) is not _Ptr:
        _trap("invalid pointer")
    if p.frame is not None and not p.frame[0]:
        _trap("access through a pointer to a returned activation")


def _rd(buf, off):
    v = buf[off]
    if v is _UNDEF:
        _trap("read of undef value")
    if v is _PAD:
        _trap("access to the interior of a value")
    return v


def _idx(i, n):
    if i < 0 or i >= n:
        _trap("array index out of bounds")
    return i


def _vrd(buf, off, n, stride):
    return [_rd(buf, off + k * stride) for k in range(n)]


def _chk_ptr(p):
    # Geometry was validated at construction and ordinary attribute
    # mutation is prohibited. Only check the helper's pointer precondition.
    if type(p) is not _Ptr:
        _trap("invalid pointer")
    if p.buf is None:
        _trap("null pointer")


def _chk_deref(p, msg):
    # Call only after _chk_ptr.
    #
    # Zero-sized pointees cannot be loaded or stored through these
    # scalar access helpers. In particular, off == hi must never pass
    # merely because stride == 0.
    if (
        p.stride <= 0
        or p.off < p.lo
        or p.off >= p.hi
        or p.off + p.stride > p.hi
    ):
        _trap(msg)

    # Scalar boxes may have a logical extent larger than len(buf),
    # but the entry actually accessed must exist.
    if p.off >= len(p.buf):
        _trap(msg)


def _derive(p, off, stride, lo, hi, msg):
    # Internal helper: callers have already established that p is a
    # valid non-null pointer. Construction validates child geometry exactly once.
    child = _Ptr(p.buf, off, stride, lo, hi, p.frame)

    # Internal consistency is not enough: derived bounds must also
    # remain within the parent's extent.
    if child.lo < p.lo or child.hi > p.hi:
        _trap(msg)

    return child


def _padd(p, n):
    if type(p) is not _Ptr:
        _trap("invalid pointer arithmetic")
    if p.buf is None:
        _trap("pointer arithmetic on null")
    _need_int(n, "invalid pointer offset")
    _chk_ptr(p)

    off = p.off + n * p.stride
    if off < p.lo or off > p.hi:
        _trap("pointer arithmetic out of object bounds")

    return _derive(p, off, p.stride, p.lo, p.hi, "pointer extent out of parent bounds")


def _pdiff(p, q):
    if type(p) is not _Ptr or type(q) is not _Ptr:
        _trap("invalid pointer subtraction")
    if p.buf is None or q.buf is None or p.buf is not q.buf:
        _trap("cross-object pointer subtraction")

    _chk_ptr(p)
    _chk_ptr(q)
    if p.stride <= 0:
        _trap("invalid pointer stride")

    return (p.off - q.off) // p.stride


def _peq(p, q):
    if type(p) is not _Ptr or type(q) is not _Ptr:
        _trap("invalid pointer comparison")
    return p.buf is q.buf and p.off == q.off


def _prel(p, q):
    if type(p) is not _Ptr or type(q) is not _Ptr:
        _trap("invalid pointer comparison")
    if p.buf is None or q.buf is None or p.buf is not q.buf:
        _trap("relational compare of cross-object pointers")

    _chk_ptr(p)
    _chk_ptr(q)
    return p.off - q.off


def _load(p):
    if type(p) is not _Ptr:
        _trap("invalid pointer dereference")
    if p.buf is None:
        _trap("null pointer dereference")

    _live(p)
    _chk_ptr(p)
    _chk_deref(p, "pointer dereference out of bounds")
    return _rd(p.buf, p.off)


def _store(p, v):
    if type(p) is not _Ptr:
        _trap("invalid pointer store")
    if p.buf is None:
        _trap("null pointer store")

    _live(p)
    _chk_ptr(p)
    _chk_deref(p, "pointer store out of bounds")

    # Do not use _rd here: initializing _UNDEF is valid, but overwriting
    # the interior of an existing wider value is not.
    if p.buf[p.off] is _PAD:
        _trap("access to the interior of a value")

    p.buf[p.off] = v


def _pidx(p, i, n, estride):
    if type(p) is not _Ptr:
        _trap("invalid ptrindex pointer")
    if p.buf is None:
        _trap("ptrindex on null pointer")

    for v in (i, n, estride):
        _need_int(v, "invalid ptrindex argument")

    _live(p)
    _chk_ptr(p)

    if p.off >= p.hi:
        _trap("ptrindex on one-past-end pointer")
    if n < 0 or estride < 0:
        _trap("invalid ptrindex extent")
    if i < 0 or i > n:
        _trap("ptrindex index out of range")

    # i == n constructs a legal one-past-end pointer, not an
    # accessible element. _derive traps a child extent past the parent.
    hi = p.off + n * estride

    return _derive(
        p, p.off + i * estride, estride, p.off, hi,
        "ptrindex extent out of parent bounds"
    )


def _pfield(p, foff, flen, slen):
    # Provenance of a field pointer is the whole containing struct
    # (SPEC 7.5 rule 15): arithmetic may roam across sibling fields.
    if type(p) is not _Ptr:
        _trap("invalid ptrfield pointer")
    if p.buf is None:
        _trap("ptrfield on null pointer")

    for v in (foff, flen, slen):
        _need_int(v, "invalid ptrfield argument")

    _live(p)
    _chk_ptr(p)

    if p.off >= p.hi:
        _trap("ptrfield on one-past-end pointer")
    if foff < 0 or flen < 0 or slen < 0:
        _trap("invalid ptrfield extent")
    if foff + flen > slen:
        _trap("ptrfield field out of struct bounds")

    hi = p.off + slen

    return _derive(
        p, p.off + foff, flen, p.off, hi,
        "ptrfield extent out of parent bounds"
    )
