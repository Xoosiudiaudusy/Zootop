"""Fast backends for the hot loops, selectable by flag; the pure-Python modules stay the
reference implementation and the default.

Three tiers, best available wins unless disabled:

* ``negpluribus._fastcore`` - the C++ core (csrc/, build with ``python scripts/build_fast.py``):
  evaluator, Monte-Carlo equity, engine, abstraction, multithreaded MCCFR.  Bit-identical to
  the Python reference where the reference is deterministic (evaluator values, E[HS] with the
  same seeds, infoset keys, single-thread training).
* ``phevaluator`` (pip) - C perfect-hash 7-card evaluator, wrapped so that it returns the
  *same integers* as ``negpluribus.evaluator.evaluate`` (a 7462-entry translation table).
* pure Python - always there.

Environment flags (read once at import, except the cache caps which are read per trainer):

* ``NEGPLURIBUS_FAST_EVAL`` = ``1`` (default) / ``0``: use compiled evaluator/equity/canonical
  helpers when available.  ``0`` forces the reference code everywhere (useful to reproduce the
  legacy numbers exactly, e.g. old E[HS] caches).
* ``NEGPLURIBUS_BACKEND`` = ``python`` (default) / ``cpp``: default traversal backend of
  ``MCCFRTrainer`` when the ``backend=`` argument is not given.
* ``NEGPLURIBUS_THREADS``: default thread count of the C++ trainer (default: all cores).
* ``NEGPLURIBUS_BUCKET_CACHE``: capacities of the per-street bucket caches of the C++ core,
  ``"flop,turn,river"`` entries with optional K/M/G suffixes (``"4M,32M,4M"``; one value applies
  to all three; ``0`` disables a cache).  Default ``DEFAULT_CACHE_CAPS``.  The caches are bounded
  (docs/backends.md, "Bounded bucket caches"): a cap only changes speed and memory, never the
  numbers.
* ``NEGPLURIBUS_VERIFY_KEYS`` = ``0`` (default) / ``1``: test mode of the C++ trainers (read per
  trainer).  Nodes are found by numeric keys (docs/backends.md, "Numeric node keys"); with ``1``
  every lookup also builds the key string the old way and checks it against the node's, and
  ``train()`` raises on a mismatch.  About as slow as string keys; ``tests/conftest.py`` turns it
  on for the test suite.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

_core = None
_core_error: Optional[str] = None
try:  # pragma: no cover - depends on the build
    from .. import _fastcore as _core  # type: ignore[attr-defined]
except Exception as exc:  # ImportError or a loader error
    _core = None
    _core_error = repr(exc)

# per-street bucket cache capacities (entries) of the C++ bucketers: flop, turn, river.  The
# core rounds each up to a power of two of 8-way sets (4,000,000 -> 4,194,304 slots = 32 MB,
# 32,000,000 -> 33,554,432 slots = 256 MB).  All 1.29M canonical flop forms fit in 4M slots
# and all 13.96M turn forms in 32M (42% load), so a long potential-aware run keeps its turn
# buckets instead of recomputing most of them (4M turn slots: 73-86% of a 50M run's CPU time);
# the river (123M forms) stays a working set.  Each street's table is allocated on its first
# insert, so games that stop earlier never pay for it (preflop-only: nothing, flop games:
# 32 MB); a 4-street game: 320 MB per trainer.  docs/backends.md "Bounded bucket caches" and
# "Speed-ups 2026-09-24".
DEFAULT_CACHE_CAPS: Tuple[int, int, int] = (4_000_000, 32_000_000, 4_000_000)


def core():
    """The compiled ``_fastcore`` module, or ``None`` when it is not built."""
    return _core


def core_available() -> bool:
    return _core is not None


def fast_eval_enabled() -> bool:
    return os.environ.get("NEGPLURIBUS_FAST_EVAL", "1").strip().lower() not in ("0", "false", "no", "off")


def verify_keys_enabled() -> bool:
    """Test mode of the C++ trainers (``NEGPLURIBUS_VERIFY_KEYS``, default off)."""
    return os.environ.get("NEGPLURIBUS_VERIFY_KEYS", "0").strip().lower() in ("1", "true", "yes", "on")


def default_backend() -> str:
    b = os.environ.get("NEGPLURIBUS_BACKEND", "python").strip().lower() or "python"
    if b not in ("python", "cpp"):
        raise ValueError(f"NEGPLURIBUS_BACKEND must be 'python' or 'cpp', got {b!r}")
    return b


def resolve_backend(backend: Optional[str]) -> str:
    b = (backend or default_backend()).lower()
    if b not in ("python", "cpp"):
        raise ValueError(f"backend must be 'python' or 'cpp', got {backend!r}")
    if b == "cpp" and _core is None:
        raise RuntimeError(
            "backend='cpp' requested but negpluribus._fastcore is not built "
            f"({_core_error}); run `python scripts/build_fast.py`"
        )
    return b


def default_threads() -> int:
    v = os.environ.get("NEGPLURIBUS_THREADS")
    if v:
        return max(1, int(v))
    return max(1, os.cpu_count() or 1)


def _parse_cap(text: str) -> int:
    t = text.strip().lower().replace("_", "")
    if not t:
        raise ValueError("empty cache cap")
    mult = 1
    if t[-1] in "kmg":
        mult = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}[t[-1]]
        t = t[:-1]
    v = int(float(t) * mult)
    if v < 0:
        raise ValueError("cache cap must be >= 0")
    return v


def parse_cache_caps(text: str) -> Tuple[int, int, int]:
    """``"4M,4M,4M"`` / ``"2M"`` / ``"0,4000000,1M"`` -> (flop, turn, river) entries."""
    parts = [p for p in text.split(",") if p.strip()]
    if len(parts) == 1:
        c = _parse_cap(parts[0])
        return (c, c, c)
    if len(parts) > 3:
        raise ValueError("NEGPLURIBUS_BUCKET_CACHE: at most three values (flop, turn, river)")
    caps = [_parse_cap(p) for p in parts]
    while len(caps) < 3:
        caps.append(DEFAULT_CACHE_CAPS[len(caps)])
    return (caps[0], caps[1], caps[2])


def default_cache_caps() -> Tuple[int, int, int]:
    v = os.environ.get("NEGPLURIBUS_BUCKET_CACHE")
    if v and v.strip():
        return parse_cache_caps(v)
    return DEFAULT_CACHE_CAPS


def resolve_cache_caps(caps: Optional[Sequence[int] | int | str]) -> Tuple[int, int, int]:
    """``None`` -> env / defaults; an int -> the same cap for all three streets; a string as the
    env var; a sequence -> (flop, turn, river), missing streets take the defaults."""
    if caps is None:
        return default_cache_caps()
    if isinstance(caps, str):
        return parse_cache_caps(caps)
    if isinstance(caps, int):
        return (caps, caps, caps)
    vals = [int(c) for c in caps]
    if len(vals) > 3:
        raise ValueError("cache_caps: at most three values (flop, turn, river)")
    d = default_cache_caps()
    while len(vals) < 3:
        vals.append(d[len(vals)])
    if any(v < 0 for v in vals):
        raise ValueError("cache_caps must be >= 0")
    return (vals[0], vals[1], vals[2])


def describe() -> str:
    from .. import evaluator as _evaluator  # noqa: F401  (installs the evaluator hook)
    from .evaluator import evaluator_tier

    caps = ",".join(str(c) for c in default_cache_caps())
    return (
        f"core={'built' if _core is not None else 'missing'}"
        + (f" ({_core_error})" if _core is None and _core_error else "")
        + f", evaluator={evaluator_tier()}, backend={default_backend()}, threads={default_threads()}, bucket_cache={caps}"
    )
