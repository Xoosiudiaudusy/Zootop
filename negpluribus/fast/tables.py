"""Precomputed bucket tables of the C++ core (csrc/handindex.h, csrc/buckettable.h).

Every postflop bucket is a pure function of the hand's canonical form (the Monte-Carlo runs are
seeded with the form's hash), so it can be computed once for all classes - flop 1,286,792, turn
13,960,050, river 123,156,254, one byte each - and looked up during training.  The numbers are
the ones ``bucket()`` computes; only the time to get them changes.

    python scripts/build_bucket_table.py --buckets data/buckets_<tag>.json   # once per abstraction
    NEGPLURIBUS_BUCKET_TABLES=data/bucket_tables python scripts/train_blueprint.py ... --backend cpp

Files are named after the bucketer's identity (kind + fingerprint of the fitted parameters), so
one directory can hold the tables of many abstractions and a trainer only picks up the table of
its own bucketer.  ``NEGPLURIBUS_BUCKET_TABLES`` unset or empty: no tables (the default).
"""
from __future__ import annotations

import os
import threading
from typing import Dict, Optional, Sequence

from . import core

ENV = "NEGPLURIBUS_BUCKET_TABLES"
STREETS = (1, 2, 3)  # flop, turn, river

_loaded: Dict[str, object] = {}
_lock = threading.Lock()


def table_name(core_bucketer) -> str:
    ident = core_bucketer.identity
    return f"buckets_{ident['kind']}_{ident['n_buckets']}_{ident['fingerprint']:016x}.npbt"


def table_dir(directory: Optional[str] = None) -> Optional[str]:
    d = directory if directory is not None else os.environ.get(ENV, "")
    return d or None


def build_tables(core_bucketer, path: str, streets: Sequence[int] = STREETS, threads: Optional[int] = None,
                 progress=None, every: float = 10.0):
    """Tabulate ``streets`` of a fitted core bucketer and write them to ``path``."""
    from . import default_threads

    t = core().BucketTables()
    for s in streets:
        t.build(core_bucketer, int(s), threads=int(threads or default_threads()),
                progress=(lambda d, n, s=s: progress(s, d, n)) if progress else None, every=every)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    t.save(path)
    return t


def load_tables(path: str, core_bucketer):
    """The tables in ``path``, checked against ``core_bucketer``; loaded once per process."""
    key = os.path.abspath(path)
    with _lock:
        t = _loaded.get(key)
        if t is None:
            t = core().BucketTables()
            t.load(path, core_bucketer)
            _loaded[key] = t
        return t


def tabulated(core_bucketer, directory: Optional[str] = None):
    """``core_bucketer`` wrapped with its tables when ``directory`` (default: $NEGPLURIBUS_BUCKET_TABLES)
    holds a table built for it; otherwise ``core_bucketer`` unchanged."""
    d = table_dir(directory)
    if d is None:
        return core_bucketer
    path = os.path.join(d, table_name(core_bucketer))
    if not os.path.exists(path):
        return core_bucketer
    return core().TabulatedBucketer(core_bucketer, load_tables(path, core_bucketer))


class CoreBuckets:
    """A fitted Python bucketer whose ``bucket()`` answers from its C++ twin (``core_bucketer``), and so
    from the bucket table when ``NEGPLURIBUS_BUCKET_TABLES`` holds one: the same buckets, checked on
    construction on ``check`` random hands per street against the Python bucketer itself.  Everything
    else (``n_buckets``, ``kind``, ...) is the Python bucketer's.  For play and evaluation, where the
    Python bucketer would compute every new hand's features (exact potential-aware: ~40 ms each)."""

    def __init__(self, bucketer, check: int = 20, seed: int = 12345):
        import random

        from .trainer import core_bucketer

        self._py = bucketer
        self._core = core_bucketer(bucketer)
        rng = random.Random(seed)
        for n_board in (0, 3, 4, 5):
            for _ in range(check):
                cards = rng.sample(range(52), 2 + n_board)
                hole, board = cards[:2], cards[2:]
                a, b = bucketer.bucket(hole, board), self._core.bucket(hole, board)
                if a != b:
                    raise RuntimeError(f"C++ bucket {b} != Python bucket {a} for hole {hole} board {board}")

    def bucket(self, hole, board) -> int:
        return self._core.bucket(hole, board)

    def __getattr__(self, name):
        return getattr(self._py, name)
