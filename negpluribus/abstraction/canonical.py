"""Suit isomorphism: the one *lossless* abstraction.

Poker does not care which suit is which: ``Ah Kh`` on ``2h 7c 9s`` is exactly
the same situation as ``As Ks`` on ``2s 7c 9h``.  Relabelling suits therefore
merges hands without losing anything, and it shrinks the game a lot:

    distinct (hole, board) combos      after suit isomorphism
    preflop     1 326                   169
    flop       25 989 600           1 286 792
    turn      305 377 800          13 960 050
    river   2 809 475 760         123 156 254

(the board is one sorted set, as in ``canonical_form`` below; Burnside count over the
24 suit permutations.  Keys that keep the flop, turn and river cards apart, as in
Waugh 2013, have 55 190 538 turn and 2 428 287 420 river forms.)

``canonical_form`` tries all 24 suit permutations and returns the
lexicographically smallest (sorted hole, sorted board).  Cheap enough for a
cache key; equity is then computed once per canonical form.
"""
from __future__ import annotations

from itertools import permutations
from typing import Sequence, Tuple

_PERMS = list(permutations(range(4)))


def _relabel(cards: Sequence[int], perm: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted((c >> 2) * 4 + perm[c & 3] for c in cards))


def canonical_form(hole: Sequence[int], board: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    best = None
    for perm in _PERMS:
        cand = (_relabel(hole, perm), _relabel(board, perm))
        if best is None or cand < best:
            best = cand
    return best  # type: ignore[return-value]


def canonical_key(hole: Sequence[int], board: Sequence[int]) -> Tuple[int, ...]:
    h, b = canonical_form(hole, board)
    return h + (-1,) + b


# ---- fast backend hook (negpluribus/fast): same tuples from C++ for the (2 hole, <=5 board)
# case every caller uses; anything else falls through to the reference above.
canonical_form_py, canonical_key_py = canonical_form, canonical_key
try:
    from ..fast import core as _fast_core, fast_eval_enabled as _fast_eval_enabled

    _core = _fast_core() if _fast_eval_enabled() else None
except Exception:  # pragma: no cover
    _core = None
if _core is not None:
    _cf, _ck = _core.canonical_form, _core.canonical_key

    def canonical_form(hole: Sequence[int], board: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:  # noqa: F811
        if len(hole) == 2 and len(board) <= 5:
            return _cf(hole, board)
        return canonical_form_py(hole, board)

    def canonical_key(hole: Sequence[int], board: Sequence[int]) -> Tuple[int, ...]:  # noqa: F811
        if len(hole) == 2 and len(board) <= 5:
            return _ck(hole, board)
        return canonical_key_py(hole, board)
