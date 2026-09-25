"""Pure-Python 5..7 card hand evaluator.

``evaluate(cards)`` returns an int: *higher is better*.  The value encodes the
hand category in the high digits and the five relevant ranks below it, so two
hands compare correctly with plain ``<``/``>``.

Speed is ~100k+ evaluations/s on CPython, enough for the simulation and
evaluation layer.  When we get to abstraction/CFR at scale we can swap in
``phevaluator`` (C perfect-hash) behind the same function signature.
"""
from __future__ import annotations

from typing import Sequence

HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8

CATEGORY_NAMES = [
    "high card",
    "pair",
    "two pair",
    "trips",
    "straight",
    "flush",
    "full house",
    "quads",
    "straight flush",
]

_BASE = 15


def _encode(cat: int, ranks: Sequence[int]) -> int:
    v = cat
    # always pack exactly 5 slots so categories dominate
    padded = list(ranks) + [0] * (5 - len(ranks))
    for r in padded:
        v = v * _BASE + r
    return v


def _straight_high(mask: int) -> int:
    """mask: bit r set if rank r present.  Returns high rank of best straight or -1."""
    m = (mask << 1) | ((mask >> 12) & 1)  # bit i -> rank i-1, bit0 = wheel ace
    for t in range(13, 3, -1):
        if ((m >> (t - 4)) & 0x1F) == 0x1F:
            return t - 1
    return -1


def evaluate(cards: Sequence[int]) -> int:
    n = len(cards)
    if n < 5 or n > 7:
        raise ValueError("evaluate expects 5..7 cards")

    counts = [0] * 13
    suit_counts = [0] * 4
    suit_masks = [0] * 4
    mask = 0
    for c in cards:
        r = c >> 2
        s = c & 3
        counts[r] += 1
        suit_counts[s] += 1
        suit_masks[s] |= 1 << r
        mask |= 1 << r

    # --- flush / straight flush -------------------------------------------
    for s in range(4):
        if suit_counts[s] >= 5:
            sh = _straight_high(suit_masks[s])
            if sh >= 0:
                return _encode(STRAIGHT_FLUSH, [sh])
            fm = suit_masks[s]
            top = []
            for r in range(12, -1, -1):
                if fm >> r & 1:
                    top.append(r)
                    if len(top) == 5:
                        break
            flush_value = _encode(FLUSH, top)
            break
    else:
        flush_value = -1

    # --- rank multiplicity ---------------------------------------------------
    quads = trips = -1
    pairs = []  # descending
    for r in range(12, -1, -1):
        c = counts[r]
        if c == 4:
            quads = r
        elif c == 3:
            if trips < 0:
                trips = r
            else:
                pairs.append(r)  # second trips counts as a pair for full house
        elif c == 2:
            pairs.append(r)

    if quads >= 0:
        kicker = max(r for r in range(13) if counts[r] and r != quads)
        return _encode(QUADS, [quads, kicker])

    if trips >= 0 and pairs:
        return _encode(FULL_HOUSE, [trips, pairs[0]])

    if flush_value >= 0:
        return flush_value

    sh = _straight_high(mask)
    if sh >= 0:
        return _encode(STRAIGHT, [sh])

    if trips >= 0:
        kick = [r for r in range(12, -1, -1) if counts[r] and r != trips][:2]
        return _encode(TRIPS, [trips] + kick)

    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kicker = max(r for r in range(13) if counts[r] and r != p1 and r != p2)
        return _encode(TWO_PAIR, [p1, p2, kicker])

    if len(pairs) == 1:
        p = pairs[0]
        kick = [r for r in range(12, -1, -1) if counts[r] and r != p][:3]
        return _encode(PAIR, [p] + kick)

    top = [r for r in range(12, -1, -1) if counts[r]][:5]
    return _encode(HIGH_CARD, top)


def category(value: int) -> int:
    return value // (_BASE ** 5)


def describe(value: int) -> str:
    return CATEGORY_NAMES[category(value)]


# ---- fast backend hook (negpluribus/fast): same integers, only faster.  ``evaluate_py`` keeps
# the reference implementation reachable; NEGPLURIBUS_FAST_EVAL=0 disables the swap.
evaluate_py = evaluate
try:
    from .fast.evaluator import make_fast_evaluate as _make_fast_evaluate

    _fast_evaluate = _make_fast_evaluate(evaluate_py)
except Exception:  # pragma: no cover - the reference must never depend on the extension
    _fast_evaluate = None
if _fast_evaluate is not None:
    evaluate = _fast_evaluate
