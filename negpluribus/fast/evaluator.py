"""Compiled 5..7 card evaluators behind the ``negpluribus.evaluator.evaluate`` signature.

Tier 1: ``_fastcore.evaluate`` - a C++ port of the reference algorithm, returns the very
        same integer.
Tier 2: ``phevaluator`` (pip, C perfect hash).  Its rank runs 1 (royal flush) .. 7462 (worst
        high card), lower = better.  A 7462-entry table translates the rank into the reference
        value, built once (~50 ms) by evaluating one representative hand per rank class with the
        pure-Python evaluator, so the numbers are again identical, not merely order-preserving.
Tier 3: the pure-Python evaluator (``make_fast_evaluate`` returns ``None``).
"""
from __future__ import annotations

from itertools import combinations, combinations_with_replacement
from typing import Callable, List, Optional, Sequence

from . import core, fast_eval_enabled

_tier = "python"


def evaluator_tier() -> str:
    return _tier


def _phe_table(evaluate_py: Callable[[Sequence[int]], int], phe5) -> List[int]:
    """phevaluator rank (1..7462) -> reference value, via one representative per rank class."""
    table = {}
    # non-flush hands: every rank multiset of size 5 except five-of-a-kind (6175 classes)
    for ranks in combinations_with_replacement(range(13), 5):
        if len(set(ranks)) == 1:
            continue
        cards = []
        seen = {}
        for r in ranks:
            s = seen.get(r, 0)
            seen[r] = s + 1
            cards.append(r * 4 + s)
        if len({c & 3 for c in cards}) == 1:  # five distinct ranks all in suit 0 -> not a flush
            cards[-1] = (cards[-1] & ~3) | 1
        table[phe5(*cards)] = evaluate_py(cards)
    # flushes: every 5-subset of ranks in one suit (1287 classes)
    for ranks in combinations(range(13), 5):
        cards = [r * 4 for r in ranks]
        table[phe5(*cards)] = evaluate_py(cards)
    if len(table) != 7462 or set(table) != set(range(1, 7463)):
        raise RuntimeError("phevaluator rank classes do not map 1:1 onto the reference evaluator")
    return [table[i] for i in range(1, 7463)]


def make_fast_evaluate(evaluate_py: Callable[[Sequence[int]], int]) -> Optional[Callable[[Sequence[int]], int]]:
    """Return a drop-in replacement for ``evaluate`` or ``None`` (keep the reference)."""
    global _tier
    if not fast_eval_enabled():
        return None
    c = core()
    if c is not None:
        _tier = "cpp"
        _ev = c.evaluate

        def evaluate_cpp(cards: Sequence[int]) -> int:
            n = len(cards)
            if n < 5 or n > 7:
                raise ValueError("evaluate expects 5..7 cards")
            return _ev(cards)

        return evaluate_cpp
    try:
        from phevaluator._pheval import evaluate_5cards, evaluate_6cards, evaluate_7cards
    except Exception:
        return None
    table = _phe_table(evaluate_py, evaluate_5cards)
    fns = {5: evaluate_5cards, 6: evaluate_6cards, 7: evaluate_7cards}
    _tier = "phevaluator"

    def evaluate_phe(cards: Sequence[int]) -> int:
        fn = fns.get(len(cards))
        if fn is None:
            raise ValueError("evaluate expects 5..7 cards")
        return table[fn(*cards) - 1]

    return evaluate_phe
