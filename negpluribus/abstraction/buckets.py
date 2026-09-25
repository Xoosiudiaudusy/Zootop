"""Card abstraction by expected hand strength (E[HS]).

Idea: on each postflop street, compute how much equity a (hole, board) has
against one random hand, then cut the 0..1 equity line into ``n`` intervals so
that each interval holds the same share of random situations (equal-frequency
buckets).  Two hands in the same bucket are treated as the same hand by CFR.

This is the simplest "lossy" card abstraction.  It is what the first poker
bots used; Pluribus uses a richer *potential-aware* version (it also looks at
how equity is distributed over future cards, so a flush draw and a weak pair
with the same equity land in different buckets).  We start with E[HS] because
it is easy to see and to reason about, and the interface (``bucket(hole,
board) -> int``) does not change when we upgrade.

Preflop is kept lossless: 169 starting-hand classes, no equity involved.
"""
from __future__ import annotations

import bisect
import json
import random
from typing import Dict, List, Optional, Sequence

from ..cards import ALL_HOLE_CLASSES, hole_class
from ..engine import Street
from ..equity import equity_vs_random
from .canonical import canonical_key

_CLASS_INDEX = {c: i for i, c in enumerate(ALL_HOLE_CLASSES)}
POSTFLOP = (Street.FLOP, Street.TURN, Street.RIVER)
BOARD_SIZE = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


class EquityBucketer:
    def __init__(self, n_buckets: int = 10, samples: int = 200):
        self.n_buckets = n_buckets
        self.samples = samples
        self.boundaries: Dict[int, List[float]] = {}  # street -> n-1 cut points
        self._cache: Dict[tuple, float] = {}

    # ----------------------------------------------------------------- equity
    def ehs(self, hole: Sequence[int], board: Sequence[int], rng: Optional[random.Random] = None) -> float:
        """Equity vs one random hand, cached per canonical (suit-relabelled) form.

        The Monte-Carlo runs on the canonical *representative* (``key[:2]``, ``key[3:]``) with a
        generator seeded from the CPython hash of the key, so the value is a pure function of
        the canonical form: the same number in every process, every backend and after a cache
        eviction (the C++ core's caches are bounded).  Up to 2026-09-23 it ran on whichever
        representative was seen first, which made the cached value order-dependent.
        """
        key = canonical_key(hole, board)
        v = self._cache.get(key)
        if v is None:
            rng = rng or random.Random(hash(key) & 0xFFFFFFFF)
            v = equity_vs_random(key[:2], key[3:], 1, self.samples, rng)
            self._cache[key] = v
        return v

    # -------------------------------------------------------------------- fit
    def fit(self, n_situations: int = 2000, seed: int = 0, verbose: bool = False) -> "EquityBucketer":
        """Sample random (hole, board) per street and place cut points at equal quantiles."""
        rng = random.Random(seed)
        for street in POSTFLOP:
            k = BOARD_SIZE[street]
            eqs = []
            for _ in range(n_situations):
                cards = rng.sample(range(52), 2 + k)
                eqs.append(equity_vs_random(cards[:2], cards[2:], 1, self.samples, rng))
            eqs.sort()
            self.boundaries[int(street)] = [
                eqs[int(len(eqs) * b / self.n_buckets)] for b in range(1, self.n_buckets)
            ]
            if verbose:
                cuts = " ".join(f"{c:.2f}" for c in self.boundaries[int(street)])
                print(f"  {street.name.lower():>5}: cuts at {cuts}")
        return self

    # ----------------------------------------------------------------- bucket
    def bucket(self, hole: Sequence[int], board: Sequence[int]) -> int:
        if len(board) == 0:
            return _CLASS_INDEX[hole_class(hole[0], hole[1])]
        street = {3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}[len(board)]
        cuts = self.boundaries.get(int(street))
        if cuts is None:
            raise RuntimeError("bucketer not fitted; call fit() or load()")
        return bisect.bisect_right(cuts, self.ehs(hole, board))

    def n_buckets_for(self, street: Street) -> int:
        return len(ALL_HOLE_CLASSES) if street == Street.PREFLOP else self.n_buckets

    # ------------------------------------------------------------------- io
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"n_buckets": self.n_buckets, "samples": self.samples,
                 "boundaries": {str(k): v for k, v in self.boundaries.items()}},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "EquityBucketer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("kind", "ehs") != "ehs":  # old files carry no "kind"; other kinds live in potential.py
            raise ValueError(f"{path} is a {d['kind']!r} bucketer; use negpluribus.abstraction.load_bucketer()")
        b = cls(d["n_buckets"], d["samples"])
        b.boundaries = {int(k): v for k, v in d["boundaries"].items()}
        return b
