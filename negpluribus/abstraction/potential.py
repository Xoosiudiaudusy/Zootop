"""Potential-aware card abstraction: cluster hands by where their equity is *going*.

E[HS] (``buckets.py``) answers "how much do I win right now against a random hand".  Two
hands with the same answer can be very different hands: on ``Kh 9h 2c`` the nut flush draw
``Ah Qh`` and the second pair ``Ts 9s`` both sit at E[HS] ~ 0.72, yet after one more card the
draw is either ~0.9 (a heart comes, 9 of 47 cards) or ~0.55 (it does not), while the pair
stays near 0.7 on almost every card.  A blueprint that sees one bucket for both plays them
the same way; potential-aware buckets keep them apart.

The idea is the one of Ganzfried & Sandholm, "Potential-aware imperfect-recall abstraction
with earth mover's distance in imperfect-information games", AAAI 2014 (foundational and old,
but still what Libratus / Pluribus-style blueprints use): a hand's feature on a street is the
*distribution* of its equity on the next street, and hands are clustered with k-means under
the earth mover's distance (EMD) between those distributions.

Concretely, per street:

* **flop**: for each of the 47 possible turn cards, E[HS] on that turn (Monte-Carlo with
  ``samples`` runouts vs one random hand); histogram of the 47 values over ``bins`` equal
  bins of [0, 1];
* **turn**: the same over the 46 river cards (river equity is Monte-Carlo too);
* **river**: nothing is left to draw, so the feature is the exact equity vs one random hand
  (all 990 opponent combos) and the buckets are equal-frequency cut points, exactly the E[HS]
  rule (a 1-bin histogram would be the same thing);
* **preflop**: the lossless 169 starting-hand classes, identical to ``EquityBucketer``.

For a one-dimensional histogram EMD is the L1 distance between the cumulative distributions
(with equal bins), and the EMD "mean" of a set of histograms is the mean CDF, so k-means runs
in CDF space with the L1 metric.  After clustering the clusters are **ordered by the mean
equity of their members**, so the bucket index still grows (approximately) with hand
strength: ``exploit/model.py::key_strength`` reads ``b / (n_buckets - 1)`` as a 0..1 strength
and needs that monotonicity.

Reproducibility: the feature of a hand is computed on its *canonical* (suit-relabelled)
representative with a ``random.Random`` seeded from the CPython hash of the canonical key,
so it is the same number in every process and every backend, and it is cached per canonical
form (``EquityBucketer.ehs`` does the same since 2026-09-23).  The C++ core
(``csrc/abstraction.h::PotentialBucketer``) ports this byte for byte: same draws, same
histogram, same argmin.

Cost: a flop feature is 47 x ``samples`` Monte-Carlo runouts (2 evaluations each), i.e.
~9 400 evaluations at the default ``samples=100`` against ~300 for an E[HS] bucket; a river
feature is 990 evaluations.  Numbers in docs/buckets.md.

Granularity caveat (measured, docs/buckets.md): a flush draw and a made hand of equal E[HS]
are 0.6-1.3 bins apart under EMD, while 8 clusters per street have a radius of ~0.4-0.5 bins,
so at the game's ``n_buckets=8`` only the strongly bimodal draws get a cluster of their own;
with 12-16 clusters most draw-vs-made pairs separate.  The paper used thousands.
"""
from __future__ import annotations

import bisect
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..cards import ALL_HOLE_CLASSES, hole_class
from ..engine import Street
from ..equity import equity_vs_random
from ..evaluator import evaluate
from .buckets import BOARD_SIZE, POSTFLOP, EquityBucketer, _CLASS_INDEX
from .canonical import canonical_key

STREET_OF_BOARD = {3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}
HISTOGRAM_STREETS = (Street.FLOP, Street.TURN)


# ---------------------------------------------------------------- features (reference)
def next_street_histogram(hole: Sequence[int], board: Sequence[int], samples: int, bins: int) -> Tuple[List[int], float]:
    """Counts over ``bins`` equal bins of [0, 1] of E[HS] after each possible next card (47 on
    the flop, 46 on the turn, in card order), and the mean of those equities.  One
    ``random.Random`` seeded from the canonical key is consumed sequentially, so the result is
    a pure function of (hole, board, samples, bins)."""
    used = set(hole) | set(board)
    rng = random.Random(hash(canonical_key(hole, board)) & 0xFFFFFFFF)
    hole_l, board_l = list(hole), list(board)
    counts = [0] * bins
    total = 0.0
    n = 0
    for c in range(52):
        if c in used:
            continue
        e = equity_vs_random(hole_l, board_l + [c], 1, samples, rng)
        b = int(e * bins)
        if b >= bins:
            b = bins - 1
        counts[b] += 1
        total += e
        n += 1
    return counts, total / n


def river_equity(hole: Sequence[int], board: Sequence[int]) -> float:
    """Exact equity vs one random hand on a 5-card board: all C(45, 2) = 990 opponent combos."""
    used = set(hole) | set(board)
    rest = [c for c in range(52) if c not in used]
    board_l = list(board)
    mine = evaluate(list(hole) + board_l)
    won = 0.0
    n = 0
    for i in range(len(rest)):
        for j in range(i + 1, len(rest)):
            v = evaluate([rest[i], rest[j]] + board_l)
            if mine > v:
                won += 1.0
            elif mine == v:
                won += 0.5
            n += 1
    return won / n


# ---- fast backend hook (negpluribus/fast): the same numbers from C++ (same MT draws from the
# same seed, same summation order); NEGPLURIBUS_FAST_EVAL=0 or an older core keeps the reference.
next_street_histogram_py, river_equity_py = next_street_histogram, river_equity
try:
    from ..fast import core as _fast_core, fast_eval_enabled as _fast_eval_enabled

    _core = _fast_core() if _fast_eval_enabled() else None
except Exception:  # pragma: no cover
    _core = None
if _core is not None and hasattr(_core, "potential_histogram") and hasattr(_core, "river_equity_exact"):
    _ph, _re = _core.potential_histogram, _core.river_equity_exact

    def next_street_histogram(hole: Sequence[int], board: Sequence[int], samples: int, bins: int) -> Tuple[List[int], float]:  # noqa: F811
        counts, mean = _ph(hole, board, samples, bins)
        return list(counts), mean

    def river_equity(hole: Sequence[int], board: Sequence[int]) -> float:  # noqa: F811
        return _re(hole, board)


# ------------------------------------------------------------------ EMD helpers
def histogram_cdf(counts: Sequence[int]) -> List[float]:
    n = sum(counts)
    out = []
    cum = 0
    for c in counts:
        cum += c
        out.append(cum / n)
    return out


def emd(cdf_a: Sequence[float], cdf_b: Sequence[float]) -> float:
    """Earth mover's distance between two histograms on the same equal bins = L1 distance of
    their CDFs (in units of one bin width).  Summed in index order, as the C++ port does."""
    d = 0.0
    for a, b in zip(cdf_a, cdf_b):
        d += abs(a - b)
    return d


def nearest_centroid(cdf: Sequence[float], centroids: Sequence[Sequence[float]]) -> int:
    best, best_d = 0, None
    for j, cen in enumerate(centroids):
        d = emd(cdf, cen)
        if best_d is None or d < best_d:
            best, best_d = j, d
    return best


def kmeans_emd(points: List[List[float]], k: int, rng: random.Random, max_iter: int = 100) -> Tuple[List[List[float]], List[int]]:
    """k-means on CDFs under the L1 (= EMD) metric: k-means++ seeding, Lloyd iterations with
    the mean CDF as centroid (the EMD barycentre of 1-D histograms on equal bins).
    Returns (centroids, assignment)."""
    n = len(points)
    if n < k:
        raise ValueError(f"need at least {k} situations to fit {k} clusters, got {n}")
    dim = len(points[0])
    centroids = [list(points[rng.randrange(n)])]
    while len(centroids) < k:
        w = []
        for p in points:
            d = min(emd(p, c) for c in centroids)
            w.append(d * d)
        s = sum(w)
        if s <= 0:
            centroids.append(list(points[rng.randrange(n)]))
            continue
        r = rng.random() * s
        acc = 0.0
        pick = n - 1
        for i, wi in enumerate(w):
            acc += wi
            if r < acc:
                pick = i
                break
        centroids.append(list(points[pick]))
    assign = [-1] * n
    for _ in range(max_iter):
        new = [nearest_centroid(p, centroids) for p in points]
        if new == assign:
            break
        assign = new
        for j in range(k):
            members = [i for i in range(n) if assign[i] == j]
            if not members:
                far = max(range(n), key=lambda i: emd(points[i], centroids[assign[i]]))
                centroids[j] = list(points[far])
                assign[far] = j
                continue
            centroids[j] = [sum(points[i][t] for i in members) / len(members) for t in range(dim)]
    return centroids, assign


# ------------------------------------------------------------------ the bucketer
class PotentialAwareBucketer:
    """Same interface as ``EquityBucketer``: ``bucket(hole, board) -> int``, ``n_buckets_for``,
    ``fit``, ``save`` / ``load``; ``boundaries`` holds the river cut points so that code which
    checks ``bucketer.boundaries`` for "fitted" keeps working."""

    kind = "potential"

    def __init__(self, n_buckets: int = 8, samples: int = 100, bins: int = 10):
        self.n_buckets = n_buckets
        self.samples = samples  # Monte-Carlo runouts per next-street card (flop / turn features)
        self.bins = bins
        self.centroids: Dict[int, List[List[float]]] = {}  # street -> n_buckets CDFs (ordered by strength)
        self.centroid_mean_equity: Dict[int, List[float]] = {}  # informational: mean E[HS] of each cluster
        self.centroid_share: Dict[int, List[float]] = {}  # informational: share of fit situations per cluster
        self.boundaries: Dict[int, List[float]] = {}  # river -> n-1 cut points (E[HS] rule)
        self._cache: Dict[tuple, int] = {}  # canonical key -> bucket
        self._features: Dict[tuple, Tuple[List[float], float]] = {}  # canonical key -> (CDF, mean), fit / analysis

    # ------------------------------------------------------------ features
    def feature(self, hole: Sequence[int], board: Sequence[int]) -> Tuple[List[float], float]:
        """(CDF of the next-street equity histogram, mean equity) for a flop or turn situation,
        computed on the canonical representative (what ``bucket`` clusters); cached."""
        key = canonical_key(hole, board)
        f = self._features.get(key)
        if f is None:
            counts, mean = next_street_histogram(key[:2], key[3:], self.samples, self.bins)
            f = (histogram_cdf(counts), mean)
            self._features[key] = f
        return f

    def river_ehs(self, hole: Sequence[int], board: Sequence[int]) -> float:
        key = canonical_key(hole, board)
        return river_equity(key[:2], key[3:])

    # ------------------------------------------------------------------ fit
    def fit(self, n_situations: int = 1000, seed: int = 0, verbose: bool = False, max_iter: int = 100) -> "PotentialAwareBucketer":
        """Sample ``n_situations`` random (hole, board) per street; k-means (EMD) on the flop /
        turn histograms, equal-frequency cut points on the river equity."""
        rng = random.Random(seed)
        for street in POSTFLOP:
            k = BOARD_SIZE[street]
            situations = [rng.sample(range(52), 2 + k) for _ in range(n_situations)]
            if street == Street.RIVER:
                eqs = sorted(self.river_ehs(cards[:2], cards[2:]) for cards in situations)
                self.boundaries[int(street)] = [eqs[int(len(eqs) * b / self.n_buckets)] for b in range(1, self.n_buckets)]
                if verbose:
                    cuts = " ".join(f"{c:.2f}" for c in self.boundaries[int(street)])
                    print(f"  {street.name.lower():>5}: exact equity, cuts at {cuts}")
                continue
            feats = [self.feature(cards[:2], cards[2:]) for cards in situations]
            cdfs = [f[0] for f in feats]
            means = [f[1] for f in feats]
            centroids, assign = kmeans_emd(cdfs, self.n_buckets, rng, max_iter=max_iter)
            # order clusters by the mean equity of their members: bucket index ~ hand strength
            stats = []
            for j in range(self.n_buckets):
                members = [means[i] for i in range(len(means)) if assign[i] == j]
                stats.append((sum(members) / len(members) if members else 0.0, len(members) / len(means)))
            order = sorted(range(self.n_buckets), key=lambda j: stats[j][0])
            self.centroids[int(street)] = [centroids[j] for j in order]
            self.centroid_mean_equity[int(street)] = [stats[j][0] for j in order]
            self.centroid_share[int(street)] = [stats[j][1] for j in order]
            if verbose:
                print(f"  {street.name.lower():>5}: {self.n_buckets} EMD clusters on {n_situations} situations")
                for b, j in enumerate(order):
                    print(f"         b{b}: mean eq {stats[j][0]:.2f}  share {stats[j][1]:.2f}  {self.describe_centroid(centroids[j])}")
        self._cache.clear()
        return self

    @staticmethod
    def describe_centroid(cdf: Sequence[float]) -> str:
        """Bin masses of a centroid as a compact bar string (one char per bin, 0..9)."""
        prev = 0.0
        out = []
        for c in cdf:
            out.append(str(min(9, int(round((c - prev) * 10)))))
            prev = c
        return "[" + "".join(out) + "]"

    # --------------------------------------------------------------- bucket
    def bucket(self, hole: Sequence[int], board: Sequence[int]) -> int:
        if len(board) == 0:
            return _CLASS_INDEX[hole_class(hole[0], hole[1])]
        key = canonical_key(hole, board)
        v = self._cache.get(key)
        if v is None:
            v = self._bucket_canonical(key)
            self._cache[key] = v
        return v

    def _bucket_canonical(self, key: tuple) -> int:
        hole, board = key[:2], key[3:]
        street = STREET_OF_BOARD[len(board)]
        if street == Street.RIVER:
            cuts = self.boundaries.get(int(street))
            if cuts is None:
                raise RuntimeError("bucketer not fitted; call fit() or load()")
            return bisect.bisect_right(cuts, river_equity(hole, board))
        centroids = self.centroids.get(int(street))
        if centroids is None:
            raise RuntimeError("bucketer not fitted; call fit() or load()")
        counts, _ = next_street_histogram(hole, board, self.samples, self.bins)
        return nearest_centroid(histogram_cdf(counts), centroids)

    def n_buckets_for(self, street: Street) -> int:
        return len(ALL_HOLE_CLASSES) if street == Street.PREFLOP else self.n_buckets

    @property
    def fitted(self) -> bool:
        return all(int(s) in self.centroids for s in HISTOGRAM_STREETS) and int(Street.RIVER) in self.boundaries

    # ------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "n_buckets": self.n_buckets,
            "samples": self.samples,
            "bins": self.bins,
            "centroids": {str(k): v for k, v in self.centroids.items()},
            "centroid_mean_equity": {str(k): v for k, v in self.centroid_mean_equity.items()},
            "centroid_share": {str(k): v for k, v in self.centroid_share.items()},
            "boundaries": {str(k): v for k, v in self.boundaries.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PotentialAwareBucketer":
        if d.get("kind") != cls.kind:
            raise ValueError(f"not a potential-aware bucketer file (kind={d.get('kind', 'ehs')!r}); use load_bucketer()")
        b = cls(d["n_buckets"], d["samples"], d["bins"])
        b.centroids = {int(k): [list(c) for c in v] for k, v in d["centroids"].items()}
        b.centroid_mean_equity = {int(k): list(v) for k, v in d.get("centroid_mean_equity", {}).items()}
        b.centroid_share = {int(k): list(v) for k, v in d.get("centroid_share", {}).items()}
        b.boundaries = {int(k): list(v) for k, v in d["boundaries"].items()}
        return b

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "PotentialAwareBucketer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ------------------------------------------------------------------ kind dispatch
BUCKET_KINDS = ("ehs", "potential")


def make_bucketer(kind: str, n_buckets: int, samples: Optional[int] = None, **kw):
    """``"ehs"`` -> ``EquityBucketer`` (default 150 samples), ``"potential"`` ->
    ``PotentialAwareBucketer`` (default 100 runouts per next-street card, 10 bins)."""
    if kind == "ehs":
        return EquityBucketer(n_buckets=n_buckets, samples=150 if samples is None else samples, **kw)
    if kind == "potential":
        return PotentialAwareBucketer(n_buckets=n_buckets, samples=100 if samples is None else samples, **kw)
    raise ValueError(f"bucket kind must be one of {BUCKET_KINDS}, got {kind!r}")


def bucketer_kind(bucketer) -> str:
    return getattr(bucketer, "kind", "ehs")


def load_bucketer(path: str):
    """Load either kind from JSON: files without a ``kind`` field are the old E[HS] format."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    kind = d.get("kind", "ehs")
    if kind == "ehs":
        return EquityBucketer.load(path)
    if kind == "potential":
        return PotentialAwareBucketer.from_dict(d)
    raise ValueError(f"unknown bucketer kind {kind!r} in {path}")
