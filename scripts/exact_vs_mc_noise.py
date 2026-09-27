"""How often do Monte-Carlo potential features put a hand in another bucket than the exact feature?

    python scripts/exact_vs_mc_noise.py --k 16,64,256 --flop 600 --turn 3000

For each k: PotentialAwareBucketer(n_buckets=k, samples=100, bins=10).fit(n_situations=75*k, seed=0)
(Monte-Carlo features, as in the project; k-means as in the project, or --numpy-kmeans-from K: the same
algorithm on numpy arrays for k >= K, identical steps and random draws, sums possibly in another order).
Then on fresh random deals (own seed): the nearest centroid (EMD = L1 on CDFs, first minimum) of the
Monte-Carlo feature (the bucketer's own) and of the exact feature (core.exact_feature).  Per street:
share of hands in another bucket (95% interval), mean |shift| of the bucket index, and how much worse
(EMD to the exact feature) the Monte-Carlo pick is than the best centroid, absolute and relative.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from negpluribus.abstraction import PotentialAwareBucketer  # noqa: E402
from negpluribus.abstraction import potential as pot  # noqa: E402
from negpluribus.abstraction.potential import emd, histogram_cdf  # noqa: E402
from negpluribus.fast import core  # noqa: E402


def kmeans_emd_numpy(points, k, rng, max_iter=100):
    """potential.kmeans_emd on numpy arrays: k-means++ seeding with the same draws, Lloyd iterations
    with the mean CDF, empty clusters re-seeded with the farthest point; first minimum on ties."""
    P = np.asarray(points, dtype=np.float64)
    n = len(P)
    if n < k:
        raise ValueError(f"need at least {k} situations to fit {k} clusters, got {n}")
    centroids = [P[rng.randrange(n)].copy()]
    dmin = np.abs(P - centroids[0]).sum(axis=1)
    while len(centroids) < k:
        w = dmin * dmin
        s = float(w.sum())
        if s <= 0:
            c = P[rng.randrange(n)].copy()
        else:
            r = rng.random() * s
            cum = np.cumsum(w)
            pick = int(np.searchsorted(cum, r, side="right"))
            c = P[min(pick, n - 1)].copy()
        centroids.append(c)
        dmin = np.minimum(dmin, np.abs(P - c).sum(axis=1))
    C = np.array(centroids)
    assign = np.full(n, -1)
    for _ in range(max_iter):
        D = np.abs(P[:, None, :] - C[None, :, :]).sum(axis=2)
        new = D.argmin(axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            members = np.nonzero(assign == j)[0]
            if len(members) == 0:
                dist = np.abs(P - C[assign]).sum(axis=1)
                far = int(dist.argmax())
                C[j] = P[far]
                assign[far] = j
                continue
            C[j] = P[members].mean(axis=0)
    return [list(map(float, c)) for c in C], [int(a) for a in assign]


def nearest(cdf, cens):
    best, bd = 0, None
    for j, c in enumerate(cens):
        d = emd(cdf, c)
        if bd is None or d < bd:
            best, bd = j, d
    return best, bd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="16,64,256")
    ap.add_argument("--flop", type=int, default=600)
    ap.add_argument("--turn", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=12345, help="seed of the test deals")
    ap.add_argument("--numpy-kmeans-from", type=int, default=128)
    args = ap.parse_args()
    c = core()
    original = pot.kmeans_emd
    for k in [int(x) for x in args.k.split(",")]:
        pot.kmeans_emd = kmeans_emd_numpy if k >= args.numpy_kmeans_from else original
        t0 = time.time()
        bk = PotentialAwareBucketer(n_buckets=k, samples=100, bins=10).fit(n_situations=75 * k, seed=0)
        print(f"\n== k = {k}: fit on {75 * k} situations per street in {time.time() - t0:.0f}s"
              f"{' (numpy k-means)' if k >= args.numpy_kmeans_from else ''}", flush=True)
        rng = random.Random(args.seed)
        for street, n_board, n in ((1, 3, args.flop), (2, 4, args.turn)):
            cens = bk.centroids[street]
            diff = 0
            shifts, worse_abs, worse_rel = [], [], []
            t0 = time.time()
            for _ in range(n):
                cards = rng.sample(range(52), 2 + n_board)
                hole, board = cards[:2], cards[2:]
                cdf_mc = bk.feature(hole, board)[0]
                counts, _ = c.exact_feature(hole, board, 10)
                cdf_ex = histogram_cdf(counts)
                b_mc, _ = nearest(cdf_mc, cens)
                b_ex, d_best = nearest(cdf_ex, cens)
                d_mc = emd(cdf_ex, cens[b_mc])
                if b_mc != b_ex:
                    diff += 1
                shifts.append(abs(b_mc - b_ex))
                worse_abs.append(d_mc - d_best)
                worse_rel.append(d_best)
            p = diff / n
            half = 1.96 * math.sqrt(p * (1 - p) / n)
            mean_best = sum(worse_rel) / n
            mean_worse = sum(worse_abs) / n
            name = "flop" if street == 1 else "turn"
            print(f"  {name}: n={n}  other bucket {100 * p:5.1f}% +/- {100 * half:.1f}%  mean |shift| {sum(shifts) / n:.2f} "
                  f"(over those moved {sum(shifts) / max(1, diff):.2f})  EMD to exact: best {mean_best:.3f}, MC pick +{mean_worse:.4f} "
                  f"(+{100 * mean_worse / mean_best:.1f}%)  [{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
