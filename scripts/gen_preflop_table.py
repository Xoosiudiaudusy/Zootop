"""Generate negpluribus/preflop_equity_6max.json.

For each of the 169 starting-hand classes, Monte-Carlo the all-in equity vs
N random opponents (default 5 = 6-max) and store it.  ~1 minute on CPython
with the default sample count.

    python scripts/gen_preflop_table.py [--samples 3000] [--opponents 5]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.cards import ALL_HOLE_CLASSES, RANKS, make_card  # noqa: E402
from negpluribus.equity import equity_vs_random  # noqa: E402


def representative(cls: str):
    r1 = RANKS.index(cls[0])
    r2 = RANKS.index(cls[1])
    if len(cls) == 2:  # pair
        return [make_card(r1, 0), make_card(r2, 1)]
    if cls[2] == "s":
        return [make_card(r1, 0), make_card(r2, 0)]
    return [make_card(r1, 0), make_card(r2, 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--opponents", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = {}
    t0 = time.time()
    for i, cls in enumerate(ALL_HOLE_CLASSES):
        out[cls] = equity_vs_random(
            representative(cls), [], args.opponents, args.samples, rng
        )
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/169  {time.time() - t0:.0f}s", flush=True)

    path = os.path.join(
        os.path.dirname(__file__), "..", "negpluribus", "preflop_equity_6max.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0, sort_keys=True)
    ranked = sorted(out.items(), key=lambda kv: -kv[1])
    print("top 10:", [f"{k}={v:.3f}" for k, v in ranked[:10]])
    print("bottom 5:", [f"{k}={v:.3f}" for k, v in ranked[-5:]])
    print("saved", os.path.abspath(path))


if __name__ == "__main__":
    main()
