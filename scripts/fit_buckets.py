"""Fit a postflop bucketer on its own: what train_blueprint.py does when data/buckets_<tag>.json is
missing (make_bucketer(kind, n) then fit(n_situations, seed)), so that its bucket table
(scripts/build_bucket_table.py) can be built before the training starts.

    python scripts/fit_buckets.py --buckets 64 --kind potential --situations 4800 --seed 0 \
        --out data/buckets_<tag>.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import BUCKET_KINDS, make_bucketer  # noqa: E402
from negpluribus.fast.power import disable_power_throttling  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", type=int, required=True)
    ap.add_argument("--kind", choices=list(BUCKET_KINDS), default="potential")
    ap.add_argument("--situations", type=int, required=True, help="random situations per street (train_blueprint --fit-situations)")
    ap.add_argument("--seed", type=int, default=0, help="the seed train_blueprint passes from --seed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; not overwritten")
    disable_power_throttling()  # scheduling only
    bk = make_bucketer(args.kind, args.buckets, None)
    t0 = time.perf_counter()
    bk.fit(n_situations=args.situations, seed=args.seed, verbose=True)
    bk.save(args.out)
    print(f"fitted in {time.perf_counter() - t0:.0f}s, saved {args.out}")


if __name__ == "__main__":
    main()
