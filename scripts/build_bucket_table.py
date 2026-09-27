"""Precompute the postflop buckets of a fitted bucketer for every hand class (fast/tables.py).

    python scripts/build_bucket_table.py --buckets data/buckets_<tag>.json [--out data/bucket_tables] [--threads 16]

Writes <out>/buckets_<kind>_<n>_<fingerprint>.npbt (flop 1.3 MB + turn 14 MB + river 123 MB).
Training then uses it when NEGPLURIBUS_BUCKET_TABLES=<out> is set; the buckets are the same
numbers bucket() computes (checked here on --check random hands per street before writing).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.fast import core, default_threads  # noqa: E402
from negpluribus.fast.tables import STREETS, table_name  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", required=True, help="fitted bucketer JSON (data/buckets_<tag>.json)")
    ap.add_argument("--out", default=os.path.join(DATA, "bucket_tables"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--streets", default="1,2,3", help="1 flop, 2 turn, 3 river")
    ap.add_argument("--check", type=int, default=20000, help="random hands per street compared with bucket()")
    ap.add_argument("--features-dir", default=None,
                    help="exact potential-aware buckets: take the flop / turn from saved exact features "
                         "(exact_features_<street>_b<bins>.bin from core.build_exact_features) when present")
    args = ap.parse_args()

    from negpluribus.fast.trainer import core_bucketer

    os.environ.pop("NEGPLURIBUS_BUCKET_TABLES", None)  # the reference must be the plain bucketer
    cbk = core_bucketer(load_bucketer(args.buckets), cache_caps=(0, 0, 0))
    streets = [int(s) for s in args.streets.split(",")]
    threads = args.threads or default_threads()
    path = os.path.join(args.out, table_name(cbk))
    print(f"bucketer {cbk.identity}\n-> {path}, {threads} threads", flush=True)
    t = core().BucketTables()
    rng = random.Random(0)
    exact = bool(getattr(cbk, "exact", False))
    for s in streets:
        t0 = time.time()
        feat = None
        if exact and args.features_dir and s in (1, 2):
            feat = os.path.join(args.features_dir, f"exact_features_{'flop' if s == 1 else 'turn'}_b{cbk.bins}.bin")
            if not os.path.exists(feat):
                feat = None
        if feat:
            t.build_from_features(cbk, s, feat, threads)
        else:
            t.build(cbk, s, threads=threads, every=30.0,
                    progress=lambda d, n: print(f"  street {s}: {d / n:6.1%}  {time.time() - t0:7.0f}s", flush=True))
        dt = time.time() - t0
        print(f"street {s}: {t.size(s):,} classes in {dt:.0f}s" + (f" (from {feat})" if feat else ""), flush=True)
        bad = 0
        # an exact flop bucket by the definition costs ~70 ms: check fewer of those
        n_check = min(args.check, 500) if exact and s == 1 else args.check
        for _ in range(n_check):
            cards = rng.sample(range(52), 4 + s)
            if t.lookup(cards[:2], cards[2:]) != cbk.bucket(cards[:2], cards[2:]):
                bad += 1
        print(f"  check: {bad} mismatches in {n_check} random hands", flush=True)
        if bad:
            print("MISMATCH: table not written", file=sys.stderr)
            return 1
    os.makedirs(args.out, exist_ok=True)
    t.save(path)
    print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
