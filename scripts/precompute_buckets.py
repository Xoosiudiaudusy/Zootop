"""Compute every flop and turn bucket of a bucketer once and save them, for searches and trainings.

    python scripts/precompute_buckets.py --buckets data/buckets_hunl200w3_pot16_s0.json \
        --out data/bucketcache_hunl200w3_pot16_s0.bin --threads 14

Potential-aware buckets cost about 0.4 ms per flop or turn form (a Monte-Carlo histogram), and a
search on a new flop meets tens of thousands of turn forms in its rollouts and turn infosets.  This
walks every suit-canonical board (1,755 flops, 16,432 turns) with every hole, which fills the C++
bucketer's cache with every canonical form (1.29M flop, 13.96M turn), and saves the cache
(csrc/bucketcache.h: the bucketer's identity, then the cached words; loading a cache of other
buckets is refused).  Load it with ``bucketer.load_cache(path)`` on a core bucketer whose turn cache
holds 64M entries and whose flop cache holds 8M (``core_bucketer(bk, (8_000_000, 64_000_000, 4_000_000))``),
so nothing is evicted.  River buckets are not stored: searches compute them per board in one batch
(0.2 ms per board).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.fast.power import disable_power_throttling  # noqa: E402
from negpluribus.fast.trainer import core_bucketer  # noqa: E402

CAPS = (8_000_000, 64_000_000, 4_000_000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--streets", default="1,2", help="1 flop, 2 turn (3 river: 123M forms, not worth it)")
    args = ap.parse_args()
    disable_power_throttling()
    bk = core_bucketer(load_bucketer(args.buckets), CAPS)
    streets = [int(s) for s in args.streets.split(",") if s.strip()]
    for s in streets:
        t = time.perf_counter()
        visited = bk.precompute(s, args.threads)
        st = bk.cache_stats()[{1: "flop", 2: "turn", 3: "river"}[s]]
        print(f"street {s}: {visited:,} (board, hole) pairs, {st['size']:,} forms cached, {st['computes']:,} computed, "
              f"{st['evictions']:,} evicted, {time.perf_counter() - t:.0f}s on {args.threads} threads", flush=True)
    t = time.perf_counter()
    bk.save_cache(args.out, streets)
    print(f"saved {args.out}: {os.path.getsize(args.out):,} bytes in {time.perf_counter() - t:.1f}s")
    fresh = core_bucketer(load_bucketer(args.buckets), CAPS)
    t = time.perf_counter()
    loaded = fresh.load_cache(args.out)
    print(f"load check: {loaded} (street, entries, evicted) in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
