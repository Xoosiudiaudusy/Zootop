"""Evaluate an archetype (or later: your bot) against a line-up with duplicate deals.

    python scripts/run_eval.py --hero tag --villains nit,station,maniac,lag,passive --deals 300
    python scripts/run_eval.py --hero lag --baseline tag ...      # paired gain of lag over tag

Output is bb/100 with a 95% confidence interval, plus a breakdown by position
and the HUD the villains generated (so you can see what an exploiter would see).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.agents import make_agent  # noqa: E402
from negpluribus.eval import compare_heroes, duplicate_match  # noqa: E402
from negpluribus.stats import StatsTracker  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", default="tag")
    ap.add_argument("--baseline", default=None, help="second hero for a paired comparison")
    ap.add_argument("--villains", default="nit,station,maniac,lag,passive")
    ap.add_argument("--deals", type=int, default=200, help="each deal is played 6x (hero in every seat)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vil_names = args.villains.split(",")
    villains = [make_agent(v, seed=100 + i, label=f"{v}#{i + 1}") for i, v in enumerate(vil_names)]
    hero = make_agent(args.hero, seed=1)
    t = time.perf_counter()
    if args.baseline:
        base = make_agent(args.baseline, seed=1, label=args.baseline + "(base)")
        ra, rb, gain, ci = compare_heroes(hero, base, villains, n_deals=args.deals, seed=args.seed)
        print(ra)
        print(rb)
        print(f"\nGAIN of {hero.name} over {base.name}: {gain:+.2f} bb/100  (95% CI +/-{ci:.2f})  "
              f"[{'significant' if abs(gain) > ci else 'not significant'}]")
    else:
        tracker = StatsTracker()
        res = duplicate_match(hero, villains, n_deals=args.deals, seed=args.seed, tracker=tracker)
        print(res)
        print("\nHUD after the match:")
        print(tracker.report([hero.name] + [v.name for v in villains]))
    print(f"\n({time.perf_counter() - t:.1f}s)")


if __name__ == "__main__":
    main()
