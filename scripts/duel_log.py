"""Summaries of the per-deal logs of scripts/eval_archetypes.py --log (also while the match runs).

    python scripts/duel_log.py data/duel_search05.jsonl
    python scripts/duel_log.py data/overbet_search05.jsonl --paired data/overbet_blueprint.jsonl

Per opponent: deals, hands, bb/100 raw and card-luck corrected with the 95% CI (one deal = one
sample), and the seconds per hand (the hero's decisions, the whole hand).  --paired: the difference
to another log deal by deal (the same --seed means the same decks and seats), which removes the card
luck both logs share.  --first/--last: a window of deals.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def read(path: str) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by an interrupted run
            out[row["opponent"]][int(row["deal"])] = row
    return out


def ci(xs: List[float], per: float) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("inf")
    m = sum(xs) / n
    if n < 2:
        return m / per * 100, float("inf")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m / per * 100, 1.96 * sd / math.sqrt(n) / per * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--paired", default=None, help="another log of the same seed: deal-by-deal differences")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=None)
    args = ap.parse_args()
    a = read(args.log)
    b = read(args.paired) if args.paired else None
    for opp, deals in sorted(a.items()):
        keys = sorted(d for d in deals if d >= args.first and (args.last is None or d <= args.last))
        if not keys:
            continue
        rows = [deals[d] for d in keys]
        per = len(rows[0].get("events") or [1, 1])  # hands per deal (the hero in every seat)
        hands = per * len(rows)
        raw = [r["raw"] for r in rows]
        m, c = ci(raw, per)
        line = f"vs {opp}: {len(rows)} deals ({keys[0]}..{keys[-1]}), {hands} hands: {m:+.1f} bb/100 +/-{c:.1f}"
        if all("corrected" in r for r in rows):
            mc, cc = ci([r["corrected"] for r in rows], per)
            line += f"; luck-corrected {mc:+.1f} +/-{cc:.1f}"
        hs = sum(r.get("hero_s", 0.0) for r in rows) / hands
        ws = sum(r.get("hand_s", 0.0) for r in rows) / hands
        line += f"; {hs:.2f}s hero / {ws:.2f}s per hand"
        print(line)
        if b is not None and opp in b:
            common = [d for d in keys if d in b[opp]]
            if common:
                dr = [deals[d]["raw"] - b[opp][d]["raw"] for d in common]
                m, c = ci(dr, per)
                out = f"   paired with {args.paired} on {len(common)} deals: {m:+.1f} bb/100 +/-{c:.1f}"
                if all("corrected" in deals[d] and "corrected" in b[opp][d] for d in common):
                    mc, cc = ci([deals[d]["corrected"] - b[opp][d]["corrected"] for d in common], per)
                    out += f"; luck-corrected {mc:+.1f} +/-{cc:.1f}"
                print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
