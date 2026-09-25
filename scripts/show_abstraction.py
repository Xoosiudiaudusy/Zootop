"""Look at the abstraction with your own eyes.

    python scripts/show_abstraction.py                 # fit buckets (~20s), show everything
    python scripts/show_abstraction.py --board "Kh 9h 2c" --hands "Ah Qh,Kd 3d,9s 8s,7c 7d,Jh Th"

Sections:
  1. why the full game is too big (counts)
  2. equity buckets: cut points per street, and which example hands share a bucket
  3. bet-size grid on a real spot, plus how off-grid sizes are mapped back
  4. the information-set key CFR will use for that spot
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import BetGrid, EquityBucketer, PotentialAwareBucketer, infoset_key, load_bucketer, pseudo_harmonic  # noqa: E402
from negpluribus.abstraction.actions import raise_name  # noqa: E402
from negpluribus.cards import cards_from_str, cards_to_str  # noqa: E402
from negpluribus.engine import CALL, FOLD, HandState, raise_to  # noqa: E402

CACHE = os.path.join(os.path.dirname(__file__), "..", "negpluribus", "buckets_ehs10.json")
CACHE_POT = os.path.join(os.path.dirname(__file__), "..", "negpluribus", "buckets_pot10.json")


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="Kh 9h 2c")
    ap.add_argument("--hands", default="Ah Qh,Kd 3d,9s 8s,7c 7d,Jh Th,As 5s,Kc Kd,6d 5d")
    ap.add_argument("--buckets", type=int, default=10)
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()

    section("1. Why the full game cannot be tabulated")
    print("distinct (hole, board) situations:      raw            after suit isomorphism")
    for name, raw, iso in [("preflop", 1_326, 169), ("flop", 25_989_600, 1_286_792),
                           ("turn", 305_377_800, 13_960_050), ("river", 2_809_475_760, 123_156_254)]:
        print(f"  {name:>8}: {raw:>16,}   {iso:>16,}")
    print("(board as one set, as our bucket key; keys that keep the flop, turn and river cards")
    print(" apart have 55,190,538 turn and 2,428,287,420 river forms)")
    print("and each of those is multiplied by every betting sequence; with any chip amount")
    print("allowed the sequences are effectively infinite.  So: merge hands (buckets) and")
    print("restrict bet sizes (grid).")

    section(f"2. Equity buckets (E[HS], {args.buckets} equal-frequency buckets per street)")
    if os.path.exists(CACHE) and not args.refit:
        bk = EquityBucketer.load(CACHE)
        print(f"loaded {CACHE}")
    else:
        bk = EquityBucketer(n_buckets=args.buckets, samples=200)
        print("fitting on random situations (this takes ~20s)…")
        bk.fit(n_situations=1500, seed=0, verbose=True)
        bk.save(CACHE)
    # the potential-aware twin (same interface): EMD clusters of the next-street equity histogram
    if os.path.exists(CACHE_POT) and not args.refit:
        pk = load_bucketer(CACHE_POT)
        print(f"loaded {CACHE_POT}")
    else:
        pk = PotentialAwareBucketer(n_buckets=args.buckets)
        print("fitting potential-aware buckets (k-means under EMD over next-street equity histograms)…")
        pk.fit(n_situations=1200, seed=0, verbose=True)
        pk.save(CACHE_POT)
    board = cards_from_str(args.board)
    hands = [cards_from_str(h) for h in args.hands.split(",")]
    print(f"\nboard [{cards_to_str(board)}]:")
    rows = sorted(((bk.ehs(h, board), h) for h in hands), reverse=True)
    for eq, h in rows:
        shape = ""
        if len(board) in (3, 4):
            cdf, _ = pk.feature(h, board)
            shape = f"   next-card equity histogram {pk.describe_centroid(cdf)}"
        print(f"  {cards_to_str(h)}  equity vs random {eq:5.2f}  -> E[HS] bucket {bk.bucket(h, board)}   "
              f"potential-aware bucket {pk.bucket(h, board)}{shape}")
    print("\nhands in the same bucket are indistinguishable to the blueprint.  A flush draw and a")
    print("weak pair can share an E[HS] bucket.  The potential-aware bucketer clusters the histogram")
    print("(mass per 0.1 of equity after the next card, weakest bin first, in tenths) instead, and")
    print("keeps them apart when the clusters are fine enough; see docs/buckets.md.")

    section("3. Bet-size grid on a real spot")
    grid = BetGrid()
    h = HandState([10_000] * 6, button=0, seed=3)
    obs = h.observe()
    print(f"preflop, {obs.position} to act, pot={obs.pot}, to_call={obs.to_call}:")
    for name in grid.abstract_actions(obs):
        print(f"  {name:>5} -> {grid.to_concrete(obs, name)}")
    h.apply(raise_to(300))
    h.apply(FOLD)
    obs = h.observe()
    print(f"\nafter UTG raises to 300, {obs.position} to act, pot={obs.pot}, to_call={obs.to_call}:")
    for name in grid.abstract_actions(obs):
        print(f"  {name:>5} -> {grid.to_concrete(obs, name)}")
    print("\npseudo-harmonic mapping of off-grid sizes onto {0.5, 1.0} pot:")
    for x in (0.55, 0.66, 0.75, 0.85, 0.95):
        rng = random.Random(1)
        n = sum(1 for _ in range(1000) if pseudo_harmonic(x, (0.5, 1.0), rng) == 0.5)
        print(f"  observed {x:.2f} pot -> r0.5 with prob {n / 1000:.2f}, else r1  (nearest-size would say {raise_name(0.5 if x < 0.75 else 1.0)})")

    section("4. Information-set key for a spot")
    h = HandState([10_000] * 6, button=0, seed=7)
    h.apply(raise_to(250))   # UTG opens 2.5bb
    h.apply(FOLD)
    h.apply(CALL)            # CO calls
    h.apply(FOLD)
    h.apply(FOLD)
    h.apply(CALL)            # BB calls -> flop
    h.apply(CALL)            # BB checks
    obs = h.observe()        # UTG to act on flop
    key = infoset_key(obs, bk, grid)
    print(f"hero {obs.position} holds [{cards_to_str(obs.hole)}] on [{cards_to_str(obs.board)}], pot={obs.pot}")
    print(f"  key = {key}")
    print("  street | position | live players | card bucket | abstract history (streets split by '/')")
    print("\nCFR keeps one row of regrets per such key.  Every hand that produces the same key")
    print("is played the same way: that is the whole point of abstraction.")


if __name__ == "__main__":
    main()
