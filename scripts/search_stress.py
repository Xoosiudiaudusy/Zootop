"""Crash hunt for the subgame search: many random roots, several threads, no result checks.

    python scripts/search_stress.py --blueprint B.bin --buckets K.json --roots 5000 --seed 1

HU game of the blueprint (200bb, preflop 0.5/1/3, postflop 0.5/1/2/4, 3 raises).  Random hands to a random street
with off-grid raise sizes (inserted into the search), all-ins and unequal stacks (20-400bb); a random depth rule,
2..14 threads, fixed iterations or a short time budget; then likelihood() of both seats.  It prints a line every 20
roots and "OK n" at the end: a crash is the only finding.  Linux: build with -fsanitize=address,undefined (see the
search-speed report in comms) to see the first bad access.  Windows: enable WER LocalDumps for python.exe to get a
minidump (0xC0000409 is a fail-fast, which no in-process handler, faulthandler included, sees).
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from negpluribus import fast
from negpluribus.abstraction import load_bucketer
from negpluribus.cards import Deck
from negpluribus.cfr import GameSpec
from negpluribus.engine import HandState, Street, raise_to
from negpluribus.fast.blueprint import load_blueprint
from negpluribus.fast.trainer import core_bucketer, spec_to_dict

core = fast.core()
ap = argparse.ArgumentParser()
ap.add_argument("--blueprint", required=True)
ap.add_argument("--buckets", required=True)
ap.add_argument("--roots", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
bk = load_bucketer(args.buckets)
spec = GameSpec(n_players=2, stack_bb=200, max_street=Street.RIVER, preflop_fracs=(0.5, 1.0, 3.0),
                postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3, n_buckets=bk.n_buckets, bucket_kind=getattr(bk, "kind", "ehs"))
bp = load_blueprint(args.blueprint)
game = core.SearchGame(spec_to_dict(spec), core_bucketer(bk), bp.lookup)
rng = random.Random(args.seed)
n_roots = args.roots
t0 = time.time()
done = 0
while done < n_roots:
    target = rng.choice([Street.PREFLOP, Street.FLOP, Street.FLOP, Street.TURN, Street.RIVER])
    stacks = [rng.choice([200, 200, rng.randint(20, 400)]) * spec.bb for _ in range(2)]
    order = list(range(52))
    rng.shuffle(order)
    st = HandState(stacks, rng.randrange(2), spec.sb, spec.bb, 0, deck=Deck.from_order(order), max_street=spec.max_street)
    acts = []
    while not st.is_terminal and (st.street < target or (st.street == target and rng.random() < 0.5)):
        obs = st.observe(st.current_player)
        r = rng.random()
        if obs.can_raise and r < 0.25:
            a = raise_to(rng.randint(obs.min_raise_to, obs.max_raise_to))  # off-grid: an inserted size
        elif obs.can_raise and r < 0.35:
            a = raise_to(obs.max_raise_to)  # all-in
        else:
            names = [n for n in spec.grid.abstract_actions(obs) if n != "f"] or ["c"]
            a = spec.grid.to_concrete(obs, rng.choice(names))
        acts.append((int(a.type), int(a.amount)))
        st.apply(a)
    if st.is_terminal:
        continue
    obs = st.observe(st.current_player)
    depth = rng.choice(["pluribus", "pluribus", "end", "hu_flop_limit", "next_street"])
    iters = {0: 300, 1: 1500, 2: 5000, 3: 15000}[int(st.street)]
    timed = rng.random() < 0.3
    try:
        s = core.SubgameSearch(game, list(st.starting_stacks), st.button, acts, list(obs.board), obs.seat, list(obs.hole),
                               iterations=0 if timed else iters, time_budget=rng.uniform(0.05, 0.4) if timed else 0.0,
                               threads=rng.choice([2, 4, 8, 14]), seed=rng.randrange(1 << 30), depth=depth)
    except Exception as e:  # a root the search refuses (reported, not a crash)
        print("refused:", e, flush=True)
        continue
    r = s.solve()
    for seat in range(2):
        s.likelihood(seat)
    done += 1
    if done % 20 == 0:
        print(f"{done} roots, {time.time() - t0:.0f}s", flush=True)
print("OK", done)
