"""The search's public-tree traversal (csrc/search.h, traverse_tree) against the engine traversal it replaces
(legacy_traverse=True): the same search bit for bit - final and average strategies, every seat's likelihood,
table size and node / leaf / rollout counts - on random hands of 2 and 3 players with unequal stacks and antes
(all-ins, side pots, split pots), every root street, and the depth rules with leaves."""
from __future__ import annotations

import os
import random

import pytest

from negpluribus import fast
from negpluribus.abstraction import EquityBucketer
from negpluribus.cards import Deck
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import HandState, Street, raise_to

core = fast.core()
pytestmark = pytest.mark.skipif(core is None, reason="C++ core not built (python scripts/build_fast.py)")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def games():
    from negpluribus.fast.trainer import core_bucketer, spec_to_dict

    p = os.path.join(DATA, "buckets_3p_15bb_flop.json")
    bk = EquityBucketer.load(p) if os.path.exists(p) else EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)
    out = {}
    for players, ante in ((2, 0), (3, 10)):
        spec = GameSpec(n_players=players, stack_bb=30, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2,
                        preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0), ante=ante)
        t = MCCFRTrainer(spec, bk, seed=players, backend="cpp", threads=4).train(3000)
        out[players] = (spec, core.SearchGame(spec_to_dict(spec), core_bucketer(bk), t.blueprint().lookup))
    return out


def random_hand(spec, rng, target, stacks):
    grid = spec.grid
    for _ in range(2000):
        order = list(range(52))
        rng.shuffle(order)
        st = HandState(stacks, rng.randrange(spec.n_players), spec.sb, spec.bb, spec.ante, deck=Deck.from_order(order),
                       max_street=spec.max_street)
        acts = []
        while not st.is_terminal and st.street < target:
            obs = st.observe(st.current_player)
            if obs.can_raise and rng.random() < 0.2:
                a = raise_to(rng.randint(obs.min_raise_to, obs.max_raise_to))
            else:
                names = [n for n in grid.abstract_actions(obs) if n != "f" or rng.random() < 0.2]
                a = grid.to_concrete(obs, rng.choice(names))
            acts.append((int(a.type), int(a.amount)))
            st.apply(a)
        if not st.is_terminal and st.street == target:
            return st, acts
    raise RuntimeError("no hand reached the street")


def run(game, st, acts, **kw):
    obs = st.observe(st.current_player)
    s = core.SubgameSearch(game, list(st.starting_stacks), st.button, acts, list(obs.board), obs.seat, list(obs.hole),
                           time_budget=0.0, threads=1, seed=3, **kw)
    r = s.solve()
    lik = [s.likelihood(seat) if not st.players[seat].folded else None for seat in range(st.n)]
    keep = ("final", "average", "iterations", "table_size", "nodes_touched", "forced", "leaves", "leaf_evals", "rollouts", "rollout_steps")
    return {k: r[k] for k in keep}, lik


@pytest.mark.parametrize("players", [2, 3])
def test_public_tree_traversal_is_the_engine_traversal(games, players):
    spec, game = games[players]
    rng = random.Random(100 + players)
    n_cases = 0
    for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER):
        for depth in ("end", "pluribus", "hu_flop_limit"):
            if street == Street.PREFLOP and depth == "end":
                continue  # a whole hand per iteration: slow and nothing new
            for _ in range(2):
                stacks = [rng.randint(6, 40) * spec.bb for _ in range(players)]  # unequal: all-ins, side pots
                st, acts = random_hand(spec, rng, street, stacks)
                iters = 300 if street <= Street.FLOP else 1500
                a = run(game, st, acts, iterations=iters, depth=depth)
                b = run(game, st, acts, iterations=iters, depth=depth, legacy_traverse=True)
                assert a == b, (players, street, depth, stacks, acts)
                n_cases += 1
    assert n_cases >= 20
