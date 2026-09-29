"""Regret-based pruning in the subgame search (SearchParams::prune_mode; type b, off by default).

* off: the search is the one without the option (the bit-for-bit gate is search_bench --compare, 12/12);
* on: one thread and a fixed seed give the same numbers every time; actions are pruned; the outputs are
  distributions; our actual hole's decision on the real path is never pruned (the forced / focused path);
* every mode (absolute, x t, relative) and the window (prune_start / prune_stop) take effect; bad values are refused.
"""
from __future__ import annotations

import os
import random

import pytest

from negpluribus import fast
from negpluribus.abstraction import EquityBucketer
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import Street

core = fast.core()
pytestmark = pytest.mark.skipif(core is None, reason="C++ core not built")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def game():
    from negpluribus.fast.trainer import core_bucketer, spec_to_dict

    p = os.path.join(DATA, "buckets_3p_15bb_flop.json")
    bk = EquityBucketer.load(p) if os.path.exists(p) else EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)
    spec = GameSpec(n_players=2, stack_bb=30, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2,
                    preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    t = MCCFRTrainer(spec, bk, seed=2, backend="cpp", threads=1).train(4000)
    return spec, core.SearchGame(spec_to_dict(spec), core_bucketer(bk), t.blueprint().lookup)


def search(spec, game, line, **kw):
    rng = random.Random(41)
    order = list(range(52))
    rng.shuffle(order)
    st = spec.new_hand(order, button=0)
    acts = []
    for name in line:
        a = spec.grid.to_concrete(st.observe(st.current_player), name)
        acts.append((int(a.type), int(a.amount)))
        st.apply(a)
    obs = st.observe(st.current_player)
    params = dict(iterations=3000, time_budget=0.0, threads=1, seed=1, depth="end")
    params.update(kw)
    return core.SubgameSearch(game, list(st.starting_stacks), st.button, acts, list(obs.board), obs.seat, list(obs.hole), **params)


FLOP = ["r1", "c"]
TURN = ["r1", "c", "c", "c"]


@pytest.mark.parametrize("mode,below", [(1, 20.0), (2, 0.5), (3, 0.1)])
def test_pruning_is_deterministic_on_one_thread_and_prunes(game, mode, below):
    spec, g = game
    a = search(spec, g, FLOP, prune_mode=mode, prune_below=below).solve()
    b = search(spec, g, FLOP, prune_mode=mode, prune_below=below).solve()
    assert a["final"] == b["final"] and a["average"] == b["average"] and a["pruned"] == b["pruned"]
    assert a["pruned"] > 0
    off = search(spec, g, FLOP).solve()
    assert off["pruned"] == 0 and a["nodes_touched"] < off["nodes_touched"]
    for r in (a, off):
        assert abs(sum(r["average"]) - 1.0) < 1e-9 and abs(sum(r["final"]) - 1.0) < 1e-9


def test_window_and_values(game):
    spec, g = game
    none = search(spec, g, TURN, prune_mode=3, prune_below=0.1, prune_start=0.5, prune_stop=0.5).solve()
    assert none["pruned"] == 0  # an empty window: never prunes
    late = search(spec, g, TURN, prune_mode=3, prune_below=0.1, prune_start=0.5, prune_stop=0.0).solve()
    full = search(spec, g, TURN, prune_mode=3, prune_below=0.1, prune_start=0.0, prune_stop=0.0).solve()
    assert 0 < late["pruned"] < full["pruned"]
    with pytest.raises(ValueError):
        search(spec, g, TURN, prune_mode=4, prune_below=1.0)
    with pytest.raises(ValueError):
        search(spec, g, TURN, prune_mode=3, prune_below=0.0)
    with pytest.raises(ValueError):
        search(spec, g, TURN, prune_mode=3, prune_below=0.1, prune_start=0.7, prune_stop=0.5)


def test_pruning_with_threads_gives_distributions(game):
    spec, g = game
    r = search(spec, g, TURN, prune_mode=3, prune_below=0.1, threads=4, iterations=0, time_budget=0.5).solve()
    assert r["pruned"] > 0 and abs(sum(r["average"]) - 1.0) < 1e-9
