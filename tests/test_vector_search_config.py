"""The flags of the next search experiments (E10 = E2 + --search-vector with vector iterations per street, E11 = E10 +
--search-river-buckets 200 --search-vector-discount dcfr; docs/search_vs_blueprint.md, handoff 28.09 22:40) from the
command line to the solver, and the vector CFR's discount variants, which no test ran before.

  * parse_street_budgets / add_search_args / search_config_from_args: every flag lands in SearchConfig as the
    agent reads it (street numbers, integer iterations, the discount code), and the defaults stay off;
  * the agent: a vector-eligible root (turn / river, heads-up, no leaves) solves exactly the vector iterations of its
    street, a flop root the MCCFR's fixed iterations (a reproducible run must not fall back to the clock);
  * vector_discount 1 (CFR+) and 2 (DCFR 1.5/0/2): valid strategies, deterministic on one thread, other numbers than
    Linear CFR, and a river root's exploitability falls with the iterations as Linear's does.
Tiny HU 30bb game (fixed E[HS] cut points, a 1-thread C++ blueprint), one thread everywhere.
"""
from __future__ import annotations

import argparse
import random

import pytest

from negpluribus import fast
from negpluribus.engine import Street

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "SubgameSearch"), reason="C++ core with the search not built")

TURN = ["r1", "c", "c", "c"]                            # SB opens, BB calls, checked to the turn: SB decides
RIVER_BET = ["r1", "c", "c", "c", "c", "c", "c", "r1"]  # a pot bet on the river: BB decides
FLOP = ["r1", "c"]


def test_street_budgets_parse_names_to_street_numbers():
    from negpluribus.agents.core_search import parse_street_budgets

    assert parse_street_budgets("turn=300, river=600") == {2: 300.0, 3: 600.0}
    assert parse_street_budgets("PREFLOP=2,flop=1.5") == {0: 2.0, 1: 1.5}
    assert parse_street_budgets("") == {} and parse_street_budgets(None) == {}
    for bad in ("turn", "turn=", "fourth=1", "turn:3", "turn=x"):
        with pytest.raises(ValueError):
            parse_street_budgets(bad)


def _config(*argv):
    from negpluribus.agents.core_search import add_search_args, search_config_from_args

    ap = argparse.ArgumentParser()
    add_search_args(ap)
    return search_config_from_args(ap.parse_args(list(argv)))


def test_the_e10_e11_flags_reach_the_search_config():
    cfg = _config("--search-vector", "--search-vector-iterations", "turn=300,river=600", "--search-river-buckets", "200",
                  "--search-vector-discount", "dcfr", "--search-street-iterations", "flop=1400000",
                  "--search-play", "average", "--search-threads", "1")
    assert cfg.vector_cfr and cfg.vector_street_iterations == {2: 300, 3: 600}
    assert all(type(v) is int for v in cfg.vector_street_iterations.values())
    assert cfg.street_iterations == {1: 1_400_000} and type(cfg.street_iterations[1]) is int
    assert cfg.river_buckets == 200 and cfg.vector_discount == 2 and not cfg.river_exact
    assert cfg.threads == 1 and cfg.play == "average" and cfg.iters(Street.FLOP) == 1_400_000
    assert [_config("--search-vector-discount", d).vector_discount for d in ("linear", "cfr+", "dcfr")] == [0, 1, 2]
    assert _config("--search-river-exact").river_exact


def test_search_defaults_keep_the_new_options_off():
    cfg = _config()
    assert not cfg.vector_cfr and cfg.vector_street_iterations == {} and cfg.vector_discount == 0
    assert cfg.river_buckets == 0 and not cfg.river_exact
    assert cfg.time_budget == 2.0 and cfg.iterations == 0 and cfg.preflop_offgrid == 0.3 and cfg.preflop_unknown
    assert cfg.search_from_street == 1 and cfg.search_to_street == 3 and cfg.depth == "pluribus"
    off = _config("--no-preflop-search")
    assert off.preflop_offgrid == float("inf") and not off.preflop_unknown


@pytest.fixture(scope="module")
def game():
    from negpluribus.abstraction import EquityBucketer
    from negpluribus.agents.core_search import SearchResources
    from negpluribus.cfr.game import GameSpec
    from negpluribus.cfr.mccfr import MCCFRTrainer

    bk = EquityBucketer(n_buckets=4, samples=30)
    bk.boundaries = {1: [0.35, 0.5, 0.65], 2: [0.35, 0.5, 0.65], 3: [0.35, 0.5, 0.65]}
    spec = GameSpec(n_players=2, stack_bb=30, max_street=Street.RIVER, n_buckets=4, max_raises_per_street=2,
                    preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    bp = MCCFRTrainer(spec, bk, seed=2, backend="cpp", threads=1).train(3000).blueprint()
    return spec, SearchResources.build(spec, bk, bp, cache_caps=(1 << 16,) * 3)


def _line(spec, names, seed=41):
    rng = random.Random(seed)
    order = list(range(52))
    rng.shuffle(order)
    st = spec.new_hand(order, button=0)
    for name in names:
        st.apply(spec.grid.to_concrete(st.observe(st.current_player), name))
    return st


@pytest.mark.parametrize("line,street,want", [(TURN, Street.TURN, 7), (RIVER_BET, Street.RIVER, 9), (FLOP, Street.FLOP, 50)])
def test_the_agent_solves_the_vector_iterations_of_the_street(game, line, street, want):
    from negpluribus.agents.core_search import CoreSearchAgent, SearchConfig

    spec, res = game
    cfg = SearchConfig(iterations=50, threads=1, vector_cfr=True, vector_street_iterations={2: 7, 3: 9},
                       river_buckets=6, vector_discount=2)
    hero = CoreSearchAgent(res, cfg, seed=1)
    st = _line(spec, line)
    assert st.street == street
    hero.act(st.observe(st.current_player))
    last = hero.last
    assert last["street"] == street and last["result"]["iterations"] == want
    assert last["search"].vector_eligible == (street >= Street.TURN)
    assert not hero.stats.errors


def _search(game, st, **kw):
    spec, res = game
    obs = st.observe(st.current_player)
    acts = [(int(e.action.type), int(e.action.amount)) for e in st.events]
    params = dict(iterations=0, time_budget=0.0, threads=1, seed=1, depth="end", vector_cfr=True)
    params.update(kw)
    return core.SubgameSearch(res.game, list(st.starting_stacks), st.button, acts, list(obs.board), obs.seat, list(obs.hole),
                              **params)


@pytest.mark.parametrize("discount", [1, 2])
def test_cfr_plus_and_dcfr_are_deterministic_valid_and_converge_on_a_river_root(game, discount):
    spec, _ = game
    st = _line(spec, RIVER_BET)
    assert st.street == Street.RIVER
    runs = []
    for _ in range(2):
        s = _search(game, st, iterations=40, vector_discount=discount, seed=5)
        r = s.solve()
        assert abs(sum(r["final"]) - 1) < 1e-9 and abs(sum(r["average"]) - 1) < 1e-9
        assert all(p >= 0 for p in r["final"] + r["average"])
        runs.append((r["final"], r["average"], s.likelihood(0), s.likelihood(1)))
    assert runs[0] == runs[1]
    linear = _search(game, st, iterations=40, vector_discount=0, seed=5).solve()
    assert (linear["final"], linear["average"]) != runs[0][:2]  # another weighting, other numbers
    ex = {}
    for iters in (10, 300):
        s = _search(game, st, iterations=iters, vector_discount=discount)
        s.solve()
        ex[iters] = s.river_exploitability(0)[0]
    assert 0 <= ex[300] < 0.5 * ex[10], ex


@pytest.mark.parametrize("discount", [
    0,
    1,
    2,  # H2 fixed 29.09 (a66d70c): the played average is the table's average under every discount
])
def test_the_played_average_is_the_solvers_own_average(game, discount):
    """r["average"] (what CoreSearchAgent samples with play="average") must be the average strategy the solver's table
    holds for our hole at the root (r["average_table"]), for every vector discount.  Measured on this game's river root
    at 500 iterations: Linear and CFR+ agree to 1e-3; DCFR's differ by 0.01-0.04 (0.05 on the turn root)."""
    spec, _ = game
    st = _line(spec, RIVER_BET)
    r = _search(game, st, iterations=500, vector_discount=discount).solve()
    gap = max(abs(a - b) for a, b in zip(r["average"], r["average_table"]))
    assert gap < 0.005, (gap, r["average"], r["average_table"])


@pytest.mark.parametrize("discount", [1, 2])
def test_cfr_plus_and_dcfr_lower_a_turn_root_below_the_blueprint(game, discount):
    spec, _ = game
    st = _line(spec, TURN)
    s = _search(game, st, iterations=200, vector_discount=discount)
    s.solve()
    after = s.subgame_exploitability(0, 2)[0]
    blueprint = s.subgame_exploitability(2, 2)[0]
    assert 0 <= after < blueprint, (after, blueprint)
