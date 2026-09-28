"""Vector Linear CFR in the subgame search (SearchParams::vector_cfr, csrc/search.h).

* where it runs: 2 live players, a turn or river root, no leaves, no frozen round (elsewhere the MCCFR);
* one thread and a fixed seed: the same numbers every time;
* river roots (no chance): the average strategy's exact exploitability falls with iterations and ends far
  below the blueprint's; with 4 threads too;
* turn roots (one river card sampled per iteration): the average beats the MCCFR's at equal iterations x work;
* our actual hole plays the action taken on the real path; the outputs are distributions.
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
pytestmark = pytest.mark.skipif(core is None, reason="C++ core not built (python scripts/build_fast.py)")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def trained():
    from negpluribus.fast.trainer import core_bucketer, spec_to_dict

    p = os.path.join(DATA, "buckets_3p_15bb_flop.json")
    bk = EquityBucketer.load(p) if os.path.exists(p) else EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)
    out = {}
    for players, iters in ((2, 6000), (3, 4000)):
        spec = GameSpec(n_players=players, stack_bb=30, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2,
                        preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
        t = MCCFRTrainer(spec, bk, seed=players, backend="cpp", threads=4).train(iters)
        out[players] = (spec, core.SearchGame(spec_to_dict(spec), core_bucketer(bk), t.blueprint().lookup))
    return out


def play(spec, line, seed=41):
    rng = random.Random(seed)
    order = list(range(52))
    rng.shuffle(order)
    st = spec.new_hand(order, button=0)
    acts = []
    for name in line:
        a = spec.grid.to_concrete(st.observe(st.current_player), name)
        acts.append((int(a.type), int(a.amount)))
        st.apply(a)
    return st, acts


def search(game, st, acts, **kw):
    obs = st.observe(st.current_player)
    params = dict(iterations=0, time_budget=0.0, threads=1, seed=1, depth="end")
    params.update(kw)
    return core.SubgameSearch(game, list(st.starting_stacks), st.button, acts, list(obs.board), obs.seat, list(obs.hole), **params)


RIVER = ["r1", "c", "c", "c", "c", "c"]           # SB opens, BB calls, checked to the river
TURN = ["r1", "c", "c", "c"]                      # ... to the turn
RIVER_BET = ["r1", "c", "c", "c", "c", "c", "c", "r1"]  # a pot bet on the river: BB decides


def test_vector_cfr_runs_only_where_it_applies(trained):
    spec2, game2 = trained[2]
    st, acts = play(spec2, RIVER)
    assert search(game2, st, acts, vector_cfr=True).vector_eligible
    assert not search(game2, st, acts).vector_eligible  # off by default
    st, acts = play(spec2, TURN)
    assert search(game2, st, acts, vector_cfr=True).vector_eligible
    assert not search(game2, st, acts, vector_cfr=True, depth="next_street").vector_eligible  # leaves: the MCCFR
    st, acts = play(spec2, ["r1", "c"])
    assert not search(game2, st, acts, vector_cfr=True).vector_eligible  # a flop root
    spec3, game3 = trained[3]
    st, acts = play(spec3, ["c", "c", "c", "c", "c", "c"])  # three live players on the turn
    assert st.street == Street.TURN
    assert not search(game3, st, acts, vector_cfr=True).vector_eligible


@pytest.mark.parametrize("line", [RIVER, TURN])
def test_one_thread_is_deterministic(trained, line):
    spec, game = trained[2]
    st, acts = play(spec, line)
    outs = []
    for _ in range(2):
        s = search(game, st, acts, iterations=30, vector_cfr=True, seed=5)
        r = s.solve()
        outs.append((r["final"], r["average"], r["iterations"], s.likelihood(0), s.likelihood(1)))
        assert abs(sum(r["final"]) - 1) < 1e-9 and abs(sum(r["average"]) - 1) < 1e-9
    assert outs[0] == outs[1]


def test_river_root_converges_far_below_the_blueprint(trained):
    spec, game = trained[2]
    st, acts = play(spec, RIVER_BET)
    assert st.street == Street.RIVER
    ex = {}
    for iters in (10, 300):
        s = search(game, st, acts, iterations=iters, vector_cfr=True)
        s.solve()
        ex[iters] = s.river_exploitability(0)
    blueprint = s.river_exploitability(2)
    pot_bb = s.root_info()["pot"] / spec.bb
    assert ex[300][0] < 0.5 * ex[10][0], ex
    assert ex[300][0] < 0.1 * blueprint[0], (ex, blueprint)
    assert ex[300][0] < 0.01 * pot_bb, (ex, pot_bb)
    assert all(v >= -1e-9 for v in ex[300][:3])
    # a river root runs on one thread whatever the thread count (no chance to spread): the same numbers
    s4 = search(game, st, acts, iterations=300, threads=4, vector_cfr=True)
    s4.solve()
    assert s4.river_exploitability(0) == ex[300]


def test_turn_root_beats_the_mccfr_at_equal_work(trained):
    """One vector iteration walks every hole pair; the MCCFR samples one per traversal.  Give the MCCFR
    1326 x the vector's iterations (more work than the vector's): the vector's average is still less exploitable."""
    spec, game = trained[2]
    st, acts = play(spec, TURN)
    assert st.street == Street.TURN
    sv = search(game, st, acts, iterations=200, vector_cfr=True)
    sv.solve()
    sm = search(game, st, acts, iterations=200 * 1326)
    sm.solve()
    ev, em = sv.subgame_exploitability(0, 2)[0], sm.subgame_exploitability(0, 2)[0]
    assert ev < em, (ev, em)


def test_our_actual_hole_plays_the_real_action(trained):
    spec, game = trained[2]
    st, acts = play(spec, RIVER_BET)
    obs = st.observe(st.current_player)
    ours = core.combo_index(*sorted(obs.hole))
    s = search(game, st, acts, iterations=50, vector_cfr=True)
    s.solve()
    checked = 0
    for k, step in enumerate(s.path()):
        if step["actor"] != obs.seat:
            continue
        row = s.path_strategies(k, 0)[ours]
        assert row[step["index"]] == 1.0
        checked += 1
    assert checked >= 1


# ---- river_exact: a turn root's river infosets by hand strength instead of the blueprint's buckets

def test_river_exact_class_is_the_strength_on_the_board(trained):
    spec, game = trained[2]
    st, acts = play(spec, TURN)
    s = search(game, st, acts, iterations=3, vector_cfr=True, river_exact=True)
    s.solve()
    board = list(st.observe(st.current_player).board)
    classes = s.river_classes()
    rivers = [r for r in range(52) if r not in board]
    assert all(classes[r] > 0 for r in rivers) and all(classes[c] == 0 for c in board)
    combos = [(a, b) for a in range(52) for b in range(a + 1, 52)]
    for r in rivers[:6]:
        b5 = board + [r]
        seen = {}
        for i, (a, b) in enumerate(combos):
            k = s.river_class(r, i)
            if a in b5 or b in b5:
                assert k == -1
                continue
            strength = core.evaluate([a, b] + b5)
            seen.setdefault(strength, set()).add(k)
        # one class per strength, classes in the order of the strengths, 0 .. n - 1
        assert all(len(v) == 1 for v in seen.values())
        ks = [next(iter(seen[x])) for x in sorted(seen)]
        assert ks == list(range(len(ks))) and classes[r] == len(ks)


def test_river_exact_one_thread_is_deterministic_and_off_elsewhere(trained):
    spec, game = trained[2]
    st, acts = play(spec, TURN)
    outs = []
    for _ in range(2):
        s = search(game, st, acts, iterations=20, vector_cfr=True, river_exact=True, seed=4)
        r = s.solve()
        outs.append((r["final"], r["average"], s.likelihood(0), s.likelihood(1), s.subgame_exploitability(0, 2)))
    assert outs[0] == outs[1]
    # a river root is lossless already: river_exact changes nothing there
    st, acts = play(spec, RIVER_BET)
    a = search(game, st, acts, iterations=30, vector_cfr=True, river_exact=True)
    b = search(game, st, acts, iterations=30, vector_cfr=True)
    ra, rb = a.solve(), b.solve()
    assert (ra["final"], ra["average"]) == (rb["final"], rb["average"]) and a.river_bytes == 0


def test_river_exact_lowers_the_turn_floor(trained):
    """With the river by buckets the turn subgame's exploitability stops at the abstraction's floor (~0.5-0.65 here,
    not falling from 600 to 12000 iterations); with the exact river it keeps falling (0.2 at 3000, 0.04 at 12000).
    Each river card's rows learn only in the iterations that deal it, so the exact river starts slower."""
    spec, game = trained[2]
    st, acts = play(spec, TURN)
    ex = {}
    for exact in (False, True):
        s = search(game, st, acts, iterations=3000, threads=4, vector_cfr=True, river_exact=exact)
        s.solve()
        ex[exact] = s.subgame_exploitability(0, 2)[0]
    assert ex[True] < 0.5 * ex[False], ex
