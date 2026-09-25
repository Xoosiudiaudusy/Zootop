"""Acceptance tests for the fast backends (docs/backends.md).

Everything that needs the C++ core (``negpluribus._fastcore``) is skipped automatically when it
is not built (``python scripts/build_fast.py``); the multiprocess driver and the phevaluator
tier are tested whenever they are available.
"""
from __future__ import annotations

import os
import random

import pytest

from negpluribus import fast
from negpluribus.abstraction import EquityBucketer, infoset_key
from negpluribus.cards import ALL_HOLE_CLASSES, Deck
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import ActionType, HandState, Street, raise_to
from negpluribus.equity import equity_vs_random, equity_vs_random_py
from negpluribus.evaluator import evaluate_py

core = fast.core()
needs_core = pytest.mark.skipif(core is None, reason="C++ core not built (python scripts/build_fast.py)")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CLASS_EQUITY = os.path.join(DATA, "class_equity.json")
needs_equity_table = pytest.mark.skipif(not os.path.exists(CLASS_EQUITY), reason="needs data/class_equity.json")


def pushfold_spec(stack=10):
    return GameSpec(n_players=2, stack_bb=stack, max_street=Street.PREFLOP, preflop_fracs=(), forbid_open_limp=True)


@pytest.fixture(scope="module")
def flop_bucketer():
    p = os.path.join(DATA, "buckets_3p_15bb_flop.json")
    if os.path.exists(p):
        return EquityBucketer.load(p)
    return EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)


# =========================================================== 1. evaluator / equity
@needs_core
def test_cpp_evaluator_identical_on_100k_hands():
    rng = random.Random(20260923)
    for _ in range(100_000):
        cards = rng.sample(range(52), rng.choice((5, 6, 7)))
        assert core.evaluate(cards) == evaluate_py(cards), cards
    with pytest.raises(ValueError):
        core.evaluate([0, 1, 2, 3])


@pytest.mark.skipif(pytest.importorskip("phevaluator", reason="phevaluator not installed") is None, reason="")
def test_phevaluator_tier_identical_on_100k_hands():
    from phevaluator._pheval import evaluate_5cards, evaluate_6cards, evaluate_7cards

    from negpluribus.fast.evaluator import _phe_table

    table = _phe_table(evaluate_py, evaluate_5cards)
    assert len(table) == 7462 and len(set(table)) == 7462
    fns = {5: evaluate_5cards, 6: evaluate_6cards, 7: evaluate_7cards}
    rng = random.Random(7)
    for _ in range(100_000):
        cards = rng.sample(range(52), rng.choice((5, 6, 7)))
        assert table[fns[len(cards)](*cards) - 1] == evaluate_py(cards), cards


@needs_core
def test_cpp_equity_bit_identical_with_seeded_rng():
    """Same random.Random state in -> same equity and same generator state out."""
    for i in range(200):
        r = random.Random(500 + i)
        cards = r.sample(range(52), 7)
        n_board = r.choice((0, 3, 4, 5))
        hole, board = cards[:2], cards[2:2 + n_board]
        n_opp, samples = r.choice((1, 2, 3)), r.choice((20, 100, 200))
        a_rng, b_rng = random.Random(i), random.Random(i)
        a = equity_vs_random_py(hole, board, n_opp, samples, a_rng)
        b = equity_vs_random(hole, board, n_opp, samples, b_rng)
        assert a == b and a_rng.getstate() == b_rng.getstate()
    # unseeded: statistically the same thing (Monte-Carlo noise only)
    hole, board = [51, 47], [3, 20, 33]
    fast_eq = sum(equity_vs_random(hole, board, 1, 2000) for _ in range(5)) / 5
    slow_eq = equity_vs_random_py(hole, board, 1, 2000, random.Random(1))
    assert abs(fast_eq - slow_eq) < 0.03


def _sample_setsize(k: int) -> int:
    """CPython's random.sample threshold: the pool branch when n <= setsize, else the set branch."""
    import math

    return 21 + (4 ** math.ceil(math.log(k * 3, 4)) if k > 5 else 0)


@needs_core
def test_cpp_sample_matches_cpython_random_sample():
    """PyRandom::sample (stack pool / 64-bit "selected" mask up to 64 items, heap beyond) draws
    exactly what random.Random.sample draws: both CPython branches, several consecutive calls
    from one generator, and the same generator state afterwards (same number of MT words)."""
    if not hasattr(core, "random_sample_seq"):
        pytest.skip("the built core predates random_sample_seq; run `python scripts/build_fast.py`")
    gen = random.Random(20260924)
    branches = set()
    cases = 0
    for n in (0, 1, 2, 5, 13, 21, 22, 30, 45, 46, 47, 50, 52, 63, 64, 65, 100, 277, 278, 300):
        for k in sorted({0, 1, 2, 3, 4, 5, 6, 7, 10, 15, 21, 22, 30, 64, 65, n // 2, n}):
            if k > n:
                continue
            pool = n <= _sample_setsize(k)
            branches.add(("pool" if pool else "set", "stack" if n <= 64 else "heap"))
            population = gen.sample(range(100_000), n)
            for _ in range(3):
                seed = gen.getrandbits(64)
                ref = random.Random(seed)
                want = [ref.sample(population, k) for _ in range(4)]
                got, state = core.random_sample_seq(seed, population, k, 4)
                assert [list(s) for s in got] == want, (n, k, seed)
                assert tuple(state) == ref.getstate()[1], (n, k, seed)
                cases += 1
    assert branches == {("pool", "stack"), ("pool", "heap"), ("set", "stack"), ("set", "heap")}
    assert cases > 500
    # the shapes the bucketers draw (deck of 47/46/45 cards, 4/3/2 cards: the set branch), many seeds
    for n, k in ((47, 4), (46, 3), (45, 2), (50, 15)):
        deck = list(range(52 - n, 52))
        for seed in range(300):
            ref = random.Random(seed)
            got, state = core.random_sample_seq(seed, deck, k, 20)
            assert [list(s) for s in got] == [ref.sample(deck, k) for _ in range(20)], (n, k, seed)
            assert tuple(state) == ref.getstate()[1]
    # k outside 0..n raises like CPython (the port used to loop forever)
    for bad_k in (4, -1):
        with pytest.raises(ValueError):
            random.Random(0).sample([1, 2, 3], bad_k)
        with pytest.raises(ValueError):
            core.random_sample(0, [1, 2, 3], bad_k)
        with pytest.raises(ValueError):
            core.random_sample_seq(0, [1, 2, 3], bad_k, 1)


@needs_core
def test_cpp_equity_on_complete_boards_matches_python():
    """On a complete 5-card board the port evaluates our hand once instead of once per sample:
    the same floats and the same generator state as the reference for 1..5 opponents, including
    boards that play (every showdown a tie, the 1/(ties+1) split)."""
    from negpluribus.cards import cards_from_str as cs

    rng = random.Random(24092026)
    spots = []
    for _ in range(100):
        cards = rng.sample(range(52), 7)
        spots.append((cards[:2], cards[2:]))
    spots += [
        (cs("2c 7d"), cs("As Ks Qs Js Ts")),  # royal flush on board: everybody ties
        (cs("3h 4h"), cs("2c 2d 2h 2s Ac")),  # quads + ace kicker on board: everybody ties
        (cs("Ah Kd"), cs("Qc Jd Th 3s 3c")),  # the nut straight: ties with any other ace-king
    ]
    n_cases = 0
    for i, (hole, board) in enumerate(spots):
        for n_opp in (1, 2, 3, 5):
            for samples in (1, 10, 100):
                seed = 1000 * i + 10 * n_opp + samples
                a_rng, b_rng = random.Random(seed), random.Random(seed)
                a = equity_vs_random_py(hole, board, n_opp, samples, a_rng)
                b = equity_vs_random(hole, board, n_opp, samples, b_rng)
                c = core.equity_vs_random_seeded(hole, board, n_opp, samples, seed)
                assert a == b == c and a_rng.getstate() == b_rng.getstate(), (hole, board, n_opp, samples)
                n_cases += 1
    assert n_cases == 4 * 3 * len(spots)
    # the royal flush on board: every showdown a tie, 1 / (n + 1) per sample (up to the float sum)
    for n_opp in (1, 2, 3, 5):
        eq = equity_vs_random(cs("2c 7d"), cs("As Ks Qs Js Ts"), n_opp, 50, random.Random(n_opp))
        assert eq == pytest.approx(1.0 / (n_opp + 1), abs=1e-12)


@needs_core
def test_cpp_ehs_and_buckets_match_python_bucketer(flop_bucketer):
    cbk = core.Bucketer(flop_bucketer.n_buckets, flop_bucketer.samples, dict(flop_bucketer.boundaries))
    rng = random.Random(3)
    pyb = EquityBucketer(flop_bucketer.n_buckets, flop_bucketer.samples)
    pyb.boundaries = dict(flop_bucketer.boundaries)
    for _ in range(300):
        cards = rng.sample(range(52), 7)
        n_board = rng.choice((3, 4, 5))
        hole, board = cards[:2], cards[2:2 + n_board]
        assert cbk.ehs(hole, board) == pyb.ehs(hole, board)
        assert cbk.bucket(hole, board) == pyb.bucket(hole, board)
    assert cbk.cache_size() == len(pyb._cache)


# ======================================================================= 2. engine
def _random_hand_params(rng, n):
    short = rng.random() < 0.5
    stacks = [rng.choice([250, 600, 1500, 10_000]) if short else 10_000 for _ in range(n)]
    order = list(range(52))
    rng.shuffle(order)
    return stacks, order, rng.randrange(n), rng.choice([Street.PREFLOP, Street.FLOP, Street.RIVER]), rng.choice([0, 0, 0, 10])


@needs_core
@pytest.mark.parametrize("n", [2, 3, 6])
def test_cpp_engine_matches_python_on_random_hands(n):
    hands = 0
    for i in range(2000 // 3 + 1):
        rng = random.Random(n * 100_000 + i)
        stacks, order, button, max_street, ante = _random_hand_params(rng, n)
        h = HandState(stacks, button, 50, 100, ante, deck=Deck.from_order(order), max_street=max_street)
        c = core.Hand(stacks, button, 50, 100, ante, order, int(max_street))
        while not h.is_terminal:
            seat = h.current_player
            assert c.current_player == seat and not c.is_terminal
            obs, co = h.observe(seat), c.observe(seat)
            for k in ("pot", "to_call", "stack", "min_raise_to", "max_raise_to", "can_raise", "can_fold",
                      "raises_this_street", "n_active", "stacks", "street_bets", "folded", "all_in", "board", "hole"):
                assert getattr(obs, k) == co[k], (k, getattr(obs, k), co[k])
            assert obs.position == co["position"] and int(obs.street) == co["street"]
            a = rng.choice(obs.legal_actions())
            if a.type == ActionType.RAISE and rng.random() < 0.5:
                a = raise_to(rng.randint(obs.min_raise_to, obs.max_raise_to))
            ev, cev = h.apply(a), c.apply(int(a.type), a.amount)
            assert (int(ev.street), ev.seat, int(ev.action.type), ev.action.amount, ev.to_call, ev.pot_before,
                    ev.facing_raise, ev.raises_this_street, ev.paid, ev.all_in, ev.stack_after) == tuple(
                cev[k] for k in ("street", "seat", "type", "amount", "to_call", "pot_before", "facing_raise",
                                 "raises_this_street", "paid", "all_in", "stack_after"))
        assert c.is_terminal
        r, cr = h.record(), c.record()
        assert r.net == cr["net"]
        assert r.showdown_seats == cr["showdown_seats"]
        assert r.winners == cr["winners"] and r.board == cr["board"] and r.saw_flop == cr["saw_flop"]
        assert len(r.events) == cr["n_events"]
        hands += 1
    assert hands >= 667


# ========================================================================= 3. keys
@needs_core
def test_cpp_infoset_keys_equal_python_keys(flop_bucketer):
    from negpluribus.fast.trainer import core_bucketer, spec_to_dict

    specs = [
        GameSpec(n_players=3, stack_bb=15, max_street=Street.FLOP, n_buckets=8),
        GameSpec(n_players=2, stack_bb=20, max_street=Street.FLOP, n_buckets=8),
        GameSpec(n_players=6, stack_bb=25, max_street=Street.RIVER, n_buckets=8, preflop_fracs=(0.5, 1.0), postflop_fracs=(0.33, 0.75, 1.5)),
        pushfold_spec(),
    ]
    # fresh caches on both sides: the E[HS] Monte-Carlo runs on the first representative seen
    pyb = EquityBucketer(flop_bucketer.n_buckets, flop_bucketer.samples)
    pyb.boundaries = dict(flop_bucketer.boundaries)
    cbk = core_bucketer(pyb)
    n_keys = 0
    for si, spec in enumerate(specs):
        game = core.Game(spec_to_dict(spec), cbk)
        grid = spec.grid
        for i in range(60):
            rng = random.Random(si * 1000 + i)
            order = list(range(52))
            rng.shuffle(order)
            button = rng.randrange(spec.n_players)
            h, c = spec.new_hand(order, button), game.new_hand(order, button)
            while not h.is_terminal:
                obs = h.observe(h.current_player)
                legal = grid.abstract_actions(obs)
                assert c.legal() == legal
                assert c.key() == infoset_key(obs, pyb, grid)
                a = rng.choice(legal)
                act = grid.to_concrete(obs, a)
                assert c.concrete(a) == (int(act.type), act.amount)
                h.apply(act)
                c.apply_name(a)
                n_keys += 1
    assert n_keys >= 500


# ================================================================ 4. blueprint quality
def _pushfold_tables(trainer):
    def prob(pos, cls, action, hist=""):
        node = trainer.nodes.get(f"P|{pos}|2|b{ALL_HOLE_CLASSES.index(cls)}|{hist}")
        return 0.0 if node is None else dict(zip(node.actions, node.average_strategy())).get(action, 0.0)

    push = {c: prob("BTN/SB", c, "a") for c in ALL_HOLE_CLASSES}
    call = {c: prob("BB", c, "c", "a") for c in ALL_HOLE_CLASSES}
    return push, call


@needs_core
@needs_equity_table
def test_cpp_pushfold_blueprint_reaches_low_exploitability_and_matches_python():
    from negpluribus.cfr.exploit import ClassEquity, exact_exploitability

    equity = ClassEquity.load(CLASS_EQUITY)
    spec = pushfold_spec()
    cpp = MCCFRTrainer(spec, seed=0, backend="cpp").train(100_000)  # all cores, < 1 s
    res = exact_exploitability(spec, cpp.strategy(), equity)
    assert res.bb100 <= 6.0, str(res)
    py = MCCFRTrainer(spec, seed=0, backend="python").train(100_000)  # ~15 s
    res_py = exact_exploitability(spec, py.strategy(), equity)
    push_c, call_c = _pushfold_tables(cpp)
    push_p, call_p = _pushfold_tables(py)
    d_push = sum(abs(push_c[c] - push_p[c]) for c in ALL_HOLE_CLASSES) / 169
    d_call = sum(abs(call_c[c] - call_p[c]) for c in ALL_HOLE_CLASSES) / 169
    # Not seed-identical (threads > 1).  Measured 2026-09-23 at 100k iterations: cpp x16 vs python
    # 0.160 / 0.144 (push / call), while python seed 0 vs python seed 1 is 0.172 / 0.147 - the
    # backend difference is the algorithm's own seed-to-seed noise.  Tolerance = that noise + margin.
    assert d_push < 0.2 and d_call < 0.2, (d_push, d_call, res.bb100, res_py.bb100)
    assert abs(res.bb100 - res_py.bb100) < 3.0, (res.bb100, res_py.bb100)
    # with a bigger budget the tables tighten: 1M iterations, 1 thread (== python) vs all cores
    one = MCCFRTrainer(spec, seed=0, backend="cpp", threads=1).train(1_000_000)
    many = MCCFRTrainer(spec, seed=0, backend="cpp").train(1_000_000)
    p1, c1 = _pushfold_tables(one)
    pm, cm = _pushfold_tables(many)
    assert sum(abs(p1[c] - pm[c]) for c in ALL_HOLE_CLASSES) / 169 < 0.15  # measured 0.102
    assert sum(abs(c1[c] - cm[c]) for c in ALL_HOLE_CLASSES) / 169 < 0.10  # measured 0.054
    assert exact_exploitability(spec, many.strategy(), equity).bb100 < 1.5  # measured 0.49 (python/x1: 0.74)


# ================================================== 5. flag, equivalence, checkpoints
@needs_core
def test_single_thread_cpp_trainer_is_bit_identical_to_python(flop_bucketer):
    for spec, bk, iters in (
        (pushfold_spec(), None, 2000),
        (GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,)), None, 1500),
        (GameSpec(n_players=3, stack_bb=15, max_street=Street.FLOP, n_buckets=8), flop_bucketer, 200),
    ):
        py = MCCFRTrainer(spec, bk, seed=4, backend="python").train(iters)
        cpp = MCCFRTrainer(spec, bk, seed=4, backend="cpp", threads=1).train(iters)
        assert cpp.iteration == py.iteration and cpp.nodes_touched == py.nodes_touched
        assert set(cpp.nodes) == set(py.nodes)
        for k, n in py.nodes.items():
            m = cpp.nodes[k]
            assert (m.actions, m.regret, m.strategy_sum, m.visits) == (n.actions, n.regret, n.strategy_sum, n.visits), k
        assert cpp.strategy().table == py.strategy().table


@needs_core
def test_backend_flag_and_env_var(monkeypatch):
    from negpluribus.exploit.rnr import RNRTrainer
    from negpluribus.fast.trainer import CppMCCFRTrainer

    spec = pushfold_spec()
    assert type(MCCFRTrainer(spec)) is MCCFRTrainer
    assert type(MCCFRTrainer(spec, backend="python")) is MCCFRTrainer
    t = MCCFRTrainer(spec, backend="cpp", threads=2)
    assert type(t) is CppMCCFRTrainer and t.backend == "cpp" and t.threads == 2
    monkeypatch.setenv("NEGPLURIBUS_BACKEND", "cpp")
    assert type(MCCFRTrainer(spec)) is CppMCCFRTrainer
    assert type(MCCFRTrainer(spec, backend="python")) is MCCFRTrainer
    # RNRTrainer follows the same flag when its model is tabular (None / OpponentModel / BlueprintStrategy)
    from negpluribus.fast.rnr import CppRNRTrainer

    r = RNRTrainer(spec, None, opponent_model=None, p_model=0.0)
    assert type(r) is CppRNRTrainer and r.backend == "cpp"
    r = RNRTrainer(spec, None, opponent_model=None, p_model=0.0, backend="python")
    assert type(r) is RNRTrainer and r.backend == "python"
    monkeypatch.setenv("NEGPLURIBUS_BACKEND", "bogus")
    with pytest.raises(ValueError):
        MCCFRTrainer(spec)


@needs_core
def test_checkpoints_are_interchangeable(tmp_path):
    spec = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))
    cpp = MCCFRTrainer(spec, seed=1, backend="cpp", threads=4).train(3000)
    p = tmp_path / "cpp.json"
    cpp.save_checkpoint(str(p))
    py = MCCFRTrainer(spec, seed=1).load_checkpoint(str(p))
    assert py.iteration == 3000 and len(py.nodes) == cpp.n_nodes
    for k, n in cpp.nodes.items():
        m = py.nodes[k]
        assert (m.actions, m.regret, m.strategy_sum, m.visits) == (n.actions, n.regret, n.strategy_sum, n.visits)
    py.train(10)
    q = tmp_path / "py.json"
    py.save_checkpoint(str(q))
    back = MCCFRTrainer(spec, seed=1, backend="cpp").load_checkpoint(str(q))
    assert back.iteration == 3010 and back.n_nodes == len(py.nodes)
    back.train(10)
    assert back.iteration == 3020
    s = back.strategy()
    assert len(s) == back.n_nodes and all(abs(sum(p) - 1) < 1e-9 for _, p in s.table.values())


# =============================================================== RNR (exploit layer)
@needs_core
def test_cpp_rnr_is_bit_identical_to_python_and_dispatches(monkeypatch):
    from negpluribus.exploit.model import OpponentModel
    from negpluribus.exploit.rnr import RNRTrainer
    from negpluribus.fast.rnr import CppRNRTrainer

    spec = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))
    bp = MCCFRTrainer(spec, seed=0, backend="cpp", threads=1).train(4000).strategy()
    model = OpponentModel(bp)
    model.theta[("preflop", True)] = {"f": -0.8, "c": 0.9, "r": -0.1}   # a station facing a raise
    model.theta[("preflop", False)] = {"f": 0.3, "c": 0.4, "r": -0.7}
    for p_model, warm in ((0.6, bp), (1.0, None), (0.0, bp)):
        py = RNRTrainer(spec, None, model, p_model, seed=3, warm_start=warm, warm_visits=30.0, backend="python").train(1500)
        cpp = RNRTrainer(spec, None, model, p_model, seed=3, warm_start=warm, warm_visits=30.0, backend="cpp", threads=1)
        assert type(cpp) is CppRNRTrainer and cpp.backend == "cpp"
        cpp.train(1500)
        assert cpp.iteration == py.iteration and cpp.nodes_touched == py.nodes_touched
        for table_py, table_cpp in ((py.hero_nodes, cpp.hero_nodes), (py.opp_nodes, cpp.opp_nodes)):
            assert set(table_py) == set(table_cpp)
            for k, n in table_py.items():
                m = table_cpp[k]
                assert (m.actions, m.regret, m.strategy_sum, m.visits) == (n.actions, n.regret, n.strategy_sum, n.visits), k
        assert cpp.strategy().table == py.strategy().table
        assert cpp.rational_opponent().table == py.rational_opponent().table
    # a non-tabular model keeps the Python traversal even when cpp is requested
    class FoldCall:
        def policy(self, key, legal):
            return [1.0 / len(legal)] * len(legal)

    t = RNRTrainer(spec, None, FoldCall(), 0.5, backend="cpp")
    assert type(t) is RNRTrainer and t.backend == "python"
    monkeypatch.setenv("NEGPLURIBUS_BACKEND", "cpp")
    assert type(RNRTrainer(spec, None, model, 0.5)) is CppRNRTrainer
    # multi-threaded run: same tables reached, checkpoint round trip
    many = RNRTrainer(spec, None, model, 0.6, seed=3, warm_start=bp, backend="cpp").train(3000)
    assert many.iteration == 3000 and len(many.hero_nodes) > 100 and len(many.opp_nodes) > 100


# ============================================================== multiprocess driver
def test_train_parallel_merges_workers_and_keeps_linear_weights():
    spec = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))
    tr = MCCFRTrainer(spec, seed=0)
    tr.train(100)
    tr.train_parallel(400, workers=2, sync_every=100)
    assert tr.iteration == 500
    assert tr.nodes_touched > 100 * 10
    assert len(tr.nodes) > 100
    # every node stays a valid regret-matching table; strategy sums grew with Linear weights
    for n in tr.nodes.values():
        assert len(n.regret) == len(n.actions) == len(n.strategy_sum)
        assert all(x >= 0 for x in n.strategy_sum)
    strat = tr.strategy()
    assert all(abs(sum(p) - 1) < 1e-9 for _, p in strat.table.values())
    # keeps training serially afterwards from the merged state
    tr.train(5)
    assert tr.iteration == 505


# ============================================================ bounded bucket caches
def _river_spec(kind: str) -> GameSpec:
    return GameSpec(n_players=2, stack_bb=20, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2, bucket_kind=kind)


@pytest.fixture(scope="module")
def tiny_potential():
    from negpluribus.abstraction import PotentialAwareBucketer

    return PotentialAwareBucketer(n_buckets=8, samples=6, bins=10).fit(n_situations=120, seed=3)


@needs_core
def test_ehs_is_a_pure_function_of_the_canonical_key(flop_bucketer):
    """Two suit representatives of one canonical form give the same E[HS] (Python and C++): the
    Monte-Carlo runs on the canonical representative, so the cached value never depends on
    which representative was seen first, and an evicted entry recomputes to the same number."""
    from negpluribus.abstraction import canonical_key
    from negpluribus.cards import rank_of, suit_of
    from negpluribus.fast.trainer import core_bucketer

    def fresh():
        b = EquityBucketer(flop_bucketer.n_buckets, flop_bucketer.samples)
        b.boundaries = dict(flop_bucketer.boundaries)
        return b

    py1, py2 = fresh(), fresh()
    c1, c2 = core_bucketer(py1), core_bucketer(py2)
    rng = random.Random(5)
    for _ in range(300):
        cards = rng.sample(range(52), 7)
        n_board = rng.choice((3, 4, 5))
        hole, board = cards[:2], cards[2:2 + n_board]
        perm = list(range(4))
        rng.shuffle(perm)
        hole2 = [rank_of(c) * 4 + perm[suit_of(c)] for c in hole]
        board2 = [rank_of(c) * 4 + perm[suit_of(c)] for c in board]
        key = canonical_key(hole, board)
        assert key == canonical_key(hole2, board2)
        ref = equity_vs_random(list(key[:2]), list(key[3:]), 1, flop_bucketer.samples, random.Random(hash(key) & 0xFFFFFFFF))
        assert py1.ehs(hole, board) == py2.ehs(hole2, board2) == c1.ehs(hole, board) == c2.ehs(hole2, board2) == ref
        assert py1.bucket(hole, board) == py2.bucket(hole2, board2) == c1.bucket(hole, board) == c2.bucket(hole2, board2)


@needs_core
@pytest.mark.parametrize("kind", ["ehs", "potential"])
def test_bounded_bucket_cache_is_bit_identical(kind, flop_bucketer, tiny_potential):
    """A 64-entry cache (constant eviction), no cache at all and a roomy cache give identical
    regrets, strategy sums, visits and strategies; all equal the Python reference.  The cap only
    changes memory and speed because every cached value is a pure function of the canonical key."""
    bk = flop_bucketer if kind == "ehs" else tiny_potential
    spec = _river_spec(kind)
    runs = {}
    for label, caps in (("tiny", (64, 64, 64)), ("none", 0), ("roomy", (1_000_000, 1_000_000, 1_000_000))):
        tr = MCCFRTrainer(spec, bk, seed=7, backend="cpp", threads=1, cache_caps=caps).train(80)
        runs[label] = tr
    tables = {k: t.export_tables() for k, t in runs.items()}
    assert tables["tiny"] == tables["none"] == tables["roomy"]
    assert runs["tiny"].strategy().table == runs["roomy"].strategy().table
    stats = {k: t.cache_stats() for k, t in runs.items()}
    for street in ("flop", "turn", "river"):
        assert stats["tiny"][street]["capacity"] == 64 and stats["tiny"][street]["evictions"] > 0
        assert stats["none"][street]["capacity"] == 0 and stats["none"][street]["size"] == 0
        assert stats["roomy"][street]["capacity"] == 1 << 20 and stats["roomy"][street]["evictions"] == 0
        assert stats["roomy"][street]["size"] == stats["roomy"][street]["computes"] > 0
        # without a cache every bucketer call is a compute; the trainer's per-iteration memo calls
        # the bucketer at most once per (seat, street) and iteration, 2 x 80 here (before the memo
        # it was one call per decision node: 997 / 1152 / 1054 flop / turn / river computes in
        # the E[HS] run with the fitted fallback bucketer, 144 / 140 / 122 with it, measured
        # 2026-09-24); the roomy cache computes each canonical form once
        assert stats["roomy"][street]["computes"] <= stats["none"][street]["computes"] <= 2 * 80
        assert stats["tiny"][street]["computes"] >= stats["roomy"][street]["computes"]
    # the Python reference (unbounded dict cache) agrees bit for bit on the 4-street game
    py = MCCFRTrainer(spec, bk, seed=7, backend="python").train(80)
    assert py.iteration == runs["roomy"].iteration and py.nodes_touched == runs["roomy"].nodes_touched
    ref = {k: (n.actions, n.regret, n.strategy_sum, n.visits) for k, n in py.nodes.items()}
    got = {k: (list(a), list(r), list(s), v) for k, (a, r, s, v) in tables["roomy"].items()}
    assert got == ref
    assert {k[0] for k in ref} == {"P", "F", "T", "R"}  # every street's buckets went through the memo


@needs_core
@pytest.mark.parametrize("kind", ["ehs", "potential"])
def test_cpp_rnr_bit_identical_on_the_4_street_game(kind, flop_bucketer, tiny_potential):
    """RNR, cpp x1 == the Python reference on the 4-street game (csrc/rnr.h shares the trainer's
    per-iteration bucket memo; model table and warm start on turn and river keys too)."""
    from negpluribus.exploit.model import OpponentModel
    from negpluribus.exploit.rnr import RNRTrainer

    bk = flop_bucketer if kind == "ehs" else tiny_potential
    spec = _river_spec(kind)
    bp = MCCFRTrainer(spec, bk, seed=1, backend="cpp", threads=1).train(150).strategy()
    model = OpponentModel(bp)
    model.theta[("turn", "")] = {"f": -0.4, "c": 0.6, "r": -0.2}
    model.theta[("river", "r")] = {"f": -0.9, "c": 0.8, "r": 0.1}
    py = RNRTrainer(spec, bk, model, 0.6, seed=5, warm_start=bp, backend="python").train(60)
    cpp = RNRTrainer(spec, bk, model, 0.6, seed=5, warm_start=bp, backend="cpp", threads=1).train(60)
    assert cpp.iteration == py.iteration and cpp.nodes_touched == py.nodes_touched
    for table_py, table_cpp in ((py.hero_nodes, cpp.hero_nodes), (py.opp_nodes, cpp.opp_nodes)):
        assert set(table_py) == set(table_cpp)
        for k, n in table_py.items():
            m = table_cpp[k]
            assert (m.actions, m.regret, m.strategy_sum, m.visits) == (n.actions, n.regret, n.strategy_sum, n.visits), k
    assert {k[0] for k in py.hero_nodes} == {"P", "F", "T", "R"}


@needs_core
def test_cache_caps_from_kwarg_and_env(monkeypatch, flop_bucketer):
    spec = GameSpec(n_players=2, stack_bb=15, max_street=Street.FLOP, n_buckets=8)
    t = MCCFRTrainer(spec, flop_bucketer, backend="cpp", threads=1, cache_caps=(1000, 2000, 0))
    assert t.cache_caps == (1000, 2000, 0)
    caps = {s: v["capacity"] for s, v in t.cache_stats().items()}
    assert caps == {"flop": 1024, "turn": 2048, "river": 0}  # rounded up to whole 8-way sets
    assert MCCFRTrainer(spec, flop_bucketer, backend="cpp", threads=1, cache_caps="3M").cache_caps == (3_000_000,) * 3
    assert MCCFRTrainer(spec, flop_bucketer, backend="cpp", threads=1, cache_caps=(5,)).cache_caps[0] == 5
    monkeypatch.setenv("NEGPLURIBUS_BUCKET_CACHE", "1K,2k,0")
    assert fast.default_cache_caps() == (1000, 2000, 0)
    assert MCCFRTrainer(spec, flop_bucketer, backend="cpp", threads=1).cache_caps == (1000, 2000, 0)
    monkeypatch.setenv("NEGPLURIBUS_BUCKET_CACHE", "2M")
    assert MCCFRTrainer(spec, flop_bucketer, backend="cpp", threads=1).cache_caps == (2_000_000,) * 3
    monkeypatch.delenv("NEGPLURIBUS_BUCKET_CACHE")
    t = MCCFRTrainer(spec, flop_bucketer, backend="cpp", threads=1)
    assert t.cache_caps == fast.DEFAULT_CACHE_CAPS == (4_000_000, 32_000_000, 4_000_000)
    # 4M / 32M / 4M entries -> 32 / 256 / 32 MB of 8-byte slots, each allocated on its street's first insert
    caps = {s: v["capacity"] for s, v in t.cache_stats().items()}
    assert caps == {"flop": 4_194_304, "turn": 33_554_432, "river": 4_194_304}
    with pytest.raises(ValueError):
        fast.parse_cache_caps("1,2,3,4")
    # the Python backend accepts and ignores the kwarg
    assert MCCFRTrainer(spec, flop_bucketer, backend="python", cache_caps=(1, 1, 1)).backend == "python"


@needs_core
def test_cpp_resume_is_bit_identical_and_nodes_view_is_live(tmp_path, flop_bucketer):
    """A checkpoint carries the iteration counter (Linear CFR weights continue at t+1) and every
    thread's RNG state, so resume + train == one uninterrupted run; ``nodes`` is a live view."""
    spec = GameSpec(n_players=2, stack_bb=15, max_street=Street.FLOP, n_buckets=8)
    straight = MCCFRTrainer(spec, flop_bucketer, seed=3, backend="cpp", threads=1).train(700)
    first = MCCFRTrainer(spec, flop_bucketer, seed=3, backend="cpp", threads=1).train(400)
    p = tmp_path / "ck.json"
    first.save_checkpoint(str(p))
    resumed = MCCFRTrainer(spec, flop_bucketer, seed=99, backend="cpp", threads=1).load_checkpoint(str(p))
    assert resumed.iteration == 400
    resumed.train(300)
    assert resumed.iteration == 700 == straight.iteration
    assert resumed.export_tables() == straight.export_tables()
    assert resumed.strategy().table == straight.strategy().table
    # multi-threaded: every thread's stream is restored, the counter continues
    many = MCCFRTrainer(spec, flop_bucketer, seed=1, backend="cpp", threads=4).train(200)
    many.save_checkpoint(str(p))
    states = many._core.rng_states()
    assert len(states) == 4
    again = MCCFRTrainer(spec, flop_bucketer, seed=1, backend="cpp", threads=4).load_checkpoint(str(p))
    assert again._core.rng_states() == states and again.iteration == 200
    again.train(10)
    assert again.iteration == 210
    # the Python trainer loads the same file (extra fields ignored) and continues at 201
    py = MCCFRTrainer(spec, flop_bucketer, seed=1).load_checkpoint(str(p))
    assert py.iteration == 200 and len(py.nodes) == many.n_nodes
    py.train(1)
    assert py.iteration == 201
    # nodes: a live read-through view, no snapshot held between exports
    from negpluribus.fast.trainer import NodeView

    view = straight.nodes
    assert isinstance(view, NodeView) and len(view) == straight.n_nodes and not hasattr(straight, "_nodes_cache")
    key = next(iter(view))
    assert key in view and view.get("nope") is None and view.get(key).actions == view[key].actions
    visits_before = sum(n.visits for n in view.values())
    straight.train(50)
    assert sum(n.visits for n in straight.nodes.values()) > visits_before  # same object, new numbers
    assert straight.nodes is view and len(view) == straight.n_nodes
    snap = view.snapshot()
    assert isinstance(snap, dict) and len(snap) == len(view)
    with pytest.raises(KeyError):
        view["nope"]


@needs_core
def test_cpp_rnr_resume_and_views(tmp_path, flop_bucketer):
    from negpluribus.exploit.model import OpponentModel
    from negpluribus.exploit.rnr import RNRTrainer
    from negpluribus.fast.trainer import NodeView

    spec = GameSpec(n_players=2, stack_bb=15, max_street=Street.FLOP, n_buckets=8)
    bp = MCCFRTrainer(spec, flop_bucketer, seed=0, backend="cpp", threads=1).train(300).strategy()
    model = OpponentModel(bp)
    model.theta[("flop", True)] = {"f": -0.5, "c": 0.7, "r": -0.2}
    # the warm-start prior of a key is fixed when the key is created (from planned_iters at that
    # moment, as in the reference), so the comparison is train(200) + train(100) in one object
    # against train(200) + checkpoint + load + train(100)
    straight = RNRTrainer(spec, flop_bucketer, model, 0.6, seed=3, warm_start=bp, backend="cpp", threads=1).train(200)
    first = RNRTrainer(spec, flop_bucketer, model, 0.6, seed=3, warm_start=bp, backend="cpp", threads=1).train(200)
    p = tmp_path / "rnr.json"
    first.save_checkpoint(str(p))
    resumed = RNRTrainer(spec, flop_bucketer, model, 0.6, seed=3, warm_start=bp, backend="cpp", threads=1).load_checkpoint(str(p))
    assert resumed.iteration == 200
    straight.train(100)
    resumed.train(100)
    assert resumed.iteration == 300 == straight.iteration
    assert resumed._core.export_hero() == straight._core.export_hero()
    assert resumed._core.export_opp() == straight._core.export_opp()
    assert isinstance(straight.hero_nodes, NodeView) and isinstance(straight.opp_nodes, NodeView)
    assert len(straight.hero_nodes) == straight._core.n_hero and straight.nodes is straight.hero_nodes
    assert straight.cache_stats()["flop"]["computes"] > 0


# ====================================================== numeric node keys, flat table
FOREIGN_BIT = 1 << 63


@needs_core
def test_numeric_keys_equal_the_parsed_key_strings(flop_bucketer):
    """The key a trainer computes without the string (fields + history hashed event by event) is the
    key a checkpoint import computes by parsing the string, along random trajectories in four games;
    strings that are not in canonical form get foreign keys that no traversal key can equal."""
    from negpluribus.fast.trainer import core_bucketer, spec_to_dict

    specs = [
        GameSpec(n_players=3, stack_bb=15, max_street=Street.FLOP, n_buckets=8),
        GameSpec(n_players=2, stack_bb=100, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2),
        GameSpec(n_players=6, stack_bb=25, max_street=Street.RIVER, n_buckets=8, preflop_fracs=(0.5, 1.0), postflop_fracs=(0.33, 0.75, 1.5)),
        pushfold_spec(),
    ]
    pyb = EquityBucketer(flop_bucketer.n_buckets, flop_bucketer.samples)
    pyb.boundaries = dict(flop_bucketer.boundaries)
    cbk = core_bucketer(pyb)
    seen = {}
    for si, spec in enumerate(specs):
        game = core.Game(spec_to_dict(spec), cbk)
        for i in range(150):
            rng = random.Random(7000 + si * 1000 + i)
            order = list(range(52))
            rng.shuffle(order)
            c = game.new_hand(order, rng.randrange(spec.n_players))
            while not c.is_terminal:
                key, nk = c.key(), tuple(c.numeric_key())
                assert nk == tuple(game.numeric_key(key)), key
                assert not nk[1] & FOREIGN_BIT
                assert seen.setdefault((spec.n_players, nk), key) == key, (key, seen[(spec.n_players, nk)])
                c.apply_name(rng.choice(c.legal()))
    assert len(seen) > 2000
    game = core.Game(spec_to_dict(specs[1]), cbk)
    canonical = "F|BB|2|b3|c r0.5"
    for other in ("F|BB|2|b03|c r0.5", "F|bb|2|b3|c r0.5", "F|BB|02|b3|c r0.5", "F|BB|2|3|c r0.5", "X|BB|2|b3|c r0.5",
                  "F|BB|2|b3", "F|UTG|2|b3|c r0.5", "", "F|BB|2|b99999999999|"):
        k = tuple(game.numeric_key(other))
        assert k[1] & FOREIGN_BIT and k != tuple(game.numeric_key(canonical)), other
        assert k == tuple(game.numeric_key(other))  # deterministic
    assert tuple(game.numeric_key("F|BB|2|b3|c r0.5 ")) != tuple(game.numeric_key(canonical))  # history bytes count


@needs_core
def test_verify_keys_is_on_in_tests_and_catches_a_wrong_stored_key(flop_bucketer, monkeypatch):
    """tests/conftest.py turns the test mode on: every lookup compares the node's stored key string
    with the string built the old way.  A node stored under the wrong string is reported."""
    from negpluribus.exploit.model import OpponentModel
    from negpluribus.exploit.rnr import RNRTrainer

    spec = GameSpec(n_players=2, stack_bb=15, max_street=Street.FLOP, n_buckets=8)
    tr = MCCFRTrainer(spec, flop_bucketer, seed=3, backend="cpp", threads=2).train(300)
    assert tr._core.verify_keys
    bp = tr.strategy()
    keys = tr._core.keys()
    for k in keys:
        assert tr._core._debug_relabel(k, "x" + k)
    assert sorted(tr._core.keys()) == sorted("x" + k for k in keys)
    with pytest.raises(RuntimeError, match="numeric infoset key mismatch"):
        tr.train(50)
    # production mode: no check (and no cost)
    monkeypatch.setenv("NEGPLURIBUS_VERIFY_KEYS", "0")
    fast_tr = MCCFRTrainer(spec, flop_bucketer, seed=3, backend="cpp", threads=1).train(50)
    assert not fast_tr._core.verify_keys
    for k in fast_tr._core.keys():
        fast_tr._core._debug_relabel(k, "x" + k)
    fast_tr.train(5)
    # the RNR trainer checks too (hero and opponent tables, and the opponent-model index)
    monkeypatch.setenv("NEGPLURIBUS_VERIFY_KEYS", "1")
    r = RNRTrainer(spec, flop_bucketer, OpponentModel(bp), 0.6, seed=3, warm_start=bp, backend="cpp", threads=1).train(200)
    assert r._core.verify_keys
    for k in r._core.opp_keys():
        r._core._debug_relabel(k, "x" + k, False)
    with pytest.raises(RuntimeError, match="numeric infoset key mismatch"):
        r.train(50)


@needs_core
def test_flat_node_table_concurrent_inserts_never_drop_or_duplicate():
    """16 threads look up the same 100k keys in different random orders, three times each, on a
    table that starts with 16 slots (so it doubles 14 times under them): every key is created
    exactly once, every lookup returns that node with the key's own string."""
    n = 100_000
    r = core._table_stress(threads=16, n_keys=n, rounds=3, initial_capacity=16, seed=1)
    assert r["created"] == r["size"] == r["for_each"] == n, r
    assert r["other_node"] == r["wrong_key"] == r["missing"] == 0, r
    assert r["lookups"] == 16 * 3 * n
    assert r["resizes"] >= 10 and r["capacity"] >= 2 * n, r


@needs_core
def test_cpp_threads_grow_the_node_table_without_losing_or_duplicating_nodes(flop_bucketer):
    """A 16-thread run of the 4-street game grows its table several times (4096 slots at the start);
    afterwards every infoset is there once and findable, and every key was verified at every lookup."""
    t = MCCFRTrainer(_river_spec("ehs"), flop_bucketer, seed=11, backend="cpp", threads=16).train(3000)
    stats = t._core.table_stats()
    keys = t._core.keys()
    assert len(keys) == len(set(keys)) == t.n_nodes == stats["size"]
    assert stats["resizes"] >= 2 and stats["capacity"] >= 2 * stats["size"], stats
    assert all(t._core.get_node(k) is not None for k in keys)
    assert t._core.get_node("F|BB|2|b3|no such history") is None


@needs_core
def test_export_import_keep_contents_including_non_canonical_keys(flop_bucketer):
    """Export and import move the same (key -> values) contents in any order; keys that are not in
    canonical form (possible in a hand-made checkpoint) stay separate nodes and round-trip."""
    spec = GameSpec(n_players=3, stack_bb=15, max_street=Street.FLOP, n_buckets=8)
    a = MCCFRTrainer(spec, flop_bucketer, seed=5, backend="cpp", threads=4).train(500)
    table = {k: (list(x), list(r), list(s), v) for k, (x, r, s, v) in a.export_tables().items()}
    canonical = next(k for k in table if k.startswith("F|"))
    parts = canonical.split("|")
    odd = "|".join(parts[:3] + ["b0" + parts[3][1:]] + parts[4:])  # leading zero in the bucket
    extra = {odd: (["c"], [1.0], [2.0], 3), "weird key": (["f", "c"], [0.5, -0.5], [1.0, 1.0], 7), "P|XYZ|2|b1|": (["c"], [0.0], [0.0], 0)}
    b = MCCFRTrainer(spec, flop_bucketer, seed=5, backend="cpp", threads=1)
    rows = dict(reversed(list({**table, **extra}.items())))  # another insertion order
    b._core.import_nodes(rows, True)
    got = {k: (list(x), list(r), list(s), v) for k, (x, r, s, v) in b.export_tables().items()}
    assert got == {**table, **extra}
    assert b.n_nodes == len(table) + 3
    assert list(b._core.get_node("weird key")[1]) == [0.5, -0.5] and b._core.get_node(odd)[3] == 3
    assert list(b._core.get_node(canonical)[1]) == table[canonical][1]
    for k in extra:
        assert b._core.numeric_key(k)[1] & FOREIGN_BIT
    assert not any(b._core.numeric_key(k)[1] & FOREIGN_BIT for k in table)
    # the imported table keeps training exactly like the original one
    c = MCCFRTrainer(spec, flop_bucketer, seed=5, backend="cpp", threads=1)
    c._core.import_nodes(table, True)
    d = MCCFRTrainer(spec, flop_bucketer, seed=5, backend="cpp", threads=1)
    d._core.import_nodes(dict(reversed(list(table.items()))), True)
    c.train(100)
    d.train(100)
    assert c.export_tables() == d.export_tables()


@needs_core
def test_cpp_equity_rejects_more_than_two_hole_or_five_board_cards():
    state = random.Random(0).getstate()[1]
    for hole, board in (([0, 1, 2], []), ([0, 1], [2, 3, 4, 5, 6, 7])):
        with pytest.raises(ValueError):
            core.equity_vs_random_seeded(hole, board, 1, 10, 0)
        with pytest.raises(ValueError):
            core.equity_vs_random(hole, board, 1, 10, state)
    # fewer cards are fine, as in the reference (one hole card and no board: 6-card hands)
    assert core.equity_vs_random_seeded([51], [], 1, 50, 0) == equity_vs_random_py([51], [], 1, 50, random.Random(0))
