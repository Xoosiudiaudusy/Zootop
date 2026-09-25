"""Potential-aware card buckets (negpluribus/abstraction/potential.py, docs/buckets.md).

(a) the draw-vs-made-hand signature, (b) Python == C++ bit for bit, (c) the interface
(preflop, keys, save/load, kind dispatch), (e) nothing here needs the C++ core: every test
that does is skipped automatically when it is not built.  The Monte-Carlo sample counts are
deliberately small; the fits are deterministic (seeded), so the assertions are reproducible.
"""
from __future__ import annotations

import json
import random
import re

import pytest

from negpluribus import fast
from negpluribus.abstraction import (
    BetGrid,
    EquityBucketer,
    PotentialAwareBucketer,
    bucketer_kind,
    infoset_key,
    load_bucketer,
    make_bucketer,
)
from negpluribus.abstraction import potential as potential_mod
from negpluribus.abstraction.potential import emd, histogram_cdf, kmeans_emd
from negpluribus.agents.blueprint import BlueprintAgent
from negpluribus.cards import ALL_HOLE_CLASSES, cards_from_str
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import CALL, HandState, Street, raise_to
from negpluribus.equity import equity_vs_random, equity_vs_random_py

core = fast.core()
needs_core = pytest.mark.skipif(core is None, reason="C++ core not built (python scripts/build_fast.py)")
needs_potential_core = pytest.mark.skipif(
    core is None or not hasattr(core, "PotentialBucketer"), reason="C++ core without PotentialBucketer (rebuild)"
)

cs = cards_from_str
FIT_SAMPLES = 40  # runouts per next-street card in the test fits (100 is the production default)
FIT_SITUATIONS = 400


def flop_spec(kind: str = "potential", n_buckets: int = 8) -> GameSpec:
    return GameSpec(n_players=2, stack_bb=20, max_street=Street.FLOP, n_buckets=n_buckets, bucket_kind=kind)


@pytest.fixture(scope="module")
def pot8() -> PotentialAwareBucketer:
    return PotentialAwareBucketer(n_buckets=8, samples=FIT_SAMPLES, bins=10).fit(n_situations=FIT_SITUATIONS, seed=1)


@pytest.fixture(scope="module")
def pot16(pot8) -> PotentialAwareBucketer:
    b = PotentialAwareBucketer(n_buckets=16, samples=FIT_SAMPLES, bins=10)
    b._features.update(pot8._features)  # same situations, same features: no second Monte-Carlo pass
    return b.fit(n_situations=FIT_SITUATIONS, seed=1)


@pytest.fixture(scope="module")
def ehs8() -> EquityBucketer:
    return EquityBucketer(n_buckets=8, samples=150).fit(n_situations=400, seed=1)


@pytest.fixture(scope="module")
def tiny() -> PotentialAwareBucketer:
    """A cheap fit for the interface / equivalence tests."""
    return PotentialAwareBucketer(n_buckets=8, samples=6, bins=10).fit(n_situations=120, seed=3)


# ================================================================ (a) the signature
# (draw, made hand with a similar E[HS], board).  E[HS] values are printed (run with -s).
FLOP_PAIRS = [
    ("Ah Qh", "Qs 9d", "Kh 9h 2c"),  # nut flush draw + overcards vs second pair
    ("Jh Th", "9s 8s", "Kh 9h 2c"),  # flush draw + gutshot vs second pair
]
TURN_PAIRS = [
    ("Ah Qh", "4s 7s", "Kh 9h 2c 7d"),  # nut flush draw + overcards vs third pair
    ("Jh Th", "As 2s", "Kh 9h 2c 7d"),  # flush draw + gutshot vs bottom pair, top kicker
]


def _ehs(hand: str, board: str) -> float:
    return equity_vs_random(cs(hand), cs(board), 1, 4000, random.Random(11))


def _spread(cdf) -> float:
    """Mean absolute deviation of the histogram around its mean, in equity units."""
    bins = len(cdf)
    masses = [c - (cdf[i - 1] if i else 0.0) for i, c in enumerate(cdf)]
    centers = [(i + 0.5) / bins for i in range(bins)]
    mean = sum(m * c for m, c in zip(masses, centers))
    return sum(m * abs(c - mean) for m, c in zip(masses, centers))


def test_draw_and_made_hand_have_the_same_ehs_but_different_histograms(pot8, ehs8):
    """The feature itself: equal equity now, very different equity distributions next street."""
    for draw, made, board in FLOP_PAIRS + TURN_PAIRS:
        e_draw, e_made = _ehs(draw, board), _ehs(made, board)
        print(f"\n{board}: {draw} E[HS]={e_draw:.3f}  {made} E[HS]={e_made:.3f}  "
              f"E[HS] buckets {ehs8.bucket(cs(draw), cs(board))} / {ehs8.bucket(cs(made), cs(board))}")
        assert abs(e_draw - e_made) < 0.035, (draw, made, e_draw, e_made)
        cdf_d, mean_d = pot8.feature(cs(draw), cs(board))
        cdf_m, mean_m = pot8.feature(cs(made), cs(board))
        assert abs(mean_d - mean_m) < 0.06  # the histogram mean is E[HS] up to Monte-Carlo noise
        d = emd(cdf_d, cdf_m)
        s_d, s_m = _spread(cdf_d), _spread(cdf_m)
        print(f"   EMD(draw, made) = {d:.2f} bins;  spread {s_d:.3f} vs {s_m:.3f};  "
              f"draw {pot8.describe_centroid(cdf_d)}  made {pot8.describe_centroid(cdf_m)}")
        assert d > 0.45, (draw, made, d)
        # same mean, wider distribution: the draw either gets there or misses
        assert s_d > 1.3 * s_m, (draw, made, s_d, s_m)
    # two made hands of the same strength stay close
    same_kind = emd(pot8.feature(cs("Qs 9d"), cs("Kh 9h 2c"))[0], pot8.feature(cs("Ts 9d"), cs("Kh 9h 2c"))[0])
    assert same_kind < 0.4, same_kind


def test_signature_pairs_split_into_different_potential_buckets(pot8, pot16, ehs8):
    """The user's acceptance test.  At 16 clusters every pair splits; at the game's 8 clusters the
    two strongly bimodal draws (flush draw + gutshot) do, the A-high flush draws are reported
    (their EMD to the made hand, ~0.7 bins, is about the radius of an 8-cluster; docs/buckets.md)."""
    for draw, made, board in FLOP_PAIRS + TURN_PAIRS:
        b8 = pot8.bucket(cs(draw), cs(board)), pot8.bucket(cs(made), cs(board))
        b16 = pot16.bucket(cs(draw), cs(board)), pot16.bucket(cs(made), cs(board))
        be = ehs8.bucket(cs(draw), cs(board)), ehs8.bucket(cs(made), cs(board))
        print(f"\n{board}: {draw} vs {made}: potential-8 b{b8[0]}/b{b8[1]}  potential-16 b{b16[0]}/b{b16[1]}  "
              f"E[HS]-8 b{be[0]}/b{be[1]}{'  (same E[HS] bucket)' if be[0] == be[1] else ''}")
        assert b16[0] != b16[1], (draw, made, b16)
        if draw == "Jh Th":
            assert b8[0] != b8[1], (draw, made, b8)
    # the E[HS] bucketer cannot tell them apart in general: at least half of the pairs share a bucket
    shared = sum(ehs8.bucket(cs(d), cs(b)) == ehs8.bucket(cs(m), cs(b)) for d, m, b in FLOP_PAIRS + TURN_PAIRS)
    assert shared >= 2, shared


def test_buckets_are_ordered_by_strength(pot8):
    board = cs("Kh 9h 2c")
    assert pot8.bucket(cs("Kc Kd"), board) >= pot8.bucket(cs("Kd 3d"), board) > pot8.bucket(cs("7c 7d"), board) > pot8.bucket(cs("6d 5d"), board)
    assert pot8.bucket(cs("Kc Kd"), board) == 7
    assert pot8.bucket(cs("6d 5d"), board) == 0
    for street in (Street.FLOP, Street.TURN):
        means = pot8.centroid_mean_equity[int(street)]
        assert means == sorted(means)
    river = cs("Kh 9h 2c 7d 3s")
    assert pot8.bucket(cs("Kc Kd"), river) == 7 and pot8.bucket(cs("6d 5d"), river) == 0
    assert pot8.bucket(cs("Kc Kd"), river) > pot8.bucket(cs("Ad 9d"), river) > pot8.bucket(cs("6d 5d"), river)


# ===================================================================== (c) interface
def test_preflop_is_the_same_169_classes(tiny, ehs8):
    for c1 in range(52):
        for c2 in range(c1 + 1, 52):
            assert tiny.bucket([c1, c2], []) == ehs8.bucket([c1, c2], [])
    assert tiny.bucket(cs("Ah Ad"), []) == 0 == ALL_HOLE_CLASSES.index("AA")
    assert tiny.n_buckets_for(Street.PREFLOP) == 169
    for s in (Street.FLOP, Street.TURN, Street.RIVER):
        assert tiny.n_buckets_for(s) == 8


def test_infoset_keys_keep_their_format(tiny, ehs8):
    grid = BetGrid()
    h = HandState([10_000] * 6, button=0, seed=7)
    for a in (raise_to(250), "f", CALL, "f", "f", CALL, CALL):
        h.apply(a if a != "f" else __import__("negpluribus.engine", fromlist=["FOLD"]).FOLD)
    obs = h.observe()
    key = infoset_key(obs, tiny, grid)
    assert re.fullmatch(r"[PFTR]\|[A-Z/]+\|\d\|b\d+\|.*", key), key
    street, pos, n_active, bucket, hist = key.split("|")
    assert street == "F" and pos == "UTG" and n_active == "3" and hist == "r0.5 f c f f c/c"
    assert 0 <= int(bucket[1:]) < 8
    key_ehs = infoset_key(obs, ehs8, grid)
    assert key.split("|")[:3] == key_ehs.split("|")[:3] and key.split("|")[4] == key_ehs.split("|")[4]
    # deterministic and cached
    assert infoset_key(obs, tiny, grid) == key


def test_save_load_roundtrip_and_kind_dispatch(tmp_path, tiny, ehs8):
    p = tmp_path / "pot.json"
    tiny.save(str(p))
    d = json.load(open(p, encoding="utf-8"))
    assert d["kind"] == "potential" and d["bins"] == 10 and d["samples"] == 6
    b2 = load_bucketer(str(p))
    assert type(b2) is PotentialAwareBucketer and bucketer_kind(b2) == "potential"
    assert b2.centroids == tiny.centroids and b2.boundaries == tiny.boundaries and b2.centroid_mean_equity == tiny.centroid_mean_equity
    b3 = PotentialAwareBucketer.load(str(p))
    rng = random.Random(9)
    for _ in range(100):
        cards = rng.sample(range(52), 7)
        n_board = rng.choice((0, 3, 4, 5))
        hole, board = cards[:2], cards[2:2 + n_board]
        assert b2.bucket(hole, board) == b3.bucket(hole, board) == tiny.bucket(hole, board)
    # old E[HS] files (no "kind") keep loading through both entry points
    q = tmp_path / "ehs.json"
    ehs8.save(str(q))
    assert "kind" not in json.load(open(q, encoding="utf-8"))
    e2, e3 = EquityBucketer.load(str(q)), load_bucketer(str(q))
    assert type(e2) is type(e3) is EquityBucketer and e2.boundaries == e3.boundaries == ehs8.boundaries
    assert bucketer_kind(e2) == "ehs"
    # the wrong loader says so instead of returning a broken object
    with pytest.raises(ValueError):
        EquityBucketer.load(str(p))
    with pytest.raises(ValueError):
        PotentialAwareBucketer.load(str(q))


def test_gamespec_bucket_kind():
    assert type(GameSpec().make_bucketer()) is EquityBucketer and GameSpec().bucket_kind == "ehs"
    spec = flop_spec("potential")
    b = spec.make_bucketer()
    assert type(b) is PotentialAwareBucketer and b.n_buckets == 8 and b.samples == 100 and b.bins == 10
    assert spec.make_bucketer(samples=7).samples == 7
    assert "(potential)" in spec.describe() and "(ehs)" in GameSpec().describe()
    with pytest.raises(ValueError):
        GameSpec(bucket_kind="ochs")
    with pytest.raises(ValueError):
        make_bucketer("ochs", 8)
    assert type(make_bucketer("ehs", 5)) is EquityBucketer and make_bucketer("ehs", 5).samples == 150


def test_unfitted_bucketer_is_rejected_and_fitted_one_trains(tiny):
    spec = flop_spec("potential")
    with pytest.raises(ValueError):
        MCCFRTrainer(spec, PotentialAwareBucketer(n_buckets=8, samples=6))
    with pytest.raises(RuntimeError):
        PotentialAwareBucketer(n_buckets=8, samples=6).bucket(cs("Ah Qh"), cs("Kh 9h 2c"))
    tr = MCCFRTrainer(spec, tiny, seed=2).train(40)
    assert tr.iteration == 40 and len(tr.nodes) > 50
    assert all(k.split("|")[3].startswith("b") for k in tr.nodes)
    flop_buckets = {int(k.split("|")[3][1:]) for k in tr.nodes if k.startswith("F|")}
    assert flop_buckets and flop_buckets <= set(range(8))
    hero = BlueprintAgent(tr.strategy(), tiny, spec.grid, seed=1)
    from negpluribus.agents import make_agent
    from negpluribus.eval import duplicate_match

    res = duplicate_match(hero, [make_agent("station", seed=2)], n_deals=10, seed=1, sb=spec.sb, bb=spec.bb,
                          stack_bb=spec.stack_bb, max_street=spec.max_street)
    assert res.n_hands == 20


def test_train_parallel_carries_the_potential_bucketer_to_workers(tiny):
    spec = flop_spec("potential")
    tr = MCCFRTrainer(spec, tiny, seed=0)
    tr.train(5)
    tr.train_parallel(20, workers=2, sync_every=5)
    assert tr.iteration == 25 and len(tr.nodes) > 50


def test_kmeans_emd_basics():
    rng = random.Random(0)
    cdf_lo = histogram_cdf([5, 3, 1, 0, 0])
    cdf_hi = histogram_cdf([0, 0, 1, 3, 5])
    assert emd(cdf_lo, cdf_lo) == 0.0 and emd(cdf_lo, cdf_hi) == pytest.approx(emd(cdf_hi, cdf_lo))
    pts = [cdf_lo] * 10 + [cdf_hi] * 10
    cents, assign = kmeans_emd(pts, 2, rng)
    assert assign[:10] == [assign[0]] * 10 and assign[10:] == [assign[10]] * 10 and assign[0] != assign[10]
    with pytest.raises(ValueError):
        kmeans_emd(pts[:1], 2, rng)


# ====================================================== (b) Python == C++ bit for bit
def _random_situation(rng, n_board):
    cards = rng.sample(range(52), 2 + n_board)
    return cards[:2], cards[2:]


@needs_potential_core
def test_cpp_features_bit_identical_to_pure_python_reference(monkeypatch):
    from negpluribus.abstraction.potential import next_street_histogram_py, river_equity_py

    rng = random.Random(2026)
    # fully pure chain (Python histogram over the pure-Python equity loop) on a smaller set...
    for n_board in (3, 4):
        for _ in range(60):
            hole, board = _random_situation(rng, n_board)
            monkeypatch.setattr(potential_mod, "equity_vs_random", equity_vs_random_py)
            a = next_street_histogram_py(hole, board, 5, 10)
            monkeypatch.setattr(potential_mod, "equity_vs_random", equity_vs_random)
            b = core.potential_histogram(hole, board, 5, 10)
            assert a[0] == list(b[0]) and a[1] == b[1], (hole, board, a, b)
    # ...and the Python histogram over the (already bit-identical) compiled equity on a larger one
    for n_board in (3, 4):
        for _ in range(1000):
            hole, board = _random_situation(rng, n_board)
            a = next_street_histogram_py(hole, board, 6, 10)
            b = core.potential_histogram(hole, board, 6, 10)
            assert a[0] == list(b[0]) and a[1] == b[1], (hole, board, a, b)
    for _ in range(2000):
        hole, board = _random_situation(rng, 5)
        assert river_equity_py(hole, board) == core.river_equity_exact(hole, board), (hole, board)


@needs_potential_core
def test_cpp_turn_histogram_with_production_samples_matches_pure_python(monkeypatch):
    """The turn feature at the production sample count (46 river cards x 100 runouts): every
    runout is on a complete board, where the port evaluates our hand once per river card instead
    of once per runout.  Same counts and mean as the fully pure-Python chain."""
    from negpluribus.abstraction.potential import next_street_histogram_py

    monkeypatch.setattr(potential_mod, "equity_vs_random", equity_vs_random_py)
    rng = random.Random(24_09_2026)
    for _ in range(6):
        hole, board = _random_situation(rng, 4)
        a = next_street_histogram_py(hole, board, 100, 10)
        b = core.potential_histogram(hole, board, 100, 10)
        assert a[0] == list(b[0]) and a[1] == b[1], (hole, board, a, b)


@needs_potential_core
def test_cpp_buckets_equal_python_buckets_on_5000_situations_per_street(tiny, monkeypatch):
    from negpluribus.fast.trainer import core_bucketer

    monkeypatch.setattr(potential_mod, "next_street_histogram", potential_mod.next_street_histogram_py)
    monkeypatch.setattr(potential_mod, "river_equity", potential_mod.river_equity_py)
    pyb = PotentialAwareBucketer.from_dict(tiny.to_dict())  # fresh cache, reference code path
    cbk = core_bucketer(pyb)
    assert type(cbk).__name__ == "PotentialBucketer" and cbk.fitted
    assert cbk.n_buckets == 8 and cbk.samples == 6 and cbk.bins == 10
    rng = random.Random(77)
    for n_board in (0, 3, 4, 5):
        n = 5000 if n_board else 1326
        for _ in range(n):
            hole, board = _random_situation(rng, n_board)
            assert pyb.bucket(hole, board) == cbk.bucket(hole, board), (hole, board)
    assert cbk.cache_size() == len(pyb._cache)
    # features too (CDF and mean equity)
    for n_board in (3, 4):
        for _ in range(200):
            hole, board = _random_situation(rng, n_board)
            cdf, mean = pyb.feature(hole, board)
            ccdf, cmean = cbk.feature(hole, board)
            assert list(ccdf) == cdf and cmean == mean
    for _ in range(200):
        hole, board = _random_situation(rng, 5)
        assert cbk.river_ehs(hole, board) == pyb.river_ehs(hole, board)


@needs_potential_core
def test_cpp_trainer_and_rnr_bit_identical_with_potential_bucketer(tiny):
    from negpluribus.exploit.model import OpponentModel
    from negpluribus.exploit.rnr import RNRTrainer
    from negpluribus.fast.rnr import CppRNRTrainer
    from negpluribus.fast.trainer import CppMCCFRTrainer, core_bucketer, spec_to_dict

    spec = flop_spec("potential")
    py = MCCFRTrainer(spec, tiny, seed=4, backend="python").train(300)
    cpp = MCCFRTrainer(spec, tiny, seed=4, backend="cpp", threads=1).train(300)
    assert type(cpp) is CppMCCFRTrainer and cpp.backend == "cpp"
    assert cpp.iteration == py.iteration and cpp.nodes_touched == py.nodes_touched
    assert set(cpp.nodes) == set(py.nodes)
    for k, n in py.nodes.items():
        m = cpp.nodes[k]
        assert (m.actions, m.regret, m.strategy_sum, m.visits) == (n.actions, n.regret, n.strategy_sum, n.visits), k
    assert cpp.strategy().table == py.strategy().table
    # keys along random trajectories through the C++ game object
    pyb = PotentialAwareBucketer.from_dict(tiny.to_dict())
    game = core.Game(spec_to_dict(spec), core_bucketer(pyb))
    grid = spec.grid
    n_keys = 0
    for i in range(120):
        rng = random.Random(500 + i)
        order = list(range(52))
        rng.shuffle(order)
        h, c = spec.new_hand(order, i % 2), game.new_hand(order, i % 2)
        while not h.is_terminal:
            obs = h.observe(h.current_player)
            legal = grid.abstract_actions(obs)
            assert c.legal() == legal and c.key() == infoset_key(obs, pyb, grid)
            a = rng.choice(legal)
            h.apply(grid.to_concrete(obs, a))
            c.apply_name(a)
            n_keys += 1
    assert n_keys >= 200
    # RNR (exploit layer) with the new bucketer: python == cpp x1, and the multithreaded run works
    bp = py.strategy()
    model = OpponentModel(bp)
    model.theta[("flop", "")] = {"f": 0.2, "c": 0.5, "r": -0.7}
    r_py = RNRTrainer(spec, tiny, model, 0.6, seed=3, warm_start=bp, backend="python").train(150)
    r_cpp = RNRTrainer(spec, tiny, model, 0.6, seed=3, warm_start=bp, backend="cpp", threads=1)
    assert type(r_cpp) is CppRNRTrainer
    r_cpp.train(150)
    assert r_cpp.iteration == r_py.iteration and r_cpp.nodes_touched == r_py.nodes_touched
    for table_py, table_cpp in ((r_py.hero_nodes, r_cpp.hero_nodes), (r_py.opp_nodes, r_cpp.opp_nodes)):
        assert set(table_py) == set(table_cpp)
        for k, n in table_py.items():
            m = table_cpp[k]
            assert (m.actions, m.regret, m.strategy_sum, m.visits) == (n.actions, n.regret, n.strategy_sum, n.visits), k
    many = MCCFRTrainer(spec, tiny, seed=4, backend="cpp").train(2000)
    assert many.iteration == 2000 and many.n_nodes > len(py.nodes)
    s = many.strategy()
    assert all(abs(sum(p) - 1) < 1e-9 for _, p in s.table.values())


@needs_core
def test_core_bucketer_dispatch_keeps_equity_bucketer(ehs8):
    from negpluribus.fast.trainer import core_bucketer

    cbk = core_bucketer(ehs8)
    assert type(cbk).__name__ == "Bucketer" and cbk.n_buckets == 8 and cbk.samples == 150
    assert cbk.bucket(cs("Kc Kd"), cs("Kh 9h 2c")) == ehs8.bucket(cs("Kc Kd"), cs("Kh 9h 2c"))
