"""AIVAT (negpluribus/eval/aivat.py, aivat_values.py, csrc/aivat.h; docs/aivat.md).

  * the pieces the estimator is built on, each against the code it mirrors: the random streams, the
    agent's sampling law and translation coins (against BetGrid.from_concrete with stubbed coins),
    the agent model (against BlueprintAgent's own keys and probabilities);
  * the C++ evaluator equals the Python reference on the same seeds: every branch value vector bit for
    bit, every term and the hand value to rounding;
  * unbiasedness (Burch et al. 2018, Theorem 1) on a small heads-up game: in blueprint self-play with
    alternating seats the exact value is 0 and the AIVAT mean must be within its CI of it; against a
    random bettor (off-grid sizes: the translation coins matter) the paired difference AIVAT - net
    must have mean 0; every kind of correction term must have mean 0, for the self-play heuristic and
    for an arbitrary one (AdditiveCardValues: Lemma 1 holds for ANY value function);
  * variance reduction on the same game;
  * logged range strategies (docs/aivat.md section 7): the format round-trips, C++ = reference with them, and an
    agent that samples from its logged quantized rows (what the spec asks of the search agent) is evaluated
    without bias.
"""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import pytest

from negpluribus import fast
from negpluribus.abstraction import EquityBucketer
from negpluribus.agents.base import RandomAgent
from negpluribus.agents.blueprint import BlueprintAgent
from negpluribus.cards import Deck
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import ActionType, HandState, Street, raise_to
from negpluribus.eval import aivat as A
from negpluribus.eval import aivat_values as V
from negpluribus.table import play_hand

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "AivatEvaluator"), reason="C++ core with AIVAT not built")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
THREADS = int(os.environ.get("NEGPLURIBUS_TEST_THREADS", "2"))  # evaluator threads in these tests


@pytest.fixture(scope="module")
def game():
    from negpluribus.fast.trainer import core_bucketer
    from negpluribus.eval.aivat_fast import make_game

    p = os.path.join(DATA, "buckets_3p_15bb_flop.json")
    bk = EquityBucketer.load(p) if os.path.exists(p) else EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)
    spec = GameSpec(n_players=2, stack_bb=30, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2,
                    preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    t = MCCFRTrainer(spec, bk, seed=3, backend="cpp", threads=1).train(6000)
    bp = t.blueprint()
    cbk = core_bucketer(bk)
    g = make_game(spec, cbk, bp)
    rt = core.AivatRootTable.build(g, 2, 11, THREADS)
    return {"spec": spec, "bk": bk, "cbk": cbk, "bp": bp, "game": g, "root": rt}


def play(g, n, opponent, seed):
    """n heads-up hands, x = the blueprint agent in alternating seats, as AivatHand records."""
    spec, bp = g["spec"], g["bp"]
    rng = random.Random(seed)
    x = BlueprintAgent(bp, g["cbk"], spec.grid, seed=seed, name="x")
    y = BlueprintAgent(bp, g["cbk"], spec.grid, seed=seed + 1, name="y") if opponent == "self" else RandomAgent(seed=seed + 1)
    out = []
    for i in range(n):
        order = list(range(52))
        rng.shuffle(order)
        xs = i % 2
        agents = [x, y] if xs == 0 else [y, x]
        x.reset(seed * 100003 + i)
        y.reset(seed * 7919 + i)
        rec = play_hand(agents, list(spec.stacks), button=0, sb=spec.sb, bb=spec.bb, deck=Deck.from_order(order), max_street=spec.max_street)
        out.append(A.AivatHand(hand_id=i, stacks=tuple(spec.stacks), button=0, sb=spec.sb, bb=spec.bb, known_seat=xs,
                               holes=(tuple(rec.hole_cards[0]), tuple(rec.hole_cards[1])), board=tuple(rec.board),
                               actions=tuple((int(e.action.type), int(e.action.amount)) for e in rec.events), net=rec.net[xs]))
    return out


# ------------------------------------------------------------------------------ building blocks
def test_streams_and_laws_match_the_core():
    rng = random.Random(1)
    for _ in range(50):
        parts = [rng.getrandbits(64) for _ in range(rng.randrange(1, 6))]
        assert A.stream_seed(*parts) == core.aivat_stream_seed(parts)
        s = A.stream_seed(*parts)
        r = A.CounterRng(s)
        assert [r.uniform() for _ in range(5)] == core.aivat_uniforms(s, 5)
        probs = [rng.random() for _ in range(rng.randrange(1, 7))]
        tot = sum(probs)
        probs = [p / tot for p in probs]
        assert A.sampling_law(probs) == core.aivat_sampling_law(probs)
        p = rng.random()
        assert A.coin_prob(p) == core.aivat_coin_prob(p)


def test_sampling_law_is_the_agents_rule():
    """P(choice = i) of BlueprintAgent.act's loop, counted exactly on a grid of r values."""
    for probs in ([0.2, 0.3, 0.5], [0.33333, 0.33333, 0.33333], [0.50001, 0.5, 0.0], [0.0, 1.0], [1.0]):
        law = A.sampling_law(probs)
        assert abs(sum(law) - 1.0) < 1e-15
        counts = [0] * len(probs)
        n = 200000
        for k in range(n):
            r = (k + 0.5) / n
            acc, choice = 0.0, len(probs) - 1
            for i, p in enumerate(probs):
                acc += p
                if r < acc:
                    choice = i
                    break
            counts[choice] += 1
        for c, p in zip(counts, law):
            assert abs(c / n - p) < 2e-5, (probs, law, counts)


def _random_raise_events(spec, rng, n):
    """Engine events of random raises (every size between the min raise and the all-in)."""
    out = []
    while len(out) < n:
        order = list(range(52))
        rng.shuffle(order)
        st = spec.new_hand(order, button=rng.randrange(2))
        while not st.is_terminal and len(out) < n:
            obs = st.observe(st.current_player)
            if obs.can_raise and rng.random() < 0.6:
                ev = st.apply(raise_to(rng.randint(obs.min_raise_to, obs.max_raise_to)))
                out.append(ev)
            else:
                st.apply(spec.grid.to_concrete(obs, "c"))
    return out


def test_translation_outcomes_are_from_concrete_with_its_coin():
    """Tokens: from_concrete with the coin at 0 and just below 1; probability: the fraction of the
    2^53 coin values giving the low token, found by bisection on a stubbed coin."""
    spec = GameSpec(n_players=2, stack_bb=100, max_street=Street.RIVER, preflop_fracs=(0.5, 1.0, 3.0),
                    postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3)
    grid = spec.grid
    rng = random.Random(4)
    n_random = 0
    for ev in _random_raise_events(spec, rng, 400):
        outs = A.translation_outcomes(grid, ev)
        lo = grid.from_concrete(ev, ev.all_in, A._StubRng(0.0))
        hi = grid.from_concrete(ev, ev.all_in, A._StubRng(1.0 - 2.0 ** -53))
        if lo == hi:
            assert outs == [(lo, 1.0)]
            continue
        n_random += 1
        a, b = 0, 1 << 53  # smallest k with from_concrete(k / 2^53) == hi
        while a < b:
            mid = (a + b) // 2
            if grid.from_concrete(ev, ev.all_in, A._StubRng(mid / 2.0 ** 53)) == hi:
                b = mid
            else:
                a = mid + 1
        assert outs == [(lo, a / 2.0 ** 53), (hi, 1.0 - a / 2.0 ** 53)], (outs, lo, hi, a)
    assert n_random > 100


def test_the_model_is_the_blueprint_agent(game):
    """For hands the agent played: the model's rows for the agent's actual hole contain, for the
    coin outcome the agent drew, exactly the law of the probabilities the agent looked up."""
    spec, bp = game["spec"], game["bp"]
    model = A.BlueprintAgentModel(bp, game["cbk"], spec.grid)
    rng = random.Random(7)
    checked = 0

    class Recorder:
        def __init__(self, inner):
            self.inner, self.calls = inner, []

        def policy(self, key, legal):
            p = self.inner.policy(key, legal)
            self.calls.append((key, list(legal), p))
            return p

    for i in range(60):
        rec_bp = Recorder(bp)
        x = BlueprintAgent(rec_bp, game["cbk"], spec.grid, seed=i, name="x")
        y = RandomAgent(seed=1000 + i)
        order = list(range(52))
        rng.shuffle(order)
        xs = i % 2
        st = spec.new_hand(order, button=0)
        branches = [A.Branch(1.0, [], A.disjoint(st.players[1 - xs].hole).astype(float))]
        x.reset(i)
        while not st.is_terminal:
            seat = st.current_player
            if seat == xs:
                obs = st.observe(seat)
                n0 = len(rec_bp.calls)
                act = x.act(obs)
                key, legal, probs = rec_bp.calls[n0]
                hist = key.split("|", 4)[4]
                acts, sig = model.decision(st, xs, branches)
                ci = A.combo_index(obs.hole)
                match = [s for br, s in zip(branches, sig) if br.history() == hist]
                assert len(match) == 1, (hist, [b.history() for b in branches])
                row = match[0][ci]
                expect = np.zeros(len(acts))
                if probs is None:
                    expect[acts.index(spec.grid.to_concrete(obs, "c"))] = 1.0
                else:
                    for n, p in zip(legal, A.sampling_law(list(probs))):
                        expect[acts.index(spec.grid.to_concrete(obs, n))] += p
                assert np.array_equal(row, expect)
                assert act in acts
                checked += 1
            else:
                act = y.act(st.observe(seat))
            ev = st.apply(act)
            branches = A.split_branches(branches, int(ev.street), A.translation_outcomes(spec.grid, ev))
    assert checked > 60


# --------------------------------------------------------------------------- C++ = reference
def test_cpp_equals_the_python_reference(game):
    from negpluribus.eval.aivat_fast import FastAivat, TERM_NAMES

    spec, bp = game["spec"], game["bp"]
    v0, v1 = game["root"].values
    rt = V.RootTable(V.pair_orbits(), np.array([v0, v1]))
    spv = V.SelfPlayValues(bp, game["cbk"], spec.grid, root_table=rt, rollouts={0: 1, 1: 1, 2: 1}, eq_samples=30, seed=5)
    fa = FastAivat(game["game"], (1, 1, 1), 30, 5, game["root"])
    model = A.BlueprintAgentModel(bp, game["cbk"], spec.grid)
    kinds = set()
    for h in play(game, 10, "random", 21) + play(game, 4, "self", 22):
        tr = []
        r = A.aivat_hand(h, model, spv, trace=tr)
        c = fa.evaluate(h, trace=True)
        assert [n for n, _ in tr] == [n for n, _ in c["trace"]]
        for (n, vp), (_, vc) in zip(tr, c["trace"]):
            assert np.array_equal(vp, np.array(vc)), (h.hand_id, n)
        assert r.net == c["net"] == h.net
        assert abs(r.value - c["value"]) <= 1e-9 * (1 + abs(r.value))
        assert [t.kind for t in r.terms] == [TERM_NAMES[k] for k, _, _, _ in c["terms"]]
        for t, (_, _, _, v) in zip(r.terms, c["terms"]):
            assert abs(t.value - v) <= 1e-9 * (1 + abs(t.value))
        kinds |= {t.kind for t in r.terms}
    assert spv.stats["rollouts"] > 0 and spv.stats["river_trees"] > 0
    assert {"root", "seat", "x", "flop", "turn", "river"} <= kinds
    # root-table entries recomputed by the reference: a class's value is computed on its first pair in
    # scan order (c, then d)
    orb = V.pair_orbits()
    flat = orb.ravel()
    spv_root = V.SelfPlayValues(bp, game["cbk"], spec.grid, seed=11)
    for ci, di in ((5, 700), (1200, 3)):
        o = int(orb[ci, di])
        first = int(np.nonzero(flat == o)[0][0])
        rc, rd = divmod(first, A.N_COMBOS)
        assert V.root_rollout_value(spv_root, spec, 1, rc, rd, o, 2) == v1[o]
        assert V.root_rollout_value(spv_root, spec, 0, rc, rd, o, 2) == v0[o]


# ------------------------------------------------------------------------------ unbiasedness
def _z(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean() / (xs.std(ddof=1) / math.sqrt(len(xs))))


def test_unbiased_in_self_play_and_against_a_random_bettor(game):
    from negpluribus.eval.aivat_fast import FastAivat, TERM_NAMES

    fa = FastAivat(game["game"], (1, 2, 2), 50, 9, game["root"])
    # self-play, alternating seats: the exact value is 0
    hands = play(game, 1500, "self", 31)
    res = fa.evaluate_many(hands, THREADS)
    assert all(not r["trace"] for r in res)
    av = np.array([r["value"] for r in res]) / 100
    raw = np.array([h.net for h in hands]) / 100
    assert abs(_z(av)) < 4.0, (av.mean(), av.std())
    assert av.std() < 0.35 * raw.std(), (av.std(), raw.std())  # variance reduction (self-play: large)
    by_kind = {}
    for r in res:
        for k, _, _, v in r["terms"]:
            by_kind.setdefault(TERM_NAMES[k], []).append(v)
    for k, v in by_kind.items():
        if k != "seat" and len(v) > 30:
            assert abs(_z(v)) < 4.0, (k, np.mean(v), len(v))
    # against a random bettor: E[AIVAT - net] = 0
    hands = play(game, 1000, "random", 32)
    res = fa.evaluate_many(hands, THREADS)
    diff = np.array([r["value"] - h.net for r, h in zip(res, hands)])
    assert abs(_z(diff)) < 4.0, (diff.mean(), diff.std())
    by_kind = {}
    for r in res:
        for k, _, _, v in r["terms"]:
            by_kind.setdefault(TERM_NAMES[k], []).append(v)
    for k, v in by_kind.items():
        if k != "seat" and len(v) > 30:
            assert abs(_z(v)) < 4.0, (k, np.mean(v), len(v))


def test_every_term_has_mean_zero_for_an_arbitrary_value_function(game):
    """Lemma 1 for ANY u: pseudo-random values with exact expectations over the chance nodes."""
    spec, bp = game["spec"], game["bp"]
    model = A.BlueprintAgentModel(bp, game["cbk"], spec.grid)
    vals = V.AdditiveCardValues(scale=300.0, seed=3)
    by_kind = {}
    diffs = []
    for h in play(game, 250, "random", 41):
        r = A.aivat_hand(h, model, vals)
        diffs.append(r.value - r.net)
        for t in r.terms:
            by_kind.setdefault(t.kind, []).append(t.value)
    for k, v in by_kind.items():
        if len(v) > 30:
            assert abs(_z(v)) < 4.0, (k, np.mean(v), len(v))
    assert abs(_z(diffs)) < 4.0


# ------------------------------------------------------------------ logged range strategies (section 7)
def play_logged(g, n, seed):
    """x = an agent that plays like the blueprint (deterministic translation) but samples from its quantized
    rows for every hole, logged as LoggedRows (what docs/aivat.md section 7 asks of the search agent);
    y = a random bettor."""
    spec, bp = g["spec"], g["bp"]
    model = A.BlueprintAgentModel(bp, g["cbk"], spec.grid)
    rng = random.Random(seed)
    y = RandomAgent(seed=seed + 1)
    out = []
    for i in range(n):
        order = list(range(52))
        rng.shuffle(order)
        xs = i % 2
        st = spec.new_hand(order, button=0)
        y.reset(seed * 7919 + i)
        branches = [A.Branch(1.0, [], A.disjoint(st.players[1 - xs].hole).astype(float))]
        rows, acts_all = [], []
        while not st.is_terminal:
            seat = st.current_player
            if seat == xs:
                acts, sig = model.decision(st, xs, branches)
                lr = A.LoggedRows.from_probs([(int(a.type), int(a.amount)) for a in acts], sig[0], list(st.board))
                m = lr.matrix(list(st.board))
                row = np.rint(m[A.combo_index(st.players[xs].hole)] * A.Q_LOG).astype(int)
                u = rng.randrange(A.Q_LOG)
                act = acts[int(np.searchsorted(np.cumsum(row), u, side="right"))]
                rows.append(lr)
            else:
                act = y.act(st.observe(seat))
            acts_all.append((int(act.type), int(act.amount)))
            ev = st.apply(act)
            branches = A.split_branches(branches, int(ev.street), [(spec.grid.from_concrete(ev, ev.all_in, None), 1.0)])
        rec = st.record()
        out.append(A.AivatHand(hand_id=i, stacks=tuple(spec.stacks), button=0, sb=spec.sb, bb=spec.bb, known_seat=xs,
                               holes=(tuple(rec.hole_cards[0]), tuple(rec.hole_cards[1])), board=tuple(rec.board),
                               actions=tuple(acts_all), net=rec.net[xs], x_rows=tuple(rows)))
    return out


def test_logged_rows_roundtrip_and_quantization():
    rng = np.random.default_rng(1)
    board = [3, 17, 40]
    p = rng.random((A.N_COMBOS, 4)) ** 3
    lr = A.LoggedRows.from_probs([(0, 0), (1, 0), (2, 300), (2, 900)], p, board)
    assert A.LoggedRows.from_json(json.loads(json.dumps(lr.to_json()))) == lr
    m = lr.matrix(board)
    ok = A.disjoint(board)
    assert (np.rint(m[ok] * A.Q_LOG).astype(np.int64).sum(axis=1) == A.Q_LOG).all()
    pn = p[ok] / p[ok].sum(axis=1, keepdims=True)
    assert np.abs(m[ok] - pn).max() <= 1.0 / A.Q_LOG
    assert not m[~ok].any()


def test_logged_rows_cpp_equals_reference_and_stay_unbiased(game):
    from negpluribus.eval.aivat_fast import FastAivat

    spec, bp = game["spec"], game["bp"]
    v0, v1 = game["root"].values
    rt = V.RootTable(V.pair_orbits(), np.array([v0, v1]))
    spv = V.SelfPlayValues(bp, game["cbk"], spec.grid, root_table=rt, rollouts={0: 1, 1: 1, 2: 1}, eq_samples=30, seed=5)
    fa1 = FastAivat(game["game"], (1, 1, 1), 30, 5, game["root"])
    model = A.BlueprintAgentModel(bp, game["cbk"], spec.grid)
    for h in play_logged(game, 8, 51):
        tr = []
        r = A.aivat_hand(h, model, spv, trace=tr)
        c = fa1.evaluate(h, trace=True)
        assert [n for n, _ in tr] == [n for n, _ in c["trace"]]
        for (n, vp), (_, vc) in zip(tr, c["trace"]):
            assert np.array_equal(vp, np.array(vc)), (h.hand_id, n)
        assert abs(r.value - c["value"]) <= 1e-9 * (1 + abs(r.value))
    fa = FastAivat(game["game"], (1, 2, 2), 50, 9, game["root"])
    hands = play_logged(game, 800, 53)
    res = fa.evaluate_many(hands, THREADS)
    assert all(not r["trace"] for r in res)
    diff = np.array([r["value"] - h.net for r, h in zip(res, hands)])
    assert abs(_z(diff)) < 4.0, (diff.mean(), diff.std())
