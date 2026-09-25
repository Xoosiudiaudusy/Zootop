import random

import pytest

from negpluribus.abstraction import BetGrid, EquityBucketer, history_string, infoset_key
from negpluribus.agents import make_agent
from negpluribus.agents.blueprint import BlueprintAgent
from negpluribus.cards import ALL_HOLE_CLASSES
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.cfr.strategy import BlueprintStrategy
from negpluribus.engine import CALL, FOLD, HandState, Street, raise_to
from negpluribus.eval import duplicate_match


# ------------------------------------------------------------ engine bits
def test_max_street_preflop_runs_out_board():
    h = HandState([1000, 1000], button=0, seed=1, max_street=Street.PREFLOP)
    h.apply(CALL)
    h.apply(CALL)
    assert h.is_terminal and len(h.board) == 5 and len(h.showdown_seats) == 2


def test_max_street_flop_stops_after_flop_betting():
    h = HandState([2000, 2000], button=0, seed=2, max_street=Street.FLOP)
    h.apply(CALL)
    h.apply(CALL)
    assert h.street == Street.FLOP
    h.apply(CALL)
    h.apply(CALL)
    assert h.is_terminal and len(h.board) == 5


def test_clone_is_independent_and_shares_future_cards():
    h = HandState([2000, 2000], button=0, seed=3)
    c = h.clone()
    c.apply(raise_to(300))
    assert h.current_player == 0 and h.pot == 150
    assert c.pot == 400
    # same deck order: both see the same flop if both go there
    h.apply(CALL)
    h.apply(CALL)
    c.apply(CALL)
    assert h.board == c.board


# ---------------------------------------------------------- push/fold toy
@pytest.fixture(scope="module")
def pushfold():
    spec = GameSpec(n_players=2, stack_bb=10, max_street=Street.PREFLOP, preflop_fracs=(1.0,))
    trainer = MCCFRTrainer(spec, seed=0, linear=True).train(30000)  # ~8s
    return spec, trainer


def test_pushfold_strategy_is_sane(pushfold):
    spec, trainer = pushfold
    def opening(cls):
        node = trainer.nodes[f"P|BTN/SB|2|b{ALL_HOLE_CLASSES.index(cls)}|"]
        return dict(zip(node.actions, node.average_strategy()))
    aa, junk = opening("AA"), opening("72o")
    assert aa["f"] < 0.05
    assert aa["r1"] + aa["a"] > 0.9
    assert junk["f"] > aa["f"] + 0.3
    # premium hands are played at least as aggressively as junk overall
    assert (aa["r1"] + aa["a"]) > (junk["r1"] + junk["a"])


def test_blueprint_beats_random_and_caller(pushfold):
    spec, trainer = pushfold
    strat = trainer.strategy()
    for vil in ("random", "caller", "tag"):
        hero = BlueprintAgent(strat, trainer.bucketer, spec.grid, seed=1)
        res = duplicate_match(
            hero, [make_agent(vil, seed=5)], n_deals=3000, seed=3,
            sb=spec.sb, bb=spec.bb, stack_bb=spec.stack_bb, max_street=spec.max_street,
        )
        # an all-in game is very noisy even with duplicate deals: demand significance only
        # against the weakest opponent, and "not significantly losing" against the others
        # (exploitability, not bb/100 vs dummies, is the real ruler: see test_exploit.py).
        # 3000 deals (+/-11 bb/100): since the translation follows Ganzfried & Sandholm (all-in
        # as upper neighbour, 24.09) the edge vs random is ~+41 instead of ~+66 (measured on
        # 40k hands); 600 deals (+/-24) no longer resolved it.
        if vil == "random":
            assert res.bb100 - res.ci95 > 0, (res.bb100, res.ci95)
        else:
            assert res.bb100 + res.ci95 > 0, (vil, res.bb100, res.ci95)
        assert hero.fallback_rate < 0.05


def test_strategy_roundtrip_and_checkpoint(tmp_path, pushfold):
    spec, trainer = pushfold
    strat = trainer.strategy()
    p = tmp_path / "bp.json"
    strat.save(str(p))
    s2 = BlueprintStrategy.load(str(p))
    key = next(iter(strat.table))
    names, probs = strat.table[key]
    assert s2.policy(key, names) == pytest.approx(probs, abs=1e-4)
    assert s2.policy("nope", ["f", "c"]) is None
    ck = tmp_path / "ck.json"
    trainer.save_checkpoint(str(ck))
    t2 = MCCFRTrainer(spec, seed=1).load_checkpoint(str(ck))
    assert t2.iteration == trainer.iteration and len(t2.nodes) == len(trainer.nodes)
    t2.train(5)
    assert t2.iteration == trainer.iteration + 5


# --------------------------------------------------- flop game smoke test
def test_flop_game_trains_and_plays():
    spec = GameSpec(n_players=3, stack_bb=15, max_street=Street.FLOP, n_buckets=4)
    bk = EquityBucketer(n_buckets=4, samples=40).fit(n_situations=120, seed=2)
    trainer = MCCFRTrainer(spec, bk, seed=0).train(60)
    assert any(k.startswith("F|") for k in trainer.nodes)
    hero = BlueprintAgent(trainer.strategy(), bk, spec.grid, seed=1)
    res = duplicate_match(
        hero, [make_agent("caller", seed=1), make_agent("random", seed=2)], n_deals=5, seed=1,
        sb=spec.sb, bb=spec.bb, stack_bb=spec.stack_bb, max_street=spec.max_street,
    )
    assert res.n_hands == 15


def test_needs_fitted_bucketer_for_postflop():
    with pytest.raises(ValueError):
        MCCFRTrainer(GameSpec(max_street=Street.FLOP))


# ------------------------------------------- pseudo-harmonic wiring (finding #1)
def test_event_rng_makes_translation_consistent_within_hand():
    grid = BetGrid()
    h = HandState([10_000] * 2, button=0)
    h.apply(raise_to(230))  # increment 130 / pot-after-call 200 = 0.65 -> p(r0.5) = 0.64: a genuinely random spot
    h.apply(CALL)
    h.apply(CALL)
    # deterministic mode always maps 0.6 to r0.5
    assert history_string(h.events, grid) == "r0.5 c/c"
    # randomized: same nonce -> same string every time it is asked, on any later street
    def rng_for(nonce):
        return lambda i: random.Random(nonce * 1_000_003 + i)
    seen = {history_string(h.events, grid, rng_for(n)) for n in range(200)}
    assert seen == {"r0.5 c/c", "r1 c/c"}  # both sides of the coin occur across hands
    for n in range(20):
        a = history_string(h.events, grid, rng_for(n))
        b = history_string(h.events, grid, rng_for(n))
        assert a == b
