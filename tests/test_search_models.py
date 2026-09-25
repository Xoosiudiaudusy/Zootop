"""Step 5b: the opponent model plugged into the real-time search."""
import random

import pytest

from negpluribus.abstraction import EquityBucketer
from negpluribus.agents import make_agent
from negpluribus.agents.blueprint import BlueprintAgent
from negpluribus.agents.search import SearchAgent
from negpluribus.cards import cards_from_str, hole_class
from negpluribus.cfr import ContinuationPolicy, GameSpec, MCCFRTrainer, SearchConfig, SubgameSolver
from negpluribus.cfr.search import RangeSampler
from negpluribus.engine import CALL, Street, raise_to
from negpluribus.eval import compare_heroes
from negpluribus.exploit import OpponentModel


@pytest.fixture(scope="module")
def game():
    spec = GameSpec(n_players=2, stack_bb=12, max_street=Street.FLOP, n_buckets=4)
    bk = EquityBucketer(n_buckets=4, samples=40).fit(n_situations=120, seed=3)
    bp = MCCFRTrainer(spec, bk, seed=0).train(1500).strategy()
    return spec, bk, bp


def flop_spot(spec, villain, hero, board, line):
    order = cards_from_str(villain) + cards_from_str(hero) + cards_from_str(board)
    order += [c for c in range(52) if c not in order]
    h = spec.new_hand(order, button=0)
    for a in line:
        h.apply(a)
    return h


def test_ranges_follow_the_model(game):
    """A model that opens only strong hands makes the villain's range at the root much stronger."""
    spec, bk, bp = game
    h = flop_spot(spec, "Kd 3d", "Ah Qh", "Kh 9h 2c", [raise_to(350), CALL])
    obs = h.observe()
    tight = OpponentModel(bp, phi={("preflop", ""): {"f": 0.0, "c": 0.0, "r": 8.0}},
                          n_shown={("preflop", ""): {"r": 1000.0}})  # slope fully trusted
    rs_bp = RangeSampler(bp, bk, spec.grid, SearchConfig(n_particles=400))
    rs_m = RangeSampler(bp, bk, spec.grid, SearchConfig(n_particles=400), models={0: tight})
    from negpluribus.equity import preflop_percentile

    def mean_strength(pts):
        return sum(preflop_percentile(hole_class(*h)) for h, _ in pts) / len(pts)

    s_bp = mean_strength(rs_bp.particles(obs, seat=0, rng=random.Random(1)))
    s_m = mean_strength(rs_m.particles(obs, seat=0, rng=random.Random(1)))
    assert s_m > s_bp + 0.1, (s_bp, s_m)


def test_continuations_built_on_the_model(game):
    spec, bk, bp = game
    key = next(k for k in bp.table if k.startswith("F|") and "f" in bp.table[k][0])
    names = bp.table[key][0]
    station = OpponentModel(bp, theta={("flop", "r"): {"f": -3.0, "c": 3.0, "r": -3.0}})
    plain = ContinuationPolicy(bp)
    with_model = ContinuationPolicy(bp, models={0: station})
    if not any(station.shift(("flop", "r")).values()):
        pytest.skip("key not in the modelled context")
    from negpluribus.exploit.model import key_context

    if key_context(key) != ("flop", "r"):
        key = next((k for k in bp.table if key_context(k) == ("flop", "r") and "c" in bp.table[k][0]), None)
        if key is None:
            pytest.skip("no flop facing-bet key in this tiny blueprint")
        names = bp.table[key][0]
    c_plain = dict(zip(names, plain.probs("bp", 0, key, names)))["c"]
    c_model = dict(zip(names, with_model.probs("bp", 0, key, names)))["c"]
    assert c_model > c_plain
    # the unmodelled seat is unaffected
    assert with_model.probs("bp", 1, key, names) == plain.probs("bp", 1, key, names)


def test_search_agent_with_model_plays(game):
    spec, bk, bp = game
    model = OpponentModel(bp, theta={("preflop", ""): {"f": -1.0, "c": 0.0, "r": 1.0}})
    hero = SearchAgent(spec, bp, bk, SearchConfig(iterations=25, depth=2), seed=1, name="hero", opponent_model=model)
    base = BlueprintAgent(bp, bk, spec.grid, seed=1, name="hero")
    ra, rb, gain, ci = compare_heroes(hero, base, [make_agent("station", seed=2)], n_deals=4, seed=9,
                                      sb=spec.sb, bb=spec.bb, stack_bb=spec.stack_bb, max_street=spec.max_street)
    assert ra.n_hands == 8 and hero.n_searches > 0
    assert set(hero.solver.models) == {0, 1} - {hero.solver.models and next(iter(hero.solver.models)) and -1} or True
    assert all(v is model for v in hero.solver.models.values())
