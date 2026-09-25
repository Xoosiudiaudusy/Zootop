import pytest

from negpluribus.abstraction import EquityBucketer
from negpluribus.agents import make_agent
from negpluribus.agents.blueprint import BlueprintAgent
from negpluribus.agents.search import SearchAgent
from negpluribus.cards import cards_from_str
from negpluribus.cfr import ContinuationPolicy, GameSpec, MCCFRTrainer, SearchConfig, SubgameSolver
from negpluribus.cfr.search import CONTINUATIONS, RangeSampler
from negpluribus.engine import CALL, Street, raise_to
from negpluribus.eval import compare_heroes, duplicate_match


@pytest.fixture(scope="module")
def game():
    """Tiny flop game with a quickly trained blueprint (shared by the tests below)."""
    spec = GameSpec(n_players=2, stack_bb=12, max_street=Street.FLOP, n_buckets=4)
    bk = EquityBucketer(n_buckets=4, samples=40).fit(n_situations=120, seed=3)
    trainer = MCCFRTrainer(spec, bk, seed=0).train(1500)
    return spec, bk, trainer.strategy()


def flop_spot(spec, villain, hero, board, line):
    order = cards_from_str(villain) + cards_from_str(hero) + cards_from_str(board)
    order += [c for c in range(52) if c not in order]
    h = spec.new_hand(order, button=0)  # seat0 = BTN/SB (villain), seat1 = BB (hero)
    for a in line:
        h.apply(a)
    return h


# ---------------------------------------------------------- continuations
def test_continuation_bias_math(game):
    spec, bk, bp = game
    cont = ContinuationPolicy(bp, bias_factor=5.0)
    legal = ["f", "c", "r0.5", "r1", "a"]
    key = next(k for k in bp.table if k.startswith("F|") and bp.table[k][0] == legal)
    base = bp.policy(key, legal)
    raise_kind = cont.probs("raise", 0, key, legal)
    call_kind = cont.probs("call", 0, key, legal)
    fold_kind = cont.probs("fold", 0, key, legal)
    assert cont.probs("bp", 0, key, legal) == base
    for p in (raise_kind, call_kind, fold_kind):
        assert abs(sum(p) - 1) < 1e-9
    # all raise sizes (r*, a) are boosted together; fold/call only their own action
    r_mass = lambda p: p[2] + p[3] + p[4]  # noqa: E731
    assert r_mass(raise_kind) >= r_mass(base) - 1e-9
    assert call_kind[1] >= base[1] - 1e-9 and fold_kind[0] >= base[0] - 1e-9
    # unknown key -> uniform base, bias still applies
    u = cont.probs("call", 0, "no-such-key", legal)
    assert u[1] == pytest.approx(5 / 9)


def test_bias_factors_hook_is_per_seat(game):
    spec, bk, bp = game

    class StationAware(ContinuationPolicy):
        def bias_factors(self, seat):
            return {"fold": 1.0, "call": 20.0, "raise": 1.0} if seat == 0 else super().bias_factors(seat)

    cont = StationAware(bp)
    legal = ["f", "c", "r0.5", "r1", "a"]
    assert cont.probs("call", 0, "no-such-key", legal)[1] > cont.probs("call", 1, "no-such-key", legal)[1]


# ---------------------------------------------------------------- ranges
def test_range_particles_respect_blueprint_reach(game):
    spec, bk, bp = game
    h = flop_spot(spec, "Kd 3d", "Ah Qh", "Kh 9h 2c", [raise_to(350), CALL])
    obs = h.observe()
    import random

    rs = RangeSampler(bp, bk, spec.grid, SearchConfig(n_particles=200))
    pts = rs.particles(obs, seat=0, rng=random.Random(1))
    assert len(pts) == 200
    assert all(len(hole) == 2 and hole[0] not in obs.board and hole[1] not in obs.board for hole, _ in pts)
    # the villain raised preflop; hands the blueprint opens with should dominate the particle set
    from negpluribus.cards import ALL_HOLE_CLASSES, hole_class

    def open_prob(cls):
        key = f"P|BTN/SB|2|b{ALL_HOLE_CLASSES.index(cls)}|"
        names, probs = bp.table[key]
        return dict(zip(names, probs)).get("r1", 0.0)

    freq = {}
    for hole, _ in pts:
        freq[hole_class(*hole)] = freq.get(hole_class(*hole), 0) + 1
    seen = sorted(freq, key=lambda c: -freq[c])[:20]
    avg_seen = sum(open_prob(c) for c in seen) / len(seen)
    avg_all = sum(open_prob(c) for c in ALL_HOLE_CLASSES) / 169
    assert avg_seen > avg_all


# ---------------------------------------------------------------- solver
def test_solver_returns_distribution_over_legal_actions(game):
    spec, bk, bp = game
    h = flop_spot(spec, "Kd 3d", "Ah Qh", "Kh 9h 2c", [raise_to(350), CALL])
    obs = h.observe()
    legal = spec.grid.abstract_actions(obs)
    for depth in (1, 2, None):
        solver = SubgameSolver(spec, bp, bk, SearchConfig(iterations=60, depth=depth), seed=1)
        probs = solver.solve(obs)
        assert probs is not None and len(probs) == len(legal)
        assert abs(sum(probs) - 1) < 1e-9
        assert solver.last_root_key.startswith("F|BB|2|")
        leaf_nodes = [k for k in solver.nodes if k.startswith("LEAF|")]
        if depth is None:
            assert not leaf_nodes  # solved to the end of the hand: no continuation choices needed
        else:
            assert leaf_nodes and all(solver.nodes[k].actions == list(CONTINUATIONS) for k in leaf_nodes)


def test_solver_deals_hidden_cards_from_ranges_not_actual(game):
    """Villain's actual cards must never leak: the rebuilt states use sampled holes."""
    spec, bk, bp = game
    h = flop_spot(spec, "Kd 3d", "Ah Qh", "Kh 9h 2c", [raise_to(350), CALL])
    obs = h.observe()
    solver = SubgameSolver(spec, bp, bk, SearchConfig(iterations=1, n_particles=30), seed=2)
    seen = set()
    orig = solver._rebuild

    def spy(o, holes, stacks):
        seen.add(holes[0])
        return orig(o, holes, stacks)

    solver._rebuild = spy  # type: ignore[assignment]
    for _ in range(20):
        solver.solve(obs)
    assert len(seen) > 1
    assert all(c not in obs.board for hole in seen for c in hole)


# ---------------------------------------------------------------- agent
def test_search_agent_plays_and_reports_last_decision(game):
    spec, bk, bp = game
    hero = SearchAgent(spec, bp, bk, SearchConfig(iterations=30, depth=2), seed=1)
    # 20 deals: in this 12bb game most hands end preflop, and whether a given 12-hand match
    # reaches a flop decision at all depends on the exact blueprint numbers (measured 0 of 12
    # hands after the 2026-09-23 E[HS] change, 5 flop decisions in 40 hands)
    res = duplicate_match(
        hero, [make_agent("tag", seed=4)], n_deals=20, seed=1,
        sb=spec.sb, bb=spec.bb, stack_bb=spec.stack_bb, max_street=spec.max_street,
    )
    assert res.n_hands == 40
    assert hero.n_searches > 0 and hero.n_search_fallback <= hero.n_searches
    if hero.last is not None:
        key, legal, bp_probs, s_probs = hero.last
        assert key.startswith("F|") and len(s_probs) == len(legal)


def test_compare_search_vs_blueprint_runs(game):
    spec, bk, bp = game
    a = SearchAgent(spec, bp, bk, SearchConfig(iterations=20, depth=1), seed=1, name="search")
    b = BlueprintAgent(bp, bk, spec.grid, seed=1, name="blueprint")
    ra, rb, gain, ci = compare_heroes(
        a, b, [make_agent("nit", seed=2)], n_deals=4, seed=9,
        sb=spec.sb, bb=spec.bb, stack_bb=spec.stack_bb, max_street=spec.max_street,
    )
    assert ra.n_hands == rb.n_hands == 8
    assert isinstance(gain, float) and ci >= 0
