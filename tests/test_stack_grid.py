"""The depth grid (negpluribus/agents/stack_grid.py, docs/stack_grid_design.md): one agent per stack depth, picked
per hand by the effective stack.

  * the rule against a direct computation: the nearest point by ratio (a boundary to the deeper point), explicit
    bounds, stacks above and below the grid, the manifest's validation;
  * the effective stack: heads-up min(own, other); 3-max min(own, max(live opponents)) and the review's alternatives
    (min and median of the live opponents), folded opponents left out, the spec's stack when a observation has none;
  * the wrapper: the point chosen on the first decision and kept for the hand, the LRU of loaded points with counts
    that survive eviction, decision_info on every decision (the selection on the first one), end_hand to the agent
    that played the hand;
  * on tiny trained blueprints: at a grid point the grid plays bit for bit like the point's own blueprint agent, and
    after a reset a hand does not depend on the points loaded or played before; a point's blueprint of another depth
    is refused; the search factory shares one C++ bucketer between the points;
  * reading a log back (scripts/aivat_eval.py --blueprints): the point per hand from the hero's first decision, from
    the rule for a hand without a hero decision, and a hand at other stacks than the root table's refused.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import random
from collections import Counter
from fractions import Fraction
from types import SimpleNamespace

import pytest

from negpluribus import fast
from negpluribus.abstraction import EquityBucketer
from negpluribus.agents.base import Agent, CallingAgent, RandomAgent
from negpluribus.agents.blueprint import BlueprintAgent
from negpluribus.agents.stack_grid import GridPoint, StackGrid, StackGridAgent, StackRule, depths
from negpluribus.cards import Deck
from negpluribus.cfr.game import GameSpec
from negpluribus.engine import CALL, FOLD, HandState, Street
from negpluribus.table import play_hand

core = fast.core()
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
POINTS = [20, 40, 50, 70, 84, 100, 150, 200]  # the pot16 grid of data/stack_grid_pot16_s0.json


def nearest_by_ratio(points, s):
    """Direct computation: the point with the least |log(s / p)|, the deeper one on a tie (exact: compare s / p with q / s)."""
    s = Fraction(s)
    best = points[0]
    for p in points[1:]:
        # p is at least as near as best iff max(s/p, p/s) <= max(s/best, best/s)
        d_p = max(s / p, Fraction(p) / s)
        d_b = max(s / best, Fraction(best) / s)
        if d_p <= d_b:
            best = p
    return best


# ============================================================ the rule
def test_nearest_log_is_the_nearest_point_by_ratio_with_ties_to_the_deeper():
    rule = StackRule("nearest_log")
    rng = random.Random(1)
    for _ in range(3000):
        s = Fraction(rng.randrange(1, 60000), 100)  # 0.01 .. 600 bb in chips of a 100-chip big blind
        assert POINTS[rule.pick(POINTS, s)] == nearest_by_ratio(POINTS, s), s
    for p in POINTS:
        assert POINTS[rule.pick(POINTS, p)] == p
    # the boundary between 20 and 40 is sqrt(800) = 28.2842...; on it (exactly) the deeper point
    assert POINTS[rule.pick(POINTS, Fraction(2828427, 100000))] == 20
    assert POINTS[rule.pick(POINTS, Fraction(2828428, 100000))] == 40
    assert [25, 100][rule.pick([25, 100], 50)] == 100 and [25, 100][rule.pick([25, 100], Fraction(4999, 100))] == 25
    # outside the grid: the bottom and the top point
    assert POINTS[rule.pick(POINTS, 1)] == 20 and POINTS[rule.pick(POINTS, Fraction(1, 2))] == 20
    assert POINTS[rule.pick(POINTS, 201)] == 200 and POINTS[rule.pick(POINTS, 10_000)] == 200
    # between 150 and 200 the boundary is sqrt(30000) = 173.2
    assert POINTS[rule.pick(POINTS, 173)] == 150 and POINTS[rule.pick(POINTS, Fraction(17321, 100))] == 200


# the rule of data/stack_grid_pot16_s0.json (master 9387b90: step 2's cross matrix; inclusive upper depths)
MANIFEST_RULE = {"kind": "boundaries", "below_grid": 20, "above_grid": 200,
                 "upper": {"20": 28, "40": 46, "50": 59, "70": 76, "84": 92, "100": 110, "122": 135, "150": 173, "200": 10000}}
GRID9 = [20, 40, 50, 70, 84, 100, 122, 150, 200]


def test_boundaries_rule_of_the_manifest_and_its_validation():
    rule = StackRule.from_manifest(MANIFEST_RULE, GRID9)
    rule.validate(GRID9)
    assert rule.upper == (28, 46, 59, 76, 92, 110, 135, 173, 10000)
    want = {5: 20, 20: 20, 28: 20, Fraction(2801, 100): 40, 46: 40, Fraction(4601, 100): 50, 50: 50, 59: 50, 60: 70,
            76: 70, 77: 84, 92: 84, 93: 100, 100: 100, 110: 100, 111: 122, 135: 122, 136: 150, 150: 150, 173: 150,
            174: 200, 200: 200, 10000: 200, 10001: 200}
    for s, p in want.items():
        assert GRID9[rule.pick(GRID9, s)] == p, s
    for p in GRID9:  # every point at its own stack (the acceptance points 50, 100, 150 among them)
        assert GRID9[rule.pick(GRID9, p)] == p
    # below_grid / above_grid are the manifest's (here not the ends of the grid)
    odd = StackRule.from_manifest(dict(MANIFEST_RULE, below_grid=40, above_grid=150), GRID9)
    assert [GRID9[odd.pick(GRID9, s)] for s in (19, 20, 10000, 10001)] == [40, 20, 200, 150]
    # a point without an upper depth is refused, not skipped
    no50 = {k: v for k, v in MANIFEST_RULE["upper"].items() if k != "50"}
    with pytest.raises(ValueError, match=r"no entry for the point\(s\) \[50\]"):
        StackRule.from_manifest(dict(MANIFEST_RULE, upper=no50), GRID9)
    # the entry of an absent point is allowed (its depths go to the next deeper point); an unknown depth is not
    g8 = [s for s in GRID9 if s != 122]
    r8 = StackRule.from_manifest(MANIFEST_RULE, g8, absent=[122])
    r8.validate(g8)
    assert g8[r8.pick(g8, 110)] == 100 and g8[r8.pick(g8, 111)] == 150
    with pytest.raises(ValueError, match="no points of the grid"):
        StackRule.from_manifest(MANIFEST_RULE, g8)
    with pytest.raises(ValueError, match="must map"):
        StackRule.from_manifest({"kind": "boundaries", "upper": [28, 46]}, GRID9)
    for bad in ({**MANIFEST_RULE["upper"], "40": 50}, {**MANIFEST_RULE["upper"], "40": 39}):  # 50 -> 40; 40 -> 50
        with pytest.raises(ValueError, match="must be in"):
            StackRule.from_manifest(dict(MANIFEST_RULE, upper=bad), GRID9).validate(GRID9)
    with pytest.raises(ValueError, match="is not a point"):
        StackRule.from_manifest(dict(MANIFEST_RULE, above_grid=300), GRID9).validate(GRID9)
    with pytest.raises(ValueError, match="takes no 'upper'"):
        StackRule("nearest_log", (28,)).validate(GRID9)
    with pytest.raises(ValueError, match="rule kind"):
        StackRule("linear").validate(GRID9)
    assert "20bb <= 28" in rule.describe(GRID9) and "below the grid 20bb, above it 200bb" in rule.describe(GRID9)


# ============================================================ the effective stack
def test_effective_stack_heads_up_and_three_max_with_the_alternatives():
    bb = 100
    for seat in (0, 1):
        d = depths([5000, 12000], seat, [True, True], bb)
        assert d == {"eff": 50, "min": 50, "median": 50}
    stacks = [3000, 10000, 20000]  # 30 / 100 / 200 bb (docs/stack_depth.md section 6)
    assert depths(stacks, 2, [True] * 3, bb) == {"eff": 100, "min": 30, "median": 65}
    assert depths(stacks, 0, [True] * 3, bb) == {"eff": 30, "min": 30, "median": 30}
    assert depths(stacks, 1, [True] * 3, bb) == {"eff": 100, "min": 30, "median": 100}  # median of 30 and 200 = 115
    assert depths(stacks, 2, [True, False, True], bb) == {"eff": 30, "min": 30, "median": 30}  # the 100bb one folded
    assert depths([3000, 10000, 20000, 4000], 2, [True] * 4, bb)["median"] == 40  # median of 30, 40, 100
    assert depths([3050, 10000], 1, [True, True], bb)["eff"] == Fraction(61, 2)  # exact, not rounded


class Fake(Agent):
    """An inner agent that checks / calls and records what it was asked."""

    made = []

    def __init__(self, point: GridPoint):
        super().__init__(name=f"fake{point.stack_bb}")
        self.point = point
        self.n_decisions = self.n_fallback = self.ends = 0
        self.resets = []
        self.draws = []
        Fake.made.append(self)

    def reset(self, seed=None):
        super().reset(seed)
        self.resets.append(seed)

    def act(self, obs):
        self.n_decisions += 1
        self.draws.append(self.rng.random())
        if self.draws[-1] < 0.25:
            self.n_fallback += 1
        return CALL

    def end_hand(self, record, my_seat):
        self.ends += 1

    def decision_info(self):
        return {"played": "fake", "n": self.n_decisions}


def fake_grid(stacks=(20, 40, 50, 100, 200), **kw):
    pts = [GridPoint(s, f"data/bp{s}.bin") for s in stacks]
    return StackGridAgent(pts, Fake, StackRule("nearest_log"), **kw)


def hand(agents, stacks, button=0, seed=0):
    order = list(range(52))
    random.Random(seed).shuffle(order)
    return play_hand(agents, list(stacks), button, 50, 100, deck=Deck.from_order(order))


def test_the_point_is_chosen_at_the_first_decision_and_kept_for_the_hand():
    g = fake_grid(seed=7)
    infos = []
    orig = g.act

    def spy(obs):
        a = orig(obs)
        infos.append(g.decision_info())
        return a

    g.act = spy
    cases = [((5000, 5000), 50), ((1000, 1000), 20), ((500, 900), 20), ((50000, 60000), 200), ((5000, 30000), 50),
             ((9000, 10000), 100), ((2800, 2800), 20), ((3000, 3000), 40)]  # sqrt(20 * 40) = 28.3
    for k, (stacks, want) in enumerate(cases):
        infos.clear()
        g.reset(100 + k)
        rec = hand([g, CallingAgent(name="caller")], stacks, button=k % 2, seed=k)
        assert infos, "the hero had no decision"
        assert all(i["point"] == want and i["blueprint"] == f"data/bp{want}.bin" and i["played"] == "fake" for i in infos)
        eff = min(stacks) / 100
        assert infos[0]["eff_bb"] == eff and infos[0]["eff_min_bb"] == eff and infos[0]["eff_median_bb"] == eff
        assert infos[0]["point_min"] == infos[0]["point_median"] == want
        assert all("eff_bb" not in i for i in infos[1:])  # the selection is reported once per hand
        assert g.current_point is None  # end_hand closed the hand
        assert len(rec.events) > 2
    assert g.hands_by_point == Counter({20: 3, 50: 2, 200: 1, 100: 1, 40: 1})


def test_three_max_formula_through_the_agent_and_the_spec_stack_fallback():
    g = fake_grid((30, 60, 100, 200), seed=1)
    # 30 / 100 / 200 bb: seat 0 (under the gun, 30bb) and seat 1 (small blind, 100bb) act, then seat 2 (big blind, 200bb)
    for before, eff, want in (([CALL, CALL], (100, 30, 65), (100, 30, 60)),   # median of 30 and 100: 65 -> 60 (< 77.5)
                              ([CALL, FOLD], (30, 30, 30), (30, 30, 30)),     # the 100bb one folded: only the 30bb one
                              ([FOLD, CALL], (100, 100, 100), (100, 100, 100))):
        st = HandState([3000, 10000, 20000], 0, 50, 100, deck=Deck.from_order(list(range(52))))
        for a in before:
            st.apply(a)
        assert st.current_player == 2
        g.reset(3)
        g.act(st.observe(2))
        info = g.decision_info()
        assert (info["eff_bb"], info["eff_min_bb"], info["eff_median_bb"]) == eff
        assert (info["point"], info["point_min"], info["point_median"]) == want
    # no starting stacks in the observation (a foreign wrapper): the fallback stack for every seat
    st = HandState([5000, 5000], 0, 50, 100, deck=Deck.from_order(list(range(52))))
    obs = dataclasses.replace(st.observe(st.current_player), starting_stacks=[])
    g = fake_grid(seed=1, fallback_stack_bb=100)
    g.act(obs)
    assert g.decision_info()["point"] == 100 and g.decision_info()["eff_bb"] == 100
    with pytest.raises(ValueError, match="no starting stacks"):
        fake_grid(seed=1).act(obs)


def test_lru_eviction_keeps_the_counts_and_reset_reaches_every_loaded_agent():
    Fake.made = []
    g = fake_grid(max_loaded=2, seed=5)
    plan = [20, 40, 20, 50, 40, 40, 200, 20]
    loaded_after = []
    for k, s in enumerate(plan):
        g.reset(1000 + k)
        hand([g, CallingAgent(name="caller")], (s * 100, s * 100), button=k % 2, seed=k)
        loaded_after.append(g.loaded_points)
    assert loaded_after == [[20], [20, 40], [40, 20], [20, 50], [50, 40], [50, 40], [40, 200], [200, 20]]
    assert g.loads == Counter({20: 2, 40: 2, 50: 1, 200: 1}) and g.evictions == 4
    assert len(Fake.made) == 6 and len(g.inner_agents()) == 2
    assert g.n_decisions == sum(f.n_decisions for f in Fake.made) and g.n_decisions > len(plan)
    assert g.n_fallback == sum(f.n_fallback for f in Fake.made) and 0 < g.fallback_rate < 1
    # a new agent is reset with the last reset's seed when made; a reset reaches every loaded agent
    assert [f.resets[0] for f in Fake.made] == [1000 + k for k in (0, 1, 3, 4, 6, 7)]
    g.reset(42)
    assert all(f.resets[-1] == 42 for f in g.inner_agents())
    # end_hand went to the agent that played each hand (one per hand)
    assert sum(f.ends for f in Fake.made) == len(plan)
    with pytest.raises(ValueError, match="max_loaded"):
        fake_grid(max_loaded=0)
    with pytest.raises(ValueError, match="sorted"):
        fake_grid((20, 50, 40))


def test_a_new_hand_without_reset_or_end_hand_is_noticed():
    """Replaying hands without reset / end_hand (a driver that never calls them): the point is chosen again when the
    hole cards change or the events are not a continuation."""
    g = fake_grid(seed=2)
    for stacks, want in (((5000, 5000), 50), ((20000, 20000), 200), ((5000, 5000), 50)):
        st = HandState(list(stacks), 0, 50, 100, deck=Deck.from_order(list(range(52))))
        g.act(st.observe(st.current_player))  # the same hole cards every time: the events tell the new hand
        assert g.decision_info()["point"] == want and "eff_bb" in g.decision_info()


# ============================================================ tiny trained blueprints
def small_spec(stack: int) -> GameSpec:
    return GameSpec(n_players=2, stack_bb=stack, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2,
                    preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """Three tiny blueprints (20, 30 and 45bb; one 8-bucket E[HS] bucketer, the same bets) and their manifest in
    <tmp>/data/, with an absent 60bb point."""
    if core is None:
        pytest.skip("C++ core not built")
    from negpluribus.cfr.mccfr import MCCFRTrainer
    from negpluribus.fast.trainer import core_bucketer

    root = tmp_path_factory.mktemp("grid")
    data = root / "data"
    data.mkdir()
    p = os.path.join(DATA, "buckets_3p_15bb_flop.json")
    bk = EquityBucketer.load(p) if os.path.exists(p) else EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)
    bk.save(str(data / "buckets_small.json"))
    points = []
    for stack in (20, 30, 45):
        t = MCCFRTrainer(small_spec(stack), bk, seed=stack, backend="cpp", threads=2).train(3000)
        t.save_blueprint(str(data / f"blueprint_small{stack}.bin"))
        points.append({"stack_bb": stack, "blueprint": f"data/blueprint_small{stack}.bin", "buckets": "data/buckets_small.json"})
    points.append({"stack_bb": 60, "blueprint": "data/blueprint_small60.bin", "present": False})
    manifest = {"bucketer": "data/buckets_small.json", "grid": {"preflop": [1.0], "postflop": [0.5, 1.0], "max_raises": 2},
                "points": points[::-1], "rule": {"kind": "nearest_log"}}
    path = data / "stack_grid_small.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return SimpleNamespace(root=str(root), path=str(path), bk=bk, cbk=core_bucketer(bk), manifest=manifest)


def duel(hero, stack, seed=3, deals=25, log=None, opponent=None):
    """Duplicate deals against the random agent (any raise size): per hand the events and the hero's decisions."""
    from negpluribus.eval.duel import duplicate_duel

    out = []

    def on_hand(d, seat, rec, luck, infos):
        out.append(([(e.street, e.seat, int(e.action.type), e.action.amount) for e in rec.events],
                    [(i["street"], i["action"]) for i in infos]))
        if log is not None:
            log.append({"deal": d, "hero_seat": seat, "button": rec.button, "stacks": list(rec.starting_stacks),
                        "holes": rec.hole_cards, "board": rec.board,
                        "events": [[int(e.street), e.seat, int(e.action.type), int(e.action.amount)] for e in rec.events],
                        "net_bb": rec.net[seat] / 100, "luck_bb": None,
                        "hero": [{k: v for k, v in i.items() if k != "s"} for i in infos]})

    duplicate_duel(hero, [opponent or RandomAgent(seed=5, name="random")], n_deals=deals, seed=seed, sb=50, bb=100,
                   stack_bb=stack, luck=False, on_hand=on_hand)
    return out


def test_the_manifest_loader(small, tmp_path):
    g = StackGrid.load(small.path)
    assert g.stacks == [20, 30, 45] and g.absent == (60,) and g.root == small.root
    assert g.resolve("data/x.bin") == os.path.normpath(os.path.join(small.root, "data", "x.bin"))
    assert g.bet_grid().preflop_fracs == (1.0,) and g.max_raises == 2 and g.rule == StackRule("nearest_log")
    assert g.point_at(30).blueprint == "data/blueprint_small30.bin" and g.pick(26).stack_bb == 30
    assert g.find("C:/elsewhere/blueprint_small45.bin").stack_bb == 45
    with pytest.raises(KeyError):
        g.find("data/blueprint_other.bin")
    assert "points 20bb, 30bb, 45bb; absent: 60bb; rule nearest_log" in g.describe()
    bad = dict(small.manifest, points=small.manifest["points"] + [{"stack_bb": 90, "blueprint": "data/missing.bin"}])
    (tmp_path / "data").mkdir()
    bounds = {"kind": "boundaries", "upper": {"20": 24, "30": 37, "45": 1000, "60": 2000}, "below_grid": 20, "above_grid": 45}
    path = tmp_path / "data" / "bounds.json"
    path.write_text(json.dumps(dict(small.manifest, rule=bounds)), encoding="utf-8")
    gb = StackGrid.load(str(path), root=small.root)
    assert gb.rule.upper == (24, 37, 1000) and [gb.pick(s).stack_bb for s in (10, 24, 25, 37, 38, 1000, 1001)] == [20, 20, 30, 30, 45, 45, 45]
    for name, d, err, match in (("missing", bad, FileNotFoundError, "missing"),
                                ("twice", dict(small.manifest, points=small.manifest["points"] + [small.manifest["points"][1]]),
                                 ValueError, "two points"),
                                ("list", dict(small.manifest, rule={"kind": "boundaries", "upper": [25]}), ValueError, "must map"),
                                ("no45", dict(small.manifest, rule={"kind": "boundaries", "upper": {"20": 24, "30": 37}}),
                                 ValueError, r"bad_no45.json: rule 'upper' has no entry for the point\(s\) \[45\]")):
        name = "bad_" + name
        path = tmp_path / "data" / f"{name}.json"
        path.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(err, match=match):
            StackGrid.load(str(path), root=small.root)


def test_at_a_grid_point_the_grid_plays_bit_for_bit_as_the_point_agent(small):
    """Acceptance 1 of the design on the tiny grid: at stacks equal to a point, the grid agent (every point loadable,
    two kept) and the single blueprint agent of that point, reset by the duel with the same seeds, play the same
    hands to the last action."""
    from negpluribus.fast.blueprint import load_blueprint

    g = StackGrid.load(small.path)
    make = g.blueprint_factory(small.cbk)
    for stack in (20, 30, 45):
        spec = small_spec(stack)
        single = BlueprintAgent(load_blueprint(g.resolve(g.point_at(stack).blueprint)), small.cbk, spec.grid, seed=1, name="hero")
        grid = g.agent(make, max_loaded=2, seed=1, name="hero")
        a, b = duel(single, stack), duel(grid, stack)
        assert a == b and len(a) == 50
        assert set(grid.hands_by_point) == {stack} and grid.loads == Counter({stack: 1})
        assert grid.n_decisions == single.n_decisions and grid.n_fallback == single.n_fallback


def test_after_a_reset_a_hand_does_not_depend_on_the_points_played_before(small):
    g = StackGrid.load(small.path)
    make = g.blueprint_factory(small.cbk)
    ref = duel(g.agent(make, max_loaded=1, seed=1, name="hero"), 30)
    warm = g.agent(make, max_loaded=1, seed=9, name="hero")
    duel(warm, 20, seed=4, deals=5)
    duel(warm, 45, seed=5, deals=5)  # 20 evicted, 45 loaded
    assert duel(warm, 30) == ref and warm.evictions == 2
    both = g.agent(make, max_loaded=3, seed=1, name="hero")
    duel(both, 45, deals=5)
    assert duel(both, 30) == ref and both.loaded_points == [45, 30]
    # 30bb deals at stacks off the grid are played at the nearest point: 26bb -> 30, 24bb -> 20 (sqrt(600) = 24.5)
    off = g.agent(make, seed=1, name="hero")
    duel(off, 26, deals=5)
    duel(off, 24, deals=5)
    assert set(off.hands_by_point) == {30, 20}


def test_a_point_blueprint_of_another_depth_or_bucketer_is_refused(small, tmp_path):
    from negpluribus.fast.trainer import core_bucketer

    d = json.loads(json.dumps(small.manifest))
    d["points"] = [p for p in d["points"] if p.get("present", True)]
    d["points"][0]["blueprint"] = "data/blueprint_small30.bin"  # the 45bb point pointing at the 30bb file (reversed list)
    (tmp_path / "data").mkdir()
    path = tmp_path / "data" / "wrong.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    g = StackGrid.load(str(path), root=small.root)
    make = g.blueprint_factory(small.cbk)
    with pytest.raises(ValueError, match="not the 45bb point"):
        make(g.point_at(45))
    other = EquityBucketer(n_buckets=8, samples=150)
    other.boundaries = {k: [x * 0.999 for x in v] for k, v in small.bk.boundaries.items()}
    make = StackGrid.load(small.path).blueprint_factory(core_bucketer(other))
    with pytest.raises(ValueError, match="bucketer_fingerprint"):
        make(StackGrid.load(small.path).point_at(20))


def test_the_search_factory_shares_one_bucketer_between_the_points(small):
    """The hook for the next step: a search agent per point, on SearchResources of the point's blueprint at the point's
    stack, all on the first point's C++ bucketer; a hand is played (fixed iterations, one thread)."""
    from negpluribus.agents.core_search import SearchConfig

    if not hasattr(core, "SubgameSearch"):
        pytest.skip("no search in the C++ core")
    g = StackGrid.load(small.path)
    make = g.search_factory(small_spec(20), small.bk, SearchConfig(iterations=100, threads=1))
    a, b = make(g.point_at(20)), make(g.point_at(45))
    assert a.res.core_bucketer is b.res.core_bucketer and a.res.spec.stack_bb == 20 and b.res.spec.stack_bb == 45
    grid = g.agent(make, seed=1, name="hero")
    out = duel(grid, 45, deals=3, opponent=CallingAgent(name="caller"))
    assert len(out) == 6 and set(grid.hands_by_point) == {45}
    assert grid.inner_agents()[0].n_searches > 0 and not grid.inner_agents()[0].stats.errors


def load_aivat_script():
    spec = importlib.util.spec_from_file_location("aivat_eval_script", os.path.join(ROOT, "scripts", "aivat_eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_aivat_selection_reads_the_point_of_every_hand_from_the_log(small, tmp_path):
    """scripts/aivat_eval.py --blueprints: per hand the blueprint the hero's first decision names (here hands at 20 and
    45bb in one log), the rule at the logged stacks for a hand without a hero decision; a hand at other stacks than
    the root table's is refused."""
    g = StackGrid.load(small.path)
    make = g.blueprint_factory(small.cbk)
    grid = g.agent(make, seed=1, name="hero")
    log = []
    duel(grid, 20, deals=15, log=log)
    n20 = len(log)
    duel(grid, 45, deals=15, log=log)
    silent = [r for r in log if not r["hero"]]
    assert silent, "no hand without a hero decision (the small blind folding to it)"
    for r in log:
        want = 20 if r["stacks"][0] == 2000 else 45
        assert g.point_for_hand(r).stack_bb == want
        if r["hero"]:
            assert r["hero"][0]["blueprint"] == g.point_at(want).blueprint and r["hero"][0]["point"] == want
            assert g.point_for_hand({**r, "hero": [{"point": want}]}).stack_bb == want
    old = {k: v for k, v in silent[0].items() if k != "stacks"}  # a line without stacks: the default stack
    assert g.point_for_hand(old, 45).stack_bb == 45 and g.point_for_hand(old, 20).stack_bb == 20
    with pytest.raises(KeyError):
        g.point_for_hand(old)
    with pytest.raises(KeyError):
        g.point_for_hand({**log[0], "hero": [{"street": 0}]})  # not a grid hero
    # the script's reading: the fixed-stack part of the log is accepted, a hand at other stacks refused
    mod = load_aivat_script()
    path = tmp_path / "hands.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in log[:n20]), encoding="utf-8")
    args = SimpleNamespace(slumbot=None, duel=str(path), stack_bb=20, known="hero")
    entries, _ = mod.load_hands(args, g)
    assert len(entries) == n20 and {e["point"] for _, e in entries} == {20}
    assert all(e["blueprint"] == "data/blueprint_small20.bin" and h.stacks == (2000, 2000) for h, e in entries)
    path.write_text("".join(json.dumps(r) + "\n" for r in log), encoding="utf-8")
    with pytest.raises(SystemExit, match="fixed stacks only"):
        mod.load_hands(args, g)
    args45 = SimpleNamespace(slumbot=None, duel=str(path), stack_bb=45, known="hero")
    with pytest.raises(SystemExit, match=r"\[2000, 2000\] are not the root table's \[4500, 4500\]"):
        mod.load_hands(args45, g)
