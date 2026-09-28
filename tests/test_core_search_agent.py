"""Real-time search, part 3: the search agent at the table (negpluribus/agents/core_search.py), the
duplicate duel with the card-luck correction (negpluribus/eval/duel.py) and the value overbettor.

Deterministic pieces are checked for identity (the duel against duplicate_match, the luck
correction against a direct computation, the equity against Python, the ranges passed between
rounds against the previous search's likelihood); the agent's random play for legality, coverage
and the documented decision rules."""
from __future__ import annotations

import itertools
import os
import random

import pytest

from negpluribus import fast
from negpluribus.abstraction import EquityBucketer
from negpluribus.agents.base import RandomAgent
from negpluribus.agents.blueprint import BlueprintAgent
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import ActionType, Street, raise_to
from negpluribus.evaluator import evaluate

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "SubgameSearch"), reason="C++ core not built")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def game():
    from negpluribus.agents.core_search import SearchResources

    p = os.path.join(DATA, "buckets_3p_15bb_flop.json")
    bk = EquityBucketer.load(p) if os.path.exists(p) else EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)
    spec = GameSpec(n_players=2, stack_bb=30, max_street=Street.RIVER, n_buckets=bk.n_buckets, max_raises_per_street=2,
                    preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    t = MCCFRTrainer(spec, bk, seed=2, backend="cpp", threads=4).train(6000)
    return spec, bk, SearchResources.build(spec, bk, t.blueprint())


def agent(game, **kw):
    from negpluribus.agents.core_search import CoreSearchAgent, SearchConfig

    spec, bk, res = game
    cfg = dict(iterations=300, threads=2)
    cfg.update(kw)
    return CoreSearchAgent(res, SearchConfig(**cfg), seed=3)


def test_plays_legal_hands_against_arbitrary_sizes_with_nothing_off_the_map(game):
    """Against the random agent (any raise size): every action legal (the engine would raise), every
    postflop decision searched, nothing played by a default, and the per-hand state reset."""
    from negpluribus.eval.duel import duplicate_duel

    spec, bk, res = game
    hero = agent(game)
    r = duplicate_duel(hero, [RandomAgent(seed=5, name="random")], n_deals=12, seed=1, sb=spec.sb, bb=spec.bb,
                       stack_bb=spec.stack_bb, luck=False)
    s = hero.stats
    assert r.n_hands == 24 and s.n_decisions > 24 and not s.errors
    assert s.off_map == 0 and hero.fallback_rate == 0.0
    postflop = sum(v for k, v in s.decisions.items() if k > 0)
    assert postflop > 0 and sum(v for k, v in s.searches.items() if k > 0) == postflop
    assert s.preflop_reasons.get("off-grid size", 0) > 0 and s.inserted > 0


def test_a_failed_search_is_played_by_the_blueprint_and_counted(game, monkeypatch):
    """A search that raises (as solve() does when its node table cannot grow: tests/test_search_oom.py) is played by
    the blueprint and counted: SearchStats.errors, the summary line of the check files next to "off the map", and the
    decision info (played = "blueprint after a search error" and the error, both kept by eval_archetypes
    --log-hands), so a run under memory pressure is visible.  Every search fails here (deterministic: a real table
    growth happens or not depending on the hands)."""
    from negpluribus.agents.core_search import SearchStats
    from negpluribus.eval.duel import duplicate_duel

    assert "off the map 0 (0.00%); search errors 0 (fallback to the blueprint)" in SearchStats().summary()
    spec, bk, res = game
    hero = agent(game)
    calls = []

    def failing(obs, reason):
        calls.append(int(obs.street))
        raise RuntimeError("search failed: out of memory growing the node table")

    monkeypatch.setattr(hero, "_search", failing)
    infos = []
    r = duplicate_duel(hero, [RandomAgent(seed=5, name="random")], n_deals=6, seed=1, sb=spec.sb, bb=spec.bb,
                       stack_bb=spec.stack_bb, luck=False, on_hand=lambda d, seat, rec, luck, inf: infos.extend(inf))
    s = hero.stats
    failed = [i for i in infos if i.get("played") == "blueprint after a search error"]
    assert r.n_hands == 12 and s.n_errors == len(failed) == len(calls) > 0, s.summary()
    assert all(i["error"] == "RuntimeError: search failed: out of memory growing the node table" for i in failed)
    assert f"search errors {s.n_errors} (fallback to the blueprint)" in s.summary()


def test_preflop_blueprint_unless_the_size_is_far_off_the_grid_or_the_key_unknown(game):
    from negpluribus.agents.core_search import raise_offgrid_distance

    spec, bk, res = game
    grid = spec.grid
    st = spec.new_hand(list(range(52)), button=0)
    obs = st.observe(st.current_player)
    for name in grid.abstract_actions(obs):
        a = grid.to_concrete(obs, name)
        if a.type == ActionType.RAISE:
            assert raise_offgrid_distance(obs, a.amount, grid) == 0.0
    pot_raise = grid.to_concrete(obs, "r1").amount
    level = obs.street_bets_max()
    assert raise_offgrid_distance(obs, level + int(1.5 * (pot_raise - level)), grid) == pytest.approx(0.5, abs=0.02)
    # on the grid: the blueprint answers (no search)
    hero = agent(game)
    st.apply(grid.to_concrete(obs, "r1"))
    hero.act(st.observe(st.current_player))
    assert hero.stats.blueprint == 1 and hero.stats.n_searches == 0
    # far off the grid: a preflop search from the start of the hand, the size inserted
    st = spec.new_hand(list(range(52)), button=0)
    obs = st.observe(st.current_player)
    st.apply(raise_to(level + 3 * (pot_raise - level)))
    hero = agent(game)
    hero.act(st.observe(st.current_player))
    assert hero.stats.preflop_reasons["off-grid size"] == 1 and hero.last["street"] == 0
    assert hero.stats.inserted == 1 and hero.last["search"].root_info()["street"] == 0
    assert hero.last["search"].root_info()["limit_street"] == 0  # a preflop root stops at the flop
    # the blueprint without a strategy at our key: searched instead of check/call
    hero = agent(game)

    class NoStrategy:
        def policy(self, key, legal):
            return None

    hero.blueprint_agent.strategy = NoStrategy()
    st = spec.new_hand(list(range(52)), button=0)
    st.apply(grid.to_concrete(st.observe(st.current_player), "r1"))
    hero.act(st.observe(st.current_player))
    assert hero.stats.preflop_reasons["no blueprint strategy"] == 1 and hero.stats.off_map == 0
    assert hero.stats.blueprint == 0 and hero.stats.n_searches == 1


def test_preflop_blueprint_decisions_equal_a_blueprint_agent_with_the_same_seed(game):
    """For paired comparisons: reset with the same seed, the search agent's preflop blueprint
    decisions are the blueprint agent's own (the same random draws)."""
    spec, bk, res = game
    hero = agent(game)
    ref = BlueprintAgent(res.blueprint, bk, spec.grid, seed=99)
    rng = random.Random(12)
    compared = 0
    for k in range(200):
        order = list(range(52))
        rng.shuffle(order)
        st = spec.new_hand(order, button=k % 2)
        hero.reset(k)
        ref.reset(k)
        while not st.is_terminal and st.street == Street.PREFLOP:
            obs = st.observe(st.current_player)
            a = hero.act(obs)
            if hero.last is None or hero.stats.n_searches == 0:
                assert a == ref.act(obs)
                compared += 1
            st.apply(a)
        if hero.stats.n_searches:
            break
    assert compared > 100


def test_later_rounds_take_their_ranges_from_the_searched_rounds(game):
    """At the turn the search gets, for every live seat, the flop search's likelihood of that seat's
    flop actions (each factor floored at range_floor)."""
    spec, bk, res = game
    hero = agent(game, range_floor=0.01)
    captured = []
    orig = hero._overrides

    def spy(obs):
        out = orig(obs)
        captured.append((int(obs.street), out))
        return out

    hero._overrides = spy
    rng = random.Random(4)
    for _ in range(200):  # play until a hand reaches a turn decision of ours after a flop decision
        order = list(range(52))
        rng.shuffle(order)
        st = spec.new_hand(order, button=0)
        hero.reset(rng.getrandbits(32))
        captured.clear()
        flop_search = None
        while not st.is_terminal and st.street <= Street.TURN:
            obs = st.observe(st.current_player)
            if obs.seat == 0:
                a = hero.act(obs)
                if obs.street == Street.FLOP:
                    flop_search = hero.last["search"]
            else:
                names = [n for n in spec.grid.abstract_actions(obs) if n != "f"]
                a = spec.grid.to_concrete(obs, rng.choice(names))
            if st.street == Street.TURN and obs.seat == 0:
                break
            st.apply(a)
        turn = [o for s, o in captured if s == Street.TURN]
        if flop_search is not None and turn:
            break
    else:
        pytest.fail("no hand with a flop and a turn decision")
    ov = turn[0]
    assert sorted(seat for street, seat, _ in ov) == [0, 1] and all(street == Street.FLOP for street, _, _ in ov)
    flop_actions = [(int(e.action.type), int(e.action.amount) if e.action.type == ActionType.RAISE else 0)
                    for e in obs.events if e.street == Street.FLOP]
    for street, seat, w in ov:
        want, _ = flop_search.likelihood(seat, flop_actions, floor=0.01)
        assert w == want
        assert min(x for x in w if x > 0) >= 0.01 ** sum(1 for e in obs.events if e.street == Street.FLOP and e.seat == seat)


def test_play_mode_samples_the_chosen_strategy(game):
    spec, bk, res = game
    rng = random.Random(8)
    for play in ("average", "final"):
        hero = agent(game, play=play)
        for _ in range(15):
            order = list(range(52))
            rng.shuffle(order)
            st = spec.new_hand(order, button=0)
            for name in ("r1", "c", "c", "c"):  # a turn decision
                st.apply(spec.grid.to_concrete(st.observe(st.current_player), name))
            hero.act(st.observe(st.current_player))
            r, i = hero.last["result"], hero.last["choice"]
            assert hero.last["street"] == Street.TURN and r[play][i] > 0.0


def test_the_duel_deals_and_seeds_exactly_as_duplicate_match(game):
    from negpluribus.eval import duplicate_match
    from negpluribus.eval.duel import duplicate_duel

    spec, bk, res = game
    kw = dict(n_deals=30, seed=4, sb=spec.sb, bb=spec.bb, stack_bb=spec.stack_bb)
    a = duplicate_match(BlueprintAgent(res.blueprint, bk, spec.grid, seed=1, name="hero"), [RandomAgent(seed=2)], **kw)
    b = duplicate_duel(BlueprintAgent(res.blueprint, bk, spec.grid, seed=1, name="hero"), [RandomAgent(seed=2)], luck=False, **kw)
    assert b.raw == a.per_deal_bb and b.n_hands == a.n_hands and len(b.hand_seconds) == 60
    # the per-hand hook: every hand, the hero's seat and card luck, the hero's decisions
    hands = []
    c = duplicate_duel(BlueprintAgent(res.blueprint, bk, spec.grid, seed=1, name="hero"), [RandomAgent(seed=2)], luck=True,
                       on_hand=lambda d, seat, rec, luck, infos: hands.append((d, seat, rec, luck, infos)), **kw)
    assert c.raw == a.per_deal_bb and len(hands) == 60 and [h[1] for h in hands] == [0, 1] * 30
    for d in range(30):
        net = sum(rec.net[seat] / spec.bb for _, seat, rec, _, _ in hands[2 * d: 2 * d + 2])
        luck = sum(lk for _, _, _, lk, _ in hands[2 * d: 2 * d + 2])
        assert net == pytest.approx(c.raw[d], abs=1e-9) and c.corrected[d] == pytest.approx(net - luck, abs=1e-9)
    assert sum(len(h[4]) for h in hands) == c.hero_decisions
    assert all(i["street"] in (0, 1, 2, 3) and len(i["action"]) == 2 for h in hands for i in h[4])


def _py_equity(h, o, board):
    rest = [c for c in range(52) if c not in set(h) | set(o) | set(board)]
    won = n = 0.0
    for extra in itertools.combinations(rest, 5 - len(board)):
        bd = list(board) + list(extra)
        a1, a2 = evaluate(list(h) + bd), evaluate(list(o) + bd)
        won += 1.0 if a1 > a2 else 0.5 if a1 == a2 else 0.0
        n += 1
    return won / n


def test_exact_equity_is_a_martingale_over_the_cards_to_come():
    rng = random.Random(3)
    for _ in range(6):
        cards = rng.sample(range(52), 7)
        h, o, flop = cards[:2], cards[2:4], cards[4:7]
        e_flop = core.equity_vs_hand(h, o, flop)
        assert e_flop == pytest.approx(_py_equity(h, o, flop), abs=1e-12)
        turns = [c for c in range(52) if c not in cards]
        e_turns = [core.equity_vs_hand(h, o, flop + [t]) for t in turns]
        assert sum(e_turns) / len(e_turns) == pytest.approx(e_flop, abs=1e-12)
        t = turns[0]
        rivers = [c for c in turns if c != t]
        e_rivers = [core.equity_vs_hand(h, o, flop + [t, r]) for r in rivers]
        assert sum(e_rivers) / len(e_rivers) == pytest.approx(e_turns[0], abs=1e-12)
    pre = core.equity_vs_hand([48, 49], [0, 5], [], 1)
    assert core.equity_vs_hand([48, 49], [0, 5], [], 8) == pre  # exact counts: the same on any thread count
    assert pre + core.equity_vs_hand([0, 5], [48, 49], [], 8) == pytest.approx(1.0, abs=1e-12)


def test_chance_correction_against_a_direct_computation(game):
    """Pots at the deals from a manual count of the chips, equities by enumeration in Python."""
    from negpluribus.eval.duel import chance_correction, deal_pots
    from negpluribus.table import play_hand
    from negpluribus.cards import Deck

    spec, bk, res = game
    rng = random.Random(6)
    checked = 0
    for k in range(40):
        order = list(range(52))
        rng.shuffle(order)
        rec = play_hand([RandomAgent(seed=k), RandomAgent(seed=k + 100)], [spec.stack_bb * spec.bb] * 2, k % 2,
                        spec.sb, spec.bb, deck=Deck.from_order(order))
        # the chips: blinds, then each action's payment; a street's deal sees the matched chips
        contrib = [0, 0]
        sb_seat = rec.button  # heads-up: the button posts the small blind
        contrib[sb_seat], contrib[1 - sb_seat] = spec.sb, spec.bb
        want = [spec.sb + spec.bb]
        street = 0
        for ev in rec.events:
            while ev.street > street:
                street += 1
                want.append(2 * min(contrib))
            contrib[ev.seat] += ev.paid
        n = {0: 1, 3: 2, 4: 3, 5: 4}[len(rec.board)]
        while len(want) < n:
            want.append(2 * min(contrib))
        assert deal_pots(rec) == want[:n]
        if len(rec.board) >= 3 and checked < 6:
            e0 = core.equity_vs_hand(rec.hole_cards[0], rec.hole_cards[1], [])
            for seat in (0, 1):
                h, o = rec.hole_cards[seat], rec.hole_cards[1 - seat]
                # preflop and flop equities from C++ (checked against Python in the martingale test)
                eqs = [e0 if seat == 0 else 1.0 - e0, core.equity_vs_hand(h, o, rec.board[:3])]
                eqs += [_py_equity(h, o, rec.board[:nb]) for nb in (4, 5)]
                eqs = eqs[:n]
                corr = (eqs[0] - 0.5) * want[0] + sum((eqs[i] - eqs[i - 1]) * want[i] for i in range(1, n))
                assert chance_correction(rec, seat) == pytest.approx(corr, abs=1e-9)
            checked += 1
    assert checked >= 5


def test_the_value_overbettor_overbets_only_its_strong_raises(game):
    from negpluribus.agents.overbettor import ValueOverbettor

    spec, bk, res = game
    grid = spec.grid
    ob = ValueOverbettor(res.blueprint, bk, grid, mult=2.5, seed=4)
    ref = BlueprintAgent(res.blueprint, bk, grid, seed=4)
    rng = random.Random(9)
    overbets = same = 0
    for k in range(300):
        order = list(range(52))
        rng.shuffle(order)
        st = spec.new_hand(order, button=k % 2)
        ob.reset(k)
        ref.reset(k)
        while not st.is_terminal:
            obs = st.observe(st.current_player)
            a = ob.act(obs)
            b = ref.act(obs)
            if a != b:  # only a raise of the blueprint becomes a bigger raise, short of the all-in
                assert b.type == ActionType.RAISE and a.type == ActionType.RAISE and b.amount < a.amount < obs.max_raise_to
                level, top = obs.street_bets_max(), max(grid.fracs_for(obs.street))
                want = level + top * 2.5 * (obs.pot + obs.to_call)
                assert a.amount == min(int(round(want)), int(round(level + 0.8 * (obs.max_raise_to - level))))
                overbets += 1
            else:
                same += 1
            st.apply(a)
    assert overbets > 5 and same > 3 * overbets and ob.n_overbets == overbets  # about 15-25 of 780 decisions


def test_the_search_can_key_its_later_rounds_on_a_bucketer_of_its_own(game, tmp_path):
    """SearchResources with a search bucketer: the game keys the subgame's later rounds on it while the inner blueprint
    agent keeps the blueprint's bucketer; the agent plays legal hands, every postflop decision searched; a saved cache
    of the search bucketer loads; a search bucketer with the blueprint's identity is the blueprint's (nothing separate)."""
    from negpluribus.abstraction import PotentialAwareBucketer
    from negpluribus.agents.core_search import CoreSearchAgent, SearchConfig, SearchResources
    from negpluribus.eval.duel import duplicate_duel
    from negpluribus.fast.trainer import core_bucketer

    spec, bk, res = game
    fine = PotentialAwareBucketer(n_buckets=24, samples=6, bins=10).fit(n_situations=200, seed=5)
    cache = str(tmp_path / "fine_cache.bin")
    c = core_bucketer(fine)
    c.bucket([0, 1], [10, 20, 30])
    c.bucket([0, 1], [10, 20, 30, 40])
    c.save_cache(cache, [1, 2])
    r2 = SearchResources.build(spec, bk, res.blueprint, search_bucketer=fine, search_cache_path=cache)
    assert r2.game.separate_buckets and r2.search_bucketer is fine and r2.search_core_bucketer is not None
    assert [x[0] for x in r2.search_cache_loaded] == [1, 2] and all(x[1] >= 1 for x in r2.search_cache_loaded)
    assert "the subgame's rounds after its root: 24 potential" in r2.describe_buckets()
    hero = CoreSearchAgent(r2, SearchConfig(iterations=300, threads=2), seed=3)
    assert hero.blueprint_agent.bucketer is bk
    r = duplicate_duel(hero, [RandomAgent(seed=5, name="random")], n_deals=12, seed=1, sb=spec.sb, bb=spec.bb,
                       stack_bb=spec.stack_bb, luck=False)
    s = hero.stats
    assert r.n_hands == 24 and not s.errors and s.off_map == 0
    postflop = sum(v for k, v in s.decisions.items() if k > 0)
    assert postflop > 0 and sum(v for k, v in s.searches.items() if k > 0) == postflop
    r3 = SearchResources.build(spec, bk, res.blueprint, search_bucketer=bk)
    assert not r3.game.separate_buckets and r3.search_bucketer is None and r3.search_core_bucketer is None
    assert not res.game.separate_buckets and "after its root: the blueprint's" in res.describe_buckets()


def test_resources_of_two_blueprints_share_one_bucketer_bit_for_bit(game, tmp_path):
    """Two SearchResources of different blueprints (30bb and 20bb) on one card abstraction, the second built on the
    first's C++ bucketers (the blueprint's, and the subgame's own with its saved cache): on one thread at fixed
    iterations every search is bit for bit that of two independent resources (direct solves at flop / turn / river
    roots: strategies, ranges, likelihoods; and whole hands of the search agent).  A C++ bucketer of another
    abstraction, or a cache path next to a shared bucketer, is refused."""
    from negpluribus.abstraction import PotentialAwareBucketer
    from negpluribus.agents.core_search import CoreSearchAgent, SearchConfig, SearchResources
    from negpluribus.eval.duel import duplicate_duel
    from negpluribus.fast.trainer import core_bucketer

    spec, bk, res = game
    spec20 = GameSpec(n_players=2, stack_bb=20, max_street=Street.RIVER, n_buckets=bk.n_buckets, max_raises_per_street=2,
                      preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
    bp20 = MCCFRTrainer(spec20, bk, seed=4, backend="cpp", threads=4).train(4000).blueprint()
    fine = PotentialAwareBucketer(n_buckets=24, samples=6, bins=10).fit(n_situations=200, seed=5)
    cache = str(tmp_path / "fine_cache.bin")
    c = core_bucketer(fine)
    c.bucket([0, 1], [10, 20, 30])
    c.save_cache(cache, [1, 2])
    games = [(spec, res.blueprint), (spec20, bp20)]

    def solves(r, sp):
        out = []
        for line in (["r1", "c"], ["r1", "c", "c", "c"], ["r1", "c", "c", "c", "c", "c"]):
            order = list(range(52))
            random.Random(len(line)).shuffle(order)
            st = sp.new_hand(order, button=0)
            acts = []
            for name in line:
                a = sp.grid.to_concrete(st.observe(st.current_player), name)
                acts.append((int(a.type), int(a.amount) if a.type == ActionType.RAISE else 0))
                st.apply(a)
            obs = st.observe(st.current_player)
            s = core.SubgameSearch(r.game, list(st.starting_stacks), st.button, acts, list(obs.board), obs.seat,
                                   list(obs.hole), iterations=1500, time_budget=0.0, threads=1, seed=3, depth="end")
            x = s.solve()
            out.append(({k: x[k] for k in ("final", "average", "iterations", "table_size", "nodes_touched")},
                        s.ranges(), [s.likelihood(p) for p in range(2)]))
        return out

    def hands(r, sp):
        hero = CoreSearchAgent(r, SearchConfig(iterations=300, threads=1), seed=3)
        log = []
        duplicate_duel(hero, [RandomAgent(seed=5, name="random")], n_deals=6, seed=2, sb=sp.sb, bb=sp.bb,
                       stack_bb=sp.stack_bb, luck=False,
                       on_hand=lambda d, seat, rec, luck, infos: log.append(
                           ([(e.seat, int(e.action.type), e.action.amount) for e in rec.events],
                            [{k: v for k, v in i.items() if k not in ("s", "search_s")} for i in infos])))
        assert hero.stats.n_searches > 0 and not hero.stats.errors
        return log

    for sbk in (None, fine):
        alone = [SearchResources.build(sp, bk, bp, search_bucketer=sbk, search_cache_path=cache if sbk else None)
                 for sp, bp in games]
        first = SearchResources.build(spec, bk, res.blueprint, search_bucketer=sbk, search_cache_path=cache if sbk else None)
        second = SearchResources.build(spec20, bk, bp20, search_bucketer=sbk, core_bucketer=first.core_bucketer,
                                       search_core_bucketer=first.search_core_bucketer)
        assert second.core_bucketer is first.core_bucketer and second.cache_loaded is None
        assert second.search_core_bucketer is first.search_core_bucketer and (sbk is None) == (second.search_core_bucketer is None)
        assert second.game.separate_buckets == (sbk is not None)
        for (sp, _), a, b in zip(games, alone, (first, second)):
            assert solves(a, sp) == solves(b, sp)
            assert hands(a, sp) == hands(b, sp)
    with pytest.raises(ValueError, match="not the C\\+\\+ twin"):
        SearchResources.build(spec20, bk, bp20, core_bucketer=first.search_core_bucketer)
    with pytest.raises(ValueError, match="cache_path with a shared core_bucketer"):
        SearchResources.build(spec20, bk, bp20, core_bucketer=first.core_bucketer, cache_path=cache)
    with pytest.raises(ValueError, match="without the Python search_bucketer"):
        SearchResources.build(spec20, bk, bp20, core_bucketer=first.core_bucketer, search_core_bucketer=first.search_core_bucketer)
