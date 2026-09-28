"""The search agent at unequal stacks and at depths other than its blueprint's (negpluribus/agents/core_search.py,
csrc/search.h): the subgame is built from the hand's real starting stacks (``obs.starting_stacks``), which is what
the stack-depth work relies on (a 200bb blueprint's search at a 50bb table, carry-stack duels where the two stacks
differ every hand, 3-max with three different stacks).

The duels of tests/test_core_search_agent.py only use equal stacks at the blueprint's depth.  Here, heads-up and
3-max, stacks shallower and deeper than the blueprint's 30bb and unequal between seats: every hero action is legal
(the engine refuses anything else), every postflop decision is searched without an error or an off-map default,
and at a searched decision the subgame's own largest raise is the hero's all-in of THIS hand: min(own stack, the
deepest live opponent's) is what the engine lets it put in, never the blueprint's 30bb.
"""
from __future__ import annotations

import random

import pytest

from negpluribus import fast
from negpluribus.agents.base import CallingAgent, RandomAgent
from negpluribus.cards import Deck
from negpluribus.engine import ActionType
from negpluribus.table import play_hand

core = fast.core()
pytestmark = pytest.mark.skipif(core is None or not hasattr(core, "SubgameSearch"), reason="C++ core with the search not built")


@pytest.fixture(scope="module")
def resources():
    from negpluribus.abstraction import EquityBucketer
    from negpluribus.agents.core_search import SearchResources
    from negpluribus.cfr.game import GameSpec
    from negpluribus.cfr.mccfr import MCCFRTrainer
    from negpluribus.engine import Street

    bk = EquityBucketer(n_buckets=4, samples=30)
    bk.boundaries = {1: [0.35, 0.5, 0.65], 2: [0.35, 0.5, 0.65], 3: [0.35, 0.5, 0.65]}
    out = {}
    for players, iters in ((2, 1500), (3, 1500)):
        spec = GameSpec(n_players=players, stack_bb=30, max_street=Street.RIVER, n_buckets=4, max_raises_per_street=2,
                        preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))
        bp = MCCFRTrainer(spec, bk, seed=players, backend="cpp", threads=1).train(iters).blueprint()
        out[players] = SearchResources.build(spec, bk, bp, cache_caps=(1 << 16,) * 3)
    return out


class _Watch:
    """Wraps the hero: checks each searched decision's largest raise against the engine's all-in of this hand."""

    def __init__(self, inner):
        self.inner, self.checked = inner, 0
        self.name = inner.name

    def reset(self, seed=None):
        self.inner.reset(seed)

    def end_hand(self, record, seat):
        self.inner.end_hand(record, seat)

    def act(self, obs):
        n = self.inner.stats.n_searches
        a = self.inner.act(obs)
        if self.inner.stats.n_searches > n and obs.can_raise:  # this decision was searched
            r = self.inner.last["result"]
            raises = [amt for t, amt in zip(r["types"], r["amounts"]) if t == int(ActionType.RAISE)]
            if raises:
                assert max(raises) == obs.max_raise_to, (obs.starting_stacks, obs.seat, raises, obs.max_raise_to)
                assert min(raises) >= obs.min_raise_to
                self.checked += 1
        return a


@pytest.mark.parametrize("players,stacks", [
    (2, [1200, 3000]),          # 12bb vs 30bb: the short stack's all-in bounds both
    (2, [6000, 900]),           # deeper than the blueprint, against 9bb
    (2, [8000, 8000]),          # 80bb both: a 30bb blueprint's search at another depth
    (3, [3000, 1500, 500]),     # three different stacks (side pots)
    (3, [800, 6000, 2500]),
])
def test_the_search_plays_the_hands_real_stacks(resources, players, stacks):
    from negpluribus.agents.core_search import CoreSearchAgent, SearchConfig

    res = resources[players]
    hero = CoreSearchAgent(res, SearchConfig(iterations=40, threads=1), seed=1, name="hero")
    watch = _Watch(hero)
    rng = random.Random(sum(stacks))
    for i in range(12):
        seat = i % players
        # a random bettor (any size) and a caller (hands that reach the later streets) as the opponents
        agents = [RandomAgent(seed=100 * i + k, name=f"r{k}") if (i + k) % 2 else CallingAgent(name=f"c{k}")
                  for k in range(players)]
        agents[seat] = watch
        order = list(range(52))
        rng.shuffle(order)
        hero.reset(i)
        rec = play_hand(agents, list(stacks), button=i % players, deck=Deck.from_order(order))
        assert sum(rec.net) == 0 and -rec.net[seat] <= stacks[seat]
    s = hero.stats
    assert not s.errors, s.summary()
    assert s.off_map == 0, s.summary()
    postflop = sum(v for k, v in s.decisions.items() if k > 0)
    assert sum(v for k, v in s.searches.items() if k > 0) == postflop
    assert s.n_decisions >= 8 and watch.checked >= 1, s.summary()


def _without_starting_stacks():
    """HU 30bb vs 12bb, SB opens to 300, BB calls: the flop observation, with its starting_stacks emptied (as a foreign
    wrapper or an old record would give it)."""
    from dataclasses import replace

    from negpluribus.engine import CALL, HandState, raise_to

    h = HandState([3000, 1200], button=0)
    h.apply(raise_to(300))
    h.apply(CALL)
    obs = h.observe()
    assert obs.starting_stacks == [3000, 1200]
    return replace(obs, starting_stacks=[])


@pytest.mark.xfail(strict=True, reason="L8: negpluribus/abstraction/infoset.py starting_stacks() rebuilds the stacks from the "
                                        "events, which do not hold the posted blinds (its docstring says so)")
def test_starting_stacks_are_rebuilt_with_the_blinds():
    from negpluribus.abstraction.infoset import starting_stacks

    assert starting_stacks(_without_starting_stacks()) == [3000, 1200]


@pytest.mark.xfail(strict=True, reason="L8: CoreSearchAgent._starting_stacks falls back to the spec's stack for every seat "
                                        "instead of rebuilding the hand's own stacks")
def test_the_search_agent_rebuilds_the_hands_stacks_without_starting_stacks(resources):
    from negpluribus.agents.core_search import CoreSearchAgent, SearchConfig

    hero = CoreSearchAgent(resources[2], SearchConfig(iterations=10, threads=1), seed=1)
    assert hero._starting_stacks(_without_starting_stacks()) == [3000, 1200]
