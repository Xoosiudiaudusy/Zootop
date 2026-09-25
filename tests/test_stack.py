"""Integration tests: agents + table + tracker + duplicate evaluation."""
import random

from negpluribus.agents import STYLES, make_agent
from negpluribus.agents.base import CallingAgent
from negpluribus.cards import Deck
from negpluribus.engine import CALL, FOLD, raise_to
from negpluribus.eval import compare_heroes, duplicate_match
from negpluribus.stats import StatsTracker
from negpluribus.table import CashTable, play_hand


def test_all_archetypes_play_legal_hands():
    agents = [make_agent(n, seed=i) for i, n in enumerate(list(STYLES) + ["random", "caller"])][:6]
    table = CashTable(agents, seed=5)
    res = table.play(200)
    assert len(res.records) == 200
    assert sum(res.net) == 0


def test_tracker_vpip_pfr_basic():
    """Hand: UTG raises, everyone folds.  UTG: vpip+pfr; others: not."""
    tr = StatsTracker()
    names = list("ABCDEF")
    stacks = [10000] * 6

    class Scripted(CallingAgent):
        def __init__(self, plan):
            super().__init__()
            self.plan = plan

        def act(self, obs):
            return self.plan(obs)

    def utg(obs):
        return raise_to(300) if obs.seat == 3 else FOLD

    agents = [Scripted(lambda o: FOLD if o.can_fold else CALL) for _ in range(6)]
    agents[3] = Scripted(utg)
    rec = play_hand(agents, stacks, button=0, deck=Deck(seed=1))
    tr.observe_hand(rec, names)
    assert tr.players["D"].raw("vpip") == 1.0 and tr.players["D"].raw("pfr") == 1.0
    assert tr.players["A"].raw("vpip") == 0.0
    assert tr.players["C"].raw("vpip") == 0.0  # BB folding is not VPIP
    # smoothed value with 1 hand of evidence sits between prior and observation
    d = tr.players["D"].stat("vpip")
    assert 0.25 < d < 1.0
    assert 0 < tr.players["D"].confidence("vpip") < 0.2


def test_tracker_orders_archetypes_by_looseness():
    names = ["nit", "tag", "lag", "station", "maniac", "passive"]
    agents = [make_agent(n, seed=i) for i, n in enumerate(names)]
    res = CashTable(agents, seed=11).play(400)
    tr = StatsTracker()
    for r in res.records:
        tr.observe_hand(r, names)
    v = {n: tr.players[n].raw("vpip") for n in names}
    assert v["nit"] < v["tag"] < v["lag"] < v["maniac"]
    assert v["station"] > v["tag"]
    p = {n: tr.players[n].raw("pfr") for n in names}
    assert p["maniac"] > p["lag"] > p["station"]
    assert p["passive"] < p["tag"]
    assert tr.players["station"].af < tr.players["maniac"].af
    assert tr.players["station"].raw("fold_to_cbet") is None or tr.players["station"].raw("fold_to_cbet") < 0.5
    assert len(tr.players["nit"].feature_vector()) == 15 * 2 + len(tr.players["nit"].hud) + 1


def test_duplicate_match_is_reproducible_and_symmetric():
    vil = [make_agent(n, seed=i) for i, n in enumerate(["nit", "station", "maniac", "lag", "passive"])]
    a = duplicate_match(make_agent("tag", seed=9), vil, n_deals=20, seed=3)
    b = duplicate_match(make_agent("tag", seed=9), vil, n_deals=20, seed=3)
    assert a.per_deal_bb == b.per_deal_bb
    assert a.n_hands == 120
    assert a.ci95 > 0


def test_compare_heroes_same_agent_zero_gain():
    vil = [make_agent(n, seed=i) for i, n in enumerate(["nit", "station", "maniac", "lag", "passive"])]
    ra, rb, gain, ci = compare_heroes(make_agent("tag", seed=1), make_agent("tag", seed=1), vil, n_deals=15, seed=2)
    assert abs(gain) < 1e-9 and ci == 0.0


def test_station_loses_to_tag_over_many_hands():
    """Weak sanity check on archetype ordering: a calling station should lose to a TAG in a 6-max pool."""
    vil = [make_agent(n, seed=i) for i, n in enumerate(["tag", "tag", "nit", "lag", "passive"])]
    r_tag = duplicate_match(make_agent("tag", seed=1), vil, n_deals=60, seed=4)
    r_station = duplicate_match(make_agent("station", seed=1), vil, n_deals=60, seed=4)
    assert r_tag.bb100 > r_station.bb100
