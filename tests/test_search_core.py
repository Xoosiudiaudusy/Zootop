"""Real-time search, part 1: the C++ subgame core (csrc/search.h, docs/search_design.md).

Unit checks of the parts the solver is built on, each against an independent computation:
  * the root = the public state at the start of the current round, rebuilt from the real actions
    (against the Python engine replaying the same actions);
  * the ranges = exact weights over the 1326 combos by Bayes over the blueprint (against a
    brute-force Python loop over combos, keys and BlueprintStrategy.policy: bit-identical);
  * an off-grid action of this round is inserted at the node where it happened;
  * our own actions of this round are fixed for our actual hole only;
  * deals never share a card, and follow the joint distribution of the ranges.
"""
from __future__ import annotations

import math
import os
import random
from collections import Counter

import pytest

from negpluribus import fast
from negpluribus.abstraction import EquityBucketer
from negpluribus.abstraction.infoset import infoset_key_for_bucket
from negpluribus.cards import Deck
from negpluribus.cfr.game import GameSpec
from negpluribus.cfr.mccfr import MCCFRTrainer
from negpluribus.engine import CALL, FOLD, ActionType, HandState, Street, raise_to

core = fast.core()
pytestmark = pytest.mark.skipif(core is None, reason="C++ core not built (python scripts/build_fast.py)")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
COMBOS = [(a, b) for a in range(52) for b in range(a + 1, 52)]
COMBO_INDEX = {c: i for i, c in enumerate(COMBOS)}
MIN_PROB = 1e-3


def to_action(t: int, amount: int):
    if t == int(ActionType.RAISE):
        return raise_to(amount)
    return CALL if t == int(ActionType.CALL) else FOLD


@pytest.fixture(scope="module")
def bucketer():
    p = os.path.join(DATA, "buckets_3p_15bb_flop.json")
    if os.path.exists(p):
        return EquityBucketer.load(p)
    return EquityBucketer(n_buckets=8, samples=150).fit(n_situations=300, seed=0)


def river_spec(players: int = 2) -> GameSpec:
    return GameSpec(n_players=players, stack_bb=30, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2,
                    preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0))


@pytest.fixture(scope="module")
def trained(bucketer):
    """(spec, trainer) per player count: small blueprints, enough keys that both the found and the
    missing-key paths of the range update are exercised."""
    from negpluribus.fast.trainer import core_bucketer, spec_to_dict

    out = {}
    for players, iters in ((2, 6000), (3, 4000)):
        spec = river_spec(players)
        t = MCCFRTrainer(spec, bucketer, seed=players, backend="cpp", threads=4).train(iters)
        game = core.SearchGame(spec_to_dict(spec), core_bucketer(bucketer), t.blueprint().lookup)
        out[players] = (spec, t, game, core_bucketer(bucketer))
    return out


def random_hand(spec: GameSpec, rng: random.Random, target_street: Street, off_grid: float = 0.25, max_tries: int = 500):
    """Play random grid actions (and now and then an off-grid raise) with the Python engine until a
    decision on ``target_street``; returns (state, actions as (type, amount))."""
    grid = spec.grid
    for _ in range(max_tries):
        order = list(range(52))
        rng.shuffle(order)
        st = spec.new_hand(order, button=rng.randrange(spec.n_players))
        acts = []
        while not st.is_terminal and st.street < target_street:
            obs = st.observe(st.current_player)
            legal = grid.abstract_actions(obs)
            if obs.can_raise and rng.random() < off_grid:
                lo, hi = obs.min_raise_to, obs.max_raise_to
                a = raise_to(rng.randint(lo, hi))
            else:
                name = rng.choice(legal)
                if name == "f" and rng.random() < 0.7:  # fewer folds, longer hands
                    name = "c"
                a = grid.to_concrete(obs, name)
            acts.append((int(a.type), int(a.amount)))
            st.apply(a)
        if st.is_terminal or st.street != target_street:
            continue
        # a few actions into the round, still somebody to act
        for _ in range(rng.randrange(3)):
            obs = st.observe(st.current_player)
            legal = [n for n in grid.abstract_actions(obs) if n != "f"]
            a = grid.to_concrete(obs, rng.choice(legal))
            probe = st.clone()
            probe.apply(a)
            if probe.is_terminal or probe.street != target_street:
                break
            acts.append((int(a.type), int(a.amount)))
            st.apply(a)
        return st, acts
    raise RuntimeError("no hand reached the street")


def make_search(game, st: HandState, acts, **kw):
    obs = st.observe(st.current_player)
    params = dict(iterations=0, time_budget=0.3, threads=4, seed=1)
    params.update(kw)
    return core.SubgameSearch(game, list(st.starting_stacks), st.button, acts, list(obs.board), obs.seat, list(obs.hole), **params)


def brute_ranges(spec: GameSpec, strategy, cbucketer, st_final: HandState, acts, root_street: Street, min_prob=MIN_PROB):
    """Per seat and combo: prod over the seat's actions before the root of max(sigma, min_prob), sigma =
    BlueprintStrategy.policy at the key the blueprint agent would build (deterministic translation),
    uniform when the key is unknown; 0 for combos on the board; None for seats folded at the root."""
    grid = spec.grid
    board = list(st_final.board)
    order = list(range(52))
    st = spec.new_hand(order, button=st_final.button)
    steps = []
    for a in acts:
        if st.street >= root_street:
            break
        obs = st.observe(st.current_player)
        legal = grid.abstract_actions(obs)
        ev = st.apply(to_action(*a))
        name = grid.from_concrete(ev, ev.all_in, None)
        steps.append((obs, legal, name))
    folded = [p.folded for p in st.players]
    out = []
    for seat in range(spec.n_players):
        if folded[seat]:
            out.append(None)
            continue
        w_seat = []
        for c0, c1 in COMBOS:
            if c0 in board or c1 in board:
                w_seat.append(0.0)
                continue
            w = 1.0
            for obs, legal, name in steps:
                if obs.seat != seat:
                    continue
                n_board = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[obs.street]
                b = cbucketer.bucket([c0, c1], board[:n_board])
                key = infoset_key_for_bucket(obs, b, grid)
                p = strategy.policy(key, legal)
                if p is None:
                    sigma = 1.0 / len(legal)
                else:
                    sigma = p[legal.index(name)] if name in legal else 0.0
                w *= max(sigma, min_prob)
            w_seat.append(w)
        out.append(w_seat)
    return out


# ============================================================ root and ranges
@pytest.mark.parametrize("players", [2, 3])
def test_root_is_the_start_of_the_round_rebuilt_from_real_actions(trained, players):
    spec, _, game, _ = trained[players]
    rng = random.Random(100 + players)
    n_checked = 0
    for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER):
        for _ in range(6):
            st, acts = random_hand(spec, rng, street)
            s = make_search(game, st, acts)
            info = s.root_info()
            # the Python engine at the start of the same round
            ref = spec.new_hand(list(range(52)), button=st.button)
            k = 0
            for a in acts:
                if ref.street >= street:
                    break
                ref.apply(to_action(*a))
                k += 1
            assert info["street"] == int(street) == int(ref.street)
            assert info["pot"] == ref.pot
            assert info["stacks"] == [p.stack for p in ref.players]
            assert info["street_bets"] == [p.street_bet for p in ref.players]
            assert info["invested"] == [p.invested for p in ref.players]
            assert info["folded"] == [p.folded for p in ref.players]
            assert info["all_in"] == [p.all_in for p in ref.players]
            assert info["to_act"] == ref.current_player
            assert info["n_active"] == sum(1 for p in ref.players if not p.folded)
            assert info["current_bet"] == ref.current_bet and info["raises_this_street"] == ref.raises_this_street == 0
            assert info["n_events"] == k == len(ref.events)
            assert len(s.path()) == len(acts) - k
            n_checked += 1
    assert n_checked == 24


@pytest.mark.parametrize("players", [2, 3])
def test_ranges_equal_a_brute_force_bayes_over_the_blueprint(trained, players):
    spec, t, game, cbk = trained[players]
    strategy = t.strategy()  # the same floats as the C++ lookup (full precision)
    rng = random.Random(7 + players)
    n_found = n_hands = 0
    for street in (Street.FLOP, Street.TURN, Street.RIVER):
        for _ in range(3):
            st, acts = random_hand(spec, rng, street)
            s = make_search(game, st, acts)
            got = s.ranges()
            want = brute_ranges(spec, strategy, cbk, st, acts, street)
            assert [g is None for g in got] == [w is None for w in want]
            for g, w in zip(got, want):
                if w is not None:
                    assert g == w  # bit for bit: same sigma, same products in the same order
                    n_found += sum(1 for x in w if 0 < x < 1)
            n_hands += 1
            # conditioned on our hole: the other seats lose every combo holding one of our cards
            obs = st.observe(st.current_player)
            cond = s.ranges(conditioned=True, our_seat=obs.seat, hole=list(obs.hole))
            for seat, (g, c) in enumerate(zip(got, cond)):
                if g is None:
                    continue
                for i, (a, b) in enumerate(COMBOS):
                    blocked = seat != obs.seat and (a in obs.hole or b in obs.hole)
                    assert c[i] == (0.0 if blocked else g[i])
    assert n_hands == 9 and n_found > 1000  # informative ranges, not all uniform


def test_likelihood_overrides_replace_the_blueprint_for_their_street(trained):
    spec, t, game, cbk = trained[2]
    st, acts = random_hand(spec, random.Random(3), Street.TURN, off_grid=0.0)
    s = make_search(game, st, acts)
    base = s.ranges()
    rng = random.Random(5)
    v = [rng.random() for _ in range(1326)]
    obs = st.observe(st.current_player)
    other = 1 - obs.seat
    s2 = make_search(game, st, acts, overrides=[(int(Street.FLOP), other, v)])
    got = s2.ranges()
    strategy = t.strategy()
    # expected: the blueprint product over preflop only, times v for the flop
    pre = brute_ranges(spec, strategy, cbk, st, acts, Street.FLOP)
    for i, (a, b) in enumerate(COMBOS):
        if a in obs.board or b in obs.board:
            assert got[other][i] == 0.0
        else:
            assert got[other][i] == pre[other][i] * v[i]
    assert got[obs.seat] == base[obs.seat]
    with pytest.raises(ValueError):  # an override for the root's own street means nothing
        make_search(game, st, acts, overrides=[(int(Street.TURN), other, v)])


# ============================================================ insertion and fixing
def _flop_hand(spec, rng_seed=11, flop_board=None):
    """SB/button raises pot preflop, BB calls; the flop is dealt from a fixed deck."""
    order = list(range(52))
    random.Random(rng_seed).shuffle(order)
    if flop_board:
        rest = [c for c in order if c not in flop_board]
        order = rest[:4] + list(flop_board) + rest[4:]
    st = spec.new_hand(order, button=0)
    acts = []
    for name in ("r1", "c"):
        a = spec.grid.to_concrete(st.observe(st.current_player), name)
        acts.append((int(a.type), int(a.amount)))
        st.apply(a)
    assert st.street == Street.FLOP
    return st, acts


def _act(st, acts, a):
    acts.append((int(a.type), int(a.amount)))
    st.apply(a)


def test_an_off_grid_action_is_inserted_where_it_was_taken(trained):
    spec, _, game, _ = trained[2]
    st, acts = _flop_hand(spec)
    obs = st.observe(st.current_player)  # BB, first to act on the flop
    pot = obs.pot
    off = raise_to(int(0.7 * pot))  # 0.7 pot: between the grid's 0.5 and 1
    assert off.amount not in {spec.grid.to_concrete(obs, n).amount for n in spec.grid.abstract_actions(obs)}
    _act(st, acts, off)
    s = make_search(game, st, acts, time_budget=0.5)
    path = s.path()
    assert len(path) == 1 and path[0]["inserted"] and path[0]["actor"] == 1
    node = path[0]
    assert node["actions"][-1] == f"x{off.amount}" and node["amounts"][-1] == off.amount and node["index"] == len(node["actions"]) - 1
    assert node["actions"][:-1] == spec.grid.abstract_actions(obs)  # the grid's actions come first, unchanged
    r = s.solve()
    assert r["visited"] and abs(sum(r["final"]) - 1) < 1e-9 and abs(sum(r["average"]) - 1) < 1e-9
    assert r["actions"] == spec.grid.abstract_actions(st.observe(st.current_player))  # our node: the grid
    # the inserted action is a live action of the subgame: some BB holes play it
    played = 0
    for a, b in COMBOS[::7]:
        if a in obs.board or b in obs.board or a in st.players[0].hole or b in st.players[0].hole:
            continue
        p = s._probe_path(0, a, b)
        if p is not None and p[-1] > 0:
            played += 1
    assert played > 0
    # we call, BB raises off-grid again: a second insertion in the same round, same root
    _act(st, acts, spec.grid.to_concrete(st.observe(st.current_player), "r0.5"))
    obs2 = st.observe(st.current_player)
    off2 = raise_to(min(obs2.max_raise_to - 1, obs2.min_raise_to + 37))
    _act(st, acts, off2)
    s2 = make_search(game, st, acts, time_budget=0.5)
    ins = [p["inserted"] for p in s2.path()]
    assert ins == [True, False, True]
    assert s2.root_info()["pot"] == s.root_info()["pot"] and s2.root_info()["n_events"] == 2  # same root
    r2 = s2.solve()
    assert r2["visited"] and len(r2["final"]) == len(spec.grid.abstract_actions(st.observe(st.current_player)))


def test_our_taken_actions_are_fixed_for_our_actual_hole_only(trained):
    spec, _, game, _ = trained[2]
    # a rainbow flop of three ranks: every combo is its own class, so the class of our hole is ours alone
    board = [48, 37, 22]  # As Kh 7d style: three suits, three ranks
    st, acts = _flop_hand(spec, flop_board=board)
    assert sorted(st.board) == sorted(board)
    _act(st, acts, spec.grid.to_concrete(st.observe(st.current_player), "c"))      # BB checks
    _act(st, acts, spec.grid.to_concrete(st.observe(st.current_player), "r0.5"))   # we (SB) bet
    _act(st, acts, spec.grid.to_concrete(st.observe(st.current_player), "r1"))     # BB raises
    obs = st.observe(st.current_player)
    assert obs.seat == 0
    s = make_search(game, st, acts, time_budget=0.6)
    r = s.solve()
    assert r["forced"] > 0 and r["visited"]
    h = list(obs.hole)
    # our bet node (path index 1), our actual hole: never updated (every visit was forced) ...
    mine = s._node_at_path(1, h[0], h[1])
    assert mine is not None and all(x == 0.0 for x in mine["regret"]) and all(x == 0.0 for x in mine["strategy_sum"])
    probe = s._probe_path(1, h[0], h[1])
    assert probe[s.path()[1]["index"]] == 1.0 and sum(probe) == 1.0
    # ... while our other holes decide freely there, and so do the opponent's holes at its nodes
    used = set(board) | set(h) | set(st.players[1].hole)
    others = [(a, b) for a, b in COMBOS if not ({a, b} & used)]
    free = [s._node_at_path(1, a, b) for a, b in others[::11]]
    assert sum(1 for x in free if x is not None and any(v != 0.0 for v in x["regret"])) > 5
    opp = [s._node_at_path(k, a, b) for k in (0, 2) for a, b in others[::17]]
    assert sum(1 for x in opp if x is not None and any(v != 0.0 for v in x["regret"])) > 5
    # an opponent's real action is not forced: its strategy there is not a point mass for every hole
    probes = [s._probe_path(2, a, b) for a, b in others[::13]]
    idx = s.path()[2]["index"]
    assert any(p is not None and p[idx] < 1.0 for p in probes)


# ============================================================ deals
def test_deals_never_share_a_card_and_follow_the_joint_distribution(trained):
    spec, _, game, _ = trained[3]
    st, acts = random_hand(spec, random.Random(21), Street.TURN, off_grid=0.0)
    s = make_search(game, st, acts)
    obs = st.observe(st.current_player)
    live = [i for i, w in enumerate(s.ranges()) if w is not None]
    board = set(obs.board)
    deals = s._sample_deals(40000, False, 3)
    for d in deals:
        cards = [d[2 * i] for i in live] + [d[2 * i + 1] for i in live] + d[2 * spec.n_players:]
        assert len(cards) == len(set(cards)) and not (set(cards) & board)
    focused = s._sample_deals(3000, True, 4)
    for d in focused:
        assert sorted(d[2 * obs.seat: 2 * obs.seat + 2]) == sorted(obs.hole)
        cards = [d[2 * i] for i in live] + [d[2 * i + 1] for i in live] + d[2 * spec.n_players:]
        assert len(cards) == len(set(cards))
    # two-player joint distribution: P(ours = h) is proportional to r_us(h) * sum of r_opp over holes
    # disjoint from h (card removal); compare with the sampled frequencies of the 12 likeliest holes
    spec2, _, game2, _ = trained[2]
    st2, acts2 = random_hand(spec2, random.Random(22), Street.RIVER, off_grid=0.0)
    s2 = make_search(game2, st2, acts2)
    r = s2.ranges()
    us = st2.current_player
    opp = 1 - us
    exact = []
    for i, (a, b) in enumerate(COMBOS):
        tot = sum(r[opp][j] for j, (x, y) in enumerate(COMBOS) if not ({a, b} & {x, y})) if r[us][i] > 0 else 0.0
        exact.append(r[us][i] * tot)
    z = sum(exact)
    n = 60000
    cnt = Counter(COMBO_INDEX[tuple(sorted(d[2 * us: 2 * us + 2]))] for d in s2._sample_deals(n, False, 9))
    top = sorted(range(1326), key=lambda i: -exact[i])[:12]
    for i in top:
        p = exact[i] / z
        sd = math.sqrt(p * (1 - p) / n)
        assert abs(cnt[i] / n - p) < 5 * sd + 1e-4, (COMBOS[i], cnt[i] / n, p)
    # the rest of the board is uniform over the unseen cards
    runouts = Counter(d[-1] for d in s2._sample_deals(20000, False, 5)) if len(st2.board) < 5 else None
    if runouts:
        assert max(runouts.values()) / min(runouts.values()) < 1.6


# ============================================================ exact check inside a river subgame
def test_river_subgame_average_strategy_converges_below_the_blueprint(trained):
    """A river root has no chance left: the exact best response over all hole pairs measures the
    search's strategy inside its own subgame.  The average strategy must get less exploitable with
    more iterations and end far below the blueprint played in the same spots."""
    spec, _, game, _ = trained[2]
    rng = random.Random(41)
    order = list(range(52))
    rng.shuffle(order)
    st = spec.new_hand(order, button=0)
    acts = []
    for name in ["r1", "c", "c", "c", "c", "c"]:  # SB opens, BB calls; checked down to the river
        a = spec.grid.to_concrete(st.observe(st.current_player), name)
        _act(st, acts, a)
    assert st.street == Street.RIVER
    ex = []
    for iters in (2_000, 1_000_000):
        s = make_search(game, st, acts, iterations=iters, time_budget=0.0, threads=4, seed=2)
        s.solve()
        ex.append(s.river_exploitability(0))
    blueprint = s.river_exploitability(2)
    pot_bb = s.root_info()["pot"] / spec.bb
    assert ex[1][0] < ex[0][0] and ex[1][0] < 0.25 * blueprint[0], (ex, blueprint)
    assert ex[1][0] < 0.05 * pot_bb, (ex, pot_bb)
    assert all(v >= -1e-9 for v in ex[1])  # a best response never loses against the strategy it answers


# ============================================================ the solver itself
def test_solve_respects_budgets_and_returns_distributions(trained):
    spec, _, game, _ = trained[2]
    st, acts = random_hand(spec, random.Random(31), Street.RIVER, off_grid=0.0)
    s = make_search(game, st, acts, iterations=3000, time_budget=0.0, threads=3)
    r = s.solve()
    assert r["iterations"] == 3000 and r["threads"] == 3
    assert abs(sum(r["final"]) - 1) < 1e-9 and abs(sum(r["average"]) - 1) < 1e-9 and r["visited"]
    s = make_search(game, st, acts, iterations=0, time_budget=0.4, threads=4)
    r = s.solve()
    assert 0.4 <= r["seconds"] < 0.6 and r["iterations"] > 100
    lik, missing = s.likelihood(1 - st.current_player)
    assert len(lik) == 1326 and all(0.0 <= x <= 1.0 for x in lik)
    with pytest.raises(RuntimeError):  # not our turn: nothing to solve
        core.SubgameSearch(game, list(st.starting_stacks), st.button, acts, list(st.board), 1 - st.current_player,
                           list(st.players[1 - st.current_player].hole), iterations=10).solve()
