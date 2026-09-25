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


def test_exact_evaluator_on_turn_and_river_roots(trained):
    """The prefix-sum showdowns equal the pairwise ones on river roots; on turn roots (the river card a
    chance node) the profile's values of the two seats sum to zero, best responses gain >= 0, and a
    best response on the river alone gains no more than one on the turn and the river."""
    spec, _, game, _ = trained[2]
    st, acts = line_hand(spec, ["r1", "c", "c", "c", "c", "c"])
    s = make_search(game, st, acts, iterations=20_000, time_budget=0.0, threads=4)
    s.solve()
    for kind in (0, 1, 2):
        assert list(s.river_exploitability(kind)) == pytest.approx(s._river_exploitability_slow(kind), rel=1e-9, abs=1e-12)
    st, acts = line_hand(spec, ["r1", "c", "c", "c"])
    assert st.street == Street.TURN
    # deterministic: the river cards' subtrees are summed in card order, whatever the thread count
    blueprint = [make_search(game, st, acts, threads=t).subgame_exploitability(2, b) for t in (1, 4) for b in (2, 3)]
    assert blueprint[:2] == blueprint[2:]
    for depth in ("end", "next_street"):
        s = make_search(game, st, acts, iterations=20_000, time_budget=0.0, threads=4, depth=depth)
        s.solve()
        for kind in (0, 1, 2):
            full = s.subgame_exploitability(kind, 2)
            river = s.subgame_exploitability(kind, 3)
            assert abs(full[3] + full[4]) < 1e-9 and full[3] == river[3], (depth, kind, full)
            assert min(full[1], full[2], river[1], river[2]) > -1e-9, (depth, kind, full, river)
            assert river[1] <= full[1] + 1e-9 and river[2] <= full[2] + 1e-9, (depth, kind, full, river)


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


# ============================================================ part 2: depth limits and leaves
def line_hand(spec: GameSpec, names, seed: int = 5):
    """A hand along grid action names (button 0), from a shuffled deck."""
    order = list(range(52))
    random.Random(seed).shuffle(order)
    st = spec.new_hand(order, button=0)
    acts = []
    for name in names:
        a = spec.grid.to_concrete(st.observe(st.current_player), name)
        _act(st, acts, a)
    return st, acts


LINES = {  # root street -> line to a decision on it (2 and 3 players, button 0)
    2: {Street.PREFLOP: [], Street.FLOP: ["r1", "c"], Street.TURN: ["r1", "c", "c", "c"], Street.RIVER: ["r1", "c", "c", "c", "c", "c"]},
    3: {Street.PREFLOP: [], Street.FLOP: ["r1", "c", "c"], Street.TURN: ["r1", "c", "c", "c", "c", "c"],
        Street.RIVER: ["r1", "c", "c", "c", "c", "c", "c", "c", "c"]},
}


@pytest.mark.parametrize("players", [2, 3])
def test_leaves_are_exactly_where_each_depth_rule_puts_them(trained, players):
    spec, _, game, _ = trained[players]
    for root, line in LINES[players].items():
        st, acts = line_hand(spec, line)
        assert st.street == root
        for mode in ("end", "pluribus", "hu_flop_limit", "next_street"):
            s = make_search(game, st, acts, depth=mode)
            info = s.root_info()
            if mode == "end" and root <= Street.FLOP:  # the whole rest of the game: too big to enumerate here
                assert info["limit_street"] == 3 and info["raise_limit"] == 0
                continue
            leaves, terminals, decisions = s._leaves(3_000_000)
            assert terminals > 0 and decisions > 0
            if mode == "end" or root == Street.RIVER:
                expect = None
            elif root == Street.PREFLOP:
                expect = Street.FLOP  # every rule: the first round searches to its end
            elif mode == "pluribus":
                expect = Street.TURN if (root == Street.FLOP and players > 2) else None
            elif mode == "hu_flop_limit":
                expect = Street.TURN if root == Street.FLOP else None
            else:
                expect = Street(int(root) + 1)
            if expect is None:
                assert leaves == [] and info["limit_street"] == 3, (mode, root, leaves[:2])
                continue
            assert leaves, (mode, root)
            raise_rule = mode == "pluribus" and root == Street.FLOP and players > 2
            for lf in leaves:
                if lf["reason"] == "next_street":
                    # the first node of the next street: nothing played on it yet, its cards dealt,
                    # at least two players left to act (else the engine runs the board out)
                    assert lf["street"] == int(expect) and lf["raises"] == 0, lf
                    assert lf["n_board"] == {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[expect]
                    assert lf["choosers"] >= 2
                else:
                    # right after the second raise (an all-in may leave one player to answer it)
                    assert raise_rule and lf["street"] == int(Street.FLOP) and lf["raises"] == 2, lf
                    assert lf["actions"][-1][0] == 2 and lf["choosers"] >= 1
            reasons = {lf["reason"] for lf in leaves}
            assert reasons == ({"next_street", "raise_limit"} if raise_rule else {"next_street"}), (mode, root, reasons)


def test_raise_limit_spares_the_real_path_up_to_our_decision(trained):
    spec, _, game, _ = trained[3]
    st, acts = line_hand(spec, ["r1", "c", "c", "r0.5", "r1"])  # flop: SB bets, BB raises: two raises, BTN to act
    assert st.street == Street.FLOP and st.raises_this_street == 2 and st.current_player == 0
    s = make_search(game, st, acts, depth="pluribus")
    leaves, terminals, decisions = s._leaves(100_000)
    path = [(p["type"], p["amount"]) for p in s.path()]
    assert len(path) == 2
    # nothing on the real path up to our decision is a leaf (our decision comes after the second raise) ...
    assert not any(lf["actions"] == path[:j] for lf in leaves for j in range(3))
    # ... and every action of ours ends in a leaf right there (or the hand ends)
    below = [lf for lf in leaves if lf["actions"][:2] == path]
    assert below and all(len(lf["actions"]) == 3 and lf["reason"] == "raise_limit" for lf in below)
    ours = len(s._rollout_policy(path, st.players[0].hole[0], st.players[0].hole[1], 0)[0])
    assert len(below) <= ours
    r = s.solve()
    assert r["visited"] and r["leaves"] > 0


def test_continuations_multiply_fold_call_or_raises_by_five_and_renormalise(trained):
    kinds = [0, 1, 2, 2, 2]
    p = [0.1, 0.4, 0.2, 0.2, 0.1]

    def py(choice):
        q = list(p)
        want = {1: 0, 2: 1, 3: 2}.get(choice)
        if want is None:
            return q
        s = 0.0
        for i, k in enumerate(kinds):
            if k == want:
                q[i] *= 5.0
            s += q[i]
        return [x / s for x in q]

    for choice in range(4):
        assert core.apply_continuation(kinds, p, choice) == py(choice)
    assert core.apply_continuation(kinds, p, 3) == [0.1 / 3.0, 0.4 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.5 / 3.0]
    # the rollout policy at real states: BlueprintStrategy.policy at the key BlueprintAgent builds
    # (bucket, history translated deterministically, off-grid sizes included), check / call when the
    # key is unknown, then the continuation
    spec, t, game, cbk = trained[2]
    strategy = t.strategy()
    grid = spec.grid
    rng = random.Random(8)
    checked = unknown = 0
    for trial in range(60):
        st, acts = random_hand(spec, rng, rng.choice([Street.FLOP, Street.TURN, Street.RIVER]), off_grid=0.3)
        root = int(st.street)
        # search from the root of this round, ask for the policy at the current state
        s = make_search(game, st, acts, depth="next_street" if root < 3 else "end")
        k0 = s.root_info()["n_events"]
        obs = st.observe(st.current_player)
        for hole in ([obs.hole[0], obs.hole[1]], None):
            if hole is None:
                free = [c for c in range(52) if c not in obs.board and c not in obs.hole]
                hole = rng.sample(free, 2)
            b = cbk.bucket(hole, list(obs.board))
            legal = grid.abstract_actions(obs)
            base = strategy.policy(infoset_key_for_bucket(obs, b, grid), legal)
            if base is None:
                unknown += 1
                base = [1.0 if n == "c" else 0.0 for n in legal]
            kinds_here = [0 if n == "f" else 1 if n == "c" else 2 for n in legal]
            for choice in range(4):
                names, got = s._rollout_policy(acts[k0:], hole[0], hole[1], choice)
                assert names == legal
                want = core.apply_continuation(kinds_here, base, choice)
                assert got == pytest.approx(want, abs=1e-15), (trial, choice)
                checked += 1
    assert checked > 400 and unknown < checked / 4


def test_leaf_choices_are_shared_by_a_players_indistinguishable_leaves(trained):
    spec, _, game, _ = trained[2]
    st, acts = _flop_hand(spec, flop_board=[0, 20, 44])  # a one-suit flop: the other three suits are interchangeable
    s = make_search(game, st, acts, depth="hu_flop_limit", iterations=4000, time_budget=0.0, threads=2, debug_leaves=20000)
    r = s.solve()
    log = s._leaf_log()
    assert len(log) == 20000 and r["leaves"] > 0
    groups = {}
    for e in log:
        groups.setdefault((e["seat"], e["cls"], e["path"]), []).append(e)
    # one key per (player, class of its hole on the flop, public path) ...
    for (seat, cls, path), es in groups.items():
        assert len({e["key"] for e in es}) == 1
    # ... whatever the turn card or the other player's hole at that leaf
    varied = [es for es in groups.values()
              if len({e["board"][3] for e in es}) > 1 and len({e["combos"][1 - e["seat"]] for e in es}) > 1]
    assert len(varied) > 10
    # and different classes, seats or paths never share a key
    keys = {}
    for (seat, cls, path), es in groups.items():
        assert keys.setdefault(es[0]["key"], (seat, cls, path)) == (seat, cls, path)
    # suit-isomorphic holes (one class) do share it
    assert any(len({e["combo"] for e in es}) > 1 for es in groups.values())


def test_leaf_values_use_the_given_number_of_rollouts(trained):
    spec, _, game, _ = trained[2]
    st, acts = line_hand(spec, ["r1", "c"])
    for n in (3, 1, 5):
        s = make_search(game, st, acts, depth="hu_flop_limit", iterations=300, time_budget=0.0, threads=2, rollouts=n)
        r = s.solve()
        assert r["leaf_evals"] > 0 and r["rollouts"] == n * r["leaf_evals"], (n, r["rollouts"], r["leaf_evals"])
        assert r["leaves"] > 0 and r["rollout_steps"] >= r["rollouts"]
    r = make_search(game, st, acts, depth="end", iterations=300, time_budget=0.0, threads=2).solve()
    assert r["leaves"] == r["leaf_evals"] == r["rollouts"] == 0


def test_search_tables_and_presampled_actions(trained, bucketer):
    """The per-search turn and river bucket tables give the bucketer's buckets; with pre-sampled
    actions a rollout plays one legal action of positive blueprint probability per infoset and
    continuation, the same every time."""
    from negpluribus.abstraction import PotentialAwareBucketer
    from negpluribus.fast.trainer import core_bucketer, spec_to_dict

    spec, t, _, _ = trained[2]
    pot = PotentialAwareBucketer(n_buckets=8, samples=6, bins=10).fit(n_situations=120, seed=3)
    pspec = GameSpec(n_players=2, stack_bb=30, max_street=Street.RIVER, n_buckets=8, max_raises_per_street=2,
                     preflop_fracs=(1.0,), postflop_fracs=(0.5, 1.0), bucket_kind="potential")
    pbk = core_bucketer(pot)
    pt = MCCFRTrainer(pspec, pot, seed=1, backend="cpp", threads=4).train(3000)
    game = core.SearchGame(spec_to_dict(pspec), pbk, pt.blueprint().lookup)
    st, acts = line_hand(pspec, ["r1", "c"])
    s = make_search(game, st, acts, depth="hu_flop_limit")
    info = s.root_info()
    assert info["river_table_boards"] == 1176
    rng = random.Random(4)
    free = [c for c in range(52) if c not in st.board]
    for _ in range(300):
        cards = rng.sample(free, 4)
        hole, extra = cards[:2], cards[2:]
        assert s._later_bucket(hole[0], hole[1], list(st.board) + extra[:1]) == pbk.bucket(hole, list(st.board) + extra[:1])
        assert s._later_bucket(hole[0], hole[1], list(st.board) + extra) == pbk.bucket(hole, list(st.board) + extra)
    # pre-sampled continuation actions (a game of its own: the choice is per SearchGame)
    game2 = core.SearchGame(spec_to_dict(spec), core_bucketer(bucketer), t.blueprint().lookup)
    rng = random.Random(9)
    checks = 0
    for trial in range(40):
        st, acts = random_hand(spec, rng, rng.choice([Street.FLOP, Street.TURN, Street.RIVER]), off_grid=0.0)
        s = make_search(game2, st, acts)
        k0 = s.root_info()["n_events"]
        h = st.players[st.current_player].hole
        full = {c: s._rollout_policy(acts[k0:], h[0], h[1], c) for c in range(4)}
        game2.presample(trial)
        for c in range(4):
            names, p = s._rollout_policy(acts[k0:], h[0], h[1], c)
            assert names == full[c][0]
            if max(full[c][1]) < 1.0 or p != full[c][1]:  # a known key: one action, of positive probability
                assert sorted(p) == [0.0] * (len(p) - 1) + [1.0]
                assert full[c][1][p.index(1.0)] > 0.0
                assert s._rollout_policy(acts[k0:], h[0], h[1], c)[1] == p
                checks += 1
        game2.clear_presampled()
        assert s._rollout_policy(acts[k0:], h[0], h[1], 0) == full[0]
    assert checks > 40


def test_our_average_is_accumulated_every_iteration(trained):
    """solve()["average"] adds up our decision's strategy once per iteration (our hole's own reach
    there is fixed), so it exists even when our hole is too unlikely in our range for the opponents'
    traversals to deal it (the node's accumulated average, "average_table", then stays uniform);
    where they do deal it, the two agree."""
    spec, _, game, _ = trained[2]
    st, acts = line_hand(spec, ["r1", "c", "c", "c", "c", "c", "c", "r1"])  # river: BB checks, SB bets pot
    assert st.street == Street.RIVER and st.current_player == 1
    seat = st.current_player
    s = make_search(game, st, acts, iterations=40_000, time_budget=0.0, threads=2)
    r = s.solve()
    tv = 0.5 * sum(abs(a - b) for a, b in zip(r["average"], r["average_table"]))
    assert abs(sum(r["average"]) - 1) < 1e-9 and tv < 0.15, (r["average"], r["average_table"])
    ours = s.class_of(core.combo_index(*st.players[seat].hole))
    w = s.ranges()[seat]
    s2 = make_search(game, st, acts, iterations=40_000, time_budget=0.0, threads=2)
    s2.set_reach(seat, [x * 1e-12 if s2.class_of(c) == ours else x for c, x in enumerate(w)])
    r2 = s2.solve()
    n = len(r2["average"])
    assert r2["average_table"] == pytest.approx([1.0 / n] * n)  # never dealt to the opponents' traversals
    assert abs(sum(r2["average"]) - 1) < 1e-9 and max(r2["average"]) > 1.0 / n + 0.2, r2["average"]


def test_a_search_with_leaves_converges_in_its_own_model(trained):
    """Leaves at the river start: the exact exploitability inside the search's own model (the best
    responder deviating on the turn and choosing its best continuation at a leaf, before the river
    card; the river then played by the continuations) falls with iterations, so the leaf values the
    search learns from are those of the model; a best responder free on the river gains at least as
    much, and both measure the same profile."""
    spec, _, game, _ = trained[2]
    st, acts = line_hand(spec, ["r1", "c", "c", "c"])
    assert st.street == Street.TURN
    out = {}
    for iters in (2_000, 100_000):
        s = make_search(game, st, acts, iterations=iters, time_budget=0.0, threads=4, depth="next_street")
        s.solve()
        for kind in (0, 1):
            model = s.subgame_exploitability(kind, 7)
            full = s.subgame_exploitability(kind, 2)
            assert model[3] == full[3] and model[4] == full[4]  # the same profile
            assert min(model[1], model[2]) > -1e-9 and model[1] <= full[1] + 1e-9 and model[2] <= full[2] + 1e-9, (model, full)
        out[iters] = s.subgame_exploitability(0, 7)[0]
    assert out[100_000] < 0.4 * out[2_000], out
    s = make_search(game, st, acts, iterations=100, time_budget=0.0, threads=2, depth="end")
    s.solve()
    with pytest.raises(RuntimeError, match="leaf game"):
        s.subgame_exploitability(0, 7)


def test_a_frozen_round_is_played_as_the_source_search_says(trained):
    """freeze_round (a measurement tool): every player plays the root's round as another solved search
    of the same spot says, and solve() learns only the later rounds.  On a river root everything is
    frozen: the profile, our strategy and the exact exploitability are the source's.  On a turn root
    the turn rows are the source's and the river is learned."""
    spec, _, game, _ = trained[2]
    st, acts = line_hand(spec, ["r1", "c", "c", "c", "c", "c"])
    assert st.street == Street.RIVER
    src = make_search(game, st, acts, iterations=5000, time_budget=0.0, threads=4)
    rs = src.solve()
    k = len(src.path())
    for kind in (0, 1):
        s = make_search(game, st, acts, iterations=500, time_budget=0.0, threads=4)
        s.freeze_round(src, kind)
        assert s.is_frozen
        r = s.solve()
        want = rs["average_table"] if kind == 0 else rs["final"]  # the profile's average (the node's)
        assert r["table_size"] == 0 and r["average"] == want and r["final"] == want
        assert s.path_strategies(k, 0) == s.path_strategies(k, 1) == src.path_strategies(k, kind)
        assert s.subgame_exploitability(0, 3) == s.subgame_exploitability(1, 3) == src.subgame_exploitability(kind, 3)
    other_st, other_acts = line_hand(spec, ["r1", "c", "c", "c", "c", "c"], seed=6)
    with pytest.raises(ValueError, match="another spot"):
        make_search(game, other_st, other_acts).freeze_round(src, 0)
    with pytest.raises(RuntimeError, match="solve the source"):
        s.freeze_round(make_search(game, st, acts), 0)
    s.freeze_round(None)
    assert not s.is_frozen
    # a turn root: the turn frozen to a search with leaves at the river start, the river learned
    st, acts = line_hand(spec, ["r1", "c", "c", "c"])
    assert st.street == Street.TURN
    src = make_search(game, st, acts, iterations=20_000, time_budget=0.0, threads=4, depth="next_street")
    rs = src.solve()
    k = len(src.path())
    river = {}
    for iters in (1, 20_000):
        s = make_search(game, st, acts, iterations=iters, time_budget=0.0, threads=4, depth="end")
        s.freeze_round(src, 0)
        r = s.solve()
        assert r["average"] == rs["average_table"] and s.path_strategies(k, 1) == src.path_strategies(k, 0)
        assert r["table_size"] > 0 and r["leaves"] == 0
        river[iters] = s.subgame_exploitability(0, 3)[0]
    assert river[20_000] < 0.5 * river[1], river  # the river play learned (after one iteration: uniform)


def test_bucket_caches_save_load_and_river_batch(bucketer, tmp_path):
    from negpluribus.abstraction import PotentialAwareBucketer
    from negpluribus.fast.trainer import core_bucketer

    assert core.canonical_boards(3) == 1755 and core.canonical_boards(4) == 16432
    cbk = core_bucketer(bucketer)
    rng = random.Random(2)
    forms = []
    for _ in range(300):
        cards = rng.sample(range(52), 6)
        forms.append((cards[:2], cards[2:]))
    values = [cbk.bucket(h, b) for h, b in forms]
    path = str(tmp_path / "cache.bin")
    cbk.save_cache(path, [1, 2])
    fresh = core_bucketer(bucketer)
    loaded = fresh.load_cache(path)
    assert [x[0] for x in loaded] == [1, 2] and sum(x[1] for x in loaded) >= 300 and all(x[2] == 0 for x in loaded)
    assert [fresh.bucket(h, b) for h, b in forms] == values
    assert fresh.cache_stats()["flop"]["computes"] == 0 and fresh.cache_stats()["turn"]["computes"] == 0  # all served by the cache
    other = EquityBucketer(bucketer.n_buckets, bucketer.samples)
    other.boundaries = {k: [x + 1e-9 for x in v] for k, v in bucketer.boundaries.items()}
    with pytest.raises(RuntimeError, match="other buckets"):
        core_bucketer(other).load_cache(path)
    # batch river buckets: the numbers of bucket() for potential-aware buckets; none for E[HS]
    pot = core_bucketer(PotentialAwareBucketer(n_buckets=8, samples=6, bins=10).fit(n_situations=120, seed=3))
    for _ in range(4):
        board = rng.sample(range(52), 5)
        allb = pot.river_buckets_all(board)
        for a, b in rng.sample(COMBOS, 200):
            if a in board or b in board:
                assert allb[core.combo_index(a, b)] == 255
            else:
                assert allb[core.combo_index(a, b)] == pot.bucket([a, b], board)
    assert cbk.river_buckets_all(rng.sample(range(52), 5)) is None
