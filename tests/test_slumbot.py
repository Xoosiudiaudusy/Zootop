"""Slumbot client: action-string parser, incr formatting, seat mapping and blinds, the HTTP client's
retry rules, and whole matches against a local mock server that speaks the same JSON protocol
(our engine + a bot on the server side).  Nothing here touches the internet."""
from __future__ import annotations

import json
import ssl
import urllib.error

import pytest

from negpluribus.abstraction import BetGrid, EquityBucketer
from negpluribus.agents import CallingAgent, GridRandomAgent, RandomAgent
from negpluribus.agents.base import Agent
from negpluribus.agents.blueprint import BlueprintAgent
from negpluribus.cards import cards_from_str
from negpluribus.cfr.strategy import BlueprintStrategy
from negpluribus.engine import CALL, FOLD, Street, raise_to
from negpluribus.slumbot import (
    HandAborted,
    MockSlumbot,
    Move,
    ProtocolError,
    SlumbotClient,
    SlumbotError,
    TransportError,
    bb100_ci,
    bot_seat,
    format_incr,
    hero_seat,
    make_deck,
    play_hand,
    read_log,
    replay,
    run_match,
    split_action,
)
from negpluribus.slumbot.runner import hand_seed

GRID = BetGrid(preflop_fracs=(1.0, 3.0), postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3)
HERO_HOLE = cards_from_str("Ac 9d")
BOT_HOLE = cards_from_str("Kh Kd")
BOARD = cards_from_str("2c 7d Jh 4s 9c")  # our pair of nines loses to Slumbot's kings


def play(action, client_pos=0, board=None, bot=None):
    """Replay ``action`` with our cards in our seat (client_pos 0 = we are the BB)."""
    return replay(action, make_deck(hero_seat(client_pos), HERO_HOLE, BOARD if board is None else board, bot))


def mock_client(mock, **kw):
    return SlumbotClient(host="http://mock", transport=mock.transport, min_interval=0, sleep=lambda s: None, **kw)


class Shover(Agent):
    """Goes all-in whenever it may raise, otherwise checks/calls."""

    def act(self, obs):
        return obs.clamp_raise(obs.max_raise_to) if obs.can_raise else CALL


class MinRaiser(Agent):
    def act(self, obs):
        return obs.clamp_raise(obs.min_raise_to) if obs.can_raise else CALL


class Folder(Agent):
    def act(self, obs):
        return FOLD if obs.can_fold else CALL


class OpenTo250(Agent):
    """First to act preflop: raises to 250 (an off-grid size); otherwise checks/calls."""

    def act(self, obs):
        if obs.street == Street.PREFLOP and not obs.events:
            return raise_to(250)
        return CALL


# ------------------------------------------------------------------ parser
def test_split_action_samples_from_the_protocol():
    assert split_action("b200c/kk/kk/kb200") == [
        [Move("b", 200), Move("c")], [Move("k"), Move("k")], [Move("k"), Move("k")], [Move("k"), Move("b", 200)]]
    assert split_action("b20000c///") == [[Move("b", 20000), Move("c")], [], [], []]
    assert split_action("b200c/kb400") == [[Move("b", 200), Move("c")], [Move("k"), Move("b", 400)]]
    assert split_action("") == [[]]
    assert split_action("ck/") == [[Move("c"), Move("k")], []]


@pytest.mark.parametrize("bad", ["x", "b", "bk", "b200c/kk/kk/kk/", "k1", "b-100", "B200"])
def test_split_action_rejects_bad_syntax(bad):
    with pytest.raises(ProtocolError):
        split_action(bad)


def test_replay_river_sample():
    st = play("b200c/kk/kk/kb200").state  # we are the BB: SB raised, we called, checks, SB bets the river
    assert st.street == Street.RIVER and st.current_player == hero_seat(0) == 1
    assert st.current_bet == 200 and st.pot == 600 and st.to_call_for(1) == 200
    assert st.board == BOARD


def test_replay_pot_size_flop_bet_sample():
    st = play("b200c/kb400", board=BOARD[:3]).state
    assert st.street == Street.FLOP and st.pot == 800 and st.current_player == 1
    assert [p.street_bet for p in st.players] == [400, 0]  # 400 = the pot after preflop: a pot-size bet


@pytest.mark.parametrize("action", ["b20000c///", "b20000c"])
def test_replay_all_in_with_empty_streets(action):
    rp = play(action, bot=BOT_HOLE)  # we (BB, Ac 9d) call Slumbot's (SB, Kh Kd) preflop shove
    st = rp.state
    assert st.is_terminal and st.board == BOARD and st.pot == 40000
    assert [p.all_in for p in st.players] == [True, True]
    assert st.record().net[hero_seat(0)] == -20000
    assert not rp.ended_by_fold


def test_replay_flop_all_in_with_two_empty_streets():
    rp = play("b200c/b10000b19800c//", bot=BOT_HOLE)  # all-in raise below a full raise is allowed
    assert rp.state.is_terminal and rp.state.pot == 40000 and rp.state.record().net[1] == -20000


def test_replay_limp_check_and_fold():
    st = play("ck/", board=BOARD[:3]).state
    assert st.street == Street.FLOP and st.pot == 200 and st.current_player == 1  # BB first postflop
    rp = play("f", board=[])
    assert rp.ended_by_fold and rp.state.record().net[hero_seat(0)] == 50  # SB folded to us


@pytest.mark.parametrize("action", [
    "b300b500",             # re-raise by exactly the last increment
    "cb300",                # limp, BB raises by one big blind (min)
    "b200c/b100",           # min bet postflop = the big blind
    "b200c/b400b800",       # min raise postflop
    "b200c/b10000b19800",   # all-in raise smaller than a full raise
    "b200c",                # street over, no trailing '/'
    "b200c/",               # street over, trailing '/'
])
def test_replay_accepts_legal_edges(action):
    replay(action, make_deck(1, HERO_HOLE, BOARD[:3]))


@pytest.mark.parametrize("action", [
    "k",                    # SB cannot check preflop
    "b150",                 # raise below the minimum (to 200)
    "b100",                 # 'raise' to the big blind is no raise
    "b20001",               # more than the stack
    "b300b499",             # re-raise below the last increment
    "cb150",                # BB raise below one big blind
    "b200c/b99",            # bet below the big blind
    "b200c/b400b799",       # raise below the bet
    "b200c/c",              # call with nothing to call
    "b200c/f",              # fold when checking is free
    "b200/c",               # '/' before the street is over
    "b200ck",               # missing '/' after a finished street
    "b200c//",              # '/' closing a street that has not started
    "fk",                   # move after the hand is over
    "f/",                   # characters after a fold
    "b20000c/",             # partial slashes after an all-in call
    "b20000c//",
    "b200c/kk/kk/kkk",      # move after the showdown
])
def test_replay_rejects_illegal(action):
    with pytest.raises(ProtocolError):
        replay(action, make_deck(1, HERO_HOLE, BOARD))


# ------------------------------------------------------ seats, blinds, incr
def test_seat_mapping():
    assert hero_seat(1) == 0 and bot_seat(1) == 1  # client_pos 1: we are the SB = our button seat 0
    assert hero_seat(0) == 1 and bot_seat(0) == 0  # client_pos 0: we are the BB
    with pytest.raises(ProtocolError):
        hero_seat(2)


def test_blinds_and_order_follow_client_pos():
    st = play("", client_pos=1, board=[]).state  # we are the SB
    me, bot = hero_seat(1), bot_seat(1)
    assert st.players[me].street_bet == 50 and st.players[bot].street_bet == 100
    assert st.current_player == me and st.to_call_for(me) == 50 and st.pot == 150  # SB acts first preflop
    assert st.players[me].hole == HERO_HOLE
    st = play("b200c", client_pos=0, board=BOARD[:3]).state  # we are the BB
    assert st.players[hero_seat(0)].street_bet == 0 and st.current_player == hero_seat(0)  # BB first postflop


def test_format_incr():
    obs = play("", client_pos=1, board=[]).state.observe(0)  # SB, 50 to call
    assert format_incr(CALL, obs) == "c" and format_incr(FOLD, obs) == "f"
    assert format_incr(raise_to(obs.min_raise_to), obs) == "b200"  # min raise: preflop the blind counts
    assert format_incr(raise_to(obs.max_raise_to), obs) == "b20000"  # all-in
    for bad in (199, 20001):
        with pytest.raises(ValueError):
            format_incr(raise_to(bad), obs)
    obs = play("b300", client_pos=0, board=[]).state.observe(1)  # BB facing a raise to 300
    assert format_incr(raise_to(obs.min_raise_to), obs) == "b500" and format_incr(CALL, obs) == "c"
    obs = play("b200c", client_pos=0, board=BOARD[:3]).state.observe(1)  # BB first on the flop
    assert format_incr(CALL, obs) == "k"
    assert format_incr(raise_to(obs.min_raise_to), obs) == "b100"
    assert format_incr(raise_to(obs.max_raise_to), obs) == "b19800"  # all-in counts this street only
    with pytest.raises(ValueError):
        format_incr(FOLD, obs)


def test_observation_does_not_depend_on_unknown_cards():
    """Slumbot's hole cards and the undealt board are placeholders; the agent's view must not
    change when they change."""
    action, board = "b300c/kb450", BOARD[:3]
    a = replay(action, make_deck(1, HERO_HOLE, board)).state
    b = replay(action, make_deck(1, HERO_HOLE, board, bot_hole=BOT_HOLE)).state
    assert a.players[0].hole != b.players[0].hole  # the bot's (unknown) cards differ...
    assert a.observe(1) == b.observe(1)  # ...and our observation is identical
    assert a.observe(1).hole == HERO_HOLE and a.observe(1).board == board


def test_make_deck_rejects_inconsistent_cards():
    with pytest.raises(ProtocolError):
        make_deck(0, HERO_HOLE, [HERO_HOLE[0]] + BOARD[1:3])
    with pytest.raises(ProtocolError):
        make_deck(0, HERO_HOLE, BOARD, bot_hole=[BOARD[0], BOT_HOLE[0]])


# ------------------------------------------------------------- HTTP client
class ScriptedTransport:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, dict(payload)))
        a = self.answers.pop(0)
        if isinstance(a, BaseException):
            raise a
        status, body = a
        return status, body if isinstance(body, bytes) else json.dumps(body).encode()


def test_client_retries_5xx_with_growing_backoff():
    ok = {"action": "", "client_pos": 1, "hole_cards": ["Ac", "9d"], "board": [], "token": "T1"}
    tr = ScriptedTransport([(503, b"busy"), (502, b""), urllib.error.URLError("reset"), (200, ok)])
    sleeps = []
    c = SlumbotClient(host="https://example.invalid", transport=tr, min_interval=0, sleep=sleeps.append, backoff_base=1.0)
    assert c.new_hand() == ok and c.token == "T1"
    assert c.n_requests == 4 and c.n_retries == 3
    assert tr.calls[0][0] == "https://example.invalid/slumbot/api/new_hand" and tr.calls[0][1] == {}
    assert len(sleeps) == 3 and 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 2.0 and 2.0 <= sleeps[2] <= 4.0


def test_client_gives_up_after_max_retries():
    tr = ScriptedTransport([urllib.error.URLError("down")] * 3)
    sleeps = []
    c = SlumbotClient(host="https://x", transport=tr, min_interval=0, sleep=sleeps.append, max_retries=2)
    with pytest.raises(TransportError):
        c.new_hand()
    assert c.n_requests == 3 and len(sleeps) == 2


def test_client_does_not_retry_client_errors_or_certificate_failures():
    tr = ScriptedTransport([(400, {"error_msg": "Illegal action"})])
    c = SlumbotClient(host="https://x", token="T", transport=tr, min_interval=0, sleep=lambda s: None)
    with pytest.raises(SlumbotError, match="Illegal action") as e:
        c.act("k")
    assert e.value.status == 400 and c.n_requests == 1
    tr = ScriptedTransport([(200, {"error_msg": "Bad token"})])
    c = SlumbotClient(host="https://x", token="T", transport=tr, min_interval=0, sleep=lambda s: None)
    with pytest.raises(SlumbotError, match="Bad token"):
        c.act("c")
    cert = urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))
    c = SlumbotClient(host="https://x", transport=ScriptedTransport([cert]), min_interval=0, sleep=lambda s: None)
    with pytest.raises(TransportError):
        c.new_hand()
    assert c.n_requests == 1
    with pytest.raises(SlumbotError):
        SlumbotClient(host="https://x", transport=ScriptedTransport([])).act("c")  # no token yet


def test_client_keeps_the_latest_token_and_paces_requests():
    tr = ScriptedTransport([(200, {"token": "T1"}), (200, {"token": "T2"}), (200, {})])
    now = [100.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    c = SlumbotClient(host="https://x", transport=tr, min_interval=0.5, sleep=sleep, clock=lambda: now[0])
    c.new_hand()
    c.act("c")
    now[0] += 0.2
    c.act("k")
    assert [p for _, p in tr.calls] == [{}, {"token": "T1", "incr": "c"}, {"token": "T2", "incr": "k"}]
    assert c.token == "T2" and c.n_token_changes == 1
    assert sleeps == pytest.approx([0.5, 0.3])  # never two requests closer than min_interval


# ------------------------------------------------------ matches vs the mock
def _check_against_truth(log, mock, expect_all_verified=True):
    hands = [r for r in read_log(log) if r["type"] == "hand"]
    assert len(hands) == len(mock.finished)
    for rec, truth in zip(hands, mock.finished):
        assert rec["status"] == "ok"
        assert rec["winnings"] == truth.client_net
        assert rec["action"] == truth.action and rec["client_pos"] == truth.client_pos
        assert rec["hero_cards"] == truth.client_cards and rec["board"] == truth.board
        assert rec["check"] in (("match",) if expect_all_verified else ("match", "unverified"))
        if rec["check"] == "match":
            assert rec["engine_net"] == truth.client_net
        assert rec["session_check"] == "match"  # Slumbot-style session counters agree hand by hand
    return hands


def test_match_over_local_http_matches_server_accounting(tmp_path):
    """The default urllib transport against the mock running as a real HTTP server."""
    mock = MockSlumbot(RandomAgent(seed=3), seed=5)
    log = str(tmp_path / "http.jsonl")
    with mock.serve() as host:
        client = SlumbotClient(host=host, min_interval=0, timeout=10)
        tally = run_match(client, GridRandomAgent(GRID, seed=1), log, 40, seed=7, out=lambda s: None)
    hands = _check_against_truth(log, mock)
    assert tally.n_hands == 40 and sum(tally.winnings) == sum(h.client_net for h in mock.finished)
    assert {r["client_pos"] for r in hands} == {0, 1}  # both seats played
    assert mock.requests[0][0] == "/slumbot/api/new_hand" and "token" not in mock.requests[0][1]
    assert all(p.get("token") for _, p in mock.requests[1:])


@pytest.mark.parametrize("opts", [
    dict(allin_slashes=True, reveal_bot_cards="showdown"),
    dict(allin_slashes=False, street_slash=False, reveal_bot_cards="showdown"),
    dict(allin_slashes=True, reveal_bot_cards="never"),
])
def test_all_in_hands_with_empty_streets(tmp_path, opts):
    mock = MockSlumbot(Shover(), seed=11, **opts)
    log = str(tmp_path / "allin.jsonl")
    tally = run_match(mock_client(mock), CallingAgent(), log, 30, out=lambda s: None)
    hands = _check_against_truth(log, mock, expect_all_verified=opts["reveal_bot_cards"] != "never")
    for rec in hands:
        tail = "c///" if opts["allin_slashes"] else "c"
        assert rec["action"] in ("b20000" + tail, "cb20000" + tail)  # we call as BB / limp-call as SB
        assert abs(rec["winnings"]) in (0, 20000) and len(rec["board"]) == 5
        if opts["reveal_bot_cards"] == "never":
            assert rec["check"] == "unverified" and rec["bot_cards"] is None
    assert tally.n_hands == 30 and sum(tally.winnings) == sum(h.client_net for h in mock.finished)


def test_random_play_all_protocol_variants(tmp_path):
    for i, opts in enumerate([dict(), dict(allin_slashes=False, street_slash=False), dict(rotate_token_every=3)]):
        mock = MockSlumbot(RandomAgent(seed=20 + i), seed=30 + i, **opts)
        client = mock_client(mock)
        log = str(tmp_path / f"v{i}.jsonl")
        run_match(client, GridRandomAgent(GRID, seed=40 + i), log, 60, out=lambda s: None)
        _check_against_truth(log, mock)
        if opts.get("rotate_token_every"):
            assert client.n_token_changes == 19  # a new token before hands 4, 7, ..., 58


def test_bot_folding_in_the_new_hand_response(tmp_path):
    mock = MockSlumbot(Folder(), seed=2, first_client_pos=0)  # we are the BB first: Slumbot acts first
    rec = play_hand(mock_client(mock), CallingAgent(), hand_no=0, trace=True)
    assert rec["action"] == "f" and rec["winnings"] == 50 and rec["decisions"] == []
    assert rec["check"] == "match" and rec["check_basis"] == "fold"
    assert len(rec["responses"]) == 1 and rec["responses"][0]["winnings"] == 50  # the new_hand answer ended it
    assert "token" not in rec["responses"][0] and "token" not in rec["final"]
    assert rec["responses"][0]["_token"] == "new" and rec["responses"][0]["_ms"] >= 0


def test_double_applied_move_after_a_lost_response_aborts_the_hand():
    mock = MockSlumbot(MinRaiser(), seed=4, first_client_pos=1)  # we are the SB and act first
    mock.lose_next_act = 1  # our first move is applied, then the answer is lost (504) and retried
    with pytest.raises(HandAborted, match="did not send") as e:
        play_hand(mock_client(mock), CallingAgent(), hand_no=0)
    assert e.value.record["status"] == "error" and e.value.record["n_retries"] == 1
    assert "winnings" not in e.value.record


def test_runner_stops_after_repeated_failures(tmp_path):
    mock = MockSlumbot(RandomAgent(seed=1), seed=1)
    mock.fail_next = [400] * 10  # every request refused
    log = str(tmp_path / "fail.jsonl")
    tally = run_match(mock_client(mock), CallingAgent(), log, 50, max_errors=3, out=lambda s: None)
    assert tally.n_errors == 3 and tally.n_hands == 0 and len(mock.requests) == 3
    assert bb100_ci(tally.winnings) == (0.0, float("inf"))
    errors = [r for r in read_log(log) if r["type"] == "hand"]
    assert [r["status"] for r in errors] == ["error"] * 3 and all("winnings" not in r for r in errors)
    assert errors[0]["error_response"] == {"error_msg": "injected failure 400"}


class FailNthAct:
    """Transport that answers the n-th act request with a 400 (the mock never sees it)."""

    def __init__(self, mock, n):
        self.mock, self.n, self.seen = mock, n, 0

    def __call__(self, url, payload, timeout):
        if url.endswith("/act"):
            self.seen += 1
            if self.seen == self.n:
                return 400, json.dumps({"error_msg": "injected"}).encode()
        return self.mock.transport(url, payload, timeout)


def test_abandoned_hand_shows_up_as_a_session_gap(tmp_path):
    """A hand we had to abandon has no result on our side; if the server counts it (here: the
    next new_hand forfeits it), Slumbot's session counters jump and its chips are recorded, so
    results + gap chips add up to the server's total."""
    mock = MockSlumbot(RandomAgent(seed=3), seed=5, abandoned_hand="forfeit")
    client = SlumbotClient(host="http://mock", transport=FailNthAct(mock, 4), min_interval=0, sleep=lambda s: None)
    log = str(tmp_path / "gap.jsonl")
    tally = run_match(client, GridRandomAgent(GRID, seed=1), log, 12, out=lambda s: None)
    hands = [r for r in read_log(log) if r["type"] == "hand"]
    assert [r["status"] for r in hands].count("error") == 1 and tally.n_errors == 1
    forfeits = [h for h in mock.finished if h.forfeit]
    assert len(forfeits) == 1
    gaps = [r for r in hands if r.get("session_gap")]
    assert len(gaps) == 1 and gaps[0]["session_gap"] == {"hands": 1, "chips": forfeits[0].client_net}
    assert tally.gap_hands == 1 and tally.gap_chips == forfeits[0].client_net
    assert sum(tally.winnings) + tally.gap_chips == sum(h.client_net for h in mock.finished)
    assert all(r["session_check"] in ("match", "gap") for r in hands if r["status"] == "ok")


def test_runner_resumes_the_log(tmp_path):
    log = str(tmp_path / "resume.jsonl")
    mock = MockSlumbot(RandomAgent(seed=5), seed=6)
    run_match(mock_client(mock), GridRandomAgent(GRID, seed=1), log, 6, seed=3, out=lambda s: None)
    with open(log, "a", encoding="utf-8") as f:
        f.write('{"type": "hand", "hand": 99, "stat')  # a line cut short by a crash
    mock2 = MockSlumbot(RandomAgent(seed=7), seed=8)
    tally = run_match(mock_client(mock2), GridRandomAgent(GRID, seed=1), log, 100, seed=3, until=10, out=lambda s: None)
    recs = read_log(log)
    runs = [r for r in recs if r["type"] == "run"]
    hands = [r for r in recs if r["type"] == "hand"]
    assert len(runs) == 2 and runs[1]["first_hand"] == 6
    assert [r["hand"] for r in hands] == list(range(10)) and tally.n_hands == 10
    assert [r["seed"] for r in hands] == [hand_seed(3, i) for i in range(10)]
    assert tally.winnings == [h.client_net for h in mock.finished + mock2.finished]


# --------------------------------------------------- blueprint decision log
def _tiny_blueprint_agent():
    bk = EquityBucketer(n_buckets=4, samples=30)
    bk.boundaries = {1: [0.35, 0.5, 0.65], 2: [0.35, 0.5, 0.65], 3: [0.35, 0.5, 0.65]}
    names = ["f", "c", "r1", "r3", "a"]
    table = {f"P|BB|2|b{i}|r1": (names, [0.0, 1.0, 0.0, 0.0, 0.0]) for i in range(169)}  # BB vs a raise: call
    return BlueprintAgent(BlueprintStrategy(table), bk, GRID, seed=1)


def test_blueprint_decisions_are_logged_with_translation_and_off_map():
    mock = MockSlumbot(OpenTo250(), seed=9, first_client_pos=0)  # Slumbot (SB) opens to 250 = 0.75 pot
    agent = _tiny_blueprint_agent()
    rec = play_hand(mock_client(mock), agent, hand_no=0, seed=1)
    first, second = rec["decisions"][0], rec["decisions"][1]
    assert first["before"] == "b250" and first["legal"] == ["f", "c", "r1", "r3", "a"]
    assert first["key"].startswith("P|BB|2|b") and first["key"].endswith("|r1")  # 0.75 pot read as r1
    assert first["off_map"] is False and first["abstract"] == "c" and first["incr"] == "c"
    assert first["translations"] == [
        {"i": 0, "street": "preflop", "to": 250, "frac": 0.75, "all_in": False, "mapped": "r1", "on_grid": False}]
    assert second["street"] == "flop" and second["off_map"] is True and second["incr"] == "k"  # fallback: check
    assert rec["check"] == "match" and rec["winnings"] == mock.finished[0].client_net
    assert agent.n_fallback == sum(d["off_map"] for d in rec["decisions"])


def test_bb100_ci():
    assert bb100_ci([100, -100]) == pytest.approx((0.0, 196.0))
    assert bb100_ci([200, 200, 200]) == pytest.approx((200.0, 0.0))
    assert bb100_ci([150])[1] == float("inf")
