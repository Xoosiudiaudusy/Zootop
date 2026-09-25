"""Slumbot hands in our engine: replay the action string, ask our agent, send its move back.

At every decision the whole action string is replayed into a fresh ``HandState`` (two seats,
button 0, blinds 50/100, stacks 20,000).  The engine then checks every Slumbot move against the
rules (turn order, check/call/fold legality, min-raise, all-in cap, street boundaries) and gives
our agent the same ``Observation`` it gets in training and evaluation, including the ``Event``
log from which ``BlueprintAgent`` translates Slumbot's bet sizes onto its grid.

The deck is built to fit what we know: our hole cards in our seat, the revealed board in board
order.  Slumbot's hole cards are unknown during the hand; its seat gets placeholder cards (the
lowest unused ones) and the not-yet-dealt board gets placeholders too.  Neither can reach the
agent: an ``Observation`` carries only our hole cards and the dealt board (tested in
tests/test_slumbot.py).  At the end of the hand the real board and, when Slumbot shows them, its
hole cards go into the deck, and the engine's chip count for our seat is compared with Slumbot's
``winnings``.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, List, Optional, Sequence, Tuple

from ..abstraction import ALL_IN, BetGrid
from ..agents.base import Agent
from ..agents.blueprint import BlueprintAgent
from ..cards import Deck
from ..engine import (
    CALL,
    FOLD,
    Action,
    ActionType,
    HandState,
    Observation,
    STREET_NAMES,
    Street,
    raise_to,
)
from .client import SlumbotClient, SlumbotError, TransportError
from .protocol import (
    BIG_BLIND,
    BUTTON_SEAT,
    NUM_STREETS,
    POSITION_NAMES,
    SMALL_BLIND,
    STACK_SIZE,
    Move,
    ProtocolError,
    cards_str,
    format_incr,
    hero_seat,
    parse_cards,
    parse_incr,
    split_action,
)

AppliedMove = Tuple[int, int, Move]  # (street index, engine seat, move)

BOT_CARD_KEYS = ("bot_hole_cards", "bot_cards", "opp_hole_cards")  # only the first is expected


# ------------------------------------------------------------------ deck + replay
def new_state(deck: Deck) -> HandState:
    """A fresh Slumbot hand in our engine (button = seat 0 = small blind)."""
    return HandState([STACK_SIZE, STACK_SIZE], BUTTON_SEAT, SMALL_BLIND, BIG_BLIND, deck=deck, max_street=Street.RIVER)


def make_deck(
    hero: int,
    hero_hole: Sequence[int],
    board: Sequence[int],
    bot_hole: Optional[Sequence[int]] = None,
) -> Deck:
    """Deck order for ``new_state``: seat 0's two cards, seat 1's two cards, then the board.

    Unknown cards (Slumbot's hole cards before a showdown, board cards not dealt yet) are the
    lowest cards not in use; they never reach our agent."""
    if len(hero_hole) != 2:
        raise ProtocolError(f"need 2 hole cards, got {len(hero_hole)}")
    if len(board) > 5:
        raise ProtocolError(f"board with {len(board)} cards")
    if bot_hole is not None and len(bot_hole) != 2:
        raise ProtocolError(f"need 2 bot hole cards, got {len(bot_hole)}")
    known = list(hero_hole) + list(board) + (list(bot_hole) if bot_hole is not None else [])
    if len(set(known)) != len(known):
        raise ProtocolError(f"duplicate cards in hole/board/bot cards: {cards_str(known)}")
    spare = iter(c for c in range(52) if c not in set(known))
    holes: List[List[int]] = [[], []]
    holes[hero] = list(hero_hole)
    holes[1 - hero] = list(bot_hole) if bot_hole is not None else [next(spare), next(spare)]
    runout = list(board) + [next(spare) for _ in range(5 - len(board))]
    order = holes[0] + holes[1] + runout
    used = set(order)
    order += [c for c in range(52) if c not in used]
    return Deck.from_order(order)


def engine_action(state: HandState, seat: int, mv: Move, action: str = "") -> Action:
    """One Slumbot move -> engine action, with Slumbot's legality rules."""
    to_call = state.to_call_for(seat)
    if mv.code == "k":
        if to_call:
            raise ProtocolError(f"check facing a bet ({to_call} to call) in {action!r}")
        return CALL
    if mv.code == "c":
        if not to_call:
            raise ProtocolError(f"call with nothing to call in {action!r}")
        return CALL
    if mv.code == "f":
        if not to_call:
            raise ProtocolError(f"fold when checking is free in {action!r}")
        return FOLD
    if mv.code == "b":
        can, lo, hi = state.raise_bounds(seat)
        if not can:
            raise ProtocolError(f"{mv} where no bet/raise is possible in {action!r}")
        if not lo <= mv.amount <= hi:
            raise ProtocolError(f"{mv} outside the legal range [b{lo}, b{hi}] in {action!r}")
        return raise_to(mv.amount)
    raise ProtocolError(f"unknown move {mv!r}")


class Replay:
    """The engine state after an action string, and who made each move."""

    def __init__(self, state: HandState, moves: List[AppliedMove]):
        self.state = state
        self.moves = moves

    @property
    def ended_by_fold(self) -> bool:
        return bool(self.moves) and self.moves[-1][2].code == "f"


def replay(action: str, deck: Deck) -> Replay:
    """Apply a Slumbot action string to a fresh hand.  Raises ``ProtocolError`` on anything the
    rules or the grammar forbid: an illegal move, a '/' where the street is not over, a move on
    a street that is over without a '/', moves after the hand ended, extra characters after a
    fold, or a partial set of slashes after an all-in call (the official parser accepts either
    none or all of them)."""
    streets = split_action(action)
    state = new_state(deck)
    moves: List[AppliedMove] = []
    for st, street_moves in enumerate(streets):
        if st > 0 and not state.is_terminal and int(state.street) != st:
            raise ProtocolError(f"'/' before the {STREET_NAMES[st - 1]} was over in {action!r}")
        for mv in street_moves:
            if state.is_terminal:
                raise ProtocolError(f"{mv} after the hand was over in {action!r}")
            if int(state.street) != st:
                raise ProtocolError(f"missing '/' before {mv} in {action!r}")
            seat = state.current_player
            assert seat is not None
            state.apply(engine_action(state, seat, mv, action))
            moves.append((st, seat, mv))
    if state.is_terminal and moves:
        last_street = moves[-1][0]
        slashes_after = len(streets) - 1 - last_street
        if moves[-1][2].code == "f":
            if slashes_after:
                raise ProtocolError(f"characters after a fold in {action!r}")
        elif last_street < NUM_STREETS - 1 and slashes_after not in (0, NUM_STREETS - 1 - last_street):
            raise ProtocolError(f"after an all-in call expected no '/' or {NUM_STREETS - 1 - last_street}, got {slashes_after} in {action!r}")
    return Replay(state, moves)


# --------------------------------------------------------------- decision logging
class _RecordingStrategy:
    """Wraps a blueprint strategy; answers exactly as the wrapped one and remembers the last
    lookup (key, legal abstract actions, probabilities or None = off-map)."""

    def __init__(self, inner):
        self.inner = inner
        self.last: Optional[Tuple[str, List[str], Optional[List[float]]]] = None

    def policy(self, key, legal):
        probs = self.inner.policy(key, legal)
        self.last = (key, list(legal), None if probs is None else list(probs))
        return probs

    def __len__(self) -> int:
        return len(self.inner)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def history_names(key: str) -> List[str]:
    """Abstract action names in an infoset key's history, one per event
    (key = ``street|position|n_active|b<bucket>|<history>``, streets in the history split by '/')."""
    parts = key.split("|", 4)
    if len(parts) < 5:
        return []
    return parts[4].replace("/", " ").split()


def translations(events, names: Sequence[str], hero: int) -> List[dict]:
    """How each of Slumbot's bets/raises was read by our grid (the pseudo-harmonic mapping the
    agent used, taken from its infoset key).  ``frac`` is the observed raise increment over the
    pot after calling; ``on_grid``: within half a chip of the mapped size (or an all-in read as
    all-in), i.e. no translation was needed."""
    out = []
    for i, ev in enumerate(events):
        if ev.seat == hero or ev.action.type != ActionType.RAISE:
            continue
        name = names[i] if i < len(names) else None
        x = BetGrid.observed_frac(ev)
        pot_after_call = ev.pot_before + ev.to_call
        if name == ALL_IN:
            on_grid = bool(ev.all_in)
        elif name and name.startswith("r"):
            on_grid = abs((x - float(name[1:])) * pot_after_call) <= 0.5 + 1e-9
        else:
            on_grid = False
        out.append({
            "i": i,
            "street": STREET_NAMES[ev.street],
            "to": ev.action.amount,
            "frac": round(x, 4),
            "all_in": bool(ev.all_in),
            "mapped": name,
            "on_grid": on_grid,
        })
    return out


class DecisionExplainer:
    """What our agent decided and why, for the per-hand log.

    For a ``BlueprintAgent`` it wraps ``agent.strategy`` in a recording proxy (same answers,
    nothing in the agent changes), so the log shows the exact infoset key the agent looked up,
    its legal abstract actions and probabilities, whether it was off-map (key never seen in
    training -> the agent's check/call fallback), which abstract action the concrete move is,
    and how every Slumbot bet was translated.  Other agents get only the concrete move."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.proxy: Optional[_RecordingStrategy] = None
        if isinstance(agent, BlueprintAgent):
            if isinstance(agent.strategy, _RecordingStrategy):
                self.proxy = agent.strategy
            else:
                self.proxy = _RecordingStrategy(agent.strategy)
                agent.strategy = self.proxy

    def before_act(self) -> None:
        if self.proxy is not None:
            self.proxy.last = None

    def explain(self, obs: Observation, action: Action, hero: int) -> dict:
        if self.proxy is None or self.proxy.last is None:
            return {}
        key, legal, probs = self.proxy.last
        grid: BetGrid = self.agent.grid  # type: ignore[attr-defined]
        abstract = next((n for n in legal if grid.to_concrete(obs, n) == action), None)
        return {
            "key": key,
            "legal": legal,
            "probs": None if probs is None else [round(p, 4) for p in probs],
            "abstract": abstract,
            "off_map": probs is None,
            "translations": translations(obs.events, history_names(key), hero),
        }


# -------------------------------------------------------------------- one hand
class HandAborted(Exception):
    """A hand that could not be finished; ``record`` holds what is known (status "error")."""

    def __init__(self, message: str, record: dict):
        super().__init__(message)
        self.record = record


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _client_pos(resp: dict) -> int:
    pos = resp.get("client_pos")
    if isinstance(pos, bool) or not isinstance(pos, int) or pos not in (0, 1):
        raise ProtocolError(f"client_pos must be 0 or 1, got {pos!r}")
    return pos


def bot_cards_of(resp: dict) -> Tuple[Optional[List[int]], Optional[str]]:
    """Slumbot's hole cards if the response shows them (and under which key)."""
    for k in BOT_CARD_KEYS:
        v = resp.get(k)
        if v:
            return parse_cards(v), k
    return None, None


def check_continuation(prev: List[AppliedMove], ours: Optional[AppliedMove], moves: List[AppliedMove], hero: int, action: str) -> None:
    """The new action string must be the old one, then our move, then Slumbot's moves only.
    Catches a server that misread our move and a retried ``act`` that was applied twice."""
    expected = prev + ([ours] if ours is not None else [])
    if moves[: len(expected)] != expected:
        want = "".join(str(m[2]) for m in expected)
        raise ProtocolError(f"{action!r} does not continue the hand as we know it (expected the moves {want!r} first)")
    for st, seat, mv in moves[len(expected):]:
        if seat == hero:
            raise ProtocolError(f"{action!r} holds a move for us that we did not send ({mv} on the {STREET_NAMES[st]})")


def play_hand(
    client: SlumbotClient,
    agent: Agent,
    explainer: Optional[DecisionExplainer] = None,
    *,
    hand_no: Optional[int] = None,
    seed: Optional[int] = None,
    trace: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Play one hand against the server; return its log record (see docs/slumbot.md).

    ``seed``: the agent is re-seeded with it before the hand (``Agent.reset``), so a hand's
    random choices depend only on (seed, what Slumbot did).  ``trace``: also keep every raw
    response of the hand under ``"responses"``, with the token value replaced by our own
    ``"_token"`` ("absent" / "same" / "new") and the request's latency ``"_ms"``.  Raises
    ``HandAborted`` (with the partial record) on protocol, server or network errors."""
    explainer = explainer or DecisionExplainer(agent)
    t0 = clock()
    rec: dict = {"type": "hand", "hand": hand_no, "time_utc": _utc_now(), "status": "ok", "seed": seed}
    decisions: List[dict] = []
    rec["decisions"] = decisions
    counters0 = (client.n_requests, client.n_retries, client.n_token_changes)
    responses: List[dict] = []
    if trace:
        rec["responses"] = responses

    def request(send, *args) -> dict:
        r = send(*args)
        if trace:
            entry = {k: v for k, v in r.items() if k != "token"}
            entry.update(_token=client.last_token_status, _ms=round(client.last_latency * 1000))
            responses.append(entry)
        return r

    def close() -> None:
        rec["wall_s"] = round(clock() - t0, 3)
        rec["n_requests"] = client.n_requests - counters0[0]
        rec["n_retries"] = client.n_retries - counters0[1]
        rec["token_changes"] = client.n_token_changes - counters0[2]

    if seed is not None:
        agent.reset(seed)
    try:
        resp = request(client.new_hand)
        client_pos = _client_pos(resp)
        hero = hero_seat(client_pos)
        hole = parse_cards(resp.get("hole_cards"))
        if len(hole) != 2:
            raise ProtocolError(f"hole_cards {resp.get('hole_cards')!r}")
        rec.update(client_pos=client_pos, position=POSITION_NAMES[client_pos], hero_cards=cards_str(hole))
        prev: List[AppliedMove] = []
        ours: Optional[AppliedMove] = None
        while resp.get("winnings") is None:
            _same_hand(resp, client_pos, hole)
            action = resp.get("action")
            board = parse_cards(resp.get("board"))
            rp = replay(action, make_deck(hero, hole, board))
            check_continuation(prev, ours, rp.moves, hero, action)
            state = rp.state
            if state.is_terminal:
                raise ProtocolError(f"no winnings, but {action!r} is a finished hand")
            if state.current_player != hero:
                raise ProtocolError(f"not our turn after {action!r}")
            if state.board != board:
                raise ProtocolError(f"board {cards_str(board)} does not fit the street of {action!r}")
            obs = state.observe(hero)
            explainer.before_act()
            t = clock()
            act = agent.act(obs)
            think = clock() - t
            incr = format_incr(act, obs)
            dec = {
                "street": STREET_NAMES[obs.street],
                "before": action,
                "pot": obs.pot,
                "to_call": obs.to_call,
                "action": str(act),
                "incr": incr,
                "think_s": round(think, 4),
            }
            dec.update(explainer.explain(obs, act, hero))
            decisions.append(dec)
            prev, ours = rp.moves, (int(obs.street), hero, parse_incr(incr))
            resp = request(client.act, incr)
        _same_hand(resp, client_pos, hole)
        _finish(rec, resp, hero, hole, prev, ours, agent)
    except (ProtocolError, SlumbotError, TransportError, ValueError) as exc:
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, SlumbotError) and exc.response is not None:
            rec["error_response"] = {k: v for k, v in exc.response.items() if k != "token"}
        close()
        raise HandAborted(rec["error"], rec) from exc
    close()
    return rec


def _same_hand(resp: dict, client_pos: int, hole: List[int]) -> None:
    """client_pos and our hole cards must not change within a hand (checked when present)."""
    if "client_pos" in resp and _client_pos(resp) != client_pos:
        raise ProtocolError(f"client_pos changed within the hand ({client_pos} -> {resp.get('client_pos')})")
    if resp.get("hole_cards") is not None and parse_cards(resp.get("hole_cards")) != hole:
        raise ProtocolError(f"hole cards changed within the hand ({cards_str(hole)} -> {resp.get('hole_cards')})")


def _finish(rec: dict, resp: dict, hero: int, hole: List[int], prev: List[AppliedMove], ours: Optional[AppliedMove], agent: Agent) -> None:
    """Final response: log it, replay it with every card we know, cross-check the chips.

    Slumbot's ``winnings`` are what the match counts.  If our replay of the final action string
    fails, the hand keeps its winnings (the chips did move) and is flagged ``replay-failed``
    rather than dropped, so a failure cannot bias the bb/100."""
    winnings = resp.get("winnings")
    if isinstance(winnings, bool) or not isinstance(winnings, (int, float)) or int(winnings) != winnings:
        raise ProtocolError(f"winnings must be a whole number of chips, got {winnings!r}")
    winnings = int(winnings)
    action = resp.get("action")
    rec.update(action=action, winnings=winnings, final={k: v for k, v in resp.items() if k != "token"})
    try:
        _cross_check(rec, resp, hero, hole, prev, ours, agent, winnings, action)
    except ProtocolError as exc:
        rec["engine_net"] = None
        rec["check"] = "replay-failed"
        rec["check_basis"] = str(exc)


def _cross_check(rec: dict, resp: dict, hero: int, hole: List[int], prev: List[AppliedMove], ours: Optional[AppliedMove],
                 agent: Agent, winnings: int, action: str) -> None:
    board = parse_cards(resp.get("board"))
    bot_hole, bot_key = bot_cards_of(resp)
    rec.update(board=cards_str(board), bot_cards=cards_str(bot_hole) if bot_hole else None)
    notes = []
    try:
        deck = make_deck(hero, hole, board, bot_hole)
    except ProtocolError as exc:  # e.g. Slumbot's cards collide with ours: keep its number, flag it
        notes.append(f"bot cards unusable: {exc}")
        bot_hole = None
        deck = make_deck(hero, hole, board)
    rp = replay(action, deck)
    check_continuation(prev, ours, rp.moves, hero, action)
    state = rp.state
    if not state.is_terminal:
        raise ProtocolError(f"winnings given, but {action!r} is not a finished hand")
    if rp.ended_by_fold:
        verifiable, why = True, "fold"
    elif bot_hole is None:
        verifiable, why = False, "showdown, bot cards not shown"
    elif len(board) < 5:
        verifiable, why = False, f"showdown, only {len(board)} board cards shown"
    else:
        verifiable, why = True, "showdown"
    record = state.record()
    engine_net = record.net[hero]
    rec["engine_net"] = engine_net if verifiable else None
    rec["check"] = ("match" if engine_net == winnings else "MISMATCH") if verifiable else "unverified"
    rec["check_basis"] = why
    if bot_key and bot_key != BOT_CARD_KEYS[0]:
        notes.append(f"bot cards under key {bot_key!r}")
    if notes:
        rec["notes"] = notes
    if verifiable:
        agent.end_hand(record, hero)  # only records whose cards and chips we could confirm
