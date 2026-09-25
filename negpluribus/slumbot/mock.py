"""A local stand-in for the Slumbot server: the same JSON protocol, our engine and any of our
agents on the server side.  For tests and dry runs; it never talks to the internet.

Two ways to reach it:

* ``mock.transport`` - an injectable transport for ``SlumbotClient`` (no sockets);
* ``with mock.serve() as host:`` - a real ``http.server`` on 127.0.0.1 (random port), so the
  client's default urllib transport is exercised end to end.

The server-side string encoding is written independently of the client's parser
(``adapter.replay``): each engine action becomes ``k``/``c``/``f``/``bN`` and slashes are written
as the official sample client describes them.  Options reproduce the protocol variants the
sample parser accepts (no trailing '/' after a finished street, no slashes after an all-in call),
token rotation, and faults (5xx answers, a response lost after the move was applied).

Seen on the real server (5 hands, 2026-09-24) and copied here: the bot's hole cards come with
every finished hand, folds included, and the final response carries the session counters
``session_num_hands`` / ``session_total``.  Not known, so strict or configurable here: what a
``new_hand`` does while a hand is still open (``abandoned_hand``: refuse, or forfeit the open hand
as if we had folded), and Slumbot's ``baseline_winnings`` (not reproduced).
"""
from __future__ import annotations

import json
import random
import re
import threading
import urllib.parse
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Iterator, List, Optional, Tuple

from ..agents.base import Agent
from ..cards import Deck
from ..engine import CALL, FOLD, ActionType, HandState, Street, raise_to
from .client import API_PREFIX
from .protocol import BIG_BLIND, BUTTON_SEAT, NUM_STREETS, SMALL_BLIND, STACK_SIZE, cards_str

_INCR = re.compile(r"^(?:[kcf]|b[0-9]+)$")


@dataclass
class _Session:
    token: str
    next_client_pos: int
    state: Optional[HandState] = None
    action: str = ""
    client_pos: int = 0
    hands: int = 0  # hands dealt
    finished_hands: int = 0  # hands with a result (forfeits included): the session counters
    total: int = 0


@dataclass
class MockHand:
    """What the mock knows about a finished hand (for tests: the ground truth)."""

    token: str
    client_pos: int
    action: str
    client_net: int
    client_cards: List[str]
    bot_cards: List[str]
    board: List[str]
    showdown: bool
    forfeit: bool = False


class MockSlumbot:
    def __init__(
        self,
        bot: Agent,
        seed: int = 0,
        reveal_bot_cards: str = "always",  # "always" (as seen live) | "showdown" | "never"
        allin_slashes: bool = True,  # write "b20000c///" (True) or "b20000c" (False)
        street_slash: bool = True,  # write the '/' as soon as a street ends (False: only before the next move)
        alternate: bool = True,  # client_pos alternates between hands of a session
        rotate_token_every: int = 0,  # >0: new token every k hands (the old one stops working)
        first_client_pos: Optional[int] = None,  # client_pos of a session's first hand (None: random)
        abandoned_hand: str = "error",  # new_hand while a hand is open: "error" (400) or "forfeit" it
    ):
        if reveal_bot_cards not in ("showdown", "always", "never"):
            raise ValueError(reveal_bot_cards)
        if abandoned_hand not in ("error", "forfeit"):
            raise ValueError(abandoned_hand)
        self.abandoned_hand = abandoned_hand
        self.bot = bot
        self.rng = random.Random(seed)
        self.reveal = reveal_bot_cards
        self.allin_slashes = allin_slashes
        self.street_slash = street_slash
        self.alternate = alternate
        self.rotate_token_every = rotate_token_every
        self.first_client_pos = first_client_pos
        self.sessions: Dict[str, _Session] = {}
        self.finished: List[MockHand] = []
        self.requests: List[Tuple[str, dict]] = []
        self.fail_next: List[int] = []  # statuses to answer the next requests with (not processed)
        self.lose_next_act = 0  # process the next N act requests, then answer 504 as if the reply got lost
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- dispatch
    def handle(self, path: str, payload: dict) -> Tuple[int, dict]:
        with self._lock:
            self.requests.append((path, dict(payload)))
            if self.fail_next:
                status = self.fail_next.pop(0)
                return status, {"error_msg": f"injected failure {status}"}
            if path == f"{API_PREFIX}/new_hand":
                return self._new_hand(payload)
            if path == f"{API_PREFIX}/act":
                status, data = self._act(payload)
                if status == 200 and self.lose_next_act > 0:
                    self.lose_next_act -= 1
                    return 504, {"error_msg": "gateway timeout (injected after processing)"}
                return status, data
            return 404, {"error_msg": f"no such endpoint {path}"}

    def transport(self, url: str, payload: dict, timeout: float) -> Tuple[int, bytes]:
        """Injectable transport for ``SlumbotClient`` (JSON round trip, like the wire)."""
        path = urllib.parse.urlsplit(url).path
        status, data = self.handle(path, json.loads(json.dumps(payload)))
        return status, json.dumps(data).encode("utf-8")

    @contextmanager
    def serve(self, host: str = "127.0.0.1", port: int = 0) -> Iterator[str]:
        """Run the mock as an HTTP server in a background thread; yields its base URL."""
        mock = self

        class Handler(BaseHTTPRequestHandler):
            wbufsize = 64 * 1024  # headers + body leave in one send (no Nagle / delayed-ACK stall)

            def do_POST(self):  # noqa: N802 - http.server API
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                    if not isinstance(payload, dict):
                        raise ValueError("not an object")
                except ValueError:
                    status, data = 400, {"error_msg": "body is not a JSON object"}
                else:
                    status, data = mock.handle(urllib.parse.urlsplit(self.path).path, payload)
                body = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):  # keep test output quiet
                pass

        server = ThreadingHTTPServer((host, port), Handler)
        thread = threading.Thread(target=server.serve_forever, name="mock-slumbot", daemon=True)
        thread.start()
        try:
            yield f"http://{host}:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    # ------------------------------------------------------------------- endpoints
    def _new_hand(self, payload: dict) -> Tuple[int, dict]:
        token = payload.get("token")
        sess = self.sessions.get(token) if token else None
        if token and sess is None:
            return 400, {"error_msg": "unknown token"}
        if sess is None:
            first = self.rng.randrange(2) if self.first_client_pos is None else self.first_client_pos
            sess = _Session(token=uuid.UUID(int=self.rng.getrandbits(128)).hex, next_client_pos=first)
            self.sessions[sess.token] = sess
        if sess.state is not None and not sess.state.is_terminal:
            if self.abandoned_hand == "error":
                return 400, {"error_msg": "a hand is in progress"}
            self._forfeit(sess)
        if self.rotate_token_every and sess.hands and sess.hands % self.rotate_token_every == 0:
            del self.sessions[sess.token]
            sess.token = uuid.UUID(int=self.rng.getrandbits(128)).hex
            self.sessions[sess.token] = sess
        sess.client_pos = sess.next_client_pos
        if self.alternate:
            sess.next_client_pos = 1 - sess.client_pos
        sess.state = HandState([STACK_SIZE, STACK_SIZE], BUTTON_SEAT, SMALL_BLIND, BIG_BLIND,
                               deck=Deck(seed=self.rng.getrandbits(32)), max_street=Street.RIVER)
        sess.action = ""
        sess.hands += 1
        self._bot_moves(sess)
        return 200, self._response(sess, old_action="")

    def _act(self, payload: dict) -> Tuple[int, dict]:
        sess = self.sessions.get(payload.get("token") or "")
        if sess is None:
            return 400, {"error_msg": "unknown token"}
        st = sess.state
        if st is None or st.is_terminal:
            return 400, {"error_msg": "no hand in progress"}
        incr = payload.get("incr")
        if not isinstance(incr, str) or not _INCR.match(incr):
            return 400, {"error_msg": f"bad incr {incr!r}"}
        seat = 1 - sess.client_pos
        if st.current_player != seat:
            return 400, {"error_msg": "not your turn"}
        to_call = st.to_call_for(seat)
        if incr == "k" and to_call == 0:
            act = CALL
        elif incr in ("c", "f") and to_call > 0:
            act = CALL if incr == "c" else FOLD
        elif incr[0] == "b":
            can, lo, hi = st.raise_bounds(seat)
            if not (can and lo <= int(incr[1:]) <= hi):
                return 400, {"error_msg": f"illegal bet {incr} (legal b{lo}..b{hi})" if can else "no bet allowed"}
            act = raise_to(int(incr[1:]))
        else:
            return 400, {"error_msg": f"illegal action {incr}"}
        old = sess.action
        self._apply(sess, act)
        self._bot_moves(sess)
        return 200, self._response(sess, old_action=old)

    # --------------------------------------------------------------- game logic
    def _pad(self, sess: _Session, streets_closed: int) -> None:
        sess.action += "/" * max(0, streets_closed - sess.action.count("/"))

    def _apply(self, sess: _Session, act) -> None:
        st = sess.state
        street = int(st.street)
        to_call = st.to_call_for(st.current_player)
        if act.type == ActionType.FOLD:
            tok = "f"
        elif act.type == ActionType.CALL:
            tok = "k" if to_call == 0 else "c"
        else:
            tok = f"b{act.amount}"
        self._pad(sess, street)  # the '/' that closes earlier streets, if not written yet
        st.apply(act)
        sess.action += tok
        if st.is_terminal:
            if tok != "f" and street < NUM_STREETS - 1 and self.allin_slashes:
                self._pad(sess, NUM_STREETS - 1)  # all-in called before the river: close every street
        elif int(st.street) != street and self.street_slash:
            self._pad(sess, int(st.street))

    def _bot_moves(self, sess: _Session) -> None:
        st = sess.state
        bot = sess.client_pos  # engine seat of the bot (client_pos 0 = client is BB = seat 1)
        while not st.is_terminal and st.current_player == bot:
            self._apply(sess, self.bot.act(st.observe(bot)))
        if st.is_terminal:
            rec = st.record()
            client = 1 - sess.client_pos
            self._record(sess, rec.net[client], showdown=len(rec.showdown_seats) > 1)
            self.bot.end_hand(rec, bot)

    def _record(self, sess: _Session, client_net: int, showdown: bool, forfeit: bool = False) -> None:
        st = sess.state
        client, bot = 1 - sess.client_pos, sess.client_pos
        self.finished.append(MockHand(
            token=sess.token, client_pos=sess.client_pos, action=sess.action, client_net=client_net,
            client_cards=cards_str(st.players[client].hole), bot_cards=cards_str(st.players[bot].hole),
            board=cards_str(st.board), showdown=showdown, forfeit=forfeit,
        ))
        sess.finished_hands += 1
        sess.total += client_net

    def _forfeit(self, sess: _Session) -> None:
        """The open hand ends as if the client had folded: it loses what it has put in."""
        self._record(sess, -sess.state.players[1 - sess.client_pos].invested, showdown=False, forfeit=True)
        sess.state = None

    def _response(self, sess: _Session, old_action: str) -> dict:
        st = sess.state
        client, bot = 1 - sess.client_pos, sess.client_pos
        r = {
            "old_action": old_action,
            "action": sess.action,
            "client_pos": sess.client_pos,
            "hole_cards": cards_str(st.players[client].hole),
            "board": cards_str(st.board),
            "token": sess.token,
        }
        if st.is_terminal:
            rec = st.record()
            r["winnings"] = rec.net[client]
            showdown = len(rec.showdown_seats) > 1
            if self.reveal == "always" or (self.reveal == "showdown" and showdown):
                r["bot_hole_cards"] = cards_str(st.players[bot].hole)
            r["session_num_hands"] = sess.finished_hands
            r["session_total"] = sess.total
        return r
