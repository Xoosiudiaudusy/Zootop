# -*- coding: utf-8 -*-
r"""NegativePluribus web table: play heads-up against a trained blueprint in the browser.

    python webapp\server.py            # -> http://127.0.0.1:8777

Serves the static UI and a small JSON API on top of the project's own engine,
abstraction and blueprint code (the same classes compare_checkpoints.py uses):
HandState drives the hand, BlueprintAgent plays the bot seat, CoreBuckets +
the precomputed bucket tables make decisions instant.

Which bots the table offers is configuration, not code: webapp/bots.json (in git,
paths relative to the project root, e.g. data/blueprint_*.bin) plus an optional
webapp/bots.local.json (git-ignored: this machine's extra bots, absolute paths or
paths relative to the project root). A bot whose files are missing is listed as
unavailable instead of breaking the server. Blueprints and bucket tables are read
where they are; data/ may be a junction to another drive.
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

WEBAPP = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(WEBAPP)
sys.path.insert(0, PROJECT)

from negpluribus.abstraction import BetGrid, infoset_key, load_bucketer  # noqa: E402
from negpluribus.agents.blueprint import BlueprintAgent  # noqa: E402
from negpluribus.cfr.game import GameSpec  # noqa: E402
from negpluribus.engine import ActionType, HandState, Street  # noqa: E402
from negpluribus.fast import core as fast_core  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402
from negpluribus.fast.tables import CoreBuckets  # noqa: E402

HOST = "127.0.0.1"  # local only: the table is not reachable from other machines
PORT = int(os.environ.get("NP_WEB_PORT", "8777"))
SB, BB = 50, 100

# --- which bots the table offers: webapp/bots.json + webapp/bots.local.json ------
CONFIG = os.path.join(WEBAPP, "bots.json")              # in git: the project's own blueprints
LOCAL_CONFIG = os.path.join(WEBAPP, "bots.local.json")  # git-ignored: this machine's extras


def _path(p: str) -> str:
    """Absolute paths as they are; relative ones from the project root (so "data/..." follows
    the data/ junction wherever the data really lives)."""
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(PROJECT, p))


def _bot(item: dict, source: str) -> dict:
    bp = _path(item["blueprint"])
    bk = _path(item["bucketer"])
    td = _path(item.get("tables_dir") or "data/bucket_tables")
    missing = [os.path.basename(p) for p in (bp, bk) if not os.path.exists(p)]
    note = item.get("note", "")
    label = item.get("label", os.path.basename(bp))
    if missing:
        label += " — нет файлов"
        note = (note + " · " if note else "") + "не найдено: " + ", ".join(missing)
    return {
        "id": item.get("id") or os.path.splitext(os.path.basename(bp))[0],
        "label": label, "note": note,
        "stack_bb": int(item.get("stack_bb", 200)),
        "preflop": tuple(item.get("preflop", (0.5, 1.0, 3.0))),
        "postflop": tuple(item.get("postflop", (0.5, 1.0, 2.0, 4.0))),
        "max_raises": int(item.get("max_raises", 3)),
        "blueprint": bp, "bucketer": bk, "tables_dir": td,
        "available": not missing, "source": source,
    }


def load_bots() -> tuple[list[dict], str | None]:
    """Bots from bots.json, then bots.local.json (a local entry with the same id replaces the
    shared one). Returns (bots, default id): the configured default if its files exist, else the
    first available bot."""
    bots: dict[str, dict] = {}
    default = None
    for path, source in ((CONFIG, "bots.json"), (LOCAL_CONFIG, "bots.local.json")):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8-sig") as f:  # sig: Notepad/PowerShell write a BOM
                cfg = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"{source}: {e}", flush=True)
            continue
        items = cfg.get("bots", []) if isinstance(cfg, dict) else cfg
        if isinstance(cfg, dict) and cfg.get("default"):
            default = cfg["default"]
        for item in items:
            try:
                b = _bot(item, source)
                bots[b["id"]] = b
            except Exception as e:  # noqa: BLE001
                print(f"{source} item: {e}", flush=True)
    out = list(bots.values())
    avail = [b["id"] for b in out if b["available"]]
    if default not in avail:
        default = avail[0] if avail else None
    return out, default


BOTS, DEFAULT_BOT = load_bots()


class Bot:
    """A loaded blueprint plus everything needed to act at the table."""

    def __init__(self, cfg):
        self.cfg = cfg
        os.environ["NEGPLURIBUS_BUCKET_TABLES"] = cfg["tables_dir"]  # read at tabulated() time
        bucketer = load_bucketer(cfg["bucketer"])
        bucketer = CoreBuckets(bucketer, check=10)  # the duels' own path: C++ + table, checked
        self.strategy = load_blueprint(cfg["blueprint"])
        self.spec = GameSpec(
            n_players=2, stack_bb=cfg["stack_bb"], sb=SB, bb=BB, max_street=Street.RIVER,
            preflop_fracs=tuple(cfg["preflop"]), postflop_fracs=tuple(cfg["postflop"]),
            max_raises_per_street=cfg["max_raises"], n_buckets=bucketer.n_buckets,
            bucket_kind=getattr(bucketer, "kind", "potential"),
        )
        self.grid: BetGrid = self.spec.grid
        self.agent = BlueprintAgent(self.strategy, bucketer, self.grid, seed=random.randrange(1 << 30), name="bot")
        self.n_infosets = len(self.strategy)


def describe_event(ev, grid: BetGrid) -> dict:
    t = ev.action.type
    kind = "fold"
    if t == ActionType.CALL:
        kind = "check" if ev.to_call == 0 else "call"
    elif t == ActionType.RAISE:
        if ev.all_in:
            kind = "allin"
        elif ev.street == Street.PREFLOP or ev.raises_this_street > 0:
            kind = "raise"
        else:
            kind = "bet"
    frac = BetGrid.observed_frac(ev) if t == ActionType.RAISE else None
    return {
        "street": ev.street.name.lower(), "seat": ev.seat, "kind": kind,
        "to": ev.action.amount if t == ActionType.RAISE else 0, "paid": ev.paid,
        "frac": round(frac, 3) if frac is not None else None,
        "toCall": ev.to_call, "potBefore": ev.pot_before, "allIn": ev.all_in,
    }


class Game:
    """One table: hero seat 0, bot seat 1, button alternates, stacks reset each hand."""

    def __init__(self, bot: Bot):
        self.bot = bot
        self.grid = bot.grid
        self.rng = random.Random(random.randrange(1 << 30))
        self.hand_no = 0
        self.button = self.rng.randrange(2)
        self.net_chips = 0  # hero's cumulative
        self.state: HandState | None = None
        self.hand_log: list[dict] = []
        self.bot_last: dict | None = None
        self.record = None
        self.history: list[dict] = []  # finished hands of this session

    # ---------------------------------------------------------------- session
    def reset_stats(self) -> None:
        """Zero the session counters (hands, net, history); keep the same bot."""
        self.hand_no = 0
        self.net_chips = 0
        self.state = None
        self.record = None
        self.hand_log = []
        self.bot_last = None
        self.history = []
        self.button = self.rng.randrange(2)

    # ---------------------------------------------------------------- hand flow
    def new_hand(self) -> None:
        stacks = [self.bot.cfg["stack_bb"] * BB] * 2
        self.button = 1 - self.button
        self.hand_no += 1
        self.record = None
        self.bot_last = None
        self.hand_log = []
        self.bot.agent.reset(seed=self.rng.randrange(1 << 30))
        self.state = HandState(stacks, self.button, SB, BB, seed=self.rng.randrange(1 << 30),
                               max_street=Street.RIVER)
        if self.state.current_player == 1:  # bot acts first (it is SB/BTN)
            self._bot_step()

    def hero_act(self, name: str, amount: int | None = None) -> None:
        st = self.state
        assert st is not None and not st.terminal and st.current_player == 0
        obs = st.observe(0)
        legal = self.grid.abstract_actions(obs)
        if name == "raise":  # free sizing: the engine takes any raise-to, the bot's
            # pseudo-harmonic translation maps it onto its grid (abstraction/actions.py)
            if not obs.can_raise:
                raise ValueError("рейз невозможен")
            action = obs.clamp_raise(int(amount if amount is not None else obs.min_raise_to))
        else:
            if name not in legal:
                raise ValueError(f"illegal action {name!r}; legal: {legal}")
            action = self.grid.to_concrete(obs, name)
        n0 = len(st.events)
        st.apply(action)
        self._collect(n0)
        while not st.terminal and st.current_player == 1:
            self._bot_step()

    def _bot_step(self) -> None:
        st = self.state
        obs = st.observe(1)
        legal = self.grid.abstract_actions(obs)
        agent = self.bot.agent
        if agent._new_hand or agent._nonce is None:  # the same guard act() uses: the key
            agent._nonce = agent.rng.getrandbits(32)  # translation needs the hand's nonce
            agent._new_hand = False
        key = infoset_key(obs, agent.bucketer, self.grid, event_rng=agent._event_rng)
        probs = self.strategy_policy(key, legal)
        n0 = len(st.events)
        action = self.bot.agent.act(obs)
        st.apply(action)
        self._collect(n0)
        self.bot_last = {
            "key": key, "street": obs.street.name.lower(),
            "legal": legal, "probs": None if probs is None else [round(p, 4) for p in probs],
            "fallback": probs is None,
        }

    def strategy_policy(self, key, legal):
        return self.bot.strategy.policy(key, legal)

    def _collect(self, since: int) -> None:
        st = self.state
        for ev in st.events[since:]:
            self.hand_log.append(describe_event(ev, self.grid))

    # ---------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        st = self.state
        if st is None:
            return {"started": False, "bot": self.bot_public()}
        terminal = None
        if st.terminal:
            if self.record is None:
                self.record = st.record()
                self.bot.agent.end_hand(self.record, 1)
                self.net_chips += self.record.net[0]
                self.history.append({
                    "hand": self.hand_no,
                    "netHero": self.record.net[0],
                    "showdown": bool(self.record.showdown_seats),
                    "winners": list(self.record.winners),
                    "board": list(self.record.board),
                    "heroHole": list(self.record.hole_cards[0]),
                    "botHole": list(self.record.hole_cards[1]),
                    "log": list(self.hand_log),
                })
                self.history = self.history[-30:]
            rec = self.record
            terminal = {
                "showdown": bool(rec.showdown_seats),
                "winners": list(rec.winners),
                "netHero": rec.net[0],
                "netBot": rec.net[1],
                "showdownSeats": list(rec.showdown_seats),
            }
        obs = None
        legal = []
        raise_bounds = None
        if not st.terminal and st.current_player == 0:
            obs = st.observe(0)
            if obs.can_raise:
                raise_bounds = {"min": obs.min_raise_to, "max": obs.max_raise_to}
            for name in self.grid.abstract_actions(obs):
                act = self.grid.to_concrete(obs, name)
                if act.type == ActionType.RAISE:
                    chips, kind = act.amount, "raise"
                else:
                    chips, kind = (0, "check") if obs.to_call == 0 else (obs.to_call, "call")
                if name == "f":
                    kind = "fold"
                elif name == "a":
                    kind = "allin"
                legal.append({"name": name, "kind": kind, "chips": chips})
        p = st.players
        cfg = self.bot.cfg
        return {
            "started": True,
            "bot": self.bot_public(),
            "hand": self.hand_no,
            "buttonHero": st.button == 0,
            "heroPos": "BTN/SB" if st.button == 0 else "BB",
            "botPos": "BB" if st.button == 0 else "BTN/SB",
            "street": st.street.name.lower(),
            "board": list(st.board),
            "heroHole": list(p[0].hole),
            "botHole": list(p[1].hole) if st.terminal else None,
            "stacks": [p[0].stack, p[1].stack],
            "streetBets": [p[0].street_bet, p[1].street_bet],
            "invested": [p[0].invested, p[1].invested],
            "pot": st.pot,
            "toCall": st.to_call_for(0) if (not st.terminal and st.current_player == 0) else 0,
            "turn": "hero" if (not st.terminal and st.current_player == 0) else ("bot" if not st.terminal else "none"),
            "canFold": bool(obs and obs.can_fold),
            "raiseBounds": raise_bounds,
            "legal": legal,
            "log": self.hand_log,
            "terminal": terminal,
            "session": {
                "hands": self.hand_no, "netChips": self.net_chips,
                "netBb": round(self.net_chips / BB, 2),
                "bb100": round(self.net_chips / BB / max(1, self.hand_no) * 100, 2),
            },
            "history": self.history,
            "botLast": self.bot_last,
            "spec": self.bot.spec.describe(),
        }

    def bot_public(self) -> dict:
        a = self.bot.agent
        fb = {"decisions": a.n_decisions, "fallbacks": a.n_fallback}
        return {"id": self.bot.cfg["id"], "label": self.bot.cfg["label"],
                "note": self.bot.cfg["note"], "infosets": self.n_infosets_safe(), "offmap": fb}

    def n_infosets_safe(self) -> int:
        try:
            return len(self.bot.strategy)
        except Exception:
            return 0


# ---------------------------------------------------------------------- server
class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.game: Game | None = None
        self.bots: dict[str, Bot] = {}
        self.load_lock = threading.Lock()
        self.loading: str | None = None

    def get_bot(self, bot_id: str) -> Bot:
        with self.load_lock:
            if bot_id not in self.bots:
                cfg = next((b for b in BOTS if b["id"] == bot_id), None)
                if cfg is None:
                    raise ValueError(f"нет такого бота: {bot_id!r}")
                if not cfg["available"]:
                    raise ValueError(f"у бота {cfg['id']} нет файлов: {cfg['note']}")
                self.loading = bot_id
                try:
                    self.bots[bot_id] = Bot(cfg)
                finally:
                    self.loading = None
            return self.bots[bot_id]


SESSION = Session()


class QuietServer(ThreadingHTTPServer):
    """Keeps the console clean when a browser tab is closed mid-request.

    ``allow_reuse_address = False``: on Windows the default (SO_REUSEADDR) lets a second
    copy of the server silently bind the same port, and browser requests then land on a
    random one of the two — a killed copy leaves tabs with endlessly pending fetches."""

    allow_reuse_address = 0

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]}:{self.client_address[1]} "
              f"{(fmt % args).strip()}", flush=True)

    def log_code(self, code, path, t0):
        print(f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]}:{self.client_address[1]} "
              f"{path} -> {code} in {time.perf_counter() - t0:.3f}s", flush=True)

    def end_headers(self):
        # One request per connection: browser extensions and local proxies sometimes break
        # keep-alive sockets silently, which leaves fetch() pending forever. Closing after
        # every response makes each request independent of that.
        self.send_header("Connection", "close")
        self.close_connection = True
        super().end_headers()

    # ------------------------------------------------------------------ helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        if hasattr(self, "_t0"):
            self.log_code(code, urlparse(self.path).path, self._t0)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self._json({"error": "not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    # ------------------------------------------------------------------- routes
    def do_GET(self):
        self._t0 = time.perf_counter()
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._file(os.path.join(WEBAPP, "static", "index.html"), "text/html; charset=utf-8")
        elif path == "/app.css":
            self._file(os.path.join(WEBAPP, "static", "app.css"), "text/css; charset=utf-8")
        elif path == "/app.js":
            self._file(os.path.join(WEBAPP, "static", "app.js"), "text/javascript; charset=utf-8")
        elif path == "/api/bots":
            with SESSION.lock:
                active = SESSION.game.bot.cfg["id"] if SESSION.game else None
            self._json({
                "bots": [{
                    "id": b["id"], "label": b["label"], "note": b["note"],
                    "stackBb": b["stack_bb"], "spec": (f"2 игрока, {b['stack_bb']}bb, сетка префлоп "
                            f"{b['preflop']} / постфлоп {b['postflop']} + олл-ин, до {b['max_raises']} рейзов на улицу"),
                    "loaded": b["id"] in SESSION.bots, "active": b["id"] == active,
                    "available": b["available"],
                } for b in BOTS],
                "default": DEFAULT_BOT,
                "loading": SESSION.loading,
            })
        elif path == "/api/state":
            with SESSION.lock:
                snap = SESSION.game.snapshot() if SESSION.game else {"started": False}
            self._json(snap)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        self._t0 = time.perf_counter()
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/select":
                bot = SESSION.get_bot(body.get("id") or DEFAULT_BOT)
                with SESSION.lock:
                    SESSION.game = Game(bot)
                with SESSION.lock:
                    snap = SESSION.game.snapshot()
                self._json(snap)
                return
            with SESSION.lock:
                game = SESSION.game
                if game is None:
                    raise ValueError("бот не выбран")
                if path == "/api/new":
                    if game.state is not None and not game.state.terminal and game.hand_no > 0:
                        raise ValueError("раздача ещё не окончена")
                    game.new_hand()
                elif path == "/api/action":
                    if game.state is None or game.state.terminal or game.state.current_player != 0:
                        raise ValueError("сейчас не ваш ход")
                    game.hero_act(str(body.get("name", "")),
                                  int(body["amount"]) if body.get("amount") is not None else None)
                elif path == "/api/reset":
                    game.reset_stats()
                else:
                    self._json({"error": "not found"}, 404)
                    return
                snap = game.snapshot()
            self._json(snap)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": str(e)}, 400)


def main() -> None:
    print("NegativePluribus web table")
    print("проект:", PROJECT)
    pid_file = os.path.join(WEBAPP, "server.pid")  # stop.bat reads it
    with open(pid_file, "w", encoding="ascii") as f:
        f.write(str(os.getpid()))

    def _drop_pid():
        try:
            with open(pid_file, encoding="ascii") as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(pid_file)
        except OSError:
            pass

    import atexit
    atexit.register(_drop_pid)
    for b in BOTS:
        print(f"  {'+' if b['available'] else '-'} {b['id']:<14} {b['label']}  [{b['source']}]")
    if DEFAULT_BOT is None:
        print("нет ни одного бота с файлами: проверь webapp/bots.json и webapp/bots.local.json")
    srv = QuietServer((HOST, PORT), Handler)

    if DEFAULT_BOT is not None:  # preload the default bot so the first hand starts fast
        threading.Thread(target=lambda: SESSION.get_bot(DEFAULT_BOT), daemon=True).start()
    url = f"http://{HOST}:{PORT}"
    print(f"открой в браузере: {url}  (бот по умолчанию грузится пару секунд)")
    if not os.environ.get("NP_WEB_NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nстоп.")


if __name__ == "__main__":
    main()
