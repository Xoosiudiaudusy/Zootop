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

import itertools
import json
import math
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
from negpluribus.evaluator import evaluate  # noqa: E402
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


# --- hand names ------------------------------------------------------------------------------
# evaluate() packs the category and five ranks (0 = deuce .. 12 = ace) in base 15, see evaluator.py
_B = 15
RANK_CH = "23456789TJQKA"
_GEN_PL = ["двоек", "троек", "четвёрок", "пятёрок", "шестёрок", "семёрок", "восьмёрок", "девяток",
           "десяток", "вальтов", "дам", "королей", "тузов"]
_NOM_PL = ["двойки", "тройки", "четвёрки", "пятёрки", "шестёрки", "семёрки", "восьмёрки", "девятки",
           "десятки", "вальты", "дамы", "короли", "тузы"]
_GEN_SG = ["двойки", "тройки", "четвёрки", "пятёрки", "шестёрки", "семёрки", "восьмёрки", "девятки",
           "десятки", "валета", "дамы", "короля", "туза"]
_NOM_SG = ["двойка", "тройка", "четвёрка", "пятёрка", "шестёрка", "семёрка", "восьмёрка", "девятка",
           "десятка", "валет", "дама", "король", "туз"]
_EN_SG = ["Deuce", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King", "Ace"]
_EN_PL = ["Deuces", "Threes", "Fours", "Fives", "Sixes", "Sevens", "Eights", "Nines", "Tens", "Jacks", "Queens",
          "Kings", "Aces"]


def _decode(value: int) -> tuple[int, list[int]]:
    ranks = []
    for _ in range(5):
        ranks.insert(0, value % _B)
        value //= _B
    return value, ranks


def hand_name_ru(value: int) -> str:
    cat, r = _decode(value)
    if cat == 0:
        return f"Старшая карта — {_NOM_SG[r[0]]}"
    if cat == 1:
        return f"Пара {_GEN_PL[r[0]]}"
    if cat == 2:
        return f"Две пары: {_NOM_PL[r[0]]} и {_NOM_PL[r[1]]}"
    if cat == 3:
        return f"Тройка {_GEN_PL[r[0]]}"
    if cat == 4:
        return f"Стрит до {_GEN_SG[r[0]]}"
    if cat == 5:
        return f"Флеш до {_GEN_SG[r[0]]}"
    if cat == 6:
        return f"Фулл-хаус: {_NOM_PL[r[0]]} и {_NOM_PL[r[1]]}"
    if cat == 7:
        return f"Каре {_GEN_PL[r[0]]}"
    return "Роял-флеш" if r[0] == 12 else f"Стрит-флеш до {_GEN_SG[r[0]]}"


def hand_name_en(value: int) -> str:
    """PokerStars wording for the hand-history export."""
    cat, r = _decode(value)
    return [
        lambda: f"high card {_EN_SG[r[0]]}",
        lambda: f"a pair of {_EN_PL[r[0]]}",
        lambda: f"two pair, {_EN_PL[r[0]]} and {_EN_PL[r[1]]}",
        lambda: f"three of a kind, {_EN_PL[r[0]]}",
        lambda: f"a straight, {_EN_SG[max(r[0] - 4, -1)] if r[0] > 3 else 'Ace'} to {_EN_SG[r[0]]}",
        lambda: f"a flush, {_EN_SG[r[0]]} high",
        lambda: f"a full house, {_EN_PL[r[0]]} full of {_EN_PL[r[1]]}",
        lambda: f"four of a kind, {_EN_PL[r[0]]}",
        lambda: "a Royal Flush" if r[0] == 12 else f"a straight flush, {_EN_SG[r[0] - 4] if r[0] > 3 else 'Ace'} to {_EN_SG[r[0]]}",
    ][cat]()


def hand_info(hole: list[int], board: list[int]) -> dict | None:
    """The best five cards and their Russian name; before the flop only a pocket pair has a name."""
    cards = list(hole) + list(board)
    if len(cards) < 5:
        if len(hole) == 2 and hole[0] // 4 == hole[1] // 4:
            return {"name": f"Пара {_GEN_PL[hole[0] // 4]}", "best": list(hole), "cat": 1}
        return None
    best = max(itertools.combinations(cards, 5), key=evaluate)
    value = evaluate(best)
    return {"name": hand_name_ru(value), "best": list(best), "cat": value // _B ** 5, "value": value}


def card_str(c: int) -> str:
    return RANK_CH[c // 4] + "cdhs"[c % 4]


_BOARD_N = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


class Game:
    """One table: hero seat 0, bot seat 1, button alternates.

    Stacks carry over from hand to hand until someone is broke (``carry``); then the table is
    re-seated with fresh stacks (``rebuy``). With ``carry`` off every hand starts at the bot's
    training depth, as in the duels. The blueprint's infoset key has no stack depth in it, so at
    other depths the bot plays its training-depth strategy clamped to the chips it has."""

    def __init__(self, bot: Bot, carry: bool = True):
        self.bot = bot
        self.grid = bot.grid
        self.rng = random.Random(random.randrange(1 << 30))
        self.start_stack = bot.cfg["stack_bb"] * BB
        self.carry = carry
        self.stacks = [self.start_stack] * 2
        self.button = self.rng.randrange(2)
        self.state: HandState | None = None
        self.hand_log: list[dict] = []
        self.frames: list[dict] = []
        self.record = None
        self.hand_started = ""
        self.reset_stats()

    # ---------------------------------------------------------------- session
    def reset_stats(self) -> None:
        """Zero the session counters (hands, net, history, matches) and re-seat both players."""
        self.hand_no = 0
        self.net_chips = 0
        self.nets: list[int] = []            # hero's net per hand, chips
        self.history: list[dict] = []        # finished hands for the UI (last 30)
        self.records: list[tuple] = []       # (hand_no, time, HandRecord) for the export
        self.session_id = int(time.time())    # hand numbers stay unique across sessions (trackers dedupe)
        self.matches = {"hero": 0, "bot": 0}  # who broke whom (carry mode)
        self.stacks = [self.start_stack] * 2
        self.fresh_next = False
        self.state = None
        self.record = None
        self.hand_log = []
        self.frames = []
        self.button = self.rng.randrange(2)

    def rebuy(self) -> None:
        self.stacks = [self.start_stack] * 2

    def set_carry(self, carry: bool) -> None:
        if carry != self.carry:
            self.carry = carry
            self.fresh_next = True  # either way the next hand starts at the training depth

    def busted(self) -> str | None:
        if not self.carry or self.fresh_next:
            return None
        if self.stacks[0] <= 0:
            return "hero"
        if self.stacks[1] <= 0:
            return "bot"
        return None

    # ---------------------------------------------------------------- hand flow
    def new_hand(self) -> None:
        if self.busted():
            raise ValueError("у одного из игроков кончились фишки: «Сесть заново»")
        if self.fresh_next or not self.carry:
            self.stacks = [self.start_stack] * 2
            self.fresh_next = False
        stacks = list(self.stacks)
        self.button = 1 - self.button
        self.hand_no += 1
        self.record = None
        self.hand_log = []
        self.hand_started = time.strftime("%Y/%m/%d %H:%M:%S", time.gmtime())
        self.bot.agent.reset(seed=self.rng.randrange(1 << 30))
        self.state = HandState(stacks, self.button, SB, BB, seed=self.rng.randrange(1 << 30),
                               max_street=Street.RIVER)
        p = self.state.players
        self.frames = [{"street": "preflop", "boardN": 0, "pot": self.state.pot,
                        "bets": [p[0].street_bet, p[1].street_bet], "stacks": [p[0].stack, p[1].stack],
                        "heroHand": self._hand_label(0)}]
        while not self.state.terminal and self.state.current_player == 1:  # bot is SB/BTN
            self._bot_step()

    def _event_rng(self):
        """The bot's own translation RNG for this hand (so the hero's review reads the history
        exactly as the bot did)."""
        agent = self.bot.agent
        if agent._new_hand or agent._nonce is None:  # the same guard act() uses
            agent._nonce = agent.rng.getrandbits(32)
            agent._new_hand = False
        return agent._event_rng

    def _policy(self, obs) -> dict:
        """The blueprint's mix at this decision, for the review after the hand."""
        legal = self.grid.abstract_actions(obs)
        key = infoset_key(obs, self.bot.agent.bucketer, self.grid, event_rng=self._event_rng())
        probs = self.bot.strategy.policy(key, legal)
        opts = []
        for i, name in enumerate(legal):
            act = self.grid.to_concrete(obs, name)
            if name == "f":
                kind, chips = "fold", 0
            elif name == "a":
                kind, chips = "allin", act.amount
            elif act.type == ActionType.RAISE:
                kind, chips = "raise", act.amount
            else:
                kind, chips = ("check", 0) if obs.to_call == 0 else ("call", obs.to_call)
            opts.append({"name": name, "kind": kind, "chips": chips,
                         "p": None if probs is None else round(probs[i], 4)})
        return {"opts": opts, "fallback": probs is None, "key": key}

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
        policy = self._policy(obs)
        n0 = len(st.events)
        st.apply(action)
        self._collect(n0)
        ev = st.events[n0]
        policy["chosen"] = self.grid.from_concrete(ev, ev.all_in, None)
        self.hand_log[n0]["policy"] = policy
        while not st.terminal and st.current_player == 1:
            self._bot_step()

    def _bot_step(self) -> None:
        st = self.state
        obs = st.observe(1)
        policy = self._policy(obs)
        n0 = len(st.events)
        action = self.bot.agent.act(obs)
        st.apply(action)
        self._collect(n0)
        ev = st.events[n0]
        policy["chosen"] = self.grid.from_concrete(ev, ev.all_in, None)  # the bot bets on its grid
        self.hand_log[n0]["policy"] = policy

    def _collect(self, since: int) -> None:
        """Log the new events and, for each, the public table right after it (a frame): the
        client replays the frames one by one instead of jumping to the final state."""
        st = self.state
        for ev in st.events[since:]:
            e = describe_event(ev, self.grid)
            self.hand_log.append(e)
            f = self.frames[-1]
            bets = [0, 0] if e["street"] != f["street"] else list(f["bets"])
            stacks = list(f["stacks"])
            if ev.action.type == ActionType.RAISE:
                bets[ev.seat] = ev.action.amount
            else:
                bets[ev.seat] += ev.paid
            stacks[ev.seat] -= ev.paid
            n = _BOARD_N[e["street"]]
            self.frames.append({"street": e["street"], "boardN": n,
                                "pot": ev.pot_before + ev.paid, "bets": bets, "stacks": stacks,
                                "heroHand": self._hand_label(0, n)})

    # ---------------------------------------------------------------- finish
    def _finish_hand(self) -> None:
        """Called once when the hand is over: book the result."""
        st = self.state
        self.record = st.record()
        self.bot.agent.end_hand(self.record, 1)
        net = self.record.net[0]
        self.net_chips += net
        self.nets.append(net)
        self.stacks = [p.stack for p in st.players]
        who = self.busted()
        if who:
            self.matches["bot" if who == "hero" else "hero"] += 1
        self.records.append((self.hand_no, self.hand_started, self.record))
        self.history.append({
            "hand": self.hand_no,
            "netHero": net,
            "showdown": bool(self.record.showdown_seats),
            "winners": list(self.record.winners),
            "board": list(self.record.board),
            "heroHole": list(self.record.hole_cards[0]),
            "botHole": list(self.record.hole_cards[1]),
            "heroPos": "BTN" if st.button == 0 else "BB",
            "log": list(self.hand_log),
            "heroHand": self._hand_name(0), "botHand": self._hand_name(1),
        })
        self.history = self.history[-30:]

    def _hand_label(self, seat: int, board_n: int = 0) -> str | None:
        """The hand's name with the first ``board_n`` board cards (frames replay street by street)."""
        info = hand_info(self.state.players[seat].hole, self.state.board[:board_n])
        return info["name"] if info else None

    def _hand_name(self, seat: int) -> dict | None:
        st = self.state
        info = hand_info(st.players[seat].hole, st.board)
        if info:
            info.pop("value", None)
        return info

    def session_stats(self) -> dict:
        n = len(self.nets)
        mean = self.net_chips / BB / n if n else 0.0
        ci = None
        if n >= 2:
            xs = [x / BB for x in self.nets]
            var = sum((x - mean) ** 2 for x in xs) / (n - 1)
            ci = round(1.96 * math.sqrt(var / n) * 100, 1)
        series, acc = [0.0], 0
        for x in self.nets:
            acc += x
            series.append(round(acc / BB, 2))
        if len(series) > 301:  # thin to ~300 points, keep the last one
            step = len(series) / 300
            series = [series[int(i * step)] for i in range(300)] + [series[-1]]
        return {"hands": n, "netChips": self.net_chips, "netBb": round(self.net_chips / BB, 2),
                "bb100": round(mean * 100, 2), "ci95": ci, "series": series, "matches": self.matches}

    # ---------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        st = self.state
        base = {"bot": self.bot_public(), "carry": self.carry, "startStack": self.start_stack,
                "session": self.session_stats(), "history": self.history, "spec": self.bot.spec.describe(),
                "busted": self.busted()}
        if st is None:
            return {"started": False, "stacks": list(self.stacks), **base}
        terminal = None
        if st.terminal:
            if self.record is None:
                self._finish_hand()
                base.update(session=self.session_stats(), history=self.history, busted=self.busted())
            rec = self.record
            invested = [p.invested for p in st.players]
            returned = [max(0, invested[i] - invested[1 - i]) for i in range(2)]
            terminal = {
                "showdown": bool(rec.showdown_seats),
                "winners": list(rec.winners),
                "netHero": rec.net[0],
                "netBot": rec.net[1],
                "showdownSeats": list(rec.showdown_seats),
                "potWon": sum(invested) - sum(returned),
                "heroHand": self._hand_name(0),
                "botHand": self._hand_name(1),
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
        # the blueprint's mixes are shown only after the hand: live they would leak the bot's hand
        log = self.hand_log if st.terminal else [{k: v for k, v in e.items() if k != "policy"}
                                                  for e in self.hand_log]
        return {
            "started": True,
            **base,
            "hand": self.hand_no,
            "buttonHero": st.button == 0,
            "heroPos": "BTN" if st.button == 0 else "BB",
            "botPos": "BB" if st.button == 0 else "BTN",
            "street": st.street.name.lower(),
            "board": list(st.board),
            "heroHole": list(p[0].hole),
            "botHole": list(p[1].hole) if st.terminal else None,
            "heroHand": self._hand_name(0),
            "stacks": [p[0].stack, p[1].stack],
            "startStacks": list(st.starting_stacks),
            "streetBets": [p[0].street_bet, p[1].street_bet],
            "pot": st.pot,
            "toCall": st.to_call_for(0) if (not st.terminal and st.current_player == 0) else 0,
            "turn": "hero" if (not st.terminal and st.current_player == 0) else ("bot" if not st.terminal else "none"),
            "raiseBounds": raise_bounds,
            "legal": legal,
            "log": log,
            "frames": self.frames,
            "terminal": terminal,
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

    # ---------------------------------------------------------------- export
    def export_pokerstars(self) -> str:
        """The session's hands as PokerStars hand histories (play-money chips), for trackers."""
        names = ["Hero", "NegPluribus"]
        out = []
        for no, started, rec in self.records:
            btn = rec.button
            sb_seat, bb_seat = btn, 1 - btn
            lines = [
                f"PokerStars Hand #{self.session_id * 10_000 + no}:  Hold'em No Limit ({SB}/{BB}) - {started} UTC",
                f"Table 'NegativePluribus' 2-max (Play Money) Seat #{btn + 1} is the button",
            ]
            for s in range(2):
                lines.append(f"Seat {s + 1}: {names[s]} ({rec.starting_stacks[s]} in chips)")
            lines.append(f"{names[sb_seat]}: posts small blind {min(SB, rec.starting_stacks[sb_seat])}")
            lines.append(f"{names[bb_seat]}: posts big blind {min(BB, rec.starting_stacks[bb_seat])}")
            lines.append("*** HOLE CARDS ***")
            lines.append(f"Dealt to Hero [{' '.join(card_str(c) for c in rec.hole_cards[0])}]")
            street = Street.PREFLOP
            street_bet = [min(SB, rec.starting_stacks[sb_seat]) if s == sb_seat else min(BB, rec.starting_stacks[bb_seat])
                          for s in range(2)]
            invested = list(street_bet)
            folded_on = None
            board = rec.board
            heads = {Street.FLOP: ("FLOP", 3), Street.TURN: ("TURN", 4), Street.RIVER: ("RIVER", 5)}

            def street_line(s):
                name, n = heads[s]
                if n == 3:
                    return f"*** FLOP *** [{' '.join(card_str(c) for c in board[:3])}]"
                return (f"*** {name} *** [{' '.join(card_str(c) for c in board[:n - 1])}] "
                        f"[{card_str(board[n - 1])}]")

            for ev in rec.events:
                while ev.street != street:
                    street = Street(int(street) + 1)
                    street_bet = [0, 0]
                    if street in heads and len(board) >= heads[street][1]:
                        lines.append(street_line(street))
                who = names[ev.seat]
                tail = " and is all-in" if ev.all_in else ""
                t = ev.action.type
                if t == ActionType.FOLD:
                    lines.append(f"{who}: folds")
                    folded_on = (ev.seat, street)
                elif t == ActionType.CALL:
                    lines.append(f"{who}: checks" if ev.paid == 0 else f"{who}: calls {ev.paid}{tail}")
                else:
                    cur = max(street_bet)
                    if cur == 0:
                        lines.append(f"{who}: bets {ev.action.amount}{tail}")
                    else:
                        lines.append(f"{who}: raises {ev.action.amount - cur} to {ev.action.amount}{tail}")
                street_bet[ev.seat] += ev.paid
                invested[ev.seat] += ev.paid
            returned = [max(0, invested[i] - invested[1 - i]) for i in range(2)]
            for s in range(2):
                if returned[s]:
                    lines.append(f"Uncalled bet ({returned[s]}) returned to {names[s]}")
            # all-in run-out: the remaining streets are dealt without actions
            while int(street) < int(Street.RIVER) and folded_on is None and len(board) == 5:
                street = Street(int(street) + 1)
                lines.append(street_line(street))
            final = [rec.starting_stacks[s] + rec.net[s] for s in range(2)]
            collected = [final[s] - (rec.starting_stacks[s] - invested[s]) - returned[s] for s in range(2)]
            values = {}
            if rec.showdown_seats:
                lines.append("*** SHOW DOWN ***")
                for s in rec.showdown_seats:
                    values[s] = evaluate(rec.hole_cards[s] + board)
                    lines.append(f"{names[s]}: shows [{' '.join(card_str(c) for c in rec.hole_cards[s])}] "
                                 f"({hand_name_en(values[s])})")
            for s in range(2):
                if collected[s] > 0:
                    lines.append(f"{names[s]} collected {collected[s]} from pot")
            lines.append("*** SUMMARY ***")
            lines.append(f"Total pot {sum(invested) - sum(returned)} | Rake 0")
            if board:
                lines.append(f"Board [{' '.join(card_str(c) for c in board)}]")
            for s in range(2):
                role = " (button) (small blind)" if s == sb_seat else " (big blind)"
                if folded_on and folded_on[0] == s:
                    where = {Street.PREFLOP: "before Flop", Street.FLOP: "on the Flop",
                             Street.TURN: "on the Turn", Street.RIVER: "on the River"}[folded_on[1]]
                    res = f"folded {where}"
                elif s in values:
                    cards = ' '.join(card_str(c) for c in rec.hole_cards[s])
                    res = (f"showed [{cards}] and won ({collected[s]}) with {hand_name_en(values[s])}"
                           if collected[s] > 0 else f"showed [{cards}] and lost with {hand_name_en(values[s])}")
                else:
                    res = f"collected ({collected[s]})"
                lines.append(f"Seat {s + 1}: {names[s]}{role} {res}")
            out.append("\n".join(lines))
        return "\n\n\n".join(out) + ("\n" if out else "")


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
        elif path == "/api/export":
            with SESSION.lock:
                text = SESSION.game.export_pokerstars() if SESSION.game else ""
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition",
                             f'attachment; filename="negpluribus_{time.strftime("%Y%m%d_%H%M")}.txt"')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            self.log_code(200, path, self._t0)
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
                    carry = SESSION.game.carry if SESSION.game else True
                    SESSION.game = Game(bot, carry=carry)
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
                elif path == "/api/rebuy":  # re-seat both players with fresh stacks, deal
                    if game.state is not None and not game.state.terminal:
                        raise ValueError("раздача ещё не окончена")
                    game.rebuy()
                    game.new_hand()
                elif path == "/api/settings":
                    if "carry" in body:
                        game.set_carry(bool(body["carry"]))
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
