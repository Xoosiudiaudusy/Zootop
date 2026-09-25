"""A Slumbot match: N hands, strictly one at a time, one JSONL line per hand, running bb/100.

The log is append-only and the match resumes from it: hands already logged count in the
statistics, new hands continue the numbering, and the agent's per-hand seed is derived from the
hand number, so a resumed run is the same run continued.  Each invocation first writes a
``{"type": "run", ...}`` line (what played: blueprint, buckets, grid, seed, host); every hand is a
``{"type": "hand", ...}`` line (format in docs/slumbot.md).

Accounting: the chips are Slumbot's ``winnings``.  Two cross-checks run on every hand:

* our engine's count for the replayed hand (``engine_net``, ``check``) whenever it can be
  computed (a fold, or a showdown where Slumbot shows its cards; live it showed them every hand);
* Slumbot's own session counters (``session_num_hands`` / ``session_total`` in the final
  response): each hand must add one hand and exactly its winnings (``session_check``).  A jump of
  more than one hand means the server counted hands we have no result for (a hand we had to
  abandon); their chips are logged as ``session_gap`` and summed in the summary, so they stay
  visible even though they cannot enter the bb/100.

A hand the client could not finish (network or protocol failure) has no winnings; it is logged
with ``status: "error"``, counted separately and never enters the bb/100.  After ``max_errors``
failed hands in a row the match stops.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ..agents.base import Agent
from .adapter import DecisionExplainer, HandAborted, play_hand
from .client import SlumbotClient
from .protocol import BIG_BLIND


def bb100_ci(winnings: Iterable[int], bb: int = BIG_BLIND) -> Tuple[float, float]:
    """(bb/100, half-width of the 95% CI) from per-hand chip results (normal approximation,
    sample standard deviation).  The half-width is inf below two hands."""
    xs = [w / bb for w in winnings]
    n = len(xs)
    if n == 0:
        return 0.0, math.inf
    mean = sum(xs) / n
    if n < 2:
        return mean * 100.0, math.inf
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean * 100.0, 1.96 * math.sqrt(var / n) * 100.0


@dataclass
class Tally:
    """Running totals over the hand records of a log."""

    winnings: List[int] = field(default_factory=list)  # chips per hand with winnings, log order
    positions: List[str] = field(default_factory=list)
    n_errors: int = 0
    checks: Counter = field(default_factory=Counter)
    n_decisions: int = 0
    n_blueprint_decisions: int = 0
    n_off_map: int = 0
    n_bot_bets: int = 0
    n_bot_bets_off_grid: int = 0
    wall: List[float] = field(default_factory=list)
    n_hand_records: int = 0
    session_checks: Counter = field(default_factory=Counter)
    gap_hands: int = 0  # hands Slumbot counted that we have no result for
    gap_chips: int = 0  # ... and their chips (ours), from Slumbot's session_total

    def add(self, rec: dict) -> None:
        if rec.get("type") != "hand":
            return
        self.n_hand_records += 1
        if rec.get("status") == "ok" and rec.get("winnings") is not None:
            self.winnings.append(int(rec["winnings"]))
            self.positions.append(rec.get("position", "?"))
            self.checks[rec.get("check", "?")] += 1
            if rec.get("wall_s") is not None:
                self.wall.append(float(rec["wall_s"]))
            if rec.get("session_check"):
                self.session_checks[rec["session_check"]] += 1
            gap = rec.get("session_gap")
            if gap:
                self.gap_hands += int(gap.get("hands", 0))
                self.gap_chips += int(gap.get("chips", 0))
        else:
            self.n_errors += 1
        for d in rec.get("decisions") or []:
            self.n_decisions += 1
            if "off_map" in d:  # only blueprint decisions say whether they were on the map
                self.n_blueprint_decisions += 1
                self.n_off_map += int(bool(d["off_map"]))
        # Slumbot's bets once per hand: our last decision lists all of them (every bet or raise
        # gets an answer from us, since nobody can raise against an all-in)
        last = next((d for d in reversed(rec.get("decisions") or []) if "translations" in d), None)
        if last is not None:
            self.n_bot_bets += len(last["translations"])
            self.n_bot_bets_off_grid += sum(1 for t in last["translations"] if not t.get("on_grid"))

    @property
    def n_hands(self) -> int:
        return len(self.winnings)

    def bb100(self) -> Tuple[float, float]:
        return bb100_ci(self.winnings)

    def by_position(self) -> Dict[str, Tuple[int, float, float]]:
        out = {}
        for pos in sorted(set(self.positions)):
            ws = [w for w, p in zip(self.winnings, self.positions) if p == pos]
            out[pos] = (len(ws),) + bb100_ci(ws)
        return out

    def summary(self) -> str:
        m, ci = self.bb100()
        lines = [f"hands with a result: {self.n_hands}   total {sum(self.winnings):+d} chips = {sum(self.winnings) / BIG_BLIND:+.1f} bb"
                 f"   {m:+.1f} bb/100 (95% CI +/-{ci:.1f})"]
        for pos, (n, pm, pci) in self.by_position().items():
            lines.append(f"  as {pos}: {n} hands, {pm:+.1f} bb/100 (+/-{pci:.1f})")
        lines.append(f"failed hands (no result, not in bb/100): {self.n_errors}")
        lines.append("engine cross-check: " + (", ".join(f"{k} {v}" for k, v in sorted(self.checks.items())) or "none"))
        if self.session_checks:
            lines.append("Slumbot session counters: " + ", ".join(f"{k} {v}" for k, v in sorted(self.session_checks.items())))
        if self.gap_hands:
            lines.append(f"hands Slumbot counted without a result on our side: {self.gap_hands}, "
                         f"{self.gap_chips:+d} chips for us (not in the bb/100 above)")
        if self.n_blueprint_decisions:
            lines.append(f"our decisions: {self.n_decisions}, off-map {self.n_off_map} "
                         f"({self.n_off_map / self.n_blueprint_decisions:.1%} of {self.n_blueprint_decisions} blueprint lookups)")
        elif self.n_decisions:
            lines.append(f"our decisions: {self.n_decisions}")
        if self.n_bot_bets:
            lines.append(f"Slumbot bets/raises seen at our decisions: {self.n_bot_bets}, off our grid {self.n_bot_bets_off_grid} "
                         f"({self.n_bot_bets_off_grid / self.n_bot_bets:.1%})")
        if self.wall:
            lines.append(f"wall time per hand: mean {sum(self.wall) / len(self.wall):.2f}s, max {max(self.wall):.2f}s")
        return "\n".join(lines)


# --------------------------------------------------------------------------- log
def read_log(path: str) -> List[dict]:
    """All parseable records of a JSONL log (a line cut short by a crash is skipped)."""
    out: List[dict] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_record(path: str, rec: dict) -> None:
    """Append one JSON line and flush it to disk (a crash loses at most the hand in play)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    needs_newline = False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"  # a line cut short by a crash: start a fresh one
    with open(path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


class SessionCheck:
    """Slumbot's running session counters against ours (see the module docstring).
    ``fresh``: the session starts with our first request (no token yet), so it is at 0 hands."""

    def __init__(self, fresh: bool = False) -> None:
        self.n: Optional[int] = 0 if fresh else None
        self.total: Optional[int] = 0 if fresh else None

    def check(self, rec: dict) -> None:
        final = rec.get("final") or {}
        n, total = final.get("session_num_hands"), final.get("session_total")
        if rec.get("status") != "ok" or not isinstance(n, int) or not isinstance(total, int):
            return
        w = int(rec["winnings"])
        if n == 1:  # a new session (our first hand without a token, or the server started one)
            rec["session_check"] = "match" if total == w else "MISMATCH"
        elif self.n is not None and n == self.n + 1:
            rec["session_check"] = "match" if total - self.total == w else "MISMATCH"
        elif self.n is not None and n > self.n + 1:
            rec["session_check"] = "gap"
            rec["session_gap"] = {"hands": n - self.n - 1, "chips": total - self.total - w}
        else:
            rec["session_check"] = "unknown"
        self.n, self.total = n, total


def hand_seed(base_seed: int, hand_no: int) -> int:
    """The agent's seed for hand ``hand_no`` of a log (so a resumed run continues the sequence)."""
    return (base_seed * 1_000_003 + hand_no) & 0xFFFFFFFF


def _fmt_ci(ci: float) -> str:
    return "inf" if math.isinf(ci) else f"{ci:.1f}"


def hand_line(rec: dict, tally: Tally) -> str:
    m, ci = tally.bb100()
    if rec.get("status") != "ok":
        return f"#{rec.get('hand'):>5} ERROR {rec.get('error')} ({rec.get('wall_s', 0):.1f}s)"
    check = rec.get("check")
    eng = "" if rec.get("engine_net") is None else f" engine {rec['engine_net']:+d}"
    bot = f" vs [{' '.join(rec['bot_cards'])}]" if rec.get("bot_cards") else ""
    sess = rec.get("session_check")
    sess = f" [Slumbot session {sess}{': ' + str(rec['session_gap']) if rec.get('session_gap') else ''}]" if sess and sess != "match" else ""
    return (f"#{rec.get('hand'):>5} {rec.get('position', '?'):>2} [{' '.join(rec.get('hero_cards', []))}]{bot} "
            f"board [{' '.join(rec.get('board', []))}] {rec.get('action')!s:<28} {rec['winnings']:+7d}{eng} ({check}){sess}"
            f" | n={tally.n_hands} {m:+.1f} bb/100 +/-{_fmt_ci(ci)} | {rec.get('wall_s', 0):.2f}s")


# ------------------------------------------------------------------------ match
def run_match(
    client: SlumbotClient,
    agent: Agent,
    log_path: str,
    hands: int,
    *,
    seed: int = 0,
    until: Optional[int] = None,
    max_errors: int = 3,
    meta: Optional[dict] = None,
    out: Callable[[str], None] = print,
    pause_between_hands: float = 0.0,
    trace: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> Tally:
    """Play ``hands`` more hands (fewer if ``until`` total results are reached first), one at a
    time, appending to ``log_path``.  ``trace`` keeps every raw response in the hand records.
    Returns the tally over the whole log."""
    records = read_log(log_path)
    tally = Tally()
    for r in records:
        tally.add(r)
    hand_no = tally.n_hand_records
    if tally.n_hand_records:
        m, ci = tally.bb100()
        out(f"resuming {log_path}: {tally.n_hand_records} hand records, {tally.n_hands} with a result, "
            f"{m:+.1f} bb/100 +/-{_fmt_ci(ci)}")
    run = {"type": "run", "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "host": client.host,
           "agent": getattr(agent, "name", type(agent).__name__), "seed": seed, "hands_requested": hands,
           "first_hand": hand_no}
    run.update(meta or {})
    append_record(log_path, run)
    explainer = DecisionExplainer(agent)
    session = SessionCheck(fresh=client.token is None)
    in_a_row = 0
    for _ in range(hands):
        if until is not None and tally.n_hands >= until:
            break
        try:
            rec = play_hand(client, agent, explainer, hand_no=hand_no, seed=hand_seed(seed, hand_no), trace=trace)
            session.check(rec)
            in_a_row = 0
        except HandAborted as exc:
            rec = exc.record
            in_a_row += 1
        append_record(log_path, rec)
        tally.add(rec)
        out(hand_line(rec, tally))
        hand_no += 1
        if in_a_row >= max_errors:
            out(f"stopping: {in_a_row} failed hands in a row")
            break
        if pause_between_hands > 0:
            sleep(pause_between_hands)
    return tally
