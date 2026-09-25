"""Playing our agents against Slumbot (slumbot.com), a strong public heads-up NLHE bot.

* ``protocol`` - constants (50/100, 200bb), the action-string grammar, seat mapping, ``incr``;
* ``client``   - the HTTP client: one request at a time, pacing, timeouts, backoff;
* ``adapter``  - replay of Slumbot's action string into our engine, one-hand driver, decision log,
                 chip cross-check;
* ``runner``   - a resumable match with a JSONL log and a running bb/100 with its 95% CI;
* ``mock``     - a local server with the same protocol (tests, dry runs).

docs/slumbot.md has the verified protocol, the log format and what is still uncertain.
"""
from .adapter import DecisionExplainer, HandAborted, make_deck, play_hand, replay
from .client import DEFAULT_HOST, SlumbotClient, SlumbotError, TransportError
from .mock import MockSlumbot
from .protocol import (
    BIG_BLIND,
    SMALL_BLIND,
    STACK_SIZE,
    Move,
    ProtocolError,
    bot_seat,
    format_incr,
    hero_seat,
    split_action,
)
from .runner import Tally, bb100_ci, read_log, run_match

__all__ = [
    "BIG_BLIND", "DEFAULT_HOST", "DecisionExplainer", "HandAborted", "MockSlumbot", "Move", "ProtocolError",
    "SMALL_BLIND", "STACK_SIZE", "SlumbotClient", "SlumbotError", "Tally", "TransportError", "bb100_ci",
    "bot_seat", "format_incr", "hero_seat", "make_deck", "play_hand", "read_log", "replay", "run_match",
    "split_action",
]
