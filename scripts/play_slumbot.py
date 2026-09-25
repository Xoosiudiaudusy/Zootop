"""Play a blueprint against Slumbot (https://slumbot.com), one hand at a time.

    python scripts/play_slumbot.py --blueprint data/blueprint_X.json --buckets data/buckets_X.json \
        --hands 1000 --log data/slumbot/X.jsonl

    # dry run against the local mock server (nothing leaves the machine)
    python scripts/play_slumbot.py --blueprint ... --buckets ... --hands 200 --mock random

Every hand is one JSON line in --log (format in docs/slumbot.md); a running bb/100 with its 95%
confidence interval is printed after each hand.  Re-running with the same --log resumes: the old
hands count, new ones are appended (--hands N plays N more; --until N stops once the log holds N
hands with a result).  Slumbot's game is heads-up 50/100 with 200bb stacks reset every hand, so the
blueprint must be trained for 2 players, 200bb, betting through the river; the default grid is the
wide one (preflop 1 and 3 pots, postflop 0.5/1/2/4 pots, 3 raises per street).

Politeness: requests are strictly sequential with --pause seconds between them, failed requests
are retried with exponential backoff (network errors, 5xx), and the match stops after
--max-errors failed hands in a row.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import bucketer_kind, load_bucketer  # noqa: E402
from negpluribus.agents import CallingAgent, GridRandomAgent, RandomAgent, make_agent  # noqa: E402
from negpluribus.agents.blueprint import BlueprintAgent  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402
from negpluribus.slumbot import BIG_BLIND, DEFAULT_HOST, SMALL_BLIND, STACK_SIZE, MockSlumbot, SlumbotClient, run_match  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def fracs(s: str):
    return tuple(float(x) for x in s.split(",") if x.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", default="blueprint", help="blueprint (default), or caller / gridrandom / random for plumbing tests")
    ap.add_argument("--blueprint", help="average strategy from scripts/train_blueprint.py (.bin or .json; looked up in C++ when the core is built)")
    ap.add_argument("--buckets", help="the bucketer JSON the blueprint was trained with")
    ap.add_argument("--preflop-fracs", default="1.0,3.0")
    ap.add_argument("--postflop-fracs", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--max-raises", type=int, default=3)
    ap.add_argument("--hands", type=int, default=100, help="hands to play in this run")
    ap.add_argument("--until", type=int, default=None, help="stop once the log holds this many hands with a result")
    ap.add_argument("--log", default=None, help="JSONL log (appended; default data/slumbot/<blueprint name>.jsonl)")
    ap.add_argument("--seed", type=int, default=0, help="agent seed; hand i of the log uses a seed derived from (seed, i)")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds per HTTP request")
    ap.add_argument("--max-retries", type=int, default=4, help="retries per request (network errors, 5xx), exponential backoff")
    ap.add_argument("--max-errors", type=int, default=3, help="stop after this many failed hands in a row")
    ap.add_argument("--mock", default=None, metavar="BOT",
                    help="play the local mock server instead of slumbot.com, with this bot on its side "
                         "(random, caller, gridrandom or an archetype: nit, tag, lag, station, maniac, ...)")
    ap.add_argument("--mock-seed", type=int, default=0)
    ap.add_argument("--trace", action="store_true", help="also log every raw server response of each hand (token removed)")
    args = ap.parse_args()

    spec = None
    meta = {"argv": sys.argv[1:]}
    grid = GameSpec(preflop_fracs=fracs(args.preflop_fracs), postflop_fracs=fracs(args.postflop_fracs),
                    max_raises_per_street=args.max_raises).grid
    if args.agent == "blueprint":
        if not (args.blueprint and args.buckets):
            ap.error("--agent blueprint needs --blueprint and --buckets")
        bk = load_bucketer(args.buckets)
        spec = GameSpec(
            n_players=2, stack_bb=STACK_SIZE // BIG_BLIND, sb=SMALL_BLIND, bb=BIG_BLIND, max_street=Street.RIVER,
            preflop_fracs=fracs(args.preflop_fracs), postflop_fracs=fracs(args.postflop_fracs),
            max_raises_per_street=args.max_raises, n_buckets=bk.n_buckets, bucket_kind=bucketer_kind(bk),
        )
        t = time.perf_counter()
        bp = load_blueprint(args.blueprint)
        name = os.path.splitext(os.path.basename(args.blueprint))[0]
        agent = BlueprintAgent(bp, bk, spec.grid, name=name, seed=args.seed)
        print(f"game: {spec.describe()}")
        print(f"blueprint: {len(bp):,} infosets from {args.blueprint} ({time.perf_counter() - t:.1f}s)")
        meta.update(blueprint=args.blueprint, buckets=args.buckets, game=spec.describe())
    else:
        agent = {"caller": lambda: CallingAgent(seed=args.seed),
                 "gridrandom": lambda: GridRandomAgent(grid, seed=args.seed),
                 "random": lambda: RandomAgent(seed=args.seed)}[args.agent]()
        name = args.agent

    log = args.log or os.path.join(DATA, "slumbot", f"{name}{'_mock' if args.mock else ''}.jsonl")
    if args.mock:
        bot = GridRandomAgent(grid, seed=args.mock_seed) if args.mock == "gridrandom" else make_agent(args.mock, seed=args.mock_seed)
        mock = MockSlumbot(bot, seed=args.mock_seed)
        client = SlumbotClient(host="http://mock.local", transport=mock.transport, min_interval=0,
                               timeout=args.timeout, max_retries=args.max_retries, log=print)
        meta["mock_bot"] = args.mock
    else:
        client = SlumbotClient(host=args.host, timeout=args.timeout, max_retries=args.max_retries,
                               min_interval=args.pause, log=print)
    print(f"opponent: {'local mock (' + args.mock + ')' if args.mock else client.host}   log: {log}")
    try:
        tally = run_match(client, agent, log, args.hands, seed=args.seed, until=args.until,
                          max_errors=args.max_errors, meta=meta, trace=args.trace)
    except KeyboardInterrupt:
        print("\ninterrupted: the hand in play (if any) is not in the log; its result on the server is unknown")
        return 130
    print()
    print(tally.summary())
    print(f"requests: {client.n_requests} ({client.n_retries} retries, {client.n_token_changes} token changes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
