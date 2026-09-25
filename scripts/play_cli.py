"""Play 6-max against the archetype bots in the terminal.

    python scripts/play_cli.py --villains nit,station,maniac,lag,passive --hands 50

Type  f / c / r <amount-to> / a (all-in) / q.  A live HUD of the bots' stats
(as *you* would see them after N hands) is printed every hand so you can watch
the tracker converge.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.agents import make_agent  # noqa: E402
from negpluribus.agents.base import Agent  # noqa: E402
from negpluribus.cards import cards_to_str  # noqa: E402
from negpluribus.engine import CALL, FOLD, STREET_NAMES, Action, Observation  # noqa: E402
from negpluribus.stats import StatsTracker  # noqa: E402
from negpluribus.table import CashTable, format_record  # noqa: E402


class HumanAgent(Agent):
    name = "you"

    def act(self, obs: Observation) -> Action:
        print(f"\n[{STREET_NAMES[obs.street]}] board: [{cards_to_str(obs.board)}]  pot={obs.pot}  "
              f"you ({obs.position}): [{cards_to_str(obs.hole)}] stack={obs.stack}  to_call={obs.to_call}")
        for e in obs.events[-6:]:
            print(f"    seat{e.seat}: {e.action}")
        opts = ["c=" + ("check" if obs.to_call == 0 else f"call {obs.to_call}")]
        if obs.can_fold:
            opts.insert(0, "f=fold")
        if obs.can_raise:
            opts.append(f"r <to> (min {obs.min_raise_to}, max {obs.max_raise_to})")
            opts.append("a=all-in")
        while True:
            s = input("  " + " | ".join(opts) + " > ").strip().lower()
            if s == "q":
                raise KeyboardInterrupt
            if s == "f" and obs.can_fold:
                return FOLD
            if s in ("c", ""):
                return CALL
            if s == "a" and obs.can_raise:
                return obs.clamp_raise(obs.max_raise_to)
            if s.startswith("r") and obs.can_raise:
                try:
                    return obs.clamp_raise(int(s[1:].strip()))
                except ValueError:
                    pass
            print("  ?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--villains", default="nit,station,maniac,lag,passive")
    ap.add_argument("--hands", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hide-styles", action="store_true", help="label bots bot1..bot5 so you have to read them")
    args = ap.parse_args()

    vil = args.villains.split(",")
    bots = [make_agent(v, seed=args.seed + i, label=(f"bot{i + 1}" if args.hide_styles else v)) for i, v in enumerate(vil)]
    agents = [HumanAgent()] + bots
    names = [a.name for a in agents]
    table = CashTable(agents, seed=args.seed)
    tracker = StatsTracker()
    try:
        for h in range(args.hands):
            rec = table.play(1).records[-1]
            tracker.observe_hand(rec, names)
            print(format_record(rec, names))
            print(f"session: {table.result.net[0] / 100:+.1f} bb over {h + 1} hands")
            print("HUD:\n" + tracker.report(names[1:]))
    except KeyboardInterrupt:
        print("\nbye")
    if args.hide_styles:
        print("styles were:", dict(zip(names[1:], vil)))


if __name__ == "__main__":
    main()
