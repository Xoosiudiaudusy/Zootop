"""The AIVAT known player of a duel hand log, reproduced exactly: the blueprint villain of
scripts/eval_archetypes.py --opponents blueprint (name "blueprint0") is reset before every hand from
(seed, deal, hero seat, name) as negpluribus/eval/duel.py does, so replaying its decisions with the same
BlueprintAgent on the logged cards and actions must give the logged action every time.  A mismatch would
mean the log was not played by the agent the AIVAT model describes (docs/aivat.md, section 1).

    python scripts/check_model_duel.py --hands SP/p3/hands_duel2_bp.jsonl --blueprint ... --buckets ... [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.agents.blueprint import BlueprintAgent  # noqa: E402
from negpluribus.cfr.game import GameSpec  # noqa: E402
from negpluribus.engine import Action, ActionType, Street  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", required=True, help="--log-hands file of eval_archetypes.py")
    ap.add_argument("--blueprint", required=True, help="the villain's blueprint")
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--tables", default=None, help="bucket-table directory (optional)")
    ap.add_argument("--seed", type=int, default=0, help="the duel's --seed")
    ap.add_argument("--villain", default="blueprint0", help="the villain agent's name (its reseed tag)")
    ap.add_argument("--stack", type=int, default=200)
    ap.add_argument("--preflop-fracs", default="0.5,1.0,3.0")
    ap.add_argument("--postflop-fracs", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--max-raises", type=int, default=3)
    args = ap.parse_args()
    bk = load_bucketer(args.buckets)
    spec = GameSpec(n_players=2, stack_bb=args.stack, max_street=Street.RIVER,
                    preflop_fracs=tuple(float(x) for x in args.preflop_fracs.split(",")),
                    postflop_fracs=tuple(float(x) for x in args.postflop_fracs.split(",")),
                    max_raises_per_street=args.max_raises, n_buckets=bk.n_buckets, bucket_kind=getattr(bk, "kind", "ehs"))
    if args.tables:
        from negpluribus.fast.tables import tabulated
        from negpluribus.fast.trainer import core_bucketer
        bk = tabulated(core_bucketer(bk), args.tables)
    bp = load_blueprint(args.blueprint)
    agent = BlueprintAgent(bp, bk, spec.grid, name=args.villain, seed=10)
    tag = zlib.crc32(args.villain.encode())
    t0 = time.perf_counter()
    hands = decisions = mismatches = 0
    examples = []
    with open(args.hands, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            r = row["hero_seat"]
            v = 1 - r
            holes, board = row["holes"], row["board"]
            used = set(holes[0]) | set(holes[1]) | set(board)
            order = list(holes[0]) + list(holes[1]) + list(board) + [c for c in range(52) if c not in used]
            st = spec.new_hand(order, button=row["button"])
            assert [list(p.hole) for p in st.players] == [list(h) for h in holes], "dealing order"
            agent.reset(seed=(args.seed * 7_919 + row["deal"] * 31 + r) ^ tag)
            hands += 1
            for street, seat, t, amount in row["events"]:
                assert st.current_player == seat and int(st.street) == street, "replay out of step"
                if seat == v:
                    a = agent.act(st.observe(seat))
                    got = (int(a.type), int(a.amount) if a.type == ActionType.RAISE else 0)
                    decisions += 1
                    if got != (t, amount):
                        mismatches += 1
                        if len(examples) < 5:
                            examples.append((row["deal"], r, street, (t, amount), got))
                st.apply(Action(ActionType(t), amount))
    print(f"hands {hands:,}; villain decisions {decisions:,}; mismatches {mismatches}; {time.perf_counter() - t0:.0f}s")
    for ex in examples:
        print("  mismatch: deal %d hero seat %d street %d logged %s replayed %s" % ex)


if __name__ == "__main__":
    main()
