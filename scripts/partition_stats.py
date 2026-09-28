"""Would a tree-partitioned trainer work on this game?  Subtrees from the flop (or turn) on separate machines.

    python scripts/partition_stats.py --players 3 --stack 100 --buckets K.json --train 2000000 --sample 50000

Trains the flat batched trainer --train iterations (a realistic strategy), then runs the forward pass of
--sample more iterations without updating and reports, per boundary street:
* subtrees (one per preflop line that reaches the flop, or per line to the turn), cells per subtree (memory);
* work per subtree (node visits, terminals included) and the head's share (the betting above the boundary);
* crossings per iteration (a subtree root reached: a value the owner must send back to the head);
* the best split over K machines by work (greedy, largest first): the busiest machine's share.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from negpluribus import fast  # noqa: E402
from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.fast.trainer import core_bucketer, spec_to_dict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--players", type=int, default=3)
ap.add_argument("--stack", type=int, default=100)
ap.add_argument("--preflop-fracs", default="0.5,1.0,3.0")
ap.add_argument("--postflop-fracs", default="0.5,1.0,2.0,4.0")
ap.add_argument("--max-raises", type=int, default=3)
ap.add_argument("--buckets", required=True)
ap.add_argument("--train", type=int, default=2_000_000)
ap.add_argument("--sample", type=int, default=50_000)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--scale-buckets", type=int, default=64, help="also report cells as if the game had this many buckets")
args = ap.parse_args()
fr = lambda s: tuple(float(x) for x in s.split(",") if x.strip())
bk = load_bucketer(args.buckets)
spec = GameSpec(n_players=args.players, stack_bb=args.stack, max_street=Street.RIVER, preflop_fracs=fr(args.preflop_fracs),
                postflop_fracs=fr(args.postflop_fracs), max_raises_per_street=args.max_raises, n_buckets=bk.n_buckets,
                bucket_kind=getattr(bk, "kind", "ehs"))
core = fast.core()
ft = core.FlatTrainer(spec_to_dict(spec), core_bucketer(bk), 0, True, args.threads)
ft.batch_size = args.batch
gs = ft.game_stats()
print(f"game: {spec.describe()}; cells {gs['cells']:,}, infosets {gs['infosets']:,}", flush=True)
t = time.time()
if args.train:
    ft.train(args.train)
print(f"trained {args.train:,} iterations in {time.time() - t:.0f}s", flush=True)
lo = ft.iteration + 1
hi = lo + args.sample - 1
scale = args.scale_buckets / bk.n_buckets
for boundary, name in ((1, "flop"), (2, "turn")):
    t = time.time()
    s = ft.partition_stats(boundary, lo, hi)
    R = len(s["roots"])
    items = s["items"]
    tot_items = sum(items) + s["head_items"]
    tot_cells = sum(s["cells"]) + s["head_cells"]
    cross = sum(s["crossings"])
    iters = args.sample
    print(f"\n== boundary: {name} ({time.time() - t:.0f}s)")
    print(f"subtrees {R:,}; head (above the boundary): {s['head_cells'] / tot_cells:.4%} of cells "
          f"({s['head_cells'] * scale:,.0f} at {args.scale_buckets} buckets... preflop rows do not scale), "
          f"{s['head_items'] / tot_items:.2%} of work")
    order = sorted(range(R), key=lambda i: -items[i])
    big = order[:5]
    print("largest subtrees by work: " + ", ".join(f"{items[i] / tot_items:.1%} work / {s['cells'][i] / tot_cells:.1%} cells" for i in big))
    bigc = max(s["cells"]) / tot_cells
    print(f"largest subtree by cells: {bigc:.1%} of all cells")
    print(f"crossings: {cross / iters:.1f} per iteration -> x 8 bytes (one double back to the head) = "
          f"{cross / iters * 8 * 16384 / 1e6:.2f} MB per batch of 16384")
    hr = s["head_records"] / iters
    print(f"update records per iteration: head {hr:.1f}, subtrees {s['sub_records'] / iters:.1f}; head records to every machine "
          f"(12 bytes each: cell + value) = {hr * 12 * 16384 / 1e6:.1f} MB per batch of 16384")
    hd = s["head_distinct"] / max(1, s["batches"])
    print(f"distinct head cells updated per batch of 16384 (regret or strategy sum): {hd:,.0f} -> summed per machine, "
          f"{hd * 12 / 1e6:.1f} MB per machine per batch")
    h = s["head_items"] / tot_items
    print("head replicated on every machine (only boundary values travel, bit for bit): speed-up " +
          ", ".join(f"{K} machines x{1 / (h + (1 - h) / K):.2f}" for K in (2, 3, 4, 8)))
    for K in (2, 3, 4, 8):
        load = [0] * K
        cells = [0] * K
        for i in order:
            k = min(range(K), key=lambda j: load[j])
            load[k] += items[i]
            cells[k] += s["cells"][i]
        print(f"  {K} machines: busiest {max(load) / (sum(items) or 1):.1%} of subtree work (ideal {1 / K:.1%}), "
              f"most cells {max(cells) / tot_cells:.1%}")
