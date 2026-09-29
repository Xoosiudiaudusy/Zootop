"""AIVAT bench: first N hands of a duel log, known hero, 1+ threads; fingerprint of every number (hex), optional trace sha."""
import argparse, hashlib, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("--n", type=int, default=200); ap.add_argument("--first", type=int, default=0)
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--trace", action="store_true")
ap.add_argument("--out"); ap.add_argument("--cmp"); ap.add_argument("--reps", type=int, default=1)
a = ap.parse_args()
sys.path.insert(0, a.root)
from negpluribus import fast
from negpluribus.abstraction import load_bucketer
from negpluribus.cfr.game import GameSpec
from negpluribus.engine import Street
from negpluribus.eval.aivat import hand_from_duel
from negpluribus.eval.aivat_fast import FastAivat, make_game, root_table, hand_to_dict
from negpluribus.fast.tables import table_name
from negpluribus.fast.blueprint import load_blueprint
from negpluribus.fast.trainer import core_bucketer
S = "/tmp/claude-0/-home-user-Zootop/780badec-1fee-5bac-85c5-e13d71ebb617/scratchpad"
B = "/home/user/duel200"
core = fast.core()
bk = load_bucketer(f"{B}/buckets_base_s0.json")
spec = GameSpec(n_players=2, stack_bb=200, max_street=Street.RIVER, preflop_fracs=(0.5, 1.0, 3.0), postflop_fracs=(0.5, 1.0, 2.0, 4.0),
                max_raises_per_street=3, n_buckets=bk.n_buckets, bucket_kind=getattr(bk, "kind", "ehs"))
cbk = core_bucketer(bk)
tables = core.BucketTables(); tables.load(os.path.join("/home/user/data/bucket_tables_pot16", table_name(cbk)), cbk)
tb = core.TabulatedBucketer(cbk, tables)
bp = load_blueprint(f"{B}/blueprint_base_s0.bin", backend="cpp", n_players=2)
game = make_game(spec, tb, bp)
rt = root_table(game, 256, seed=0, threads=4, cache_path=f"{S}/root_base.npz",
                identity={"blueprint": "blueprint_base_s0.bin", "buckets": "buckets_base_s0.json", "grid": ["0.5,1.0,3.0", "0.5,1.0,2.0,4.0", 3, 200]})
ev = FastAivat(game, (4, 8, 8), 2000, 0, rt)
lines = [json.loads(l) for l in open(f"{S}/hands2000.jsonl")]
hands = [hand_to_dict(hand_from_duel(r, stack=20000, hand_id=i, known="hero")) for i, r in enumerate(lines)][a.first:a.first + a.n]
best = 1e9
for rep in range(a.reps):
    t = time.perf_counter()
    if a.trace:
        res = [ev.evaluate(h, True) for h in hands]
    else:
        res = ev.evaluate_many(hands, a.threads)
    dt = time.perf_counter() - t; best = min(best, dt)
fp = {}
for h, r in zip(hands, res):
    d = {"net": r["net"], "value": float(r["value"]).hex(), "base": float(r["base"]).hex(),
         "terms": [[t[0], t[1], t[2], float(t[3]).hex()] for t in r["terms"]], "rollouts": r.get("rollouts"), "steps": r.get("rollout_steps")}
    if a.trace:
        d["trace"] = hashlib.sha1(repr([(n, [float(x).hex() for x in v]) for n, v in r["trace"]]).encode()).hexdigest()
    fp[str(h["hand_id"])] = d
print(f"{len(hands)} hands, {a.threads} thr: {best:.2f}s best of {a.reps}, {best / len(hands) * a.threads:.4f} thread-s/hand, {len(hands)/best:.2f} hands/s")
if a.out: json.dump(fp, open(a.out, "w"))
if a.cmp:
    ref = json.load(open(a.cmp)); bad = [k for k in fp if fp[k] != ref.get(k)]
    print(f"compare: {len(fp) - len(bad)} identical, {len(bad)} different", bad[:5])
