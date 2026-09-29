"""v1 vs a v2 heuristic on the same duel log: per-hand AIVAT values, bb/100 with CIs (a duplicate deal = one sample),
sd per hand, hands for +-5, the paired difference, seconds per hand."""
import argparse, json, math, os, sys, time
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("--n", type=int, default=2000); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--variants", default="v1:0,v2:1"); ap.add_argument("--log", default="/tmp/claude-0/-home-user-Zootop/780badec-1fee-5bac-85c5-e13d71ebb617/scratchpad/hands2000.jsonl")
ap.add_argument("--save", default=None)
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
S = "/tmp/claude-0/-home-user-Zootop/780badec-1fee-5bac-85c5-e13d71ebb617/scratchpad"; B = "/home/user/duel200"
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
recs = [json.loads(l) for l in open(a.log)][: a.n]
hands = [hand_to_dict(hand_from_duel(r, stack=20000, hand_id=i, known="hero")) for i, r in enumerate(recs)]
deals = [int(r["deal"]) for r in recs]
def stat(x):
    x = np.asarray(x) / 100.0
    g = {}
    for d, v in zip(deals, x): g.setdefault(d, []).append(v)
    per = np.array([np.mean(v) for v in g.values()])
    sd_deal = per.std(ddof=1); n = len(x)
    sd_eff = sd_deal * math.sqrt(n / len(per))
    return x.mean() * 100, 1.96 * sd_deal / math.sqrt(len(per)) * 100, x.std(ddof=1), (1.96 * sd_eff * 100 / 5) ** 2
res = {}
for spec_v in a.variants.split(","):
    name, *kv = spec_v.split(":")
    kw = dict(x.split("=") for x in kv)
    seed = int(kw.pop("seed", 0))
    kwi = {k: int(v) for k, v in kw.items()}
    if kwi.pop("preflop", 0):
        from negpluribus.eval.aivat_fast import preflop_equity
        kwi["preflop"] = preflop_equity(1, "/home/user/avbench/preflop_equity_exact.npz")
    ev = FastAivat(game, (4, 8, 8), 2000, seed, rt, **kwi)
    t = time.perf_counter(); out = ev.evaluate_many(hands, a.threads); dt = time.perf_counter() - t
    res[name] = {"value": [r["value"] for r in out], "net": [r["net"] for r in out], "sec": dt, "rollouts": sum(r["rollouts"] for r in out),
                 "thread_s": sum(r["seconds"] for r in out)}
    bad = [i for i, r in enumerate(out) if not math.isfinite(r["value"])]
    assert not bad, bad[:5]
raw = stat(next(iter(res.values()))["net"])
print(f"{len(hands)} hands, {len(set(deals))} deals, {a.threads} threads")
print(f"{'':10} {'bb/100':>8} {'95% CI':>8} {'sd/hand':>8} {'hands +-5':>10} {'s/hand/thr':>10} {'rollouts/hand':>13}")
print(f"{'raw':10} {raw[0]:+8.1f} {raw[1]:8.1f} {raw[2]:8.2f} {raw[3]:10,.0f}")
for name, r in res.items():
    m, ci, sd, need = stat(r["value"])
    print(f"{name:10} {m:+8.1f} {ci:8.1f} {sd:8.2f} {need:10,.0f} {r['thread_s'] / len(hands):10.4f} {r['rollouts'] / len(hands):13,.0f}")
names = list(res)
for nm in names[1:]:
    d = np.array(res[nm]["value"]) - np.array(res[names[0]]["value"])
    m, ci, sd, _ = stat(d)
    print(f"paired {nm} - {names[0]}: {m:+.2f} +- {ci:.2f} bb/100 (sd of the difference per hand {sd:.3f} bb)")
# rollout noise: two seeds of the same heuristic (name_s0, name_s1)
for nm in names:
    if nm.endswith("_s0") and nm[:-3] + "_s1" in res:
        d = np.array(res[nm[:-3] + "_s1"]["value"]) - np.array(res[nm]["value"])
        print(f"rollout noise {nm[:-3]}: sd per hand {d.std(ddof=1) / 100 / math.sqrt(2):.3f} bb (two seeds)")
if a.save: json.dump(res, open(a.save, "w"))
