"""Exploitability vs time of the search's configurations on the turn (or river) roots of scripts/search_quality.py:
the MCCFR, the vector CFR with the river by the blueprint's buckets, by K strength buckets shared by the river cards
(river_buckets), exactly per river card (river_exact), and the vector CFR's weightings (Linear, CFR+, DCFR).

    python scripts/river_abstraction_bench.py --blueprint B.bin --buckets K.json --threads 4 --seconds 1,2,4,8 \\
        --configs mccfr,vec,k16,k64,k200,k500,exact
    python scripts/river_abstraction_bench.py ... --street river --configs vec,vec+,vecd
    python scripts/river_abstraction_bench.py ... --cache C.bin --iterations 800 --configs vec,vecd,k16,k16d,k200,k200d

--iterations: fixed iterations per search instead of --seconds (vector iterations for the vector configs, MCCFR
iterations for mccfr), e.g. the agent's --search-vector-iterations; --cache: the buckets' saved cache
(scripts/precompute_buckets.py).  At the end: seconds per search, the process's peak memory (working set and commit;
per configuration when each runs in its own process; loading the blueprint may set it) and the search's own memory
(the largest working set and commit right after a solve, the search alive, minus the values after loading).

Configs: mccfr | vec (blueprint's river buckets) | k<K> (K strength buckets) | exact | warm<K>_<percent> (exact river
warm-started from K buckets after <percent> % of the budget); a suffix "+" = CFR+,
"d" = DCFR(1.5, 0, 2) (e.g. k200d, vecd).  Exact exploitability of the average strategy (subgame_exploitability,
kind 0, the best responder deviating from the root's street), bb per deal of the subgame; MCCFR: mean of --seeds.
On river roots (--street river) the river's abstraction options act only on turn roots: k<K> is vec there (and k<K>d is
vecd, k<K>+ is vec+), exact and warm are vec as well -- the same numbers, printed under their names.
"""
import argparse
import ctypes
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from negpluribus import fast  # noqa: E402
from negpluribus.abstraction import load_bucketer  # noqa: E402
from negpluribus.cfr import GameSpec  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.fast.blueprint import load_blueprint  # noqa: E402
from negpluribus.fast.trainer import core_bucketer, spec_to_dict  # noqa: E402

core = fast.core()
FLOP_LINE = ["r1", "c", "c", "r0.5", "c"]
SPOTS = {
    "turn, BB first to act": FLOP_LINE,
    "turn, BB faces a pot bet": FLOP_LINE + ["c", "r1"],
    "river, BB first to act": FLOP_LINE + ["c", "c"],
    "river, BB faces a pot bet": FLOP_LINE + ["c", "c", "c", "r1"],
}


def parse(cfg):
    disc = 0
    if cfg.endswith("+"):
        disc, cfg = 1, cfg[:-1]
    elif cfg.endswith("d"):
        disc, cfg = 2, cfg[:-1]
    if cfg == "mccfr":
        return dict(vector_cfr=False)
    if cfg == "vec":
        return dict(vector_cfr=True, vector_discount=disc)
    if cfg == "exact":
        return dict(vector_cfr=True, river_exact=True, vector_discount=disc)
    if cfg.startswith("warm"):  # warm<K>_<percent>: K bucket phase for <percent> % of the budget, then the exact river
        k, pct = cfg[4:].split("_")
        return dict(vector_cfr=True, river_exact=True, river_buckets=int(k), river_warm=float(pct) / 100.0)
    if cfg.startswith("k"):
        return dict(vector_cfr=True, river_buckets=int(cfg[1:]), vector_discount=disc)
    raise SystemExit(f"unknown config {cfg}")


def memory():
    """(peak working set, peak commit, working set, commit) of this process in MB (Windows; elsewhere the max RSS)."""
    if os.name == "nt":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                               "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return (c.PeakWorkingSetSize / 2**20, c.PeakPagefileUsage / 2**20, c.WorkingSetSize / 2**20,
                    c.PrivateUsage / 2**20)
        return (float("nan"),) * 4
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return rss, rss, rss, rss


ap = argparse.ArgumentParser()
ap.add_argument("--blueprint", required=True)
ap.add_argument("--buckets", required=True)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--seconds", default="1,2,4,8")
ap.add_argument("--iterations", default="", help="fixed iterations per search, e.g. 800 or 400,800 (instead of --seconds)")
ap.add_argument("--cache", default=None, help="saved bucket cache of --buckets (scripts/precompute_buckets.py)")
ap.add_argument("--street", choices=("turn", "river"), default="turn")
ap.add_argument("--hands", type=int, default=2)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--configs", default="mccfr,vec,k16,k64,k200,k500,exact")
args = ap.parse_args()
bk = load_bucketer(args.buckets)
spec = GameSpec(n_players=2, stack_bb=200, max_street=Street.RIVER, preflop_fracs=(0.5, 1.0, 3.0),
                postflop_fracs=(0.5, 1.0, 2.0, 4.0), max_raises_per_street=3, n_buckets=bk.n_buckets,
                bucket_kind=getattr(bk, "kind", "ehs"))
bp = load_blueprint(args.blueprint)
cbk = core_bucketer(bk, (8_000_000, 64_000_000, 4_000_000))
if args.cache:
    print(f"bucket cache: {cbk.load_cache(args.cache)}", flush=True)
game = core.SearchGame(spec_to_dict(spec), cbk, bp.lookup)
fixed = bool(args.iterations.strip())
budgets = [int(x) for x in args.iterations.split(",")] if fixed else [float(x) for x in args.seconds.split(",")]
unit = "it" if fixed else "s"
ws0, commit0, cws0, ccommit0 = memory()
print(f"loaded: peak working set {ws0:,.0f} MB, peak commit {commit0:,.0f} MB; now {cws0:,.0f} / {ccommit0:,.0f} MB",
      flush=True)
search_mem = [0.0, 0.0]  # the largest (working set, commit) right after a solve, minus the values after loading
configs = [c.strip() for c in args.configs.split(",") if c.strip()]
rng = random.Random(args.seed)
table = {}  # (config, seconds) -> [eps per root]
its = {}
secs = {}  # (config, budget) -> [seconds per search]
print(f"{args.threads} threads, {args.street} roots; exploitability of the average strategy, bb per deal", flush=True)
for spot, line in SPOTS.items():
    for h in range(args.hands):
        order = list(range(52))
        rng.shuffle(order)
        if not spot.startswith(args.street):
            continue
        st = spec.new_hand(order, button=0)
        acts = []
        for name in line:
            obs = st.observe(st.current_player)
            a = spec.grid.to_concrete(obs, name)
            acts.append((int(a.type), int(a.amount)))
            st.apply(a)
        obs = st.observe(st.current_player)
        br = int(st.street)
        print(f"\n[{spot} #{h}] board {obs.board}", flush=True)
        for sec in budgets:
            cells = []
            for cfg in configs:
                kw = parse(cfg)
                seeds = range(1, args.seeds + 1) if cfg == "mccfr" else (1,)
                ex = []
                for sd in seeds:
                    s = core.SubgameSearch(game, list(st.starting_stacks), 0, acts, list(obs.board), obs.seat, list(obs.hole),
                                           iterations=sec if fixed else 0, time_budget=0.0 if fixed else sec,
                                           threads=args.threads, seed=sd, depth="pluribus", **kw)
                    r = s.solve()
                    _, _, cws, ccommit = memory()
                    search_mem[0] = max(search_mem[0], cws - cws0)
                    search_mem[1] = max(search_mem[1], ccommit - ccommit0)
                    ex.append(s.subgame_exploitability(0, br)[0])
                    its.setdefault((cfg, sec), []).append(r["iterations"] / max(r["seconds"], 1e-9))
                    secs.setdefault((cfg, sec), []).append(r["seconds"])
                e = statistics.fmean(ex)
                table.setdefault((cfg, sec), []).append(e)
                cells.append(f"{cfg} {e:7.3f}")
            print(f"  {sec:5g}{unit}  " + "  ".join(cells), flush=True)
print("\nmean over roots (iterations per second in brackets):")
print(f"  {'budget':>8s}  " + "".join(f"{c:>18s}" for c in configs))
for sec in budgets:
    print(f"  {sec:>6g}{unit:2s}  " + "".join(f"{statistics.fmean(table[(c, sec)]):9.3f} ({statistics.fmean(its[(c, sec)]):>6,.0f})" for c in configs))
print("seconds per search (mean over roots):")
for sec in budgets:
    print(f"  {sec:>6g}{unit:2s}  " + "".join(f"{statistics.fmean(secs[(c, sec)]):18.2f}" for c in configs))
ws, commit, _, _ = memory()
print(f"process peak: working set {ws:,.0f} MB, commit {commit:,.0f} MB (after loading: {ws0:,.0f} / {commit0:,.0f} MB)", flush=True)
print(f"search memory (largest after a solve, the search alive, over loading): working set +{search_mem[0]:,.0f} MB, "
      f"commit +{search_mem[1]:,.0f} MB", flush=True)
