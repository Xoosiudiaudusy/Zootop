"""E[HS] buckets vs potential-aware buckets: train two blueprints, play them against each other
and against the archetypes on identical deals (docs/buckets.md, acceptance test (d)).

    python scripts/compare_buckets.py                                  # 300k iterations, cpp backend, all cores
    python scripts/compare_buckets.py --iters 30000 --h2h-deals 300 --arch-deals 150   # quick look

Game: GameSpec(n_players=2, stack_bb=20, max_street=FLOP, n_buckets=8).  Both blueprints get the
same seed and the same number of iterations; the bucketers are fitted with the same seed.  A
third blueprint (E[HS], seed 1) is the noise floor: how much two blueprints of the *same*
abstraction differ.  Evaluation is the Python reference (``duplicate_match`` /
``compare_heroes``): the hero's random stream is keyed by its slot, so two heroes compared on
the same deals differ only where their policies differ.  Outputs: data/blueprint_<tag>.json,
data/buckets_<tag>.json and the markdown tables on stdout.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus import fast  # noqa: E402
from negpluribus.abstraction import EquityBucketer, PotentialAwareBucketer, load_bucketer  # noqa: E402
from negpluribus.abstraction import potential as potential_mod  # noqa: E402
from negpluribus.agents import make_agent  # noqa: E402
from negpluribus.agents.blueprint import BlueprintAgent  # noqa: E402
from negpluribus.cfr.game import GameSpec  # noqa: E402
from negpluribus.cfr.mccfr import MCCFRTrainer  # noqa: E402
from negpluribus.cfr.strategy import BlueprintStrategy  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.eval import compare_heroes, duplicate_match  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
ARCHETYPES = ("tag", "nit", "maniac", "station")


def fit_or_load(spec: GameSpec, tag: str, n_situations: int, seed: int):
    path = os.path.join(DATA, f"buckets_{tag}.json")
    if os.path.exists(path):
        bk = load_bucketer(path)
        print(f"buckets {tag}: loaded {path}")
        return bk, 0.0
    bk = spec.make_bucketer()
    t0 = time.perf_counter()
    bk.fit(n_situations=n_situations, seed=seed, verbose=True)
    dt = time.perf_counter() - t0
    bk.save(path)
    print(f"buckets {tag}: fitted in {dt:.1f}s ({type(bk).__name__}), saved {path}")
    return bk, dt


def train_or_load(spec: GameSpec, bk, tag: str, iters: int, seed: int, backend: str, threads):
    bp_path = os.path.join(DATA, f"blueprint_{tag}.json")
    if os.path.exists(bp_path):
        print(f"blueprint {tag}: loaded {bp_path}")
        return BlueprintStrategy.load(bp_path), None
    tr = MCCFRTrainer(spec, bk, seed=seed, backend=backend, threads=threads)
    t0 = time.perf_counter()
    tr.train(iters, log_every=max(1, iters // 5))
    dt = time.perf_counter() - t0
    info = {
        "time": dt,
        "infosets": len(tr.nodes) if tr.backend == "python" else tr.n_nodes,
        "nodes_touched": tr.nodes_touched,
        "cache": tr._core_bucketer.cache_size() if tr.backend == "cpp" else len(bk._cache),
        "backend": f"{tr.backend}" + (f" x{tr.threads}" if tr.backend == "cpp" else ""),
    }
    print(f"blueprint {tag}: {iters:,} iterations in {dt:.0f}s ({info['backend']}), {info['infosets']:,} infosets, "
          f"{info['cache']:,} canonical forms bucketed")
    strat = tr.strategy()
    strat.save(bp_path)
    tr.save_checkpoint(os.path.join(DATA, f"checkpoint_{tag}.json"))
    return strat, info


def bucket_cost(bk_e: EquityBucketer, bk_p: PotentialAwareBucketer, n: int = 150) -> dict:
    """Milliseconds per *new* canonical form (cache misses) on random flop / turn / river situations."""
    out = {}
    rng = random.Random(99)
    sits = {k: [rng.sample(range(52), 2 + k) for _ in range(n)] for k in (3, 4, 5)}

    def time_py(bk, label):
        for k, name in ((3, "flop"), (4, "turn"), (5, "river")):
            bk._cache.clear()
            if hasattr(bk, "_features"):
                bk._features.clear()
            t0 = time.perf_counter()
            for s in sits[k]:
                bk.bucket(s[:2], s[2:])
            out[(label, name)] = (time.perf_counter() - t0) / n * 1000

    from negpluribus.abstraction import buckets as buckets_mod
    from negpluribus.equity import equity_vs_random_py

    hooks = (potential_mod.next_street_histogram, potential_mod.river_equity, potential_mod.equity_vs_random, buckets_mod.equity_vs_random)
    # pure Python reference: no compiled helper anywhere (histogram, river equity, equity loop)
    potential_mod.next_street_histogram, potential_mod.river_equity = potential_mod.next_street_histogram_py, potential_mod.river_equity_py
    potential_mod.equity_vs_random = buckets_mod.equity_vs_random = equity_vs_random_py
    time_py(bk_p, "potential/python (pure)")
    time_py(bk_e, "ehs/python (pure)")
    potential_mod.next_street_histogram, potential_mod.river_equity, potential_mod.equity_vs_random, buckets_mod.equity_vs_random = hooks
    if fast.core() is not None:
        time_py(bk_p, "potential/python+hooks")
        time_py(bk_e, "ehs/python+hooks")
        from negpluribus.fast.trainer import core_bucketer

        for bk, label in ((bk_e, "ehs/cpp"), (bk_p, "potential/cpp")):
            cbk = core_bucketer(bk)
            for k, name in ((3, "flop"), (4, "turn"), (5, "river")):
                t0 = time.perf_counter()
                for s in sits[k]:
                    cbk.bucket(s[:2], s[2:])
                out[(label, name)] = (time.perf_counter() - t0) / n * 1000
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300_000)
    ap.add_argument("--h2h-deals", type=int, default=3000)
    ap.add_argument("--arch-deals", type=int, default=1500)
    ap.add_argument("--fit-situations", type=int, default=1200)
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["python", "cpp"], default="cpp")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-control", action="store_true", help="skip the second E[HS] blueprint (noise floor)")
    ap.add_argument("--tag", default=None, help="tag prefix (default 2p_20bb_flop_<iters>k)")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    print("fast:", fast.describe())

    base = args.tag or f"2p_20bb_flop_{args.iters // 1000}k"
    if args.buckets != 8:
        base += f"_b{args.buckets}"
    spec_e = GameSpec(n_players=2, stack_bb=20, max_street=Street.FLOP, n_buckets=args.buckets, bucket_kind="ehs")
    spec_p = GameSpec(n_players=2, stack_bb=20, max_street=Street.FLOP, n_buckets=args.buckets, bucket_kind="potential")
    print("game:", spec_e.describe(), "/", spec_p.describe())

    bk_e, fit_e = fit_or_load(spec_e, f"{base}_ehs", args.fit_situations, args.seed)
    bk_p, fit_p = fit_or_load(spec_p, f"{base}_pot", args.fit_situations, args.seed)
    print("\npotential-aware clusters (flop):")
    for b, (m, s, c) in enumerate(zip(bk_p.centroid_mean_equity[int(Street.FLOP)], bk_p.centroid_share[int(Street.FLOP)], bk_p.centroids[int(Street.FLOP)])):
        print(f"  b{b}: mean equity {m:.2f}  share {s:.2f}  {bk_p.describe_centroid(c)}")
    print("E[HS] cuts (flop):", " ".join(f"{c:.2f}" for c in bk_e.boundaries[int(Street.FLOP)]))

    bp_e, info_e = train_or_load(spec_e, bk_e, f"{base}_ehs", args.iters, args.seed, args.backend, args.threads)
    bp_p, info_p = train_or_load(spec_p, bk_p, f"{base}_pot", args.iters, args.seed, args.backend, args.threads)
    bp_c = info_c = None
    if not args.no_control:
        ctl_tag = f"{base}_ehs_seed{args.seed + 1}"
        bp_c, info_c = train_or_load(spec_e, bk_e, ctl_tag, args.iters, args.seed + 1, args.backend, args.threads)
        ctl_bk_path = os.path.join(DATA, f"buckets_{ctl_tag}.json")
        if not os.path.exists(ctl_bk_path):
            bk_e.save(ctl_bk_path)  # same E[HS] buckets under the control's tag, so exploitability.py finds them

    kw = dict(sb=spec_e.sb, bb=spec_e.bb, stack_bb=spec_e.stack_bb, max_street=spec_e.max_street)

    def hero(kind: str, name: str, seed: int = 1) -> BlueprintAgent:
        if kind == "pot":
            return BlueprintAgent(bp_p, bk_p, spec_p.grid, seed=seed, name=name)
        if kind == "ehs":
            return BlueprintAgent(bp_e, bk_e, spec_e.grid, seed=seed, name=name)
        return BlueprintAgent(bp_c, bk_e, spec_e.grid, seed=seed, name=name)

    # ---------------------------------------------------------------- (i) head to head
    print(f"\n(i) head-to-head, {args.h2h_deals} duplicate deals x 2 rotations (Python reference evaluation)")
    rows = []
    t0 = time.perf_counter()
    pairs = [("pot", "ehs"), ("ehs", "pot")]
    if bp_c is not None:
        pairs += [("ctl", "ehs"), ("ehs", "ctl"), ("pot", "ctl")]
    for h, v in pairs:
        H, V = hero(h, h), hero(v, v, seed=2)
        res = duplicate_match(H, [V], n_deals=args.h2h_deals, seed=args.seed + 11, **kw)
        rows.append((h, v, res.bb100, res.ci95, H.fallback_rate, V.fallback_rate))
        print(f"  {h:>3} (hero) vs {v:<3}: {res.bb100:+7.2f} bb/100  +/-{res.ci95:.2f}   off-map hero {H.fallback_rate:.1%} villain {V.fallback_rate:.1%}")
    t_h2h = time.perf_counter() - t0

    # ------------------------------------------------------------ (ii) vs archetypes
    print(f"\n(ii) vs archetypes, {args.arch_deals} duplicate deals x 2 rotations, paired (compare_heroes)")
    arch_rows = []
    t0 = time.perf_counter()
    for name in ARCHETYPES:
        villain = make_agent(name, seed=10)
        ra, rb, gain, ci = compare_heroes(hero("pot", "pot"), hero("ehs", "ehs"), [villain], n_deals=args.arch_deals, seed=args.seed + 23, **kw)
        row = [name, ra.bb100, ra.ci95, rb.bb100, rb.ci95, gain, ci]
        if bp_c is not None:
            rc, rb2, gain_c, ci_c = compare_heroes(hero("ctl", "ctl"), hero("ehs", "ehs"), [villain], n_deals=args.arch_deals, seed=args.seed + 23, **kw)
            row += [rc.bb100, gain_c, ci_c]
        arch_rows.append(row)
        extra = f"   control ehs(seed1)-ehs: {row[8]:+.2f} +/-{row[9]:.2f}" if bp_c is not None else ""
        print(f"  vs {name:>7}: pot {ra.bb100:+7.2f} +/-{ra.ci95:.2f}   ehs {rb.bb100:+7.2f} +/-{rb.ci95:.2f}   gain pot-ehs {gain:+.2f} +/-{ci:.2f}{extra}")
    t_arch = time.perf_counter() - t0

    # ------------------------------------------------------------------- cost
    print("\nbucket computation cost (ms per new canonical form):")
    cost = bucket_cost(bk_e, bk_p)
    labels = sorted({k[0] for k in cost})
    for label in labels:
        print(f"  {label:<28} " + "  ".join(f"{name} {cost[(label, name)]:7.3f}" for name in ("flop", "turn", "river")))

    # --------------------------------------------------------------- markdown
    print("\n\n### markdown\n")
    print(f"Game `{spec_e.describe()}`; {args.iters:,} iterations each, seed {args.seed}; fit {args.fit_situations} situations/street.\n")
    print("| blueprint | bucketer | fit time | train time | infosets | canonical forms bucketed |")
    print("|---|---|---:|---:|---:|---:|")
    for tag, bk, info, fit_t in ((f"{base}_ehs", bk_e, info_e, fit_e), (f"{base}_pot", bk_p, info_p, fit_p)):
        if info is None:
            print(f"| {tag} | {type(bk).__name__} | - | (loaded) | - | - |")
        else:
            print(f"| {tag} | {type(bk).__name__} | {fit_t:.0f} s | {info['time']:.0f} s ({info['backend']}) | {info['infosets']:,} | {info['cache']:,} |")
    if info_c is not None:
        print(f"| {base}_ehs_seed{args.seed + 1} (control) | EquityBucketer | - | {info_c['time']:.0f} s ({info_c['backend']}) | {info_c['infosets']:,} | {info_c['cache']:,} |")
    print(f"\n(i) head-to-head, {args.h2h_deals} duplicate deals x 2 rotations ({t_h2h:.0f} s):\n")
    print("| hero | villain | bb/100 | 95% CI | off-map hero / villain |")
    print("|---|---|---:|---:|---:|")
    for h, v, bb, ci, fh, fv in rows:
        print(f"| {h} | {v} | {bb:+.2f} | +/-{ci:.2f} | {fh:.1%} / {fv:.1%} |")
    print(f"\n(ii) vs archetypes, {args.arch_deals} duplicate deals x 2 rotations each, identical deals for both heroes ({t_arch:.0f} s):\n")
    if bp_c is not None:
        print("| villain | potential bb/100 | E[HS] bb/100 | gain potential - E[HS] | control: E[HS] seed 1 bb/100 | control gain seed1 - seed0 |")
        print("|---|---:|---:|---:|---:|---:|")
        for name, a, ca, b, cb, g, cg, c, gc, cgc in arch_rows:
            print(f"| {name} | {a:+.2f} +/-{ca:.2f} | {b:+.2f} +/-{cb:.2f} | **{g:+.2f} +/-{cg:.2f}** | {c:+.2f} | {gc:+.2f} +/-{cgc:.2f} |")
    else:
        print("| villain | potential bb/100 | E[HS] bb/100 | gain potential - E[HS] |")
        print("|---|---:|---:|---:|")
        for name, a, ca, b, cb, g, cg in arch_rows:
            print(f"| {name} | {a:+.2f} +/-{ca:.2f} | {b:+.2f} +/-{cb:.2f} | **{g:+.2f} +/-{cg:.2f}** |")
    print("\nbucket computation, ms per new canonical form:\n")
    print("| bucketer / backend | flop | turn | river |")
    print("|---|---:|---:|---:|")
    for label in labels:
        print(f"| {label} | " + " | ".join(f"{cost[(label, name)]:.3f}" for name in ("flop", "turn", "river")) + " |")


if __name__ == "__main__":
    main()
