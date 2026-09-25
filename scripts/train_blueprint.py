"""Train a first blueprint with MCCFR on a reduced game and measure it.

    python scripts/train_blueprint.py                       # 2 players, 20bb, preflop+flop, ~2 min
    python scripts/train_blueprint.py --players 3 --iters 20000
    python scripts/train_blueprint.py --street preflop --stack 10 --iters 5000   # push/fold toy
    python scripts/train_blueprint.py --buckets-kind potential --backend cpp --iters 300000

Outputs go to data/: blueprint_<tag>.bin (average strategy), buckets_<tag>.json,
checkpoint_<tag>.bin (regrets, for resuming with --resume).  The tag carries the bucket kind
(``2p_20bb_flop`` for E[HS], ``2p_20bb_flop_pot`` for potential-aware), so the two blueprints
do not overwrite each other.  ``--backend cpp`` / ``--threads`` select the C++ core
(docs/backends.md); the default stays the Python reference (or NEGPLURIBUS_BACKEND).

Formats: with the C++ backend checkpoints and blueprints are binary files written straight from
the C++ table (docs/backends.md, "Binary checkpoints and blueprints"); ``--json`` also writes the
old JSON files (the same bytes as before), ``--format json`` writes only JSON.  The Python
reference trainer always writes JSON.  ``--resume`` takes checkpoint_<tag>.bin, else the old
checkpoint_<tag>.json; ``scripts/export_json.py`` turns any binary file into its JSON.  Every tool
reads both formats (eval_archetypes.py, compare_checkpoints.py, play_slumbot.py).

After training the script prints:
  * a few strategy rows you can sanity-check as a player (AA vs 72o in the SB…)
  * duplicate-deal results of the blueprint vs random / caller / archetypes
    in the *same* reduced game, in bb/100 with 95% CI.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.agents import make_agent  # noqa: E402
from negpluribus.agents.blueprint import BlueprintAgent  # noqa: E402
from negpluribus.cards import ALL_HOLE_CLASSES  # noqa: E402
from negpluribus.cfr.game import GameSpec  # noqa: E402
from negpluribus.cfr.mccfr import MCCFRTrainer  # noqa: E402
from negpluribus.abstraction import BUCKET_KINDS, bucketer_kind, load_bucketer  # noqa: E402
from negpluribus.engine import Street  # noqa: E402
from negpluribus.eval import duplicate_match  # noqa: E402
from negpluribus.fast.blueprint import tagged_path  # noqa: E402
from negpluribus.fast.power import CoreMeter, disable_power_throttling  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def strategy_change(prev, cur):
    """Mean L1 distance between two average strategies over the keys they share (None if no prev).
    The C++ trainer computes the same number, to the last bit, without the two dicts
    (``trainer.strategy_change(prev_blueprint)``)."""
    if prev is None:
        return None
    tot, n = 0.0, 0
    for key, (names, probs) in cur.table.items():
        entry = prev.table.get(key)
        if entry is None or entry[0] != names:
            continue
        tot += sum(abs(a - b) for a, b in zip(probs, entry[1]))
        n += 1
    return tot / n if n else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--stack", type=int, default=20, help="stack in bb")
    ap.add_argument("--street", choices=["preflop", "flop", "turn", "river"], default="flop", help="last street with betting")
    ap.add_argument("--preflop-fracs", default="1.0", help="comma-separated pot fractions of the preflop grid (all-in is always there)")
    ap.add_argument("--postflop-fracs", default="0.5,1.0", help="comma-separated pot fractions of the postflop grid")
    ap.add_argument("--max-raises", type=int, default=3, help="raises per street before only call/fold/all-in remain")
    ap.add_argument("--checkpoint-every", type=int, default=0, help="save checkpoint + strategy every N iterations and print the L1 change of the average strategy (0 = only at the end)")
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--buckets-kind", choices=list(BUCKET_KINDS), default="ehs",
                    help="postflop card abstraction: E[HS] cut points or potential-aware EMD clusters (docs/buckets.md)")
    ap.add_argument("--fit-situations", type=int, default=1200, help="random situations per street for the bucketer fit")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--no-linear", action="store_true", help="plain CFR weighting instead of Linear CFR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-deals", type=int, default=300)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--backend", choices=["python", "cpp"], default=None, help="traversal backend (default: NEGPLURIBUS_BACKEND or python)")
    ap.add_argument("--threads", type=int, default=None, help="threads of the C++ backend (default: all cores)")
    ap.add_argument("--bucket-cache", default=None,
                    help="C++ backend: bucket-cache capacities 'flop,turn,river' in entries (K/M suffixes; default "
                         "NEGPLURIBUS_BUCKET_CACHE or 4M,32M,4M; docs/backends.md 'Bounded bucket caches')")
    ap.add_argument("--format", choices=["bin", "json"], default="bin",
                    help="C++ backend: file format of checkpoints and blueprints (default bin; the Python backend always writes json)")
    ap.add_argument("--json", action="store_true", help="C++ backend: also write the JSON files next to the binary ones")
    ap.add_argument("--no-l1", action="store_true",
                    help="skip the L1-change diagnostic at checkpoints (C++ backend: frees the ~60 bytes per infoset it keeps between checkpoints)")
    ap.add_argument("--prune-below", type=float, default=0.0,
                    help="C++ backend: regret-based pruning (Pluribus) of actions whose accumulated regret is below -X "
                         "(stored units: bb x iteration weight; Pluribus 3e8; 0 = off, the default); changes the algorithm, judge by the result")
    ap.add_argument("--prune-prob", type=float, default=0.95, help="share of iterations that prune (Pluribus: 0.95)")
    ap.add_argument("--prune-after", type=int, default=0, help="iterations before pruning starts")
    ap.add_argument("--data-dir", default=DATA, help="where buckets / checkpoints / blueprints go (default data/)")
    args = ap.parse_args()
    no_throttle = disable_power_throttling()  # scheduling only, results unchanged

    spec = GameSpec(
        n_players=args.players,
        stack_bb=args.stack,
        max_street={"preflop": Street.PREFLOP, "flop": Street.FLOP, "turn": Street.TURN, "river": Street.RIVER}[args.street],
        preflop_fracs=tuple(float(x) for x in args.preflop_fracs.split(",") if x.strip()),
        postflop_fracs=tuple(float(x) for x in args.postflop_fracs.split(",") if x.strip()),
        max_raises_per_street=args.max_raises,
        n_buckets=args.buckets,
        bucket_kind=args.buckets_kind,
    )
    tag = args.tag or f"{spec.n_players}p_{spec.stack_bb}bb_{args.street}" + ("_pot" if args.buckets_kind == "potential" else "")
    data = args.data_dir
    os.makedirs(data, exist_ok=True)
    bk_path = os.path.join(data, f"buckets_{tag}.json")

    print("game:", spec.describe())
    bucketer = spec.make_bucketer()
    if spec.needs_buckets():
        if os.path.exists(bk_path):
            bucketer = load_bucketer(bk_path)
            if bucketer_kind(bucketer) != spec.bucket_kind:
                raise SystemExit(f"{bk_path} holds {bucketer_kind(bucketer)!r} buckets, the spec wants {spec.bucket_kind!r}; use --tag")
            print("buckets: loaded", bk_path)
        else:
            print(f"buckets: fitting {spec.bucket_kind} ({args.fit_situations} situations per street)…", flush=True)
            t0 = time.perf_counter()
            bucketer.fit(n_situations=args.fit_situations, seed=args.seed, verbose=True)
            bucketer.save(bk_path)
            print(f"  fitted in {time.perf_counter() - t0:.0f}s, saved {bk_path}")

    trainer = MCCFRTrainer(spec, bucketer, seed=args.seed, linear=not args.no_linear, backend=args.backend, threads=args.threads,
                           cache_caps=args.bucket_cache)
    cpp = trainer.backend == "cpp"
    if args.prune_below > 0:
        if not cpp:
            raise SystemExit("--prune-below needs --backend cpp")
        trainer.set_pruning(args.prune_below, args.prune_prob, args.prune_after)
        print(f"pruning: regret below -{args.prune_below:g}, {args.prune_prob:.0%} of iterations after {args.prune_after:,}")
    ext = ".bin" if cpp and args.format == "bin" else ".json"
    bp_path = os.path.join(data, f"blueprint_{tag}{ext}")
    ck_path = os.path.join(data, f"checkpoint_{tag}{ext}")
    also_json = cpp and args.json and ext == ".bin"  # JSON copies next to the binary files
    print(f"backend: {trainer.backend}" + (f" x{trainer.threads} threads, bucket cache {trainer.cache_caps}" if cpp else "")
          + f", files {ext[1:]}{' + json' if also_json else ''}, Windows power throttling {'off' if no_throttle else 'not changed'}")
    if args.resume:
        # C++ backend: checkpoint_<tag>.bin or .json, whichever was written last (the C++ trainer
        # reads both; a Python-backend run on the same tag writes JSON); Python backend: the JSON
        resume_from = tagged_path(data, "checkpoint", tag) if cpp else ck_path
        if os.path.exists(resume_from):
            t = time.perf_counter()
            trainer.load_checkpoint(resume_from)
            print(f"resumed from iteration {trainer.iteration:,} ({len(trainer.nodes):,} infosets, {resume_from}, "
                  f"{time.perf_counter() - t:.1f}s)")

    def it_path(path: str) -> str:  # blueprint_<tag>.it<N>.<ext>
        root, e = os.path.splitext(path)
        return f"{root}.it{trainer.iteration}{e}"

    def save_outputs(snapshot: bool, strat=None) -> None:
        """checkpoint + blueprint (+ the .it<N> copy of the blueprint) in the chosen format(s);
        Python backend: ``strat`` is the trainer's strategy() when the caller has it already"""
        if cpp:
            # JSON copies first, so that the binary files are the newest of each pair
            # (fast.blueprint.tagged_path, --resume)
            pairs = ([(ck_path[: -len(ext)] + ".json", bp_path[: -len(ext)] + ".json")] if also_json else []) + [(ck_path, bp_path)]
            for ck, bp in pairs:
                trainer.save_checkpoint(ck)
                trainer.save_blueprint(bp)
                if snapshot:
                    shutil.copyfile(bp, it_path(bp))  # the same bytes
        else:
            trainer.save_checkpoint(ck_path)
            s = strat if strat is not None else trainer.strategy()
            s.save(bp_path)
            if snapshot:
                s.save(it_path(bp_path))

    print(f"training {args.iters:,} iterations (each = {spec.n_players} traversals)…")
    t0 = time.perf_counter()
    cores = CoreMeter()  # busy cores per checkpoint interval: ~4 instead of ~15 means the run is throttled
    if args.checkpoint_every and args.checkpoint_every < args.iters:
        # long runs: train in chunks, save after each, and report how much the AVERAGE strategy
        # still moves between checkpoints (mean L1 distance over keys present in both).  With no
        # exact epsilon in multi-street games this is the plateau diagnostic; the paired
        # checkpoint-vs-checkpoint match in scripts/compare_checkpoints.py is the other one.
        # C++ backend: the previous average strategy is kept as a C++ lookup (full precision, like
        # the old dict) and the change is computed in C++: the same number without the two dicts.
        prev = None
        done = 0
        while done < args.iters:
            step = min(args.checkpoint_every, args.iters - done)
            trainer.train(step, log_every=max(1, step // 2))
            done += step
            if cpp:
                change = trainer.strategy_change(prev) if prev is not None else None
                prev = None
                n_infosets = trainer.n_nodes
            else:
                cur = trainer.strategy()
                change = strategy_change(prev, cur)
                n_infosets = len(cur)
            cache = ""
            if hasattr(trainer, "cache_stats"):
                cache = "; bucket cache " + " ".join(
                    f"{s[0]}:{v['size']:,}/{v['capacity']:,} (miss {v['computes']:,}, evict {v['evictions']:,})"
                    for s, v in trainer.cache_stats().items() if v["capacity"])
            print(f"  checkpoint {trainer.iteration:,}: {n_infosets:,} infosets, mean L1 change vs previous "
                  f"{'n/a' if change is None else f'{change:.4f}'}, {time.perf_counter() - t0:.0f}s, {cores.lap():.1f} busy cores{cache}", flush=True)
            t_save = time.perf_counter()
            save_outputs(snapshot=True, strat=None if cpp else cur)
            if not args.no_l1:
                prev = trainer.blueprint(rounded=False) if cpp else cur
            print(f"  saved in {time.perf_counter() - t_save:.1f}s", flush=True)
        prev = None
    else:
        trainer.train(args.iters, log_every=max(1, args.iters // 10))
    print(f"done in {time.perf_counter() - t0:.0f}s, {len(trainer.nodes):,} infosets")
    # the agent's strategy: the average strategy at full precision, as trainer.strategy() had it
    # (C++ backend: the same floats from a C++ lookup, no dict)
    if cpp:
        save_outputs(snapshot=False)
        strat = trainer.blueprint(rounded=False)
    else:
        strat = trainer.strategy()
        save_outputs(snapshot=False, strat=strat)
    print("saved", bp_path)

    # ------------------------------------------------------------ sanity rows
    first_pos = "BTN/SB" if spec.n_players == 2 else "BTN"
    print(f"\nopening decision ({first_pos}, nobody acted yet) for a few hands:")
    for cls in ["AA", "AKs", "TT", "A5s", "KJo", "98s", "72o", "J3o"]:
        key = f"P|{first_pos}|{spec.n_players}|b{ALL_HOLE_CLASSES.index(cls)}|"
        row = trainer.nodes.get(key)
        if row:
            probs = "  ".join(f"{a}:{p:.2f}" for a, p in zip(row.actions, row.average_strategy()))
            print(f"  {cls:>4}: {probs}")
    prefix = f"P|{first_pos}|{spec.n_players}|"
    if cpp:  # counted in C++: no list of every key string
        n_open = trainer.count_keys(prefix, "|")
    else:
        n_open = sum(1 for k in trainer.nodes if k.startswith(prefix) and k.endswith("|"))
    print(f"  ({n_open} of 169 starting hands reached in that spot)")

    # -------------------------------------------------------------- measure
    if args.eval_deals <= 0:
        return  # no evaluation requested (long runs evaluate separately: scripts/eval_archetypes.py)
    print(f"\nduplicate-deal evaluation in the same game ({args.eval_deals} deals x {spec.n_players} rotations):")
    line_ups = {
        "random": ["random"] * (spec.n_players - 1),
        "caller": ["caller"] * (spec.n_players - 1),
        "tag": ["tag"] * (spec.n_players - 1),
        "maniac": ["maniac"] * (spec.n_players - 1),
        "nit": ["nit"] * (spec.n_players - 1),
    }
    for label, vil in line_ups.items():
        hero = BlueprintAgent(strat, bucketer, spec.grid, seed=1)
        villains = [make_agent(v, seed=10 + i, label=f"{v}#{i + 1}") for i, v in enumerate(vil)]
        res = duplicate_match(
            hero, villains, n_deals=args.eval_deals, seed=args.seed + 7,
            sb=spec.sb, bb=spec.bb, stack_bb=spec.stack_bb, max_street=spec.max_street,
        )
        print(f"  vs {label:>7}: {res.bb100:+8.1f} bb/100  (95% CI +/-{res.ci95:.1f})   off-map decisions: {hero.fallback_rate:.1%}")


if __name__ == "__main__":
    main()
