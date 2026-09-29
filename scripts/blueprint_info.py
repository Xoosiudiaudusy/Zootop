"""What a checkpoint or blueprint file is and how it was trained: its kind, iteration and game (from the file) and
its run passport (<file>.run.json: seed, batch, Linear CFR schedule, pruning, backend, threads, device, code commit,
dates, and the segments of the run across --resume).

    python scripts/blueprint_info.py data/blueprint_X.bin [data/checkpoint_X.bin ...]
    python scripts/blueprint_info.py data/blueprint_X.bin --json     # the passport as JSON

Files written before the passports existed (29.09.2026) have none; a GPU checkpoint of that time may have the older
.gpu.json (seed, batch, linear, linear_until), which is shown instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from negpluribus.fast.binfmt import file_kind  # noqa: E402
from negpluribus.fast.runinfo import file_header, file_iteration, read_passport  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true", help="print the passports as JSON")
    args = ap.parse_args()
    for path in args.files:
        kind = file_kind(path) if os.path.exists(path) else "missing"
        head = file_header(path) or {}  # (the header only: a large checkpoint is not read)
        it = head.get("iteration") if head else file_iteration(path)
        pp = read_passport(path)
        if args.json:
            print(json.dumps({"file": path, "kind": kind, "iteration": it, "infosets": head.get("infosets"),
                              "identity": head.get("identity"), "passport": pp}, indent=1))
            continue
        print(f"{path}")
        n_inf = head.get("infosets")
        print(f"  kind {kind}, iteration {'?' if it is None else f'{it:,}'}, {'?' if n_inf is None else f'{n_inf:,}'} infosets, "
              f"{os.path.getsize(path):,} bytes" if os.path.exists(path) else "  (file missing)")
        ident = head.get("identity")
        if ident:
            print(f"  game: {ident['n_players']} players, {ident['stack_bb']}bb, streets to {ident['max_street']}, "
                  f"preflop {ident['preflop_fracs']}, postflop {ident['postflop_fracs']}, raises {ident['max_raises_per_street']}, "
                  f"buckets {ident['bucketer_kind']} {ident['bucketer_n_buckets']} (fingerprint {ident['bucketer_fingerprint']:x})")
        if pp is None:
            print("  passport: none (written before passports, 29.09.2026)")
            continue
        if pp.get("legacy_gpu_json"):
            print(f"  passport: only the old .gpu.json: {pp['train']} at iteration {pp.get('iteration')}")
            continue
        if pp.get("iteration") is not None and it is not None and pp["iteration"] != it:
            print(f"  WARNING: the passport is of iteration {pp['iteration']:,}, the file of {it:,} (written by something else?)")
        print(f"  written {pp.get('written')} by code {pp.get('code')} on {pp.get('host')}")
        print(f"  training: {', '.join(f'{k}={v}' for k, v in pp.get('train', {}).items())}")
        g = pp.get("game", {})
        if g:
            print(f"  game (flags): {', '.join(f'{k}={v}' for k, v in g.items())}")
        for seg in pp.get("history", []):
            print(f"  segment {seg.get('from_iteration', 0):,} -> {seg.get('to_iteration', 0):,}: {seg.get('backend')} x{seg.get('threads')} "
                  f"threads, {seg.get('device')}, started {seg.get('started')}"
                  + ("" if seg.get("train") == pp.get("train") else f", settings {seg.get('train')}"))


if __name__ == "__main__":
    main()
