"""Convert between the binary files of the C++ trainer and the JSON files of the Python code.

    python scripts/export_json.py data/checkpoint_X.bin                 # -> data/checkpoint_X.json
    python scripts/export_json.py data/blueprint_X.bin out.json
    python scripts/export_json.py data/blueprint_X.json data/blueprint_X.bin   # JSON blueprint -> binary

The kind of the input is read from its first bytes.  Binary checkpoint -> the JSON checkpoint of
the C++ trainer (loadable by both trainers); binary blueprint -> the JSON of BlueprintStrategy.save
(probabilities rounded to 5 decimals, as in every blueprint JSON); JSON blueprint -> binary
blueprint (the game identity is unknown then: the lookup does not need it).  Keys come out in the
binary file's order (sorted by numeric key), values exactly as stored.  Without the C++ core
the pure-Python reader (negpluribus/fast/binfmt.py) does the binary -> JSON direction.
A JSON checkpoint needs no conversion: the C++ trainer resumes from it directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.fast import core  # noqa: E402
from negpluribus.fast.blueprint import file_kind  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="binary checkpoint / blueprint, or a JSON blueprint")
    ap.add_argument("dst", nargs="?", default=None, help="output (default: src with .json, or .bin for a JSON blueprint)")
    ap.add_argument("--python", action="store_true", help="use the pure-Python reader even when the C++ core is built")
    args = ap.parse_args()

    kind = file_kind(args.src)
    root = os.path.splitext(args.src)[0]
    c = None if args.python else core()
    t = time.perf_counter()
    if kind == "checkpoint":
        dst = args.dst or root + ".json"
        if c is not None:
            c.checkpoint_bin_to_json(args.src, dst)
        else:
            from negpluribus.fast.binfmt import read_checkpoint

            with open(dst, "w", encoding="utf-8") as f:
                json.dump(read_checkpoint(args.src), f)
    elif kind == "blueprint":
        dst = args.dst or root + ".json"
        if c is not None:
            c.BlueprintTable.load(args.src, keys=True).save_json(dst)
        else:
            from negpluribus.fast.binfmt import read_blueprint

            read_blueprint(args.src).save(dst)
    elif kind == "json":
        dst = args.dst or root + ".bin"
        if c is None:
            print("JSON -> binary needs the C++ core (python scripts/build_fast.py)", file=sys.stderr)
            return 2
        if dst.lower().endswith(".json"):
            print("the output of a JSON input is a binary blueprint: give it another extension", file=sys.stderr)
            return 2
        with open(args.src, "r", encoding="utf-8") as f:
            head = f.read(64)
        if not head.lstrip().startswith('{"table"'):
            print(f"{args.src}: only JSON blueprints convert to binary (a JSON checkpoint resumes as it is)", file=sys.stderr)
            return 2
        c.BlueprintTable.load(args.src, keys=True).save(dst)
    else:
        print(f"{args.src}: not a checkpoint or blueprint ({kind})", file=sys.stderr)
        return 2
    print(f"{args.src} ({kind}, {os.path.getsize(args.src):,} bytes) -> {dst} ({os.path.getsize(dst):,} bytes) "
          f"in {time.perf_counter() - t:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
