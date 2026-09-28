"""Mean L1 change of the average strategy between saved blueprints (.bin or .json): the number
``scripts/train_blueprint.py`` prints at checkpoints ("mean L1 change vs previous"), for the snapshots of a
GPU run (the GPU trainer prints no L1) or any two blueprints of the same game and abstraction.

    python scripts/blueprint_l1.py data/blueprint_X.it100M.bin data/blueprint_X.it200M.bin [more: consecutive pairs]

Same formula as ``train_blueprint.strategy_change``: over the infoset keys both blueprints have with the
same action list, the mean of sum_a |p_a - q_a|.  Memory: the current blueprint's items as a Python list
(about 200 bytes per infoset) plus both C++ lookups."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negpluribus.fast.blueprint import load_blueprint  # noqa: E402


def l1_change(prev, cur_items):
    """(mean L1 over shared keys with equal actions, shared keys, keys of cur missing in prev, action mismatches)."""
    tot, n, missing, mismatch = 0.0, 0, 0, 0
    for key, names, probs in cur_items:
        entry = prev.get(key)
        if entry is None:
            missing += 1
            continue
        if list(entry[0]) != list(names):
            mismatch += 1
            continue
        tot += sum(abs(a - b) for a, b in zip(probs, entry[1]))
        n += 1
    return (tot / n if n else None), n, missing, mismatch


def main() -> None:
    paths = sys.argv[1:]
    if len(paths) < 2:
        sys.exit(__doc__)
    prev, prev_path = None, None
    for path in paths:
        t0 = time.time()
        bp = load_blueprint(path, keys=True)
        items = bp.items()
        print(f"{path}: iteration {bp.iteration:,}, {len(items):,} infosets, loaded in {time.time() - t0:.1f}s", flush=True)
        if prev is not None:
            t0 = time.time()
            mean, n, missing, mismatch = l1_change(prev, items)
            text = "n/a" if mean is None else f"{mean:.6f}"
            print(f"  mean L1 change vs {os.path.basename(prev_path)}: {text} over {n:,} shared keys "
                  f"({missing:,} keys new, {mismatch:,} with other actions; {time.time() - t0:.1f}s)", flush=True)
        prev, prev_path = bp.lookup, path
        del items


if __name__ == "__main__":
    main()
