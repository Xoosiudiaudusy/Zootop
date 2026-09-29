"""What a training run's files are: their iteration (from the header, without loading them), the checkpoint
``--resume`` continues from, and the run passport.

**Resume.**  ``resume_checkpoint`` picks ``checkpoint_<tag>.bin`` or ``.json`` by the larger iteration (a
``--json`` run writes both at the same iteration: then the binary), never by modification time: an old
``.json`` of 40M next to a ``.bin`` of 400M is not continued.  No checkpoint, or one whose iteration cannot
be read next to the other, is an error (``ResumeError``): training must not silently start from zero and
write over the blueprint of the tag.

**Passport.**  ``<file>.run.json`` next to every checkpoint and blueprint the trainer writes (and its
``.it<N>`` snapshots): the settings that change what training computes (``TRAIN_KEYS``: seed, batch,
Linear CFR and its schedule, pruning), how it ran (backend, threads, device), the code (git commit) and the
date, and the history of the run's segments across ``--resume``.  ``--resume`` refuses to continue with
other ``TRAIN_KEYS`` than the checkpoint's passport says.  ``scripts/blueprint_info.py`` prints it.
Files written before the passports existed have none (a checkpoint of the GPU trainer may have the older
``.gpu.json``: seed, batch, linear, linear_until).
"""
from __future__ import annotations

import datetime
import functools
import json
import os
import platform
import re
import subprocess
from typing import Dict, List, Optional

from .binfmt import BLUEPRINT_MAGIC, CHECKPOINT_MAGIC, _identity, _Reader

PASSPORT_SUFFIX = ".run.json"
# the settings that change the numbers training computes (a resumed run must keep them)
TRAIN_KEYS = ("seed", "batch", "linear", "linear_until", "prune_below", "prune_prob", "prune_after", "prune_relative",
              "prune_scale_t", "regret_floor")
_HEAD = 1 << 20  # the identity is a few hundred bytes; the header is read from this much of the file


class ResumeError(ValueError):
    pass


def file_iteration(path: str) -> Optional[int]:
    """The iteration a checkpoint or blueprint was saved at, from its first bytes; None if the file has none (a
    Python ``BlueprintStrategy`` JSON) or cannot be read."""
    try:
        with open(path, "rb") as f:
            head = f.read(_HEAD)
    except OSError:
        return None
    magic = bytes(head[:8])
    if magic in (CHECKPOINT_MAGIC, BLUEPRINT_MAGIC):
        try:
            r = _Reader(head, path)
            r.take(8)
            r.u32()  # version
            r.u32()  # flags
            _identity(r)
            if magic == CHECKPOINT_MAGIC:
                r.u8()  # trainer kind
            return int(r.i64())
        except Exception:
            return None
    m = re.match(rb'\s*\{\s*"iteration"\s*:\s*(-?\d+)', head)  # both trainers write "iteration" first
    return int(m.group(1)) if m else None


def file_header(path: str) -> Optional[dict]:
    """What the first bytes of a binary checkpoint / blueprint say, without reading the rest (a 1.4 GB checkpoint is
    not loaded): {"kind", "identity" (the game, None if unknown), "iteration", "infosets" (the first table's nodes of
    a checkpoint, the records of a blueprint; None if the header is longer than the part read)}; None for other files."""
    try:
        with open(path, "rb") as f:
            head = f.read(_HEAD)
    except OSError:
        return None
    magic = bytes(head[:8])
    if magic not in (CHECKPOINT_MAGIC, BLUEPRINT_MAGIC):
        return None
    r = _Reader(head, path)
    out: dict = {"kind": "checkpoint" if magic == CHECKPOINT_MAGIC else "blueprint", "identity": None, "iteration": None, "infosets": None}
    try:
        r.take(8)
        r.u32()
        r.u32()
        out["identity"] = _identity(r)
        if magic == CHECKPOINT_MAGIC:
            r.u8()  # trainer kind
            out["iteration"] = int(r.i64())
            r.u8()  # linear
            for _ in range(r.u32()):  # RNG streams: 624 words + an index each
                r.take(624 * 4 + 4)
            for _ in range(r.u16()):
                r.str16()
            if r.u32() > 0:
                r.str16()
                out["infosets"] = int(r.u64())
        else:
            out["iteration"] = int(r.i64())
            r.u8()  # rounded
            r.u8()  # packed
            r.u32()  # codec players
            for _ in range(r.u16()):
                r.str16()
            out["infosets"] = int(r.u64())
    except Exception:
        pass  # (a header longer than the part read: what was read so far)
    return out


def resume_checkpoint(data_dir: str, tag: str, binary: bool = True) -> str:
    """The checkpoint ``--resume`` continues from (see the module docstring); raises ResumeError."""
    stem = os.path.join(data_dir, f"checkpoint_{tag}")
    names = [stem + ".bin", stem + ".json"] if binary else [stem + ".json"]
    found = [p for p in names if os.path.exists(p)]
    if not found:
        raise ResumeError(f"--resume: no checkpoint to continue ({' or '.join(names)} does not exist); "
                          "drop --resume to start a new run (it writes over the files of this tag)")
    if len(found) == 1:
        return found[0]
    its = {p: file_iteration(p) for p in found}
    bad = [p for p, it in its.items() if it is None]
    if bad:
        raise ResumeError(f"--resume: both {found[0]} and {found[1]} exist and the iteration of {', '.join(bad)} cannot be read; "
                          "move away the one not to continue")
    a, b = found  # .bin, .json
    return a if its[a] >= its[b] else b


def passport_path(path: str) -> str:
    return path + PASSPORT_SUFFIX


def read_passport(path: str) -> Optional[dict]:
    """The passport of a checkpoint / blueprint file (None: none), or the older ``.gpu.json`` as a passport with
    its keys only."""
    for p, legacy in ((passport_path(path), False), (path + ".gpu.json", True)):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if legacy:
                d = {"legacy_gpu_json": True, "iteration": d.get("iteration"),
                     "train": {k: d[k] for k in ("seed", "batch", "linear", "linear_until") if k in d}}
            return d
    return None


def write_passport(path: str, passport: dict) -> None:
    """``<path>.run.json`` (text first, then a temporary file renamed over the old one)."""
    text = json.dumps(passport, indent=1, sort_keys=False)
    tmp = passport_path(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, passport_path(path))


def train_diff(saved: dict, now: dict) -> Dict[str, tuple]:
    """The TRAIN_KEYS on which the saved passport's settings and this run's differ: {key: (saved, now)}.  Keys
    the passport does not have are not compared (an older passport); the seed only matters with batches (the
    Philox deals; unbatched runs continue the RNG streams stored in the checkpoint)."""
    st = saved.get("train", {})
    out = {}
    for k in TRAIN_KEYS:
        if k not in st or k not in now:
            continue
        if k == "seed" and not (now.get("batch") or st.get("batch")):
            continue
        if st[k] != now[k]:
            out[k] = (st[k], now[k])
    return out


@functools.lru_cache(maxsize=None)
def code_version() -> str:
    """The git commit of the code (``+dirty`` with local changes), or ``unknown``."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        rev = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True,
                               text=True, timeout=10)
        return rev.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def now_text() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def new_passport(train: dict, run: dict, game: dict, iteration: int, history: Optional[List[dict]] = None) -> dict:
    return {"format": 1, "iteration": int(iteration), "written": now_text(), "code": code_version(),
            "host": platform.node(), "train": dict(train), "run": dict(run), "game": dict(game),
            "history": list(history or [])}
