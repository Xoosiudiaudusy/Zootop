"""Pure-Python readers of the binary checkpoint and blueprint files (layouts: csrc/persist.h and
docs/backends.md, "Binary checkpoints and blueprints").

The C++ core reads and writes these files itself; this module is for machines without the core
(``scripts/export_json.py`` turns a file back into JSON with it) and is the tests' independent
check of the layout.  ``verify=True`` also checks the checksums (slow in Python: about 1 s per
10 MB).
"""
from __future__ import annotations

import array
import struct
import sys
from typing import Dict, Optional

from ..cfr.strategy import BlueprintStrategy

CHECKPOINT_MAGIC = b"NPCKPT01"
BLUEPRINT_MAGIC = b"NPBLUE01"
MT_WORDS = 624
_M64 = (1 << 64) - 1


def file_kind(path: str) -> str:
    """``checkpoint`` / ``blueprint`` (binary), ``json``, ``unknown`` or ``unreadable``."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return "unreadable"
    if head == CHECKPOINT_MAGIC:
        return "checkpoint"
    if head == BLUEPRINT_MAGIC:
        return "blueprint"
    s = head.lstrip(b" \t\r\n")
    if not head:
        return "unreadable"
    return "json" if s[:1] == b"{" else "unknown"


def stream_checksum(data) -> int:
    """csrc/binio.h StreamHash over ``data`` (little-endian 8-byte words, then the tail)."""
    h, count = 0x243F6A8885A308D3, 0
    n8 = len(data) - len(data) % 8
    words = array.array("Q")
    words.frombytes(bytes(data[:n8]))
    if sys.byteorder != "little":
        words.byteswap()
    for w in words:
        h ^= w
        h = (h * 0x9E3779B97F4A7C15) & _M64
        h ^= h >> 31
    count = n8
    fill = len(data) - n8
    if fill:
        w = int.from_bytes(bytes(data[n8:]), "little")
        h ^= w
        h = (h * 0x9E3779B97F4A7C15) & _M64
        h ^= h >> 31
    h ^= count + fill
    h = (h * 0xFF51AFD7ED558CCD) & _M64
    h ^= h >> 33
    return h


class _Reader:
    def __init__(self, data: bytes, path: str):
        self.b = memoryview(data)
        self.p = 0
        self.path = path

    def take(self, n: int) -> memoryview:
        if n < 0 or self.p + n > len(self.b):
            raise ValueError(f"unexpected end of file: {self.path}")
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))

    def u8(self) -> int:
        return self.unpack("<B")[0]

    def u16(self) -> int:
        return self.unpack("<H")[0]

    def u32(self) -> int:
        return self.unpack("<I")[0]

    def i32(self) -> int:
        return self.unpack("<i")[0]

    def u64(self) -> int:
        return self.unpack("<Q")[0]

    def i64(self) -> int:
        return self.unpack("<q")[0]

    def f64(self) -> float:
        return self.unpack("<d")[0]

    def str16(self) -> str:
        return bytes(self.take(self.u16())).decode("utf-8")

    def arr(self, code: str, n: int) -> array.array:
        a = array.array(code)
        a.frombytes(bytes(self.take(n * a.itemsize)))
        if sys.byteorder != "little":
            a.byteswap()
        return a

    def checksum(self, what: str, verify: bool) -> None:
        end = self.p
        stored = self.u64()
        if verify and stored != stream_checksum(self.b[:end]):
            raise ValueError(f"corrupt file ({what} checksum): {self.path}")


def _identity(r: _Reader) -> Optional[dict]:
    d = {"key_scheme": r.u32()}
    if not r.u8():
        return None
    for name in ("n_players", "stack_bb", "sb", "bb", "ante", "max_street", "max_raises_per_street", "n_buckets"):
        d[name] = r.i32()
    d["forbid_open_limp"] = bool(r.u8())
    d["allow_all_in"] = bool(r.u8())
    for name in ("preflop_fracs", "postflop_fracs"):
        d[name] = [r.f64() for _ in range(r.u16())]
    d["grid_names"] = [r.str16() for _ in range(r.u16())]
    d["bucketer_kind"] = r.str16()
    d["bucketer_n_buckets"] = r.i32()
    d["bucketer_samples"] = r.i32()
    d["bucketer_bins"] = r.i32()
    d["bucketer_fingerprint"] = r.u64()
    return d


def _open(path: str, magic: bytes) -> _Reader:
    with open(path, "rb") as f:
        data = f.read()
    r = _Reader(data, path)
    if bytes(r.take(8)) != magic:
        raise ValueError(f"{path} is not a {magic.decode()} file")
    version = r.u32()
    if version != 1:
        raise ValueError(f"{path}: format version {version} is not supported")
    r.u32()  # flags
    return r


def read_checkpoint_identity(path: str) -> Optional[dict]:
    """The game a binary checkpoint was saved for (spec, grid names, bucketer fingerprint)."""
    return _identity(_open(path, CHECKPOINT_MAGIC))


def read_checkpoint(path: str, verify: bool = False) -> dict:
    """A binary checkpoint as the dict of its JSON form: ``{"iteration", "linear", "nodes": {key:
    [names, regret, strategy_sum, visits]}[, "opp_nodes"], "backend", "threads", "rng_states"}``
    (nodes in file order)."""
    r = _open(path, CHECKPOINT_MAGIC)
    _identity(r)
    r.u8()  # trainer kind
    out: dict = {"iteration": r.i64(), "linear": bool(r.u8())}
    states = []
    for _ in range(r.u32()):
        words = list(r.arr("I", MT_WORDS))
        states.append(words + [r.i32()])
    names = [r.str16() for _ in range(r.u16())]
    for _ in range(r.u32()):
        tname = r.str16()
        nodes: Dict[str, list] = {}
        for _ in range(r.u64()):
            r.take(16)  # numeric key
            key = r.str16()
            n = r.u8()
            acts = [names[i] for i in bytes(r.take(n))]
            reg = list(r.arr("d", n))
            ss = list(r.arr("d", n))
            nodes[key] = [acts, reg, ss, r.i64()]
        out[tname] = nodes
    r.checksum("checkpoint", verify)
    out.update(backend="cpp", threads=len(states), rng_states=states)
    return out


def read_blueprint_arrays(path: str, verify: bool = False) -> dict:
    """The raw contents of a binary blueprint: header fields, names, and the arrays."""
    r = _open(path, BLUEPRINT_MAGIC)
    d: dict = {"identity": _identity(r), "iteration": r.i64(), "rounded": bool(r.u8())}
    packed = r.u8() == 1  # probabilities as u32 k (k / 100000, the double round(p, 5) gives), else f64
    d["codec_players"] = r.u32()
    d["names"] = [r.str16() for _ in range(r.u16())]
    n, m = r.u64(), r.u64()
    d["keys"] = r.arr("Q", 2 * n)
    d["offsets"] = r.arr("I", n + 1)
    d["ids"] = bytes(r.take(m))
    d["probs"] = array.array("d", (k / 100000 for k in r.arr("I", m))) if packed else r.arr("d", m)
    r.checksum("blueprint records", verify)
    d["key_strings"] = [r.str16() for _ in range(n)]
    r.checksum("blueprint key strings", verify)
    return d


def read_blueprint(path: str, verify: bool = False) -> BlueprintStrategy:
    """A binary blueprint as a ``BlueprintStrategy`` (keys in file order)."""
    d = read_blueprint_arrays(path, verify)
    names, off, ids, probs = d["names"], d["offsets"], d["ids"], d["probs"]
    table: Dict[str, tuple] = {}
    for i, key in enumerate(d["key_strings"]):
        a, b = off[i], off[i + 1]
        table[key] = ([names[j] for j in ids[a:b]], list(probs[a:b]))
    return BlueprintStrategy(table)
