"""The web table (webapp/): its bot list against the files it names, and its HTTP API against other web pages.

  * webapp/bots.json: "Game parameters must match the training" - every bot whose binary blueprint is present is
    checked against the game stored in the blueprint's header (stack, bet grid, raises, heads-up); and the training
    length its note announces ("N млн итераций") against the header's iteration.  M9 (defect hunter; confirmed on the
    PC's data 29.09): pot64x_s1 says 400M, the file is at 600,047,616; pot64x_s0 says 400M, the file is at
    1,600,063,488 (data/blueprint_<tag>.bin is the LATEST version of a tag; a fixed version is an .it<N> file).
  * L5 (defect hunter, low): the server checks no Origin / Host, so any page open in the user's browser can POST to
    http://127.0.0.1:8777/api/* (reset the statistics, change settings, act for the hero).

The bots test reads only the blueprints' headers and skips without data (NEGPLURIBUS_TEST_DATA points it at a data
folder elsewhere); the server test runs the real handler on a free port with no bot loaded.
"""
from __future__ import annotations

import http.client
import importlib.util
import json
import os
import re
import threading

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.environ.get("NEGPLURIBUS_TEST_DATA") or os.path.join(ROOT, "data")
BOTS_JSON = os.path.join(ROOT, "webapp", "bots.json")


def _resolve(p: str) -> str:
    if os.path.isabs(p):
        return p
    parts = p.replace("\\", "/").split("/")
    return os.path.join(DATA, *parts[1:]) if parts[0] == "data" else os.path.join(ROOT, *parts)


def _header(path: str):
    """(game identity, iteration) from a binary blueprint's header (layout: negpluribus/fast/binfmt.py)."""
    from negpluribus.fast import binfmt

    with open(path, "rb") as f:
        head = f.read(1 << 16)
    r = binfmt._Reader(head, path)
    assert bytes(r.take(8)) == binfmt.BLUEPRINT_MAGIC
    assert r.u32() == 1
    r.u32()
    ident = binfmt._identity(r)
    return ident, r.i64()


def _present_binary_bots():
    with open(BOTS_JSON, encoding="utf-8-sig") as f:
        bots = json.load(f)["bots"]
    out = [b for b in bots if b["blueprint"].endswith(".bin") and os.path.exists(_resolve(b["blueprint"]))]
    if not out:
        pytest.skip(f"no binary blueprint of webapp/bots.json under {DATA}")
    return out


def test_bots_json_game_parameters_match_the_blueprints():
    for b in _present_binary_bots():
        ident, _ = _header(_resolve(b["blueprint"]))
        assert ident is not None, b["id"]
        assert ident["n_players"] == 2, b["id"]
        assert ident["stack_bb"] == int(b.get("stack_bb", 200)), b["id"]
        assert ident["preflop_fracs"] == [float(x) for x in b["preflop"]], b["id"]
        assert ident["postflop_fracs"] == [float(x) for x in b["postflop"]], b["id"]
        assert ident["max_raises_per_street"] == int(b["max_raises"]), b["id"]


@pytest.mark.xfail(strict=True, reason="M9: webapp/bots.json notes say 400 млн for pot64x_s1 / pot64x_s0 while their files "
                                        "(the tags' latest versions) are at 600M / 1.6B iterations")
def test_bots_json_iteration_labels_match_the_blueprints():
    wrong = []
    for b in _present_binary_bots():
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(млн|M)\b", b.get("note", "") + " " + b.get("label", ""))
        if not m:
            continue
        said = float(m.group(1).replace(",", ".")) * 1e6
        _, it = _header(_resolve(b["blueprint"]))
        pinned = re.search(r"\.it(\d+)\.bin$", b["blueprint"])
        if abs(it - said) > 0.01 * said or (pinned and int(pinned.group(1)) != it):
            wrong.append((b["id"], int(said), it))
    assert not wrong, wrong


# ------------------------------------------------------------------------------------------------ the HTTP API
@pytest.fixture()
def server():
    spec = importlib.util.spec_from_file_location("webapp_server_under_test", os.path.join(ROOT, "webapp", "server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # reads the bot list (file checks only); no bot is loaded, no pid file is written
    srv = mod.QuietServer(("127.0.0.1", 0), mod.Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)


def _post(port, path, origin=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json", "Content-Length": "2"}
    if origin:
        headers["Origin"] = origin
    conn.request("POST", path, body=b"{}", headers=headers)
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r.status, dict(r.getheaders()), body


def test_the_api_answers_its_own_page(server):
    status, headers, body = _post(server, "/api/reset", origin=f"http://127.0.0.1:{server}")
    assert status in (200, 400) and "application/json" in headers.get("Content-Type", "")  # 400: no bot selected yet
    assert "Access-Control-Allow-Origin" not in headers


@pytest.mark.xfail(strict=True, reason="L5: webapp/server.py do_POST checks no Origin / Host: a foreign page can drive the "
                                        "table (it answers 400 here only because no bot is selected)")
def test_the_api_refuses_a_foreign_page(server):
    for path in ("/api/reset", "/api/settings", "/api/new"):
        status, _, _ = _post(server, path, origin="http://evil.example")
        assert status == 403, (path, status)
