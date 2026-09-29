"""train_blueprint.py --resume and the run passport (negpluribus/fast/runinfo.py, scripts/blueprint_info.py).

H1: --resume without a checkpoint is an error (it used to start from zero and write over the tag's blueprint); the
checkpoint is chosen by the larger iteration, not the newer file; an unreadable one next to the other is an error.
M1: the final iteration always has its .it<N> snapshot (the GPU path skipped it).
M3: --resume refuses other seed / batch / Linear CFR / pruning settings than the checkpoint's passport (and the
checkpoint's own linear flag no longer overrides --no-linear silently); --resume-override continues and the
passport keeps the segments.  A preflop toy game, one thread, a temporary data directory.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from negpluribus import fast

core = fast.core()
pytestmark = pytest.mark.skipif(core is None, reason="C++ core not built")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(ROOT, "scripts", "train_blueprint.py")


def train(tmp_path, *extra, ok=True):
    cmd = [sys.executable, "-B", SCRIPT, "--players", "2", "--stack", "10", "--street", "preflop", "--backend", "cpp",
           "--threads", "1", "--eval-deals", "0", "--tag", "t", "--data-dir", str(tmp_path), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    assert (p.returncode == 0) == ok, out[-3000:]
    return out


def test_resume_without_checkpoint_is_an_error(tmp_path):
    out = train(tmp_path, "--resume", "--iters", "100", ok=False)
    assert "no checkpoint to continue" in out
    assert not any(n.startswith(("blueprint_", "checkpoint_")) for n in os.listdir(tmp_path))


def test_resume_takes_the_larger_iteration_not_the_newer_file(tmp_path):
    from negpluribus.fast.runinfo import file_iteration

    train(tmp_path, "--iters", "1000", "--format", "json")          # checkpoint_t.json at 1000
    old_json = (tmp_path / "checkpoint_t.json").read_bytes()
    train(tmp_path, "--iters", "3000", "--overwrite")                 # checkpoint_t.bin at 3000 (a new run)
    (tmp_path / "checkpoint_t.json").write_bytes(old_json)           # the old JSON, now the newer file
    assert file_iteration(str(tmp_path / "checkpoint_t.bin")) == 3000
    assert file_iteration(str(tmp_path / "checkpoint_t.json")) == 1000
    out = train(tmp_path, "--resume", "--iters", "100")
    assert "resumed from iteration 3,000" in out
    (tmp_path / "checkpoint_t.json").write_text("not a checkpoint")  # unreadable next to the binary: refuse
    out = train(tmp_path, "--resume", "--iters", "100", ok=False)
    assert "cannot be read" in out


def test_blueprint_path_by_iteration(tmp_path):
    from negpluribus.fast.blueprint import tagged_path

    train(tmp_path, "--iters", "2000")
    shutil.copyfile(tmp_path / "blueprint_t.bin", tmp_path / "keep.bin")
    train(tmp_path, "--iters", "500", "--format", "json", "--tag", "u")
    # a binary blueprint of 2000 and a newer one of 500 converted to JSON by the checkpoint of 500: take the 2000
    train(tmp_path, "--iters", "500", "--overwrite")
    os.replace(tmp_path / "keep.bin", tmp_path / "blueprint_t.bin")
    j = subprocess.run([sys.executable, "-B", os.path.join(ROOT, "scripts", "export_json.py"), str(tmp_path / "checkpoint_t.bin"),
                        str(tmp_path / "blueprint_t.json")], capture_output=True, text=True, timeout=600)
    assert j.returncode == 0, j.stderr
    assert tagged_path(str(tmp_path), "blueprint", "t").endswith(".bin")


@pytest.mark.parametrize("flags", [["--linear-until", "500"], ["--batch", "32"], ["--no-linear"], ["--seed", "5", "--batch", "64"],
                                   ["--prune-below", "1e9"]])
def test_resume_refuses_other_training_settings(tmp_path, flags):
    base = ["--batch", "64"] if "--seed" in flags else []
    train(tmp_path, "--iters", "1000", *base)
    before = (tmp_path / "checkpoint_t.bin").read_bytes()
    out = train(tmp_path, "--resume", "--iters", "100", *flags, ok=False)
    assert "was trained with other settings" in out, out[-2000:]
    assert (tmp_path / "checkpoint_t.bin").read_bytes() == before


def test_checkpoint_linear_flag_is_checked_without_a_passport(tmp_path):
    train(tmp_path, "--iters", "1000")
    os.remove(tmp_path / "checkpoint_t.bin.run.json")
    out = train(tmp_path, "--resume", "--iters", "100", "--no-linear", ok=False)
    assert "linear: True (checkpoint) vs False (now)" in out


def test_override_continues_and_the_passport_keeps_the_segments(tmp_path):
    train(tmp_path, "--iters", "1000")
    out = train(tmp_path, "--resume", "--iters", "500", "--linear-until", "800", "--resume-override")
    assert "--resume-override" in out and "resumed from iteration 1,000" in out
    with open(tmp_path / "checkpoint_t.bin.run.json", encoding="utf-8") as f:
        pp = json.load(f)
    assert pp["iteration"] == 1500 and pp["train"]["linear_until"] == 800 and pp["train"]["seed"] == 0
    assert [(s["from_iteration"], s["to_iteration"]) for s in pp["history"]] == [(0, 1000), (1000, 1500)]
    assert pp["history"][0]["train"]["linear_until"] == 0 and pp["code"]
    info = subprocess.run([sys.executable, "-B", os.path.join(ROOT, "scripts", "blueprint_info.py"), str(tmp_path / "blueprint_t.bin")],
                          capture_output=True, text=True, timeout=600)
    assert info.returncode == 0, info.stderr
    assert "iteration 1,500" in info.stdout and "linear_until=800" in info.stdout and "segment 1,000 -> 1,500" in info.stdout


@pytest.mark.parametrize("gpu", [False, True])
def test_final_iteration_has_its_snapshot(tmp_path, gpu):
    extra = ["--batch", "64", "--gpu", "0", "--gpu-emulate"] if gpu else []
    train(tmp_path, "--iters", "1024", "--checkpoint-every", "512", *extra)
    for it in (512, 1024):
        assert (tmp_path / f"blueprint_t.it{it}.bin").exists(), it
        assert (tmp_path / f"blueprint_t.it{it}.bin.run.json").exists(), it
    out = train(tmp_path, "--resume", "--iters", "512", "--checkpoint-every", "512", *extra)  # the settings match: continues
    assert "resumed from iteration 1,024" in out and (tmp_path / "blueprint_t.it1536.bin").exists()


def test_a_new_run_does_not_overwrite_a_tags_checkpoints(tmp_path):
    train(tmp_path, "--iters", "1000")
    before = (tmp_path / "checkpoint_t.bin").read_bytes()
    out = train(tmp_path, "--iters", "100", ok=False)
    assert "--overwrite" in out and (tmp_path / "checkpoint_t.bin").read_bytes() == before
    train(tmp_path, "--iters", "100", "--overwrite")
    assert (tmp_path / "checkpoint_t.bin").read_bytes() != before


def test_resume_checks_come_before_the_buckets(tmp_path):
    base = ["--street", "flop", "--fit-situations", "40"]
    train(tmp_path, "--iters", "200", *base)
    os.remove(tmp_path / "buckets_t.json")
    out = train(tmp_path, "--resume", "--iters", "100", "--linear-until", "50", *base, ok=False)
    assert "was trained with other settings" in out and "buckets:" not in out
    assert not (tmp_path / "buckets_t.json").exists()  # refused before fitting new buckets


def test_blueprint_json_iteration_from_its_passport(tmp_path):
    from negpluribus.fast.blueprint import tagged_path

    train(tmp_path, "--iters", "2000", "--format", "json")               # blueprint_t.json: no iteration in the file
    train(tmp_path, "--iters", "500", "--overwrite")                     # blueprint_t.bin at 500, newer
    assert tagged_path(str(tmp_path), "blueprint", "t").endswith(".json")  # 2000 by the JSON's passport


def test_blueprint_info_reads_headers(tmp_path):
    from negpluribus.fast.runinfo import file_header

    train(tmp_path, "--iters", "1000")
    ck, bp = file_header(str(tmp_path / "checkpoint_t.bin")), file_header(str(tmp_path / "blueprint_t.bin"))
    assert ck["kind"] == "checkpoint" and ck["iteration"] == 1000 and ck["infosets"] > 0 and ck["identity"]["n_players"] == 2
    assert bp["kind"] == "blueprint" and bp["iteration"] == 1000 and bp["infosets"] == ck["infosets"]
    info = subprocess.run([sys.executable, "-B", os.path.join(ROOT, "scripts", "blueprint_info.py"), str(tmp_path / "blueprint_t.bin")],
                          capture_output=True, text=True, timeout=600)
    assert f"{bp['infosets']:,} infosets" in info.stdout and "game: 2 players" in info.stdout, info.stdout
