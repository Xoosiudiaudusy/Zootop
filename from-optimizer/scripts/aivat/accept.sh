#!/bin/bash
# acceptance: scripts/aivat_eval.py on the same 2000-hand duel log, 1 thread, core vs branch; out.jsonl and the summary compared
S=/tmp/claude-0/-home-user-Zootop/780badec-1fee-5bac-85c5-e13d71ebb617/scratchpad
B=/home/user/duel200
T=${1:-1}
for w in core3-wt av-wt; do
  rm -f /home/user/avbench/acc_${w}_t$T.jsonl
  cd /home/user/$w
  /home/user/venv312/bin/python /home/user/avbench/runmeas.py /home/user/avbench/acc_${w}_t$T /home/user/venv312/bin/python scripts/aivat_eval.py --duel $S/hands2000.jsonl --out /home/user/avbench/acc_${w}_t$T.jsonl \
    --blueprint $B/blueprint_base_s0.bin --buckets $B/buckets_base_s0.json --tables /home/user/data/bucket_tables_pot16 \
    --root-cache $S/root_base.npz --threads $T --no-luck
done
/home/user/venv312/bin/python - $T <<'PY'
import json, sys
T = sys.argv[1]
def rows(w):
    out = []
    for ln in open(f"/home/user/avbench/acc_{w}_t{T}.jsonl"):
        d = json.loads(ln); d.pop("seconds", None); out.append(d)
    return out
a, b = rows("core3-wt"), rows("av-wt")
print("per-hand lines (seconds removed):", "EQUAL" if a == b else "DIFFERENT", len(a), len(b))
bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
print("first differing:", bad[:5])
def summ(w):
    return [l for l in open(f"/home/user/avbench/acc_{w}_t{T}.txt") if "evaluator:" not in l and "hands, " not in l and "root table" not in l]
print("summary table (timing lines removed):", "EQUAL" if summ("core3-wt") == summ("av-wt") else "DIFFERENT")
for w in ("core3-wt", "av-wt"):
    ev = [l.strip() for l in open(f"/home/user/avbench/acc_{w}_t{T}.txt") if "evaluator:" in l]
    tm = [l.strip() for l in open(f"/home/user/avbench/acc_{w}_t{T}.time") if "Elapsed" in l or "Maximum resident" in l]
    print(w, ev, tm)
PY
