#!/usr/bin/env bash
# AIVAT for the 64-bucket duel series (scripts/duels64.sh): known player = the hero of each pairing, its
# blueprint / buckets on the bucket tables, root table per blueprint (data/aivat/root_v1_<hero>.npz).
# LIMIT hands per pairing (the evaluator resumes from its --out, so a later run with a larger LIMIT
# only scores the new hands).  Output: data/aivat/duels64_<pair>_aivat.{jsonl,txt}
set -uo pipefail
cd "$(dirname "$0")/.."
P3=.claude/worktrees/agent-ae78f3fcf39ad9442/data
LIMIT=${LIMIT:-20000}
THREADS=${THREADS:-4}
bp() { case $1 in pot16) echo "$P3/blueprint_hunl200w3_pot16_s0.bin";; *) echo "data/blueprint_hunl200w3_$1.bin";; esac; }
bk() { case $1 in pot16) echo "$P3/buckets_hunl200w3_pot16_s0.json";; *) echo "data/buckets_hunl200w3_$1.json";; esac; }
for pair in ${*:-pot64x_s0_vs_pot64_s0 pot64x_s1_vs_pot64_s1 pot64x_s0_vs_pot64_s1 pot64x_s1_vs_pot64_s0 pot64_s0_vs_pot64_s1 pot64x_s0_vs_pot64x_s1 pot64_s0_vs_pot16 pot64x_s0_vs_pot16}; do
  h=${pair%%_vs_*}
  hands=data/duels64/hands_$pair.jsonl
  [ "$(wc -l < "$hands" 2>/dev/null || echo 0)" -ge 50000 ] || { echo "skip $pair: incomplete"; continue; }
  echo "=== aivat $pair start $(date +%FT%T)" >> data/duels64/status.txt
  "${PY:-python}" -B scripts/aivat_eval.py --duel "$hands" --known hero --out "data/aivat/duels64_${pair}_aivat.jsonl" \
    --blueprint "$(bp $h)" --buckets "$(bk $h)" --tables data/bucket_tables --root-cache "data/aivat/root_v1_$h.npz" \
    --threads "$THREADS" --limit "$LIMIT" > "data/aivat/duels64_${pair}_aivat.txt" 2>&1
  echo "=== aivat $pair end $(date +%FT%T) exit $?: $(grep -a 'AIVAT v1' "data/aivat/duels64_${pair}_aivat.txt" | head -1)" >> data/duels64/status.txt
done
