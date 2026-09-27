#!/usr/bin/env bash
# The 64-bucket duel series (docs/scale_4street.md, "Blueprint на 64 корзинах"): exact features vs
# Monte-Carlo features, seed-matched and crossed, with the same-kind pairs as the seed-noise control and
# the production potential-16 blueprint as the reference.  Duplicate deals 0..24999 (--seed 0) for
# every pairing, both agents on bucket tables; hand logs for AIVAT (known player = the hero).
set -uo pipefail
cd "$(dirname "$0")/.."
P3=.claude/worktrees/agent-ae78f3fcf39ad9442/data
OUT=data/duels64
DEALS=${DEALS:-25000}
bp() { case $1 in pot16) echo "$P3/blueprint_hunl200w3_pot16_s0.bin";; *) echo "data/blueprint_hunl200w3_$1.bin";; esac; }
bk() { case $1 in pot16) echo "$P3/buckets_hunl200w3_pot16_s0.json";; *) echo "data/buckets_hunl200w3_$1.json";; esac; }
[ -n "$(ls data/bucket_tables/buckets_potential_16_* 2>/dev/null)" ] || \
  "${PY:-python}" -B scripts/build_bucket_table.py --buckets "$(bk pot16)" --out data/bucket_tables --threads 8
for pair in pot64x_s0:pot64_s0 pot64x_s1:pot64_s1 pot64x_s0:pot64_s1 pot64x_s1:pot64_s0 pot64_s0:pot64_s1 pot64x_s0:pot64x_s1 pot64_s0:pot16 pot64x_s0:pot16; do
  h=${pair%%:*}; v=${pair##*:}
  [ -s "$OUT/check_${h}_vs_${v}.txt" ] && grep -q "vs blueprint" "$OUT/check_${h}_vs_${v}.txt" && continue
  echo "=== $h vs $v start $(date +%FT%T)" >> "$OUT/status.txt"
  "${PY:-python}" -B scripts/eval_archetypes.py --spec 2p_200bb_river --preflop-fracs 0.5,1.0,3.0 --postflop-fracs 0.5,1.0,2.0,4.0 \
    --blueprint "$(bp $h)" --buckets "$(bk $h)" --opponent-blueprint "$(bp $v)" --opponent-buckets "$(bk $v)" \
    --tables data/bucket_tables --opponents blueprint --deals "$DEALS" --seed 0 --progress 5000 \
    --log "$OUT/deals_${h}_vs_${v}.jsonl" --log-hands "$OUT/hands_${h}_vs_${v}.jsonl" > "$OUT/check_${h}_vs_${v}.txt" 2>&1
  echo "=== $h vs $v end $(date +%FT%T) exit $?" >> "$OUT/status.txt"
done
echo "=== all done $(date +%FT%T)" >> "$OUT/status.txt"
