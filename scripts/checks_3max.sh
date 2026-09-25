#!/usr/bin/env bash
# Morning checks of the two 3-max night runs (docs/scale_3max.md): iteration curve (checkpoints vs
# the 100M blueprint, head to head, both directions) and archetypes (two copies at the table),
# 100k deals per row, parallel processes with fixed seeds (seed 7 duels, seed 5 archetypes).
set -u
cd "$(dirname "$0")/.."
TS="$(date +%Y%m%d_%H%M)"
LOG="data/logs/checks_3max_${TS}.log"
JOBS="data/logs/jobs_3max_${TS}"
mkdir -p "$JOBS"
exec > >(tee -a "$LOG") 2>&1
MAXJ=${MAXJ:-12}
PIDS=(); NAMES=()
stamp() { echo "=== $* ($(date '+%d.%m %H:%M:%S'))"; }
job() { local name="$1"; shift; while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 2; done; "$@" > "$JOBS/$name.txt" 2>&1 & PIDS+=($!); NAMES+=("$name"); }
SPEC="--spec 3p_100bb_river"
NARROW="--preflop-fracs 0.5,1.0 --postflop-fracs 0.5,1.0 --max-raises 2"
WIDE="--preflop-fracs 0.5,1.0,3.0 --postflop-fracs 0.5,1.0,2.0,4.0 --max-raises 3"
curve() {  # label grid tag ext iters...
  local label="$1" grid="$2" tag="$3" ext="$4"; shift 4
  for it in "$@"; do
    job "CURVE_${label}_${it}_vs_100M" python scripts/compare_checkpoints.py $SPEC $grid \
      --a "data/blueprint_${tag}.it${it}.${ext}" --b "data/blueprint_${tag}.it100000000.${ext}" \
      --buckets "data/buckets_${tag}.json" --label-a "${label}_${it}" --label-b "${label}_100M" --deals 100000 --seed 7
  done
}
accept() {  # label grid tag bp
  local label="$1" grid="$2" tag="$3" bp="$4" opp
  for opp in tag nit maniac station lag gridrandom random; do
    job "ACCEPT_${label}_${opp}" python scripts/eval_archetypes.py $SPEC $grid --blueprint "$bp" \
      --buckets "data/buckets_${tag}.json" --deals 100000 --seed 5 --opponents "$opp"
  done
}
stamp "CHECKS 3-max: narrow (hunl3m100_pot16_s0) and wide (hunl3w100_pot16_s0)"
curve narrow "$NARROW" hunl3m100_pot16_s0 json 20000000 40000000 60000000 80000000
curve wide "$WIDE" hunl3w100_pot16_s0 bin 20000000 40000000 60000000 80000000
accept narrow "$NARROW" hunl3m100_pot16_s0 data/blueprint_hunl3m100_pot16_s0.it100000000.json
accept wide "$WIDE" hunl3w100_pot16_s0 data/blueprint_hunl3w100_pot16_s0.it100000000.bin
t0=$SECONDS; wait "${PIDS[@]}"; stamp "DONE in $((SECONDS - t0))s"
for n in "${NAMES[@]}"; do echo "--- $n"; grep -E "^  |Traceback|Error" "$JOBS/$n.txt"; done
