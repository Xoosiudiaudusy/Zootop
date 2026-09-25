#!/usr/bin/env bash
# Bucket experiments, restarted after the bucket speed-ups (24.09.2026 17:45; docs/scale_4street.md,
# "Остановка и перезапуск").  The same experiment, recipes and seeds as scripts/overnight_buckets.sh.
# Only speed changes:
#   * bigger bucket caches (results are bit-identical whatever the caps: every cached value is a pure
#     function of the canonical form; tests/test_backends.py::test_bounded_bucket_cache_is_bit_identical);
#   * evaluations run as parallel processes: every row / duel is its own process with the same fixed
#     seed as in the sequential script, so each number is the one the sequential run would print;
#   * pot_s1 continues from its 20M checkpoint of the stopped run (regrets, strategy sums, iteration
#     counter and the RNG stream of every thread are restored), 30M more = 50M as in the recipe.
# Usage: scripts/overnight_buckets_v2.sh <stage> [<stage> ...]    stages 1, 2, 3 as in the old script;
#        2b = the rest of stage 2 after seed 0 (paused 24.09 18:46 for exact benchmarks)
# Log: data/logs/buckets_v2_<date>.log, raw job outputs in data/logs/jobs_<date>/
set -u
cd "$(dirname "$0")/.."
mkdir -p data/logs
TS="$(date +%Y%m%d_%H%M)"
LOG="data/logs/buckets_v2_${TS}.log"
JOBS="data/logs/jobs_${TS}"
mkdir -p "$JOBS"
exec > >(tee -a "$LOG") 2>&1
SPEC="--spec 2p_100bb_river"
GRID="--preflop-fracs 1.0 --postflop-fracs 0.5,1.0 --max-raises 2"
DEALS=100000                      # 200k hands per direction / per row
CACHE="--bucket-cache 4M,32M,128M" # 32 MB + 256 MB + 1 GB; all 13.96M turn forms fit
MAXJ=${MAXJ:-12}                  # parallel evaluation processes (single-threaded each)
PIDS=()
NAMES=()

stamp() { echo "=== $* ($(date '+%d.%m %H:%M:%S'))"; }

train() {  # kind buckets seed tag iters [extra flags]
  local kind="$1" nb="$2" seed="$3" tag="$4" iters="$5"
  shift 5
  stamp "TRAIN kind=$kind buckets=$nb seed=$seed tag=$tag iters=$iters $*"
  python scripts/train_blueprint.py --players 2 --stack 100 --street river $GRID --buckets "$nb" \
    --buckets-kind "$kind" --fit-situations 1500 --iters "$iters" --checkpoint-every 10000000 --backend cpp \
    $CACHE --seed "$seed" --eval-deals 0 --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -E "checkpoint [0-9]|fitted|loaded|resumed|backend|done in|Traceback|Error"
}

job() {  # name command...   runs in the background, at most MAXJ at once
  local name="$1"
  shift
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 2; done
  "$@" > "$JOBS/$name.txt" 2>&1 &
  PIDS+=($!)
  NAMES+=("$name")
}

collect() {  # wait for the queued jobs, then print their result lines in queue order
  local t0=$SECONDS
  stamp "EVALS: ${#NAMES[@]} jobs, up to $MAXJ at once"
  wait "${PIDS[@]}"
  stamp "EVALS DONE in $((SECONDS - t0))s"
  local n
  for n in "${NAMES[@]}"; do
    echo "--- $n"
    grep -E "^  |Traceback|Error" "$JOBS/$n.txt"
  done
  PIDS=()
  NAMES=()
}

duel() {  # tag_a tag_b   (both directions in one process, seed 7 as before)
  job "DUEL_$1_vs_$2" python scripts/compare_checkpoints.py $SPEC $GRID \
    --a "data/blueprint_$1.json" --b "data/blueprint_$2.json" \
    --buckets-a "data/buckets_$1.json" --buckets-b "data/buckets_$2.json" \
    --label-a "$1" --label-b "$2" --deals $DEALS --seed 7
}

plateau() {  # tag late_iters early_iters (first line = late vs early)
  job "PLATEAU_$1_it$2_vs_it$3" python scripts/compare_checkpoints.py $SPEC $GRID \
    --a "data/blueprint_$1.it$2.json" --b "data/blueprint_$1.it$3.json" --buckets "data/buckets_$1.json" \
    --label-a "it$2" --label-b "it$3" --deals $DEALS --seed 7
}

accept() {  # tag: one process per opponent, the same seed 5 as the sequential script
  local opp
  for opp in tag nit maniac station lag gridrandom random; do  # random rows last: pass/fail sanity only
    job "ACCEPT_$1_$opp" python scripts/eval_archetypes.py $SPEC $GRID --blueprint "data/blueprint_$1.json" \
      --buckets "data/buckets_$1.json" --deals $DEALS --seed 5 --opponents "$opp"
  done
}

stage1() {  # Г2: is the potential-aware gap a seed peculiarity?
  train potential 8 1 hunl100v2_pot_s1 30000000 --resume
  accept hunl100v2_pot_s1
  duel hunl100v2_pot_s1 hunl100v2_pot_s0
  duel hunl100v2_pot_s1 hunl100v2_ehs_s0
  duel hunl100v2_pot_s1 hunl100v2_ehs_s1
  collect
}

stage2() {  # 16 E[HS] buckets, two seeds: do 8 buckets limit the blueprint?
  train ehs 16 0 hunl100v2_ehs16_s0 40000000
  train ehs 16 1 hunl100v2_ehs16_s1 40000000
  plateau hunl100v2_ehs16_s0 40000000 20000000
  duel hunl100v2_ehs16_s0 hunl100v2_ehs16_s1
  duel hunl100v2_ehs16_s0 hunl100v2_ehs_s0
  duel hunl100v2_ehs16_s1 hunl100v2_ehs_s1
  accept hunl100v2_ehs16_s0
  accept hunl100v2_ehs16_s1
  collect
}

stage2b() {  # stage 2 resumed after the 18:46 pause for benchmarks: seed 0 is done, train seed 1, then the evaluations
  train ehs 16 1 hunl100v2_ehs16_s1 40000000
  plateau hunl100v2_ehs16_s0 40000000 20000000
  duel hunl100v2_ehs16_s0 hunl100v2_ehs16_s1
  duel hunl100v2_ehs16_s0 hunl100v2_ehs_s0
  duel hunl100v2_ehs16_s1 hunl100v2_ehs_s1
  accept hunl100v2_ehs16_s0
  accept hunl100v2_ehs16_s1
  collect
}

stage3() {  # 16 potential-aware buckets, two seeds: does distribution-awareness pay off at 16?
  train potential 16 0 hunl100v2_pot16_s0 40000000
  train potential 16 1 hunl100v2_pot16_s1 40000000
  plateau hunl100v2_pot16_s0 40000000 20000000
  duel hunl100v2_pot16_s0 hunl100v2_pot16_s1
  duel hunl100v2_pot16_s0 hunl100v2_ehs16_s0
  duel hunl100v2_pot16_s1 hunl100v2_ehs16_s1
  accept hunl100v2_pot16_s0
  accept hunl100v2_pot16_s1
  collect
}

for s in "$@"; do
  stamp "STAGE $s"
  "stage$s"
done
stamp "BUCKETS V2 DONE (stages $*), log: $LOG"
