#!/usr/bin/env bash
# Bucket experiments (approved 24.09): potential-aware seed control (Г2), then 16 buckets for BOTH
# kinds with seed controls.  Narrow grid, same game as docs/scale_4street.md so every number is
# comparable with the 8-bucket results.  Predictions were committed to docs/scale_4street.md
# BEFORE this script ran.  Log: data/logs/buckets_<date>.log
set -u
cd "$(dirname "$0")/.."
mkdir -p data/logs
LOG="data/logs/buckets_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
SPEC="--spec 2p_100bb_river"
GRID="--preflop-fracs 1.0 --postflop-fracs 0.5,1.0 --max-raises 2"
DEALS=100000   # 200k hands per direction / per row

stamp() { echo "=== $* ($(date '+%d.%m %H:%M:%S'))"; }

train() {  # kind buckets seed tag iters
  stamp "TRAIN kind=$1 buckets=$2 seed=$3 tag=$4 iters=$5"
  python scripts/train_blueprint.py --players 2 --stack 100 --street river $GRID --buckets "$2" \
    --buckets-kind "$1" --fit-situations 1500 --iters "$5" --checkpoint-every 10000000 --backend cpp \
    --seed "$3" --eval-deals 0 --tag "$4" 2>&1 | grep --line-buffered -E "checkpoint [0-9]|fitted|done in|Traceback|Error"
}

duel() {  # tag_a tag_b [blueprint_a_file] [blueprint_b_file]
  local bpa="${3:-data/blueprint_$1.json}" bpb="${4:-data/blueprint_$2.json}"
  stamp "DUEL $1 vs $2"
  python scripts/compare_checkpoints.py $SPEC $GRID --a "$bpa" --b "$bpb" \
    --buckets-a "data/buckets_$1.json" --buckets-b "data/buckets_$2.json" \
    --label-a "$1" --label-b "$2" --deals $DEALS --seed 7 2>&1 | grep --line-buffered -E "^  |Traceback|Error"
}

plateau() {  # tag late_iters early_iters
  stamp "PLATEAU $1: it$2 (first line = late vs early) vs it$3"
  python scripts/compare_checkpoints.py $SPEC $GRID --a "data/blueprint_$1.it$2.json" --b "data/blueprint_$1.it$3.json"     --buckets "data/buckets_$1.json" --label-a "it$2" --label-b "it$3" --deals $DEALS --seed 7 2>&1 | grep --line-buffered -E "^  |Traceback|Error"
}

accept() {  # tag
  stamp "ACCEPT $1: random, grid-random, archetypes"
  python scripts/eval_archetypes.py $SPEC $GRID --blueprint "data/blueprint_$1.json" --buckets "data/buckets_$1.json" \
    --deals $DEALS --seed 5 --opponents random,gridrandom,tag,nit,maniac,station,lag 2>&1 | grep --line-buffered -E "^  |Traceback|Error"
}

# 1. Г2: is the potential-aware gap a seed peculiarity?  Same recipe as pot_s0 (50M).
train potential 8 1 hunl100v2_pot_s1 50000000
accept hunl100v2_pot_s1
duel hunl100v2_pot_s1 hunl100v2_pot_s0
duel hunl100v2_pot_s1 hunl100v2_ehs_s0
duel hunl100v2_pot_s1 hunl100v2_ehs_s1

# 2. 16 E[HS] buckets, two seeds: does 8 buckets limit the blueprint?
train ehs 16 0 hunl100v2_ehs16_s0 40000000
train ehs 16 1 hunl100v2_ehs16_s1 40000000
plateau hunl100v2_ehs16_s0 40000000 20000000
duel hunl100v2_ehs16_s0 hunl100v2_ehs16_s1
duel hunl100v2_ehs16_s0 hunl100v2_ehs_s0
duel hunl100v2_ehs16_s1 hunl100v2_ehs_s1
accept hunl100v2_ehs16_s0
accept hunl100v2_ehs16_s1

# 3. 16 potential-aware buckets, two seeds: does distribution-awareness pay off at 16?
train potential 16 0 hunl100v2_pot16_s0 40000000
train potential 16 1 hunl100v2_pot16_s1 40000000
plateau hunl100v2_pot16_s0 40000000 20000000
duel hunl100v2_pot16_s0 hunl100v2_pot16_s1
duel hunl100v2_pot16_s0 hunl100v2_ehs16_s0
duel hunl100v2_pot16_s1 hunl100v2_ehs16_s1
accept hunl100v2_pot16_s0
accept hunl100v2_pot16_s1
stamp "BUCKETS DONE, log: $LOG"
