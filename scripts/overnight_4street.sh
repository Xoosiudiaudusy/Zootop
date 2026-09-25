#!/usr/bin/env bash
# Overnight: heads-up 100bb, 4 streets.  Trains five blueprints with the C++ backend (bounded
# bucket caches) and runs the acceptance after each one, so results are on disk even if nobody
# is watching.  Everything is logged to data/logs/overnight_<date>.log.
#
#   bash scripts/overnight_4street.sh            # full plan (~8-10 h on 16 threads)
#   STAGES="small" bash scripts/overnight_4street.sh      # stages: small wide pot
#
# Acceptance per the project rules: plateau (late checkpoint vs earlier one), seed control
# (same abstraction, other seed), abstraction comparison only next to the control, "vs random
# >= 0" as a Nash sanity check, grid-random (no translation) and archetypes.  The night of 23-24.09
# ran an earlier version with the ad-hoc commit_all_in rule, since removed (docs/scale_4street.md).
set -u
cd "$(dirname "$0")/.."
mkdir -p data/logs
LOG="data/logs/overnight_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
STAGES="${STAGES:-small wide pot}"
SPEC="--spec 2p_100bb_river"
SMALL="--preflop-fracs 1.0 --postflop-fracs 0.5,1.0 --max-raises 2"
WIDE="--preflop-fracs 1.0,3.0 --postflop-fracs 0.5,1.0,2.0,4.0 --max-raises 3"
DEALS=15000          # 30k hands per head-to-head: +/- ~10 bb/100
ARCH_DEALS=100000   # 200k hands per row: +/- 4..10 bb/100 (calibrated 24.09)

stamp() { echo "=== $* ($(date '+%d.%m %H:%M:%S'))"; }

train() {  # kind seed tag grid iters every
  stamp "TRAIN kind=$1 seed=$2 tag=$3 iters=$5"
  python scripts/train_blueprint.py --players 2 --stack 100 --street river $4 --buckets 8 \
    --buckets-kind "$1" --fit-situations 1500 --iters "$5" --checkpoint-every "$6" --backend cpp \
    --seed "$2" --eval-deals 0 --tag "$3" 2>&1 | grep --line-buffered -vE "^\s+iter "
}

duel() {  # grid a b bk_a bk_b label_a label_b [extra]
  stamp "DUEL $6 vs $7"
  python scripts/compare_checkpoints.py $SPEC $1 --a "$2" --b "$3" --buckets-a "$4" --buckets-b "$5" \
    --label-a "$6" --label-b "$7" --deals $DEALS ${8:-} 2>&1 | grep -E "^  "
}

accept() {  # grid tag
  local bp="data/blueprint_$2.json" bk="data/buckets_$2.json"
  stamp "ACCEPT $2: random, grid-random and archetypes"
  python scripts/eval_archetypes.py $SPEC $1 --blueprint "$bp" --buckets "$bk" --deals $ARCH_DEALS     --opponents random,gridrandom,tag,nit,maniac,station 2>&1 | grep -E "^  "
}

# Order = most important answers first:
#  1. small-grid E[HS] seed 0: does a well-trained blueprint still lose to random (translation hole)?
#  2. seed control for it
#  3. wide grid: does the structural fix (overbet sizes in the grid) remove the hole?
#  4. potential-aware vs E[HS] on 4 streets, next to the control
#  5. control for the wide grid
has() { [[ " $STAGES " == *" $1 "* ]]; }
if has small; then
  train ehs 0 hunl100v2_ehs_s0 "$SMALL" 50000000 10000000
  accept "$SMALL" hunl100v2_ehs_s0
  duel "$SMALL" data/blueprint_hunl100v2_ehs_s0.it50000000.json data/blueprint_hunl100v2_ehs_s0.it20000000.json        data/buckets_hunl100v2_ehs_s0.json data/buckets_hunl100v2_ehs_s0.json it50M it20M
  train ehs 1 hunl100v2_ehs_s1 "$SMALL" 50000000 10000000
  duel "$SMALL" data/blueprint_hunl100v2_ehs_s0.json data/blueprint_hunl100v2_ehs_s1.json        data/buckets_hunl100v2_ehs_s0.json data/buckets_hunl100v2_ehs_s1.json ehs_s0 ehs_s1_CONTROL
fi
if has wide; then
  train ehs 0 hunl100wv2_ehs_s0 "$WIDE" 40000000 10000000
  accept "$WIDE" hunl100wv2_ehs_s0
  duel "$WIDE" data/blueprint_hunl100wv2_ehs_s0.it40000000.json data/blueprint_hunl100wv2_ehs_s0.it20000000.json        data/buckets_hunl100wv2_ehs_s0.json data/buckets_hunl100wv2_ehs_s0.json it40M it20M
fi
if has pot; then
  train potential 0 hunl100v2_pot_s0 "$SMALL" 50000000 10000000
  duel "$SMALL" data/blueprint_hunl100v2_pot_s0.json data/blueprint_hunl100v2_ehs_s0.json        data/buckets_hunl100v2_pot_s0.json data/buckets_hunl100v2_ehs_s0.json pot_s0 ehs_s0
  duel "$SMALL" data/blueprint_hunl100v2_pot_s0.json data/blueprint_hunl100v2_ehs_s1.json        data/buckets_hunl100v2_pot_s0.json data/buckets_hunl100v2_ehs_s1.json pot_s0 ehs_s1
  accept "$SMALL" hunl100v2_pot_s0
fi
if has wide; then
  train ehs 1 hunl100wv2_ehs_s1 "$WIDE" 40000000 10000000
  duel "$WIDE" data/blueprint_hunl100wv2_ehs_s0.json data/blueprint_hunl100wv2_ehs_s1.json        data/buckets_hunl100wv2_ehs_s0.json data/buckets_hunl100wv2_ehs_s1.json wide_s0 wide_s1_CONTROL
fi
stamp "OVERNIGHT DONE, log: $LOG"
