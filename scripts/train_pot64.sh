#!/usr/bin/env bash
# HU 200bb, wide grid, potential-aware 64 buckets with Monte-Carlo features: the baseline of the duel
# "exact vs Monte-Carlo features" (docs/walkthrough-findings-2026-09-14.md, findings 19).  Per seed:
# fit the bucketer (6000 situations per street = 94 per cluster, the density of the production 16-bucket
# bucketer, fitted on 1500: overnight scripts, sha reproduced by the optimizer), build its bucket table, then train 400M iterations with a
# checkpoint every 50M (the L1 change of the average strategy is printed at each checkpoint).
#   bash scripts/train_pot64.sh [seeds]      default seeds 0 1; THREADS (default 12) for the training
set -euo pipefail
cd "$(dirname "$0")/.."
THREADS=${THREADS:-12}
for S in ${*:-0 1}; do
  TAG=hunl200w3_pot64_s$S
  [ -f "data/buckets_$TAG.json" ] || python -B scripts/fit_buckets.py --buckets 64 --kind potential \
      --situations 6000 --seed "$S" --out "data/buckets_$TAG.json"
  python -B scripts/build_bucket_table.py --buckets "data/buckets_$TAG.json" --out data/bucket_tables --threads 16
  NEGPLURIBUS_BUCKET_TABLES=data/bucket_tables python -B scripts/train_blueprint.py --players 2 --stack 200 --street river \
      --preflop-fracs 0.5,1.0,3.0 --postflop-fracs 0.5,1.0,2.0,4.0 --max-raises 3 --buckets 64 --buckets-kind potential \
      --fit-situations 6000 --iters 400000000 --checkpoint-every 50000000 --tag "$TAG" --eval-deals 0 --backend cpp \
      --threads "$THREADS" --seed "$S"
done
