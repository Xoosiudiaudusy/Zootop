cd /home/user/core-wt
. /home/user/venv312/bin/activate
export NEGPLURIBUS_BUCKET_TABLES=/home/user/data/bucket_tables
D=/home/user/duel
G="--players 2 --stack 100 --street river --preflop-fracs 1.0 --postflop-fracs 0.5,1.0 --max-raises 2 --buckets 8 --buckets-kind ehs --backend cpp --threads 4 --eval-deals 0 --data-dir $D --no-l1 --iters 20000000 --prune-after 2000000"
train() { t=$1; s=$2; shift 2; cp $D/buckets_base_s0.json $D/buckets_$t.json; echo "== $t"; python scripts/train_blueprint.py $G --seed $s --tag $t "$@" | grep -E "done in|pruned|pruning"; }
train t5000f_s0 0 --prune-below 5000 --prune-scale-t --regret-floor 1.033
train t50f_s0 0 --prune-below 50 --prune-scale-t --regret-floor 1.033
train p9f_s0 0 --prune-below 1e9 --regret-floor 1.033
train rel5stat_s0 0 --prune-below 5 --prune-relative
train p11stat_s0 0 --prune-below 1e11
train t5000f_s1 1 --prune-below 5000 --prune-scale-t --regret-floor 1.033
train p9f_s1 1 --prune-below 1e9 --regret-floor 1.033
C="--players 2 --stack 100 --street river --preflop-fracs 1.0 --postflop-fracs 0.5,1.0 --max-raises 2 --buckets $D/buckets_base_s0.json --deals 200000"
duel() { python scripts/compare_checkpoints.py $C --a $D/blueprint_$1.bin --b $D/blueprint_$2.bin --label-a $1 --label-b $2 > $D/duel4_$1.log 2>&1; }
duel t5000f_s0 base_s0 & duel t50f_s0 base_s0 & duel p9f_s0 base_s0 & wait
duel t5000f_s1 base_s1 & duel p9f_s1 base_s1 & wait
grep -h " vs " $D/duel4_*.log
echo ALL4_DONE
