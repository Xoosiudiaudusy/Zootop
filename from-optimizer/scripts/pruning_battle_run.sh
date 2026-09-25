set -u
cd /home/user/core-wt
. /home/user/venv312/bin/activate
D=/home/user/duel200
cp $D/buckets_try1500.json $D/buckets_ref.json
T0=$(date +%s)
echo "== table"; python scripts/build_bucket_table.py --buckets $D/buckets_ref.json --out $D/tables --threads 4 --check 5000 2>&1 | grep -v "%"
echo "table done at +$(( $(date +%s) - T0 ))s"
export NEGPLURIBUS_BUCKET_TABLES=$D/tables
G="--players 2 --stack 200 --street river --preflop-fracs 0.5,1.0,3.0 --postflop-fracs 0.5,1.0,2.0,4.0 --max-raises 3 --buckets 16 --buckets-kind potential --backend cpp --threads 4 --eval-deals 0 --data-dir $D --iters 40000000 --checkpoint-every 10000000"
P="--prune-below 10000 --prune-scale-t --regret-floor 1.033 --prune-after 4000000"
train() { t=$1; s=$2; shift 2; cp $D/buckets_ref.json $D/buckets_$t.json; echo "== $t"; python scripts/train_blueprint.py $G --seed $s --tag $t "$@" 2>&1 | grep -vE "^\s+iter "; echo "$t done at +$(( $(date +%s) - T0 ))s"; }
train base_s0 0
train prune_s0 0 $P
train ctrl_s1 1
train prune_s1 1 $P
C="--players 2 --stack 200 --street river --preflop-fracs 0.5,1.0,3.0 --postflop-fracs 0.5,1.0,2.0,4.0 --max-raises 3 --buckets $D/buckets_ref.json --deals 200000"
duel() { python scripts/compare_checkpoints.py $C --a $D/blueprint_$1.bin --b $D/blueprint_$2.bin --label-a $1 --label-b $2 > $D/duel_$1_vs_$2.log 2>&1; }
duel prune_s0 base_s0 & duel ctrl_s1 base_s0 & duel prune_s1 ctrl_s1 & wait
grep -h " vs " $D/duel_*.log
echo "ALL_DONE at +$(( $(date +%s) - T0 ))s"
