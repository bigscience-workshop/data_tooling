#!/bin/bash
#SBATCH --job-name=pseudo_crawl_shard_and_compress
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/fix_compress_cpu_p1/%x-%j.out           # output file name
#SBATCH --array=0-604   # TODO: modify according to the number of models you want to evaluated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.wang@huggingface.co
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/cc
DATA_TOOLING_REPO=$WORK/code/big_science/data_tooling

pushd $DATA_TOOLING_REPO

SEED_ID=$(python cc_pseudo_crawl/load_all_seed_ids.py --seed-path "$DATA_TOOLING_REPO"/cc_pseudo_crawl/sourcing_sheet_seeds/ --seed-index $SLURM_ARRAY_TASK_ID)
# SEED_IDS=(
# 263
# 20
# 280
# 424
# 299
# 58
# 189
# 191
# 63
# 585
# 333
# 79
# 211
# 86
# 349
# 354
# 226
# 100
# 497
# 373
# 249
# 638
# 255
# )
# SEED_ID=${SEED_IDS[$SLURM_ARRAY_TASK_ID]}


echo "Sharding and compressing seed id ${SEED_ID}"

DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/datasets-seeds/bigscience-catalogue-data/pseudo_crawl_seed--seed-id--"$SEED_ID"
SAVE_DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/datasets-compressed-shards/bigscience-catalogue-data/seed_id="$SEED_ID"

mkdir -p $SAVE_DATASET_PATH

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python cc_pseudo_crawl/shard_and_compress.py \
    --dataset-path $DATASET_PATH \
    --max-size 10_000_000_000 \
    --num-proc 4 \
    --save-num-proc 80 \
    --save-path $SAVE_DATASET_PATH \
    --save-batch-size 2 # 100 # 10 # 2

