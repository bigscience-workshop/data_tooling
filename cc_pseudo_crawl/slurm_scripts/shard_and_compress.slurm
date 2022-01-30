#!/bin/bash
#SBATCH --job-name=pseudo_crawl_shard_and_compress
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4          # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/fix_compress/%x-%j.out           # output file name
#SBATCH --array=0-8   # TODO: modify according to the number of models you want to evaluated
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.wang@huggingface.co
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/cc
DATA_TOOLING_REPO=$WORK/code/big_science/data_tooling

pushd $DATA_TOOLING_REPO

# SEED_ID=$(python cc_pseudo_crawl/load_all_seed_ids.py --seed-path "$DATA_TOOLING_REPO"/cc_pseudo_crawl/sourcing_sheet_seeds/ --seed-index $SLURM_ARRAY_TASK_ID)
SEED_IDS=(
146
396
689
429
510
198
550
661
35
)
SEED_ID=${CHECKPOINT_STEPS[$SLURM_ARRAY_TASK_ID]}

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
    --save-path $SAVE_DATASET_PATH \
    --save-batch-size 20

