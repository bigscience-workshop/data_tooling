#!/bin/bash
#SBATCH --job-name=pseudo_crawl_shard_by_id
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=1           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --array=0-605   # TODO: modify according to the number of models you want to evaluated
#SBATCH --account=six@cpu


set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/cc
DATA_TOOLING_REPO=$WORK/code/big_science/data_tooling
DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/datasets-shard-by-seed-id/bigscience-catalogue-data
pushd $DATA_TOOLING_REPO

SEED_ID=$(python cc_pseudo_crawl/load_all_seed_ids.py --seed-path "$DATA_TOOLING_REPO"/cc_pseudo_crawl/sourcing_sheet_seeds/ --seed-index $SLURM_ARRAY_TASK_ID)

SAVE_DATASET_PREFIX_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/datasets-seeds/bigscience-catalogue-data/pseudo_crawl_seed--seed-id--"$SEED_ID"
mkdir -p $SAVE_DATASET_DIR

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python cc_pseudo_crawl/merge_seed_shards.py \
    --dataset-dir $DATASET_DIR \
    --num-proc 80 \
    --save-prefix-path $SAVE_DATASET_PREFIX_PATH
