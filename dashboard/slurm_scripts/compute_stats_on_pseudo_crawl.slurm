#!/bin/bash
#SBATCH --job-name=pseudo_crawl_compute_stats_v5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/six/uty16tp/code/big_science/logs/compute_stats_v5/%x-%j.out           # output file name
#SBATCH --array=1-604
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/cc
DATA_TOOLING_REPO=$WORK/repos/sync_data_tooling/data_tooling

pushd $DATA_TOOLING_REPO

SEED_ID=$(python cc_pseudo_crawl/load_all_seed_ids.py --seed-path "$DATA_TOOLING_REPO"/cc_pseudo_crawl/sourcing_sheet_seeds/ --seed-index $SLURM_ARRAY_TASK_ID)

echo "Computing stats on seed id ${SEED_ID}"

DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/datasets-seeds/bigscience-catalogue-data/pseudo_crawl_seed--seed-id--"$SEED_ID"
SAVE_STATS_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/datasets-stats/bigscience-catalogue-data/seed_id="$SEED_ID"/stats.json
SAVE_STATS_PATH_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/datasets-stats/bigscience-catalogue-data/seed_id="$SEED_ID"/full
SAVE_STATS_PATH_FULL=$SAVE_STATS_PATH_DIR/full.jsonl.gz

mkdir -p $SAVE_STATS_PATH_DIR

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python dashboard/python_scripts/compute_stats.py \
    --dataset-path $DATASET_PATH \
    --num-proc 4 \
    --save-path-stats-json $SAVE_STATS_PATH \
    --save-path-stats-full-json $SAVE_STATS_PATH_FULL \
    --use-datasets-caching\
    --seed-id $SEED_ID\
    --save-batch-size 2 # 100 # 10 # 2
