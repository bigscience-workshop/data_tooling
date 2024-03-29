#!/bin/bash
#SBATCH --job-name=pseudo_crawl_merge_seed_shards
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/six/commun/pseudo_crawl/seeds_batch_2/logs/merge_seed_shards/%x-%j.out           # output file name #TODO change path if necessary
#SBATCH --array=0-99   #TODO set correct number
#SBATCH --account=six@cpu


set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_2/cc
DATA_TOOLING_REPO=$WORK/repos/sync_data_tooling/data_tooling

DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_2/datasets-shard-by-seed-id/bigscience-catalogue-data
pushd $DATA_TOOLING_REPO

SEED_ID=$(python cc_pseudo_crawl/python_scripts/load_all_seed_ids.py --seed-path "$DATA_TOOLING_REPO"/cc_pseudo_crawl/seeds_batch_2/sourcing_sheet_seeds/seeds.csv --seed-index $SLURM_ARRAY_TASK_ID)
echo "Merging all shards of seed id ${SEED_ID}"
SAVE_DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_2/datasets-seeds/bigscience-catalogue-data/pseudo_crawl_seed--seed-id--"$SEED_ID"

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python cc_pseudo_crawl/python_scripts/merge_seed_shards.py \
    --dataset-dir $DATASET_DIR \
    --seed-id $SEED_ID \
    --save-path $SAVE_DATASET_PATH
