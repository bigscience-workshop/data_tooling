#!/bin/bash
#SBATCH --job-name=pseudo_crawl_divide_in_subshards
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/six/uty16tp/code/big_science/logs/preprocess/%x-%j.out           # output file name
#SBATCH --array=0-9   # TODO: modify according to the number of models you want to evaluated
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/cc
DATA_TOOLING_REPO=$WORK/repos/sync_data_tooling/data_tooling
DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/datasets-redownload/bigscience-catalogue-data/pseudo_crawl_seed--"$SLURM_ARRAY_TASK_ID"--"$SLURM_ARRAY_TASK_COUNT"
SAVE_DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/datasets-divide-in-subshards/bigscience-catalogue-data
echo $DATASET_PATH
pushd $DATA_TOOLING_REPO

mkdir -p $SAVE_DATASET_DIR

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python -m cc_pseudo_crawl.python_scripts.divide_in_shards \
    --dataset-path $DATASET_PATH \
    --save-dir $SAVE_DATASET_DIR \
    --num-shards 10
