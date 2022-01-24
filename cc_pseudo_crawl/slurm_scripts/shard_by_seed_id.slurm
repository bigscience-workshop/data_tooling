#!/bin/bash
#SBATCH --job-name=pseudo_crawl_shard_by_id
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p13
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/cc
DATA_TOOLING_REPO=$WORK/code/big_science/data_tooling
DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/datasets-preprocessed-text-extracted/bigscience-catalogue-data
SAVE_DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/datasets-shard-by-seed-id/bigscience-catalogue-data
pushd $DATA_TOOLING_REPO

mkdir -p $SAVE_DATASET_DIR

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python -m cc_pseudo_crawl.redownload_warc \
    --dataset-dir $DATASET_DIR \
    --num-proc 40 \
    --save-dir $SAVE_DATASET_DIR
