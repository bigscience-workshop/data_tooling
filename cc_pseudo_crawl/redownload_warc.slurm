#!/bin/bash
#SBATCH --job-name=pseudo_crawl_redownload
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --array=0-9   # TODO: modify according to the number of models you want to evaluated
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/cc
SAVE_DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/datasets
DATA_TOOLING_REPO=$WORK/code/big_science/data_tooling
DATASET_PATH="$SAVE_DATASET_DIR"/bigscience-catalogue-data/pseudo_crawl_seed--"$SLURM_ARRAY_TASK_ID"--"$SLURM_ARRAY_TASK_COUNT"
echo $DATASET_PATH
pushd $DATA_TOOLING_REPO

python -m cc_pseudo_crawl.redownload_warc.py \
    --dataset-path $DATASET_PATH \
    --num-proc 4 \
