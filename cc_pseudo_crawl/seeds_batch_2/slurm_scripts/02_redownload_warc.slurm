#!/bin/bash
#SBATCH --job-name=pseudo_crawl_redownload
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name #TODO change path if necessary
#SBATCH --array=0-9   #TODO set correct number
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl_batch_2/cc  #TODO change path if necessary
DATA_TOOLING_REPO=$WORK/repos/sync_data_tooling/data_tooling  #TODO change path if necessary

DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl_batch_2/datasets/bigscience-catalogue-data/pseudo_crawl_seed--"$SLURM_ARRAY_TASK_ID"--"$SLURM_ARRAY_TASK_COUNT"
SAVE_DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl_batch_2/datasets-redownload/bigscience-catalogue-data
SAVE_DATASET_PATH=$SAVE_DATASET_DIR/pseudo_crawl_seed--"$SLURM_ARRAY_TASK_ID"--"$SLURM_ARRAY_TASK_COUNT"
echo $DATASET_PATH
pushd $DATA_TOOLING_REPO

mkdir -p $SAVE_DATASET_DIR

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python -m cc_pseudo_crawl.python_scripts.redownload_warc \
    --dataset-path $DATASET_PATH \
    --num-proc 4 \
    --save-path $SAVE_DATASET_PATH \
    --use-datasets-caching
