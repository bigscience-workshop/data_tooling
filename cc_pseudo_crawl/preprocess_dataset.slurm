#!/bin/bash
#SBATCH --job-name=pseudo_crawl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --array=0-9 # TODO: modify according to the number of models you want to evaluated
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CC_INDEX_FOLDER=$SCRATCH/pseudo_crawl
DATA_TOOLING_REPO=$WORK/code/big_science/data_tooling
pushd $DATA_TOOLING_REPO

# TODO run this offline
# aws s3 sync s3://commoncrawl-dev/big-science-workshop/data-sourcing-sheet/cc/ $CC_INDEX_FOLDER/cc/

python cc_pseudo_crawl/preprocess_dataset.py \
    --dataset bigscience-catalogue-data/pseudo_craw_seed \
    --cc-index-folder $CC_INDEX_FOLDER \
    --num-proc 4 \
    --shard-id $SLURM_ARRAY_TASK_ID \
    --num-shards $SLURM_ARRAY_TASK_COUNT \
    --range :10000
