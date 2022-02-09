#!/bin/bash
#SBATCH --job-name=pseudo_crawl_clean_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 4:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/six/commun/pseudo_crawl/seeds_batch_1_2/logs/clean_dataset/%x-%j.out           # output file name #TODO change path if necessary
#SBATCH --array=13
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

DATA_TOOLING_REPO=$WORK/repos/sync_data_tooling/data_tooling

DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/hub/pseudo_crawl
SAVE_DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1_2/datasets-clean/bigscience-catalogue-data
echo $DATASET_PATH
pushd $DATA_TOOLING_REPO

mkdir -p $SAVE_DATASET_DIR

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python cc_pseudo_crawl/python_scripts/pseudo_crawl_seed_to_lm_dset.py \
    --seed-id $SLURM_ARRAY_TASK_ID \
    --save-dir $SAVE_DATASET_DIR \
    --pseudo_crawl_path $DATASET_PATH \
    --batch-size 1000 \
    --num-proc 80\
    --n-records 10000 \
    --pourcentage-threshold 0.01  
