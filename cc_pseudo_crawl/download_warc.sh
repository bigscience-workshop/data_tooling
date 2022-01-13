NUM_SHARDS=$1

if [[ $NUM_SHARDS = "" ]]
then
  echo "Please feed the number of shards"
  exit
fi

CC_INDEX_FOLDER=~/bigscience/pseudo_crawl/cc
SAVE_DATASET_DIR=~/bigscience/pseudo_crawl/datasets

pushd ~/code/data_tooling

for i in $(seq 0 $((NUM_SHARDS-1)))
do
   echo "Processing shard number $i"
   python3 cc_pseudo_crawl/download_warc.py \
    --dataset bigscience-catalogue-data/pseudo_crawl_seed_dedup_url \
    --cc-index-folder $CC_INDEX_FOLDER \
    --save-dir $SAVE_DATASET_DIR \
    --num-proc 8 \
    --shard-id $i \
    --num-shards $NUM_SHARDS \
    --range :1000000
done