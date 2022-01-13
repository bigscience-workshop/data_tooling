NUM_SHARDS=$1

CC_INDEX_FOLDER=~/bigscience/pseudo_crawl


for i in {1..$1}
do
   python cc_pseudo_crawl/preprocess_dataset.py \
    --dataset bigscience-catalogue-data/pseudo_crawl_seed \
    --cc-index-folder $CC_INDEX_FOLDER \
    --save-dir $SAVE_DATASET_DIR \
    --num-proc 8 \
    --shard-id $i \
    --num-shards $NUM_SHARDS \
    --range :100000
done