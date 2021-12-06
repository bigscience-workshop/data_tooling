#!/bin/bash
LANGUAGES=('min')
SHARDS=1
THRESHOLD=1
PYTHON="/home/jovyan/conda/envs/data/bin/python"
SCRIPT="/home/jovyan/data_tooling/ac_dc/deduplicate.py"

for lang in "${LANGUAGES[@]}"; do
    echo "lang: $lang"
    $PYTHON $SCRIPT create-shards "cache/sharded_deduplicated_${lang}_v2" $SHARDS --path "oscar-corpus/OSCAR-2109" --name "deduplicated_${lang}" --split "train"
    $PYTHON $SCRIPT create-shards "cache/sharded_deduplicated_${lang}_v1" $SHARDS --path "oscar" --name "unshuffled_deduplicated_${lang}" --split "train"

    # Hash
    for i in $(seq -f "%05g" 0 "$((SHARDS - 1))"); do
        $PYTHON $SCRIPT build-hashes "cache/sharded_deduplicated_${lang}_v2/hashes_${i}" --data-files "sharded_${i}.jsonl" --path "cache/sharded_deduplicated_${lang}_v2" --split "train" --shingle-size 4 --text-column-name "text"
    done

    for i in $(seq -f "%05g" 0 "$((SHARDS - 1))"); do
        $PYTHON $SCRIPT build-hashes "cache/sharded_deduplicated_${lang}_v1/hashes_${i}" --data-files "sharded_${i}.jsonl" --path "cache/sharded_deduplicated_${lang}_v1" --split "train" --shingle-size 4 --text-column-name "text"
    done

    # Create the index file
    # $PYTHON $SCRIPT build-index "cache/sharded_deduplicated_${lang}_v1/simhash_index.ann" $(seq -s " " -f "cache/sharded_deduplicated_${lang}_v1/hashes_%05g" 0 "$((SHARDS - 1))") --split "train"
    $PYTHON $SCRIPT build-index "cache/sharded_deduplicated_${lang}_v2/simhash_index.ann" $(seq -s " " -f "cache/sharded_deduplicated_${lang}_v2/hashes_%05g" 0 "$((SHARDS - 1))") --split "train"

    # merge v2 metadata into v1
    $PYTHON $SCRIPT merge-meta \
        "cache/sharded_deduplicated_${lang}_v2/simhash_index.ann" \
        $(seq -s " " -f "--data-dirs cache/sharded_deduplicated_${lang}_v1/hashes_%05g" 0 "$((SHARDS - 1))") \
        $(seq -s " " -f "--meta-data-dirs cache/sharded_deduplicated_${lang}_v2/hashes_%05g" 0 "$((SHARDS - 1))") \
        --k 1 \
        --threshold $THRESHOLD \
        --split "train"

done
