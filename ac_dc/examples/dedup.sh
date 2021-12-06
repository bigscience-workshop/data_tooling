#!/bin/bash
LANGUAGES=('gl')
SHARDS=20
THRESHOLD=3
PYTHON="/home/jovyan/conda/envs/data/bin/python"
SCRIPT="/home/jovyan/data_tooling/ac_dc/deduplicate.py"

for lang in "${LANGUAGES[@]}"; do
    echo "lang: $lang"

    echo "Creating ${SHARDS} shards"
    $PYTHON $SCRIPT create-shards "cache/sharded_deduplicated_${lang}" $SHARDS --path "oscar-corpus/OSCAR-2109" --name "deduplicated_${lang}" --split "train"

    echo "Hashing documents"
    for i in $(seq -f "%05g" 0 "$((SHARDS - 1))"); do
        echo "Hashing shard ${i}"
        $PYTHON $SCRIPT build-hashes "cache/sharded_deduplicated_${lang}/hashes_${i}" --data-files "sharded_${i}.jsonl" --path "cache/sharded_deduplicated_${lang}" --split "train" --shingle-size 4
    done

    echo "Creating index"
    $PYTHON $SCRIPT build-index "cache/sharded_deduplicated_${lang}/simhash_index.ann" $(seq -s " " -f "cache/sharded_deduplicated_${lang}/hashes_%05g" 0 "$((SHARDS - 1))") --split "train"

    echo "Finding duplicates"
    for i in $(seq -f "%05g" 0 "$((SHARDS - 1))"); do
        echo "Querying shard ${i}"
        $PYTHON -W ignore $SCRIPT find-duplicates "cache/sharded_deduplicated_${lang}/hashes_${i}" "cache/sharded_deduplicated_${lang}/simhash_index.ann" --split "train" --k 100 --threshold $THRESHOLD
    done

    echo "Removing duplicates"
    for i in $(seq -f "%05g" 0 "$((SHARDS - 1))"); do
        echo "Cleaning shard ${i}"
        $PYTHON $SCRIPT remove-duplicates "cache/sharded_deduplicated_${lang}/hashes_${i}_duplicates" --split "train"
    done

    echo "Merging shards"
    $PYTHON $SCRIPT merge-shards "cache/sharded_deduplicated_${lang}/output" $(seq -s " " -f "cache/sharded_deduplicated_${lang}/hashes_%05g_deduplicated" 0 "$((SHARDS - 1))") --split "train"

    echo "Done"
done
