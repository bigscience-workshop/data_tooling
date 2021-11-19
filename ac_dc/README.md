## Big Science - Automated Classification & Dataset Curation - AC/DC

This is the data filtering code for BigScience.

See this document for more details..

https://docs.google.com/document/d/1bx7lzAIWALH2IX5PLAiRfkHr3025dC-ZYkEmq4zB2gI/edit


### Deduplication

#### 0. Sharding a dataset

We want to shard a dataset into multiple shards so that each node on HPC can take a shard and each shard can be further parallelized with CPU cores.

```bash
python ac_dc/deduplicate.py create-shards "cache/sharded" 5 --path "oscar-corpus/OSCAR-2109" --name "deduplicated_af" --split "train"
# or
python ac_dc/deduplicate.py create-shards "cache/sharded" 5 --path "oscar-corpus/OSCAR-2109" --name "deduplicated_af" --data-dir "local path to data directory" --split "train"
```

It loads a local dataset and segments its `train` split into 5 shards/sub-datasets under `cache/sharded`. This gives you
```
cache/sharded
├── sharded_00000.jsonl
├── sharded_00001.jsonl
├── sharded_00002.jsonl
├── sharded_00003.jsonl
└── sharded_00004.jsonl
```

#### 1. Create Simhashes
```bash
# run each command on each node
python ac_dc/deduplicate.py build-hashes "cache/deduplicated_af_hashes_00001" --data-files "sharded_00000.jsonl" --data-files "sharded_00001.jsonl" --path "cache/sharded" --split "train"
python ac_dc/deduplicate.py build-hashes "cache/deduplicated_af_hashes_00002" --data-files "sharded_00002.jsonl" --data-files "sharded_00003.jsonl" --path "cache/sharded" --split "train"
python ac_dc/deduplicate.py build-hashes "cache/deduplicated_af_hashes_00003" --data-files "sharded_00004.jsonl" --path "cache/sharded" --split "train"
```
The above commands add an addition column `hash` in the data and outputs two datasets at `cache/en_hashes_00001` and `cache/en_hashes_00002`. This is useful for large dataset and each node/worker can hash some shards of the data in parallel.

#### 2. Create a Simhash Index
```bash
python ac_dc/deduplicate.py build-index "cache/deduplicated_af_simhash_index.pkl" "cache/deduplicated_af_hashes_00001" "cache/deduplicated_af_hashes_00002" "cache/deduplicated_af_hashes_00003" --split "train"
```
This creates the index file based on ALL the hashed datasets. This is a merge step and takes O(n) time.

#### 3. Find Duplicates
```bash
# run each command on each node
LOG_LEVEL="INFO" python ac_dc/deduplicate.py find-duplicates "cache/deduplicated_af_hashes_00001" "cache/deduplicated_af_simhash_index.pkl" --split "train"
LOG_LEVEL="INFO" python ac_dc/deduplicate.py find-duplicates "cache/deduplicated_af_hashes_00002" "cache/deduplicated_af_simhash_index.pkl" --split "train"
LOG_LEVEL="INFO" python ac_dc/deduplicate.py find-duplicates "cache/deduplicated_af_hashes_00003" "cache/deduplicated_af_simhash_index.pkl" --split "train"
```
This adds another column `duplicates` into the data with the index and outputs them into `cache/en_hashes_0000{1,2,3}_duplicates`.

#### 4. Remove Duplicates
```bash
python ac_dc/deduplicate.py remove-duplicates "cache/deduplicated_af_hashes_00001_duplicates" "cache/deduplicated_af_hashes_00002_duplicates" "cache/deduplicated_af_hashes_00003_duplicates" --split "train"
```
This removes all duplicates from the given datasets and outputs `cache/en_hashes_0000{1,2,3}_deduplicated`; Partially parallelized because thre is a step finding connected components of duplicates and it takes O(n) time.

#### 5. Merge Shards
```bash
python ac_dc/deduplicate.py merge-shards "cache/simhash_deduplicated_af" "cache/deduplicated_af_hashes_00001_deduplicated" "cache/deduplicated_af_hashes_00002_deduplicated" "cache/deduplicated_af_hashes_00003_deduplicated" --split "train"
```
This merges all shards back into one dataset.


### Merge metadata from OSCAR 21.09 to OSCAR

Similar to the deduplication step, you can find an example script under `ac_dc/examples/merge.sh`