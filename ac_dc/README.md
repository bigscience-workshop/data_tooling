## Big Science - Automated Classification & Dataset Curation - AC/DC

This is the data filtering code for BigScience.

See this document for more details..

https://docs.google.com/document/d/1bx7lzAIWALH2IX5PLAiRfkHr3025dC-ZYkEmq4zB2gI/edit


### Deduplication

#### 1. Create Simhashes
```bash
python ac_dc/deduplicate.py build-hashes "cache/en_hashes_00001" --data-files "en/en_00001.jsonl.gz" --data-files "en/en_00002.jsonl.gz" --path "mhtoin/register_oscar" --split "train"
python ac_dc/deduplicate.py build-hashes "cache/en_hashes_00002" --data-files "en/en_00003.jsonl.gz" --data-files "en/en_00004.jsonl.gz" --path "mhtoin/register_oscar" --split "train"
```
The above commands add an addition column `hash` in the data and outputs two datasets at `cache/en_hashes_00001` and `cache/en_hashes_00002`. This is useful for large dataset and each node/worker can hash some shards of the data in parallel.

#### 2. Create a Simhash Index
```bash
python ac_dc/deduplicate.py build-index "cache/en_simhash_index.pkl" "cache/en_hashes_00001" "cache/en_hashes_00002" --split "train"
```
This creates the index file based on the hashed two datasets. This is a merge step and takes O(n) time.

#### 3. Find Duplicates
```bash
python ac_dc/deduplicate.py find-duplicates "cache/en_hashes_00001" "cache/en_hashes_00002" "cache/en_simhash_index.pkl" --split "train"
```
This adds another column `duplicates` into the data with the index and outputs them into `cache/en_hashes_00001_duplicates` and `cache/en_hashes_00002_duplicates`;

#### 4. Remove Duplicates
```bash
python ac_dc/deduplicate.py remove-duplicates "cache/en_hashes_00001_duplicates" "cache/en_hashes_00002_duplicates" --split "train"
```
This removes all duplicates from the given datasets and outputs `cache/en_hashes_00001_deduplicated` and `cache/en_hashes_00002_deduplicated`; Partially parallelized because is a step finding connected components of duplicates so it takes O(n) time.
