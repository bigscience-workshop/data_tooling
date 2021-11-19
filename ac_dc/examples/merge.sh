#!/bin/bash
# This script is used to merge oscar v2 meta data into v1 using simhash

# LANGUAGES=('vls' 'diq' 'cbk' 'lrc' 'rue' 'gv' 'ht' 'nap' 'sco' 'bar' 'mwl' 'ie' 'myv' 'pam' 'scn' 'rm' 'frr' 'tyv' 'so' 'dsb' 'bxr' 'eml' 'nah' 'mai' 'gn' 'vec' 'li' 'xal' 'wuu' 'kw' 'yo' 'bh' 'ia' 'bs' 'io' 'qu' 'su' 'av' 'wa' 'mrj' 'kv' 'an' 'jv' 'jbo' 'ilo' 'lmo' 'min' 'mzn' 'gd' 'hsb' 'vo' 'gom' 'krc' 'lez' 'pms' 'war' 'new' 'ast' 'bpy' 'gsw' 'oc' 'os' 'sw' 'la' 'sh' 'mhr' 'xmf' 'nds' 'tk' 'ce' 'arz' 'br' 'mt' 'uz' 'azb' 'lb' 'mg' 'sah' 'cv' 'sa' 'pnb' 'sd' 'fy' 'ceb' 'ms' 'nn' 'ga' 'ba' 'yi' 'as' 'ku' 'dv' 'ug' 'af' 'lo' 'hr' 'cy' 'am' 'ps' 'tg' 'ky' 'or' 'bo' 'ckb' 'tl' 'eo' 'tt' 'pa' 'eu' 'gl' 'si' 'km' 'mn' 'gu')
LANGUAGES=('min')

for lang in "${LANGUAGES[@]}"; do
  echo "lang: $lang"
  /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py create-shards "cache/sharded_deduplicated_${lang}_v2" 1 --path "oscar-corpus/OSCAR-2109" --name "deduplicated_${lang}" --split "train"
  (time /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py build-hashes "cache/sharded_deduplicated_${lang}_v2/hashes_00001" --data-files "sharded_00000.jsonl" --path "cache/sharded_deduplicated_${lang}_v2" --split "train") |& tee "cache/sharded_deduplicated_${lang}_v2/1-log.txt"
  (time /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py build-index "cache/sharded_deduplicated_${lang}_v2/simhash_index.pkl" "cache/sharded_deduplicated_${lang}_v2/hashes_00001" --split "train" --threshold 1) |& tee "cache/sharded_deduplicated_${lang}_v2/2-log.txt"

  /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py create-shards "cache/sharded_deduplicated_${lang}_v1" 1 --path "oscar" --name "unshuffled_deduplicated_${lang}" --split "train"
  (time /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py build-hashes "cache/sharded_deduplicated_${lang}_v1/hashes_00001" --data-files "sharded_00000.jsonl" --path "cache/sharded_deduplicated_${lang}_v1" --split "train") |& tee "cache/sharded_deduplicated_${lang}_v2/1-log.txt"
  (time LOG_LEVEL="INFO" /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py find-duplicates "cache/sharded_deduplicated_${lang}_v1/hashes_00001" "cache/sharded_deduplicated_${lang}_v2/simhash_index.pkl" --split "train") |& tee "cache/sharded_deduplicated_${lang}_v2/3-log.txt"

  (time LOG_LEVEL="INFO" /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py merge-meta \
    --data-dirs "cache/sharded_deduplicated_${lang}_v1/hashes_00001_duplicates" \
    --meta-data-dirs "cache/sharded_deduplicated_${lang}_v2/hashes_00001" --split "train") |& tee "cache/sharded_deduplicated_${lang}_v1/4-log.txt"

done

