#!/bin/bash
# This script is used for deduplicating oscar datasets

LANGUAGES=('vls' 'diq' 'cbk' 'lrc' 'rue' 'gv' 'ht' 'nap' 'sco' 'bar' 'mwl' 'ie' 'myv' 'pam' 'scn' 'rm' 'frr' 'tyv' 'so' 'dsb' 'bxr' 'eml' 'nah' 'mai' 'gn' 'vec' 'li' 'xal' 'wuu' 'kw' 'yo' 'bh' 'ia' 'bs' 'io' 'qu' 'su' 'av' 'wa' 'mrj' 'kv' 'an' 'jv' 'jbo' 'ilo' 'lmo' 'min' 'mzn' 'gd' 'hsb' 'vo' 'gom' 'krc' 'lez' 'pms' 'war' 'new' 'ast' 'bpy' 'gsw' 'oc' 'os' 'sw' 'la' 'sh' 'mhr' 'xmf' 'nds' 'tk' 'ce' 'arz' 'br' 'mt' 'uz' 'azb' 'lb' 'mg' 'sah' 'cv' 'sa' 'pnb' 'sd' 'fy' 'ceb' 'ms' 'nn' 'ga' 'ba' 'yi' 'as' 'ku' 'dv' 'ug' 'af' 'lo' 'hr' 'cy' 'am' 'ps' 'tg' 'ky' 'or' 'bo' 'ckb' 'tl' 'eo' 'tt' 'pa' 'eu' 'gl' 'si' 'km' 'mn' 'gu')

for lang in "${LANGUAGES[@]}"; do
  echo "lang: $lang"
  /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py create-shards "cache/sharded_deduplicated_${lang}" 1 --path "oscar-corpus/OSCAR-2109" --name "deduplicated_${lang}" --split "train"
  (time /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py build-hashes "cache/sharded_deduplicated_${lang}/hashes_00001" --data-files "sharded_00000.jsonl" --path "cache/sharded_deduplicated_${lang}" --split "train") |& tee "cache/sharded_deduplicated_${lang}/1-log.txt"
  (time /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py build-index "cache/sharded_deduplicated_${lang}/simhash_index.pkl" "cache/sharded_deduplicated_${lang}/hashes_00001" --split "train") |& tee "cache/sharded_deduplicated_${lang}/2-log.txt"
  (time LOG_LEVEL="INFO" /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py find-duplicates "cache/sharded_deduplicated_${lang}/hashes_00001" "cache/sharded_deduplicated_${lang}/simhash_index.pkl" --split "train" --threshold 2) |& tee "cache/sharded_deduplicated_${lang}/3-log.txt"
  (time /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py remove-duplicates "cache/sharded_deduplicated_${lang}/hashes_00001_duplicates" --split "train") |& tee "cache/sharded_deduplicated_${lang}/4-log.txt"
  /home/jovyan/conda/envs/data/bin/python /home/jovyan/data_tooling/ac_dc/deduplicate.py merge-shards "cache/sharded_deduplicated_${lang}/output"  "cache/sharded_deduplicated_${lang}/hashes_00001_deduplicated" --split "train"

done
