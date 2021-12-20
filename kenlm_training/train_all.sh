#!/bin/bash
set -e

# Languages to train on
LANGUAGES_WIKIPEDIA=( "es" "af" "ar" "arz" "as" "bn" "fr" "sw" "eu" "ca" "zh" "en" "hi" "ur" "id" "pt" "vi" "gu" "kn" "ml" "mr" "ta" "te" "yo" )
LANGUAGES_OSCAR=( "es" "af" "ar" "arz" "as" "bn" "fr" "sw" "eu" "ca" "zh" "en" "hi" "ur" "id" "pt" "vi" "gu" "kn" "ml" "mr" "te" )

NDOC_FOR_LM=1_000_000
VOCAB_SIZE=65536
SMALL_VOCAB_SIZE=40000

# Normalization parameters
REMOVE_ACCENTS=False
LOWER_CASE=False
NORMALIZE_NUMBERS=True
NORMALIZE_PUNCT=1

# OSCAR
NDOC_FOR_LM_OSCAR=1_000_000


train_language_and_dataset () {
    local lang=$1
    local dataset=$2
    if [ "$dataset" = "wikipedia" ]; then
        # 1 Download Wikipedia cirrus
        if [ -f "data/${dataset}/cirrus/gz/${lang}.json.gz" ]; then
            echo "${lang} Wikipedia cirrus was already downloaded."
        else
            echo "Downloading ${lang}"
            mkdir -p "data/${dataset}/cirrus/gz/"
            python cc_net/get_wiki_cirrus.py dl --lang "${lang}" --output_dir "data/${dataset}/cirrus/gz" --date 20211115
            echo "Downloaded Wikipedia cirrus for ${lang}"
        fi

        # 2 Extract opening text of each article
        if [ -f "data/${dataset}/cirrus/gz/${lang}.opening.txt" ]; then
            echo "Wikipedia openings were already extracted for ${lang}"
        else
            echo "Extracting ${lang}"
            python cc_net/get_wiki_cirrus.py opening \
                --n_docs ${NDOC_FOR_LM} \
                --file "data/${dataset}/cirrus/gz/${lang}.json.gz" \
                --output "data/${dataset}/cirrus/gz/${lang}.opening.txt" \
                --accent ${REMOVE_ACCENTS} \
                --case ${LOWER_CASE} \
                --numbers ${NORMALIZE_NUMBERS} \
                --punct ${NORMALIZE_PUNCT}
        fi
    else
        # 1 & 2 Download and preprocess dataset from HF hub
        if [ -f "data/${dataset}/cirrus/gz/${lang}.opening.txt" ]; then
            echo "Wikipedia openings were already extracted for ${lang}"
        else
            echo "Downloading OSCAR ${lang}"
            mkdir -p "data/${dataset}/cirrus/gz/"
            python cc_net/get_hf_dataset.py dl \
                --dataset "${dataset}" \
                --output_file "data/${dataset}/cirrus/gz/${lang}.opening.txt" \
                --name "unshuffled_deduplicated_${lang}" \
                --split "train" \
                --max_docs $NDOC_FOR_LM_OSCAR
        fi
    fi

    # 3 Train sentence piece tokenizer
    if [ -f "data/${dataset}/lm_sp/${lang}.sp.model" ]; then
        echo "Sentence piece tokenizer was already trained for ${lang}"
    else
        echo "Training sentence piece tokenizer for ${lang}"
        mkdir -p "data/${dataset}/lm_sp"
        ./bin/spm_train --input="data/${dataset}/cirrus/gz/${lang}.opening.txt" \
            --vocab_size=${VOCAB_SIZE} --hard_vocab_limit \
            --character_coverage=0.9995 \
            --model_type=unigram \
            --model_prefix="data/${dataset}/lm_sp/${lang}.sp" \
        || echo "WARNING: Corpus is too small, will train smaller model" && \
        ./bin/spm_train --input="data/${dataset}/cirrus/gz/${lang}.opening.txt" \
            --vocab_size=${SMALL_VOCAB_SIZE} \
            --character_coverage=0.9995 \
            --model_type=unigram \
            --model_prefix="data/${dataset}/lm_sp/${lang}.sp"

        echo "Trained SentencePiece model with $(wc -l data/"${dataset}"/lm_sp/"${lang}".sp.vocab) pieces"
    fi

    # 4 Tokenize openings dataset
    if [ -f "data/${dataset}/cirrus/sp/${lang}.opening.txt" ]; then
        echo "Openings dataset already tokenized for ${lang}"
    else
        mkdir -p "data/${dataset}/cirrus/sp"
        echo "Tokenizing openings dataset for ${lang}"
        ./bin/spm_encode \
            --model="data/${dataset}/lm_sp/${lang}.sp.model" \
            --output_format=piece \
            "data/${dataset}/cirrus/gz/${lang}.opening.txt" > "data/${dataset}/cirrus/sp/${lang}.opening.txt"
        echo "Tokenized openings dataset for ${lang}"
    fi

    # 5 Train KenLM model on tokenized dataset
    if [ -f "data/${dataset}/lm_sp/${lang}.arpa" ] || [ -f "data/${dataset}/lm_sp/${lang}.arpa.bin" ]; then
        echo "KenLM model already trained for ${lang}"
    else
        echo "Training KenLM model for ${lang}"
        mkdir -p tmp
        ./bin/lmplz -o 5 -S 8G -T tmp --vocab_estimate ${VOCAB_SIZE}  --discount_fallback \
            < "data/${dataset}/cirrus/sp/${lang}.opening.txt" > "data/${dataset}/lm_sp/${lang}.arpa"
        echo "Trained KenLM model for ${lang}"
    fi

    # 6 Convert KenLM model to binary
    if [ -f "data/${dataset}/lm_sp/${lang}.arpa.bin" ]; then
        echo "KenLM model already converted to binary for ${lang}"
    else
        echo "Converting KenLM model to binary for ${lang}"
        ./bin/build_binary "data/${dataset}/lm_sp/${lang}.arpa" "data/${dataset}/lm_sp/${lang}.arpa.bin"
        echo "Converted KenLM model to binary for ${lang}"
        rm "data/${dataset}/lm_sp/${lang}.arpa"
    fi

}



for lang in "${LANGUAGES_WIKIPEDIA[@]}"
do
    train_language_and_dataset "$lang" wikipedia
done

for lang in "${LANGUAGES_OSCAR[@]}"
do
    train_language_and_dataset "$lang" oscar
done
