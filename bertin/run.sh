#!/bin/bash
# From https://arxiv.org/pdf/1907.11692.pdf
python -c "import jax; print('TPUs', jax.device_count())"
./run_mlm_flax.py \
    --output_dir="./outputs" \
    --model_type="roberta" \
    --config_name="./configs/large" \
    --tokenizer_name="./" \
    --dataset_name="mc4" \
    --dataset_config_name="es" \
    --dataset_streamnig \
    --max_seq_length="128" \
    --pad_to_max_length  \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="128" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --adam_epsilon="1e-6" \
    --learning_rate="4e-4" \
    --weight_decay="0.01" \
    --save_strategy="steps" \
    --save_steps="10000" \
    --save_total_limit="5" \
    --warmup_steps="30000" \
    --overwrite_output_dir \
    --num_train_steps="500000" \
    --eval_steps="10000" \
    --logging_steps="500" \
    --dtype="bfloat16" 2>&1 | tee run.log
