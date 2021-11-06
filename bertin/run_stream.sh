#! /bin/bash
# From https://arxiv.org/pdf/1907.11692.pdf for base model
python -c "import jax; print('TPUs', jax.device_count())"
python ./run_mlm_flax_stream.py \
    --output_dir="./outputs" \
    --model_type="roberta" \
    --config_name="./configs/base" \
    --tokenizer_name="./configs/base" \
    --dataset_name="./mc4" \
    --dataset_config_name="es" \
    --train_file="path/to/mc4-es-train-50M-XXX.jsonl" \
    --max_seq_length="128" \
    --pad_to_max_length  \
    --per_device_train_batch_size="256" \
    --per_device_eval_batch_size="256" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --adam_epsilon="1e-6" \
    --learning_rate="6e-4" \
    --weight_decay="0.01" \
    --save_steps="10000" \
    --save_total_limit="5" \
    --warmup_steps="24000" \
    --overwrite_output_dir \
    --num_train_steps="250000" \
    --eval_steps="10000" \
    --dtype="bfloat16" \
    --logging_steps="500" 2>&1 | tee run_stream.log
