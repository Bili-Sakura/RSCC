#!/bin/bash

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="path/to/logs/train_${TIMESTAMP}.log"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detect GPUs

# DeepSpeed configuration
deepspeed=path/to/zero3.json

# Model configuration
llm=path/to/pretrained/model
cache_dir=path/to/cache  # Cache directory

# Training hyperparameters
lr=1e-6
mm_projector_lr=1e-5  # Projector-specific LR
vision_tower_lr=1e-6  # Vision encoder LR
batch_size=1
grad_accum_steps=4

# Training entry point
entry_file=path/to/train_qwen.py

# Dataset configuration
datasets="rscc_subset%100"
data_flatten=True

# Output configuration
run_name="sft_for_rscc_model"
output_dir=path/to/output/model

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path \"${llm}\" \
    --cache_dir ${cache_dir} \
    --dataset_use ${datasets} \
    --data_flatten ${data_flatten} \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels $((512*512*3)) \
    --min_pixels $((512*512*3)) \
    --eval_strategy \"no\" \
    --save_strategy \"steps\" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --vision_tower_lr ${vision_tower_lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type \"cosine\" \
    --logging_steps 10 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to none \
    --optim adamw_torch"

# Launch training with nohup and logging
nohup torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} > ${LOG_FILE} 2>&1 &

# Display log file location
echo "Training started. Logs are being written to: ${LOG_FILE}"
echo "You can track progress with: tail -f ${LOG_FILE}"