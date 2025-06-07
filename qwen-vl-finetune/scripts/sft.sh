#!/bin/bash

# ----------------------------
# Distributed training config
# ----------------------------
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=1

# ----------------------------
# DeepSpeed configuration
# ----------------------------
# switched to a Zero-2 + offload config that works well with LoRA
deepspeed=./scripts/zero3_offload.json
# deepspeed=./scripts/deepspeed_zero2_offload.json

# ----------------------------
# Model configuration
# ----------------------------
# llm=Qwen/Qwen2.5-VL-7B-Instruct
llm='/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/weights/Qwen2.5-VL-7B-Instruct'

# ----------------------------
# Training hyperparameters
# ----------------------------
lr=2e-5
batch_size=4
grad_accum_steps=4

# ----------------------------
# Training entry point
# ----------------------------
entry_file=./qwenvl/train/train_qwen.py

# ----------------------------
# Dataset config
# ----------------------------
datasets=healthgpt_reconstruction

# ----------------------------
# Output config
# ----------------------------
run_name="qwen2vl-lora-baseline"
output_dir=./output
logging_dir="${output_dir}/tensorboard_logs"

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --logging_dir ${logging_dir} \
    --report_to tensorboard \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    "

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

