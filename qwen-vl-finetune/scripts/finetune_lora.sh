#!/bin/bash

# ----------------------------
# Distributed training config
# ----------------------------
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=2

# ----------------------------
# DeepSpeed configuration
# ----------------------------
# deepspeed=./scripts/zero2.json
deepspeed=./scripts/zero3_offload.json

# ----------------------------
# Model configuration
# ----------------------------
llm='/home/jack/Projects/yixin-llm/yixin-llm-data/MedicalGPT/weights/Qwen2.5-VL-7B-Instruct'

# ----------------------------
# Training hyperparameters
# ----------------------------
lr=5e-5
batch_size=2
grad_accum_steps=4
epochs=1

# Effective batch size = 1 * 2 * 128 = 256
# Steps per epoch = 212,449 / 256 = 830 steps
# Total steps = 830 steps

# ----------------------------
# Training entry point
# ----------------------------
entry_file=./qwenvl/train/train_qwen.py

# ----------------------------
# Dataset config
# ----------------------------
datasets=single_tool_multi_round,multi_tool_multi_round,multi_tool_single_round

# ----------------------------
# Output config
# ----------------------------
output_dir=/data3/qwen-weights/output_7b_final
run_name="qwen2vl-lora-final"

# ----------------------------
# Calculate dynamic save steps
# ----------------------------
# Save every ~10% of total steps
total_samples=228189
effective_bs=$(( NPROC_PER_NODE * batch_size * grad_accum_steps ))   # 2*1*128 = 256
total_steps=$(( total_samples / effective_bs ))                      # 830
save_steps=$(( total_steps / 4 ))                                    # 207

# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --lora_enable True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 25600 \
    --min_pixels 256 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    "

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

