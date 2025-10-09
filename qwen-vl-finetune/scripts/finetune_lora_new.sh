#!/bin/bash

# ====================== 
# LoRA Fine-tuning Script for Qwen2.5-VL-7B
# ======================

# Configuration
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # Or local path to downloaded model
DATA_PATH="./data/your_dataset.json"        # Your training data
IMAGE_FOLDER="./data/images"                 # Your images folder
OUTPUT_DIR="./checkpoints/qwen2.5-vl-7b-lora"

# LoRA Configuration
USE_LORA=true
USE_QLORA=false  # Set to true for 4-bit quantization (saves memory)
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"  # Comma-separated

# Training Configuration  
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-4
NUM_TRAIN_EPOCHS=3
MAX_LENGTH=2048
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.1

# System Configuration
export CUDA_VISIBLE_DEVICES=0  # Adjust based on your GPU setup

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python scripts/train_lora.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --use_lora $USE_LORA \
    --use_qlora $USE_QLORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --freeze_vision_tower true \
    --freeze_language_model false \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --max_length $MAX_LENGTH \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --dataloader_num_workers 4 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to "none"

echo "Training completed! LoRA adapters saved to: $OUTPUT_DIR"