#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,7

N_PROC_PER_NODE=2
N_NODE=1

PORT=10086
ADDR=localhost

MODEL_NAME="/data/public/multimodal/yuanziqi/models/Qwen2.5-VL-7B-Instruct"
PROCESSOR_NAME="/data/public/multimodal/yuanziqi/models/Qwen2.5-VL-7B-Instruct"

MODEL_ARGS="
    --model_name_or_path=${MODEL_NAME} \
    --processor_name=${PROCESSOR_NAME} 
"

DATA_MIXTURE="videochat_flash_pretrain"
MAX_NUM_FRAMES=4
MAX_PIXELS=451584

DATA_ARGS="
    --data_mixture ${DATA_MIXTURE} \
    --max_num_frames=${MAX_NUM_FRAMES} \
    --max_pixels=${MAX_PIXELS}
"

MODEL_MAX_LENGTH=8192
OUTPUT_DIR="runs/train/qwen2.5_vl_align/model"

WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE=cosine
NUM_TRAIN_EPOCHS=1
GRADIENT_ACCUMULATION_STEPS=1
PER_DEVICE_TRAIN_BATCH_SIZE=4

ENCODER_LEARNING_RATE=1e-3
ENCODER_STEP_MAX=5000
FREEZE_ENCODER=true
PROJECTOR_LEARNING_RATE=1e-3
PROJECTOR_STEP_MAX=5000
FREEZE_PROJECTOR=false
COMPRESSOR_LEARNING_RATE=5e-3
COMPRESSOR_STEP_MAX=10000
FREEZE_COMPRESSOR=false
LLM_LEARNING_RATE=1e-5
LLM_STEP_MAX=5000
FREEZE_LLM=true

LOGGING_STEPS=2

DO_TRAIN=true
DO_EVAL=false


MAX_STEPS=-1

MAX_GRAD_NORM=1.0

LOGGING_DIR="./logs"
SAVE_STEPS=40000
SAVE_SAFETENSORS=true

EVAL_BATCH_SIZE=1
JUST_DEBUG=true

TRAINING_ARGS="
    --tune_vision_tower=false \
    --tune_language_model=false \
    --tune_projector=true \
    --projector_lr=1e-3 \
    --model_max_length=${MODEL_MAX_LENGTH} \
    --output_dir=${OUTPUT_DIR} \

    --weight_decay=${WEIGHT_DECAY} \
    --warmup_ratio=${WARMUP_RATIO} \
    --lr_scheduler_type=${LR_SCHEDULER_TYPE} \
    --num_train_epochs=${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --evaluation_strategy=no \
    --save_strategy=steps \
    --save_steps=100000 \ 
    --save_total_limit=1 \
    --bf16 \
    --gradient_checkpointing=true \
    --dataloader_num_workers=0 \
    --report_to=tensorboard
"

CMD="torchrun \
    --nproc_per_node=$N_PROC_PER_NODE \
    --nnodes=$N_NODE \
    --master_addr=$ADDR \
    --master_port=$PORT \
    ../train_qwen2_vl.py \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $DATA_ARGS"

nvidia-smi

echo "$CMD"

$CMD