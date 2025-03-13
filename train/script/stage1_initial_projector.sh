#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,7

N_PROC_PER_NODE=2
N_NODE=1

PORT=10086
ADDR=localhost

MODEL_NAME="/home/lijun2/multimodal/VideoNIAH/train/script/stage1/ckpt_final"
TOKENIZER_NAME="/home/lijun2/multimodal/checkpoints/Qwen2-VL-7B-Instruct"
PROCESSOR_NAME="/home/lijun2/multimodal/checkpoints/Qwen2-VL-7B-Instruct"
CONFIG_NAME="/home/lijun2/multimodal/VideoNIAH/train/config/config.json"
MAX_NUM_FRAMES=256
MAX_PIXELS=200476
LOAD_RESUME=true

MODEL_ARGS="
    --model_name_or_path=${MODEL_NAME} \
    --config_name=${CONFIG_NAME} \
    --tokenizer_name=${TOKENIZER_NAME} \
    --processor_name=${PROCESSOR_NAME} \
    --max_num_frames=${MAX_NUM_FRAMES} \
    --max_pixels=${MAX_PIXELS} \
    --load_resume=${LOAD_RESUME}
"

DATA_PATH="/data/public/multimodal/yuanziqi/datasets/pretraining_datasets/lmms-lab_LLaVA-ReCap-558K"
IMAGE_PATH="None"
VIDEO_PATH="None"

DATA_ARGS="
    --data_path=${DATA_PATH} \
    --image_path=${IMAGE_PATH} \
    --video_path=${VIDEO_PATH}
"

ENCODER_LEARNING_RATE=1e-3
ENCODER_STEP_MAX=5000
FREEZE_ENCODER=true
PROJECTOR_LEARNING_RATE=1e-3
PROJECTOR_STEP_MAX=5000
FREEZE_PROJECTOR=false
COMPRESSOR_LEARNING_RATE=1e-4
COMPRESSOR_STEP_MAX=5000
FREEZE_COMPRESSOR=false
LLM_LEARNING_RATE=1e-6
LLM_STEP_MAX=5000
FREEZE_LLM=true

LOGGING_STEPS=2
OUTPUT_DIR="./output"
DO_TRAIN=true
DO_EVAL=false
GRADIENT_ACCUMULATION_STEPS=10
WEIGHT_DECAY=0.01
MAX_STEPS=-1
NUM_TRAIN_EPOCHS=3.0
MAX_GRAD_NORM=1.0
WARMUP_RATIO=0.1
LOGGING_DIR="./logs"
SAVE_STEPS=2000
SAVE_SAFETENSORS=true
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1

TRAINING_ARGS="
    --encoder_learning_rate=${ENCODER_LEARNING_RATE} \
    --encoder_step_max=${ENCODER_STEP_MAX} \
    --freeze_encoder=${FREEZE_ENCODER} \
    --projector_learning_rate=${PROJECTOR_LEARNING_RATE} \
    --projector_step_max=${PROJECTOR_STEP_MAX} \
    --freeze_projector=${FREEZE_PROJECTOR} \
    --compressor_learning_rate=${COMPRESSOR_LEARNING_RATE} \
    --compressor_step_max=${COMPRESSOR_STEP_MAX} \
    --freeze_compressor=${FREEZE_COMPRESSOR} \
    --llm_learning_rate=${LLM_LEARNING_RATE} \
    --llm_step_max=${LLM_STEP_MAX} \
    --freeze_llm=${FREEZE_LLM} \
    --logging_steps=${LOGGING_STEPS} \
    --output_dir=${OUTPUT_DIR} \
    --do_train=${DO_TRAIN} \
    --do_eval=${DO_EVAL} \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --weight_decay=${WEIGHT_DECAY} \
    --max_steps=${MAX_STEPS} \
    --num_train_epochs=${NUM_TRAIN_EPOCHS} \
    --max_grad_norm=${MAX_GRAD_NORM} \
    --warmup_ratio=${WARMUP_RATIO} \
    --logging_dir=${LOGGING_DIR} \
    --save_steps=${SAVE_STEPS} \
    --save_safetensors=${SAVE_SAFETENSORS} \
    --bf16 \
    --per_device_train_batch_size=${TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size=${EVAL_BATCH_SIZE}
"

CMD="torchrun \
    --nproc_per_node=$N_PROC_PER_NODE \
    --nnodes=$N_NODE \
    --master_addr=$ADDR \
    --master_port=$PORT \
    ../sft_qwen2_vl.py \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $DATA_ARGS"

nvidia-smi

echo "$CMD"

$CMD