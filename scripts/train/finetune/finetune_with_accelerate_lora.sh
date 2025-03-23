# input arguments
TRAIN_FILE=$1
OUTPUT_DIR=$2
SEED=$3
SOURCE=$4

# wandb configuration
source secrets.sh
export WANDB_NAME=Qwen2.5-7B-Instruct_nnetnav_${SOURCE}_seed${SEED}_lora

# huggingface configuration
export HF_REPO_ID=$WANDB_NAME
export HF_REPO_REVISION=main
export HF_HOME=$HOME/.cache/huggingface

# configuration
OUTPUT_DIR="$2/qwen2.5-7b-instruct_nnetnav_${SOURCE}_seed=${SEED}"
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=64
MODEL_AND_TOKENIZER_NAME=/model-weights/Qwen2.5-7B-Instruct/
SCRIPT=$HOME/Workspace/open-instruct/open_instruct/finetune.py
DATASET_LOCAL_CACHE_DIR=$HOME/.cache/
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training ${MODEL_AND_TOKENIZER_NAME} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.

accelerate launch \
    --config_file configs/ds_configs/stage3_qlora.yaml \
    --main_process_port 29501 \
    $SCRIPT \
    --model_name_or_path $MODEL_AND_TOKENIZER_NAME \
    --use_lora \
    --use_flash_attn \
    --use_liger_kernel \
    --gradient_checkpointing \
    --tokenizer_name $MODEL_AND_TOKENIZER_NAME \
    --dataset_mixer_list $TRAIN_FILE 1.0 \
    --max_seq_length 20000 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --logging_steps 1 \
    --checkpointing_steps "epoch" \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --hf_repo_id $HF_REPO_ID \
    --hf_repo_revision $HF_REPO_REVISION \
    --push_to_hub True \
    --exp_name $WANDB_NAME \
    --dataset_local_cache_dir $DATASET_LOCAL_CACHE_DIR
