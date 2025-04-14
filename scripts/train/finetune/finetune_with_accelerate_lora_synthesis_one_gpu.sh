# input arguments
OUTPUT_DIR=/projects/r2llab/arthur/checkpoints/train_synthesizer

# wandb configuration
source secrets.sh
WANDB_NAME=Qwen2.5-1.5B-Instruct_synthesis
export WANDB_NAME=${WANDB_NAME:0:64} # this is to avoid wandb error
export WANDB_PROJECT=data-synthesis-training

# huggingface configuration
export HF_REPO_ID=$WANDB_NAME
export HF_REPO_REVISION=main
export HF_HOME=$HOME/.cache/huggingface

# configuration
NUM_GPUS=1
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=2
MODEL_AND_TOKENIZER_NAME=Qwen/Qwen2.5-1.5B-Instruct
SCRIPT=$HOME/Workspace/open-instruct/open_instruct/finetune.py
DATASET_LOCAL_CACHE_DIR="$HOME/.cache"
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
EPOCHS=3
echo "Training ${MODEL_AND_TOKENIZER_NAME} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    $SCRIPT \
    --model_name_or_path $MODEL_AND_TOKENIZER_NAME \
    --use_flash_attn \
    --use_liger_kernel \
    --max_seq_length 1536 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --dataset_mixer_list ArthurChen189/financial_news_sentiment_analysis_sft 1.0 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --logging_steps 1 \
    --report_to wandb \
    --wandb_entity $WANDB_ENTITY \
    --hf_repo_id $HF_REPO_ID \
    --hf_repo_revision $HF_REPO_REVISION \
    --push_to_hub True \
    --exp_name $WANDB_NAME \
    --wandb_project_name $WANDB_PROJECT \
    --dataset_local_cache_dir $DATASET_LOCAL_CACHE_DIR \
    --keep_last_n_checkpoints 1
