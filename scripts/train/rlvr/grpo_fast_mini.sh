exp_name="base_smollm_grpo_${RANDOM}"
python open_instruct/grpo_fast.py \
    --exp_name $exp_name \
    --output_dir output/dummy \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 512 \
    --max_prompt_token_length 256 \
    --response_length 256 \
    --pack_length 512 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path HuggingFaceTB/SmolLM2-135M \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 10000 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --num_evals 20 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    # --with_tracking
