#!/bin/bash
set -x

# GRPO launch script for the compression policy, tuned for a SINGLE local GPU dev
# box (your proposal's Sec 4.1 plan assumes the shared WITS SLURM cluster — this is
# the single-GPU version to get the pipeline running end-to-end first; scale
# n_gpus_per_node / batch sizes up once you move to the cluster).
#
# Hyperparameters below follow the proposal's Sec 3.5.4 explicitly where it states a
# value, and ORBIT's own defaults elsewhere:
#   - algorithm.adv_estimator=grpo, group size K=4                (Sec 3.5.4, Eq 3.3)
#   - clip_ratio_low=clip_ratio_high=0.2 (eps_clip)                (Sec 3.5.4: "DeepSeekMath defaults")
#   - use_kl_loss=True, kl_loss_coef=0.04 (beta), low_var_kl        (Sec 3.5.4: "beta=0.04", Eq 3.4)
#   - LoRA rank 16 on q_proj/k_proj/v_proj/o_proj                  (Sec 3.5.1)
#
# LoRA flags (lora_rank/lora_alpha/target_modules) are written against verl's
# documented PEFT support but NOT verified against your installed verl version here
# (verl wasn't available to inspect in the sandbox this was written in — it's a
# further dependency inside third_party/rllm, not vendored as source). Run with
# `--cfg job` (Hydra's dry-run flag) first, or just watch the first few seconds of
# stdout, to confirm these keys resolve before committing a long run:
#   python scripts/train_textworld_compression.py --cfg job <rest of the flags>

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1

TASKS_CONFIG=${TASKS_CONFIG:-configs/textworld_compression_config.yaml}
if [ ! -f "$TASKS_CONFIG" ]; then
    echo "Error: Tasks config file not found: $TASKS_CONFIG"
    exit 1
fi

# AGENT_NAME selects which policy trains: compressive_agent (the actual research
# contribution) | static_append_agent | heuristic_summary_agent (baselines — you
# would not normally GRPO-train these, but the switch exists for completeness/debug).
AGENT_NAME=${AGENT_NAME:-compressive_agent}

# Point at the SFT warm-start checkpoint (Sec 3.5.2) once you have one, e.g.
#   MODEL_PATH=./checkpoints/sft_warmstart bash scripts/train_textworld_compression.sh
# Falls back to the base model for smoke-testing the pipeline before SFT is ready
# (expect near-zero reward and lots of malformed <summarise/> turns without SFT —
# that's the "silent advantage collapse" risk Sec 3.5.2 is designed to avoid, so
# don't judge the compression mechanism's quality from a no-SFT run).
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}

MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')
EXPERIMENT_NAME="textworld-compression-${AGENT_NAME}-${MODEL_NAME}"

python scripts/train_textworld_compression.py \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    +data.tasks_config_path="$TASKS_CONFIG" \
    rllm.agent.name="$AGENT_NAME" \
    rllm.agent.max_steps=40 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules="[q_proj,k_proj,v_proj,o_proj]" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    rllm.compact_filtering.enable=False \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=True \
    rllm.compact_filtering.mask_max_response_length_exceeded=True \
    rllm.compact_filtering.mask_max_turns_exceeded=False \
    rllm.compact_filtering.mask_timeout=True \
    rllm.rejection_sample.enable=False \
    rllm.stepwise_advantage.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='textworld-context-compression' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=25 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=60 \
    "$@"
