set -x

export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/alfworld/base.yaml"

# 请根据实际情况设置模型路径
ALFWORLD_MODEL_PATH=${ALFWORLD_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}

TRAIN_PATH="$PROJECT_DIR/data/corpus/alfworld/train.parquet"
VAL_PATH="$PROJECT_DIR/data/corpus/alfworld/validation.parquet"

PROJECT_NAME='ALFWorld_ARFT'
EXP_NAME='alfworld_step_level_adv'

python3 -m arft.main_agent_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$ALFWORLD_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$CONFIG_PATH" \
    reward_model.enable=False \
    custom_reward_function.path=recipe/alfworld/reward_fn.py \
    custom_reward_function.name=compute_score \
    critic.model.path="$ALFWORLD_MODEL_PATH" \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=5 "$@"

