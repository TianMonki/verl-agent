#!/usr/bin/env bash
# WebShop + SGPO (algorithm.adv_estimator=sgpo, recipe/sgpo/core_sgpo.py).
# lam=null cross-validates lambda per batch; lam=1.0 recovers GiGPO's step term. Assumes an 8-GPU node.
set -x

ENGINE=${ENGINE:-vllm}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
DATA_DIR=${DATA_DIR:-"$HOME/data/verl-agent/text"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sgpo_webshop}

# WebShop env package must be importable; pyserini needs a JDK 11+ JVM (jdk4py ships one).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/agent_system/environments/env_package/webshop/webshop:${PYTHONPATH:-}"
export JAVA_HOME="$(python3 -c 'import jdk4py; print(jdk4py.JAVA_HOME)' 2>/dev/null || echo "${JAVA_HOME:-}")"
[ -n "$JAVA_HOME" ] && export PATH="$JAVA_HOME/bin:$PATH"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=sgpo \
    algorithm.sgpo.dp_w=1.0 \
    algorithm.sgpo.lam=null \
    algorithm.gamma=0.95 \
    algorithm.use_kl_in_reward=False \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.003 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.history_length=2 \
    env.rollout.n=8 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=200 $@
