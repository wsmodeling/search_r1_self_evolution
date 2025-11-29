export CUDA_VISIBLE_DEVICES=0
export DATA_DIR='data/nq_search'

WAND_PROJECT='Search-R1-baseline2'

# Clean up any existing Ray instances to avoid actor registry conflicts
echo "Cleaning up existing Ray instances..."
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
sleep 2
rm -rf /dev/shm/ray_tmp
echo "Ray cleanup complete"

mkdir -p /dev/shm/ray_tmp
export RAY_TMPDIR=/dev/shm/ray_tmp
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
ray start --head \
  --port=6379 \
  --num-cpus=20 \
  --num-gpus=1 \
  --temp-dir=/dev/shm/ray_tmp \
  --node-ip-address=127.0.0.1
echo "Ray start complete"

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-em
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-it-em

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# update EXPERIMENT_NAME to avoid overwriting previous logs and checkpoints
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-em-test
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-em
export BASE_MODEL='Qwen/Qwen2.5-7B'
export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-em-test
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-it-em

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

# Memory Database Configuration
# Set memory_db.enable=true to enable storing low-reward responses and retrieving them for training
# The database will save responses with scores below low_reward_threshold and retrieve bad responses to augment training batches
# This helps the model learn from repeated exposure to challenging examples

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=8192 \
    data.max_response_length=500 \
    data.max_start_length=6144 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=False \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    enable_revision=true \
    enable_transfer_learning=false \
    enable_prompt_response_verification=false \
    2>&1 | tee $EXPERIMENT_NAME.log


# memory_db.enable=true \
# memory_db.db_path=./memory_db/responses_${EXPERIMENT_NAME}.json \
# memory_db.low_reward_threshold=0.3 \
# memory_db.retrieval_ratio=0.2 \
# memory_db.min_retrieval_score=-1.0 \
# memory_db.max_retrieval_score=0.5 \

# Memory Database Parameters Explained:
# - memory_db.enable: Set to true to enable, false to disable (default behavior without memory_db)
# - memory_db.db_path: JSON file path to store responses (use ${EXPERIMENT_NAME} for unique db per experiment)
# - memory_db.low_reward_threshold: Save responses with reward score < this value (e.g., 0.3)
# - memory_db.retrieval_ratio: Proportion of batch to augment with bad responses (0.2 = 20%)
# - memory_db.min_retrieval_score: Minimum score threshold for retrieving bad responses (-1.0)
# - memory_db.max_retrieval_score: Maximum score threshold for retrieving bad responses (0.5)
#
# To enable memory_db: Change memory_db.enable=false to memory_db.enable=true
# The feature is backward-compatible: if disabled or omitted, training works as before