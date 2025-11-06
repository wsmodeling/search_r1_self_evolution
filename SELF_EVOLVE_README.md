# Launch a local retrieval server.
source /workspace/wsmodeling/miniconda3/etc/profile.d/conda.sh && conda activate /workspace/username/retriever && cd /workspace/wsmodeling/search_r1_self_evolution

export HF_HOME=/workspace/username/hf_cache && export TRANSFORMERS_CACHE=$HF_HOME/transformers && export HF_DATASETS_CACHE=$HF_HOME/datasets && export HF_MODULES_CACHE=$HF_HOME/modules && bash retrieval_launch.sh

# Run GRPO in a separated terminal
source /workspace/wsmodeling/miniconda3/etc/profile.d/conda.sh && conda activate /workspace/wsmodeling/searchr1 && cd /workspace/wsmodeling/search_r1_self_evolution && wandb login
export RAY_TMPDIR=/workspace/wsmodeling/ray_tmp && CUDA_LAUNCH_BLOCKING=1 && bash train_grpo.sh