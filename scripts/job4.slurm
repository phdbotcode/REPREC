#!/bin/bash
#SBATCH --job-name=
#SBATCH --account=
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --output=
#SBATCH --error=


# HuggingFace token for gated models (e.g. Llama-3.1)
export HF_TOKEN=" "
export HF_HOME=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print GPU info
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"
nvidia-smi
echo "========================================"

# Run training
cd scripts
/apps/python/3.12/bin/python 04_train_injector.py \
--data_dir "outputs/data/beauty" \
--dataset "beauty"  \
--sasrec_ckpt "outputs/checkpoints/beauty/sasrec_beauty.pt" \
--llm_model meta-llama/Llama-3.2-3B-Instruct \
--output_ckpt outputs/checkpoints/injector_3B_6_beauty.pt \
--output_dir outputs/results/injector_3B_6_beauty \
--num_soft_tokens 6 \
--hidden_dim 128 \
--num_epochs 3 
