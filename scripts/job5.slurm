#!/bin/bash
#SBATCH --job-name=
#SBATCH --account=
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=ALL

# HuggingFace token for gated models (e.g. Llama-3.1)
export HF_TOKEN=" "
export HF_HOME=" "

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
/apps/python/3.12/bin/python 05_train_lora.py \
--data_dir "outputs/data/beauty" \
--dataset "beauty"  \
--sasrec_ckpt "outputs/checkpoints/beauty/sasrec_beauty.pt" \
--llm_model meta-llama/Llama-3.2-3B-Instruct \
--checkpoint_dir outputs/checkpoints/lora_3B_beauty_8 \
--results_dir outputs/results/lora_3B_beauty_8 \
--conditioning_mode "none" \
--lora_r 8 \
