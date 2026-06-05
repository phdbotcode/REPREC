#!/bin/bash
#SBATCH --job-name=
#SBATCH --account=
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --output=
#SBATCH --error=

# HuggingFace token for gated models (Llama-3.1)
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

cd scripts
/apps/python/3.12/bin/python 02_train_sasrec.py \
--data_dir "outputs/data/beauty" \
--dataset "beauty" \
--output_dir "outputs/checkpoints/beauty" \
--results_dir "outputs/results/beauty" \
--emb_dim 128 \
echo "========================================"
echo "Test job finished at: $(date)"
echo "========================================"
