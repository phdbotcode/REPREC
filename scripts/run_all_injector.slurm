#!/bin/bash
#SBATCH --job-name=
#SBATCH --account=
#SBATCH --time=130:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --output=
#SBATCH --error=

# =============================================================================
# run_all_injector.slurm
# End-to-end REPREC injector pipeline in a single SLURM job.
# Chains: data prep → SASRec → LLM pairs → injector training.
#
# Usage — submit with default settings:
#   sbatch run_all_injector.slurm
#
# Usage — skip steps already completed (pass via --export at submit time):
#   sbatch --export=ALL,SKIP_DATA_PREP=1                          run_all_injector.slurm
#   sbatch --export=ALL,SKIP_SASREC=1                             run_all_injector.slurm
#   sbatch --export=ALL,SKIP_PAIRS=1                              run_all_injector.slurm
#   sbatch --export=ALL,SKIP_DATA_PREP=1,SKIP_SASREC=1            run_all_injector.slurm
#   sbatch --export=ALL,SKIP_DATA_PREP=1,SKIP_SASREC=1,SKIP_PAIRS=1  run_all_injector.slurm
#
# Usage — override dataset or model settings at submit time:
#   sbatch --export=ALL,DATASET=sports                            run_all_injector.slurm
#   sbatch --export=ALL,DATASET=toys,NUM_SOFT_TOKENS=2            run_all_injector.slurm
# =============================================================================

# ── Environment ───────────────────────────────────────────────────────────────
# HuggingFace token for gated models (Llama-3.2-3B-Instruct)
export HF_TOKEN=" "
export HF_HOME=" "
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Config — edit defaults here, or override at submit time via --export ──────
DATASET="${DATASET:-beauty}"                    # beauty | sports | toys | pet_supplies | tools_home
LLM_MODEL="${LLM_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"

# SASRec (job2)
EMB_DIM="${EMB_DIM:-128}"

# Injector (job4)
NUM_SOFT_TOKENS="${NUM_SOFT_TOKENS:-6}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"

# Skip flags — set to 1 to skip, 0 to run (override via --export at submit time)
SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"           # skip 01_prepare_data.py
SKIP_SASREC="${SKIP_SASREC:-0}"                 # skip 02_train_sasrec.py
SKIP_PAIRS="${SKIP_PAIRS:-0}"                   # skip 03_build_llm_pairs.py

# ── Derived paths (match individual job conventions) ──────────────────────────
DATA_DIR="outputs/data/${DATASET}"
CKPT_DIR="outputs/checkpoints/${DATASET}"
SASREC_CKPT="outputs/checkpoints/${DATASET}/sasrec_${DATASET}.pt"
SASREC_RESULTS="outputs/results/${DATASET}"
INJECTOR_CKPT="outputs/checkpoints/injector_3B_${NUM_SOFT_TOKENS}_${DATASET}.pt"
INJECTOR_RESULTS="outputs/results/injector_3B_${NUM_SOFT_TOKENS}_${DATASET}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset:          $DATASET"
echo "LLM model:        $LLM_MODEL"
echo "Soft tokens (m):  $NUM_SOFT_TOKENS"
echo "SASRec emb dim:   $EMB_DIM"
echo "Injector epochs:  $NUM_EPOCHS"
echo "Skip data prep:   $SKIP_DATA_PREP"
echo "Skip SASRec:      $SKIP_SASREC"
echo "Skip pairs:       $SKIP_PAIRS"
echo "========================================"
nvidia-smi
echo "========================================"

cd scripts

# ── Step 1: Prepare data ──────────────────────────────────────────────────────
echo "========================================"
echo "STEP 1/4 — Data preparation"
echo "========================================"

if [[ "$SKIP_DATA_PREP" == "1" ]]; then
    echo "Skipped (SKIP_DATA_PREP=1)"
else
    echo "Started at: $(date)"
    /apps/python/3.12/bin/python 01_prepare_data.py \
        --dataset    "$DATASET" \
        --output_dir "$DATA_DIR"
    echo "Step 1 finished at: $(date)"
fi

echo "========================================"

# ── Step 2: Train SASRec (job2.slurm) ─────────────────────────────────────────
echo "========================================"
echo "STEP 2/4 — Train SASRec encoder"
echo "========================================"

if [[ "$SKIP_SASREC" == "1" ]]; then
    echo "Skipped (SKIP_SASREC=1)"
    if [[ ! -f "../$SASREC_CKPT" ]]; then
        echo "ERROR: SASRec checkpoint not found at $SASREC_CKPT"
        exit 1
    fi
    echo "Using existing checkpoint: $SASREC_CKPT"
else
    echo "Started at: $(date)"
    /apps/python/3.12/bin/python 02_train_sasrec.py \
        --data_dir    "$DATA_DIR" \
        --dataset     "$DATASET" \
        --output_dir  "$CKPT_DIR" \
        --results_dir "$SASREC_RESULTS" \
        --emb_dim     "$EMB_DIM"
    echo "Step 2 finished at: $(date)"
fi

echo "========================================"

# ── Step 3: Build LLM pairs (job3.slurm) ──────────────────────────────────────
echo "========================================"
echo "STEP 3/4 — Build LLM training pairs"
echo "========================================"

if [[ "$SKIP_PAIRS" == "1" ]]; then
    echo "Skipped (SKIP_PAIRS=1)"
else
    echo "Started at: $(date)"
    /apps/python/3.12/bin/python 03_build_llm_pairs.py \
        --data_dir    "$DATA_DIR" \
        --dataset     "$DATASET" \
        --sasrec_ckpt "$SASREC_CKPT"
    echo "Step 3 finished at: $(date)"
fi

echo "========================================"

# ── Step 4: Train injector (job4.slurm) ───────────────────────────────────────
echo "========================================"
echo "STEP 4/4 — Train REPREC injector"
echo "Started at: $(date)"
echo "========================================"

/apps/python/3.12/bin/python 04_train_injector.py \
    --data_dir         "$DATA_DIR" \
    --dataset          "$DATASET" \
    --sasrec_ckpt      "$SASREC_CKPT" \
    --llm_model        "$LLM_MODEL" \
    --output_ckpt      "$INJECTOR_CKPT" \
    --output_dir       "$INJECTOR_RESULTS" \
    --num_soft_tokens  "$NUM_SOFT_TOKENS" \
    --hidden_dim       "$HIDDEN_DIM" \
    --num_epochs       "$NUM_EPOCHS"

echo "========================================"
echo "Step 4 finished at: $(date)"
echo "========================================"

echo "========================================"
echo "All steps complete at: $(date)"
echo "Injector checkpoint : $INJECTOR_CKPT"
echo "Results directory   : $INJECTOR_RESULTS"
echo "========================================"
