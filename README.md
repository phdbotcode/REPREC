# Frozen but Informed: Can User Representation Injection Enable Effective Sequential Recommendation?
 
> **REPREC** — Representation-driven Parameter-Efficient Recommendation system.
> Codebase for the paper submitted to RecSys 2026.
 
---
 
## What is this?
 
Modern LLM-based recommender systems typically improve personalization by fine-tuning the LLM backbone — an approach that is expensive, couples the recommendation model to a specific LLM version, and requires full retraining whenever the underlying encoder or LLM changes.
 
**REPREC asks: do you actually need to fine-tune the LLM at all?**
 
Inspired by projection-based conditioning in multimodal learning (ClipCap, BLIP-2), we reformulate sequential recommendation as a **user representation alignment problem**. A frozen SASRec encoder produces a compact user embedding. A small learned MLP — the *injector* — maps that embedding into a set of soft prefix tokens prepended to a frozen LLaMA-3B backbone. The LLM is never modified. Only the injector (~2.4M parameters) is trained end-to-end.
 
This design has three practical properties:
- **Parameter-efficient**: the injector is ~0.08% of the full model's parameters.
- **Modular**: the encoder and LLM can be swapped independently without retraining both.
- **Cold-user friendly**: because the behavioral signal is injected via the user embedding (not via long textual histories), REPREC consistently outperforms LoRA fine-tuning for cold and medium users across all five benchmark datasets.
---
 
## Architecture
<img width="4312" height="2099" alt="Picture1" src="https://github.com/user-attachments/assets/53f32abc-6e4d-488b-8c0b-70860fc4faf2" />


## Results Summary
 
**Overall performance** (HIT@5, m=6, max_history=50 vs LoRA r=8 at matched parameter budget):
<img width="977" height="332" alt="Screenshot 2026-06-05 at 14 40 06" src="https://github.com/user-attachments/assets/500c710e-0149-40f3-b4b3-42dbd0789bb2" />


**Cold, Medium, and Warm User Analysis**
<img width="872" height="778" alt="hit10_clean" src="https://github.com/user-attachments/assets/f92cbf17-640d-4e8a-b26f-78f25036f59f" />


**Efficiency (REPREC_cheap: train ℓ=10, evaluate ℓ=50)**
<img width="2088" height="892" alt="warm_users_comparison" src="https://github.com/user-attachments/assets/2720b25e-2ed1-4bf9-a26d-1ba36028411a" />


<img width="1488" height="314" alt="image" src="https://github.com/user-attachments/assets/ddb35681-9dbf-401a-b29c-a6144d04d311" />


Average speedup: 1.51× with 0.82×–1.00× of LoRA warm-user performance.

## Repository Structure
```
REPREC/
├── run_all_injector.sh             # ⚡ One-shot end-to-end pipeline (start here)
├── scripts/                        # Executable pipeline scripts + SLURM jobs
│   ├── 01_prepare_data.py          # Download and preprocess Amazon datasets
│   ├── 02_train_sasrec.py          # Train the SASRec sequential encoder
│   ├── 03_build_llm_pairs.py       # Build (user, item, label) pairs for LLM training
│   ├── 04_train_injector.py        # Train the REPREC injector MLP
│   ├── 05_train_lora.py            # Train the LoRA baseline
│   ├── 06_eval_ranker.py           # Evaluate any trained model (HIT@K)
│   ├── job2.slurm                  # SLURM job: train SASRec
│   ├── job3.slurm                  # SLURM job: build LLM pairs
│   ├── job4.slurm                  # SLURM job: train injector
│   └── job5_lora.slurm             # SLURM job: train LoRA
│
└── src/llm4rec/                    # Core library
    ├── data/                       # Dataset loading, sequence construction, negative sampling
    ├── evaluation/                 # HIT@K evaluation, user-regime splitting
    ├── llm/                        # LLM loading, prompt templates, answer extraction
    ├── plotting/                   # Figure generation (parameter vs. HIT@5, user regime bars)
    ├── sasrec/                     # SASRec model definition and training loop
    ├── training/                   # Injector MLP, LoRA training utilities
    ├── utils/                      # Logging, checkpointing, reproducibility helpers
    └── __init__.py
```
 
---
 
## Datasets
 
We evaluate on five Amazon product review benchmarks (2018 release). All data is preprocessed with the standard 5-core filter (retain users and items with ≥5 interactions).
 
| Dataset | Users | Items | Interactions | Avg. Seq. Len | Sparsity |
|---------|-------|-------|-------------|---------------|---------|
| Beauty | 22,363 | 12,101 | 198,498 | 8.88 | 99.927% |
| Sports & Outdoors | 35,598 | 18,357 | 296,241 | 8.32 | 99.955% |
| Toys & Games | 19,412 | 11,924 | 167,247 | 8.62 | 99.928% |
| Pet Supplies | 19,856 | 8,510 | 157,836 | 7.95 | 99.907% |
| Tools & Home | 16,638 | 10,217 | 134,476 | 8.08 | 99.921% |
 
**Split**: last interaction → test, second-to-last → validation, remainder → train.
 
**Negative sampling**: 200 negatives per test user, fixed pre-training with `seed=2024`. Hybrid strategy: 100 popularity-weighted (Laplace-smoothed) + 100 uniform random. Identical sets shared across SASRec, LoRA, and all REPREC variants to ensure fair comparison.
 
**Download and preprocess:**
```bash
cd scripts
python 01_prepare_data.py --dataset beauty --output_dir outputs/data/beauty
```
Replace `beauty` with `sports`, `toys`, `pet_supplies`, or `tools_home` as needed.
 
---
 
## Installation
 
```bash
git clone https://github.com/phdbotcode/REPREC.git
cd REPREC
```
 
**Requirements**: Python 3.12, PyTorch, HuggingFace Transformers, PEFT, a GPU with ≥40GB VRAM (A100 recommended for full training).
 
You will need a HuggingFace token to access `meta-llama/Llama-3.2-3B-Instruct`:
```bash
export HF_TOKEN=your_token_here
export HF_HOME=/path/to/hf_cache
```
 
---
 
## Quick Start
 
To run the full pipeline (data prep → SASRec → LLM pairs → injector) in a single SLURM submission, use `run_all_injector.slurm`. **Before submitting**, fill in the SLURM header and `HF_TOKEN` at the top of the file, then submit:
 
 
```bash
export HF_TOKEN=hf_your_token_here
export HF_HOME=/path/to/hf_cache
 
sbatch run_all_injector.slurm
```

**All available options:**
 
| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | `beauty` \| `sports` \| `toys` \| `pet_supplies` \| `tools_home` |
| `--base_dir` | `outputs` | Root directory for all checkpoints, results, and logs |
| `--emb_dim` | `64` | SASRec embedding dimension |
| `--num_soft_tokens` | `6` | Number of soft prefix tokens (`m`) |
| `--hidden_dim` | `128` | Injector MLP hidden dimension |
| `--num_epochs` | `3` | Injector training epochs |
| `--max_history` | `50` | Max prompt history length at training time |
| `--llm_model` | `meta-llama/Llama-3.2-3B-Instruct` | HuggingFace model ID |
| `--skip_data_prep` | `false` | Skip Step 1 if data already prepared |
| `--skip_sasrec` | `false` | Skip Step 2 if SASRec checkpoint exists |
| `--skip_pairs` | `false` | Skip Step 3 if LLM pairs already built |
 
**Common usage patterns:**
 
```bash
# Standard run — full pipeline on Sports
bash run_all_injector.sh --dataset sports
 
# REPRECcheap variant — train on short history, evaluate on long (1.51x speedup)
bash run_all_injector.sh --dataset beauty --max_history 10
 
# Resume after a failed run (data and SASRec already done)
bash run_all_injector.sh --dataset toys --skip_data_prep --skip_sasrec --skip_pairs
 
# Reproduce soft-token ablation
bash run_all_injector.sh --dataset beauty --num_soft_tokens 2
bash run_all_injector.sh --dataset beauty --num_soft_tokens 4
bash run_all_injector.sh --dataset beauty --num_soft_tokens 8
```
 
The injector checkpoint is saved to `outputs/checkpoints/injector_m{m}_h{hidden}_hist{history}_{dataset}.pt`.
 
---
<img width="3974" height="1007" alt="reprec_param_efficiency" src="https://github.com/user-attachments/assets/b8cd1687-3454-4558-a8be-0efe2ff2df9b" />



## Running the Pipeline (Step by Step)
 
Each step can also be run individually. This is useful when running on HPC with separate SLURM jobs, or when debugging a single stage.
 
### Step 1 — Prepare data
 
```bash
python scripts/01_prepare_data.py \
  --dataset beauty \
  --output_dir outputs/data/beauty
```
 
### Step 2 — Train SASRec encoder
 
```bash
python scripts/02_train_sasrec.py \
  --data_dir outputs/data/beauty \
  --dataset beauty \
  --output_dir outputs/checkpoints/beauty \
  --results_dir outputs/results/beauty \
  --emb_dim 128
```
 
Trains a two-block, two-head SASRec model for 150 epochs with early stopping (patience=50). The frozen checkpoint is used by all downstream steps.
 
### Step 3 — Build LLM training pairs
 
```bash
python scripts/03_build_llm_pairs.py \
  --data_dir outputs/data/beauty \
  --dataset beauty \
  --sasrec_ckpt outputs/checkpoints/beauty/sasrec_beauty.pt
```
 
Constructs binary (user, candidate item, yes/no label) pairs with the default prompt template and saves them for injector and LoRA training.
 
### Step 4 — Train REPREC injector
 
```bash
python scripts/04_train_injector.py \
  --data_dir outputs/data/beauty \
  --dataset beauty \
  --sasrec_ckpt outputs/checkpoints/beauty/sasrec_beauty.pt \
  --llm_model meta-llama/Llama-3.2-3B-Instruct \
  --output_ckpt outputs/checkpoints/injector_3B_6_beauty.pt \
  --output_dir outputs/results/injector_3B_6_beauty \
  --num_soft_tokens 6 \
  --hidden_dim 128 \
  --num_epochs 3
```
 
Only the injector MLP is trained. The SASRec encoder and LLaMA backbone remain fully frozen throughout.
 
### Step 5 — Train LoRA baseline (optional)
 
```bash
python scripts/05_train_lora.py \
  --data_dir outputs/data/beauty \
  --dataset beauty \
  --sasrec_ckpt outputs/checkpoints/beauty/sasrec_beauty.pt \
  --llm_model meta-llama/Llama-3.2-3B-Instruct \
  --checkpoint_dir outputs/checkpoints/lora_3B_beauty_8 \
  --results_dir outputs/results/lora_3B_beauty_8 \
  --conditioning_mode none \
  --lora_r 8
```
 
Applies LoRA (r=8) to `q_proj` and `v_proj` across all 28 LLaMA transformer layers. No collaborative conditioning.
 
### Step 6 — Evaluate
 
```bash
python scripts/06_eval_ranker.py \
  --data_dir outputs/data/beauty \
  --dataset beauty \
  --model_type reprec          # or: lora, sasrec, zero_shot, frozen_injector
  --ckpt outputs/checkpoints/injector_3B_6_beauty.pt
```
 
Reports HIT@5 and HIT@10 overall and broken down by cold (1–5), medium (6–20), and warm (20+) user segments.
 
---
 
## SLURM Jobs (HPC)
 
All jobs are configured for a single A100 (40GB) node. Fill in `--job-name`, `--account`, `--output`, `--error`, `--mail-user`, `HF_TOKEN`, and `HF_HOME` before submitting.
 
| Job file | Script | 
|----------|--------|
| `job2.slurm` | `02_train_sasrec.py` |
| `job3.slurm` | `03_build_llm_pairs.py` |
| `job4.slurm` | `04_train_injector.py` |
| `job5_lora.slurm` | `05_train_lora.py` |

Submit in order:
```bash
sbatch scripts/job2.slurm
# wait for completion
sbatch scripts/job3.slurm
# wait for completion
sbatch scripts/job4.slurm   # REPREC
sbatch scripts/job5_lora.slurm  # LoRA baseline (can run in parallel with job4)
```
 
---

## Hyperparameters
 
**SASRec** (shared encoder across all methods):
 
| Parameter | Value |
|-----------|-------|
| Embedding dim | 64 |
| Attention heads | 2 |
| Transformer blocks | 2 |
| Dropout | 0.2 |
| Batch size | 128 |
| Learning rate | 1e-3 |
| Epochs | 150 (early stop patience 50) |
 
**Injector and LoRA** (LLM-based methods):
 
| Parameter | REPREC | LoRA |
|-----------|--------|------|
| LLM backbone | `meta-llama/Llama-3.2-3B-Instruct` | same |
| dtype | bf16 | bf16 |
| MLP hidden dim | 128 | — |
| MLP activation | GELU | — |
| MLP dropout | 0.1 | — |
| LoRA rank `r` | — | 8 |
| LoRA α | — | 16 |
| LoRA target modules | — | q_proj, v_proj |
| Batch size | 8 | 8 |
| Learning rate | 1e-4 | 2e-4 |
| Epochs | 3 | 3 |
| LR scheduler | cosine | cosine |
| Warmup ratio | 0.05 | 0.05 |
