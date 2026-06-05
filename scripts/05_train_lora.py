#!/usr/bin/env python
"""05 — Train LoRA adapters on LLaMA for recommendation scoring.

Three conditioning modes (--conditioning_mode):

* **none** (Model A): LoRA-only on text prompts.
* **injector** (Model B): LoRA + trainable SoftPromptInjector MLP.
* **frozen_projector** (Model C): LoRA + frozen random projection.

Frozen components (always):
    - SASRec  (pre-trained in step 02)
    - LLaMA base weights

Trainable:
    - Model A: LoRA adapter weights only
    - Model B: LoRA adapter weights + injector MLP
    - Model C: LoRA adapter weights only (projector is frozen)

Inputs:
    outputs/data/{dataset}/llm_train_pairs.jsonl
    outputs/data/{dataset}/train_pair_embeddings.pt   (pair_idx → emb)
    outputs/data/{dataset}/user_embeddings.npz        (full-seq for eval)
    outputs/checkpoints/sasrec_{dataset}.pt

Outputs:
    outputs/checkpoints/lora_{dataset}/        (PEFT adapter dir)
    outputs/results/lora_{dataset}.json
    outputs/results/all_runs.jsonl  (append)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm4rec.data.datasets import LLMPairDataset, PrebuiltPairDataset, load_pairs_jsonl
from llm4rec.data.negatives import load_negatives
from llm4rec.evaluation.compute import build_compute_record, count_params
from llm4rec.evaluation.rank_eval import evaluate_ranker
from llm4rec.evaluation.report import append_run_summary, save_metrics
from llm4rec.llm.collate import LLMCollator
from llm4rec.llm.llama_backbone import load_llama
from llm4rec.llm.lora import build_lora_model
from llm4rec.sasrec.infer import load_user_embeddings
from llm4rec.sasrec.model import SASRec
from llm4rec.training.train_lora import LoRATrainer
from llm4rec.utils.io import load_checkpoint, load_config, load_json
from llm4rec.utils.logging import get_logger
from llm4rec.utils.seed import set_seed

logger = get_logger(__name__)

VAL_NEG_QUICK = 100       # negatives per user for training-time validation
VAL_SUBSET_USERS = 5048   # random user subset for training-time validation (0 = all)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA adapters on LLaMA")
    parser.add_argument("--config", default="../configs/llm_lora.yaml")
    parser.add_argument("--sasrec_config", default="../configs/sasrec.yaml")
    parser.add_argument("--eval_config", default="../configs/eval.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sasrec_ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--conditioning_mode",
        choices=["none", "injector", "frozen_projector"],
        default="none",
        help="none = Model A (LoRA only), "
             "injector = Model B (LoRA + trainable injector), "
             "frozen_projector = Model C (LoRA + frozen random projection)",
    )
    parser.add_argument("--llm_model", default=None,
                        help="Override LLaMA model name (e.g. meta-llama/Llama-3.2-3B-Instruct)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override output checkpoint directory")
    parser.add_argument("--results_dir", default=None,
                        help="Override output results directory")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override number of training epochs (for quick testing)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early-stopping patience")
    parser.add_argument("--lora_r", type=int, default=None,
                        help="Override LoRA rank r (default: from config file)")
    parser.add_argument("--max_history", type=int, default=None,
                        help="Override max_history_items (default: from config file)")
    parser.add_argument("--sasrec_max_seq_len", type=int, default=None,
                        help="SASRec max_seq_len (overrides sasrec config), "
                             "must match the checkpoint's positional embedding size")
    args = parser.parse_args()

    cfg = load_config(args.config)["lora"]
    sasrec_cfg = load_config(args.sasrec_config)["sasrec"]
    eval_cfg = load_config(args.eval_config)["eval"]
    set_seed(cfg.get("seed", 42))

    # CLI overrides for quick testing
    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.patience is not None:
        cfg["training"]["patience"] = args.patience
    if args.lora_r is not None:
        cfg["r"] = args.lora_r
        logger.info("LoRA rank overridden to r=%d via --lora_r", args.lora_r)
    if args.max_history is not None:
        cfg["max_history_items"] = args.max_history
        logger.info("max_history_items overridden to %d via --max_history", args.max_history)

    # Allow config file to set conditioning_mode as a default
    conditioning_mode = args.conditioning_mode
    if conditioning_mode == "none":
        conditioning_mode = cfg.get("conditioning_mode", "none")
    logger.info("Conditioning mode: %s", conditioning_mode)

    # ── Load data ────────────────────────────────────────────────────────
    mappings = load_json(Path(args.data_dir) / "mappings.json")
    splits_raw = load_json(Path(args.data_dir) / "splits.json")
    splits = {k: {int(uid): v for uid, v in d.items()} for k, d in splits_raw.items()}
    negatives = load_negatives(str(Path(args.data_dir) / "negatives_200.jsonl"))
    num_items = mappings["num_items"]

    item_meta = None
    meta_path = Path(args.data_dir) / "item_meta.json"
    if meta_path.exists():
        raw = load_json(str(meta_path))
        item_meta = {int(k): v for k, v in raw.items()}

    # ── Load SASRec (frozen) ─────────────────────────────────────────────
    if args.sasrec_max_seq_len is not None:
        sasrec_cfg["max_seq_len"] = args.sasrec_max_seq_len
        logger.info("CLI override: sasrec_max_seq_len = %d", args.sasrec_max_seq_len)
    sasrec = SASRec(
        num_items=num_items,
        emb_dim=sasrec_cfg.get("embedding_dim", 64),
        max_len=sasrec_cfg.get("max_seq_len", 50),
        n_heads=sasrec_cfg.get("num_heads", 2),
        n_blocks=sasrec_cfg.get("num_blocks", 2),
        dropout=sasrec_cfg.get("dropout_rate", 0.2),
    )
    load_checkpoint(args.sasrec_ckpt, model=sasrec, device=args.device)
    sasrec.to(args.device).eval()
    logger.info("Loaded SASRec from %s (frozen)", args.sasrec_ckpt)

    # ── Load LLaMA + LoRA ────────────────────────────────────────────────
    llm_cfg = cfg["llm"]
    llm_model_name = args.llm_model or llm_cfg["model_name"]
    base_llm, tokenizer = load_llama(
        model_name=llm_model_name,
        dtype=llm_cfg.get("dtype", "bf16"),
        device_map=llm_cfg.get("device_map", "auto"),
        gradient_checkpointing=llm_cfg.get("gradient_checkpointing", True),
        train_mode=True,
    )

    llm_lora = build_lora_model(base_llm, cfg)
    logger.info("LLaMA + LoRA ready (r=%d, alpha=%d)",
                cfg.get("r", 8), cfg.get("lora_alpha", 16))

    # ── Load pre-computed embeddings ─────────────────────────────────────
    user_emb_path = str(Path(args.data_dir) / "user_embeddings_50128.npz")
    user_embs = load_user_embeddings(user_emb_path)
    logger.info("Loaded user embeddings (train-only): %d users", len(user_embs))

    user_emb_with_valid_path = str(Path(args.data_dir) / "user_embeddings_with_valid_50128.npz")
    user_embs_with_valid = load_user_embeddings(user_emb_with_valid_path)
    logger.info("Loaded user embeddings (with valid): %d users", len(user_embs_with_valid))

    # ── Build data loaders ───────────────────────────────────────────────
    max_history = cfg.get("max_history_items", 50)
    max_seq_len = cfg.get("max_seq_length", 512)
    batch_size = cfg["training"]["batch_size"]
    prompt_id = cfg.get("prompt_id", "2-11")

    # Training dataset: use PrebuiltPairDataset (all sub-sequence pairs)
    # for Models B/C (need pair_idx for prefix-aware embeddings),
    # and also for Model A (more training signal per user).
    train_pairs_path = str(Path(args.data_dir) / "llm_train_pairs_50128.jsonl")
    train_pairs = load_pairs_jsonl(train_pairs_path)
    train_ds = PrebuiltPairDataset(
        pairs=train_pairs,
        prompt_id=prompt_id,
        neg_per_pos=2,
        max_history=max_history,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=LLMCollator(tokenizer, max_length=max_seq_len, mode="train"),
        num_workers=0,
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )
    logger.info("Train: %d samples (from %d raw pairs in JSONL)",
                len(train_ds), len(train_pairs))
    # DEBUG 1: verify history truncation — show raw JSONL length vs after cap
    _n = min(5, len(train_ds.samples))
    _raw_lens  = [len(train_ds.samples[i]["history_text"]) for i in range(_n)]
    _cap_lens  = [min(len(train_ds.samples[i]["history_text"]), max_history) for i in range(_n)]
    _max_raw   = max(len(p["history_text"]) for p in train_ds.samples)
    logger.info("[DEBUG] Train history lengths — raw (JSONL): %s | after cap(%d): %s | max_raw_in_dataset=%d",
                _raw_lens, max_history, _cap_lens, _max_raw)
    # DEBUG 2: show a sample training prompt to confirm structure
    _sample_train = train_ds[0]
    logger.info("[DEBUG] Sample train prompt (pair_idx=%d, label=%d):\n%s",
                _sample_train["pair_idx"], _sample_train["label"],
                _sample_train["prompt"][:500])

    # Load pair-level embeddings for training (prefix-aware, indexed by pair_idx)
    pair_emb_table = None
    if conditioning_mode != "none":
        pair_emb_path = str(Path(args.data_dir) / "train_pair_embeddings_50128.pt")
        pair_emb_table = torch.load(pair_emb_path, map_location="cpu", weights_only=True)
        logger.info("Loaded pair embeddings: %s", tuple(pair_emb_table.shape))
    # DEBUG 4: confirm which embedding tables are active for this conditioning mode
    logger.info("[DEBUG] Embedding tables — pair_emb_table: %s | user_embs: %d users | conditioning_mode: %s",
                tuple(pair_emb_table.shape) if pair_emb_table is not None else "None",
                len(user_embs), conditioning_mode)

    # Validation dataset: subsample users + fewer negatives for fast training-time checks.
    # The user subset and negative pool are fixed across epochs for comparability.
    val_user_ids = sorted(splits["valid"].keys())
    total_val_users = len(val_user_ids)
    if VAL_SUBSET_USERS > 0 and total_val_users > VAL_SUBSET_USERS:
        rng = np.random.RandomState(cfg.get("seed", 42))
        val_user_ids = sorted(
            rng.choice(val_user_ids, size=VAL_SUBSET_USERS, replace=False).tolist()
        )
        logger.info("Training-time val: subsampled %d / %d users",
                     VAL_SUBSET_USERS, total_val_users)

    val_sequences = {uid: splits["train"][uid] for uid in val_user_ids
                     if uid in splits["train"]}
    val_targets = {uid: splits["valid"][uid] for uid in val_user_ids}
    val_negatives = {uid: negatives[uid][:VAL_NEG_QUICK] for uid in val_user_ids
                     if uid in negatives}

    val_ds = LLMPairDataset(
        user_sequences=val_sequences,
        targets=val_targets,
        negatives=val_negatives,
        item_meta=item_meta,
        max_history=max_history,
        prompt_id=prompt_id,
        mode="eval",
        neg_per_pos=VAL_NEG_QUICK,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=LLMCollator(tokenizer, max_length=max_seq_len, mode="eval"),
        num_workers=0,
    )
    logger.info("Training-time val: %d samples (%d users, %d neg/user, fixed pool)",
                 len(val_ds), len(val_targets), VAL_NEG_QUICK)
    # DEBUG 3: show a sample val prompt to compare structure with train
    _sample_val = val_ds[0]
    logger.info("[DEBUG] Sample val prompt (user_idx=%d, label=%d):\n%s",
                _sample_val["user_idx"], _sample_val["label"],
                _sample_val["prompt"][:500])

    # ── Conditioning module ──────────────────────────────────────────────
    injector = None
    projector = None

    if conditioning_mode == "injector":
        # Model B: trainable injector
        from llm4rec.llm.injector import SoftPromptInjector
        inj_cfg = cfg.get("injector", {})
        injector = SoftPromptInjector(
            user_dim=sasrec_cfg.get("embedding_dim", 128),
            llm_dim=base_llm.config.hidden_size,
            n_soft_tokens=inj_cfg.get("num_soft_tokens", 6),
            hidden_dim=inj_cfg.get("hidden_dim", 128),
            dropout=inj_cfg.get("dropout", 0.1),
            activation=inj_cfg.get("activation", "gelu"),
        )
        inj_params = count_params(injector)
        logger.info(
            "Model B: LoRA + Injector config: user_dim=%d, llm_dim=%d, "
            "n_soft_tokens=%d, hidden_dim=%d, dropout=%.2f, activation=%s",
            sasrec_cfg.get("embedding_dim",128), base_llm.config.hidden_size,
            inj_cfg.get("num_soft_tokens", 6), inj_cfg.get("hidden_dim", 128),
            inj_cfg.get("dropout", 0.1), inj_cfg.get("activation", "gelu"),
        )
        logger.info("Injector params: total=%d, trainable=%d",
                     inj_params["total"], inj_params["trainable"])

    elif conditioning_mode == "frozen_projector":
        # Model C: frozen random projection
        from llm4rec.llm.frozen_projector import FrozenSoftPromptProjector
        fp_cfg = cfg.get("frozen_projector", {})
        projector = FrozenSoftPromptProjector(
            d_in=sasrec_cfg.get("embedding_dim", 64),
            d_out=base_llm.config.hidden_size,
            n_soft_tokens=fp_cfg.get("n_soft_tokens", 8),
            per_token=fp_cfg.get("per_token", True),
            scale=fp_cfg.get("scale", 0.1),
            normalize=fp_cfg.get("normalize", True),
        )
        logger.info(
            "Model C: LoRA + FrozenProjector (n_soft_tokens=%d, "
            "per_token=%s, scale=%.4f, normalize=%s)",
            fp_cfg.get("n_soft_tokens", 8),
            fp_cfg.get("per_token", True),
            fp_cfg.get("scale", 0.1),
            fp_cfg.get("normalize", True),
        )

    else:
        logger.info("Model A: LoRA-only (no conditioning)")

    # ── Train ────────────────────────────────────────────────────────────
    mode_tag = {"none": "lora", "injector": "lora_injector",
                "frozen_projector": "lora_frozen_proj"}[conditioning_mode]
    ckpt_base = args.checkpoint_dir or cfg.get("checkpoint_dir", "outputs/checkpoints")
    save_dir = str(Path(ckpt_base) / f"{mode_tag}_{args.dataset}")

    trainer = LoRATrainer(
        sasrec=sasrec,
        llm_lora=llm_lora,
        tokenizer=tokenizer,
        cfg=cfg["training"],
        device=args.device,
        injector=injector,
        projector=projector,
    )

    # Log trainable parameter count
    trainable_params = trainer._trainable_params()
    total_trainable = sum(p.numel() for p in trainable_params)
    logger.info("Total trainable parameters: %d", total_trainable)

    # ── Estimate training compute ─────────────────────────────────────────
    # Correct fine-tuning FLOPs formula:
    #   forward:  2 × P_total  (reads all params, frozen + trainable)
    #   backward: 2 × P_total  (chain rule through all layers)
    #   update:   2 × P_trainable  (gradient + weight update for LoRA only)
    #   gradient_checkpointing recomputes activations (+2 × P_total extra fwd pass)
    # → No grad_ckpt: (4 × P_total + 2 × P_trainable) × T
    # → With grad_ckpt: (6 × P_total + 2 × P_trainable) × T
    total_llm_params = sum(p.numel() for p in llm_lora.parameters())
    steps_per_epoch = len(train_loader)
    num_epochs_est = cfg["training"].get("num_epochs", 3)
    total_steps = steps_per_epoch * num_epochs_est
    max_seq = cfg.get("max_seq_length", 512)
    bs = cfg["training"]["batch_size"]
    tokens_per_step = bs * max_seq
    total_tokens = total_steps * tokens_per_step
    grad_ckpt = llm_cfg.get("gradient_checkpointing", False)
    fwd_bwd_factor = 6 if grad_ckpt else 4
    flops_per_token = fwd_bwd_factor * total_llm_params + 2 * total_trainable
    flops_per_step = flops_per_token * tokens_per_step
    total_flops = flops_per_step * total_steps
    logger.info("Compute estimate (LoRA, grad_ckpt=%s): "
                 "%d steps/epoch × %d epochs = %d total steps",
                 grad_ckpt, steps_per_epoch, num_epochs_est, total_steps)
    logger.info("  FLOPs/token ≈ %d×P_total + 2×P_trainable = %d×%d + 2×%d = %.2e",
                 fwd_bwd_factor, fwd_bwd_factor, total_llm_params, total_trainable,
                 flops_per_token)
    logger.info("  FLOPs/step ≈ %.2e  |  Total FLOPs ≈ %.2e",
                 flops_per_step, total_flops)

    # ── Average token length (L_effective = L_text + m soft tokens) ──────
    _token_lengths = []
    for _batch in train_loader:
        if "attention_mask" in _batch:
            _token_lengths.append(_batch["attention_mask"].sum(dim=1).float())
        if len(_token_lengths) >= 10:
            break
    if _token_lengths:
        avg_l_text = torch.cat(_token_lengths).mean().item()
        if conditioning_mode == "injector":
            m = cfg.get("injector", {}).get("num_soft_tokens", 6)
        elif conditioning_mode == "frozen_projector":
            m = cfg.get("frozen_projector", {}).get("n_soft_tokens", 6)
        else:
            m = 0
        avg_l_effective = avg_l_text + m
        logger.info(
            "Avg token length (train sample): L_text=%.1f | m=%d soft tokens | "
            "L_effective = L_text + m = %.1f",
            avg_l_text, m, avg_l_effective,
        )

    best_llm = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        pair_emb_table=pair_emb_table,
        user_emb_table=user_embs if conditioning_mode != "none" else None,
        patience=cfg["training"].get("patience", 5),
        eval_ranking_every=1,
        save_dir=save_dir,
    )
    logger.info("LoRA adapter saved → %s", save_dir)

    # ── Evaluate on VAL and TEST (full negatives, all users) ─────────────
    logger.info("Final evaluation: all users, full negatives (up to %d/user)",
                 max(len(v) for v in negatives.values()))
    val_metrics = evaluate_ranker(
        model=best_llm,
        tokenizer=tokenizer,
        splits=splits,
        negatives=negatives,
        item_meta=item_meta,
        user_emb_table=user_embs if conditioning_mode != "none" else None,
        injector=injector,
        projector=projector,
        split="valid",
        prompt_id=eval_cfg.get("prompt_id", "2-11"),
        max_history=cfg.get("max_history_items", 50),
        batch_size=eval_cfg["scoring"]["batch_size"],
        max_seq_length=eval_cfg["scoring"]["max_seq_length"],
        ks=tuple(eval_cfg["ks"]),
        device=args.device,
    )
    test_metrics = evaluate_ranker(
        model=best_llm,
        tokenizer=tokenizer,
        splits=splits,
        negatives=negatives,
        item_meta=item_meta,
        user_emb_table=user_embs_with_valid if conditioning_mode != "none" else None,
        injector=injector,
        projector=projector,
        split="test",
        prompt_id=eval_cfg.get("prompt_id", "2-11"),
        max_history=cfg.get("max_history_items", 50),
        batch_size=eval_cfg["scoring"]["batch_size"],
        max_seq_length=eval_cfg["scoring"]["max_seq_length"],
        ks=tuple(eval_cfg["ks"]),
        device=args.device,
    )
    logger.info("VAL  %s", val_metrics)
    logger.info("TEST %s", test_metrics)

    # ── Save metrics ─────────────────────────────────────────────────────
    results_dir = Path(args.results_dir or "outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "method": mode_tag,
        "dataset": args.dataset,
        "conditioning_mode": conditioning_mode,
        "seed": cfg.get("seed", 42),
        "lora_r": cfg.get("r", 8),
        "lora_alpha": cfg.get("lora_alpha", 16),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "checkpoint": save_dir,
    }
    save_metrics(metrics, str(results_dir / f"{mode_tag}_{args.dataset}.json"))

    # ── Run summary (with compute stats) ─────────────────────────────────
    num_train = len(train_ds)
    record = build_compute_record(
        method=mode_tag,
        model=best_llm,
        cfg=cfg["training"],
        num_train_samples=num_train,
        metrics={
            "NDCG@10": test_metrics.get("NDCG@10", 0.0),
            "HR@10": test_metrics.get("HR@10", 0.0),
        },
    )
    # Override with correct fine-tuning FLOPs:
    # (fwd_bwd_factor × P_total + 2 × P_trainable) × T
    record["estimated_flops"] = total_flops
    record["tokens_processed"] = total_tokens
    record["trainable_params"] = total_trainable
    record["dataset"] = args.dataset
    record["seed"] = cfg.get("seed", 42)
    record["lora_r"] = cfg.get("r", 8)
    record["lora_alpha"] = cfg.get("lora_alpha", 16)
    record["conditioning_mode"] = conditioning_mode
    append_run_summary(str(results_dir / "all_runs.jsonl"), record)
    logger.info("Done.")


if __name__ == "__main__":
    main()
