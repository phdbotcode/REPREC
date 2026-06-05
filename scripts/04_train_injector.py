#!/usr/bin/env python
"""04 — Train the soft-prompt injector (MLP projector) on LLM YES/NO pairs.

Frozen components:
    - SASRec  (pre-trained in step 02)
    - LLaMA   (base weights, never updated)

Trainable:
    - SoftPromptInjector MLP only

Inputs:
    outputs/data/{dataset}/llm_train_pairs.jsonl
    outputs/data/{dataset}/train_pair_embeddings.pt   (pair_idx → emb)
    outputs/data/{dataset}/user_embeddings.npz        (full-seq for eval)
    outputs/checkpoints/sasrec_{dataset}.pt

Outputs:
    outputs/checkpoints/injector_{dataset}.pt
    outputs/results/injector_{dataset}.json
    outputs/results/all_runs.jsonl  (append)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm4rec.bert4rec.model import BERT4Rec
from llm4rec.data.datasets import LLMPairDataset, PrebuiltPairDataset, load_pairs_jsonl
from llm4rec.data.negatives import load_negatives
from llm4rec.evaluation.compute import build_compute_record, count_params
from llm4rec.evaluation.rank_eval import evaluate_ranker
from llm4rec.evaluation.report import append_run_summary, save_metrics
from llm4rec.llm.collate import LLMCollator
from llm4rec.llm.injector import SoftPromptInjector
from llm4rec.llm.llama_backbone import load_llama
from llm4rec.sasrec.infer import load_user_embeddings
from llm4rec.sasrec.model import SASRec
from llm4rec.training.train_injector import InjectorTrainer
from llm4rec.utils.io import load_checkpoint, load_config, load_json
from llm4rec.utils.logging import get_logger
from llm4rec.utils.seed import set_seed

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Build pair-based data loaders
# ─────────────────────────────────────────────────────────────────────────────

NEG_PER_POS_TRAIN = 2     # negatives per positive during training
VAL_NEG_QUICK = 100       # negatives per user for training-time validation
VAL_SUBSET_USERS = 5048   # random user subset for training-time validation (0 = all)


def _truncate_negatives(
    negatives: dict, max_neg: int,
) -> dict:
    """Return a copy of *negatives* with at most *max_neg* items per user."""
    return {uid: negs[:max_neg] for uid, negs in negatives.items()}


def _build_pair_loaders(
    data_dir: str,
    tokenizer,
    cfg: dict,
    splits: dict,
    negatives: dict,
    item_meta: dict | None,
    val_subset_users: int = 0,
    val_num_negatives: int = 100,
    pairs_path: str | None = None,
):
    """Build train and validation DataLoaders.

    **Training** — loads pre-built pairs from ``llm_train_pairs.jsonl``
    (produced by ``03_build_llm_pairs.py``) via :class:`PrebuiltPairDataset`.
    Each sample carries ``pair_idx`` for correct embedding lookup into
    ``train_pair_embeddings.pt``.

    **Validation** — uses :class:`LLMPairDataset` in ``"eval"`` mode with
    ``user_idx`` for embedding lookup into ``user_embeddings.npz``.
    Uses a random subset of users (``val_subset_users``) and fewer
    negatives (``val_num_negatives``) for fast training-time checks.
    The user subset and negative pool are fixed across epochs for
    comparability.
    """
    max_history = cfg.get("max_history_items", 20)
    max_seq_len = cfg.get("max_seq_length", 512)
    batch_size = cfg["training"]["batch_size"]
    prompt_id = cfg.get("prompt_id", "2-11")

    # --- Train loader (from pre-built JSONL pairs) ---
    pairs_path = pairs_path or str(Path(data_dir) / "llm_train_pairs.jsonl")
    raw_pairs = load_pairs_jsonl(pairs_path)
    logger.info("Loaded %d raw pairs from %s", len(raw_pairs), pairs_path)

    train_ds = PrebuiltPairDataset(
        pairs=raw_pairs,
        prompt_id=prompt_id,
        neg_per_pos=NEG_PER_POS_TRAIN,
        max_history=max_history,
    )

    train_collator = LLMCollator(tokenizer, max_length=max_seq_len, mode="train")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=train_collator, num_workers=0,
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )

    # --- Val loader (for ranking eval during training) ---
    # Subsample users for faster training-time validation
    val_user_ids = sorted(splits["valid"].keys())
    total_val_users = len(val_user_ids)
    if val_subset_users > 0 and total_val_users > val_subset_users:
        rng = np.random.RandomState(42)  # fixed seed → same subset every run
        val_user_ids = sorted(
            rng.choice(val_user_ids, size=val_subset_users, replace=False).tolist()
        )
        logger.info("Training-time val: subsampled %d / %d users",
                     val_subset_users, total_val_users)

    # Filter splits and negatives to the selected user subset
    val_sequences = {uid: splits["train"][uid] for uid in val_user_ids
                     if uid in splits["train"]}
    val_targets = {uid: splits["valid"][uid] for uid in val_user_ids}
    val_negatives = _truncate_negatives(
        {uid: negatives[uid] for uid in val_user_ids if uid in negatives},
        val_num_negatives,
    )
    logger.info("Training-time val: %d users, %d negatives/user (fixed pool)",
                 len(val_targets), val_num_negatives)

    val_ds = LLMPairDataset(
        user_sequences=val_sequences,
        targets=val_targets,
        negatives=val_negatives,
        item_meta=item_meta,
        max_history=max_history,
        prompt_id=prompt_id,
        mode="eval",
        neg_per_pos=val_num_negatives,
    )

    val_collator = LLMCollator(tokenizer, max_length=max_seq_len, mode="eval")
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=val_collator, num_workers=0,
    )

    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train soft-prompt injector")
    parser.add_argument("--config", default="../configs/llm_injector.yaml")
    parser.add_argument("--sasrec_config", default="../configs/sasrec.yaml")
    parser.add_argument("--bert4rec_config", default="../configs/bert4rec.yaml")
    parser.add_argument("--eval_config", default="../configs/eval.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", default="sasrec",
                        choices=["sasrec", "bert4rec"],
                        help="Sequential encoder backbone (default: sasrec)")
    parser.add_argument("--sasrec_ckpt", default=None,
                        help="SASRec checkpoint (required when --backbone sasrec)")
    parser.add_argument("--bert4rec_ckpt", default=None,
                        help="BERT4Rec checkpoint (required when --backbone bert4rec)")
    parser.add_argument("--llm_model", default=None,
                        help="HuggingFace model name (overrides config), "
                             "e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_soft_tokens", type=int, default=None,
                        help="Number of soft prompt tokens (overrides config)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Injector MLP hidden dimension (overrides config)")
    parser.add_argument("--max_history_items", type=int, default=None,
                        help="Max history items per user (overrides config)")
    parser.add_argument("--sasrec_max_seq_len", type=int, default=None,
                        help="SASRec max_seq_len (overrides sasrec config), "
                             "must match the checkpoint's positional embedding size")
    parser.add_argument("--output_ckpt", default=None,
                        help="Custom checkpoint output path, "
                             "e.g. outputs/checkpoints/injector_beauty_v2.pt")
    parser.add_argument("--output_dir", default=None,
                        help="Custom results directory. Defaults to "
                             "outputs/results/<llm_short_name>")
    parser.add_argument("--pairs_path", default=None,
                        help="Override path to llm_train_pairs .jsonl "
                             "(default: data_dir/llm_train_pairs.jsonl)")
    parser.add_argument("--pair_emb_path", default=None,
                        help="Override path to train_pair_embeddings .pt "
                             "(default: data_dir/train_pair_embeddings.pt)")
    parser.add_argument("--user_emb_path", default=None,
                        help="Override path to user_embeddings .npz used for "
                             "valid-split eval (default: data_dir/user_embeddings.npz)")
    parser.add_argument("--user_emb_test_path", default=None,
                        help="Override path to user_embeddings_with_valid .npz used for "
                             "test-split eval (default: data_dir/user_embeddings_with_valid.npz). "
                             "These encode train+valid history so the user rep is correct "
                             "right before the test item.")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--random_embeddings", action="store_true",
                        help="Replace SASRec embeddings with a shared all-zero vector "
                             "for all users at both train and eval time. Tests whether "
                             "user-specific embedding content matters vs. the MLP "
                             "learning a fixed soft-prompt bias.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Validate backbone / checkpoint combination
    if args.backbone == "sasrec" and not args.sasrec_ckpt:
        parser.error("--sasrec_ckpt is required when --backbone sasrec")
    if args.backbone == "bert4rec" and not args.bert4rec_ckpt:
        parser.error("--bert4rec_ckpt is required when --backbone bert4rec")

    cfg = load_config(args.config)["injector"]
    sasrec_cfg = load_config(args.sasrec_config)["sasrec"]
    bert4rec_cfg = load_config(args.bert4rec_config)["bert4rec"]
    eval_cfg = load_config(args.eval_config)["eval"]
    set_seed(cfg.get("seed", 42))

    # ── Log experiment config ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Experiment config")
    logger.info("=" * 60)
    logger.info("Dataset: %s | Data dir: %s", args.dataset, args.data_dir)
    logger.info("SASRec ckpt: %s", args.sasrec_ckpt)
    llm_cfg_log = cfg["llm"]
    logger.info("LLM: %s | dtype: %s | device_map: %s | grad_ckpt: %s",
                llm_cfg_log["model_name"], llm_cfg_log.get("dtype", "bf16"),
                llm_cfg_log.get("device_map", "auto"),
                llm_cfg_log.get("gradient_checkpointing", False))
    logger.info("Injector: soft_tokens=%d | hidden=%d | dropout=%.2f | activation=%s",
                cfg.get("num_soft_tokens", 8), cfg.get("hidden_dim", 256),
                cfg.get("dropout", 0.1), cfg.get("activation", "gelu"))
    train_cfg = cfg["training"]
    logger.info("Training: bs=%d | lr=%.1e | wd=%.4f | epochs=%d | warmup=%.2f | clip=%.1f | sched=%s | amp_dtype=%s",
                train_cfg["batch_size"], train_cfg["learning_rate"],
                train_cfg.get("weight_decay", 0.01), train_cfg.get("num_epochs", 5),
                train_cfg.get("warmup_ratio", 0.05), train_cfg.get("grad_clip", 1.0),
                train_cfg.get("scheduler", "cosine"), train_cfg.get("amp_dtype", "none"))
    logger.info("Data: max_history=%d | max_seq_len=%d | seed=%d",
                args.max_history_items if args.max_history_items is not None else cfg.get("max_history_items", 20),
                cfg.get("max_seq_length", 512),
                cfg.get("seed", 42))
    logger.info("=" * 60)

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

    # ── Load backbone (SASRec or BERT4Rec, frozen) ───────────────────────
    if args.backbone == "sasrec":
        if args.sasrec_max_seq_len is not None:
            sasrec_cfg["max_seq_len"] = args.sasrec_max_seq_len
            logger.info("CLI override: sasrec_max_seq_len = %d", args.sasrec_max_seq_len)
        seq_model = SASRec(
            num_items=num_items,
            emb_dim=sasrec_cfg.get("embedding_dim", 64),
            max_len=sasrec_cfg.get("max_seq_len", 50),
            n_heads=sasrec_cfg.get("num_heads", 2),
            n_blocks=sasrec_cfg.get("num_blocks", 2),
            dropout=sasrec_cfg.get("dropout_rate", 0.2),
        )
        load_checkpoint(args.sasrec_ckpt, model=seq_model, device=args.device)
        seq_model.to(args.device).eval()
        logger.info("Loaded SASRec from %s (frozen)", args.sasrec_ckpt)
    else:  # bert4rec
        seq_model = BERT4Rec(
            num_items=num_items,
            emb_dim=bert4rec_cfg.get("embedding_dim", 64),
            max_len=bert4rec_cfg.get("max_seq_len", 50),
            n_heads=bert4rec_cfg.get("num_heads", 2),
            n_blocks=bert4rec_cfg.get("num_blocks", 2),
            dropout=bert4rec_cfg.get("dropout_rate", 0.2),
        )
        load_checkpoint(args.bert4rec_ckpt, model=seq_model, device=args.device)
        seq_model.to(args.device).eval()
        logger.info("Loaded BERT4Rec from %s (frozen)", args.bert4rec_ckpt)
    # Alias kept for InjectorTrainer which accepts it as `sasrec` param
    # (model is frozen and not called during training — embeddings are pre-computed)
    sasrec = seq_model

    # ── Load LLaMA (frozen) ──────────────────────────────────────────────
    llm_cfg = cfg["llm"]
    if args.llm_model:
        llm_cfg["model_name"] = args.llm_model
        logger.info("CLI override: llm_model = %s", args.llm_model)
    if args.num_soft_tokens is not None:
        cfg["num_soft_tokens"] = args.num_soft_tokens
        logger.info("CLI override: num_soft_tokens = %d", args.num_soft_tokens)
    if args.hidden_dim is not None:
        cfg["hidden_dim"] = args.hidden_dim
        logger.info("CLI override: hidden_dim = %d", args.hidden_dim)
    if args.max_history_items is not None:
        cfg["max_history_items"] = args.max_history_items
        logger.info("CLI override: max_history_items = %d", args.max_history_items)
    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs
        logger.info("CLI override: num_epochs = %d", args.num_epochs)
    llm, tokenizer = load_llama(
        model_name=llm_cfg["model_name"],
        dtype=llm_cfg.get("dtype", "bf16"),
        device_map=llm_cfg.get("device_map", "auto"),
        gradient_checkpointing=llm_cfg.get("gradient_checkpointing", False),
        train_mode=False,
    )
    logger.info("Loaded LLaMA (frozen)")

    # ── Build injector ───────────────────────────────────────────────────
    # Infer LLM hidden dim from the model; infer user_dim from backbone embedding dim
    llm_dim = llm.config.hidden_size
    user_dim = seq_model.emb_dim
    injector = SoftPromptInjector(
        user_dim=user_dim,
        llm_dim=llm_dim,
        n_soft_tokens=cfg.get("num_soft_tokens", 8),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.1),
        activation=cfg.get("activation", "gelu"),
    )
    inj_params = count_params(injector)
    logger.info("SoftPromptInjector config: user_dim=%d, llm_dim=%d, "
                 "n_soft_tokens=%d, hidden_dim=%d, dropout=%.2f, activation=%s",
                 user_dim, llm_dim,
                 cfg.get("num_soft_tokens", 8), cfg.get("hidden_dim", 256),
                 cfg.get("dropout", 0.1), cfg.get("activation", "gelu"))
    logger.info("SoftPromptInjector params: total=%d, trainable=%d",
                 inj_params["total"], inj_params["trainable"])

    # ── Load pre-computed embeddings ─────────────────────────────────────
    # For training: flat pair_idx → embedding table
    pair_emb_path = args.pair_emb_path or str(Path(args.data_dir) / "train_pair_embeddings.pt")
    pair_emb_table = torch.load(pair_emb_path, map_location="cpu", weights_only=True)
    logger.info("Loaded pair embeddings: %s", tuple(pair_emb_table.shape))

    # For eval (valid split): train-only user embeddings
    user_emb_path = args.user_emb_path or str(Path(args.data_dir) / "user_embeddings.npz")
    user_embs = load_user_embeddings(user_emb_path)
    logger.info("Loaded user embeddings (train-only): %d users", len(user_embs))

    # For eval (test split): train+valid user embeddings so the user rep
    # reflects the state right before the test item (not one step behind)
    user_emb_test_path = (
        args.user_emb_test_path
        or str(Path(args.data_dir) / "user_embeddings_with_valid.npz")
    )
    user_embs_with_valid = load_user_embeddings(user_emb_test_path)
    logger.info("Loaded user embeddings+valid (train+valid): %d users", len(user_embs_with_valid))

    # ── Optionally replace embeddings with a shared zero vector (ablation) ──
    # All users get the same all-zeros vector at both train and eval time.
    # This tests whether user-specific embedding content matters vs. the MLP
    # simply learning a fixed soft-prompt bias that ignores the input.
    if args.random_embeddings:
        emb_dim = pair_emb_table.shape[1]
        fixed = torch.zeros(emb_dim)
        pair_emb_table = fixed.unsqueeze(0).expand(pair_emb_table.shape[0], -1).clone()
        zero_np = fixed.numpy()
        user_embs = {uid: zero_np for uid in user_embs}
        user_embs_with_valid = {uid: zero_np for uid in user_embs_with_valid}
        logger.info("ABLATION: replaced all embeddings with shared zero vector "
                    "(dim=%d) — injector sees no user-specific signal", emb_dim)

    # Text history sequences for test eval — append valid item to each user's train seq
    train_plus_valid = {
        uid: seq + [splits["valid"][uid]]
        for uid, seq in splits["train"].items()
        if uid in splits["valid"]
    }
    # DEBUG 1: verify pair_emb_table covers all pair indices in the training dataset
    # (catches mismatch between JSONL and embedding table if regenerated separately)
    _pairs_path = args.pairs_path or str(Path(args.data_dir) / "llm_train_pairs.jsonl")
    _raw_pairs_check = load_pairs_jsonl(_pairs_path)
    _max_pair_idx = max(p["pair_idx"] for p in _raw_pairs_check)
    logger.info("[DEBUG] pair_emb_table size=%d | max pair_idx in JSONL=%d | %s",
                pair_emb_table.shape[0], _max_pair_idx,
                "OK" if pair_emb_table.shape[0] > _max_pair_idx else "MISMATCH — table too small!")

    # ── Build data loaders ───────────────────────────────────────────────
    train_loader, val_loader = _build_pair_loaders(
        args.data_dir, tokenizer, cfg, splits, negatives, item_meta,
        val_subset_users=VAL_SUBSET_USERS,
        val_num_negatives=VAL_NEG_QUICK,
        pairs_path=args.pairs_path,
    )
    logger.info("Train: %d samples, Val: %d samples",
                len(train_loader.dataset), len(val_loader.dataset))
    # DEBUG 2: verify history truncation — raw JSONL lengths vs after cap
    _train_ds = train_loader.dataset
    _n = min(5, len(_train_ds.samples))
    _raw_lens = [len(_train_ds.samples[i]["history_text"]) for i in range(_n)]
    _cap_lens  = [min(len(_train_ds.samples[i]["history_text"]), cfg.get("max_history_items", 20)) for i in range(_n)]
    _max_raw   = max(len(p["history_text"]) for p in _train_ds.samples)
    logger.info("[DEBUG] Train history lengths — raw (JSONL): %s | after cap(%d): %s | max_raw_in_dataset=%d",
                _raw_lens, cfg.get("max_history_items", 20), _cap_lens, _max_raw)
    # DEBUG 3: sample training prompt to visually confirm structure and history cap
    _sample_train = _train_ds[0]
    logger.info("[DEBUG] Sample train prompt (pair_idx=%d, label=%d):\n%s",
                _sample_train["pair_idx"], _sample_train["label"],
                _sample_train["prompt"][:500])
    # DEBUG 4: sample val prompt — compare structure with train prompt
    _val_ds = val_loader.dataset
    _sample_val = _val_ds[0]
    logger.info("[DEBUG] Sample val prompt (user_idx=%d, label=%d):\n%s",
                _sample_val["user_idx"], _sample_val["label"],
                _sample_val["prompt"][:500])

    # ── Average token length after injector (L_effective = L_text + m) ───
    m = cfg.get("num_soft_tokens", 8)
    _token_lengths = []
    for _batch in train_loader:
        if "attention_mask" in _batch:
            _token_lengths.append(_batch["attention_mask"].sum(dim=1).float())
        if len(_token_lengths) >= 10:
            break
    if _token_lengths:
        avg_l_text = torch.cat(_token_lengths).mean().item()
        avg_l_effective = avg_l_text + m
        logger.info(
            "Avg token length (train sample): L_text=%.1f | m=%d soft tokens | "
            "L_effective = L_text + m = %.1f",
            avg_l_text, m, avg_l_effective,
        )

    # ── Estimate training compute ─────────────────────────────────────────
    total_llm_params = sum(p.numel() for p in llm.parameters())
    steps_per_epoch = len(train_loader)
    num_epochs = cfg["training"].get("num_epochs", 5)
    total_steps = steps_per_epoch * num_epochs
    max_seq = cfg.get("max_seq_length", 512)
    bs = cfg["training"]["batch_size"]
    tokens_per_step = bs * max_seq
    # Injector: forward through LLM (2N) + backward through frozen LLM (2N) per token
    flops_per_step = 4 * total_llm_params * tokens_per_step
    logger.info("Compute estimate (injector): "
                 "%d steps/epoch × %d epochs = %d total steps",
                 steps_per_epoch, num_epochs, total_steps)
    logger.info("  FLOPs/step ≈ 4·N·B·S = 4 × %d × %d × %d = %.2e",
                 total_llm_params, bs, max_seq, flops_per_step)
    logger.info("  Total FLOPs ≈ %.2e", flops_per_step * total_steps)

    # ── Train ────────────────────────────────────────────────────────────
    trainer = InjectorTrainer(
        sasrec=sasrec,
        llm=llm,
        injector=injector,
        tokenizer=tokenizer,
        cfg=cfg["training"],
        device=args.device,
    )

    # pair_emb_table is indexed by pair_idx (for training)
    # user_embs is indexed by user_idx (for eval)
    best_injector = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        train_emb_table=pair_emb_table,
        eval_emb_table=user_embs,
        patience=cfg["training"].get("patience", 5),
        eval_ranking_every=1,
    )

    # ── Save checkpoint ──────────────────────────────────────────────────
    if args.output_ckpt:
        ckpt_path = Path(args.output_ckpt)
    else:
        ckpt_dir = Path(cfg.get("checkpoint_dir", "outputs/checkpoints"))
        ckpt_path = ckpt_dir / f"injector_{args.dataset}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_injector.state_dict(), str(ckpt_path))
    logger.info("Injector checkpoint saved → %s", ckpt_path)

    # ── Evaluate on VAL and TEST (full negatives, all users) ─────────────
    logger.info("Final evaluation: all users, full negatives (up to %d/user)",
                 max(len(v) for v in negatives.values()))

    val_metrics = evaluate_ranker(
        model=llm,
        tokenizer=tokenizer,
        splits=splits,
        negatives=negatives,
        item_meta=item_meta,
        user_emb_table=user_embs,
        injector=best_injector,
        split="valid",
        prompt_id=eval_cfg.get("prompt_id", "2-11"),
        max_history=cfg.get("max_history_items", 20),
        batch_size=eval_cfg["scoring"]["batch_size"],
        max_seq_length=eval_cfg["scoring"]["max_seq_length"],
        ks=tuple(eval_cfg["ks"]),
        device=args.device,
    )
    test_metrics = evaluate_ranker(
        model=llm,
        tokenizer=tokenizer,
        splits=splits,
        negatives=negatives,
        item_meta=item_meta,
        user_emb_table=user_embs_with_valid,
        injector=best_injector,
        split="test",
        prompt_id=eval_cfg.get("prompt_id", "2-11"),
        max_history=cfg.get("max_history_items", 20),
        batch_size=eval_cfg["scoring"]["batch_size"],
        max_seq_length=eval_cfg["scoring"]["max_seq_length"],
        ks=tuple(eval_cfg["ks"]),
        device=args.device,
        user_sequences=train_plus_valid,
    )
    logger.info("VAL  %s", val_metrics)
    logger.info("TEST %s", test_metrics)

    # ── Save metrics ─────────────────────────────────────────────────────
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        # Derive from LLM model name: "meta-llama/Llama-3.2-1B-Instruct" → "Llama-3.2-1B-Instruct"
        llm_short_name = llm_cfg["model_name"].split("/")[-1]
        results_dir = Path("outputs/results") / llm_short_name
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "method": "injector",
        "dataset": args.dataset,
        "seed": cfg.get("seed", 42),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "checkpoint": str(ckpt_path),
    }
    metrics_name = ckpt_path.stem  # e.g. "injector_beauty_v2"
    save_metrics(metrics, str(results_dir / f"{metrics_name}.json"))

    # ── Run summary (with compute stats) ─────────────────────────────────
    inj_params = count_params(best_injector)
    num_train = len(train_loader.dataset)
    record = build_compute_record(
        method="injector",
        model=best_injector,
        cfg=cfg["training"],
        num_train_samples=num_train,
        metrics={
            "NDCG@10": test_metrics.get("NDCG@10", 0.0),
            "HR@10": test_metrics.get("HR@10", 0.0),
        },
    )
    record["dataset"] = args.dataset
    record["seed"] = cfg.get("seed", 42)
    append_run_summary(str(results_dir / "all_runs.jsonl"), record)
    logger.info("Done.")


if __name__ == "__main__":
    main()
