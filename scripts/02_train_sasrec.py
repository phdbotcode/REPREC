#!/usr/bin/env python
"""02 — Train SASRec, evaluate on val+test, save checkpoint & run summary.

Outputs:
    outputs/checkpoints/sasrec_{dataset}.pt
    outputs/results/sasrec_{dataset}.json
    outputs/results/all_runs.jsonl  (append)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm4rec.data.negatives import load_negatives
from llm4rec.evaluation.compute import build_compute_record, count_params
from llm4rec.evaluation.report import append_run_summary, save_metrics
from llm4rec.sasrec.model import SASRec
from llm4rec.sasrec.trainer import SASRecTrainer
from llm4rec.utils.io import load_config, load_json, save_checkpoint
from llm4rec.utils.logging import get_logger
from llm4rec.utils.metrics import evaluate_ranking
from llm4rec.utils.seed import set_seed

logger = get_logger(__name__)


# ── SASRec ranking evaluation (dot-product scoring) ─────────────────────────

@torch.no_grad()
def evaluate_sasrec_ranking(model, train_seqs, targets, negatives,
                            max_len, device, ks=(1, 5, 10), batch_size=256):
    """Rank 1 positive + 1000 negatives per user with SASRec dot-product."""
    model.eval()
    dev = torch.device(device)
    uids = sorted(uid for uid in targets if uid in negatives)
    all_scores = []

    for start in range(0, len(uids), batch_size):
        batch_uids = uids[start:start + batch_size]
        seqs, seq_lens = [], []
        for uid in batch_uids:
            s = train_seqs[uid][-max_len:]
            seq_lens.append(len(s))
            seqs.append(s + [0] * (max_len - len(s)))

        seq_t = torch.tensor(seqs, dtype=torch.long, device=dev)
        hidden = model(seq_t)

        for i, uid in enumerate(batch_uids):
            user_emb = hidden[i, seq_lens[i] - 1]
            cand_ids = [targets[uid]] + negatives[uid]
            cand_t = torch.tensor(cand_ids, dtype=torch.long, device=dev)
            scores = model.score_candidates(user_emb, cand_t)
            all_scores.append(scores.cpu().numpy())

    return evaluate_ranking(all_scores, ks=ks, positive_idx=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SASRec")
    parser.add_argument("--config", default="../configs/sasrec.yaml")
    parser.add_argument("--data_dir", required=True, help="e.g. outputs/data/beauty")
    parser.add_argument("--dataset", required=True, help="beauty / toys_and_games")
    parser.add_argument("--output_dir", default="outputs/checkpoints", help="directory for model checkpoints")
    parser.add_argument("--results_dir", default="outputs/results", help="directory for metrics and run summaries")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="override max_seq_len from config (e.g. 50)")
    parser.add_argument("--emb_dim", type=int, default=None,
                        help="override embedding_dim from config (e.g. 128)")
    parser.add_argument("--save_checkpoint", action=argparse.BooleanOptionalAction, default=True,
                        help="save final SASRec checkpoint (use --no_save_checkpoint to skip)")
    args = parser.parse_args()

    cfg = load_config(args.config)["sasrec"]
    set_seed(cfg.get("seed", 42))
    max_len = args.max_seq_len if args.max_seq_len is not None else cfg.get("max_seq_len", 50)
    cfg["max_seq_len"] = max_len
    emb_dim = args.emb_dim if args.emb_dim is not None else cfg.get("embedding_dim", 64)
    cfg["embedding_dim"] = emb_dim

    # ── Config summary ───────────────────────────────────────────────────
    print("=" * 60)
    print("[CONFIG] SASRec configuration")
    print(f"  max_seq_len (yaml default) : {load_config(args.config)['sasrec'].get('max_seq_len', 50)}")
    print(f"  max_seq_len (CLI override) : {args.max_seq_len}")
    print(f"  max_seq_len (effective)    : {max_len}  <-- used for training & eval")
    print(f"  embedding_dim (yaml default) : {load_config(args.config)['sasrec'].get('embedding_dim', 64)}")
    print(f"  embedding_dim (CLI override) : {args.emb_dim}")
    print(f"  embedding_dim (effective)    : {emb_dim}  <-- used for training & eval")
    print(f"  num_heads                  : {cfg.get('num_heads', 2)}")
    print(f"  num_blocks                 : {cfg.get('num_blocks', 2)}")
    print(f"  dropout_rate               : {cfg.get('dropout_rate', 0.2)}")
    print(f"  batch_size                 : {cfg.get('batch_size', 128)}")
    print(f"  num_epochs                 : {cfg.get('num_epochs', 200)}")
    print(f"  learning_rate              : {cfg.get('learning_rate', 1e-3)}")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    mappings = load_json(Path(args.data_dir) / "mappings.json")
    splits_raw = load_json(Path(args.data_dir) / "splits.json")
    splits = {k: {int(uid): v for uid, v in d.items()}
              for k, d in splits_raw.items()}
    negatives = load_negatives(str(Path(args.data_dir) / "negatives_200.jsonl"))
    num_items = mappings["num_items"]

    # ── Sample user sequence debug ───────────────────────────────────────
    sample_uid = sorted(splits["train"].keys())[0]
    full_seq = (
        splits["train"][sample_uid]
        + [splits["valid"].get(sample_uid, "?")]
        + [splits["test"].get(sample_uid, "?")]
    )
    train_seq = splits["train"][sample_uid]
    truncated = train_seq[-max_len:]
    print(f"[DATA] Sample user uid={sample_uid}")
    print(f"  full sequence length (train+valid+test) : {len(full_seq)}")
    print(f"  train split length                      : {len(train_seq)}")
    print(f"  train[-max_len:] length fed to model    : {len(truncated)}")
    print(f"  first 10 items of full sequence : {full_seq[:10]}")
    print(f"  last  10 items of full sequence : {full_seq[-10:]}")
    print(f"  first 10 items of train split   : {train_seq[:10]}")
    print(f"  last  10 items of train split   : {train_seq[-10:]}")
    print(f"  truncated input to SASRec       : {truncated[:10]} ... {truncated[-10:] if len(truncated) > 10 else '(shown above)'}")
    print("=" * 60)

    # ── Train ────────────────────────────────────────────────────────────
    trainer = SASRecTrainer(cfg, device=args.device)
    print(f"[TRAINER] SASRecTrainer.max_len = {trainer.max_len}  (must match effective max_seq_len above)")
    model = trainer.fit(splits["train"], num_items)

    # ── Save checkpoint ──────────────────────────────────────────────────
    ckpt_dir = Path(args.output_dir)
    ckpt_path = ckpt_dir / f"sasrec{max_len}{emb_dim}_{args.dataset}.pt"
    if args.save_checkpoint:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint({"config": cfg, "num_items": num_items}, ckpt_path, model=model)
        logger.info("Checkpoint saved → %s", ckpt_path)
    else:
        logger.info("Checkpoint saving skipped (--no_save_checkpoint)")

    # ── Evaluate on VAL and TEST ─────────────────────────────────────────
    # Val: input = splits["train"]  →  predict splits["valid"]
    val_metrics = evaluate_sasrec_ranking(
        model, splits["train"], splits["valid"], negatives,
        max_len, args.device,
    )

    # Test: input = splits["train"] + [splits["valid"]]  →  predict splits["test"]
    # The valid item is the most recent observed interaction and must be
    # included in the history when predicting the test item.
    test_seqs = {
        uid: splits["train"][uid] + [splits["valid"][uid]]
        for uid in splits["test"]
        if uid in splits["train"] and uid in splits["valid"]
    }
    test_metrics = evaluate_sasrec_ranking(
        model, test_seqs, splits["test"], negatives,
        max_len, args.device,
    )
    logger.info("VAL  %s", val_metrics)
    logger.info("TEST %s", test_metrics)

    # ── Save metrics ─────────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "method": "sasrec",
        "dataset": args.dataset,
        "seed": cfg.get("seed", 42),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "checkpoint": str(ckpt_path),
    }
    save_metrics(metrics, str(results_dir / f"sasrec_{args.dataset}.json"))

    # ── Run summary (with compute stats) ─────────────────────────────────
    params = count_params(model)
    summary = {
        **metrics,
        "trainable_params": params["total"],  # all SASRec params are trained
        "total_params": params["total"],
        "NDCG@10": test_metrics.get("NDCG@10", 0.0),
        "HR@10": test_metrics.get("HR@10", 0.0),
    }
    append_run_summary(str(results_dir / "all_runs.jsonl"), summary)
    logger.info("Done.")


if __name__ == "__main__":
    main()
