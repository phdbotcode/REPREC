#!/usr/bin/env python
"""03 — Build LLM YES/NO training pairs with prefix-aware backbone embeddings.

For each user, generate training examples from the **train** split only:
    step t:  history = [i1, …, i_{t-1}],  positive = i_t

Each example also gets a backbone (SASRec or BERT4Rec) user embedding
computed from the *same* prefix (no leakage).  Embeddings are saved as a
flat tensor indexed by ``pair_idx`` for easy lookup during LLM training.

Backbone choices (--backbone):
    sasrec   – causal transformer; single forward pass gives all prefix states.
    bert4rec – bidirectional transformer; requires one forward pass per prefix
               length (O(max_len) passes, each batched across users).

Outputs (SASRec, default naming):
    outputs/data/{dataset}/llm_train_pairs_{max_len}{emb_dim}.jsonl
    outputs/data/{dataset}/llm_val_pairs_{emb_dim}.jsonl
    outputs/data/{dataset}/train_pair_embeddings_{max_len}{emb_dim}.pt
    outputs/data/{dataset}/user_embeddings_{max_len}{emb_dim}.npz
    outputs/data/{dataset}/user_embeddings_with_valid_{max_len}{emb_dim}.npz

For BERT4Rec the embedding filenames get a ``_bert4rec`` suffix before ``.pt``/``.npz``.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm4rec.bert4rec.infer import (
    extract_user_embeddings as bert4rec_extract_user_embeddings,
    save_user_embeddings as bert4rec_save_user_embeddings,
)
from llm4rec.bert4rec.model import BERT4Rec
from llm4rec.data.negatives import load_negatives
from llm4rec.sasrec.infer import extract_user_embeddings, save_user_embeddings
from llm4rec.sasrec.model import SASRec
from llm4rec.utils.io import load_checkpoint, load_config, load_json
from llm4rec.utils.logging import get_logger
from llm4rec.utils.seed import set_seed

logger = get_logger(__name__)

DEBUG = True


# ─────────────────────────────────────────────────────────────────────────────
# Extract hidden states at ALL positions (for prefix-aware embeddings)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_all_hidden_states(
    model: SASRec,
    user_sequences: Dict[int, List[int]],
    max_len: int,
    batch_size: int,
    device: str,
) -> Dict[int, np.ndarray]:
    """Return ``{uid: (seq_len, D)}`` hidden states at every position."""
    model.eval()
    dev = torch.device(device)
    uids = sorted(user_sequences.keys())
    result: Dict[int, np.ndarray] = {}

    for start in range(0, len(uids), batch_size):
        batch_uids = uids[start:start + batch_size]
        seqs, lens = [], []
        for uid in batch_uids:
            s = user_sequences[uid][-max_len:]
            lens.append(len(s))
            seqs.append(s + [0] * (max_len - len(s)))

        seq_t = torch.tensor(seqs, dtype=torch.long, device=dev)
        hidden = model(seq_t).cpu().numpy()  # (B, L, D)

        for i, uid in enumerate(batch_uids):
            result[uid] = hidden[i, :lens[i]]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BERT4Rec prefix-aware hidden states
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_all_hidden_states_bert4rec(
    model: BERT4Rec,
    user_sequences: Dict[int, List[int]],
    max_len: int,
    batch_size: int,
    device: str,
) -> Dict[int, np.ndarray]:
    """Prefix-aware embeddings for BERT4Rec via mask-last-token inference.

    For each user ``u`` and prefix length ``l`` (1 … min(|seq|, max_len)):
        feed  [i₁, …, i_l, [MASK], PAD, …]  (capped to max_len)
        take the hidden state at the [MASK] position.

    Returns ``{uid: np.ndarray of shape (n_prefixes, D)}`` where
    ``n_prefixes = min(|seq|, max_len)``, matching the interface of
    :func:`extract_all_hidden_states` for SASRec.

    Since BERT4Rec is bidirectional each prefix length needs a separate
    forward pass; passes are batched across users for efficiency.
    Total forward passes = max(min(|seq|, max_len)) over all users.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    emb_dim = model.emb_dim

    uids = sorted(user_sequences.keys())

    # Allocate result arrays (n_prefixes = min(len(seq), max_len))
    result: Dict[int, np.ndarray] = {
        uid: np.zeros((min(len(user_sequences[uid]), max_len), emb_dim), dtype=np.float32)
        for uid in uids
    }

    # Maximum prefix length we actually need to compute
    max_prefix = max(min(len(user_sequences[uid]), max_len) for uid in uids)

    for l in range(1, max_prefix + 1):
        # Only users whose capped sequence length >= l
        eligible = [uid for uid in uids if min(len(user_sequences[uid]), max_len) >= l]
        if not eligible:
            continue

        for start in range(0, len(eligible), batch_size):
            batch_uids = eligible[start : start + batch_size]
            seqs: List[List[int]] = []
            mask_positions: List[int] = []

            for uid in batch_uids:
                # Prefix of length l, truncated to max_len-1 to leave slot for [MASK]
                prefix = user_sequences[uid][:l][-(max_len - 1):]
                mask_pos = len(prefix)
                seq_padded = (
                    prefix
                    + [model.mask_token_id]
                    + [0] * (max_len - len(prefix) - 1)
                )
                seqs.append(seq_padded)
                mask_positions.append(mask_pos)

            seq_t = torch.tensor(seqs, dtype=torch.long, device=dev)
            hidden = model(seq_t)  # (B, L, D)

            for i, uid in enumerate(batch_uids):
                result[uid][l - 1] = hidden[i, mask_positions[i]].cpu().numpy()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pair generation
# ─────────────────────────────────────────────────────────────────────────────

def _item_name(item_idx: int, meta: Optional[Dict[int, dict]]) -> str:
    if meta and item_idx in meta:
        t = meta[item_idx].get("title", "").strip()
        if t:
            return t
    return f"item_{item_idx}"


def generate_training_pairs(
    train_seqs: Dict[int, List[int]],
    num_items: int,
    item_meta: Optional[Dict[int, dict]],
    neg_ratio: int = 4,
    max_pairs_per_user: int = 1,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate (history, candidate, label) training pairs.

    With ``max_pairs_per_user=1`` only the last step is used (simple mode).
    With ``max_pairs_per_user=0`` (or very large) ALL steps are used.
    """
    rng = random.Random(seed)
    pairs: List[Dict[str, Any]] = []
    pair_idx = 0

    for uid, seq in sorted(train_seqs.items()):
        if len(seq) < 2:
            continue
        # Decide which steps to use (1-indexed step t means positive=seq[t-1])
        all_steps = list(range(2, len(seq) + 1))
        if max_pairs_per_user and 0 < max_pairs_per_user < len(all_steps):
            all_steps = all_steps[-max_pairs_per_user:]  # most recent

        for t in all_steps:
            history = seq[:t - 1]
            positive = seq[t - 1]
            hist_text = [_item_name(i, item_meta) for i in history]
            cand_text = _item_name(positive, item_meta)

            # Positive
            pairs.append({
                "pair_idx": pair_idx,
                "user_idx": uid,
                "prefix_len": len(history),
                "history_items": history,
                "candidate_item": positive,
                "label": 1,
                "history_text": hist_text,
                "candidate_text": cand_text,
            })
            pair_idx += 1

            # Negatives — only exclude the positive item at this step
            for _ in range(neg_ratio):
                neg = rng.randint(1, num_items)
                while neg == positive:
                    neg = rng.randint(1, num_items)
                pairs.append({
                    "pair_idx": pair_idx,
                    "user_idx": uid,
                    "prefix_len": len(history),
                    "history_items": history,
                    "candidate_item": neg,
                    "label": 0,
                    "history_text": hist_text,
                    "candidate_text": _item_name(neg, item_meta),
                })
                pair_idx += 1

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Build flat embedding table from per-pair prefix
# ─────────────────────────────────────────────────────────────────────────────

def build_pair_embedding_table(
    pairs: List[Dict[str, Any]],
    all_hidden: Dict[int, np.ndarray],
) -> torch.Tensor:
    """Create a ``(num_pairs, D)`` tensor indexed by ``pair_idx``."""
    D = next(iter(all_hidden.values())).shape[-1]
    table = torch.zeros(len(pairs), D)
    for p in pairs:
        uid = p["user_idx"]
        plen = p["prefix_len"]
        # hidden[plen-1] = state after processing items [i1…i_{plen}]
        # Cap to max_len: if prefix exceeds max_len, use the last available state
        idx = min(plen, all_hidden[uid].shape[0]) - 1
        table[p["pair_idx"]] = torch.from_numpy(all_hidden[uid][idx])
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_pairs_jsonl(pairs: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, default=str) + "\n")
    logger.info("Saved %d pairs → %s", len(pairs), path)


@torch.no_grad()
def debug_prefix_invariance(model, seq_full, max_len, device):
    import numpy as np

    dev = torch.device(device)
    model.eval()

    s = seq_full[-max_len:]
    L = len(s)
    if L < 3:
        return

    # full sequence run
    full_pad = s + [0]*(max_len - L)
    x_full = torch.tensor([full_pad], dtype=torch.long, device=dev)
    h_full = model(x_full)[0, :L].detach().cpu().numpy()

    print("\n[DEBUG] Prefix invariance check")
    for t in [1,2,3,5,10,20,30,40]:
        if t >= L:
            continue

        pref = s[:t]
        pref_pad = pref + [0]*(max_len - len(pref))
        x_pref = torch.tensor([pref_pad], dtype=torch.long, device=dev)
        h_pref = model(x_pref)[0, t-1].detach().cpu().numpy()

        diff = np.mean(np.abs(h_full[t-1] - h_pref))
        print(f"t={t:3d}  mean|full-prefix|={diff:.8f}")


def debug_embedding_series(model, uid, seq_full, max_len, device):
    import numpy as np
    dev = torch.device(device)
    model.eval()

    s = seq_full[-max_len:]
    L = len(s)

    pad = s + [0]*(max_len - L)
    x = torch.tensor([pad], dtype=torch.long, device=dev)
    h = model(x)[0, :L].detach().cpu().numpy()

    print(f"\n[DEBUG] Embedding series uid={uid} len={L}")

    def cos(a,b):
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

    for t in range(min(12, L)):
        norm = float(np.linalg.norm(h[t]))
        cprev = "-" if t==0 else f"{cos(h[t],h[t-1]):.4f}"
        print(f"t={t+1:3d} item={s[t]:6d} norm={norm:.4f} cos(prev)={cprev}")

    print("---- LAST STEPS ----")
    for t in range(max(0,L-12), L):
        norm = float(np.linalg.norm(h[t]))
        cprev = "-" if t==0 else f"{cos(h[t],h[t-1]):.4f}"
        print(f"t={t+1:3d} item={s[t]:6d} norm={norm:.4f} cos(prev)={cprev}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM training pairs")
    parser.add_argument("--config", default="../configs/dataset_amazon.yaml")
    parser.add_argument("--sasrec_config", default="../configs/sasrec.yaml")
    parser.add_argument("--bert4rec_config", default="../configs/bert4rec.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", default="sasrec",
                        choices=["sasrec", "bert4rec"],
                        help="Sequential encoder backbone (default: sasrec)")
    parser.add_argument("--sasrec_ckpt", default=None,
                        help="SASRec checkpoint path (required when --backbone sasrec)")
    parser.add_argument("--bert4rec_ckpt", default=None,
                        help="BERT4Rec checkpoint path (required when --backbone bert4rec)")
    parser.add_argument("--neg_ratio", type=int, default=4)
    parser.add_argument("--max_pairs_per_user", type=int, default=0,
                        help="0 = all steps; 1 = last step only (default)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validate backbone / checkpoint combination
    if args.backbone == "sasrec" and not args.sasrec_ckpt:
        parser.error("--sasrec_ckpt is required when --backbone sasrec")
    if args.backbone == "bert4rec" and not args.bert4rec_ckpt:
        parser.error("--bert4rec_ckpt is required when --backbone bert4rec")

    set_seed(args.seed)
    sasrec_cfg = load_config(args.sasrec_config)["sasrec"]
    bert4rec_cfg = load_config(args.bert4rec_config)["bert4rec"]
    max_len = sasrec_cfg.get("max_seq_len", 50)  # may be overridden by ckpt below

    # ── Load data ────────────────────────────────────────────────────────
    mappings = load_json(Path(args.data_dir) / "mappings.json")
    splits_raw = load_json(Path(args.data_dir) / "splits.json")
    splits = {k: {int(uid): v for uid, v in d.items()} for k, d in splits_raw.items()}

    if DEBUG:
        # pick 2 longest sequences
        uids_sorted = sorted(splits["train"].keys(), key=lambda u: len(splits["train"][u]), reverse=True)
        debug_u1 = uids_sorted[0]
        debug_u2 = uids_sorted[1]

        print("\n================ DEBUG USER SELECTION ================")
        print(f"user1: {debug_u1} len={len(splits['train'][debug_u1])}")
        print(f"user2: {debug_u2} len={len(splits['train'][debug_u2])}")
        print("u1 train seq sample:", splits["train"][debug_u1][:10], "...", splits["train"][debug_u1][-10:])
        print("u2 train seq sample:", splits["train"][debug_u2][:10], "...", splits["train"][debug_u2][-10:])
        if debug_u1 in splits["valid"]:
            print("u1 valid item:", splits["valid"][debug_u1])
        if debug_u2 in splits["valid"]:
            print("u2 valid item:", splits["valid"][debug_u2])
        print("======================================================\n")

    num_items = mappings["num_items"]

    # ── Summary statistics ────────────────────────────────────────────
    total_train_users = len(splits["train"])
    total_val_users = len(splits["valid"])
    total_test_users = len(splits.get("test", {}))

    # Count total train sub-sequences (prefix→next_item pairs) per user
    # For a train seq of length L, there are (L-1) possible sub-sequences
    # (but max_pairs_per_user may limit this)
    mppu_eff = args.max_pairs_per_user if args.max_pairs_per_user > 0 else None
    total_train_subsequences = 0
    for uid, seq in splits["train"].items():
        if len(seq) < 2:
            continue
        n_steps = len(seq) - 1  # all possible sub-sequences
        if mppu_eff and mppu_eff < n_steps:
            n_steps = mppu_eff
        total_train_subsequences += n_steps

    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total users (train):  {total_train_users}")
    print(f"Total users (val):    {total_val_users}")
    print(f"Total users (test):   {total_test_users}")
    print(f"Number of items:      {num_items}")
    print("-" * 60)
    print(f"max_pairs_per_user:   {args.max_pairs_per_user} "
          f"({'ALL sub-sequences' if args.max_pairs_per_user == 0 else 'last ' + str(args.max_pairs_per_user) + ' sub-sequence(s) per user'})")
    print(f"Total train sub-sequences (positive pairs across all users): "
          f"{total_train_subsequences}")
    print(f"Total train pairs incl. negatives (neg_ratio={args.neg_ratio}): "
          f"{total_train_subsequences * (1 + args.neg_ratio)}")
    print(f"Total val sequences:  {total_val_users}  (1 per user)")
    print(f"Total test sequences: {total_test_users}  (1 per user)")
    print("=" * 60)

    item_meta = None
    meta_path = Path(args.data_dir) / "item_meta.json"
    if meta_path.exists():
        raw = load_json(str(meta_path))
        item_meta = {int(k): v for k, v in raw.items()}

    # ── Load backbone (SASRec or BERT4Rec) ──────────────────────────────
    # Read max_seq_len from the checkpoint's saved config so model architecture
    # matches exactly what was trained (avoids pos_emb size mismatches).
    backbone_ckpt = args.sasrec_ckpt if args.backbone == "sasrec" else args.bert4rec_ckpt
    ckpt_meta = load_checkpoint(backbone_ckpt, device=args.device)
    ckpt_cfg = ckpt_meta.get("config", {})

    if args.backbone == "sasrec":
        max_len = ckpt_cfg.get("max_seq_len", max_len)
        emb_dim = ckpt_cfg.get("embedding_dim", sasrec_cfg.get("embedding_dim", 64))
        logger.info("Using max_seq_len=%d from SASRec checkpoint config", max_len)
        model = SASRec(
            num_items=num_items,
            emb_dim=emb_dim,
            max_len=max_len,
            n_heads=ckpt_cfg.get("num_heads", sasrec_cfg.get("num_heads", 2)),
            n_blocks=ckpt_cfg.get("num_blocks", sasrec_cfg.get("num_blocks", 2)),
            dropout=ckpt_cfg.get("dropout_rate", sasrec_cfg.get("dropout_rate", 0.2)),
        )
        load_checkpoint(args.sasrec_ckpt, model=model, device=args.device)
        model.to(args.device)
        logger.info("Loaded SASRec from %s", args.sasrec_ckpt)
    else:  # bert4rec
        max_len = ckpt_cfg.get("max_seq_len", bert4rec_cfg.get("max_seq_len", 50))
        emb_dim = ckpt_cfg.get("embedding_dim", bert4rec_cfg.get("embedding_dim", 64))
        logger.info("Using max_seq_len=%d from BERT4Rec checkpoint config", max_len)
        model = BERT4Rec(
            num_items=num_items,
            emb_dim=emb_dim,
            max_len=max_len,
            n_heads=ckpt_cfg.get("num_heads", bert4rec_cfg.get("num_heads", 2)),
            n_blocks=ckpt_cfg.get("num_blocks", bert4rec_cfg.get("num_blocks", 2)),
            dropout=ckpt_cfg.get("dropout_rate", bert4rec_cfg.get("dropout_rate", 0.2)),
        )
        load_checkpoint(args.bert4rec_ckpt, model=model, device=args.device)
        model.to(args.device)
        logger.info("Loaded BERT4Rec from %s", args.bert4rec_ckpt)

    if DEBUG and args.backbone == "sasrec":
        print("\n=========== RUNNING LEAKAGE TESTS ===========")
        debug_prefix_invariance(model, splits["train"][debug_u1], max_len, args.device)
        debug_prefix_invariance(model, splits["train"][debug_u2], max_len, args.device)
        print("=============================================\n")

    # ── 1. Generate training pairs ───────────────────────────────────────
    mppu = args.max_pairs_per_user if args.max_pairs_per_user > 0 else None
    pairs = generate_training_pairs(
        splits["train"], num_items, item_meta,
        neg_ratio=args.neg_ratio,
        max_pairs_per_user=mppu,
        seed=args.seed,
    )
    num_pos_train = sum(1 for p in pairs if p["label"] == 1)
    num_neg_train = sum(1 for p in pairs if p["label"] == 0)
    logger.info("Generated %d training pairs (neg_ratio=%d, max_pairs=%s)",
                len(pairs), args.neg_ratio, mppu)
    print("-" * 60)
    print("TRAIN PAIR GENERATION (actual)")
    print(f"  Positive pairs: {num_pos_train}")
    print(f"  Negative pairs: {num_neg_train}")
    print(f"  Total pairs:    {len(pairs)}")
    print("-" * 60)

    save_pairs_jsonl(pairs, str(Path(args.data_dir) / f"llm_train_pairs_{max_len}{emb_dim}.jsonl"))

    # ── 2. Prefix-aware embeddings for training ──────────────────────────
    bb_suffix = "" if args.backbone == "sasrec" else f"_{args.backbone}"

    if args.backbone == "sasrec":
        logger.info("Computing position-aware SASRec hidden states …")
        all_hidden = extract_all_hidden_states(
            model, splits["train"], max_len, batch_size=256, device=args.device,
        )
        if DEBUG:
            debug_embedding_series(model, debug_u1, splits["train"][debug_u1], max_len, args.device)
            debug_embedding_series(model, debug_u2, splits["train"][debug_u2], max_len, args.device)
    else:
        logger.info(
            "Computing prefix-aware BERT4Rec hidden states "
            "(one forward pass per prefix length) …"
        )
        all_hidden = extract_all_hidden_states_bert4rec(
            model, splits["train"], max_len, batch_size=256, device=args.device,
        )

    pair_emb = build_pair_embedding_table(pairs, all_hidden)
    emb_path = str(Path(args.data_dir) / f"train_pair_embeddings_{max_len}{emb_dim}{bb_suffix}.pt")
    torch.save(pair_emb, emb_path)
    logger.info("Pair embeddings (%s) → %s", tuple(pair_emb.shape), emb_path)

    # ── 3. Full-sequence user embeddings for eval ────────────────────────
    # 3a. Train-only embeddings (used for valid-split eval)
    if args.backbone == "sasrec":
        user_embs = extract_user_embeddings(
            model, splits["train"], max_len=max_len,
            batch_size=256, device=args.device,
        )
    else:
        user_embs = bert4rec_extract_user_embeddings(
            model, splits["train"], max_len=max_len,
            batch_size=256, device=args.device,
        )
    user_emb_path = str(Path(args.data_dir) / f"user_embeddings_{max_len}{emb_dim}{bb_suffix}.npz")
    save_user_embeddings(user_embs, user_emb_path)
    logger.info("User embeddings (%d users) → %s", len(user_embs), user_emb_path)

    # 3b. Train+valid embeddings (used for test-split eval so the user rep
    #     reflects the state right before the test item, not one step behind)
    train_plus_valid = {
        uid: seq + [splits["valid"][uid]]
        for uid, seq in splits["train"].items()
        if uid in splits["valid"]
    }
    if args.backbone == "sasrec":
        user_embs_with_valid = extract_user_embeddings(
            model, train_plus_valid, max_len=max_len,
            batch_size=256, device=args.device,
        )
    else:
        user_embs_with_valid = bert4rec_extract_user_embeddings(
            model, train_plus_valid, max_len=max_len,
            batch_size=256, device=args.device,
        )
    user_emb_with_valid_path = str(Path(args.data_dir) / f"user_embeddings_with_valid_{max_len}{emb_dim}{bb_suffix}.npz")
    save_user_embeddings(user_embs_with_valid, user_emb_with_valid_path)
    logger.info("User embeddings+valid (%d users) → %s", len(user_embs_with_valid), user_emb_with_valid_path)

    # ── 4. (Optional) val pairs for loss monitoring ──────────────────────
    negatives = load_negatives(str(Path(args.data_dir) / "negatives_200.jsonl"))
    val_pairs: List[Dict[str, Any]] = []
    pidx = 0
    for uid in sorted(splits["valid"]):
        if uid not in splits["train"] or uid not in negatives:
            continue
        history = splits["train"][uid]
        target = splits["valid"][uid]
        hist_text = [_item_name(i, item_meta) for i in history[-20:]]
        val_pairs.append({
            "pair_idx": pidx, "user_idx": uid,
            "prefix_len": len(history),
            "history_items": history[-20:],
            "candidate_item": target, "label": 1,
            "history_text": hist_text,
            "candidate_text": _item_name(target, item_meta),
        })
        pidx += 1
    save_pairs_jsonl(val_pairs, str(Path(args.data_dir) / f"llm_val_pairs_{emb_dim}.jsonl"))

    num_pos_val = sum(1 for p in val_pairs if p["label"] == 1)
    print("-" * 60)
    print("VAL PAIR GENERATION (actual)")
    print(f"  Positive pairs: {num_pos_val}")
    print(f"  Total pairs:    {len(val_pairs)}")
    print("-" * 60)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total users:                    {total_train_users}")
    print(f"  Total train sub-sequences:      {num_pos_train}  (positive pairs, cumulated over all users)")
    print(f"  Total train pairs (pos+neg):    {len(pairs)}")
    print(f"  Total val sequences:            {num_pos_val}")
    print(f"  Total test sequences:           {total_test_users}  (from splits.json)")
    print(f"  Train pair embeddings shape:    {tuple(pair_emb.shape)}")
    print(f"  User embeddings (train only):   {len(user_embs)}")
    print(f"  User embeddings (train+valid):  {len(user_embs_with_valid)}")
    print("=" * 60)

    logger.info("Done — all LLM pair artefacts saved to %s", args.data_dir)


if __name__ == "__main__":
    from llm4rec.sasrec.infer import save_user_embeddings
    main()
