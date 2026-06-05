#!/usr/bin/env python
"""06 — Unified evaluation for all recommendation methods.

Supported methods (via --method):

    sasrec               – dot-product scoring (SASRec only, no LLM)
    llm_zero_shot        – frozen LLaMA with SASRec embedding converted to
                           textual context (nearest-item description)
    injector             – frozen LLaMA + SoftPromptInjector (trained)
    lora                 – LLaMA + LoRA adapter
    llm_frozen_projector – frozen LLaMA + FrozenSoftPromptProjector
                           (fixed random projection, NO training required).
                           Uses SASRec user embeddings directly as soft
                           prompt tokens via a non-learnable random matrix.

All methods are evaluated on the SAME fixed negative pool (1000 negatives)
and the SAME data splits.

Outputs:
    outputs/results/{method}_{dataset}_eval.json
    outputs/results/all_runs.jsonl  (append)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm4rec.bert4rec.model import BERT4Rec
from llm4rec.data.negatives import load_negatives
from llm4rec.evaluation.compute import build_compute_record, count_params
from llm4rec.evaluation.report import append_run_summary, save_metrics
from llm4rec.sasrec.model import SASRec
from llm4rec.utils.io import load_checkpoint, load_config, load_json
from llm4rec.utils.logging import get_logger
from llm4rec.utils.metrics import evaluate_ranking
from llm4rec.utils.seed import set_seed

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SASRec dot-product evaluation (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_sasrec(
    model: SASRec,
    train_seqs: dict,
    targets: dict,
    negatives: dict,
    max_len: int,
    device: str,
    ks: tuple = (1, 5, 10),
    batch_size: int = 256,
) -> Dict[str, float]:
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


# ─────────────────────────────────────────────────────────────────────────────
# BERT4Rec dot-product evaluation (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_bert4rec(
    model: BERT4Rec,
    train_seqs: dict,
    targets: dict,
    negatives: dict,
    max_len: int,
    device: str,
    ks: tuple = (1, 5, 10),
    batch_size: int = 256,
) -> Dict[str, float]:
    """Rank 1 positive + negatives per user with BERT4Rec mask-last-token."""
    model.eval()
    dev = torch.device(device)
    uids = sorted(uid for uid in targets if uid in negatives)
    all_scores = []

    for start in range(0, len(uids), batch_size):
        batch_uids = uids[start : start + batch_size]
        seqs: list = []
        mask_positions: list = []

        for uid in batch_uids:
            s = train_seqs[uid][-(max_len - 1):]  # leave one slot for [MASK]
            mask_pos = len(s)
            seq_padded = s + [model.mask_token_id] + [0] * (max_len - len(s) - 1)
            seqs.append(seq_padded)
            mask_positions.append(mask_pos)

        seq_t = torch.tensor(seqs, dtype=torch.long, device=dev)
        hidden = model(seq_t)  # (B, L, D)

        for i, uid in enumerate(batch_uids):
            user_emb = hidden[i, mask_positions[i]]
            cand_ids = [targets[uid]] + negatives[uid]
            cand_t = torch.tensor(cand_ids, dtype=torch.long, device=dev)
            scores = model.score_candidates(user_emb, cand_t)
            all_scores.append(scores.cpu().numpy())

    return evaluate_ranking(all_scores, ks=ks, positive_idx=0)


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot LLM evaluation (pure text prompting)
# ─────────────────────────────────────────────────────────────────────────────

def _item_name(item_idx: int, item_meta: Optional[Dict[int, dict]]) -> str:
    if item_meta and item_idx in item_meta:
        t = item_meta[item_idx].get("title", "").strip()
        if t:
            return t
    return f"item_{item_idx}"


@torch.no_grad()
def _eval_zero_shot(
    llm,
    tokenizer,
    train_seqs: dict,
    targets: dict,
    negatives: dict,
    item_meta: Optional[Dict[int, dict]],
    max_history: int = 50,
    max_seq_length: int = 512,
    batch_size: int = 32,
    ks: tuple = (1, 5, 10),
    device: str = "cuda",
    prompt_id: str = "2-11",
) -> Dict[str, float]:
    """Zero-shot LLM scoring: purchase history text → prompt → logP(Yes).

    No SASRec embeddings are used. The user's purchase history is converted
    to item names and fed directly into the prompt template.
    """
    from llm4rec.data.prompts import format_prompt
    from llm4rec.llm.scoring import get_yes_no_token_ids, score_from_logits

    llm.eval()
    dev = torch.device(device)

    tok_ids = get_yes_no_token_ids(tokenizer)
    yes_id = tok_ids["yes"][0]
    no_id = tok_ids["no"][0]

    uids = sorted(uid for uid in targets if uid in negatives)
    all_scores = []
    total_tokens = 0
    total_seqs = 0

    for uid in uids:
        # Build history text from item names only
        history = train_seqs[uid][-max_history:]
        history_names = [_item_name(i, item_meta) for i in history]

        # Score 1 positive + N negatives
        candidates = [targets[uid]] + negatives[uid]
        user_scores = []

        for batch_start in range(0, len(candidates), batch_size):
            batch_cands = candidates[batch_start:batch_start + batch_size]
            prompts = []
            for cand in batch_cands:
                cand_name = _item_name(cand, item_meta)
                prompt = format_prompt(
                    prompt_id=prompt_id,
                    user_id=uid,
                    history=history_names,
                    candidate=cand_name,
                )
                prompts.append(prompt)

            # Tokenize
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            input_ids = encoded["input_ids"].to(dev)
            attn_mask = encoded["attention_mask"].to(dev)
            total_tokens += attn_mask.sum().item()
            total_seqs += attn_mask.shape[0]

            logits = llm(input_ids=input_ids, attention_mask=attn_mask).logits
            scores = score_from_logits(
                logits, attn_mask, yes_id=yes_id, no_id=no_id,
                mode="logprob_yes",
            )
            user_scores.extend(scores.cpu().tolist())

        all_scores.append(np.array(user_scores))

    avg_tokens = total_tokens / total_seqs if total_seqs > 0 else 0.0
    print(f"[zero-shot] avg prompt token length (excl. padding) = {avg_tokens:.1f} "
          f"over {total_seqs} prompts ({len(uids)} users)")

    return evaluate_ranking(all_scores, ks=ks, positive_idx=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ranker evaluation")
    parser.add_argument("--method", required=True,
                        choices=["sasrec", "bert4rec", "llm_zero_shot", "injector",
                                 "lora", "llm_frozen_projector", "lora_frozen_projector"])
    parser.add_argument("--config", default=None,
                        help="Method config (auto-detected if not set)")
    parser.add_argument("--sasrec_config", default="../configs/sasrec.yaml")
    parser.add_argument("--bert4rec_config", default="../configs/bert4rec.yaml")
    parser.add_argument("--eval_config", default="../configs/eval.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sasrec_ckpt", default=None,
                        help="SASRec checkpoint (required for method=sasrec)")
    parser.add_argument("--bert4rec_ckpt", default=None,
                        help="BERT4Rec checkpoint (required for method=bert4rec)")
    parser.add_argument("--split", default="test", choices=["val", "valid", "test"])
    # LLM-specific
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter_dir", default=None, help="LoRA adapter dir")
    parser.add_argument("--injector_ckpt", default=None, help="Injector .pt")
    parser.add_argument("--user_emb_path", default=None,
                        help="User embeddings .npz (train-only, for valid-split eval)")
    parser.add_argument("--user_emb_test_path", default=None,
                        help="User embeddings .npz with valid item appended "
                             "(train+valid, for test-split eval). "
                             "Defaults to user_embeddings_with_valid.npz in data_dir.")
    parser.add_argument("--n_soft_tokens", type=int, default=8,
                        help="Soft prompt tokens for llm_frozen_projector (default 8)")
    parser.add_argument("--max_history", type=int, default=None,
                        help="Max history items in prompt. Overrides config value. "
                             "Defaults: injector/lora=20 (from config), "
                             "zero_shot/frozen_projector=50 (from eval config).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sasrec_cfg = load_config(args.sasrec_config)["sasrec"]
    bert4rec_cfg = load_config(args.bert4rec_config)["bert4rec"]
    eval_cfg = load_config(args.eval_config)["eval"]
    set_seed(eval_cfg.get("seed", 42))

    # Validate method / checkpoint combination
    if args.method == "sasrec" and not args.sasrec_ckpt:
        parser.error("--sasrec_ckpt is required for method=sasrec")
    if args.method == "bert4rec" and not args.bert4rec_ckpt:
        parser.error("--bert4rec_ckpt is required for method=bert4rec")

    # Auto-detect method config
    config_map = {
        "injector": "../configs/llm_injector.yaml",
        "lora": "../configs/llm_lora.yaml",
    }
    method_cfg = {}
    if args.config:
        method_cfg = load_config(args.config)
    elif args.method in config_map:
        method_cfg = load_config(config_map[args.method])

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

    max_len = sasrec_cfg.get("max_seq_len", 50)
    ks = tuple(eval_cfg["ks"])
    split_key = "valid" if args.split in ("val", "valid") else "test"

    # For test-split eval: history should include the valid item so the user
    # representation is correct right before the test item.
    is_test = split_key == "test"
    train_plus_valid = {
        uid: seq + [splits["valid"][uid]]
        for uid, seq in splits["train"].items()
        if uid in splits["valid"]
    }
    # sequences and embedding path to use depending on the split
    eval_seqs = train_plus_valid if is_test else splits["train"]

    # ── Dispatch by method ───────────────────────────────────────────────
    if args.method == "sasrec":
        sasrec = SASRec(
            num_items=num_items,
            emb_dim=sasrec_cfg.get("embedding_dim", 64),
            max_len=max_len,
            n_heads=sasrec_cfg.get("num_heads", 2),
            n_blocks=sasrec_cfg.get("num_blocks", 2),
            dropout=sasrec_cfg.get("dropout_rate", 0.2),
        )
        load_checkpoint(args.sasrec_ckpt, model=sasrec, device=args.device)
        sasrec.to(args.device).eval()
        logger.info("Loaded SASRec from %s", args.sasrec_ckpt)
        metrics = _eval_sasrec(
            sasrec, eval_seqs, splits[split_key], negatives,
            max_len, args.device, ks=ks,
        )

    elif args.method == "bert4rec":
        bert4rec_max_len = bert4rec_cfg.get("max_seq_len", 50)
        bert4rec_model = BERT4Rec(
            num_items=num_items,
            emb_dim=bert4rec_cfg.get("embedding_dim", 64),
            max_len=bert4rec_max_len,
            n_heads=bert4rec_cfg.get("num_heads", 2),
            n_blocks=bert4rec_cfg.get("num_blocks", 2),
            dropout=bert4rec_cfg.get("dropout_rate", 0.2),
        )
        load_checkpoint(args.bert4rec_ckpt, model=bert4rec_model, device=args.device)
        bert4rec_model.to(args.device).eval()
        logger.info("Loaded BERT4Rec from %s", args.bert4rec_ckpt)
        metrics = _eval_bert4rec(
            bert4rec_model, eval_seqs, splits[split_key], negatives,
            bert4rec_max_len, args.device, ks=ks,
        )

    elif args.method == "llm_zero_shot":
        from llm4rec.llm.llama_backbone import load_llama

        llm, tokenizer = load_llama(
            model_name=args.model_name, dtype="bf16",
            device_map=args.device, train_mode=False,
        )

        max_history = args.max_history if args.max_history is not None else eval_cfg.get("max_history", 50)
        print(f"[zero-shot] max_history (history length) = {max_history}")

        metrics = _eval_zero_shot(
            llm=llm,
            tokenizer=tokenizer,
            train_seqs=eval_seqs,
            targets=splits[split_key],
            negatives=negatives,
            item_meta=item_meta,
            max_history=max_history,
            max_seq_length=eval_cfg["scoring"]["max_seq_length"],
            batch_size=eval_cfg["scoring"]["batch_size"],
            ks=ks,
            device=args.device,
            prompt_id=eval_cfg.get("prompt_id", "2-11"),
        )

    elif args.method == "injector":
        from llm4rec.evaluation.rank_eval import evaluate_ranker
        from llm4rec.llm.injector import SoftPromptInjector
        from llm4rec.llm.llama_backbone import load_llama
        from llm4rec.sasrec.infer import load_user_embeddings

        inj_cfg = method_cfg.get("injector", {})
        llm_cfg = inj_cfg.get("llm", {})
        llm, tokenizer = load_llama(
            model_name=llm_cfg.get("model_name", args.model_name),
            dtype=llm_cfg.get("dtype", "bf16"),
            device_map=llm_cfg.get("device_map", "auto"),
            train_mode=False,
        )

        # Load injector — infer architecture from checkpoint state dict
        if not args.injector_ckpt:
            raise ValueError("--injector_ckpt required for method=injector")
        state = torch.load(args.injector_ckpt, map_location="cpu", weights_only=True)

        # Infer dims from weight shapes to guarantee config ↔ checkpoint match
        # projector.0.weight → (hidden_dim, user_dim)
        # projector.3.weight → (n_soft_tokens * llm_dim, hidden_dim)
        w0 = state["projector.0.weight"]
        w3 = state["projector.3.weight"]
        inferred_hidden_dim = w0.shape[0]
        inferred_user_dim = w0.shape[1]
        # Infer llm_dim from the ln layer in the checkpoint, not from the
        # currently loaded LLM — the injector may have been trained with a
        # different model size (e.g. 1B vs 3B).
        inferred_llm_dim = state["ln.weight"].shape[0]
        inferred_n_soft = w3.shape[0] // inferred_llm_dim

        logger.info(
            "Injector checkpoint dims: user_dim=%d, hidden_dim=%d, "
            "n_soft_tokens=%d, llm_dim=%d",
            inferred_user_dim, inferred_hidden_dim, inferred_n_soft, inferred_llm_dim,
        )

        injector = SoftPromptInjector(
            user_dim=inferred_user_dim,
            llm_dim=inferred_llm_dim,
            n_soft_tokens=inferred_n_soft,
            hidden_dim=inferred_hidden_dim,
            dropout=inj_cfg.get("dropout", 0.1),
            activation=inj_cfg.get("activation", "gelu"),
        )
        injector.load_state_dict(state)
        injector.to(args.device).eval()
        logger.info("Loaded injector from %s", args.injector_ckpt)

        if is_test:
            _emb_path = (args.user_emb_test_path
                         or str(Path(args.data_dir) / "user_embeddings_with_valid.npz"))
        else:
            _emb_path = args.user_emb_path or str(Path(args.data_dir) / "user_embeddings.npz")
        user_embs = load_user_embeddings(_emb_path)

        metrics = evaluate_ranker(
            model=llm,
            tokenizer=tokenizer,
            splits=splits,
            negatives=negatives,
            item_meta=item_meta,
            user_emb_table=user_embs,
            injector=injector,
            split=args.split,
            prompt_id=eval_cfg.get("prompt_id", "2-11"),
            max_history=args.max_history if args.max_history is not None else inj_cfg.get("max_history_items", 20),
            batch_size=eval_cfg["scoring"]["batch_size"],
            max_seq_length=eval_cfg["scoring"]["max_seq_length"],
            ks=ks,
            device=args.device,
            user_sequences=eval_seqs,
        )

    elif args.method == "lora":
        from llm4rec.evaluation.rank_eval import evaluate_ranker
        from llm4rec.llm.llama_backbone import load_llama
        from llm4rec.llm.lora import load_lora

        lora_cfg = method_cfg.get("lora", {})
        llm_cfg = lora_cfg.get("llm", {})
        base_llm, tokenizer = load_llama(
            model_name=llm_cfg.get("model_name", args.model_name),
            dtype=llm_cfg.get("dtype", "bf16"),
            device_map=llm_cfg.get("device_map", "auto"),
            train_mode=False,
        )

        if not args.adapter_dir:
            raise ValueError("--adapter_dir required for method=lora")
        llm = load_lora(base_llm, args.adapter_dir, is_trainable=False)
        logger.info("Loaded LoRA adapter from %s", args.adapter_dir)

        metrics = evaluate_ranker(
            model=llm,
            tokenizer=tokenizer,
            splits=splits,
            negatives=negatives,
            item_meta=item_meta,
            split=args.split,
            prompt_id=eval_cfg.get("prompt_id", "2-11"),
            max_history=args.max_history if args.max_history is not None else lora_cfg.get("max_history_items", 20),
            batch_size=eval_cfg["scoring"]["batch_size"],
            max_seq_length=eval_cfg["scoring"]["max_seq_length"],
            ks=ks,
            device=args.device,
            user_sequences=eval_seqs,
        )

    elif args.method == "llm_frozen_projector":
        # ── LLM + frozen (untrained) random projector ─────────────────
        # No injector checkpoint is needed.  A FrozenSoftPromptProjector
        # maps SASRec user embeddings → soft prefix tokens via a fixed
        # random matrix (registered as a buffer, no gradients ever).
        # This serves as an ablation: does the *trained* injector matter,
        # or does any fixed projection of the user embedding help?
        from llm4rec.evaluation.rank_eval import evaluate_ranker
        from llm4rec.llm.frozen_projector import FrozenSoftPromptProjector
        from llm4rec.llm.llama_backbone import load_llama
        from llm4rec.sasrec.infer import load_user_embeddings

        inj_cfg = method_cfg.get("injector", {})
        llm_cfg = inj_cfg.get("llm", {})
        llm, tokenizer = load_llama(
            model_name=llm_cfg.get("model_name", args.model_name),
            dtype=llm_cfg.get("dtype", "bf16"),
            device_map=llm_cfg.get("device_map", "auto"),
            train_mode=False,
        )

        if is_test:
            _emb_path = (args.user_emb_test_path
                         or str(Path(args.data_dir) / "user_embeddings_with_valid.npz"))
        else:
            _emb_path = args.user_emb_path or str(Path(args.data_dir) / "user_embeddings.npz")
        user_embs = load_user_embeddings(_emb_path)

        # Infer user_dim from saved embeddings (first value's length)
        sample_emb = next(iter(user_embs.values()))
        user_dim = int(np.array(sample_emb).shape[-1])
        llm_dim = llm.config.hidden_size
        n_soft = args.n_soft_tokens

        projector = FrozenSoftPromptProjector(
            d_in=user_dim,
            d_out=llm_dim,
            n_soft_tokens=n_soft,
        )
        projector.to(args.device)
        logger.info(
            "FrozenSoftPromptProjector: user_dim=%d → llm_dim=%d, "
            "n_soft_tokens=%d (no training)",
            user_dim, llm_dim, n_soft,
        )

        max_history = args.max_history if args.max_history is not None else eval_cfg.get("max_history", 50)
        print(f"[llm_frozen_projector] max_history (history length) = {max_history}")

        metrics = evaluate_ranker(
            model=llm,
            tokenizer=tokenizer,
            splits=splits,
            negatives=negatives,
            item_meta=item_meta,
            user_emb_table=user_embs,
            projector=projector,
            split=args.split,
            prompt_id=eval_cfg.get("prompt_id", "2-11"),
            max_history=max_history,
            batch_size=eval_cfg["scoring"]["batch_size"],
            max_seq_length=eval_cfg["scoring"]["max_seq_length"],
            ks=ks,
            device=args.device,
            user_sequences=eval_seqs,
        )

    elif args.method == "lora_frozen_projector":
        # ── LoRA adapter + saved FrozenSoftPromptProjector ────────────────
        # Used for checkpoints trained with conditioning_mode="frozen_projector"
        # (Model C).  Both the LoRA adapter AND the saved projector W matrix
        # must be loaded — the projector's random W was fixed at training time
        # so recreating it fresh would give a different matrix.
        from llm4rec.evaluation.rank_eval import evaluate_ranker
        from llm4rec.llm.frozen_projector import FrozenSoftPromptProjector
        from llm4rec.llm.llama_backbone import load_llama
        from llm4rec.llm.lora import load_lora
        from llm4rec.sasrec.infer import load_user_embeddings

        lora_cfg = method_cfg.get("lora", {})
        llm_cfg = lora_cfg.get("llm", {})
        base_llm, tokenizer = load_llama(
            model_name=llm_cfg.get("model_name", args.model_name),
            dtype=llm_cfg.get("dtype", "bf16"),
            device_map=llm_cfg.get("device_map", "auto"),
            train_mode=False,
        )

        if not args.adapter_dir:
            raise ValueError("--adapter_dir required for method=lora_frozen_projector")
        llm = load_lora(base_llm, args.adapter_dir, is_trainable=False)
        logger.info("Loaded LoRA adapter from %s", args.adapter_dir)

        # Load the saved projector state dict (contains the fixed random W buffer)
        projector_path = Path(args.adapter_dir) / "frozen_projector.pt"
        if not projector_path.exists():
            raise FileNotFoundError(
                f"frozen_projector.pt not found in {args.adapter_dir}. "
                "Expected alongside adapter_config.json."
            )

        if is_test:
            _emb_path = (args.user_emb_test_path
                         or str(Path(args.data_dir) / "user_embeddings_with_valid.npz"))
        else:
            _emb_path = args.user_emb_path or str(Path(args.data_dir) / "user_embeddings.npz")
        user_embs = load_user_embeddings(_emb_path)
        sample_emb = next(iter(user_embs.values()))
        user_dim = int(np.array(sample_emb).shape[-1])
        llm_dim = base_llm.config.hidden_size

        # Infer n_soft_tokens and per_token from the saved W shape
        proj_state = torch.load(projector_path, map_location="cpu", weights_only=True)
        W_shape = proj_state["W"].shape   # (m, d_in, d_out) or (d_in, d_out)
        if len(W_shape) == 3:
            n_soft = W_shape[0]
            per_token = True
        else:
            n_soft = args.n_soft_tokens   # fallback to CLI arg
            per_token = False

        projector = FrozenSoftPromptProjector(
            d_in=user_dim,
            d_out=llm_dim,
            n_soft_tokens=n_soft,
            per_token=per_token,
        )
        projector.load_state_dict(proj_state)
        projector.to(args.device)
        logger.info(
            "Loaded FrozenSoftPromptProjector from %s (W shape=%s)",
            projector_path, tuple(proj_state["W"].shape),
        )

        metrics = evaluate_ranker(
            model=llm,
            tokenizer=tokenizer,
            splits=splits,
            negatives=negatives,
            item_meta=item_meta,
            user_emb_table=user_embs,
            projector=projector,
            split=args.split,
            prompt_id=eval_cfg.get("prompt_id", "2-11"),
            max_history=args.max_history if args.max_history is not None else lora_cfg.get("max_history_items", eval_cfg.get("max_history", 20)),
            batch_size=eval_cfg["scoring"]["batch_size"],
            max_seq_length=eval_cfg["scoring"]["max_seq_length"],
            ks=ks,
            device=args.device,
            user_sequences=eval_seqs,
        )

    logger.info("%s (%s, %s): %s", args.method, args.dataset, split_key, metrics)

    # ── Save metrics ─────────────────────────────────────────────────────
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "method": args.method,
        "dataset": args.dataset,
        "split": split_key,
        **metrics,
    }
    out_path = str(results_dir / f"{args.method}_{args.dataset}_eval.json")
    save_metrics(out, out_path)

    # ── Append to run summary ────────────────────────────────────────────
    summary = {
        "method": args.method,
        "dataset": args.dataset,
        "NDCG@10": metrics.get("NDCG@10", 0.0),
        "HR@10": metrics.get("HR@10", 0.0),
        "split": split_key,
    }

    # Add compute info if available from training
    if args.method == "sasrec":
        params = count_params(sasrec)
        summary["trainable_params"] = params["total"]
        summary["total_params"] = params["total"]
    elif args.method == "bert4rec":
        params = count_params(bert4rec_model)
        summary["trainable_params"] = params["total"]
        summary["total_params"] = params["total"]
    elif args.method == "injector" and args.injector_ckpt:
        params = count_params(injector)
        summary["trainable_params"] = params["trainable"]
        summary["total_params"] = params["total"]
    elif args.method == "lora" and args.adapter_dir:
        params = count_params(llm)
        summary["trainable_params"] = params["trainable"]
        summary["total_params"] = params["total"]

    append_run_summary(str(results_dir / "all_runs.jsonl"), summary)
    logger.info("Done — results saved to %s", out_path)


if __name__ == "__main__":
    main()
