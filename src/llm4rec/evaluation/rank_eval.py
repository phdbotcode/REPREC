"""Ranking evaluation: 1 positive + 1000 negatives per user.

For every test (or validation) user:

1. Load the **fixed** negative pool (identical across all model variants).
2. Build 1001 prompts: user history + candidate item + YES/NO question.
3. Score each candidate with ``logP("Yes")`` via :mod:`llm4rec.llm.scoring`.
4. Rank the 1001 items and compute **HR@K** / **NDCG@K** (single-positive).

The function :func:`evaluate_ranker` is model-agnostic: it accepts any
callable that maps ``(input_ids, attention_mask) → logits`` or that uses
``inputs_embeds`` when a ``SoftPromptInjector`` or
``FrozenSoftPromptProjector`` is provided.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llm4rec.data.datasets import LLMPairDataset
from llm4rec.data.negatives import load_negatives
from llm4rec.data.prompts import format_prompt
from llm4rec.llm.collate import LLMCollator
from llm4rec.llm.frozen_projector import FrozenSoftPromptProjector
from llm4rec.llm.injector import SoftPromptInjector, prepend_soft_prompt
from llm4rec.llm.scoring import get_yes_no_token_ids, score_from_logits
from llm4rec.training.train_injector import _prepare_emb_table
from llm4rec.utils.io import load_config, load_json, save_json
from llm4rec.utils.logging import get_logger
from llm4rec.utils.metrics import evaluate_ranking

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ranker(
    model: nn.Module,
    tokenizer: Any,
    splits: Dict[str, Any],
    negatives: Dict[int, List[int]],
    item_meta: Optional[Dict[int, dict]] = None,
    user_emb_table: Optional[Dict[int, np.ndarray] | torch.Tensor] = None,
    injector: Optional[SoftPromptInjector] = None,
    projector: Optional[FrozenSoftPromptProjector] = None,
    split: str = "test",
    prompt_id: str = "2-11",
    max_history: int = 20,
    batch_size: int = 32,
    max_seq_length: int = 512,
    ks: tuple = (1, 5, 10),
    device: str = "cuda",
    return_per_user: bool = False,
    user_sequences: Optional[Dict[int, List[int]]] = None,
) -> "Dict[str, float] | tuple[Dict[str, float], Dict[int, np.ndarray]]":
    """Run full ranking evaluation and return aggregate metrics.

    Parameters
    ----------
    model : LLaMA causal-LM (possibly PEFT-wrapped).
    tokenizer : configured tokenizer.
    splits : preprocessed splits dict with ``train``, ``valid``, ``test``.
    negatives : fixed negative pools ``{user_idx: [neg_item, …]}``.
    item_meta : optional ``{item_idx: {title, …}}`` for readable prompts.
    user_emb_table : SASRec user embeddings (required if *injector* or
        *projector* is set).
    injector : if provided, soft prompt tokens are prepended (Model B).
    projector : if provided, frozen-projected soft prompt tokens are
        prepended (Model C).  Mutually exclusive with *injector*.
    split : ``"val"`` / ``"valid"`` or ``"test"``.
    prompt_id : which prompt template to use.
    max_history : recent items to include in the prompt.
    batch_size : candidates scored per forward pass.
    max_seq_length : tokenizer max length.
    ks : tuple of K values for HR@K, NDCG@K.
    device : torch device.
    user_sequences : optional override for the text history sequences used to
        build prompts.  When ``None`` (default), ``splits["train"]`` is used.
        Pass ``train_plus_valid`` here when evaluating on the test split so
        the LLM prompt history reflects the state right before the test item.

    Returns
    -------
    dict mapping metric names (e.g. ``"HR@10"``, ``"NDCG@10"``, ``"MRR"``)
    to float values.
    """
    dev = torch.device(device)
    model.eval()

    # Resolve split key
    split_key = "valid" if split in ("val", "valid") else "test"
    targets = splits[split_key]
    train_seqs = user_sequences if user_sequences is not None else splits["train"]

    # Build eval dataset (1 pos + all negatives per user)
    # neg_per_pos=0 tells LLMPairDataset to use ALL negatives in the dict
    eval_ds = LLMPairDataset(
        user_sequences=train_seqs,
        targets=targets,
        negatives=negatives,
        item_meta=item_meta,
        max_history=max_history,
        prompt_id=prompt_id,
        mode="eval",
        neg_per_pos=0,
    )
    # Sanity check: log how many candidates per user
    n_users_eval = len(targets)
    n_samples = len(eval_ds)
    if n_users_eval > 0:
        logger.info("evaluate_ranker: %d samples for %d users (avg %.1f candidates/user)",
                     n_samples, n_users_eval, n_samples / n_users_eval)
    collator = LLMCollator(tokenizer, max_length=max_seq_length, mode="eval")
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    # YES/NO token ids
    tok_ids = get_yes_no_token_ids(tokenizer)
    yes_id = tok_ids["yes"][0]
    no_id = tok_ids["no"][0]

    # Optional: prepare embedding table for injector or projector
    emb_table = None
    use_injector = injector is not None
    use_projector = projector is not None
    if use_injector:
        if user_emb_table is None:
            raise ValueError("user_emb_table required when injector is set")
        emb_table = _prepare_emb_table(user_emb_table, dev)
        injector.eval()
    if use_projector:
        if user_emb_table is None:
            raise ValueError("user_emb_table required when projector is set")
        emb_table = _prepare_emb_table(user_emb_table, dev)
        projector.to(dev)
        projector.eval()

    print(f"[evaluate_ranker] max_history={max_history}, max_seq_length={max_seq_length}")

    # ── score all candidates ─────────────────────────────────────────────
    user_scores: Dict[int, Dict[str, list]] = defaultdict(
        lambda: {"scores": [], "labels": []}
    )

    # Detect model dtype for autocast (match model precision)
    _model_dtype = next(model.parameters()).dtype
    _use_autocast = _model_dtype in (torch.float16, torch.bfloat16)

    total_tokens = 0
    total_seqs = 0

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(dev)
        attn_mask = batch["attention_mask"].to(dev)
        user_idx = batch["user_idx"].to(dev)   # → GPU for embedding indexing
        labels_int = batch["labels_int"]        # stays CPU for aggregation

        total_seqs += attn_mask.shape[0]

        with torch.amp.autocast("cuda", dtype=_model_dtype, enabled=_use_autocast):
            if use_injector:
                user_embs = emb_table[user_idx]
                soft_prompt, _ = injector(user_embs)
                text_embeds = model.get_input_embeddings()(input_ids)
                merged = prepend_soft_prompt(text_embeds, attn_mask, soft_prompt)
                logits = model(
                    inputs_embeds=merged["inputs_embeds"],
                    attention_mask=merged["attention_mask"],
                ).logits
                score_mask = merged["attention_mask"]
            elif use_projector:
                user_embs = emb_table[user_idx]
                soft_prompt = projector(user_embs)
                text_embeds = model.get_input_embeddings()(input_ids)
                merged = prepend_soft_prompt(text_embeds, attn_mask, soft_prompt)
                logits = model(
                    inputs_embeds=merged["inputs_embeds"],
                    attention_mask=merged["attention_mask"],
                ).logits
                score_mask = merged["attention_mask"]
            else:
                logits = model(
                    input_ids=input_ids, attention_mask=attn_mask,
                ).logits
                score_mask = attn_mask

        # Count tokens using score_mask so soft-prompt tokens are included
        # for injector/projector, and text-only for base/lora.
        total_tokens += score_mask.sum().item()

        scores = score_from_logits(
            logits, score_mask, yes_id=yes_id, no_id=no_id,
            mode="logprob_yes",
        )

        user_idx_cpu = user_idx.cpu()
        for i in range(len(user_idx_cpu)):
            uid = user_idx_cpu[i].item()
            user_scores[uid]["scores"].append(scores[i].item())
            user_scores[uid]["labels"].append(labels_int[i].item())

    avg_tokens = total_tokens / total_seqs if total_seqs > 0 else 0.0
    _len_label = "text + soft tokens" if (use_injector or use_projector) else "text tokens"
    print(f"[evaluate_ranker] avg prompt length ({_len_label}, excl. padding) = {avg_tokens:.1f} "
          f"over {total_seqs} prompts ({n_users_eval} users)")

    # ── aggregate per-user → metric arrays ───────────────────────────────
    all_score_arrays: List[np.ndarray] = []
    per_user_score_arrays: Dict[int, np.ndarray] = {}
    for uid, data in user_scores.items():
        s = np.array(data["scores"])
        l = np.array(data["labels"])
        pos_mask = l == 1
        if not pos_mask.any():
            continue
        # positive first (evaluate_ranking expects positive_idx=0)
        arr = np.concatenate([s[pos_mask], s[~pos_mask]])
        all_score_arrays.append(arr)
        per_user_score_arrays[uid] = arr

    metrics = evaluate_ranking(all_score_arrays, ks=ks, positive_idx=0)
    metrics["num_users"] = len(all_score_arrays)

    logger.info(
        "Ranking eval (%s, %d users): %s",
        split_key, len(all_score_arrays),
        "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "num_users"),
    )
    if return_per_user:
        return metrics, per_user_score_arrays
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry-point: evaluate a trained ranker and write results JSON."""
    parser = argparse.ArgumentParser(description="Evaluate LLM ranker")
    parser.add_argument("--eval_cfg", required=True, help="Path to eval.yaml")
    parser.add_argument("--data_dir", required=True, help="Preprocessed data dir")
    parser.add_argument("--model_type", choices=["injector", "lora", "base"],
                        default="lora")
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--adapter_dir", default=None, help="LoRA adapter dir")
    parser.add_argument("--injector_ckpt", default=None, help="Injector .pt")
    parser.add_argument("--user_emb_path", default=None, help="User embeddings .npz")
    parser.add_argument("--split", default="test", choices=["val", "valid", "test"])
    parser.add_argument("--output", default="outputs/results/metrics.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.eval_cfg)["eval"]

    # Load data artefacts
    splits = load_json(os.path.join(args.data_dir, "splits.json"))
    # Convert string keys back to ints
    for section in ("train", "valid", "test"):
        splits[section] = {int(k): v for k, v in splits[section].items()}

    neg_path = os.path.join(args.data_dir, "negatives_200.jsonl")
    negatives = load_negatives(neg_path)

    item_meta = None
    meta_path = os.path.join(args.data_dir, "item_meta.json")
    if os.path.exists(meta_path):
        raw = load_json(meta_path)
        item_meta = {int(k): v for k, v in raw.items()}

    # Load model
    from llm4rec.llm.llama_backbone import load_llama
    model, tokenizer = load_llama(
        args.model_name, dtype="bf16", device_map=args.device,
    )

    injector = None
    user_emb_table = None

    if args.model_type == "lora" and args.adapter_dir:
        from llm4rec.llm.lora import load_lora
        model = load_lora(model, args.adapter_dir)

    if args.model_type == "injector" and args.injector_ckpt:
        from llm4rec.llm.injector import SoftPromptInjector
        # Infer dims from checkpoint
        state = torch.load(args.injector_ckpt, map_location="cpu", weights_only=False)
        injector = SoftPromptInjector()  # defaults; overridden by load
        injector.load_state_dict(state)
        injector.to(args.device)

    if args.user_emb_path:
        from llm4rec.sasrec.infer import load_user_embeddings
        user_emb_table = load_user_embeddings(args.user_emb_path)

    # Evaluate
    metrics = evaluate_ranker(
        model=model,
        tokenizer=tokenizer,
        splits=splits,
        negatives=negatives,
        item_meta=item_meta,
        user_emb_table=user_emb_table,
        injector=injector,
        split=args.split,
        prompt_id=cfg.get("prompt_id", "2-11"),
        max_history=cfg.get("max_history", 20),
        batch_size=cfg["scoring"]["batch_size"],
        max_seq_length=cfg["scoring"]["max_seq_length"],
        ks=tuple(cfg["ks"]),
        device=args.device,
    )

    save_json(metrics, args.output)
    logger.info("Results saved → %s", args.output)


if __name__ == "__main__":
    main()
