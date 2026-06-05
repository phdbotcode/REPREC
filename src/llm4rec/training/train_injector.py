"""Train the SoftPromptInjector while SASRec and LLaMA stay frozen.

Per-batch forward flow
----------------------
1. Look up pre-computed SASRec embeddings by ``pair_idx`` (training) or
   ``user_idx`` (evaluation).
2. ``injector(user_emb)`` → soft prompt tokens ``(B, m, D)``.
3. ``llm.get_input_embeddings()(input_ids)`` → text embeddings.
4. ``prepend_soft_prompt(text_embeds, attn_mask, soft_prompt, labels)``
   → merged inputs with ``-100`` over soft tokens.
5. ``llm(inputs_embeds=…, attention_mask=…, labels=…)`` → CE loss
   on answer tokens only.
6. Back-prop through frozen LLM (treated as a fixed function) into
   the injector MLP — only injector weights are updated.
"""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llm4rec.llm.injector import SoftPromptInjector, prepend_soft_prompt
from llm4rec.llm.scoring import score_from_logits
from llm4rec.training.optim import build_optimizer, build_scheduler, grad_clip, set_requires_grad
from llm4rec.utils.logging import get_logger
from llm4rec.utils.metrics import evaluate_ranking

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding-table helper
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_emb_table(
    user_embs: Dict[int, np.ndarray] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Convert a user-embedding dict to a dense lookup tensor on *device*.

    If already a ``Tensor``, just move to device.
    """
    if isinstance(user_embs, torch.Tensor):
        return user_embs.to(device)
    max_uid = max(user_embs.keys())
    dim = next(iter(user_embs.values())).shape[0]
    table = torch.zeros(max_uid + 1, dim)
    for uid, emb in user_embs.items():
        table[uid] = torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
    return table.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class InjectorTrainer:
    """Train the soft-prompt injector MLP (everything else frozen).

    Parameters
    ----------
    sasrec : trained SASRec model (frozen, used only if computing
        embeddings on the fly — otherwise ignored).
    llm : LLaMA causal-LM (frozen).
    injector : ``SoftPromptInjector`` module (the only trainable part).
    tokenizer : configured tokenizer.
    cfg : training config dict (the ``injector.training`` sub-key).
    device : torch device.
    """

    def __init__(
        self,
        sasrec: nn.Module,
        llm: nn.Module,
        injector: SoftPromptInjector,
        tokenizer: Any,
        cfg: Dict[str, Any],
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.cfg = cfg

        # Freeze SASRec and LLaMA — only injector is trainable
        self.sasrec = sasrec
        set_requires_grad(self.sasrec, False)
        self.sasrec.eval()

        self.llm = llm
        set_requires_grad(self.llm, False)
        self.llm.eval()

        self.injector = injector.to(self.device)
        set_requires_grad(self.injector, True)

        self.tokenizer = tokenizer

        # Resolve YES / NO token ids (first token for prompt-only scoring)
        from llm4rec.llm.scoring import get_yes_no_token_ids
        tok_ids = get_yes_no_token_ids(tokenizer)
        if len(tok_ids["yes"]) != 1 or len(tok_ids["no"]) != 1:
            logger.warning(
                "Yes/No are multi-token: yes=%s, no=%s — "
                "prompt-only scoring uses only the first token.",
                tok_ids["yes"], tok_ids["no"],
            )
        self.yes_id = tok_ids["yes"][0]
        self.no_id = tok_ids["no"][0]

        # Mixed precision — match autocast dtype to model dtype.
        # bf16 model weights + fp16 autocast causes instability; use bf16
        # autocast and disable GradScaler (bf16 has same exponent as fp32).
        amp_dtype_str = cfg.get("amp_dtype", None)
        if amp_dtype_str == "bf16":
            self.use_amp = True
            self.amp_dtype = torch.bfloat16
            self._use_scaler = False
        elif amp_dtype_str == "fp16" or cfg.get("fp16", False):
            self.use_amp = True
            self.amp_dtype = torch.float16
            self._use_scaler = True
        else:
            self.use_amp = False
            self.amp_dtype = torch.float32
            self._use_scaler = False
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_scaler)
        logger.info("AMP: enabled=%s  dtype=%s  GradScaler=%s",
                     self.use_amp, self.amp_dtype, self._use_scaler)

    # ── main entry ───────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        train_emb_table: torch.Tensor,
        eval_emb_table: Dict[int, np.ndarray] | torch.Tensor,
        patience: int = 5,
        eval_ranking_every: int = 1,
    ) -> SoftPromptInjector:
        """Train and return the best injector (by val loss or NDCG@10).

        Parameters
        ----------
        train_loader : DataLoader with ``LLMCollator(mode="train")``.
            Batches must contain ``pair_idx`` for embedding lookup.
        val_loader : DataLoader with ``LLMCollator(mode="eval")``.
            If ``None``, no validation is performed.
        train_emb_table : ``(num_pairs, D)`` tensor indexed by ``pair_idx``.
        eval_emb_table : user embeddings indexed by ``user_idx`` (for val
            ranking).
        patience : early-stopping patience (epochs without improvement).
        eval_ranking_every : run ranking eval every N epochs (0 = never).

        Returns
        -------
        The ``SoftPromptInjector`` with best-checkpoint weights loaded.
        """
        train_emb = train_emb_table.to(self.device)
        eval_emb = _prepare_emb_table(eval_emb_table, self.device)

        num_epochs = self.cfg.get("num_epochs", 5)
        max_norm = self.cfg.get("grad_clip", 1.0)

        optimizer = build_optimizer(self.injector, self.cfg)
        num_steps = len(train_loader) * num_epochs
        scheduler = build_scheduler(optimizer, self.cfg, num_steps)

        # --- FLOPs per epoch (one-time profiler measurement) ---
        _prof_batch = next(iter(train_loader))
        fwd_flops, step_flops = self._profile_step_flops(_prof_batch, train_emb)
        steps_per_epoch = len(train_loader)
        if fwd_flops > 0:
            logger.info(
                "FLOPs/step (profiler): fwd=%.2e  train≈%.2e (3×fwd)  per epoch≈%.2e (%d steps)",
                fwd_flops, step_flops, step_flops * steps_per_epoch, steps_per_epoch,
            )
        else:
            logger.warning("Profiler returned 0 FLOPs — ops may use unsupported kernels (e.g. FlashAttention).")

        logger.info("=" * 60)
        logger.info("Training started — %d epochs, %d batches/epoch, %d total steps",
                     num_epochs, len(train_loader), num_steps)
        logger.info("Train emb table: %s | Eval emb table: %s",
                     tuple(train_emb.shape), tuple(eval_emb.shape))
        logger.info("=" * 60)

        best_metric = float("inf")   # lower val loss = better
        best_state: Optional[dict] = None
        wait = 0
        train_start = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # ── train ────────────────────────────────────────────────
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler, train_emb, max_norm,
                epoch=epoch, num_epochs=num_epochs,
            )

            epoch_time = time.time() - epoch_start

            # ── validate ─────────────────────────────────────────────
            lr = optimizer.param_groups[0]["lr"]
            log_parts = [
                f"Epoch {epoch:3d}/{num_epochs}",
                f"train_loss {train_loss:.4f}",
                f"lr {lr:.2e}",
                f"time {epoch_time:.1f}s",
            ]
            metric = train_loss  # fallback if no val

            if val_loader is not None and eval_ranking_every > 0 and epoch % eval_ranking_every == 0:
                val_metrics = self._eval_ranking(val_loader, eval_emb)
                ndcg10 = val_metrics.get("NDCG@10", 0.0)
                hr10 = val_metrics.get("HR@10", 0.0)
                log_parts.append(f"val NDCG@10 {ndcg10:.4f}")
                log_parts.append(f"val HR@10 {hr10:.4f}")
                metric = -ndcg10  # negate so "lower = better" still applies

            logger.info(" | ".join(log_parts))

            # ── early stopping ───────────────────────────────────────
            if metric < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(self.injector.state_dict())
                wait = 0
                logger.info("  -> New best checkpoint saved")
            else:
                wait += 1
                logger.info("  -> No improvement (%d/%d patience)", wait, patience)
                if wait >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        total_time = time.time() - train_start
        logger.info("=" * 60)
        logger.info("Training complete — %d epochs in %.1fs (%.1fs/epoch)",
                     epoch, total_time, total_time / epoch)
        logger.info("=" * 60)

        if best_state is not None:
            self.injector.load_state_dict(best_state)
        return self.injector

    # ── FLOPs profiling ───────────────────────────────────────────────────

    def _profile_step_flops(
        self,
        batch: Dict[str, torch.Tensor],
        emb_table: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        """Profile forward-pass FLOPs for one step via ``torch.profiler``.

        Only the forward pass is profiled; total per-step training FLOPs are
        estimated as **3× forward** (1× fwd + 2× bwd, where frozen-LLM
        backward ≈ 1× fwd and injector backward is negligible).

        Note: ``torch.profiler`` counts FLOPs for ``aten::mm`` / ``aten::bmm``
        / ``aten::addmm``.  FlashAttention custom kernels may not report FLOPs,
        in which case ``fwd_flops`` will be 0 and a warning is logged.

        Returns
        -------
        ``(fwd_flops, step_flops)`` where ``step_flops ≈ 3 × fwd_flops``.
        Returns ``(0, 0)`` on failure.
        """
        import torch
        from torch.profiler import ProfilerActivity, profile
        try:
            torch.cuda.reset_peak_memory_stats()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_flops=True,
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                with torch.no_grad():
                    self._forward_step(batch, emb_table)
            fwd_flops = sum(e.flops for e in prof.key_averages())
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
            logger.info("Peak GPU memory (fwd profiler step): %.2f GB", peak_mem_gb)
            return fwd_flops, 3 * fwd_flops
        except Exception as exc:
            logger.warning("FLOPs profiling failed: %s", exc)
            return 0, 0

    # ── one epoch ────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        emb_table: torch.Tensor,
        max_norm: float,
        epoch: int = 0,
        num_epochs: int = 0,
    ) -> float:
        self.injector.train()
        total_loss = 0.0
        n_batches = len(loader)
        log_every = max(n_batches // 5, 1)  # log ~5 times per epoch
        nan_count = 0

        for step, batch in enumerate(loader, 1):
            loss = self._forward_step(batch, emb_table)

            # ── NaN detection ────────────────────────────────────────
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                logger.warning(
                    "  NaN/Inf loss at epoch %d step %d (count=%d). "
                    "pair_idx range=[%d, %d], batch_size=%d",
                    epoch, step, nan_count,
                    batch["pair_idx"].min().item(),
                    batch["pair_idx"].max().item(),
                    batch["pair_idx"].shape[0],
                )
                # Debug: check injector outputs
                with torch.no_grad():
                    for name, param in self.injector.named_parameters():
                        if torch.isnan(param).any():
                            logger.warning("    NaN in param: %s", name)
                        if torch.isinf(param).any():
                            logger.warning("    Inf in param: %s", name)
                if nan_count >= 10:
                    raise RuntimeError(
                        f"Too many NaN losses ({nan_count}) — aborting training. "
                        "Check embeddings, learning rate, and data."
                    )
                continue  # skip this batch

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            grad_clip(self.injector, max_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if step % log_every == 0 or step == n_batches:
                avg_loss = total_loss / max(step - nan_count, 1)
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "  [Epoch %d/%d] Step %d/%d | batch_loss %.4f | avg_loss %.4f | lr %.2e",
                    epoch, num_epochs, step, n_batches,
                    loss.item(), avg_loss, lr,
                )

        return total_loss / max(n_batches - nan_count, 1)

    # ── single forward step ──────────────────────────────────────────────

    def _forward_step(
        self,
        batch: Dict[str, torch.Tensor],
        emb_table: torch.Tensor,
    ) -> torch.Tensor:
        """Injector forward → prepend → LLM forward → loss."""
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Use pair_idx for embedding lookup (maps to train_pair_embeddings.pt)
        emb_idx = batch["pair_idx"].to(self.device)  # (B,) → GPU

        # 1. User embeddings (indexed by pair_idx)
        user_embs = emb_table[emb_idx]  # (B, d_user)

        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            # 2. Soft prompt
            soft_prompt, _ = self.injector(user_embs)  # (B, m, D)

            # 3. Text embeddings (through frozen LLM embedding layer)
            text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, L, D)

            # 4. Merge
            merged = prepend_soft_prompt(text_embeds, attn_mask, soft_prompt, labels)

            # 5. LLM forward (frozen) → CE loss on answer tokens
            outputs = self.llm(
                inputs_embeds=merged["inputs_embeds"],
                attention_mask=merged["attention_mask"],
                labels=merged["labels"],
            )

        return outputs.loss

    # ── ranking evaluation ───────────────────────────────────────────────

    @torch.no_grad()
    def _eval_ranking(
        self,
        eval_loader: DataLoader,
        emb_table: torch.Tensor,
    ) -> Dict[str, float]:
        """Score all candidates and compute NDCG@10 / HR@10."""
        self.injector.eval()

        user_scores: Dict[int, Dict[str, list]] = defaultdict(
            lambda: {"scores": [], "labels": []}
        )

        total_tokens = 0
        total_seqs = 0

        for batch in eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            user_idx = batch["user_idx"].to(self.device)  # → GPU for indexing
            labels_int = batch["labels_int"]  # 0/1 (stays CPU for aggregation)

            # Eval uses user_idx (maps to user_embeddings.npz)
            user_embs = emb_table[user_idx]

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                soft_prompt, _ = self.injector(user_embs)
                text_embeds = self.llm.get_input_embeddings()(input_ids)
                merged = prepend_soft_prompt(text_embeds, attn_mask, soft_prompt)

                logits = self.llm(
                    inputs_embeds=merged["inputs_embeds"],
                    attention_mask=merged["attention_mask"],
                ).logits

            # merged["attention_mask"] covers text + soft tokens
            total_tokens += merged["attention_mask"].sum().item()
            total_seqs += attn_mask.shape[0]

            scores = score_from_logits(
                logits, merged["attention_mask"],
                yes_id=self.yes_id, no_id=self.no_id,
                mode="logprob_yes",
            )

            # Move back to CPU for aggregation
            user_idx_cpu = user_idx.cpu()
            for i in range(len(user_idx_cpu)):
                uid = user_idx_cpu[i].item()
                user_scores[uid]["scores"].append(scores[i].item())
                user_scores[uid]["labels"].append(labels_int[i].item())

        if total_seqs > 0:
            avg_tokens = total_tokens / total_seqs
            logger.info(
                "[val] avg prompt length (text + %d soft tokens, excl. padding) = %.1f "
                "over %d prompts",
                self.injector.n_soft_tokens, avg_tokens, total_seqs,
            )

        # Aggregate per-user scores → metric arrays
        all_score_arrays = []
        for uid, data in user_scores.items():
            s = np.array(data["scores"])
            l = np.array(data["labels"])
            # Place positive score first (evaluate_ranking expects positive_idx=0)
            pos_mask = l == 1
            if not pos_mask.any():
                continue
            pos_scores = s[pos_mask]
            neg_scores = s[~pos_mask]
            all_score_arrays.append(
                np.concatenate([pos_scores, neg_scores])
            )

        return evaluate_ranking(all_score_arrays, ks=(10,), positive_idx=0)
