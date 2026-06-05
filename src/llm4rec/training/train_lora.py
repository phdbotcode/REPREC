"""Train LoRA adapters on LLaMA (base weights frozen, SASRec frozen).

Three conditioning modes:

* **Model A  (LoRA-only)** — the text prompt (with purchase history) is fed
  through ``input_ids`` directly.  Only LoRA adapter weights are trained.
* **Model B  (LoRA + Injector)** — a SASRec user embedding is projected into
  soft prompt tokens via a *trainable* ``SoftPromptInjector`` MLP and
  prepended to the text embeddings.  Both LoRA adapters **and** the injector
  MLP are trained jointly.
* **Model C  (LoRA + FrozenProjector)** — a SASRec user embedding is mapped
  to soft prompt tokens via a *frozen* random projection
  (``FrozenSoftPromptProjector``).  Only LoRA adapters are trained.

Training uses **pair-level** embeddings (prefix-aware, indexed by
``pair_idx``) from ``PrebuiltPairDataset``.  Evaluation uses **user-level**
embeddings (full-sequence, indexed by ``user_idx``).
"""

from __future__ import annotations

import copy
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llm4rec.llm.frozen_projector import FrozenSoftPromptProjector
from llm4rec.llm.injector import SoftPromptInjector, prepend_soft_prompt
from llm4rec.llm.lora import save_lora
from llm4rec.llm.scoring import score_from_logits
from llm4rec.training.optim import build_optimizer, build_scheduler, grad_clip, set_requires_grad
from llm4rec.training.train_injector import _prepare_emb_table
from llm4rec.utils.logging import get_logger
from llm4rec.utils.metrics import evaluate_ranking

logger = get_logger(__name__)


class LoRATrainer:
    """Train LoRA adapters (+ optional injector / frozen projector) on LLaMA.

    Parameters
    ----------
    sasrec : trained SASRec model (always frozen).
    llm_lora : LLaMA model already wrapped with
        :func:`~llm4rec.llm.lora.build_lora_model`.
    tokenizer : configured tokenizer.
    cfg : training config dict (the ``lora.training`` sub-key).
    device : torch device.
    injector : *Model B* — trainable soft-prompt injector (mutually
        exclusive with *projector*).
    projector : *Model C* — frozen random-projection soft-prompt
        generator (mutually exclusive with *injector*).
    """

    def __init__(
        self,
        sasrec: nn.Module,
        llm_lora: nn.Module,
        tokenizer: Any,
        cfg: Dict[str, Any],
        device: str = "cuda",
        injector: Optional[SoftPromptInjector] = None,
        projector: Optional[FrozenSoftPromptProjector] = None,
    ) -> None:
        # ── mutual exclusivity check ─────────────────────────────────────
        if injector is not None and projector is not None:
            raise ValueError(
                "injector and projector are mutually exclusive — "
                "pass at most one (Model B xor Model C)."
            )

        self.device = torch.device(device)
        self.cfg = cfg

        # Freeze SASRec
        self.sasrec = sasrec
        set_requires_grad(self.sasrec, False)
        self.sasrec.eval()

        # LLaMA + LoRA — PEFT already froze the base; adapters are trainable
        self.llm = llm_lora
        self.llm.train()

        # ── Model B: trainable injector ──────────────────────────────────
        self.injector = injector
        self.use_injector = injector is not None
        if self.use_injector:
            set_requires_grad(self.injector, True)
            self.injector.to(self.device)

        # ── Model C: frozen projector ────────────────────────────────────
        self.projector = projector
        self.use_projector = projector is not None
        if self.use_projector:
            set_requires_grad(self.projector, False)
            self.projector.to(self.device)
            self.projector.eval()
            n_proj_params = sum(p.numel() for p in self.projector.parameters())
            assert n_proj_params == 0, (
                f"FrozenSoftPromptProjector must have 0 trainable parameters, "
                f"got {n_proj_params}"
            )
            logger.info(
                "Model C enabled: FrozenSoftPromptProjector "
                "(W shape=%s, scale=%.4f, normalize=%s).  "
                "Trainable params = LoRA only.",
                tuple(self.projector.W.shape),
                self.projector.scale,
                self.projector.normalize,
            )

        self.tokenizer = tokenizer

        # YES / NO token ids for ranking evaluation
        from llm4rec.llm.scoring import get_yes_no_token_ids
        tok_ids = get_yes_no_token_ids(tokenizer)
        self.yes_id = tok_ids["yes"][0]
        self.no_id = tok_ids["no"][0]

        # Mixed precision
        self.use_amp = cfg.get("fp16", False)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ── FLOPs profiling ───────────────────────────────────────────────────

    def _profile_step_flops(
        self,
        batch: Dict[str, torch.Tensor],
        emb_table: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        """Profile forward-pass FLOPs for one step via ``torch.profiler``.

        Only the forward pass is profiled; total per-step training FLOPs are
        estimated as **3× forward** (1× fwd + 2× bwd, where LoRA backward
        ≈ 1× fwd and weight-gradient overhead is small relative to the base
        model).

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

    # ── collect trainable parameters ─────────────────────────────────────

    def _trainable_params(self) -> List[torch.nn.Parameter]:
        """Gather all trainable parameters (LoRA + optional injector).

        Note: projector has NO trainable parameters — it is excluded.
        """
        params = [p for p in self.llm.parameters() if p.requires_grad]
        if self.use_injector:
            params += [p for p in self.injector.parameters() if p.requires_grad]
        # projector: intentionally excluded (zero trainable params)
        return params

    # ── main entry ───────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        pair_emb_table: Optional[torch.Tensor] = None,
        user_emb_table: Optional[Dict[int, np.ndarray] | torch.Tensor] = None,
        patience: int = 5,
        eval_ranking_every: int = 1,
        save_dir: Optional[str] = None,
    ) -> nn.Module:
        """Train and return the best LoRA model.

        Parameters
        ----------
        train_loader : DataLoader with ``LLMCollator(mode="train")``.
            For Models B/C this should use ``PrebuiltPairDataset`` so
            batches contain ``pair_idx``.
        val_loader : DataLoader with ``LLMCollator(mode="eval")``.
        pair_emb_table : ``(num_pairs, D)`` prefix-aware SASRec embeddings
            indexed by ``pair_idx``.  Required for training when using
            injector or projector.
        user_emb_table : SASRec user embeddings indexed by ``user_idx``
            (full-sequence).  Required for eval when using injector or
            projector.
        patience : early-stopping patience.
        eval_ranking_every : run ranking eval every N epochs (0 = never).
        save_dir : directory for best LoRA adapter checkpoint.

        Returns
        -------
        The LoRA-wrapped LLM with best-checkpoint weights.
        """
        # ── prepare embedding tables ─────────────────────────────────────
        train_emb = None   # pair-indexed, for _forward_step
        eval_emb = None    # user-indexed, for _eval_ranking

        if self.use_injector or self.use_projector:
            if pair_emb_table is None:
                raise ValueError(
                    "pair_emb_table is required for training when using "
                    "injector or projector"
                )
            train_emb = pair_emb_table.to(self.device)

            if user_emb_table is not None:
                eval_emb = _prepare_emb_table(user_emb_table, self.device)
            else:
                logger.warning(
                    "user_emb_table not provided — validation ranking "
                    "will run without soft-prompt conditioning"
                )

        num_epochs = self.cfg.get("num_epochs", 3)
        max_norm = self.cfg.get("grad_clip", 1.0)

        trainable_params = self._trainable_params()
        optimizer = build_optimizer(trainable_params, self.cfg)
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
        mode_str = ("Model B (LoRA + Injector)" if self.use_injector
                     else "Model C (LoRA + FrozenProjector)" if self.use_projector
                     else "Model A (LoRA-only)")
        logger.info("Training started — %s", mode_str)
        logger.info("  %d epochs, %d batches/epoch, %d total steps",
                     num_epochs, len(train_loader), num_steps)
        logger.info("  trainable params: %d",
                     sum(p.numel() for p in trainable_params))
        if train_emb is not None:
            logger.info("  train emb table (pair-level): %s", tuple(train_emb.shape))
        if eval_emb is not None:
            logger.info("  eval emb table (user-level): %s", tuple(eval_emb.shape))
        logger.info("=" * 60)

        best_metric = float("inf")
        best_lora_state: Optional[dict] = None
        best_inj_state: Optional[dict] = None
        wait = 0
        train_start = time.time()

        for epoch in range(1, num_epochs + 1):
            # ── train ────────────────────────────────────────────────
            epoch_start = time.time()
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler, train_emb, max_norm,
                epoch=epoch, num_epochs=num_epochs,
            )
            epoch_time = time.time() - epoch_start

            # ── validate ─────────────────────────────────────────────
            log_parts = [f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | {epoch_time:.1f}s"]
            metric = train_loss

            if val_loader is not None and eval_ranking_every > 0 and epoch % eval_ranking_every == 0:
                val_metrics = self._eval_ranking(val_loader, eval_emb)
                ndcg10 = val_metrics.get("NDCG@10", 0.0)
                hr10 = val_metrics.get("HR@10", 0.0)
                log_parts.append(f"val NDCG@10 {ndcg10:.4f} | val HR@10 {hr10:.4f}")
                metric = -ndcg10

            logger.info(" | ".join(log_parts))

            # ── checkpoint + early stopping ──────────────────────────
            if metric < best_metric:
                best_metric = metric
                # Shallow-copy LoRA adapter state (small)
                best_lora_state = copy.deepcopy(
                    {k: v for k, v in self.llm.state_dict().items()}
                )
                if self.use_injector:
                    best_inj_state = copy.deepcopy(self.injector.state_dict())
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

        # ── restore best weights ─────────────────────────────────────
        if best_lora_state is not None:
            self.llm.load_state_dict(best_lora_state)
        if best_inj_state is not None and self.use_injector:
            self.injector.load_state_dict(best_inj_state)

        # ── save adapter ─────────────────────────────────────────────
        if save_dir is not None:
            save_lora(self.llm, save_dir)
            if self.use_injector:
                import os
                inj_path = os.path.join(save_dir, "injector.pt")
                torch.save(self.injector.state_dict(), inj_path)
                logger.info("Injector checkpoint saved → %s", inj_path)
            if self.use_projector:
                import os
                proj_path = os.path.join(save_dir, "frozen_projector.pt")
                torch.save(self.projector.state_dict(), proj_path)
                logger.info("FrozenProjector buffers saved → %s", proj_path)

        return self.llm

    # ── one epoch ────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        emb_table: Optional[torch.Tensor],
        max_norm: float,
        epoch: int = 0,
        num_epochs: int = 0,
    ) -> float:
        self.llm.train()
        if self.use_injector:
            self.injector.train()
        # projector stays in eval mode (frozen)

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
                    "  NaN/Inf loss at epoch %d step %d (count=%d), skipping batch",
                    epoch, step, nan_count,
                )
                if nan_count >= 10:
                    raise RuntimeError(
                        f"Too many NaN losses ({nan_count}) — aborting training. "
                        "Check embeddings, learning rate, and data."
                    )
                continue  # skip this batch

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            grad_clip(self._trainable_params(), max_norm)
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

    # ── single forward step (training — uses pair_idx) ───────────────────

    def _forward_step(
        self,
        batch: Dict[str, torch.Tensor],
        emb_table: Optional[torch.Tensor],
    ) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            if self.use_injector:
                # ── Model B: trainable injector ──────────────────────
                # Training data comes from PrebuiltPairDataset → pair_idx
                emb_idx = batch["pair_idx"] if "pair_idx" in batch else batch["user_idx"]
                user_embs = emb_table[emb_idx]
                soft_prompt, _ = self.injector(user_embs)
                text_embeds = self.llm.get_input_embeddings()(input_ids)
                merged = prepend_soft_prompt(text_embeds, attn_mask, soft_prompt, labels)
                outputs = self.llm(
                    inputs_embeds=merged["inputs_embeds"],
                    attention_mask=merged["attention_mask"],
                    labels=merged["labels"],
                )
            elif self.use_projector:
                # ── Model C: frozen projector ────────────────────────
                emb_idx = batch["pair_idx"] if "pair_idx" in batch else batch["user_idx"]
                user_embs = emb_table[emb_idx]
                soft_prompt = self.projector(user_embs)         # (B, m, D)
                text_embeds = self.llm.get_input_embeddings()(input_ids)
                merged = prepend_soft_prompt(text_embeds, attn_mask, soft_prompt, labels)
                outputs = self.llm(
                    inputs_embeds=merged["inputs_embeds"],
                    attention_mask=merged["attention_mask"],
                    labels=merged["labels"],
                )
            else:
                # ── Model A: LoRA-only ───────────────────────────────
                outputs = self.llm(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                )

        return outputs.loss

    # ── ranking evaluation (uses user_idx) ───────────────────────────────

    @torch.no_grad()
    def _eval_ranking(
        self,
        eval_loader: DataLoader,
        emb_table: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """Score all candidates and compute NDCG@10 / HR@10."""
        self.llm.eval()
        if self.use_injector:
            self.injector.eval()
        # projector is always in eval mode

        user_scores: Dict[int, Dict[str, list]] = defaultdict(
            lambda: {"scores": [], "labels": []}
        )

        total_tokens = 0
        total_seqs = 0

        for batch in eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            user_idx = batch["user_idx"]
            labels_int = batch["labels_int"]

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if (self.use_injector or self.use_projector) and emb_table is not None:
                    user_embs = emb_table[user_idx]
                    if self.use_injector:
                        soft_prompt, _ = self.injector(user_embs)
                    else:
                        soft_prompt = self.projector(user_embs)
                    text_embeds = self.llm.get_input_embeddings()(input_ids)
                    merged = prepend_soft_prompt(text_embeds, attn_mask, soft_prompt)
                    logits = self.llm(
                        inputs_embeds=merged["inputs_embeds"],
                        attention_mask=merged["attention_mask"],
                    ).logits
                    score_mask = merged["attention_mask"]
                else:
                    # Model A, or fallback if no eval emb_table
                    logits = self.llm(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                    ).logits
                    score_mask = attn_mask

            # score_mask covers text + soft tokens (if any)
            total_tokens += score_mask.sum().item()
            total_seqs += attn_mask.shape[0]

            scores = score_from_logits(
                logits, score_mask,
                yes_id=self.yes_id, no_id=self.no_id,
                mode="logprob_yes",
            )

            for i in range(len(user_idx)):
                uid = user_idx[i].item()
                user_scores[uid]["scores"].append(scores[i].item())
                user_scores[uid]["labels"].append(labels_int[i].item())

        if total_seqs > 0:
            avg_tokens = total_tokens / total_seqs
            if self.use_injector:
                _len_label = f"text + {self.injector.n_soft_tokens} soft tokens"
            elif self.use_projector:
                _len_label = f"text + {self.projector.n_soft_tokens} soft tokens"
            else:
                _len_label = "text tokens"
            logger.info(
                "[val] avg prompt length (%s, excl. padding) = %.1f over %d prompts",
                _len_label, avg_tokens, total_seqs,
            )

        # Aggregate per-user → positive-first arrays
        all_score_arrays = []
        for uid, data in user_scores.items():
            s = np.array(data["scores"])
            l = np.array(data["labels"])
            pos_mask = l == 1
            if not pos_mask.any():
                continue
            all_score_arrays.append(
                np.concatenate([s[pos_mask], s[~pos_mask]])
            )

        return evaluate_ranking(all_score_arrays, ks=(10,), positive_idx=0)
