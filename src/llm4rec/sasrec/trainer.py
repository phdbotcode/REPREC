"""SASRec training loop with sampled softmax loss and early stopping.

**Data-split strategy (no LLM leakage)**

The leave-one-out preprocessor produces per-user splits::

    original sequence:  [i1, i2, …, i_{n-2},  i_{n-1},  i_n]
                         ╰── train ──╯        valid     test

``valid`` and ``test`` are reserved **exclusively** for LLM evaluation.

Inside *this* trainer we further split the ``train`` portion::

    train portion:  [i1, i2, …, i_{n-3},  i_{n-2}]
                     ╰── sasrec_train ──╯  sasrec_val

So SASRec never touches ``i_{n-1}`` or ``i_n``, and the SASRec
validation item ``i_{n-2}`` is distinct from the LLM validation item.

**Loss** – Sampled softmax: at each non-padded time-step the positive
item is scored against ``num_neg_train`` sampled negatives and a
cross-entropy loss is applied over the resulting (1 + K) logits.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from llm4rec.data.datasets import SASRecDataset
from llm4rec.sasrec.model import SASRec
from llm4rec.utils.logging import get_logger
from llm4rec.utils.metrics import evaluate_ranking
from llm4rec.utils.seed import set_seed

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sampled softmax loss
# ─────────────────────────────────────────────────────────────────────────────

def sampled_softmax_loss(
    hidden: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    item_emb: nn.Embedding,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over positive + sampled negatives at each position.

    Parameters
    ----------
    hidden : ``(B, L, D)`` encoder outputs.
    pos_ids : ``(B, L)`` ground-truth next-item IDs.
    neg_ids : ``(B, L)`` sampled negative item IDs.
    item_emb : the shared item embedding table.
    mask : ``(B, L)`` bool — ``True`` at valid (non-pad) positions.

    Returns
    -------
    Scalar loss (mean over valid positions).
    """
    pos_emb = item_emb(pos_ids)                      # (B, L, D)
    neg_emb = item_emb(neg_ids)                      # (B, L, D)

    # Dot-product scores
    pos_score = (hidden * pos_emb).sum(-1, keepdim=True)  # (B, L, 1)
    neg_score = (hidden * neg_emb).sum(-1, keepdim=True)  # (B, L, 1)

    # Logits: [positive, negative]  →  (B, L, 2)
    logits = torch.cat([pos_score, neg_score], dim=-1)

    # Target is always index 0 (the positive) at every position
    targets = torch.zeros(logits.size(0), logits.size(1),
                          dtype=torch.long, device=logits.device)

    # Flatten to (B*L, 2) and (B*L,), then mask
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    return loss[mask_flat].mean()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SASRecTrainer:
    """Train and validate SASRec using only the *train* split.

    Parameters
    ----------
    config : dict parsed from ``configs/sasrec.yaml`` (the ``sasrec`` sub-key).
    device : ``"cuda"`` / ``"cpu"``.
    """

    def __init__(self, config: Dict[str, Any], device: str = "cuda") -> None:
        self.cfg = config
        self.device = torch.device(device)
        self.max_len = config.get("max_seq_len", 50)

    # ── sub-split ────────────────────────────────────────────────────────

    @staticmethod
    def split_for_sasrec(
        train_sequences: Dict[int, List[int]],
    ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        """Split the leave-one-out *train* set into sasrec_train / sasrec_val.

        For each user the **last item** of the train sequence becomes the
        SASRec validation target; everything before it is used for training.

        Users with fewer than 2 items in their train sequence are dropped.

        Returns
        -------
        sasrec_train : ``{uid: [item, …]}``
        sasrec_val   : ``{uid: target_item}``
        """
        sasrec_train: Dict[int, List[int]] = {}
        sasrec_val: Dict[int, int] = {}
        for uid, seq in train_sequences.items():
            if len(seq) < 2:
                continue
            sasrec_train[uid] = seq[:-1]
            sasrec_val[uid] = seq[-1]
        return sasrec_train, sasrec_val

    # ── main entry point ─────────────────────────────────────────────────

    def fit(
        self,
        train_sequences: Dict[int, List[int]],
        num_items: int,
    ) -> SASRec:
        """Train SASRec and return the best model (by NDCG@10 on sasrec_val).

        Parameters
        ----------
        train_sequences : the ``splits["train"]`` dict from preprocessing
            (mapping ``user_idx`` → item-id list).
        num_items : total item vocabulary size.

        Returns
        -------
        The ``SASRec`` model with the best validation weights loaded.
        """
        set_seed(self.cfg.get("seed", 42))

        # 1. Sub-split ────────────────────────────────────────────────────
        sasrec_train, sasrec_val = self.split_for_sasrec(train_sequences)
        logger.info(
            "SASRec sub-split: %d train users, %d val users",
            len(sasrec_train), len(sasrec_val),
        )

        # 2. Dataset / loader ─────────────────────────────────────────────
        train_ds = SASRecDataset(
            sasrec_train, num_items,
            max_len=self.max_len,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.get("batch_size", 128),
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            generator=torch.Generator().manual_seed(self.cfg.get("seed", 42)),
        )

        # 3. Model ────────────────────────────────────────────────────────
        model = SASRec(
            num_items=num_items,
            emb_dim=self.cfg.get("embedding_dim", 64),
            max_len=self.max_len,
            n_heads=self.cfg.get("num_heads", 2),
            n_blocks=self.cfg.get("num_blocks", 2),
            dropout=self.cfg.get("dropout_rate", 0.2),
        ).to(self.device)

        # 4. Optimiser ────────────────────────────────────────────────────
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.get("learning_rate", 1e-3),
            weight_decay=self.cfg.get("weight_decay", 0.0),
        )
        grad_clip = self.cfg.get("grad_clip", 5.0)

        # 5. Training loop ────────────────────────────────────────────────
        num_epochs = self.cfg.get("num_epochs", 200)
        patience = self.cfg.get("patience", 20)

        best_ndcg = -1.0
        best_state = None
        wait = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer, grad_clip)
            val_metrics = self._validate(
                model, sasrec_train, sasrec_val, num_items,
            )
            ndcg10 = val_metrics.get("NDCG@10", 0.0)

            logger.info(
                "Epoch %3d | train_loss %.4f | val_loss %.4f | "
                "HR@5 %.4f | HR@10 %.4f | NDCG@5 %.4f | NDCG@10 %.4f",
                epoch, train_loss, val_metrics.get("val_loss", 0.0),
                val_metrics.get("HR@5", 0.0), val_metrics.get("HR@10", 0.0),
                val_metrics.get("NDCG@5", 0.0), ndcg10,
            )

            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # 6. Restore best weights ─────────────────────────────────────────
        if best_state is not None:
            model.load_state_dict(best_state)
        logger.info("Best val NDCG@10: %.4f", best_ndcg)
        return model

    # ── one training epoch ───────────────────────────────────────────────

    def _train_epoch(
        self,
        model: SASRec,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
    ) -> float:
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            seq = batch["seq"].to(self.device)   # (B, L)
            pos = batch["pos"].to(self.device)   # (B, L)
            neg = batch["neg"].to(self.device)   # (B, L)

            hidden = model(seq)                  # (B, L, D)
            mask = pos != 0                      # non-pad positions

            loss = sampled_softmax_loss(hidden, pos, neg, model.item_emb, mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ── validation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(
        self,
        model: SASRec,
        sasrec_train: Dict[int, List[int]],
        sasrec_val: Dict[int, int],
        num_items: int,
        num_neg_val: int = 100,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Rank each user's held-out item against ``num_neg_val`` negatives."""
        model.eval()
        rng = random.Random(123)  # fixed for consistent validation
        uids = sorted(sasrec_val.keys())
        all_scores: list = []
        total_loss = 0.0
        n_users = 0

        for start in range(0, len(uids), batch_size):
            batch_uids = uids[start : start + batch_size]

            # Prepare right-padded sequences
            seqs: list[list[int]] = []
            seq_lens: list[int] = []
            for uid in batch_uids:
                s = sasrec_train[uid][-self.max_len :]
                seq_lens.append(len(s))
                seqs.append(s + [0] * (self.max_len - len(s)))

            seq_t = torch.tensor(seqs, dtype=torch.long, device=self.device)
            hidden = model(seq_t)  # (B, L, D)

            for i, uid in enumerate(batch_uids):
                user_emb = hidden[i, seq_lens[i] - 1]  # (D,)

                target = sasrec_val[uid]
                user_items = set(sasrec_train[uid]) | {target}

                # Sample negatives
                negs: list[int] = []
                while len(negs) < num_neg_val:
                    c = rng.randint(1, num_items)
                    if c not in user_items:
                        negs.append(c)

                # Score: positive first, then negatives
                cand = torch.tensor(
                    [target] + negs, dtype=torch.long, device=self.device,
                )
                scores = model.score_candidates(user_emb, cand)

                # Accumulate val loss (CE over positive vs negatives)
                loss = F.cross_entropy(
                    scores.unsqueeze(0),
                    torch.zeros(1, dtype=torch.long, device=self.device),
                )
                total_loss += loss.item()
                n_users += 1

                all_scores.append(scores.cpu().numpy())

        metrics = evaluate_ranking(all_scores, ks=(5, 10), positive_idx=0)
        metrics["val_loss"] = total_loss / max(n_users, 1)
        return metrics
