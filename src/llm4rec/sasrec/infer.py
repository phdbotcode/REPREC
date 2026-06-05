"""Extract user embeddings from a trained SASRec model.

After SASRec training is complete we feed each user's **full train
sequence** (from the leave-one-out split) through the model and take
the hidden state at the last non-padded position as the user
representation.  These embeddings are later consumed by the LLM
soft-prompt injector.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from llm4rec.sasrec.model import SASRec


@torch.no_grad()
def extract_user_embeddings(
    model: SASRec,
    user_sequences: Dict[int, List[int]],
    max_len: int = 50,
    batch_size: int = 256,
    device: str = "cuda",
) -> Dict[int, np.ndarray]:
    """Run trained SASRec in eval mode and collect user embeddings.

    Parameters
    ----------
    model : a trained ``SASRec`` instance.
    user_sequences : mapping ``user_idx`` → chronological item-id list.
        Typically ``splits["train"]`` from the leave-one-out preprocessor
        (the full train portion, including the item used for SASRec
        validation, since training is now finished).
    max_len : sequences longer than this are truncated to the most recent
        ``max_len`` items.
    batch_size : number of users per forward pass.
    device : torch device string.

    Returns
    -------
    dict mapping ``user_idx`` → ``np.ndarray`` of shape ``(emb_dim,)``.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    uids = sorted(user_sequences.keys())
    embeddings: Dict[int, np.ndarray] = {}

    for start in range(0, len(uids), batch_size):
        batch_uids = uids[start : start + batch_size]

        # Build right-padded tensor
        seqs: List[List[int]] = []
        for uid in batch_uids:
            s = user_sequences[uid][-max_len:]
            seqs.append(s + [0] * (max_len - len(s)))

        seq_t = torch.tensor(seqs, dtype=torch.long, device=dev)
        user_embs = model.get_last_hidden(seq_t)  # (B, D)
        user_embs_np = user_embs.cpu().numpy()

        for j, uid in enumerate(batch_uids):
            embeddings[uid] = user_embs_np[j]

    return embeddings


def save_user_embeddings(
    embeddings: Dict[int, np.ndarray],
    path: str,
) -> None:
    """Persist embeddings as a ``.npz`` archive.

    Keys are stringified user indices; values are 1-D float arrays.
    """
    np.savez(path, **{str(uid): emb for uid, emb in embeddings.items()})


def load_user_embeddings(path: str) -> Dict[int, np.ndarray]:
    """Load embeddings saved by :func:`save_user_embeddings`."""
    data = np.load(path)
    return {int(uid): data[uid] for uid in data.files}
