#!/usr/bin/env python
"""01 — Download, preprocess, split, and generate fixed negative pools.

Outputs (per dataset):
    outputs/data/{dataset}/mappings.json
    outputs/data/{dataset}/splits.json
    outputs/data/{dataset}/sequences.parquet
    outputs/data/{dataset}/item_meta.json
    outputs/data/{dataset}/negatives_200.jsonl   (val + test users)
    outputs/data/{dataset}/test_labels.jsonl     (rating + review text per test user)
    outputs/data/{dataset}/history_labels.json   (rating + review summary per train interaction)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm4rec.data.amazon_download import download_amazon_data
from llm4rec.data.negatives import sample_negatives, save_negatives
from llm4rec.data.preprocess import preprocess_category
from llm4rec.utils.io import load_config
from llm4rec.utils.logging import get_logger
from llm4rec.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Amazon data")
    parser.add_argument("--config", default="../configs/dataset_amazon.yaml")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    cfg = load_config(args.config)["dataset"]
    set_seed(args.seed)

    categories = cfg["categories"]
    raw_dir = cfg.get("raw_dir", "outputs/data/raw")
    processed_base = cfg.get("processed_dir", "outputs/data")
    min_user = cfg["preprocessing"]["min_user_interactions"]
    min_item = cfg["preprocessing"]["min_item_interactions"]
    max_seq = cfg["sequences"]["max_length"]
    num_neg = 200
    neg_seed = args.seed

    # ── 1. Download ──────────────────────────────────────────────────────
    logger.info("Downloading categories: %s", categories)
    raw = download_amazon_data(
        data_dir=raw_dir,
        categories=categories,
        download_meta=True,
    )

    # ── 2. Preprocess each category ──────────────────────────────────────
    for cat in categories:
        if cat not in raw["reviews"]:
            logger.warning("No reviews for %s, skipping", cat)
            continue

        out_dir = str(Path(processed_base) / cat.lower())
        logger.info("Preprocessing %s → %s", cat, out_dir)

        result = preprocess_category(
            reviews=raw["reviews"][cat],
            metadata=raw["meta"].get(cat),
            min_user=min_user,
            min_item=min_item,
            max_seq_len=max_seq,
            output_dir=out_dir,
        )

        # ── 3. Fixed negatives for BOTH val and test users ───────────
        splits = result["splits"]
        sequences = result["sequences"]
        num_items = result["num_items"]

        # Merge val and test users into one negative pool
        all_eval_users = set(splits["valid"].keys()) | set(splits["test"].keys())

        # Build a combined "targets" dict so sample_negatives iterates all
        combined_splits = {
            "train": splits["train"],
            "valid": splits["valid"],
            "test": {uid: splits["test"].get(uid, splits["valid"].get(uid))
                     for uid in all_eval_users},
        }

        negatives = sample_negatives(
            splits=combined_splits,
            sequences=sequences,
            num_items=num_items,
            num_negatives=num_neg,
            seed=neg_seed,
        )

        neg_path = str(Path(out_dir) / "negatives_200.jsonl")
        save_negatives(negatives, neg_path)

        # ── 4. Test labels: rating + review text per test user ────────
        # Build (user_id, item_id) → {rating, text, summary} from raw reviews.
        # Only the test interaction is saved — used for Rating and ReviewGen
        # transfer-task evaluation in 07_transfer_inference.py.
        user2idx = result["user2idx"]
        item2idx = result["item2idx"]
        idx2user = {v: k for k, v in user2idx.items()}
        idx2item = {v: k for k, v in item2idx.items()}

        # Build lookup: (raw_user_id, raw_item_id) → review fields
        review_lookup: dict = {}
        for rev in raw["reviews"][cat]:
            key = (rev["user_id"], rev["item_id"])
            review_lookup[key] = {
                "rating": rev.get("rating", None),
                "review_text": rev.get("text", ""),
                "review_summary": rev.get("summary", ""),
            }

        label_path = str(Path(out_dir) / "test_labels.jsonl")
        with open(label_path, "w") as f:
            for uid, test_iid in splits["test"].items():
                raw_uid = idx2user.get(uid, "")
                raw_iid = idx2item.get(test_iid, "")
                label = review_lookup.get((raw_uid, raw_iid), {})
                record = {
                    "user_idx": uid,
                    "item_idx": test_iid,
                    "rating": label.get("rating", None),
                    "review_text": label.get("review_text", ""),
                    "review_summary": label.get("review_summary", ""),
                }
                f.write(json.dumps(record) + "\n")
        logger.info("Test labels saved → %s  (%d users)", label_path, len(splits["test"]))

        # ── 5. History labels: rating + review summary per train interaction ──
        # Saves a lookup used by 07_transfer_inference.py to enrich history
        # prompts with per-interaction ratings and review summaries.
        history_labels: dict = {}
        for uid, train_seq in splits["train"].items():
            raw_uid = idx2user.get(uid, "")
            uid_labels: dict = {}
            for iid in train_seq:
                raw_iid = idx2item.get(iid, "")
                label = review_lookup.get((raw_uid, raw_iid), {})
                rating  = label.get("rating")
                summary = label.get("review_summary", "")
                if rating is not None or summary:
                    uid_labels[str(iid)] = {"rating": rating, "summary": summary}
            if uid_labels:
                history_labels[str(uid)] = uid_labels

        hist_path = str(Path(out_dir) / "history_labels.json")
        with open(hist_path, "w") as f:
            json.dump(history_labels, f)
        logger.info("History labels saved → %s", hist_path)

        logger.info(
            "Done %s: %d users, %d items, %d neg pools",
            cat, result["num_users"], result["num_items"], len(negatives),
        )

    logger.info("All datasets prepared.")


if __name__ == "__main__":
    main()
