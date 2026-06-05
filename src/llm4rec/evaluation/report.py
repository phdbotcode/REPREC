"""Reporting utilities: persist metrics, build summary tables.

File formats
------------
* ``results.jsonl`` — one JSON object per line, each recording one
  experimental run (method, params, FLOPs, HR@10, NDCG@10, …).
* ``compute_table.csv`` — the same data as a CSV for easy import into
  LaTeX / pandas / plotting scripts.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Union

from llm4rec.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Save / load individual metric dicts
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics(
    metrics: Dict[str, Any],
    path: str,
) -> None:
    """Write a single metrics dict as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved → %s", path)


def load_metrics(path: str) -> Dict[str, Any]:
    """Load a JSON metrics file."""
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# JSONL run summaries
# ─────────────────────────────────────────────────────────────────────────────

def append_run_summary(
    path: str,
    summary: Dict[str, Any],
) -> None:
    """Append one run summary to a JSONL file.

    Each call adds a single line.  The file is created if it does not exist.

    Parameters
    ----------
    path : path to ``results.jsonl``.
    summary : flat dict produced by
        :func:`~llm4rec.evaluation.compute.build_compute_record` (or any
        dict with string keys and JSON-serialisable values).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    logger.info("Run summary appended → %s", path)


def load_all_summaries(
    path: str,
) -> List[Dict[str, Any]]:
    """Load every run summary from a JSONL file.

    Returns
    -------
    list of dicts — one per line / experimental run.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# CSV table
# ─────────────────────────────────────────────────────────────────────────────

def save_compute_table(
    records: List[Dict[str, Any]],
    path: str,
    columns: Optional[List[str]] = None,
) -> None:
    """Write a list of run records as a CSV table.

    Parameters
    ----------
    records : list of flat dicts (e.g. from :func:`load_all_summaries`).
    path : output ``.csv`` path.
    columns : explicit column order.  If ``None``, columns are derived
        from the union of all record keys (sorted).
    """
    if not records:
        logger.warning("No records to write")
        return

    if columns is None:
        # Deterministic column order: method first, then sorted
        all_keys = set()
        for r in records:
            all_keys.update(r.keys())
        all_keys.discard("method")
        columns = ["method"] + sorted(all_keys)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    logger.info("Compute table saved → %s  (%d rows)", path, len(records))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: JSONL → CSV in one call
# ─────────────────────────────────────────────────────────────────────────────

def export_results(
    results_dir: str,
    jsonl_name: str = "results.jsonl",
    csv_name: str = "compute_table.csv",
) -> List[Dict[str, Any]]:
    """Load all summaries from JSONL and export as CSV.

    Parameters
    ----------
    results_dir : directory containing the JSONL file.
    jsonl_name : name of the JSONL file.
    csv_name : name of the CSV output.

    Returns
    -------
    The loaded records (list of dicts).
    """
    jsonl_path = os.path.join(results_dir, jsonl_name)
    csv_path = os.path.join(results_dir, csv_name)

    records = load_all_summaries(jsonl_path)
    save_compute_table(records, csv_path)
    return records
