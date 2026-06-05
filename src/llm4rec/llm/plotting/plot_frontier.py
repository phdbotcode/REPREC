"""Plot parameter-efficiency vs recommendation-quality frontiers.

Generates two figures per dataset:

1. **Trainable Parameters vs NDCG@10** (log-scale x-axis)
2. **Training FLOPs Proxy vs NDCG@10** (log-scale x-axis)

Each method (SASRec, Injector, LoRA, …) gets a distinct marker/colour
so the Pareto frontier is easy to read.

Data source
-----------
Reads ``results.jsonl`` (produced by :func:`~llm4rec.evaluation.report.append_run_summary`)
from ``outputs/results/``.  Each line must contain at least::

    {"method": str, "dataset": str, "trainable_params": int,
     "estimated_flops": int, "NDCG@10": float}
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe on headless servers)
import matplotlib.pyplot as plt

from llm4rec.evaluation.report import load_all_summaries
from llm4rec.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Style registry
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_STYLE: Dict[str, Dict[str, Any]] = {
    "sasrec":       {"marker": "s", "color": "#1f77b4", "label": "SASRec"},
    "injector":     {"marker": "^", "color": "#ff7f0e", "label": "Injector"},
    "lora":         {"marker": "o", "color": "#2ca02c", "label": "LoRA"},
    "lora+injector":{"marker": "D", "color": "#d62728", "label": "LoRA+Injector"},
}

_FALLBACK_MARKERS = ["v", "P", "X", "h", "*"]
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


def _style_for(method: str) -> Dict[str, Any]:
    """Return marker / colour / label for a method name."""
    key = method.lower().replace(" ", "").replace("_", "")
    for canon, style in _METHOD_STYLE.items():
        if canon.replace("_", "").replace("+", "") in key or key in canon.replace("_", "").replace("+", ""):
            return style
    # Fallback for unknown methods
    idx = hash(method) % len(_FALLBACK_MARKERS)
    return {
        "marker": _FALLBACK_MARKERS[idx],
        "color": _FALLBACK_COLORS[idx],
        "label": method,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_frontier(
    records: List[Dict[str, Any]],
    output_dir: str = "outputs/results",
    datasets: Optional[List[str]] = None,
) -> List[str]:
    """Generate parameter-efficiency frontier plots and save PNGs.

    Parameters
    ----------
    records : list of run-summary dicts (from ``results.jsonl``).
    output_dir : directory for saved figures.
    datasets : explicit list of dataset names to plot.  If ``None``,
        all datasets found in the records are plotted.

    Returns
    -------
    list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group records by dataset
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        ds = rec.get("dataset", "unknown")
        by_dataset[ds].append(rec)

    if datasets is not None:
        by_dataset = {k: v for k, v in by_dataset.items() if k in datasets}

    saved: List[str] = []

    for dataset, recs in sorted(by_dataset.items()):
        # Group by method
        by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in recs:
            by_method[r.get("method", "unknown")].append(r)

        # ── Figure 1: Trainable Params vs NDCG@10 ───────────────────
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for method, runs in sorted(by_method.items()):
            style = _style_for(method)
            xs = [r["trainable_params"] for r in runs if "NDCG@10" in r]
            ys = [r["NDCG@10"] for r in runs if "NDCG@10" in r]
            if not xs:
                continue
            ax1.scatter(
                xs, ys,
                marker=style["marker"], color=style["color"],
                s=100, label=style["label"], zorder=3, edgecolors="white",
                linewidths=0.6,
            )
        ax1.set_xscale("log")
        ax1.set_xlabel("Trainable Parameters", fontsize=12)
        ax1.set_ylabel("NDCG@10", fontsize=12)
        ax1.set_title(f"Parameter Efficiency — {dataset}", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()

        path1 = os.path.join(output_dir, f"{dataset}_params_vs_ndcg.png")
        fig1.savefig(path1, dpi=150)
        plt.close(fig1)
        saved.append(path1)
        logger.info("Saved %s", path1)

        # ── Figure 2: FLOPs Proxy vs NDCG@10 ────────────────────────
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for method, runs in sorted(by_method.items()):
            style = _style_for(method)
            xs = [r["estimated_flops"] for r in runs
                  if "NDCG@10" in r and "estimated_flops" in r]
            ys = [r["NDCG@10"] for r in runs
                  if "NDCG@10" in r and "estimated_flops" in r]
            if not xs:
                continue
            ax2.scatter(
                xs, ys,
                marker=style["marker"], color=style["color"],
                s=100, label=style["label"], zorder=3, edgecolors="white",
                linewidths=0.6,
            )
        ax2.set_xscale("log")
        ax2.set_xlabel("Training FLOPs (proxy)", fontsize=12)
        ax2.set_ylabel("NDCG@10", fontsize=12)
        ax2.set_title(f"Compute Efficiency — {dataset}", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()

        path2 = os.path.join(output_dir, f"{dataset}_flops_vs_ndcg.png")
        fig2.savefig(path2, dpi=150)
        plt.close(fig2)
        saved.append(path2)
        logger.info("Saved %s", path2)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot compute-vs-NDCG@10 frontier curves",
    )
    parser.add_argument(
        "--results_dir", default="outputs/results",
        help="Directory containing results.jsonl",
    )
    parser.add_argument(
        "--jsonl", default="results.jsonl",
        help="Name of the JSONL summary file",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Restrict to these dataset names (default: all)",
    )
    args = parser.parse_args()

    jsonl_path = os.path.join(args.results_dir, args.jsonl)
    records = load_all_summaries(jsonl_path)
    logger.info("Loaded %d run records from %s", len(records), jsonl_path)

    saved = plot_frontier(records, output_dir=args.results_dir, datasets=args.datasets)
    logger.info("Done — %d plots saved", len(saved))


if __name__ == "__main__":
    main()
