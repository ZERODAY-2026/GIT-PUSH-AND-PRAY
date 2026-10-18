"""
visualize.py
============
Generates publication-quality charts from CUAD evaluation results.

Plots:
1. recall_at_k.png: Recall @ 1, 3, 5, 10 curve.
2. category_retrieval.png: Category-wise retrieval recall bars.
3. category_llm.png: Category-wise LLM F1 and Groundedness comparison.
4. end_to_end_performance.png: End-to-end accuracy and retrieval recall by category.
5. failure_distribution.png: Distribution of error failure stages.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.plots")

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_recall_at_k(retrieval_data: Dict[str, Any]):
    """Plot Recall@K curve."""
    k_vals = [1, 3, 5, 10]
    recalls = [
        retrieval_data.get("recall_at_1", 0.0) * 100,
        retrieval_data.get("recall_at_3", 0.0) * 100,
        retrieval_data.get("recall_at_5", 0.0) * 100,
        retrieval_data.get("recall_at_10", 0.0) * 100,
    ]

    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.plot(k_vals, recalls, marker="o", color="#2563eb", linewidth=2.5, markersize=8)
    for k, r in zip(k_vals, recalls):
        plt.annotate(
            f"{r:.1f}%",
            (k, r),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            color="#1e3a8a",
        )

    plt.title("CUAD Retrieval Performance: Recall @ K", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Top-K Retrieved Chunks", fontsize=11)
    plt.ylabel("Recall (%)", fontsize=11)
    plt.xticks(k_vals)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out = PLOTS_DIR / "recall_at_k.png"
    plt.savefig(out)
    plt.close()
    logger.info(f"Saved: {out.name}")


def plot_category_retrieval(retrieval_data: Dict[str, Any]):
    """Plot category-wise retrieval recall."""
    breakdown = retrieval_data.get("category_breakdown", {})
    if not breakdown:
        return

    # Filter categories with positive examples, take top 12 for clean readability
    sorted_cats = sorted(
        [c for c in breakdown.items() if c[1].get("positive_examples", 0) > 0],
        key=lambda x: x[1].get("recall_at_5", 0.0),
        reverse=True,
    )[:12]

    if not sorted_cats:
        sorted_cats = sorted(breakdown.items(), key=lambda x: x[0])[:12]

    categories = [c[0][:25] for c in sorted_cats]
    r5_vals = [c[1].get("recall_at_5", 0.0) * 100 for c in sorted_cats]

    plt.figure(figsize=(9, 5), dpi=150)
    bars = plt.barh(categories[::-1], r5_vals[::-1], color="#0d9488", height=0.6)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.title("Category-wise Retrieval Recall@5", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Recall @ 5 (%)", fontsize=11)
    plt.xlim(0, 115)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    out = PLOTS_DIR / "category_retrieval.png"
    plt.savefig(out)
    plt.close()
    logger.info(f"Saved: {out.name}")


def plot_category_llm(llm_data: Dict[str, Any]):
    """Plot LLM F1 and Groundedness comparison."""
    breakdown = llm_data.get("category_breakdown", {})
    if not breakdown:
        return

    sorted_cats = sorted(breakdown.items(), key=lambda x: x[1].get("avg_f1", 0.0), reverse=True)[:10]
    cats = [c[0][:20] for c in sorted_cats]
    f1_vals = [c[1].get("avg_f1", 0.0) * 100 for c in sorted_cats]
    grounded_vals = [c[1].get("avg_groundedness", 0.0) * 100 for c in sorted_cats]

    import numpy as np
    x = np.arange(len(cats))
    width = 0.35

    plt.figure(figsize=(10, 5), dpi=150)
    plt.bar(x - width / 2, f1_vals, width, label="Token F1", color="#3b82f6")
    plt.bar(x + width / 2, grounded_vals, width, label="Groundedness", color="#10b981")

    plt.title("Gemma Comprehension: Token F1 & Groundedness by Category", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(x, cats, rotation=35, ha="right", fontsize=9)
    plt.ylabel("Score (%)", fontsize=11)
    plt.ylim(0, 110)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    out = PLOTS_DIR / "category_llm.png"
    plt.savefig(out)
    plt.close()
    logger.info(f"Saved: {out.name}")


def plot_failure_distribution(e2e_data: Dict[str, Any]):
    """Plot failure distribution pie/bar chart."""
    dist = e2e_data.get("failure_stage_distribution", {})
    if not dist:
        return

    # Filter non-zero failures or show all
    labels = [k for k in dist.keys()]
    counts = [dist[k] for k in labels]

    plt.figure(figsize=(7, 4.5), dpi=150)
    colors = ["#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#64748b", "#94a3b8"]
    bars = plt.barh(labels[::-1], counts[::-1], color=colors[:len(labels)], height=0.55)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", fontsize=9, fontweight="bold")

    plt.title("End-to-End Failure-Stage Distribution", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Count of Occurrences", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    out = PLOTS_DIR / "failure_distribution.png"
    plt.savefig(out)
    plt.close()
    logger.info(f"Saved: {out.name}")


def generate_all_plots():
    """Read available reports and generate all plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    # Stage 1 Plots
    ret_rep = RESULTS_DIR / "retrieval" / "retrieval_report.json"
    if ret_rep.exists():
        with open(ret_rep, "r", encoding="utf-8") as f:
            ret_data = json.load(f)
        plot_recall_at_k(ret_data)
        plot_category_retrieval(ret_data)

    # Stage 2 Plots
    llm_rep = RESULTS_DIR / "llm" / "llm_report.json"
    if llm_rep.exists():
        with open(llm_rep, "r", encoding="utf-8") as f:
            llm_data = json.load(f)
        plot_category_llm(llm_data)

    # Stage 3 Plots
    e2e_rep = RESULTS_DIR / "end_to_end" / "end_to_end_report.json"
    if e2e_rep.exists():
        with open(e2e_rep, "r", encoding="utf-8") as f:
            e2e_data = json.load(f)
        plot_failure_distribution(e2e_data)


if __name__ == "__main__":
    generate_all_plots()
