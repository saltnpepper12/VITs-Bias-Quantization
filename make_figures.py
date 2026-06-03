"""
make_figures.py
---------------
Regenerate every chart in assets/ directly from the raw per-image prediction
CSVs in results/. No GPU, model weights, or dataset required.

    pip install pandas matplotlib numpy
    python make_figures.py

Outputs:
    assets/01_overall_accuracy.png
    assets/02_per_group_accuracy.png
    assets/03_quantization_impact.png
    assets/04_fairness_gap.png
    assets/05_confusion_matrices.png
    assets/metrics.json
"""
import os
import ast
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# FairFace integer-label ordering (index == the `race` int in the parquet).
# Used to decode the numeric ViT-base CSV; the other CSVs already store strings.
CLASSES = ["White", "Black", "Latino_Hispanic", "East Asian",
           "Southeast Asian", "Indian", "Middle Eastern"]
SHORT = {"White": "White", "Black": "Black", "Latino_Hispanic": "Latino",
         "East Asian": "E. Asian", "Southeast Asian": "SE Asian",
         "Indian": "Indian", "Middle Eastern": "Mid. East"}

MODELS = {
    "ViT FP32":  "vit_fp32_validation",
    "ViT INT8":  "vit_int8_validation",
    "Swin FP32": "swin_fp32_validation",
    "Swin INT8": "swin_int8_validation",
}
COLORS = {"ViT FP32": "#2563eb", "ViT INT8": "#93c5fd",
          "Swin FP32": "#16a34a", "Swin INT8": "#86efac"}

CSV_DIR = "results"
OUT_DIR = "assets"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.dpi": 140, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.facecolor": "white", "figure.facecolor": "white",
})


def _decode(v):
    return v if isinstance(v, str) else CLASSES[int(v)]


def load(name):
    df = pd.read_csv(os.path.join(CSV_DIR, f"{name}.csv"))
    df["true"] = df["true_label"].map(_decode)
    df["pred"] = df["top1_pred"].map(_decode)

    def top5(v):
        lst = ast.literal_eval(v) if isinstance(v, str) else v
        return [_decode(x) for x in lst]

    df["t5"] = df["top5_preds"].map(top5)
    return df


def compute_metrics(data):
    metrics = {}
    for name, df in data.items():
        overall = (df["true"] == df["pred"]).mean() * 100
        top5 = df.apply(lambda r: r["true"] in r["t5"], axis=1).mean() * 100
        per_group = {}
        for c in CLASSES:
            sub = df[df["true"] == c]
            per_group[c] = (sub["true"] == sub["pred"]).mean() * 100
        vals = np.array([per_group[c] for c in CLASSES])
        metrics[name] = {
            "overall": overall, "top5": top5, "per_group": per_group,
            "gap": vals.max() - vals.min(), "std": vals.std(),
            "min": vals.min(), "max": vals.max(),
        }
    return metrics


def chart_overall(metrics):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = list(MODELS)
    x = np.arange(len(names)); w = 0.38
    t1 = [metrics[n]["overall"] for n in names]
    t5 = [metrics[n]["top5"] for n in names]
    b1 = ax.bar(x - w / 2, t1, w, label="Top-1 accuracy",
                color=[COLORS[n] for n in names], edgecolor="black", linewidth=.6)
    b2 = ax.bar(x + w / 2, t5, w, label="Top-5 accuracy",
                color=[COLORS[n] for n in names], edgecolor="black",
                linewidth=.6, alpha=.45, hatch="//")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Overall Accuracy — FP32 vs INT8 (FairFace race, 10,954 val images)",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(os.path.join(OUT_DIR, "01_overall_accuracy.png")); plt.close(fig)


def chart_per_group(metrics):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    names = list(MODELS)
    x = np.arange(len(CLASSES)); w = 0.2
    for i, n in enumerate(names):
        vals = [metrics[n]["per_group"][c] for c in CLASSES]
        ax.bar(x + (i - 1.5) * w, vals, w, label=n,
               color=COLORS[n], edgecolor="black", linewidth=.5)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CLASSES])
    ax.set_ylabel("Top-1 accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Per-Demographic Accuracy — Where Bias Lives",
                 fontweight="bold", fontsize=13)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False)
    fig.savefig(os.path.join(OUT_DIR, "02_per_group_accuracy.png")); plt.close(fig)


def chart_impact(metrics):
    fig, ax = plt.subplots(figsize=(12, 5))
    vit = [metrics["ViT INT8"]["per_group"][c] - metrics["ViT FP32"]["per_group"][c] for c in CLASSES]
    swin = [metrics["Swin INT8"]["per_group"][c] - metrics["Swin FP32"]["per_group"][c] for c in CLASSES]
    x = np.arange(len(CLASSES)); w = 0.36
    ax.bar(x - w / 2, vit, w, label="ViT  (INT8 − FP32)", color="#2563eb", edgecolor="black", linewidth=.5)
    ax.bar(x + w / 2, swin, w, label="Swin (INT8 − FP32)", color="#16a34a", edgecolor="black", linewidth=.5)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CLASSES])
    ax.set_ylabel("Accuracy change after INT8 (pp)")
    ax.set_title("Quantization Impact per Demographic — Negative = Accuracy Lost",
                 fontweight="bold", fontsize=13)
    ax.legend(frameon=False, loc="lower right")
    for i, (v, s) in enumerate(zip(vit, swin)):
        ax.text(i - w / 2, v + (0.15 if v >= 0 else -0.15), f"{v:+.1f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.text(i + w / 2, s + (0.15 if s >= 0 else -0.15), f"{s:+.1f}",
                ha="center", va="bottom" if s >= 0 else "top", fontsize=8)
    fig.savefig(os.path.join(OUT_DIR, "03_quantization_impact.png")); plt.close(fig)


def chart_fairness_gap(metrics):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = list(MODELS)
    gaps = [metrics[n]["gap"] for n in names]
    stds = [metrics[n]["std"] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, gaps, 0.55, color=[COLORS[n] for n in names], edgecolor="black", linewidth=.6)
    for b, g, s in zip(bars, gaps, stds):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                f"{g:.1f} pp\nσ={s:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy gap: best − worst group (pp)")
    ax.set_ylim(0, max(gaps) * 1.25)
    ax.set_title("Fairness Gap — Spread Between Best & Worst Demographic",
                 fontweight="bold", fontsize=12)
    fig.savefig(os.path.join(OUT_DIR, "04_fairness_gap.png")); plt.close(fig)


def chart_confusion(data, metrics):
    names = list(MODELS)
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    idx = {c: i for i, c in enumerate(CLASSES)}
    for ax, n in zip(axes.ravel(), names):
        df = data[n]
        cm = np.zeros((7, 7))
        for t, p in zip(df["true"], df["pred"]):
            cm[idx[t], idx[p]] += 1
        cmn = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels([SHORT[c] for c in CLASSES], rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels([SHORT[c] for c in CLASSES], fontsize=9)
        ax.set_title(f"{n}  ·  {metrics[n]['overall']:.1f}% top-1", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False)
        for i in range(7):
            for j in range(7):
                v = cmn[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > 55 else "black", fontsize=8)
    fig.suptitle("Confusion Matrices (row-normalized %) — Diagonal = Correct",
                 fontweight="bold", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_confusion_matrices.png")); plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {name: load(csv) for name, csv in MODELS.items()}
    metrics = compute_metrics(data)

    summary = {
        n: {k: (round(v, 2) if not isinstance(v, dict)
                else {kk: round(vv, 2) for kk, vv in v.items()})
            for k, v in m.items()}
        for n, m in metrics.items()
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    chart_overall(metrics)
    chart_per_group(metrics)
    chart_impact(metrics)
    chart_fairness_gap(metrics)
    chart_confusion(data, metrics)

    print("Wrote figures + metrics.json to", OUT_DIR + "/")
    for n, m in metrics.items():
        print(f"  {n:10s} top1={m['overall']:.2f}%  gap={m['gap']:.1f}pp")


if __name__ == "__main__":
    main()
