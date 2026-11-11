"""Figures for the DL notebooks (written to data/figures/dl_*.png)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from . import config as C

BLUE, ORANGE, GREEN, GREY = "#2563eb", "#ea580c", "#16a34a", "#6b7280"


def _save(fig, paths: C.Paths, name: str):
    paths.ensure()
    fig.tight_layout()
    fig.savefig(paths.figures / name, dpi=150, bbox_inches="tight")
    return fig


def input_representation(data, paths: C.Paths):
    """Detected-WAP counts per split and the distribution of standardised RSSI inputs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"internal_train": BLUE, "internal_validation": GREEN, "heldout_validation": ORANGE}
    for s, X in data.X.items():
        seen = X[:, C.N_WAPS:]
        axes[0].hist(seen.sum(1), bins=np.arange(0, 60), alpha=0.45, density=True, label=s, color=colors[s])
        z = X[:, :C.N_WAPS][seen.astype(bool)]
        axes[1].hist(z, bins=80, alpha=0.45, density=True, label=s, color=colors[s])
    axes[0].set(title="Detected WAPs per fingerprint", xlabel="presence indicators = 1", ylabel="density")
    axes[1].set(title="Standardised RSSI of detected WAPs (train-fitted scaler)", xlabel="z", ylabel="density")
    for a in axes:
        a.legend(fontsize=8)
    return _save(fig, paths, "dl_01_input_representation.png")


def training_curves(history: pd.DataFrame, paths: C.Paths, prefix: str, title: str):
    cols = [c for c in history.columns if c.startswith("val_") and ("accuracy" in c or "error" in c)]
    fig, axes = plt.subplots(1, 1 + len(cols), figsize=(5 * (1 + len(cols)), 3.6))
    axes = np.atleast_1d(axes)
    axes[0].plot(history.epoch, history.train_loss, color=BLUE, label="train (dropout on)")
    axes[0].plot(history.epoch, history.val_loss, color=ORANGE, label="internal validation")
    best = history.loc[history.selected_epoch, "epoch"].iloc[0]
    for a in axes:
        a.axvline(best, color=GREY, ls="--", lw=1)
    axes[0].set(title=f"{title}: weighted loss", xlabel="epoch", ylabel="loss")
    axes[0].legend(fontsize=8)
    for a, c in zip(axes[1:], cols):
        y = history[c] * (10.0 if "error_scaled" in c else 1.0)  # scaled error -> dataset units
        a.plot(history.epoch, y, color=GREEN)
        a.set(title=c.replace("val_", "internal val ").replace("_scaled", " (units)"), xlabel="epoch")
    return _save(fig, paths, f"{prefix}_training_curves.png")


def confusion(pred: pd.DataFrame, task: str, paths: C.Paths, name: str, title: str, ax=None):
    col = "BUILDINGID" if task == "building" else "FLOOR"
    labels = sorted(set(pred[f"{col}_TRUE"]) | set(pred[f"{col}_PRED"]))
    cm = confusion_matrix(pred[f"{col}_TRUE"], pred[f"{col}_PRED"], labels=labels)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="predicted", ylabel="true", title=title)
    if own:
        return _save(fig, paths, name)
    return ax


def coordinate_errors(pred: pd.DataFrame, paths: C.Paths, prefix: str, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    e = pred.EUCLIDEAN_ERROR
    axes[0].hist(e, bins=60, color=BLUE)
    axes[0].axvline(e.mean(), color=ORANGE, label=f"mean {e.mean():.2f}")
    axes[0].axvline(e.median(), color=GREEN, label=f"median {e.median():.2f}")
    axes[0].set(title=f"{title}: Euclidean error", xlabel="error (dataset units)", ylabel="rows")
    axes[0].legend()
    sc = axes[1].scatter(pred.LONGITUDE_TRUE, pred.LATITUDE_TRUE, c=e.clip(upper=np.quantile(e, 0.98)),
                         s=10, cmap="viridis")
    fig.colorbar(sc, ax=axes[1], label="error (clipped at p98)")
    axes[1].set(title="Spatial error at true position", xlabel="LONGITUDE", ylabel="LATITUDE")
    axes[1].set_aspect("equal", adjustable="datalim")
    stem = f"{prefix}_errors" if prefix.endswith("coordinate") else f"{prefix}_coordinate_errors"
    return _save(fig, paths, f"{stem}.png")


def error_cdf(curves: dict[str, np.ndarray], paths: C.Paths, name: str, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    palette = [BLUE, ORANGE, GREEN, GREY, "#9333ea"]
    for (label, e), col in zip(curves.items(), palette):
        s = np.sort(e)
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), label=f"{label} (mean {e.mean():.2f})", color=col)
    ax.set(xlim=(0, np.quantile(np.concatenate(list(curves.values())), 0.98)), xlabel="Euclidean error (dataset units)",
           ylabel="cumulative fraction of held-out rows", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, paths, name)


def model_comparison(table: pd.DataFrame, paths: C.Paths, name: str):
    """Bar charts of held-out building/floor accuracy and mean error across models."""
    held = table[table.split == "heldout_validation"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, task, metric, ylab in [(axes[0], "building", "accuracy", "accuracy"),
                                   (axes[1], "floor", "accuracy", "accuracy"),
                                   (axes[2], "coordinate", "mean_euclidean_error", "mean error (units)")]:
        sub = held[held.task == task]
        colors = [ORANGE if m.startswith("AML") else (BLUE if "multitask" in m else GREY) for m in sub.model]
        bars = ax.bar(sub.model.str.replace("DL ", "DL\n"), sub[metric], color=colors)
        ax.bar_label(bars, labels=[f"{v:.4f}" if metric == "accuracy" else f"{v:.2f}" for v in sub[metric]],
                     fontsize=8)
        ax.set(title=f"Held-out {task}", ylabel=ylab)
        if metric == "accuracy":
            ax.set_ylim(max(0.0, sub[metric].min() - 0.05), 1.0)
        ax.tick_params(axis="x", labelsize=8)
    return _save(fig, paths, name)
