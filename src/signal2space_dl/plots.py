"""Figures for the DL phase. Every function writes a PNG and returns the figure.

Colour roles are fixed across all figures: DL = blue, AML = orange (validated
categorical slots 1 and 2), sequential error maps use a single-hue ramp.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .evaluate import HIERARCHY_ORDER  # noqa: E402

DL_COLOR = "#2a78d6"
AML_COLOR = "#eb6834"
EXTRA_COLOR = "#1baf7a"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
UNIT = C.COORDINATE_UNIT_LABEL

# Colour follows the model, never its position in a chart.
MODEL_COLORS = {
    "DL multitask": DL_COLOR,
    "AML Random Forest": AML_COLOR,
    "DL single-task (separate nets)": EXTRA_COLOR,
}


def model_color(name: str, fallback: str = "#6250d6") -> str:
    return MODEL_COLORS.get(name, fallback)


def _plain_axes(ax):
    """Show full coordinate values instead of a 1e6 offset."""
    ax.ticklabel_format(useOffset=False, style="plain")

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#0b0b0b", "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY, "axes.grid": True, "grid.color": GRID_COLOR, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def _save(fig, path: Path | None):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_training_history(history: pd.DataFrame, title: str, path: Path | None = None):
    """Train loss and internal-validation selection score per epoch for one run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    loss_cols = [c for c in history.columns if c.startswith("train_loss_") and c != "train_loss_total"]
    palette = [DL_COLOR, AML_COLOR, EXTRA_COLOR]
    for col, color in zip(loss_cols, palette):
        axes[0].plot(history.epoch, history[col], lw=2, color=color, label=col.replace("train_loss_", ""))
    axes[0].set(title="Training loss terms (unweighted)", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(history.epoch, history.val_selection_score, lw=2, color=DL_COLOR)
    best = history.loc[history.val_selection_score.idxmin()]
    axes[1].scatter([best.epoch], [best.val_selection_score], s=60, color=DL_COLOR, zorder=3,
                    edgecolor="white", linewidth=2)
    axes[1].annotate(f"best epoch {int(best.epoch)}", (best.epoch, best.val_selection_score),
                     xytext=(8, 10), textcoords="offset points", color=TEXT_SECONDARY)
    axes[1].set(title="Internal-validation selection score (lower is better)", xlabel="Epoch", ylabel="Score")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(cm: pd.DataFrame, title: str, ax=None, cmap="Blues"):
    """Draw one confusion matrix (rows actual, columns predicted) with counts."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    values = cm.to_numpy()
    ax.imshow(values, cmap=cmap)
    ax.grid(False)
    ax.set_xticks(range(values.shape[1]), cm.columns)
    ax.set_yticks(range(values.shape[0]), cm.index)
    thresh = values.max() / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, int(values[i, j]), ha="center", va="center",
                    color="white" if values[i, j] > thresh else "#0b0b0b", fontsize=9)
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    return ax.figure


def plot_confusion_pair(cm_a: pd.DataFrame, title_a: str, cm_b: pd.DataFrame, title_b: str,
                        suptitle: str, path: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion(cm_a, title_a, axes[0], cmap="Blues")
    plot_confusion(cm_b, title_b, axes[1], cmap="Oranges")
    fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_coordinate_error_map(pred: pd.DataFrame, title: str, path: Path | None = None, n_segments: int = 500):
    """Left: true positions coloured by Euclidean error. Right: actual->predicted displacement."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    err = pred.EUCLIDEAN_ERROR.to_numpy()
    sc = axes[0].scatter(pred.LONGITUDE_TRUE, pred.LATITUDE_TRUE, c=err, cmap="Blues", s=14,
                         vmin=0, vmax=np.percentile(err, 95), edgecolors="none")
    fig.colorbar(sc, ax=axes[0], shrink=0.85, label=f"Euclidean error ({UNIT}; capped at p95)")
    axes[0].set(title="Error at each true position", xlabel=f"LONGITUDE ({UNIT})", ylabel=f"LATITUDE ({UNIT})")
    axes[0].set_aspect("equal", adjustable="datalim")
    _plain_axes(axes[0])
    rng = np.random.default_rng(C.GLOBAL_SEED)
    idx = rng.choice(len(pred), size=min(n_segments, len(pred)), replace=False)
    sub = pred.iloc[idx]
    for _, r in sub.iterrows():
        axes[1].plot([r.LONGITUDE_TRUE, r.LONGITUDE_PRED], [r.LATITUDE_TRUE, r.LATITUDE_PRED],
                     color=TEXT_SECONDARY, alpha=0.35, lw=0.7)
    axes[1].scatter(sub.LONGITUDE_TRUE, sub.LATITUDE_TRUE, s=10, color=EXTRA_COLOR, label="Actual", zorder=3)
    axes[1].scatter(sub.LONGITUDE_PRED, sub.LATITUDE_PRED, s=10, color=DL_COLOR, label="Predicted", zorder=3)
    axes[1].set(title=f"Displacement, actual to predicted ({len(sub)} sampled rows)",
                xlabel=f"LONGITUDE ({UNIT})", ylabel=f"LATITUDE ({UNIT})")
    axes[1].set_aspect("equal", adjustable="datalim")
    _plain_axes(axes[1])
    axes[1].legend(frameon=False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_error_distribution(errors: dict[str, np.ndarray], title: str, path: Path | None = None):
    """Histogram and empirical CDF of Euclidean error for each named model."""
    colors = {name: model_color(name) for name in errors}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    upper = max(np.percentile(e, 99) for e in errors.values())
    bins = np.linspace(0, upper, 50)
    for name, e in errors.items():
        axes[0].hist(e, bins=bins, histtype="step", lw=2, color=colors[name], label=f"{name} (mean {e.mean():.2f})")
        s = np.sort(e)
        axes[1].plot(s, np.arange(1, len(s) + 1) / len(s), lw=2, color=colors[name], label=name)
    axes[0].set(title="Euclidean error histogram (to p99)", xlabel=f"Euclidean error ({UNIT})", ylabel="Rows")
    axes[1].set(title="Cumulative share of rows within error", xlabel=f"Euclidean error ({UNIT})",
                ylabel="Share of rows", xlim=(0, upper))
    for ax in axes:
        ax.legend(frameon=False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_hierarchical_comparison(tables: dict[str, pd.DataFrame], title: str, path: Path | None = None):
    """Grouped bars: share of rows and mean coordinate error per hierarchy outcome."""
    names = list(tables)
    colors = [model_color(n) for n in names]
    x = np.arange(len(HIERARCHY_ORDER))
    width = 0.8 / len(names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for k, (name, color) in enumerate(zip(names, colors)):
        t = tables[name].set_index("hierarchy_outcome").reindex(HIERARCHY_ORDER)
        offs = x - 0.4 + width * (k + 0.5)
        b = axes[0].bar(offs, t.n_rows, width * 0.92, color=color, label=name)
        axes[0].bar_label(b, labels=[f"{int(v)}" for v in t.n_rows], fontsize=8, padding=2, color=TEXT_SECONDARY)
        axes[1].bar(offs, t.mean_coordinate_error.fillna(0), width * 0.92, color=color, label=name)
    labels = ["both correct", "building ok,\nfloor wrong", "building wrong,\nfloor ok", "both wrong"]
    axes[0].set(title="Rows per building/floor outcome", ylabel="Rows", yscale="log")
    axes[1].set(title="Mean coordinate error per outcome", ylabel=f"Mean Euclidean error ({UNIT})")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.legend(frameon=False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_metric_comparison(table: pd.DataFrame, title: str, path: Path | None = None):
    """Two panels (different units, never a dual axis): accuracies (%) and coordinate errors."""
    models = list(table.model.unique())
    colors = [model_color(m) for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    panels = [(axes[0], ["building_accuracy", "floor_accuracy", "building_and_floor_accuracy", "floor_macro_f1"],
               ["Building acc.", "Floor acc.", "Building+floor", "Floor macro-F1"], 100, "%"),
              (axes[1], ["mae_longitude", "mae_latitude", "mean_euclidean_error", "median_euclidean_error"],
               ["MAE longitude", "MAE latitude", "Mean Euclidean", "Median Euclidean"], 1, UNIT)]
    for ax, metrics, labels, scale, unit in panels:
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        for k, (m, color) in enumerate(zip(models, colors)):
            row = table[table.model == m].iloc[0]
            vals = [row.get(c, np.nan) * scale for c in metrics]
            b = ax.bar(x - 0.4 + width * (k + 0.5), vals, width * 0.92, color=color, label=m)
            ax.bar_label(b, labels=[f"{v:.1f}" if np.isfinite(v) else "" for v in vals], fontsize=7, padding=2,
                         color=TEXT_SECONDARY)
        ax.set_xticks(x, labels)
        ax.set_ylabel(unit)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylim(70, 102)
    axes[0].set_title("Classification (higher is better)")
    axes[1].set_title("Coordinate error (lower is better)")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_robustness(table: pd.DataFrame, title: str, path: Path | None = None):
    """Per-group floor accuracy and mean Euclidean error (e.g. held-out PHONEID)."""
    t = table.sort_values("n_rows", ascending=False).reset_index(drop=True)
    labels = [f"{g}\n(n={n})" for g, n in zip(t.group, t.n_rows)]
    seen = t.get("group_seen_in_internal_train", pd.Series([True] * len(t)))
    colors = [DL_COLOR if s else EXTRA_COLOR for s in seen]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    axes[0].bar(range(len(t)), t.floor_accuracy * 100, color=colors)
    axes[0].set(title="Floor accuracy by group", ylabel="%", ylim=(0, 105))
    axes[1].bar(range(len(t)), t.mean_euclidean_error, color=colors)
    axes[1].set(title="Mean Euclidean error by group", ylabel=UNIT)
    for ax in axes:
        ax.set_xticks(range(len(t)), labels, fontsize=8)
        ax.set_xlabel(t.group_column.iloc[0])
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=DL_COLOR, label="seen in internal training"),
                            Patch(color=EXTRA_COLOR, label="not seen in internal training")], frameon=False,
                   loc="lower left", fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)
