"""Plotly figures for the Signal2Space app. Model colours match the project's result figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core import AP, MISSING, MODEL_COLORS, UNIT

INK, MUTED, GRID = "#243A50", "#63788A", "#E3E8EC"
BUILDING_TINTS = ["#C9D6E3", "#D9D2C3", "#C7DBD1"]
LAYOUT = dict(template="plotly_white", font=dict(family="Inter, Arial, sans-serif", color=INK, size=13),
              margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15))


def _style(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(**LAYOUT, height=height, title=dict(text=title, x=0, font=dict(size=15)) if title else None)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def radio_map(background: pd.DataFrame, points: dict[str, tuple[float, float]], building: int | None = None,
              title: str = "Where the scan was taken, and where each model places it") -> go.Figure:
    """Training positions as the floor plan, plus the true and predicted positions of one scan."""
    fig = go.Figure()
    bg = background if building is None else background[background.BUILDINGID == building]
    for b, sub in bg.groupby("BUILDINGID"):
        fig.add_trace(go.Scattergl(x=sub.LONGITUDE, y=sub.LATITUDE, mode="markers", name=f"Building {b} survey points",
                                   marker=dict(size=5, color=BUILDING_TINTS[int(b) % 3]), hoverinfo="skip"))
    true = points.get("True position")
    for name, (x, y) in points.items():
        if name == "True position":
            continue
        if true is not None:
            fig.add_trace(go.Scatter(x=[true[0], x], y=[true[1], y], mode="lines", showlegend=False,
                                     line=dict(color=MODEL_COLORS[name], width=2, dash="dot"), hoverinfo="skip"))
        err = float(np.hypot(x - true[0], y - true[1])) if true is not None else None
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=name,
                                 marker=dict(size=15, color=MODEL_COLORS[name], line=dict(color="white", width=2)),
                                 hovertemplate=f"{name}<br>x=%{{x:.1f}}<br>y=%{{y:.1f}}"
                                               + (f"<br>error {err:.1f} {UNIT}" if err is not None else "") + "<extra></extra>"))
    if true is not None:
        fig.add_trace(go.Scatter(x=[true[0]], y=[true[1]], mode="markers", name="True position",
                                 marker=dict(size=20, symbol="star", color=MODEL_COLORS["True position"],
                                             line=dict(color="white", width=1.5)),
                                 hovertemplate="True position<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title=f"LATITUDE ({UNIT})", tickformat=".0f")
    fig.update_xaxes(title=f"LONGITUDE ({UNIT})", tickformat=".0f")
    return _style(fig, title, height=520)


def fingerprint(raw_original: np.ndarray, raw_used: np.ndarray) -> go.Figure:
    """Detected access points and their RSSI; hidden or dropped ones in grey."""
    det = np.flatnonzero(raw_original != MISSING)
    if det.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="No access point was detected in this scan.", showarrow=False, font=dict(size=14, color=MUTED))
        return _style(fig, "Wi-Fi fingerprint", height=300)
    used = raw_used[det] != MISSING
    order = np.argsort(raw_original[det])[::-1]
    det, used = det[order], used[order]
    labels = [AP[i] for i in det]
    vals = np.where(used, raw_used[det], raw_original[det])
    colors = ["#2A78D6" if u else "#C3CBD3" for u in used]
    fig = go.Figure(go.Bar(x=labels, y=vals + 110, base=-110, marker_color=colors,
                           customdata=np.stack([vals, np.where(used, "used", "removed")], axis=1),
                           hovertemplate="%{x}<br>%{customdata[0]:.0f} dBm (%{customdata[1]})<extra></extra>"))
    fig.update_yaxes(title="RSSI (dBm, higher is stronger)", range=[-110, 0])
    fig.update_xaxes(title=f"{det.size} detected access points, strongest first", showticklabels=det.size <= 40)
    return _style(fig, "Wi-Fi fingerprint", height=320)


def probabilities(labels: list, proba: np.ndarray, true_label, title: str) -> go.Figure:
    colors = [MODEL_COLORS["DL multitask"] if p == proba.max() else "#B8C7D6" for p in proba]
    fig = go.Figure(go.Bar(x=proba * 100, y=[f"{title.split()[0]} {l}" + ("  ← true" if l == true_label else "") for l in labels],
                           orientation="h", marker_color=colors, text=[f"{p:.1%}" for p in proba], textposition="auto",
                           insidetextfont=dict(color="white"), cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(range=[0, 130], tickvals=[0, 25, 50, 75, 100], title="probability (%)")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, title, height=90 + 42 * len(labels))


def metric_bars(comp: pd.DataFrame, metrics: list[tuple[str, str]], title: str, pct: bool) -> go.Figure:
    fig = go.Figure()
    color = {"AML Random Forest": MODEL_COLORS["AML Random Forest"], "DL multitask": MODEL_COLORS["DL multitask"],
             "DL single-task (separate nets)": MODEL_COLORS["DL coordinate network"]}
    for _, row in comp.iterrows():
        vals = [row[m] * (100 if pct else 1) for m, _ in metrics]
        fig.add_trace(go.Bar(name=row.model, x=[l for _, l in metrics], y=vals, marker_color=color.get(row.model, MUTED),
                             text=[("" if pd.isna(v) else (f"{v:.1f}" if pct else f"{v:.2f}")) for v in vals], textposition="outside",
                             textfont=dict(size=11)))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, uniformtext=dict(minsize=9, mode="show"))
    if pct:
        fig.update_yaxes(range=[80, 102], title="%")
    else:
        fig.update_yaxes(title=UNIT)
    return _style(fig, title)


def confusion(true: pd.Series, pred: pd.Series, labels: list, title: str, colorscale: str) -> go.Figure:
    cm = pd.crosstab(pd.Categorical(true, labels), pd.Categorical(pred, labels), dropna=False).to_numpy()
    fig = go.Figure(go.Heatmap(z=cm, x=[str(l) for l in labels], y=[str(l) for l in labels], colorscale=colorscale,
                               showscale=False, text=cm, texttemplate="%{text}", hovertemplate="true %{y} → predicted %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="true floor")
    fig.update_xaxes(title="predicted floor", side="bottom")
    return _style(fig, title, height=400)


def error_map(df: pd.DataFrame, title: str) -> go.Figure:
    cap = float(np.percentile(df.EUCLIDEAN_ERROR, 95))
    fig = go.Figure(go.Scattergl(x=df.LONGITUDE_TRUE, y=df.LATITUDE_TRUE, mode="markers",
                                 marker=dict(size=7, color=df.EUCLIDEAN_ERROR.clip(upper=cap), colorscale="Blues", cmin=0, cmax=cap,
                                             colorbar=dict(title=f"error<br>({UNIT})"), line=dict(width=0)),
                                 customdata=np.stack([df.EUCLIDEAN_ERROR, df.ROW_ID], axis=1),
                                 hovertemplate="scan %{customdata[1]}<br>error %{customdata[0]:.1f}<extra></extra>"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, tickformat=".0f", title=f"LATITUDE ({UNIT})")
    fig.update_xaxes(tickformat=".0f", title=f"LONGITUDE ({UNIT})")
    return _style(fig, title, height=480)


def error_cdf(errors: dict[str, np.ndarray], title: str) -> go.Figure:
    color = {"AML Random Forest": MODEL_COLORS["AML Random Forest"], "DL multitask": MODEL_COLORS["DL multitask"],
             "DL single-task": MODEL_COLORS["DL coordinate network"]}
    fig = go.Figure()
    for name, e in errors.items():
        s = np.sort(e)
        fig.add_trace(go.Scatter(x=s, y=np.arange(1, len(s) + 1) / len(s) * 100, mode="lines", name=name,
                                 line=dict(color=color[name], width=2.5),
                                 hovertemplate=name + ": %{y:.0f}% of scans within %{x:.1f}<extra></extra>"))
    fig.update_xaxes(title=f"Euclidean error ({UNIT})", range=[0, float(max(np.percentile(e, 98) for e in errors.values()))])
    fig.update_yaxes(title="% of held-out scans", range=[0, 100])
    return _style(fig, title)


def training_history(hist: pd.DataFrame) -> go.Figure:
    """Two panels (losses, validation score) instead of a dual axis."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Training loss terms", "Validation selection score (lower is better)"))
    for col, name, color in [("train_loss_building", "building loss", "#63788A"), ("train_loss_floor", "floor loss", "#EB6834"),
                             ("train_loss_coord", "coordinate loss", "#1BAF7A")]:
        fig.add_trace(go.Scatter(x=hist.epoch, y=hist[col], name=name, line=dict(color=color, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.epoch, y=hist.val_selection_score, name="validation score",
                             line=dict(color=MODEL_COLORS["DL multitask"], width=2.5)), row=1, col=2)
    best = hist.loc[hist.val_selection_score.idxmin()]
    fig.add_trace(go.Scatter(x=[best.epoch], y=[best.val_selection_score], mode="markers", name="restored best epoch",
                             marker=dict(size=11, color=MODEL_COLORS["DL multitask"], line=dict(color="white", width=2))), row=1, col=2)
    fig.add_annotation(x=best.epoch, y=best.val_selection_score, xref="x2", yref="y2", text=f"best epoch {int(best.epoch)}",
                       showarrow=True, arrowhead=0, ax=0, ay=-60, font=dict(color=INK, size=12), bgcolor="rgba(255,255,255,0.85)")
    fig.update_xaxes(title="epoch")
    return _style(fig, None, height=400)
