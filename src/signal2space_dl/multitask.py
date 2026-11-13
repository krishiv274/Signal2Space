"""DL_05: shared-trunk multitask model, integrated evaluation and AML comparison.

Selection uses internal validation only (``train.select_configuration``); the
configuration and weights are saved before the supplied held-out file is
loaded. The frozen configuration is then retrained with extra seeds purely to
report seed variability. AML predictions from ML_05 are joined on
``(SOURCE_PARTITION, ROW_ID)`` for a row-aligned comparison.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import evaluate as E
from . import plots as P
from .pipeline import TASK_PREFIX, Outputs, _context, _evaluate_frozen, _quick

MT_PREFIX = "dl_05_multitask"


def _wide(metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    """Pivot long summary rows to one row per model for one split."""
    rows = []
    for model, sub in metrics[metrics.split == split].groupby("model", sort=False):
        r = {"split": split, "model": model}
        for _, m in sub.iterrows():
            if m.task in ("building", "floor"):
                r[f"{m.task}_accuracy"] = m.accuracy
                r[f"{m.task}_macro_f1"] = m.macro_f1
            elif m.task == "building_and_floor":
                r["building_and_floor_accuracy"] = m.accuracy
            elif m.task == "coordinate":
                for k in ["mae_longitude", "mae_latitude", "mean_euclidean_error", "median_euclidean_error"]:
                    r[k] = m[k]
        rows.append(r)
    return pd.DataFrame(rows)


def run_multitask(out: Outputs = Outputs(), quick: bool = False, verbose: bool = True,
                  seeds: tuple[int, ...] = C.VARIABILITY_SEEDS) -> dict:
    from .train import save_model, select_configuration, train_model, write_json

    out.ensure()
    dev, prep, arrays, coord_norm = _context()
    grid = _quick(C.SELECTION_GRID) if quick else C.SELECTION_GRID
    best, table, histories = select_configuration(grid, arrays["internal_train"], arrays["internal_validation"],
                                                  prep, coord_norm, verbose)
    table.to_csv(out.tables / f"{MT_PREFIX}_selection.csv", index=False)
    histories.to_csv(out.tables / f"{MT_PREFIX}_training_history.csv", index=False)
    write_json({**best.config.to_dict(), "selection_coord_norm": coord_norm},
               out.models / f"{MT_PREFIX}_selected_config.json")
    save_model(best, prep, out.models / f"{MT_PREFIX}.pt")

    # ---- configuration frozen: now load the held-out partition ---------------
    heldout_frame = D.load_heldout_split()
    heldout = prep.transform(heldout_frame)
    frames = _evaluate_frozen(best.model, prep, {"internal_validation": arrays["internal_validation"],
                                                 "heldout_validation": heldout})
    rows, hier = [], []
    for split, f in frames.items():
        f.to_csv(out.tables / f"{MT_PREFIX}_{split}_predictions.csv", index=False)
        rows += E.summary_rows(f, split, "DL multitask")
        hier.append(E.hierarchical_breakdown(f, split))
    metrics = pd.DataFrame(rows)
    metrics.insert(2, "config", best.config.name)
    metrics.to_csv(out.tables / f"{MT_PREFIX}_metrics.csv", index=False)
    hier = pd.concat(hier, ignore_index=True)
    hier.to_csv(out.tables / f"{MT_PREFIX}_hierarchical_error_breakdown.csv", index=False)
    ho = frames["heldout_validation"]
    E.hierarchical_by_building_floor(ho, "heldout_validation").to_csv(
        out.tables / f"{MT_PREFIX}_hierarchical_by_building_floor.csv", index=False)
    E.floor_error_distance(ho).to_csv(out.tables / f"{MT_PREFIX}_heldout_floor_error_distance.csv", index=False)
    cm_floor = E.confusion_frame(ho.FLOOR_TRUE, ho.FLOOR_PRED, prep.floor_classes)
    cm_building = E.confusion_frame(ho.BUILDINGID_TRUE, ho.BUILDINGID_PRED, prep.building_classes)
    cm_floor.to_csv(out.tables / f"{MT_PREFIX}_heldout_floor_confusion_matrix.csv")
    cm_building.to_csv(out.tables / f"{MT_PREFIX}_heldout_building_confusion_matrix.csv")

    # ---- robustness by identifiers (post hoc; identifiers are not model inputs) ----
    train_phones = set(dev["internal_train"].PHONEID)
    robust = pd.concat([
        E.robustness_by_group(frames["internal_validation"], "USERID", "internal_validation",
                              set(dev["internal_train"].USERID)),
        E.robustness_by_group(frames["internal_validation"], "PHONEID", "internal_validation", train_phones),
        E.robustness_by_group(ho, "PHONEID", "heldout_validation", train_phones),
    ], ignore_index=True)
    robust.to_csv(out.tables / f"{MT_PREFIX}_robustness_by_identifier.csv", index=False)

    # ---- seed variability of the frozen configuration (no re-selection) ----------
    seed_rows = []
    for seed in seeds:
        res = best if seed == best.config.seed else train_model(
            dataclasses.replace(best.config, seed=seed), arrays["internal_train"], arrays["internal_validation"],
            prep, coord_norm)
        f = _evaluate_frozen(res.model, prep, {"heldout_validation": heldout})["heldout_validation"]
        wide = _wide(pd.DataFrame(E.summary_rows(f, "heldout_validation", "DL multitask")), "heldout_validation")
        seed_rows.append({"seed": seed, "best_epoch": res.best_epoch, **wide.iloc[0].drop(["split", "model"]).to_dict()})
    seed_table = pd.DataFrame(seed_rows)
    seed_table.to_csv(out.tables / f"{MT_PREFIX}_seed_variability.csv", index=False)

    # ---- AML vs DL (and single-task) comparison on identical rows ---------------
    comparison, paired = _compare(metrics, frames, out)

    # ---- figures ----------------------------------------------------------------
    import matplotlib.pyplot as plt
    P.plot_training_history(best.history, f"DL multitask: {best.config.name}",
                            out.figures / f"{MT_PREFIX}_training_history.png")
    aml_pred = _load_aml_predictions("heldout_validation")
    if aml_pred is not None:
        aml_cm = pd.read_csv(C.TABLES_DIR / "ml_05_floor_heldout_confusion_matrix.csv", index_col=0)
        P.plot_confusion_pair(cm_floor, "DL multitask", aml_cm, "AML Random Forest",
                              "Held-out floor confusion matrices", out.figures / f"{MT_PREFIX}_heldout_floor_confusion_dl_vs_aml.png")
        P.plot_error_distribution({"DL multitask": ho.EUCLIDEAN_ERROR.to_numpy(),
                                   "AML Random Forest": aml_pred.EUCLIDEAN_ERROR.to_numpy()},
                                  "Held-out coordinate error: DL vs AML",
                                  out.figures / f"{MT_PREFIX}_heldout_error_distribution_dl_vs_aml.png")
        aml_h = pd.read_csv(C.AML_HIERARCHY_PATH)
        P.plot_hierarchical_comparison({"DL multitask": hier[hier.split == "heldout_validation"],
                                        "AML Random Forest": aml_h[aml_h.split == "heldout_validation"]},
                                       "Held-out hierarchical building/floor error analysis",
                                       out.figures / f"{MT_PREFIX}_hierarchical_dl_vs_aml.png")
    else:
        P.plot_confusion(cm_floor, "Held-out floor confusion (DL multitask)").savefig(
            out.figures / f"{MT_PREFIX}_heldout_floor_confusion.png", dpi=160, bbox_inches="tight")
    P.plot_coordinate_error_map(ho, "Held-out coordinate error map (DL multitask)",
                                out.figures / f"{MT_PREFIX}_heldout_error_map.png")
    held = comparison[comparison.split == "heldout_validation"]
    P.plot_metric_comparison(held, "Held-out metrics: DL multitask vs single-task DL vs AML",
                             out.figures / f"{MT_PREFIX}_heldout_metric_comparison.png")
    P.plot_robustness(robust[(robust.split == "heldout_validation")],
                      "Held-out robustness by PHONEID (DL multitask; post-hoc)",
                      out.figures / f"{MT_PREFIX}_heldout_robustness_by_phone.png")
    plt.close("all")
    return {"selection": table, "metrics": metrics, "frames": frames, "hierarchy": hier, "robustness": robust,
            "seeds": seed_table, "comparison": comparison, "paired": paired, "best": best,
            "cm_floor": cm_floor, "cm_building": cm_building}


def _load_aml_predictions(split: str) -> pd.DataFrame | None:
    path = C.TABLES_DIR / f"ml_05_aml_{'heldout' if split == 'heldout_validation' else 'internal_validation'}_integrated_predictions.csv"
    return pd.read_csv(path) if path.is_file() else None


def _compare(mt_metrics: pd.DataFrame, mt_frames: dict, out: Outputs):
    parts = [mt_metrics.drop(columns="config")]
    for task, prefix in TASK_PREFIX.items():
        p = out.tables / f"{prefix}_metrics.csv"
        if p.is_file():
            parts.append(pd.read_csv(p).drop(columns="config"))
    if C.AML_SUMMARY_PATH.is_file():
        aml = E.load_aml_summary()
        aml["model"] = "AML Random Forest"
        aml_rows = aml.to_dict("records")
        for split in ("internal_validation", "heldout_validation"):
            aml_pred = _load_aml_predictions(split)
            if aml_pred is not None:
                aml_rows.append({"split": split, "model": "AML Random Forest", "task": "building_and_floor",
                                 "accuracy": float(aml_pred.BUILDING_FLOOR_CORRECT.mean()), "n_rows": len(aml_pred)})
        parts.append(pd.DataFrame(aml_rows))
    long = pd.concat(parts, ignore_index=True)
    wide = pd.concat([_wide(long, s) for s in ("internal_validation", "heldout_validation")], ignore_index=True)
    # Single-task rows each cover one head; merge them into one "DL single-task" row per split.
    st = wide[wide.model.str.startswith("DL single-task")]
    if len(st):
        merged = st.groupby("split", sort=False).first().reset_index()
        merged["model"] = "DL single-task (separate nets)"
        wide = pd.concat([wide[~wide.model.str.startswith("DL single-task")], merged], ignore_index=True)
    order = {"DL multitask": 0, "DL single-task (separate nets)": 1, "AML Random Forest": 2}
    wide = wide.sort_values(["split", "model"], key=lambda s: s.map(order) if s.name == "model" else s,
                            kind="stable").reset_index(drop=True)
    wide["coordinate_unit"] = C.COORDINATE_UNIT_LABEL
    wide.to_csv(out.tables / f"{MT_PREFIX}_aml_vs_dl_comparison.csv", index=False)

    paired = {}
    for split, f in mt_frames.items():
        aml_pred = _load_aml_predictions(split)
        if aml_pred is not None:
            paired[split] = E.paired_comparison(f, aml_pred)
    if paired:
        from .train import write_json
        write_json(paired, out.tables / f"{MT_PREFIX}_paired_comparison_vs_aml.json")
    return wide, paired


