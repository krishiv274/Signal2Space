"""Command-line workflow: prepare -> train -> evaluate -> compare.

Examples (from the repository root):

    python scripts/run_dl_pipeline.py all                  # full run, all four experiments
    python scripts/run_dl_pipeline.py prepare              # DL_01 outputs only
    python scripts/run_dl_pipeline.py train --experiments multitask
    python scripts/run_dl_pipeline.py evaluate             # score frozen checkpoints
    python scripts/run_dl_pipeline.py all --max-epochs 3   # quick smoke test (NOT a full run)
"""
from __future__ import annotations

import argparse
import dataclasses
import json

import pandas as pd

from . import config as C
from . import evaluate as E
from . import plots as P
from .data import prepare
from .train import train_experiment


def run_prepare(paths, cfg, verbose=True):
    data = prepare(paths, coord_scale=cfg.coord_scale, save=True)
    if verbose:
        print("Split rows:", {s: len(f) for s, f in data.frames.items()})
        print("AML split parity:", data.parity)
    P.input_representation(data, paths)
    return data


def run_train(data, paths, cfg, experiments, device="auto"):
    metas = {}
    for name in experiments:
        exp_cfg = dataclasses.replace(cfg, tasks=C.EXPERIMENTS[name])
        metas[name] = train_experiment(name, data, paths, exp_cfg, device=device)
        hist = pd.read_csv(paths.tables / f"{C.EXPERIMENT_PREFIX[name]}_training_history.csv")
        P.training_curves(hist, paths, C.EXPERIMENT_PREFIX[name], name)
    return metas


def run_evaluate(data, paths, experiments):
    preds, rows = {}, []
    for name in experiments:
        preds[name], r = E.evaluate_experiment(name, data, paths)
        rows += r
        prefix = C.EXPERIMENT_PREFIX[name]
        held = preds[name]["heldout_validation"]
        for t in C.EXPERIMENTS[name]:
            if t in ("building", "floor"):
                stem = prefix if prefix.endswith(t) else f"{prefix}_{t}"
                P.confusion(held, t, paths, f"{stem}_heldout_confusion_matrix.png",
                            f"{name}: held-out {t}")
                E.per_class_report(held, t, f"DL {name}").to_csv(
                    paths.tables / f"{stem}_heldout_per_class.csv", index=False)
            else:
                P.coordinate_errors(held, paths, prefix, f"{name} held-out")
        pd.DataFrame([x for x in r]).to_csv(paths.tables / f"{prefix}_metrics.csv", index=False)
    return preds, rows


def run_compare(preds, rows, paths):
    """Integrated DL vs AML comparison on the identical held-out partition."""
    table = pd.DataFrame(rows + E.aml_metric_rows(paths))
    order = {"AML Random Forest": 0}
    table["_o"] = table.model.map(lambda m: order.get(m, 1))
    table = table.sort_values(["split", "task", "_o", "model"]).drop(columns="_o")
    table.to_csv(paths.tables / "dl_05_model_comparison.csv", index=False)
    out = {"comparison_rows": len(table)}
    if "multitask" not in preds:
        return out
    mt = preds["multitask"]
    hier = pd.concat([E.hierarchy_breakdown(mt[s], s, "DL multitask") for s in E.EVAL_SPLITS])
    hier.to_csv(paths.tables / "dl_05_hierarchical_breakdown.csv", index=False)
    E.per_building_coordinate_error(mt["heldout_validation"], "DL multitask").to_csv(
        paths.tables / "dl_05_heldout_coordinate_error_by_building.csv", index=False)
    aml = E.load_aml_heldout(paths)
    curves = {"DL multitask": mt["heldout_validation"].EUCLIDEAN_ERROR.to_numpy()}
    if "coordinate_only" in preds:
        curves["DL coordinate_only"] = preds["coordinate_only"]["heldout_validation"].EUCLIDEAN_ERROR.to_numpy()
    if aml is not None:
        out["aml_recomputed"] = E.recompute_aml_heldout(aml)
        boot = pd.concat([E.paired_bootstrap(preds[n]["heldout_validation"], aml, f"DL {n}") for n in preds])
        boot.to_csv(paths.tables / "dl_05_vs_aml_paired_bootstrap.csv", index=False)
        out["paired_bootstrap"] = boot.to_dict(orient="records")
        aml_err = aml.EUCLIDEAN_ERROR.to_numpy()
        curves["AML Random Forest"] = aml_err
        hier = pd.concat([hier, E.hierarchy_breakdown(aml, "heldout_validation", "AML Random Forest")])
        hier.to_csv(paths.tables / "dl_05_hierarchical_breakdown.csv", index=False)
    P.error_cdf(curves, paths, "dl_05_heldout_error_cdf.png", "Held-out coordinate error CDF")
    P.model_comparison(table, paths, "dl_05_heldout_model_comparison.png")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    P.confusion(mt["heldout_validation"], "building", paths, "", "DL multitask: held-out building", ax=axes[0])
    P.confusion(mt["heldout_validation"], "floor", paths, "", "DL multitask: held-out floor", ax=axes[1])
    P._save(fig, paths, "dl_05_multitask_heldout_confusion_matrices.png")
    joint = int(((mt["heldout_validation"].BUILDINGID_TRUE == mt["heldout_validation"].BUILDINGID_PRED)
                 & (mt["heldout_validation"].FLOOR_TRUE == mt["heldout_validation"].FLOOR_PRED)).sum())
    out["dl_multitask_joint_building_floor_correct"] = joint
    (paths.tables / "dl_05_comparison_summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Signal2Space deep-learning pipeline")
    ap.add_argument("stage", choices=["prepare", "train", "evaluate", "all"])
    ap.add_argument("--experiments", nargs="+", default=list(C.EXPERIMENTS), choices=list(C.EXPERIMENTS))
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--max-epochs", type=int, default=None, help="override for smoke tests only")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    paths = C.Paths(C.find_project_root()).ensure()
    cfg = C.TrainConfig()
    if args.max_epochs is not None:
        cfg = dataclasses.replace(cfg, max_epochs=args.max_epochs)
        print(f"WARNING: max_epochs={args.max_epochs} override - this is a smoke test, not the full run.")
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)

    data = run_prepare(paths, cfg)
    if args.stage == "prepare":
        return
    if args.stage in ("train", "all"):
        run_train(data, paths, cfg, args.experiments, args.device)
    if args.stage in ("evaluate", "all"):
        preds, rows = run_evaluate(data, paths, args.experiments)
        summary = run_compare(preds, rows, paths)
        print(json.dumps(summary, indent=1, default=float))
        held = pd.read_csv(paths.tables / "dl_05_model_comparison.csv")
        held = held[held.split == "heldout_validation"]
        cols = ["model", "task", "accuracy", "macro_f1", "mae_longitude", "mae_latitude", "mean_euclidean_error"]
        print(held[[c for c in cols if c in held]].to_string(index=False))


if __name__ == "__main__":
    main()
