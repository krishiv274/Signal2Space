"""End-to-end Signal2Space DL workflow.

Stages (each notebook DL_0x calls one of them; the CLI can run them all)::

    prepare      DL_01  validate AML feature contract, fit train-only DL transforms
    building     DL_02  single-task building network (ablation)
    floor        DL_03  single-task floor network (ablation)
    coord        DL_04  single-task coordinate network (ablation)
    multitask    DL_05  select + train the shared-trunk multitask model, evaluate,
                        compare with AML and the single-task ablations
                        (implemented in ``signal2space_dl.multitask``)

Usage (from the repository root)::

    PYTHONPATH=src python -m signal2space_dl.pipeline --stage all
    PYTHONPATH=src python -m signal2space_dl.pipeline --stage multitask --quick   # smoke run

``--quick`` caps epochs and the selection grid so the whole workflow finishes in
a few minutes; its outputs are written with a ``smoke_`` prefix in a separate
folder and must not be reported as results.

Held-out discipline: every stage selects its configuration using internal
validation only, saves that choice, and only then loads the supplied held-out
file for a single evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import evaluate as E
from . import plots as P

TASK_PREFIX = {"building": "dl_02_building", "floor": "dl_03_floor", "coord": "dl_04_coordinate"}


@dataclasses.dataclass
class Outputs:
    tables: Path = C.TABLES_DIR
    figures: Path = C.FIGURES_DIR
    models: Path = C.MODELS_DIR

    @classmethod
    def smoke(cls, root: Path) -> "Outputs":
        return cls(root / "tables", root / "figures", root / "models")

    def ensure(self) -> "Outputs":
        for p in (self.tables, self.figures, self.models):
            p.mkdir(parents=True, exist_ok=True)
        return self


def _quick(grid, epochs=8):
    return tuple(dataclasses.replace(c, max_epochs=epochs, patience=3) for c in grid[:2])


def _context():
    """Load development data, fit the preprocessor, transform; no held-out access."""
    dev = D.load_development_splits()
    prep = D.fit_preprocessor(dev["internal_train"])
    arrays = {k: prep.transform(v) for k, v in dev.items()}
    coord_norm = D.centroid_baseline_error(dev["internal_train"], dev["internal_validation"])
    return dev, prep, arrays, coord_norm


# --------------------------------------------------------------------------
# DL_01
# --------------------------------------------------------------------------


def run_prepare(out: Outputs = Outputs()) -> dict:
    """DL_01 stage (implemented in ``data.prepare_and_report``)."""
    out.ensure()
    return D.prepare_and_report(out.tables, out.models)


# --------------------------------------------------------------------------
# Shared evaluation helper
# --------------------------------------------------------------------------


def _evaluate_frozen(model, prep, arrays_by_split: dict) -> dict[str, pd.DataFrame]:
    from .train import predict

    frames = {}
    for split, arr in arrays_by_split.items():
        pred = predict(model, arr, prep)
        frames[split] = E.build_prediction_frame(arr.meta, np.asarray(prep.building_classes)[arr.building],
                                                 np.asarray(prep.floor_classes)[arr.floor], arr.coords, pred,
                                                 prep.building_classes, prep.floor_classes)
    return frames


# --------------------------------------------------------------------------
# DL_02 - DL_04: single-task ablations
# --------------------------------------------------------------------------

# Architecture/regularisation candidates only; loss weights are irrelevant with one head.
SINGLE_TASK_GRID = tuple(c for c in C.SELECTION_GRID
                         if c.w_floor == 1.0 and c.w_coord == 1.0 and c.coord_beta == 0.1)


def run_single_task(task: str, out: Outputs = Outputs(), quick: bool = False, verbose: bool = True) -> dict:
    from .train import save_model, select_configuration, write_json

    assert task in TASK_PREFIX, f"task must be one of {list(TASK_PREFIX)}"
    out.ensure()
    prefix = TASK_PREFIX[task]
    dev, prep, arrays, coord_norm = _context()
    grid = tuple(dataclasses.replace(c, name=c.name.replace("mt_", f"{task}_"), tasks=(task,))
                 for c in SINGLE_TASK_GRID)
    if quick:
        grid = _quick(grid)
    best, table, histories = select_configuration(grid, arrays["internal_train"], arrays["internal_validation"],
                                                  prep, coord_norm, verbose)
    # Freeze the choice before touching held-out data.
    table.to_csv(out.tables / f"{prefix}_selection.csv", index=False)
    histories.to_csv(out.tables / f"{prefix}_training_history.csv", index=False)
    write_json(best.config.to_dict(), out.models / f"{prefix}_selected_config.json")
    save_model(best, prep, out.models / f"{prefix}.pt")

    heldout = prep.transform(D.load_heldout_split())
    frames = _evaluate_frozen(best.model, prep, {"internal_validation": arrays["internal_validation"],
                                                 "heldout_validation": heldout})
    rows = []
    for split, f in frames.items():
        f.to_csv(out.tables / f"{prefix}_{split}_predictions.csv", index=False)
        rows += E.summary_rows(f, split, f"DL single-task {task}")
    metrics = pd.DataFrame(rows)
    metrics.insert(2, "config", best.config.name)
    metrics.to_csv(out.tables / f"{prefix}_metrics.csv", index=False)

    P.plot_training_history(best.history, f"DL single-task {task}: {best.config.name}",
                            out.figures / f"{prefix}_training_history.png")
    ho = frames["heldout_validation"]
    if task == "building":
        cm = E.confusion_frame(ho.BUILDINGID_TRUE, ho.BUILDINGID_PRED, prep.building_classes)
        cm.to_csv(out.tables / f"{prefix}_heldout_confusion_matrix.csv")
        P.plot_confusion(cm, "Held-out building confusion (DL single-task)").savefig(
            out.figures / f"{prefix}_heldout_confusion_matrix.png", dpi=160, bbox_inches="tight")
    elif task == "floor":
        cm = E.confusion_frame(ho.FLOOR_TRUE, ho.FLOOR_PRED, prep.floor_classes)
        cm.to_csv(out.tables / f"{prefix}_heldout_confusion_matrix.csv")
        P.plot_confusion(cm, "Held-out floor confusion (DL single-task)").savefig(
            out.figures / f"{prefix}_heldout_confusion_matrix.png", dpi=160, bbox_inches="tight")
    else:
        P.plot_coordinate_error_map(ho, "Held-out coordinate error (DL single-task)",
                                    out.figures / f"{prefix}_heldout_error_map.png")
    import matplotlib.pyplot as plt
    plt.close("all")
    return {"selection": table, "metrics": metrics, "frames": frames, "best": best}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["prepare", "building", "floor", "coord", "multitask", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="smoke run: tiny grid/epochs, outputs under --smoke-dir")
    ap.add_argument("--smoke-dir", type=Path, default=C.PROJECT_ROOT / "outputs_smoke")
    args = ap.parse_args(argv)
    out = Outputs.smoke(args.smoke_dir) if args.quick else Outputs()
    stages = ["prepare", "building", "floor", "coord", "multitask"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"=== stage: {stage} ===")
        if stage == "prepare":
            r = run_prepare(out)
            print("\n".join(r["checks"]))
        elif stage == "multitask":
            try:
                from .multitask import run_multitask
            except ImportError as exc:  # multitask.py arrives in step S5
                raise SystemExit(f"multitask stage unavailable: {exc}") from exc
            r = run_multitask(out, quick=args.quick, seeds=(C.GLOBAL_SEED,) if args.quick else C.VARIABILITY_SEEDS)
            print(r["comparison"].to_string())
        else:
            r = run_single_task(stage, out, quick=args.quick)
            print(r["metrics"].to_string())


if __name__ == "__main__":
    main()
