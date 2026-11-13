# Signal2Space — deep learning pipeline (Phase 2)

This document describes the DL implementation, how to run it, the results, and how they compare with the AML baseline. For the dataset, the AML pipeline and the partition caveats, read [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) first.

All coordinate errors are in **dataset coordinate units**: the repository has no authoritative CRS metadata, so they are not reported as metres.

## 1. Layout

| Component | File | Purpose |
|---|---|---|
| Configuration | `src/signal2space_dl/config.py` | paths, 1,046-column feature contract, seeds, `TrainConfig`, selection grid |
| Data | `src/signal2space_dl/data.py` | load/validate AML feature tables, rebuild ML_01's split from raw data, fit train-only DL transforms, DL_01 report |
| Model | `src/signal2space_dl/model.py` | shared-trunk multitask network and loss |
| Training | `src/signal2space_dl/train.py` | training loop, early stopping, internal-validation model selection, checkpoints |
| Evaluation | `src/signal2space_dl/evaluate.py` | metrics, confusion matrices, hierarchical/robustness analysis, paired AML comparison |
| Figures | `src/signal2space_dl/plots.py` | all `data/figures/dl_*.png` |
| Runner | `src/signal2space_dl/pipeline.py` | stages and command-line interface; single-task ablations |
| Multitask | `src/signal2space_dl/multitask.py` | DL_05: multitask selection, evaluation and comparison |
| Notebooks | `Deep Learning Pipeline/DL_01` … `DL_05` | executed, documented walkthroughs of each stage |
| Tests | `tests/test_dl_*.py` | unit tests plus a checkpoint-reproduces-predictions check |

Outputs: `data/tables/dl_0x_*.csv|json`, `data/figures/dl_0x_*.png`, `models/dl/*.pt|json`.

## 2. How to run

Run ML_01–ML_05 first: DL reuses `data/processed/aml_*_features.csv`, and the comparison reads `data/tables/ml_05_*`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Either execute the notebooks `DL_01` → `DL_05` in order (Jupyter), or run everything from the command line at the repository root:

```bash
PYTHONPATH=src python -m signal2space_dl.pipeline --stage all
```

The full run takes about 18 minutes on a 4-core CPU without a GPU. Single stages: `--stage prepare|building|floor|coord|multitask`. Smoke run (tiny grid, 8 epochs, writes only to `outputs_smoke/`, which is git-ignored; never report these numbers):

```bash
PYTHONPATH=src python -m signal2space_dl.pipeline --stage all --quick
```

Tests:

```bash
PYTHONPATH=src python -m pytest -q tests
```

## 3. Data and features (DL_01)

* **Same rows as AML.** Internal training (16,085 rows) and internal validation (3,852) are read from the ML_01 tables. `data.verify_split_against_raw` rebuilds ML_01's `GroupShuffleSplit` from `trainingData.csv` and confirms the same rows and targets. The held-out partition is the same 1,111 rows of `validationData.csv`.
* **Same 1,046 predictors as AML.** These are 520 train-scaled RSSI values (undetected `100` imputed from train-only values), 520 `SEEN` missingness indicators and 6 signal summaries.
* **One DL-specific difference.** The 6 summary columns are raw dBm/count values in the AML table. Tree models do not care about scale, but a network does, so DL standardises them with internal-training mean and standard deviation. `top1_ap_index` is kept as a standardised numeric value for exact feature parity with AML, even though it is really an identifier of an access point. The strongest-AP information is also present in the RSSI block. This does not change which information each model sees.
* **Targets.** Building and floor labels are mapped to class indices from internal-training labels (buildings 0–2; floors 0–4). Coordinates are standardised with internal-training mean/std (≈ 124 and 67 units). Predictions are converted back to dataset units before any metric is computed.
* **Not inputs.** `USERID`, `PHONEID`, `SPACEID`, `RELATIVEPOSITION`, `TIMESTAMP` and the targets are carried alongside for evaluation only.
* **Descriptive overlap** (`dl_01_partition_overlap.csv`):
  * internal validation: 156 of 3,852 rows have a fingerprint identical to an internal-training row, and 1,428 share an exact coordinate pair with internal training, even though the `(building, floor, space)` groups are disjoint;
  * held-out: 0 identical fingerprints, and 26 rows on an exact internal-training position.

  Internal-validation scores are therefore optimistic. The held-out file is the stricter test.

## 4. Model

```
1,046 inputs → Dropout(p_in)
  → [Linear → BatchNorm1d → ReLU → Dropout(p)] × len(hidden_sizes)      shared trunk
      ├─ building head:   Linear(h,128) → ReLU → Linear(128, 3)         logits
      ├─ floor head:      Linear(h,128) → ReLU → Linear(128, 5)         logits
      └─ coordinate head: Linear(h,128) → ReLU → Linear(128, 2)         standardised (LONGITUDE, LATITUDE)

loss = w_b · CrossEntropy(building) + w_f · CrossEntropy(floor) + w_c · SmoothL1_β(coordinates)
```

Floor is predicted as one of 5 global floor labels. Building and floor are not forced to be consistent; the hierarchical analysis measures how often they disagree.

**Training settings.**
* Optimiser: AdamW (lr 1e-3, weight decay 1e-4), batch size 256, shuffled with a seeded generator.
* Learning rate: ReduceLROnPlateau (factor 0.5, patience 5).
* Stopping: up to 150 epochs, early stopping with patience 15 on the internal-validation selection score; the best epoch's weights are restored.
* Reproducibility: seeds for Python, NumPy and torch (`GLOBAL_SEED = 42`), plus `torch.use_deterministic_algorithms(True)` and a fixed thread count. Re-running the whole pipeline reproduced every reported metric.

**Selection score** (internal validation only, lower is better):

```
score = mean( building error rate, floor error rate, mean Euclidean error / D )
```

`D` = 125.9 units, the internal-validation error of always predicting the internal-training centroid. It puts coordinate error on roughly the same 0–1 scale as the error rates. Single-task runs use only their own term. Ties are broken by grid order, so the simpler candidate wins.

**Smooth-L1 β.** A preliminary internal-validation check (512×256 trunk, 60 epochs) gave selection scores of 0.0430 for β = 1.0, 0.0355 for β = 0.1 and 0.0378 for β = 0.02. With β = 1 the loss is quadratic for errors below one standard deviation (≈ 67–124 units), so fine-grained coordinate errors get very small gradients. β = 0.1 became the default. A β = 1.0 candidate stays in the selection grid so that this choice is recorded in `dl_05_multitask_selection.csv`.

## 5. Model selection (internal validation)

`dl_05_multitask_selection.csv`, 7 candidates:

| Candidate | Trunk | Dropout (in / hidden) | Loss weights b/f/c | β | Best epoch | Score | Val building | Val floor | Val mean Euclidean |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| **mt_512x256_d30_coordx2** (selected) | 512-256 | 0.1 / 0.3 | 1 / 1 / 2 | 0.1 | 25 | **0.0346** | 99.92% | 97.87% | 10.28 |
| mt_512x256_d30 | 512-256 | 0.1 / 0.3 | 1 / 1 / 1 | 0.1 | 26 | 0.0355 | 99.92% | 97.79% | 10.53 |
| mt_1024x512x256_d30 | 1024-512-256 | 0.1 / 0.3 | 1 / 1 / 1 | 0.1 | 15 | 0.0360 | 99.92% | 97.98% | 10.95 |
| mt_512x256_d30_floorx2 | 512-256 | 0.1 / 0.3 | 1 / 2 / 1 | 0.1 | 57 | 0.0367 | 99.92% | 97.46% | 10.56 |
| mt_512x256_d30_beta1 | 512-256 | 0.1 / 0.3 | 1 / 1 / 1 | 1.0 | 46 | 0.0375 | 99.92% | 97.85% | 11.34 |
| mt_1024x512_d30 | 1024-512 | 0.1 / 0.3 | 1 / 1 / 1 | 0.1 | 63 | 0.0438 | 99.84% | 97.69% | 13.42 |
| mt_512x256_d50 | 512-256 | 0.2 / 0.5 | 1 / 1 / 1 | 0.1 | 15 | 0.0489 | 99.92% | 96.26% | 13.65 |

The selected model has 768,906 parameters and trains in about 40 seconds on CPU. Its configuration (`models/dl/dl_05_multitask_selected_config.json`) and weights (`models/dl/dl_05_multitask.pt`) were saved **before** the held-out partition was loaded.

The single-task ablations (DL_02–DL_04) searched the four architecture candidates, each with only its own head. The selected configurations were:
* building: 512-256 (all candidates tied at 99.95%);
* floor: 1024-512-256;
* coordinates: 1024-512.

## 6. Results

### Held-out partition (1,111 rows): AML vs DL on identical rows

From `data/tables/dl_05_multitask_aml_vs_dl_comparison.csv`:

| Metric | AML Random Forest | DL multitask | DL single-task (3 separate nets) |
|---|---:|---:|---:|
| Building accuracy / macro-F1 | 99.73% / 99.71% | **99.91% / 99.88%** | 99.73% / 99.68% |
| Floor accuracy / macro-F1 | 88.93% / 87.27% | **90.64% / 89.35%** | 90.28% / 89.17% |
| Joint building + floor correct | 88.75% (986) | **90.64% (1,007)** | — |
| MAE longitude axis | 8.10 | 8.55 | **6.76** |
| MAE latitude axis | 6.81 | 6.65 | **5.83** |
| Mean Euclidean error | 11.58 | 11.88 | **9.84** |
| Median Euclidean error | **7.49** | 9.80 | 7.59 |

The joint value for single-task DL is "—" because its building and floor predictions come from different networks; it is not a joint model.

**Internal validation (3,852 rows):**

| Model | Building | Floor | Mean Euclidean |
|---|---:|---:|---:|
| AML | 99.95% | 96.42% | 8.58 |
| DL multitask | 99.92% | 97.87% | 10.28 |
| DL single-task | 99.95% | 98.18% | 7.68 |

### Is the difference real? (paired, row-aligned, held-out)

`dl_05_multitask_paired_comparison_vs_aml.json`:

* **Floor:** DL right / AML wrong on 37 rows, the reverse on 18. Exact McNemar p = 0.014.
* **Joint building + floor:** 39 vs 18 rows, p = 0.0075.
* **Building:** 3 vs 1 rows, p = 0.63 (no evidence of a difference).
* **Coordinates:** mean paired difference DL − AML = +0.30 units, bootstrap 95% CI [−0.42, +1.02]. This is not a significant difference in mean error. DL is closer than AML on 39% of rows.

### Seed variability of the frozen multitask configuration (held-out)

`dl_05_multitask_seed_variability.csv`. Retraining the frozen configuration with seeds 42, 7 and 2026 (no re-selection) gave:
* floor accuracy 90.64%, 89.65% and 90.10% (mean 90.13%, SD 0.50 pp);
* mean Euclidean error 11.88, 12.40 and 11.94 (mean 12.07, SD 0.28);
* building accuracy 99.91% for every seed.

All three seeds beat the AML floor accuracy of 88.93%. None beats the AML mean coordinate error.

### Floor confusion (held-out, DL multitask; rows = actual)

| actual \ predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 0 | 117 | 11 | 4 | 0 | 0 |
| 1 | 11 | 411 | 39 | 1 | 0 |
| 2 | 1 | 7 | 286 | 12 | 0 |
| 3 | 0 | 0 | 7 | 163 | 2 |
| 4 | 1 | 0 | 0 | 8 | 30 |

Among rows with the correct building, 97 of the 103 floor errors are off by exactly one floor (`dl_05_multitask_heldout_floor_error_distance.csv`). The most common confusion is floor 1 predicted as floor 2 (39 rows), as it is for AML.

### Hierarchical error analysis (held-out)

| Outcome | AML rows | AML mean coord. error | DL rows | DL mean coord. error |
|---|---:|---:|---:|---:|
| both correct | 986 | 10.00 | 1,007 | 11.27 |
| building correct, floor wrong | 122 | 23.27 | 103 | 17.88 |
| building wrong, floor correct | 2 | 44.31 | 0 | — |
| both wrong | 1 | 77.04 | 1 | 11.74 |

A wrong floor still roughly doubles the AML coordinate error. The multitask DL degrades less, from 11.3 to 17.9. This is consistent with the shared trunk keeping coordinate and floor predictions coherent. On rows where both models get building and floor right, AML positions are more precise. Per-cell detail: `dl_05_multitask_hierarchical_by_building_floor.csv`. Figures: `dl_05_multitask_hierarchical_dl_vs_aml.png`, `dl_05_multitask_heldout_floor_confusion_dl_vs_aml.png`, `dl_05_multitask_heldout_error_map.png`, `dl_05_multitask_heldout_error_distribution_dl_vs_aml.png`.

### Robustness by phone (post hoc, held-out)

`dl_05_multitask_robustness_by_identifier.csv` and `dl_05_multitask_heldout_robustness_by_phone.png`. Only phones 13 (369 rows) and 14 (28 rows) also appear in training.
* Phone 13: floor accuracy 93.5%.
* The 9 unseen phones range from 76.5% (phone 5, 17 rows) to 95.0% (phone 21).
* The largest unseen group, phone 20 (213 rows), reaches 90.1% with a mean error of 14.29.

Small groups are noisy, and phone identity is confounded with when and where its rows were collected, so these are descriptive differences, not causal device effects. Held-out `USERID` is a placeholder (`0`), so no user-level held-out analysis is possible. Internal-validation robustness by `USERID`/`PHONEID` is in the same table, but those users and phones also appear in training.

## 7. Interpretation

* **Building:** all models are at ceiling (≥ 99.7%). The differences are 1–3 rows and not significant.
* **Floor:** the multitask network is the best floor classifier: +1.7 pp accuracy and +2.1 pp macro-F1 over AML on held-out data, statistically significant on paired rows and stable across seeds. Because the building head is almost always right, joint building/floor correctness improves by the same amount (986 → 1,007 rows).
* **Coordinates:** sharing the trunk hurts. The single-task coordinate network is clearly the best regressor (mean 9.84 vs AML 11.58), while the multitask head is statistically indistinguishable from AML on the mean and worse on the median. The floor/building and coordinate tasks compete for trunk capacity. Even with `w_c = 2` the trunk favours features that separate floors.
* **Practical recommendation:** use the multitask network when a single model must predict building, floor and position together, especially floor. If coordinate precision matters most, use the single-task DL coordinate network.

## 8. Limitations

* All results are on the supplied UJIIndoorLoc partition. The held-out file tests a later collection campaign with mostly unseen phones in the same buildings, not unseen buildings or unseen fingerprints in general.
* Coordinate units are dataset units with an unverified CRS (see `PROJECT_OVERVIEW.md` §5).
* The selection grid is small (7 multitask, 4 per single-task network) to stay CPU-friendly. Only one seed is used during selection; the seed study covers only the frozen configuration.
* Internal validation shares users, phones, time period, some fingerprints and many exact positions with internal training. Early stopping and selection therefore favour fitting the training campaign, which partly explains the gap between internal (≈ 98% floor) and held-out (≈ 90.6%) scores.
* The project specification PDF was not available in the environment that produced this implementation. The design follows its requirements as summarised in the task brief: shared-trunk multitask model with building, floor and coordinate heads, fair AML comparison, floor confusion matrices, coordinate error maps and hierarchical error analysis.
