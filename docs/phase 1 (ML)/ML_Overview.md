# Signal2Space — machine learning pipeline (Phase 1)

This document describes the AML (applied machine learning) implementation: how to run it, how the data is prepared, how the three Random Forest models are chosen, and what they achieve. It is the Phase 1 companion to [`DL_PIPELINE.md`](DL_PIPELINE.md). For the dataset and the partition caveats, read [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) first.

All coordinate errors are in **dataset coordinate units**. ML_04 labels them EPSG:3857 projected metres, but the repository has no authoritative coordinate reference metadata, so they are not reported as verified metres here.

## 1. Layout

| Notebook | Purpose | Main outputs |
|---|---|---|
| `Machine Learning Pipeline/ML_01_data_preparation_features.ipynb` | schema and quality checks, EDA, grouped split, train-only RSSI scaling, 1,046 features | `data/processed/aml_*_features.csv`, `data/figures/ml_01_*.png` |
| `Machine Learning Pipeline/ML_02_building_classification.ipynb` | building Random Forest | `data/tables/ml_02_building_*`, `data/figures/ml_02_*.png` |
| `Machine Learning Pipeline/ML_03_floor_classification.ipynb` | floor Random Forest | `data/tables/ml_03_floor_*`, `data/figures/ml_03_*.png` |
| `Machine Learning Pipeline/ML_04_coordinate_regression.ipynb` | multi-output coordinate Random Forest | `data/tables/ml_04_coordinate_*`, `data/figures/ml_04_*.png` |
| `Machine Learning Pipeline/ML_05_integration_evaluation.ipynb` | joins the three tasks, recomputes metrics, hierarchical analysis | `data/tables/ml_05_*`, `data/figures/ml_05_*.png` |

Each task notebook writes internal-validation and held-out predictions, a metrics table and a held-out confusion matrix or error figures. ML_05 writes the integrated prediction tables (`ml_05_aml_*_integrated_predictions.csv`) that the DL phase joins against.

## 2. How to run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

Run ML_01 → ML_05 in order. ML_01 must run first: every later notebook, and the whole DL phase, reads its feature tables. The raw files `data/raw/trainingData.csv` and `data/raw/validationData.csv` must be present. Each notebook takes seconds to a few minutes on a laptop CPU; all models use `random_state=42`.

## 3. Data and features (ML_01)

### Checks and EDA

* Both files have the same 529 columns: 520 WAP fields `WAP001`–`WAP520`, 4 targets and 5 metadata fields. Neither file has blank cells.
* RSSI values range from −104 to 0 dBm in training (−102 to −34 held-out). The value **`100` means "not detected"**: 96.54% of training WAP entries and 96.83% of held-out entries are `100`.
* Training contains 637 exactly duplicated rows and 76 rows with no detected access point. Both were counted and kept.
* No raw row appears in both files. In the held-out file, `SPACEID`, `RELATIVEPOSITION` and `USERID` are placeholders (0 for every row).

Figures: `ml_01_data_quality_eda.png`, `ml_01_target_distribution_eda.png`, `ml_01_coordinate_labels_eda.png`.

### Grouped split

`GroupShuffleSplit(test_size=0.20, random_state=42)` over groups of `(BUILDINGID, FLOOR, SPACEID)` divides `trainingData.csv` into 735 groups:

| Partition | Rows | Groups | Used for |
|---|---:|---:|---|
| internal training | 16,085 | 588 | fitting the scaler and models |
| internal validation | 3,852 | 147 | choosing hyperparameters |
| held-out (`validationData.csv`) | 1,111 | — | one final evaluation per frozen model |

Grouping keeps every record from one space on the same side of the split. It does not stop identical fingerprints or identical coordinates appearing on both sides: 156 internal-validation rows have a fingerprint identical to an internal-training row, and 1,428 share an exact coordinate pair with it (`dl_01_partition_overlap.csv`).

### Feature construction

All fitted values come from internal-training rows only; validation and held-out rows are transform-only.

| Block | Count | Definition |
|---|---:|---|
| `RSSI_Z__WAPxxx` | 520 | detected readings standardized with that WAP's internal-training mean and population standard deviation; undetected readings filled with a train-only value one unit below the WAP's lowest standardized training reading. The 69 WAPs never detected in internal training fall back to global train-only statistics. |
| `SEEN__WAPxxx` | 520 | 1 if the WAP was detected, else 0 |
| `n_heard` | 1 | number of detected WAPs |
| `max_rssi_dbm`, `mean_rssi_dbm`, `std_rssi_dbm` | 3 | strongest, mean and standard deviation of detected RSSI (−110 / −110 / 0 when none) |
| `n_strong` | 1 | detected readings stronger than −70 dBm |
| `top1_ap_index` | 1 | index of the strongest WAP (−1 when none) |

That gives **1,046 predictors**. The tables also keep `ROW_ID`, `SOURCE_PARTITION`, the targets and the metadata for joins and evaluation. Those columns are excluded from every model's predictor list.

## 4. Models

Each target gets its own Random Forest, so the three predictions are made independently. Random Forests suit this data: they handle 1,046 mixed-scale, sparse inputs without further scaling, capture non-linear interactions between access points, and need little tuning. Every notebook follows the same protocol:

1. Fit a default baseline and a small grid of candidates on internal training.
2. Rank candidates on internal validation.
3. Refit the winner on internal training and freeze it.
4. Only then open the held-out table and score it once.

| Task | Notebook | Candidates | Ranking rule | Selected configuration |
|---|---|---|---|---|
| Building (3 classes) | ML_02 | 18: `n_estimators` 100/200/300 × `max_depth` None/20/30 × `min_samples_leaf` 1/2 | internal-validation accuracy, then macro F1 | 100 trees, unlimited depth, leaf size 1 (the default) |
| Floor (5 classes) | ML_03 | 8: `n_estimators` 150/300 × `max_depth` None/30 × `min_samples_leaf` 1/2; `max_features="sqrt"`, `class_weight="balanced"` | accuracy, then macro F1, then fewer trees, shallower depth | 150 trees, unlimited depth, leaf size 1, `sqrt`, balanced class weights |
| Coordinates (2 outputs) | ML_04 | 10 combinations of trees (100–200), depth (None/50), leaf size (1/2) and `max_features` (1.0, 0.5, `sqrt`) | internal-validation mean Euclidean error | 150 trees, unlimited depth, leaf size 1, `max_features=0.5` |

The floor model uses balanced class weights because floor 4 exists only in building 2 and is the smallest class (862 internal-training rows).

## 5. Model selection (internal validation)

**Building.** Every candidate scores 99.92%–99.95%. The default 100-tree model ties with the 200- and 300-tree models at 99.95% and is first in grid order, so tuning keeps the default.

**Floor** (8 candidates, from `ML_03` outputs):

| Trees | Max depth | Leaf | Accuracy | Macro F1 |
|---:|---|---:|---:|---:|
| **150** | **None** | **1** | **96.42%** | **96.78%** |
| 300 | None | 1 | 96.37% | 96.74% |
| 300 | 30 | 1 | 95.85% | 96.29% |
| 300 | None | 2 | 95.85% | 96.29% |
| 150 | None | 2 | 95.48% | 95.98% |
| 150 | 30 | 1 | 95.41% | 95.95% |
| 150 | 30 | 2 | 95.30% | 95.77% |
| 300 | 30 | 2 | 95.04% | 95.60% |

The default 100-tree baseline reaches 96.21%; tuning adds 0.21 percentage points.

**Coordinates** (`ml_04_coordinate_tuning_results.csv`, 10 candidates):

| Trees | Max depth | Leaf | `max_features` | Mean error | Median error |
|---:|---|---:|---|---:|---:|
| **150** | **None** | **1** | **0.5** | **8.58** | **6.31** |
| 200 | None | 1 | `sqrt` | 8.66 | 6.82 |
| 150 | None | 1 | `sqrt` | 8.67 | 6.84 |
| 150 | 50 | 1 | 1.0 | 9.19 | 6.47 |
| 150 | None | 1 | 1.0 | 9.20 | 6.51 |
| 200 | None | 1 | 1.0 | 9.20 | 6.47 |
| 200 | 50 | 1 | 1.0 | 9.20 | 6.44 |
| 100 | None | 1 | 1.0 | 9.26 | 6.51 |
| 150 | None | 2 | 1.0 | 9.42 | 6.58 |
| 200 | None | 2 | 1.0 | 9.43 | 6.54 |

Restricting each split to half of the features (`max_features=0.5`) decorrelates the trees and lowers the mean error from 9.26 (default) to 8.58.

## 6. Results

### Internal validation and held-out

From `data/tables/ml_05_aml_summary_metrics.csv`:

| Metric | Internal validation (3,852) | Held-out (1,111) |
|---|---:|---:|
| Building accuracy / macro F1 | 99.95% / 99.95% | **99.73% / 99.71%** |
| Floor accuracy / macro F1 | 96.42% / 96.78% | **88.93% / 87.27%** |
| Joint building + floor correct | 96.42% | **88.75%** (986 rows) |
| MAE longitude axis | 5.91 | **8.10** |
| MAE latitude axis | 5.06 | **6.81** |
| Mean / median Euclidean error | 8.58 / 6.31 | **11.58 / 7.49** |
| 75th / 95th percentile error | 11.09 / 22.77 | 13.68 / 36.21 |

The internal-to-held-out drop is expected. The held-out file was collected months later with mostly different phones, while internal validation shares users, phones, the collection period and many positions with training.

### Building confusion (held-out; rows = actual)

| actual \ predicted | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 535 | 1 | 0 |
| 1 | 1 | 305 | 1 |
| 2 | 0 | 0 | 268 |

1,108 of 1,111 rows are correct. All three errors involve building 1 or a neighbouring code.

### Floor confusion and recall (held-out; rows = actual)

| actual \ predicted | 0 | 1 | 2 | 3 | 4 | Recall |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 119 | 9 | 4 | 0 | 0 | 90.15% |
| 1 | 11 | 401 | 49 | 1 | 0 | 86.80% |
| 2 | 0 | 8 | 279 | 19 | 0 | 91.18% |
| 3 | 0 | 0 | 8 | 162 | 2 | 94.19% |
| 4 | 1 | 1 | 0 | 10 | 27 | 69.23% |

Most errors are between adjacent floors. The largest is floor 1 predicted as floor 2 (49 rows), then floor 2 predicted as floor 3 (19) and floor 4 predicted as floor 3 (10). Floor 4 has the lowest recall and only 39 held-out rows.

The weakest building-floor cells (`ml_05_aml_hierarchical_by_building_floor.csv`):

| Building | Floor | Rows | Floor accuracy | Mean coordinate error |
|---:|---:|---:|---:|---:|
| 1 | 1 | 143 | 65.7% | 15.09 |
| 2 | 2 | 54 | 68.5% | 17.16 |
| 2 | 4 | 39 | 69.2% | 26.92 |
| 1 | 0 | 30 | 76.7% | 14.72 |

Building 0 is easiest: every floor there is above 95%.

### Coordinate errors (held-out)

* Mean error by building: **7.97** (building 0), **14.60** (building 1), **15.35** (building 2).
* 32.8% of records are within 5 units, but the tail is long: the 95th percentile is 36.21 and the largest error is 135.1.

Figures:

* `ml_04_coordinate_actual_vs_predicted.png`;
* `ml_04_coordinate_error_map.png` (error at each true position);
* `ml_04_coordinate_displacement_segments.png`;
* `ml_04_coordinate_error_distribution.png`;
* `ml_05_coordinate_error_visualizations.png`.

### Hierarchical error analysis (held-out)

From `ml_05_aml_hierarchical_error_breakdown.csv`:

| Outcome | Rows | Share | Mean coordinate error | Median |
|---|---:|---:|---:|---:|
| both correct | 986 | 88.75% | 10.00 | 6.94 |
| building correct, floor wrong | 122 | 10.98% | 23.27 | 15.19 |
| building wrong, floor correct | 2 | 0.18% | 44.31 | 44.31 |
| both wrong | 1 | 0.09% | 77.04 | 77.04 |

The coordinate model never sees the floor prediction, so a wrong floor does not cause its errors. Yet when the floor is wrong, the position error more than doubles. That suggests the same hard fingerprints confuse both models. Figures: `ml_05_hierarchical_error_breakdown.png`, `ml_05_heldout_confusion_matrices.png`, `ml_05_heldout_metric_comparison.png`.

## 7. Interpretation

* **Building** is essentially solved by the fingerprint (99.73%), most likely because each building hears a largely different set of access points.
* **Floor** is the hard task, most likely because floors in the same building hear many of the same access points at different strengths. Errors are almost always one floor away, and are concentrated in a few building-floor cells.
* **Coordinates** are accurate where the radio map is dense (building 0) and much less so in buildings 1 and 2. The error distribution has a long tail.
* **Independent models** are simple and strong, but they can disagree. Nothing ties a predicted position to the predicted building or floor.

Phase 2 keeps these features, this split and these metrics and replaces the three forests with one shared-trunk neural network. It improves floor accuracy to 90.64% but does not improve coordinates; see [`DL_PIPELINE.md`](DL_PIPELINE.md).

## 8. Limitations

* Results describe the supplied UJIIndoorLoc partition: a later collection campaign in the same three buildings, not unseen buildings.
* The coordinate reference system is not verified, so errors are given in dataset units, not metres, and no projection-scale correction is applied.
* Grids are small and evaluated on one internal split, with no repeated cross-validation or confidence intervals. Internal validation overlaps heavily with training, so its scores are optimistic.
* Building, floor and coordinates are learned independently, and no constraint enforces a valid combination.
* Duplicate and all-undetected training rows were kept; their effect was not measured separately.
* Only Random Forests were tuned. The results do not rank them against other model families such as nearest-neighbour matching or gradient boosting.
