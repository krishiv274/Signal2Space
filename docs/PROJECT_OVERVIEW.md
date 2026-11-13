# Signal2Space — project overview and AML baseline

This document explains what Signal2Space predicts, how the data is represented, what the Phase 1 AML (applied machine learning) pipeline does, what its verified results are, and what the Phase 2 deep learning (DL) work adds. Every number below was checked against the files in `data/tables/` and the notebook code; where something is an assumption it says so.

## 1. What the project predicts

A phone standing inside a building can see some Wi-Fi access points (WAPs) and measures how strong each one is. That list of signal strengths is the device's **Wi-Fi fingerprint**. Signal2Space learns the mapping from a fingerprint to:

| Target | Type | Values in the data |
|---|---|---|
| `BUILDINGID` | classification | 3 buildings: 0, 1, 2 |
| `FLOOR` | classification | floors 0–3 in buildings 0 and 1; 0–4 in building 2 (5 floor labels overall) |
| `LONGITUDE`, `LATITUDE` | regression | two projected coordinate values (≈ −7,691 to −7,301 and ≈ 4,864,746 to 4,865,017) |

The data is the public **UJIIndoorLoc** dataset (`data/raw/trainingData.csv`, 19,937 rows; `data/raw/validationData.csv`, 1,111 rows).

## 2. How a fingerprint is represented

Each row has 520 columns `WAP001` … `WAP520`. A value between about −104 and 0 is the received signal strength (RSSI) in dBm; values closer to 0 are stronger. The value **`100` means "this access point was not detected"**. It is a sentinel, not a signal, so it must never be treated as a very strong reading.

Rows also contain the targets above and collection metadata: `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID` and `TIMESTAMP`. Metadata is kept for checking and analysis but is **never** used as a model input.

### The shared feature table (ML_01)

`Machine Learning Pipeline/ML_01_data_preparation_features.ipynb` turns each row into **1,046 predictors**:

| Block | Count | How it is computed |
|---|---:|---|
| `RSSI_Z__WAPxxx` | 520 | For each WAP, the mean and standard deviation of its *detected* readings are fitted on internal-training rows only; readings are standardized with them. An undetected reading is replaced by a train-fitted value one unit below that WAP's lowest standardized training reading. WAPs never detected in internal training fall back to global train-only statistics. |
| `SEEN__WAPxxx` | 520 | 1 if the WAP was detected, else 0 (explicit missingness indicator). |
| Signal summaries | 6 | `n_heard` (detected WAPs), `max_rssi_dbm`, `mean_rssi_dbm`, `std_rssi_dbm` (over detected WAPs; −110/−110/0 when none), `n_strong` (detected and > −70 dBm), `top1_ap_index` (index of the strongest WAP; −1 when none). |

## 3. Data partitions

| Partition | Rows | Built from | Used for |
|---|---:|---|---|
| internal training | 16,085 | `trainingData.csv` | fitting scalers and models |
| internal validation | 3,852 | `trainingData.csv` | choosing hyperparameters / early stopping |
| held-out validation | 1,111 | `validationData.csv` | one final evaluation after the configuration is frozen |

The internal split is a `GroupShuffleSplit` (`test_size=0.20`, `random_state=42`) grouped by `(BUILDINGID, FLOOR, SPACEID)`, so all records from one space stay on the same side. The DL code rebuilds this split from the raw CSV and confirms that it matches the saved tables row for row.

### What the partitions do and do not test (verified facts)

* `trainingData.csv` contains 637 exactly duplicated raw rows and 852 duplicated fingerprints. Training positions are concentrated on 692 distinct coordinate pairs.
* **No held-out fingerprint is identical to a training fingerprint**, and only 29 of 1,111 held-out rows lie exactly on a training coordinate pair.
* The held-out file was collected **19 Sep – 8 Oct 2013**; the training file was collected **30 May – 20 Jun 2013**.
* In the held-out file `USERID` is a placeholder (`0` for every row), and only 2 of its 11 `PHONEID`s (13 and 14) appear in training.

So the held-out score measures generalisation to a later collection campaign with mostly different devices **inside the same three buildings**. It does not measure performance in unseen buildings. Because the internal validation split shares phones, users and the collection period with training, internal scores are optimistic compared with held-out scores.

## 4. The AML pipeline (Phase 1, implemented)

| Notebook | What it does | Model and selection | Main outputs |
|---|---|---|---|
| ML_01 | schema/quality checks, EDA, grouped split, train-only RSSI scaling, 1,046 features | — | `data/processed/aml_*_features.csv`, `data/figures/ml_01_*` |
| ML_02 | building classifier | Random Forest; 18-point grid (`n_estimators` 100/200/300 × `max_depth` None/20/30 × `min_samples_leaf` 1/2); selected by internal-validation accuracy, macro-F1 tie-break → `n_estimators=100, max_depth=None, min_samples_leaf=1` | `data/tables/ml_02_*` |
| ML_03 | floor classifier | Random Forest with `class_weight="balanced"`, `max_features="sqrt"`; 8-point grid; selected by internal-validation accuracy → `n_estimators=150, max_depth=None, min_samples_leaf=1` | `data/tables/ml_03_*` |
| ML_04 | coordinate regressor | multi-output Random Forest; 10-point grid; selected by internal-validation mean Euclidean error → `n_estimators=150, max_features=0.5, min_samples_leaf=1` | `data/tables/ml_04_*`, coordinate figures |
| ML_05 | integration | joins the three tasks on `(SOURCE_PARTITION, ROW_ID)`, re-checks targets, recomputes metrics, adds a hierarchical building/floor analysis | `data/tables/ml_05_*`, `data/figures/ml_05_*` |

In every task notebook, the held-out file is opened only after the configuration is frozen.

## 5. Verified AML results

These values come from `data/tables/ml_05_aml_summary_metrics.csv` and `ml_05_aml_hierarchical_error_breakdown.csv`. They match the README.

| Metric | Internal validation (3,852 rows) | Held-out (1,111 rows) |
|---|---:|---:|
| Building accuracy / macro-F1 | 99.95% / 99.95% | **99.73% / 99.71%** |
| Floor accuracy / macro-F1 | 96.42% / 96.78% | **88.93% / 87.27%** |
| Joint building + floor correct | 96.42% | **88.75%** (986 / 1,111) |
| Coordinate MAE, longitude axis | 5.91 | **8.10** |
| Coordinate MAE, latitude axis | 5.06 | **6.81** |
| Mean / median Euclidean error | 8.58 / 6.31 | **11.58 / 7.49** |

The weakest held-out cells are building 1 floor 1 (65.7% floor accuracy), building 2 floor 2 (68.5%) and building 2 floor 4 (69.2%). Rows with the wrong floor also have about twice the coordinate error (held-out mean 23.27 vs 10.00 when both building and floor are right).

### About coordinate units

The coordinate errors above are in the **dataset's own coordinate units**. ML_04 labels them as EPSG:3857 (Web/Pseudo-Mercator) metres, citing published UJIIndoorLoc work, but the repository has no authoritative CRS metadata and the UCI variable table does not state a CRS. Even if EPSG:3857 is correct, its units are projected map-grid units. At about 40° N they overstate ground distance by roughly 1/cos(40°) ≈ 1.3×. So do **not** read these values as verified ground metres. The DL outputs use the label `dataset coordinate units`.

## 6. Phase 2 (deep learning)

The project specification asks for a **shared-trunk multitask neural network** with building and floor classification heads and a coordinate regression head. It must be compared fairly with the AML baseline using floor confusion matrices, coordinate error maps and hierarchical error analysis. The specification PDF (`AML-DL roject 9.pdf`) was not present in the cloud environment where this documentation was written, so this summary follows the requirement as described in the task brief.

The DL implementation (`src/signal2space_dl/`, `Deep Learning Pipeline/DL_01`–`DL_05`) is documented in [`docs/DL_PIPELINE.md`](DL_PIPELINE.md). For a fair comparison it:

* reuses the AML internal split and the identical 1,046-feature table (it adds only a train-fitted standardization of the 6 summary columns, which the network needs);
* uses the same held-out rows, joins on `(SOURCE_PARTITION, ROW_ID)` and uses the same metric definitions;
* selects hyperparameters on internal validation only.

## 7. Team workflow

Work is split into dependency-ordered steps S0–S5, each tracked by a GitHub issue and delivered by one pull request. See [`CONTRIBUTING.md`](../CONTRIBUTING.md).
