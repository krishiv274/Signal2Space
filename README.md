# Signal2Space

## Indoor positioning with Wi-Fi signals

Signal2Space predicts a device's building, floor, and two-dimensional position from Wi-Fi access point signal strength. Phase 1 is an AML pipeline: shared feature preparation, Random Forest classification and coordinate regression, and integrated evaluation. Phase 2 is a PyTorch deep-learning pipeline: a shared-trunk multitask network plus three single-task ablations, evaluated on the same held-out partition as AML and compared with it directly.

## Current status

- **Machine Learning:** Implemented and rerun end to end. Processed datasets, model result tables, and figures are kept separately under `data/processed/`, `data/tables/`, and `data/figures/`.
- **Current held-out results:** Building accuracy 99.73%; floor accuracy 88.93%; coordinate MAE 8.10 longitude-axis units and 6.81 latitude-axis units; mean Euclidean error 11.58 coordinate units. The coordinate notebook labels these units as EPSG:3857 metres, but the repository does not include authoritative CRS metadata; confirm the coordinate reference before treating that label as formally established.
- **Interpretation:** Results use the supplied UJIIndoorLoc train/validation partition and an internal group split. Fingerprint overlap and repeated coordinate values may occur across partitions, so these scores do not establish performance on unseen fingerprints or unseen positions.
- **Deep learning:** Implemented in `src/signal2space_dl/` and `Deep Learning Pipeline/DL_01`–`DL_05`, and run end to end (seed 42, CPU). The multitask model's held-out results are building 99.91%, floor 89.56% and mean Euclidean error 12.00. The coordinate-only DL model has a mean error of 10.51 against AML's 11.58. See [Deep learning pipeline](#deep-learning-pipeline-phase-2) and [`docs/DL_PIPELINE.md`](docs/DL_PIPELINE.md).

The dataset contains 19,937 training/reference records and 1,111 validation/test records. It has 520 WAP signal fields; `100` means the access point was not detected. [UCI UJIIndoorLoc dataset documentation](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)

## Project Structure

```mermaid
flowchart TB
    A["Signal2Space/"]

    A --> B["Machine Learning Pipeline/"]
    B --> B1["ML_01_data_preparation_features.ipynb"]
    B1 --> B2["ML_02_building_classification.ipynb"]
    B2 --> B3["ML_03_floor_classification.ipynb"]
    B3 --> B4["ML_04_coordinate_regression.ipynb"]
    B4 --> B5["ML_05_integration_evaluation.ipynb"]

    A --> C["Deep Learning Pipeline/<br/>PyTorch multitask pipeline"]
    C --> C1["DL_01_data_preparation.ipynb"]
    C1 --> C2["DL_02_building_prediction.ipynb"]
    C2 --> C3["DL_03_floor_prediction.ipynb"]
    C3 --> C4["DL_04_coordinate_prediction.ipynb"]
    C4 --> C5["DL_05_multitask_integration_evaluation.ipynb"]

    A --> D["data/"]
    D --> D1["raw/"]
    D1 --> D11["trainingData.csv"]
    D1 --> D12["validationData.csv"]

    D --> D2["processed/"]
    D2 --> D21["Canonical train/validation feature tables"]

    D --> D3["tables/"]
    D3 --> D31["Predictions, metrics, confusion data"]

    D --> D4["figures/"]
    D4 --> D41["EDA and report figures"]

    A --> E["README.md"]
    A --> F["requirements.txt<br/>AML/Runtime Lock"]
    A --> G["requirements-dl.txt<br/>AML lock + torch"]
    A --> H["src/signal2space_dl/<br/>DL package"]
    A --> I["scripts/run_dl_pipeline.py<br/>DL command line"]
    A --> J["models/<br/>Frozen DL checkpoints"]
    A --> K["docs/<br/>DL reference, team plan, viva scripts"]
    A --> L["tests/<br/>DL unit tests"]
    A --> M["handoff/<br/>Member handoff ZIPs + project handoff PDF"]

    class A root
    class B,B1,B2,B3,B4,B5 ml
    class C,C1,C2,C3,C4,C5 dl
    class D,D1,D2,D3,D4 data
    class D11,D12,D21,D31,D41,E,F files
```

## Machine Learning Flow

This diagram shows the data and model flow, not the notebook sequence.

```mermaid
flowchart TD
    A[trainingData.csv] --> B[Validate schema and inspect data quality]
    B --> C[Group split by building, floor, and space<br/>80 percent train / 20 percent internal validation]
    C --> D[Internal training split]
    C --> E[Internal validation split]
    F[validationData.csv<br/>supplied held-out test partition] --> I[Transform each split with the fitted scaler<br/>RSSI, detection indicators, signal summaries]
    D --> G[Fit RSSI scaler on internal training only]
    D --> I
    E --> I
    G --> I
    I --> J[Internal-training features]
    I --> K[Internal-validation features]
    I --> L[Held-out features]
    J --> M[Fit candidate Random Forest models<br/>building, floor, coordinate regression]
    M --> N[Select hyperparameters using internal validation]
    K --> N
    N --> O[Freeze selected configurations]
    O --> P[Evaluate frozen models on held-out features]
    L --> P
    P --> Q[Join predictions by source partition and row ID]
    Q --> R[Accuracy, macro F1, coordinate MAE]
    Q --> S[Confusion matrices, coordinate errors,<br/>hierarchical building/floor analysis]
```

## ML notebook walkthrough

The notebooks run in numeric order. ML_01 creates the shared, leakage-aware data preparation used by the three task notebooks. ML_02 through ML_04 tune and evaluate separate models for the three targets. ML_05 checks and combines their predictions. The supplied `validationData.csv` remains held out until each model configuration is selected using the internal validation split.

### ML_01 — Prepare data and engineer features

`ML_01_data_preparation_features.ipynb` loads the supplied training and validation CSVs, checks their schema and data quality, and explores the signal and target fields. It treats RSSI value `100` as an undetected access point. The training partition is divided into internal training and validation groups using building, floor, and space, so records for a group stay together. The supplied validation partition is preserved for final evaluation.

The notebook fits RSSI scaling values on internal training data only, then applies that fitted transformation to all splits. It creates 520 scaled RSSI features, 520 signal-presence indicators, and six signal-summary features: number detected, strongest, mean and standard deviation of observed RSSI, number above the strong-signal threshold, and strongest access-point index. Target and identifying metadata are retained for evaluation but excluded from model predictors. The notebook writes the internal-training, internal-validation, and held-out feature tables to `data/processed/` and saves EDA plots to `data/figures/`.

### ML_02 — Classify the building

`ML_02_building_classification.ipynb` reads the three shared feature tables and trains a Random Forest building classifier. It selects its configuration using the internal validation split, then evaluates the selected configuration once on the supplied held-out partition. It excludes target and metadata columns from predictors. Predictions, metrics, tuning results, and confusion-matrix data are saved under `data/tables/`; the held-out confusion-matrix figure is saved under `data/figures/`.

### ML_03 — Classify the floor

`ML_03_floor_classification.ipynb` uses the same prepared features and partitions to train a Random Forest floor classifier. Floor classes are encoded from the internal-training labels and that mapping is used consistently for validation and held-out predictions. Hyperparameter selection uses internal validation; final scores and the floor confusion matrix use the held-out partition. Prediction and evaluation tables are saved under `data/tables/`, and the confusion-matrix figure under `data/figures/`.

### ML_04 — Regress the coordinates

`ML_04_coordinate_regression.ipynb` trains a multi-output Random Forest regressor for longitude and latitude using the shared features. Its predictor allowlist excludes the coordinate targets and identifying metadata. Candidate settings are compared on internal validation using Euclidean coordinate error, and the selected configuration is evaluated on the held-out partition. The notebook records per-axis MAE and Euclidean error, writes predictions and tuning/evaluation tables to `data/tables/`, and saves actual-versus-predicted, displacement, and error visualizations to `data/figures/`. The notebook describes coordinate errors in metres based on an EPSG:3857 assumption; the repository does not provide authoritative CRS metadata to independently confirm that unit interpretation.

### ML_05 — Integrate and evaluate the tasks

`ML_05_integration_evaluation.ipynb` reads the task predictions and source feature tables, aligns rows using source partition and row ID, checks target alignment, and recomputes the task metrics. It then reports the combined building-and-floor correctness and breaks coordinate error down by hierarchical classification outcome. Integrated predictions, summaries, confusion data, and hierarchy tables are written to `data/tables/`; combined confusion matrices, metric comparisons, coordinate errors, and hierarchical analysis figures are saved to `data/figures/`.

The held-out rows may share Wi-Fi fingerprints or repeated two-dimensional coordinates with other partitions. The results therefore describe performance under the supplied partition; evaluating unseen fingerprints or unseen positions would require a split designed specifically for that research question.

## Selected results

| Task | Held-out result |
|---|---:|
| Building classification | Accuracy **99.73%**; macro F1 **99.71%** |
| Floor classification | Accuracy **88.93%**; macro F1 **87.27%** |
| Coordinate regression | Longitude-axis MAE **8.10**; latitude-axis MAE **6.81**; mean Euclidean error **11.58** (notebook reports metres) |
| Joint building and floor correctness | **986 / 1,111** held-out rows (**88.75%**) |

These scores are specific to the supplied split. The dataset partition does not guarantee that fingerprints or two-dimensional coordinates are unique across train and evaluation data.

## Deep learning pipeline (Phase 2)

The DL pipeline reads the raw CSVs directly. It validates the schema and recreates **the same internal group split as ML_01**: the internal-validation `ROW_ID`s are verified to be identical. It fits all preprocessing on internal-train rows only and keeps `validationData.csv` held out until every checkpoint is frozen.

Each fingerprint becomes 1,040 inputs: 520 standardised RSSI values plus 520 presence flags. The undetected sentinel `100` is filled below the observed range, and the presence flag records that the access point was missing.

The network is a shared-trunk MLP (1040 → 512 → 256) with three heads:

- **building:** 3 classes
- **floor:** 5 classes
- **coordinate:** longitude and latitude

The loss is `CE(building) + CE(floor) + Huber(coordinates / 10)`. Training uses AdamW with early stopping on the internal-validation loss. Each single-task ablation uses the same network with one head.

| Notebook | Runs | Main outputs |
|---|---|---|
| `DL_01_data_preparation` | Schema checks, split, train-only preprocessor | `data/processed/dl_01_*`, `data/tables/dl_01_*` |
| `DL_02_building_prediction` | Building-only model | `models/dl_02_*`, `data/tables/dl_02_*`, `data/figures/dl_02_*` |
| `DL_03_floor_prediction` | Floor-only model | `models/dl_03_*`, `dl_03_*` tables and figures |
| `DL_04_coordinate_prediction` | Coordinate-only model | `models/dl_04_*`, `dl_04_*` tables and figures |
| `DL_05_multitask_integration_evaluation` | Multitask model, comparison with AML, bootstrap, hierarchy | `models/dl_05_*`, `dl_05_*` tables and figures |

### Held-out comparison (same 1,111 rows, same metrics)

| Held-out | AML Random Forest | DL single-task | DL multitask |
|---|---:|---:|---:|
| Building accuracy | 99.73% | 99.82% | 99.91% |
| Floor accuracy / macro F1 | 88.93% / 87.27% | 89.92% / 88.43% | 89.56% / 88.15% |
| Coordinate MAE, longitude / latitude | 8.10 / 6.81 | 7.24 / 6.11 | 8.62 / 6.72 |
| Mean / median Euclidean error | 11.58 / 7.49 | 10.51 / 8.13 | 12.00 / 9.53 |
| Joint building + floor correct | 986 / 1,111 | — | 995 / 1,111 |

The 95% confidence intervals come from a paired bootstrap over the same held-out rows (DL minus AML).

- **Coordinate-only DL vs AML:** mean error is lower by 1.08 units (CI [−1.78, −0.37]). Its median error is higher than AML's.
- **Building and floor:** the differences are within noise. For example, multitask floor accuracy is +0.63 pp with CI [−0.72, +1.89].
- **Multitask coordinates:** sharing the trunk made coordinate error worse than the coordinate-only model.

All models were frozen before held-out scoring, so the held-out data was not used to pick a model. Coordinates are in dataset units; the EPSG:3857 label is unconfirmed.

![DL vs AML held-out comparison](data/figures/dl_05_heldout_model_comparison.png)

### Run the DL pipeline

```bash
python -m pip install -r requirements-dl.txt
```

```bash
python -m pytest -q tests
```

```bash
python scripts/run_dl_pipeline.py all
```

`all` takes about 4 minutes on a 4-core CPU. Use `evaluate` to re-score the committed checkpoints without retraining. You can also run the notebooks DL_01 through DL_05 in order; each has `RETRAIN = False` by default, so it reuses the checkpoints.

### Team handoff

The team plan, issues, branches and five member handoff ZIPs are listed in [`docs/TEAM_PLAN.md`](docs/TEAM_PLAN.md). The viva recording scripts are in [`docs/VIVA_SCRIPTS.md`](docs/VIVA_SCRIPTS.md).

## Report figures

### Held-out building and floor confusion matrices

![Held-out building and floor confusion matrices](data/figures/ml_05_heldout_confusion_matrices.png)

### Coordinate error distribution and spatial error

![Coordinate error distribution and spatial coordinate error](data/figures/ml_05_coordinate_error_visualizations.png)

### Hierarchical building/floor correctness

![Hierarchical correctness breakdown](data/figures/ml_05_hierarchical_error_breakdown.png)

## Data files

Place the supplied `trainingData.csv` and `validationData.csv` files in `data/raw/`.

- `data/processed/`: the three canonical feature tables written by ML_01.
- `data/tables/`: task predictions, tuning results, metrics, confusion-matrix data, and integrated evaluation outputs written by ML_02 through ML_05.
- `data/figures/`: selected EDA, classification, coordinate, and integrated evaluation figures.

Run the AML notebooks in numeric order from `Machine Learning Pipeline/`. ML_01 prepares the shared splits and features; ML_02, ML_03, and ML_04 produce task-specific predictions; ML_05 integrates and checks those outputs.

## Environment

The lock targets the verified Python 3.14.2 environment. Create and activate a virtual environment, then install the pinned dependencies:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

The lock includes packages required by the active AML notebooks and Jupyter execution. XGBoost is not used.

For the DL pipeline, install `requirements-dl.txt`. It includes the unchanged AML lock (`-r requirements.txt`) and adds `torch==2.15.0` and `pytest==9.2.0`. The DL run was verified on Python 3.13 (Linux, CPU); Python 3.14 on macOS with torch is pending verification by the integration owner.

On Linux, install the CPU-only torch wheel first to avoid downloading about 3 GB of CUDA libraries:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```
