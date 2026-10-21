# Signal2Space

### Indoor Localization Using Wi-Fi Signals

Signal2Space is a machine learning project focused on predicting indoor location using Wi-Fi signal strength measurements.

The project aims to predict:

* **Building**
* **Floor**
* **2D Position**

The project will explore both traditional machine learning approaches and deep multitask learning for indoor localization.

- **AML** — traditional machine learning evaluation.
- **DL** — multitask deep learning evaluation.

Both phases use the same Python environment defined in `requirements.txt`. The project data is organized under `data/raw/` and `data/processed/`.

## Project layout

```text
Signal2Space/
├── Machine Learning Pipeline/
│   ├── ML_01_data_preparation_features.ipynb
│   ├── ML_02_building_classification.ipynb
│   ├── ML_03_floor_classification.ipynb
│   ├── ML_04_coordinate_regression.ipynb
│   └── ML_05_integration_evaluation.ipynb
├── Deep Learning Pipeline/
│   ├── DL_01_data_preparation.ipynb
│   ├── DL_02_building_prediction.ipynb
│   ├── DL_03_floor_prediction.ipynb
│   ├── DL_04_coordinate_prediction.ipynb
│   └── DL_05_multitask_integration_evaluation.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── csv/
│   ├── csv_files/
├── README.md
└── requirements.txt
```

## Team pipelines

The numbered notebooks provide one pipeline per group member in each phase: data preparation and features; building prediction; floor prediction; coordinate prediction; and integration and evaluation. Members should use a shared train-test split and preprocessing definition for fair comparisons.

The AML notebooks contain the currently available building and floor work. The other phase notebooks are empty placeholders for the corresponding pipeline work. No DL implementation is included yet.

## Data files

Place the supplied UJIIndoorLoc files in `data/raw/`. Processed, reproducible data exports belong in `data/processed/`.

## Environment

Install the shared project packages with:

```bash
python -m pip install -r requirements.txt
```
