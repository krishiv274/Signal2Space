"""Signal2Space deep learning (Phase 2) package.

The package trains and evaluates a shared-trunk multitask neural network that
predicts building, floor and two-dimensional coordinates from Wi-Fi RSSI
fingerprints. It reuses the AML feature contract written by
``Machine Learning Pipeline/ML_01_data_preparation_features.ipynb`` so that the
AML and DL results are compared on identical rows and partitions.

Modules
-------
config    Paths, column contract, seeds and default hyperparameters.
data      Load/validate AML feature tables and fit train-only DL transforms.
model     Shared-trunk multitask network and loss.
train     Training loop, early stopping and internal-validation model selection.
evaluate  Metrics, confusion matrices, hierarchical and robustness analysis.
plots     Figures written to ``data/figures/``.
pipeline  End-to-end command-line entry point.
"""

__version__ = "0.1.0"
