"""Signal2Space deep-learning pipeline.

A PyTorch multitask network that predicts building, floor and two-dimensional
coordinates from UJIIndoorLoc Wi-Fi fingerprints. The modules are:

- ``config``   : paths, constants and the frozen training configuration.
- ``data``     : schema validation, the AML-compatible group split and the
                 train-only RSSI preprocessor.
- ``model``    : the shared-trunk network and the multitask loss.
- ``train``    : seeded training with validation-based early stopping.
- ``evaluate`` : frozen-model prediction, metrics and the AML comparison.
- ``plots``    : figures written to ``data/figures``.
- ``cli``      : ``python scripts/run_dl_pipeline.py`` entry point.
"""

__version__ = "1.0.0"
