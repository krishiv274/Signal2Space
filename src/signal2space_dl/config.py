"""Paths, dataset constants and the frozen DL configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root (folder containing data/raw and src/signal2space_dl)."""
    candidate = (Path.cwd() if start is None else Path(start)).resolve()
    for parent in (candidate, *candidate.parents):
        if (parent / "data" / "raw").is_dir() and (parent / "src" / "signal2space_dl").is_dir():
            return parent
    raise FileNotFoundError(
        "Could not locate the Signal2Space repository root (expected data/raw/ and src/signal2space_dl/)."
    )


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def train_csv(self) -> Path:
        return self.raw / "trainingData.csv"

    @property
    def heldout_csv(self) -> Path:
        return self.raw / "validationData.csv"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def tables(self) -> Path:
        return self.root / "data" / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "data" / "figures"

    @property
    def models(self) -> Path:
        return self.root / "models"

    def ensure(self) -> "Paths":
        for p in (self.processed, self.tables, self.figures, self.models):
            p.mkdir(parents=True, exist_ok=True)
        return self


# ---- Dataset constants (verified against the supplied CSVs by data.validate_schema) ----
N_WAPS = 520
WAP_COLUMNS = [f"WAP{i:03d}" for i in range(1, N_WAPS + 1)]
TARGET_COLUMNS = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID"]
METADATA_COLUMNS = ["SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]
RAW_COLUMNS = WAP_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS
MISSING_SIGNAL = 100  # UJIIndoorLoc sentinel: access point not detected

# ---- Split: identical to ML_01 so DL and AML share the same internal partitions ----
GROUP_COLUMNS = ["BUILDINGID", "FLOOR", "SPACEID"]
GROUP_TEST_SIZE = 0.20
GROUP_RANDOM_STATE = 42

# Coordinates are reported in dataset units. ML_04 labels them EPSG:3857 metres; the
# repository holds no authoritative CRS metadata, so DL outputs keep that caveat.
COORDINATE_UNIT_LABEL = "dataset coordinate units (EPSG:3857 metres per ML_04; CRS unconfirmed)"

TASKS = ("building", "floor", "coordinate")


@dataclass(frozen=True)
class TrainConfig:
    """Frozen hyperparameters. Chosen a priori and checked only on internal validation.

    The held-out partition (validationData.csv) is never used to choose any value here.
    """

    tasks: tuple[str, ...] = TASKS
    hidden_sizes: tuple[int, ...] = (512, 256)
    head_hidden: int = 128
    input_dropout: float = 0.20      # randomly hides inputs; mimics APs that go undetected
    dropout: float = 0.30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 150
    patience: int = 15               # early-stopping patience on validation loss
    lr_patience: int = 5             # ReduceLROnPlateau patience
    lr_factor: float = 0.5
    min_delta: float = 1e-4
    # Multitask loss: L = wb*CE(building) + wf*CE(floor) + wc*Huber(coord / coord_scale)
    loss_weights: dict = field(default_factory=lambda: {"building": 1.0, "floor": 1.0, "coordinate": 1.0})
    coord_scale: float = 10.0        # 1 loss unit ~ 10 coordinate units of error
    huber_delta: float = 1.0
    class_weighting: bool = False    # floor/building classes are moderately imbalanced; off by default
    seed: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tasks"] = list(self.tasks)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d


# Experiments run by the pipeline: three single-task ablations and the multitask model.
EXPERIMENTS = {
    "building_only": ("building",),
    "floor_only": ("floor",),
    "coordinate_only": ("coordinate",),
    "multitask": TASKS,
}

# Notebook number that owns each experiment's artifacts (dl_0N_*).
EXPERIMENT_PREFIX = {
    "building_only": "dl_02_building",
    "floor_only": "dl_03_floor",
    "coordinate_only": "dl_04_coordinate",
    "multitask": "dl_05_multitask",
}
