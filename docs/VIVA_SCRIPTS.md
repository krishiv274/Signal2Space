# Viva / recording scripts (5 minutes maximum each)

Every number below comes from the committed full run (`data/tables/dl_05_model_comparison.csv`). If DL-5 changes any number, update the script before recording. Record after your PR is merged.

Placeholders in [square brackets] are results from your own DL-1 to DL-4 work. Fill them in before recording. They are not available yet.

## Krishiv (@krishiv274): Project overview, DL architecture, integration and final AML comparison
**Total planned length: 4:55** (limit 5:00). Square brackets mean: fill in from your own results before recording.

| Time | Section | What to say | What to show on screen |
|---|---|---|---|
| 0:00–0:40 (40 s) | Problem and data | Signal2Space predicts building, floor and position from Wi-Fi fingerprints in UJIIndoorLoc: 520 access points, where 100 means not detected. We have about 20,000 training fingerprints and 1,111 held-out ones. | README top section; `data/figures/ml_01_coordinate_labels_eda.png` |
| 0:40–1:20 (40 s) | Phase 1 recap | Phase 1 was Random Forests with a group split. On held-out data they reached 99.73% building, 88.93% floor and 11.58 units mean coordinate error. | `data/tables/ml_05_aml_summary_metrics.csv`; README results table |
| 1:20–2:30 (70 s) | DL architecture | Each fingerprint becomes 1,040 inputs: 520 standardised strengths and 520 'was it heard' flags. A shared two-layer trunk feeds three heads. The loss adds two cross-entropies and a Huber loss on coordinates divided by 10, which keeps the three terms on a similar scale. | `src/signal2space_dl/model.py` (MultitaskMLP, MultitaskLoss); `docs/DL_PIPELINE.md` §4 diagram |
| 2:30–3:30 (60 s) | Integration and discipline | My job was integration. DL uses exactly the AML split; we verify the row IDs match. Every model is frozen on internal validation, and evaluation refuses to run if the checkpoint hash changed. | `evaluate.py::evaluate_experiment` (SHA check); `dl_01_split_parity.json`; GitHub PR for DL-0 and the tracker issue |
| 3:30–4:30 (60 s) | Results vs AML | Multitask: 99.91% building, 89.56% floor, 995 of 1,111 rows with both correct, against 986 for AML. The coordinate-only network is the only resolved win: mean error 10.51 against 11.58. Sharing the trunk hurt coordinates. | `DL_05` notebook comparison table; `dl_05_heldout_model_comparison.png`; `dl_05_vs_aml_paired_bootstrap.csv` |
| 4:30–4:55 (25 s) | Limits and close | These hold for the supplied split only. The CRS is unconfirmed and we used one seed. Next steps are a multi-seed run and tuning the coordinate weight on internal validation. | `docs/DL_PIPELINE.md` §9 |

**One-line summary of your contribution:** Review, verify and merge the integration PR {{PR}}, which holds the implemented DL pipeline (package, notebooks, full-run outputs, docs and handoff). Verify it on the team's macOS / Python 3.14 environment and preserve existing local work. It connects to the full project as one stage of a single pipeline: DL_01 data, then DL_02–DL_04 heads, then DL_05 comparison with AML.

**Recording tips:** use 1080p screen capture and zoom the editor to about 125%. Open every file and tab before you start. Rehearse once with a timer and cut words rather than talking faster.


## Gargi (@gigabyte2024): Data preparation, leakage controls and the data-audit tests
**Total planned length: 4:40** (limit 5:00). Square brackets mean: fill in from your own results before recording.

| Time | Section | What to say | What to show on screen |
|---|---|---|---|
| 0:00–0:35 (35 s) | My role | I own DL_01: turning the raw CSVs into network inputs without leaking held-out information. | DL-1 issue on GitHub; `DL_01_data_preparation.ipynb` title cell |
| 0:35–1:35 (60 s) | Schema and missing signals | We validate all 529 columns. About 96.5% of signal values are 100, which means not detected. We fill those below the weakest real reading and add a presence flag, so the network can tell 'absent' from 'weak'. | DL_01 section 1 output (data-quality table); `data.py::validate_schema`, `Preprocessor.transform` |
| 1:35–2:40 (65 s) | Split and leakage control | We group by building, floor and space, so a reference space never sits in both internal train and validation. The split is identical to AML's, row for row. Every statistic is fitted on the 16,085 internal-train rows only. | DL_01 section 2 (split summary, parity = true); `data.py::make_group_split` |
| 2:40–3:50 (70 s) | Audit tests | My tests run on the real files and fail if any of these guarantees break: row counts, no shared groups, matching row IDs, and fitted parameters untouched by held-out data. | `tests/test_dl_data_real.py`, run with `pytest -q`; your DL-1 PR |
| 3:50–4:40 (50 s) | Findings and link to the project | I also counted identical fingerprints across partitions [quote your number]. That is why our results describe this split, not unseen locations. Every later notebook relies on this preparation. | `dl_01_fingerprint_overlap.csv`; `dl_01_input_representation.png` |

**One-line summary of your contribution:** Prove, with automated tests on the real CSVs, that the DL split and preprocessing are leakage-free and match AML. Document the audit. It connects to the full project as one stage of a single pipeline: DL_01 data, then DL_02–DL_04 heads, then DL_05 comparison with AML.

**Recording tips:** use 1080p screen capture and zoom the editor to about 125%. Open every file and tab before you start. Rehearse once with a timer and cut words rather than talking faster.


## Agrima (@slayplayagrima): Building classification head and robustness to missing access points
**Total planned length: 4:35** (limit 5:00). Square brackets mean: fill in from your own results before recording.

| Time | Section | What to say | What to show on screen |
|---|---|---|---|
| 0:00–0:35 (35 s) | My role | I own DL_02, the building head, and its robustness analysis. | DL-2 issue; `DL_02_building_prediction.ipynb` |
| 0:35–1:35 (60 s) | Model and training | It is the shared network with just the building head: three logits and cross-entropy. Early stopping on internal validation picked epoch 13 of 28. | `model.py`; `dl_02_building_training_curves.png`; `models/dl_02_building_model.json` |
| 1:35–2:35 (60 s) | Held-out result | 99.82% held-out accuracy against 99.73% for the Random Forest. That is one row, and the bootstrap interval includes zero, so building is essentially solved by both. | DL_02 metrics and AML comparison cells; `dl_02_building_heldout_confusion_matrix.png` |
| 2:35–3:15 (40 s) | Error review | Both errors are building 0 predicted as building 1. [Quote the phone and user IDs and WAP counts from your table.] | Your error-review table in DL_02 |
| 3:15–4:35 (80 s) | Robustness | I hid 10 to 50% of the detected access points on internal validation and measured accuracy. [Quote the curve.] This tells us how the model behaves when a phone hears fewer networks. | `dl_02_building_ap_masking_robustness.png`; `evaluate.py::mask_detected_aps`; your DL-2 PR |

**One-line summary of your contribution:** Explain the building model's remaining errors and measure how accuracy degrades when detected access points go missing. It connects to the full project as one stage of a single pipeline: DL_01 data, then DL_02–DL_04 heads, then DL_05 comparison with AML.

**Recording tips:** use 1080p screen capture and zoom the editor to about 125%. Open every file and tab before you start. Rehearse once with a timer and cut words rather than talking faster.


## Abhiman (@Koray-afk): Floor classification head, class imbalance and the weighting ablation
**Total planned length: 4:40** (limit 5:00). Square brackets mean: fill in from your own results before recording.

| Time | Section | What to say | What to show on screen |
|---|---|---|---|
| 0:00–0:35 (35 s) | My role | I own DL_03, floor prediction, the hardest classification task. | DL-3 issue; `DL_03_floor_prediction.ipynb` |
| 0:35–1:35 (60 s) | Model and the gap | The floor-only network reaches 97.4% on internal validation but 89.92% on held-out. AML shows the same gap, so the held-out data, recorded later with other phones, is genuinely harder. | `dl_03_floor_training_curves.png`; DL_03 metrics cell |
| 1:35–2:35 (60 s) | Where errors happen | Floor 4 exists only in building 2 and has the lowest recall, 74% on 39 rows. Many floor-1 rows are predicted as floor 2. [Quote your by-building table.] | `dl_03_floor_heldout_confusion_matrix.png`; `dl_03_floor_heldout_per_class.csv`; `dl_03_floor_errors_by_building.csv` |
| 2:35–3:50 (75 s) | Class-weighting ablation | I fixed the rule first: adopt weighting only if internal-validation macro F1 rises by at least 0.5 points. [State the result and decision.] Held-out was scored once afterwards. | Your DL-3 PR; weighted vs unweighted table in DL_03 |
| 3:50–4:40 (50 s) | Comparison and link | Unweighted DL floor is 89.92% against AML's 88.93%. The interval includes zero, so this is not a proven improvement. Floor errors also drive the larger coordinate errors in the hierarchy analysis. | `dl_05_hierarchical_breakdown.csv`; `dl_05_vs_aml_paired_bootstrap.csv` |

**One-line summary of your contribution:** Floor is the weakest classification task (held-out 89.9%; floor 4 recall 74%). Quantify where the errors happen and test class weighting with a decision rule fixed in advance. It connects to the full project as one stage of a single pipeline: DL_01 data, then DL_02–DL_04 heads, then DL_05 comparison with AML.

**Recording tips:** use 1080p screen capture and zoom the editor to about 125%. Open every file and tab before you start. Rehearse once with a timer and cut words rather than talking faster.


## Bhawana (@bhawanaydv): Coordinate regression head, error analysis and CRS/units
**Total planned length: 4:40** (limit 5:00). Square brackets mean: fill in from your own results before recording.

| Time | Section | What to say | What to show on screen |
|---|---|---|---|
| 0:00–0:35 (35 s) | My role | I own DL_04, coordinate regression, and the question of what the coordinate units really are. | DL-4 issue; `DL_04_coordinate_prediction.ipynb` |
| 0:35–1:35 (60 s) | Model and loss | The coordinate head outputs two numbers. Targets are centred and divided by 10, and the loss is Huber, so errors above about 10 units do not dominate the gradients. | `model.py::MultitaskLoss`; `dl_04_coordinate_training_curves.png` |
| 1:35–2:45 (70 s) | Result vs AML | The coordinate-only network gets a mean error of 10.51 against 11.58 for the Random Forest. The bootstrap interval excludes zero. The DL tail is better (p90 21.3 against 24.4), but the Random Forest has the lower median. | DL_04 comparison cell; `dl_05_heldout_error_cdf.png`; `dl_05_vs_aml_paired_bootstrap.csv` |
| 2:45–3:35 (50 s) | Where error happens | [Quote your by-building and by-floor table.] Rows whose floor is predicted wrong have roughly double the coordinate error. | `dl_04_coordinate_error_by_building_floor.csv`; `dl_04_coordinate_errors.png` |
| 3:35–4:40 (65 s) | CRS verification | Converting from Web-Mercator, the points land [state result]. Web-Mercator inflates distances by about 1/cos(latitude), so ground error is about 0.77 times the reported numbers. We report dataset units and label this as an estimate. | Your CRS note in DL_04 and `docs/DL_PIPELINE.md` §9; your DL-4 PR |

**One-line summary of your contribution:** Explain where coordinate error comes from and resolve the open question of coordinate units and CRS, without changing the reported metrics. It connects to the full project as one stage of a single pipeline: DL_01 data, then DL_02–DL_04 heads, then DL_05 comparison with AML.

**Recording tips:** use 1080p screen capture and zoom the editor to about 125%. Open every file and tab before you start. Rehearse once with a timer and cut words rather than talking faster.

