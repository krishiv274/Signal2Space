# Contributing to Signal2Space

All changes reach `main` through **pull requests**. Nobody commits directly to `main`.

## Work plan

Phase 2 work is split into dependency-ordered steps. Each step has a GitHub issue titled `S<n> …` and is delivered by one pull request that says `Closes #<issue>`.

| Step | Issue | Owner (GitHub) | Delivers | Starts when |
|---|---|---|---|---|
| S0 | [#15](https://github.com/krishiv274/Signal2Space/issues/15) | Krishiv (`krishiv274`) | project/AML docs, contributing guide, DL package skeleton, dependency update, `main` protection | now |
| S1 | [#16](https://github.com/krishiv274/Signal2Space/issues/16) | Gargi (`gigabyte2024`) | `data.py`, DL_01 notebook and outputs, data tests | S0 merged |
| S2 | [#17](https://github.com/krishiv274/Signal2Space/issues/17) | Bhawana (`bhawanaydv`) | `evaluate.py`, `plots.py`, evaluation tests | S0 merged (parallel with S1) |
| S3 | [#18](https://github.com/krishiv274/Signal2Space/issues/18) | Agrima (`slayplayagrima`) | `model.py`, `train.py`, model tests | S1 merged |
| S4 | [#19](https://github.com/krishiv274/Signal2Space/issues/19) | Abhiman (`Koray-afk`) | `pipeline.py`, DL_02–DL_04 single-task notebooks and outputs | S2 and S3 merged |
| S5 | [#20](https://github.com/krishiv274/Signal2Space/issues/20) | Krishiv (`krishiv274`) | `multitask.py`, DL_05 notebook, models, results, DL docs, README | S4 merged |
| S6 | [#21](https://github.com/krishiv274/Signal2Space/issues/21) | all five members | final review of docs and results | S0–S5 merged |

Do not start a step until every step it depends on is **merged** into `main`.

## Pull-request workflow

Only repository collaborators can push branches to `krishiv274/Signal2Space`. Everyone else works from a **fork**: fork the repository on GitHub, push the branch to the fork, and open a PR from the fork into `krishiv274/Signal2Space:main`.

```bash
git checkout main
git pull origin main          # or: git pull upstream main   (fork workflow)
git checkout -b s<n>-<short-name>
# copy/edit files
git add <files>
git commit -m "S<n>: <what changed>"
git push -u origin s<n>-<short-name>
```

Then open a PR into `main` with `Closes #<issue>` in the description.

## Conventions

* Generated tables go to `data/tables/`, figures to `data/figures/`, DL model files to `models/dl/`. Prefix them with the notebook ID (`ml_0x_…`, `dl_0x_…`).
* Never fit anything (scalers, encoders, models) on `validationData.csv`. Choose configurations using internal validation only.
* Report coordinate errors in **dataset coordinate units** unless the coordinate reference system has been verified.
* Optional local checks: `PYTHONPATH=src python -m pytest -q tests`. Maintainers also run these centrally on every PR.
