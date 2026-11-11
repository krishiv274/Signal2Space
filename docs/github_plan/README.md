# GitHub plan (ready to create)

These files are the exact issue bodies for the DL phase. They could not be created automatically: GitHub refused writes from the setup session with HTTP 403 (`Resource not accessible by integration`).

To create everything, run this from the repository root as @krishiv274. You need the [`gh` CLI](https://cli.github.com) and `gh auth login`.

```bash
bash scripts/create_github_plan.sh issues
```

That creates the labels, the tracker issue, DL-0 to DL-6 (placeholders such as `{{DL-1}}` become real `#` links) and the sub-issue links. It writes the mapping to `docs/github_plan/ISSUE_NUMBERS.md`.

After the integration PR (DL-0) is merged, create the member branches from `main`:

```bash
bash scripts/create_github_plan.sh branches
```

| ID | Owner | Branch | Title |
|---|---|---|---|
| DL-0 | Krishiv (@krishiv274) | `claude/peaceful-shannon-o1t6j8` | [DL-0] Review, verify on macOS and merge the DL integration PR |
| DL-1 | Gargi (@gigabyte2024) | `dl/01-data-audit-gigabyte2024` | [DL-1] Data & leakage audit: real-data tests for DL_01 split and preprocessing |
| DL-2 | Agrima (@slayplayagrima) | `dl/02-building-robustness-slayplayagrima` | [DL-2] Building head: error review and missing-AP robustness (DL_02) |
| DL-3 | Abhiman (@Koray-afk) | `dl/03-floor-imbalance-Koray-afk` | [DL-3] Floor head: per-building errors and class-weighting ablation (DL_03) |
| DL-4 | Bhawana (@bhawanaydv) | `dl/04-coordinate-crs-bhawanaydv` | [DL-4] Coordinate head: error analysis and CRS/unit verification (DL_04) |
| DL-5 | Krishiv (@krishiv274) | `dl/05-integration-krishiv274` | [DL-5] Final integration: re-run, reconcile results, update report/PPT, tag release |
| DL-6 | Krishiv (@krishiv274) | `(no branch; links are posted as issue comments)` | [DL-6] Viva recordings (≤5 min each) for all five members |
