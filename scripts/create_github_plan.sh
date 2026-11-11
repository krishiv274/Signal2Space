#!/usr/bin/env bash
# Create the Signal2Space DL issues/labels/branches with the GitHub CLI.
#   bash scripts/create_github_plan.sh issues    # labels, tracker, DL-0..DL-6, sub-issues
#   bash scripts/create_github_plan.sh branches  # member branches from main (after DL-0 merges)
set -euo pipefail  # bash 3.2 compatible (macOS default shell)
REPO="krishiv274/Signal2Space"
PLAN="docs/github_plan"
MODE="${1:-}"
command -v gh >/dev/null || { echo "Install the GitHub CLI: https://cli.github.com"; exit 1; }

create_issue() {  # id title assignee labels bodyfile
  local id="$1" title="$2" who="$3" labels="$4" body="$5" url
  url=$(gh issue create -R "$REPO" --title "$title" --body-file "$body" --label "$labels" --assignee "$who" 2>/dev/null) || {
    echo "  ! could not assign $who (not a collaborator yet?) - creating unassigned" >&2
    url=$(gh issue create -R "$REPO" --title "$title" --body-file "$body" --label "$labels"); }
  echo "${url##*/}"
}

if [[ "$MODE" == "issues" ]]; then
  gh label create "deep-learning" -R "$REPO" --color 1d76db --force >/dev/null
  gh label create "tracker" -R "$REPO" --color 5319e7 --force >/dev/null
  gh label create "integration" -R "$REPO" --color 0e8a16 --force >/dev/null
  gh label create "data" -R "$REPO" --color c5def5 --force >/dev/null
  gh label create "building" -R "$REPO" --color bfdadc --force >/dev/null
  gh label create "floor" -R "$REPO" --color fef2c0 --force >/dev/null
  gh label create "coordinate" -R "$REPO" --color d4c5f9 --force >/dev/null
  gh label create "docs" -R "$REPO" --color 0075ca --force >/dev/null
  gh label create "viva" -R "$REPO" --color e99695 --force >/dev/null
  gh label create "step-1" -R "$REPO" --color ededed --force >/dev/null
  gh label create "step-2" -R "$REPO" --color ededed --force >/dev/null
  gh label create "step-3" -R "$REPO" --color ededed --force >/dev/null
  gh label create "step-4" -R "$REPO" --color ededed --force >/dev/null
  TMP=$(mktemp -d)
  cp "$PLAN/TRACKER.md" "$TMP/TRACKER.md"
  cp "$PLAN/DL-0.md" "$TMP/DL-0.md"
  cp "$PLAN/DL-1.md" "$TMP/DL-1.md"
  cp "$PLAN/DL-2.md" "$TMP/DL-2.md"
  cp "$PLAN/DL-3.md" "$TMP/DL-3.md"
  cp "$PLAN/DL-4.md" "$TMP/DL-4.md"
  cp "$PLAN/DL-5.md" "$TMP/DL-5.md"
  cp "$PLAN/DL-6.md" "$TMP/DL-6.md"
  TRACKER=$(gh issue create -R "$REPO" --title "DL Phase 2: multitask deep-learning pipeline, integration and handoff (tracker)" --body "Tracker - body is filled in below." --label "deep-learning,tracker" --assignee krishiv274 | sed "s#.*/##")
  echo "Tracker: #$TRACKER"
  sed -i.bak -e "s/{{TRACKER}}/#$TRACKER/g" "$TMP/DL-0.md"
  N_DL0=$(create_issue "DL-0" "[DL-0] Review, verify on macOS and merge the DL integration PR" "krishiv274" "deep-learning,integration,step-1" "$TMP/DL-0.md")
  echo "DL-0 -> #${N_DL0}"
  sed -i.bak -e "s/{{DL-0}}/#${N_DL0}/g" -e "s/{{TRACKER}}/#$TRACKER/g" "$TMP/DL-1.md"
  N_DL1=$(create_issue "DL-1" "[DL-1] Data & leakage audit: real-data tests for DL_01 split and preprocessing" "gigabyte2024" "deep-learning,data,step-2" "$TMP/DL-1.md")
  echo "DL-1 -> #${N_DL1}"
  sed -i.bak -e "s/{{DL-0}}/#${N_DL0}/g" -e "s/{{TRACKER}}/#$TRACKER/g" "$TMP/DL-2.md"
  N_DL2=$(create_issue "DL-2" "[DL-2] Building head: error review and missing-AP robustness (DL_02)" "slayplayagrima" "deep-learning,building,step-2" "$TMP/DL-2.md")
  echo "DL-2 -> #${N_DL2}"
  sed -i.bak -e "s/{{DL-0}}/#${N_DL0}/g" -e "s/{{TRACKER}}/#$TRACKER/g" "$TMP/DL-3.md"
  N_DL3=$(create_issue "DL-3" "[DL-3] Floor head: per-building errors and class-weighting ablation (DL_03)" "Koray-afk" "deep-learning,floor,step-2" "$TMP/DL-3.md")
  echo "DL-3 -> #${N_DL3}"
  sed -i.bak -e "s/{{DL-0}}/#${N_DL0}/g" -e "s/{{TRACKER}}/#$TRACKER/g" "$TMP/DL-4.md"
  N_DL4=$(create_issue "DL-4" "[DL-4] Coordinate head: error analysis and CRS/unit verification (DL_04)" "bhawanaydv" "deep-learning,coordinate,step-2" "$TMP/DL-4.md")
  echo "DL-4 -> #${N_DL4}"
  sed -i.bak -e "s/{{DL-1}}/#${N_DL1}/g" -e "s/{{DL-2}}/#${N_DL2}/g" -e "s/{{DL-3}}/#${N_DL3}/g" -e "s/{{DL-4}}/#${N_DL4}/g" -e "s/{{TRACKER}}/#$TRACKER/g" "$TMP/DL-5.md"
  N_DL5=$(create_issue "DL-5" "[DL-5] Final integration: re-run, reconcile results, update report/PPT, tag release" "krishiv274" "deep-learning,integration,docs,step-3" "$TMP/DL-5.md")
  echo "DL-5 -> #${N_DL5}"
  sed -i.bak -e "s/{{DL-5}}/#${N_DL5}/g" -e "s/{{TRACKER}}/#$TRACKER/g" "$TMP/DL-6.md"
  N_DL6=$(create_issue "DL-6" "[DL-6] Viva recordings (≤5 min each) for all five members" "krishiv274" "deep-learning,viva,step-4" "$TMP/DL-6.md")
  echo "DL-6 -> #${N_DL6}"
  sed -i.bak -e "s/{{DL-0}}/#${N_DL0}/g" -e "s/{{DL-1}}/#${N_DL1}/g" -e "s/{{DL-2}}/#${N_DL2}/g" -e "s/{{DL-3}}/#${N_DL3}/g" -e "s/{{DL-4}}/#${N_DL4}/g" -e "s/{{DL-5}}/#${N_DL5}/g" -e "s/{{DL-6}}/#${N_DL6}/g" "$TMP/TRACKER.md"
  gh issue edit "$TRACKER" -R "$REPO" --body-file "$TMP/TRACKER.md" >/dev/null
  sid=$(gh api "repos/$REPO/issues/${N_DL0}" --jq .id)
  gh api -X POST "repos/$REPO/issues/$TRACKER/sub_issues" -F sub_issue_id="$sid" >/dev/null || echo "  ! sub-issue link failed for DL-0"
  sid=$(gh api "repos/$REPO/issues/${N_DL1}" --jq .id)
  gh api -X POST "repos/$REPO/issues/$TRACKER/sub_issues" -F sub_issue_id="$sid" >/dev/null || echo "  ! sub-issue link failed for DL-1"
  sid=$(gh api "repos/$REPO/issues/${N_DL2}" --jq .id)
  gh api -X POST "repos/$REPO/issues/$TRACKER/sub_issues" -F sub_issue_id="$sid" >/dev/null || echo "  ! sub-issue link failed for DL-2"
  sid=$(gh api "repos/$REPO/issues/${N_DL3}" --jq .id)
  gh api -X POST "repos/$REPO/issues/$TRACKER/sub_issues" -F sub_issue_id="$sid" >/dev/null || echo "  ! sub-issue link failed for DL-3"
  sid=$(gh api "repos/$REPO/issues/${N_DL4}" --jq .id)
  gh api -X POST "repos/$REPO/issues/$TRACKER/sub_issues" -F sub_issue_id="$sid" >/dev/null || echo "  ! sub-issue link failed for DL-4"
  sid=$(gh api "repos/$REPO/issues/${N_DL5}" --jq .id)
  gh api -X POST "repos/$REPO/issues/$TRACKER/sub_issues" -F sub_issue_id="$sid" >/dev/null || echo "  ! sub-issue link failed for DL-5"
  sid=$(gh api "repos/$REPO/issues/${N_DL6}" --jq .id)
  gh api -X POST "repos/$REPO/issues/$TRACKER/sub_issues" -F sub_issue_id="$sid" >/dev/null || echo "  ! sub-issue link failed for DL-6"
  {
    echo "# Issue numbers"; echo; echo "| ID | Issue |"; echo "|---|---|"
    echo "| Tracker | #$TRACKER |"
    echo "| DL-0 | #${N_DL0} |"
    echo "| DL-1 | #${N_DL1} |"
    echo "| DL-2 | #${N_DL2} |"
    echo "| DL-3 | #${N_DL3} |"
    echo "| DL-4 | #${N_DL4} |"
    echo "| DL-5 | #${N_DL5} |"
    echo "| DL-6 | #${N_DL6} |"
  } > "$PLAN/ISSUE_NUMBERS.md"
  echo "Wrote $PLAN/ISSUE_NUMBERS.md - commit it and update the Notion planner Issue column."
elif [[ "$MODE" == "branches" ]]; then
  SHA=$(gh api "repos/$REPO/git/ref/heads/main" --jq .object.sha)
  gh api -X POST "repos/$REPO/git/refs" -f ref="refs/heads/dl/01-data-audit-gigabyte2024" -f sha="$SHA" >/dev/null && echo "created dl/01-data-audit-gigabyte2024" || echo "  ! dl/01-data-audit-gigabyte2024 exists or failed - left untouched"
  gh api -X POST "repos/$REPO/git/refs" -f ref="refs/heads/dl/02-building-robustness-slayplayagrima" -f sha="$SHA" >/dev/null && echo "created dl/02-building-robustness-slayplayagrima" || echo "  ! dl/02-building-robustness-slayplayagrima exists or failed - left untouched"
  gh api -X POST "repos/$REPO/git/refs" -f ref="refs/heads/dl/03-floor-imbalance-Koray-afk" -f sha="$SHA" >/dev/null && echo "created dl/03-floor-imbalance-Koray-afk" || echo "  ! dl/03-floor-imbalance-Koray-afk exists or failed - left untouched"
  gh api -X POST "repos/$REPO/git/refs" -f ref="refs/heads/dl/04-coordinate-crs-bhawanaydv" -f sha="$SHA" >/dev/null && echo "created dl/04-coordinate-crs-bhawanaydv" || echo "  ! dl/04-coordinate-crs-bhawanaydv exists or failed - left untouched"
  gh api -X POST "repos/$REPO/git/refs" -f ref="refs/heads/dl/05-integration-krishiv274" -f sha="$SHA" >/dev/null && echo "created dl/05-integration-krishiv274" || echo "  ! dl/05-integration-krishiv274 exists or failed - left untouched"
else
  echo "usage: bash scripts/create_github_plan.sh issues|branches"; exit 2
fi
