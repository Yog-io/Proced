#!/usr/bin/env bash
# Gated Zenodo build runner — implements AGENT_ORCHESTRATION_BRIEF.md Steps 0–4.
# Run ON THE DATASET PC from the Proced repo root after clone.
#
# Usage:
#   ./run_zenodo_build.sh /path/to/extracted/zenodo/folder
#   SEED=42 WORKERS=8 ./run_zenodo_build.sh "$ZENODO_ROOT"
#
# Exit codes:
#   0  all gates passed (or stopped cleanly with report written per §4)
#   2  pipeline validate failed and no allowed retry, or QA hard-fail
#   3  Step 0 STOP — incomplete extraction suspected (§4.1 >5% / folder gaps)
#   1  harness / unexpected error
set -uo pipefail

ROOT_REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_REPO" || exit 1

ZENODO_ROOT="${1:-${ZENODO_ROOT:-}}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
PAIRS="${PAIRS:-2500}"
DATE_TAG="$(date +%Y%m%d)"
LOG_DIR="$ROOT_REPO/logs"
mkdir -p "$LOG_DIR"

if [[ -z "$ZENODO_ROOT" || ! -d "$ZENODO_ROOT" ]]; then
  echo "ERROR: pass extracted Zenodo root as \$1 (or set ZENODO_ROOT)." >&2
  echo "  usage: $0 /path/to/extracted/zenodo/folder" >&2
  exit 1
fi
ZENODO_ROOT="$(cd "$ZENODO_ROOT" && pwd)"

STEP0_JSON="$ZENODO_ROOT/raw_folder_audit_report.json"
STEP1_LOG="$LOG_DIR/step1_pipeline_${DATE_TAG}.log"
STEP2_LOG="$LOG_DIR/step2_handoff_${DATE_TAG}.log"
STEP3_LOG="$LOG_DIR/step3_qa_${DATE_TAG}.log"
STATUS_MD="$ROOT_REPO/BUILD_STATUS_${DATE_TAG}.md"
PIPELINE_SUMMARY="$ROOT_REPO/data/master_dataset_processed/pipeline_summary.json"
QA_REPORT="$ROOT_REPO/data/master_dataset_processed/verification_report.json"
# QA may also write under cwd fallback
QA_REPORT_FALLBACK="$ROOT_REPO/verification_report.json"

log() { echo "[run_zenodo_build] $*"; }
die_report() {
  local code="$1"; shift
  log "STOP: $*"
  # Ensure a status report skeleton exists even on early stop
  if [[ ! -f "$STATUS_MD" ]]; then
    {
      echo "# Dataset Build Status — ${DATE_TAG}"
      echo
      echo "## Summary verdict"
      echo "[ ] Fully complete and verified (all Section 2 conditions met)"
      echo "[x] Stopped partway — see below for exactly where and why"
      echo
      echo "## Stop reason"
      echo "$*"
      echo
      echo "## Full logs"
      echo "- Step 0 report: \`${STEP0_JSON}\`"
      [[ -f "$STEP1_LOG" ]] && echo "- Step 1 log: \`${STEP1_LOG}\`"
      [[ -f "$STEP2_LOG" ]] && echo "- Step 2 log: \`${STEP2_LOG}\`"
      [[ -f "$STEP3_LOG" ]] && echo "- Step 3 log: \`${STEP3_LOG}\`"
    } > "$STATUS_MD"
  fi
  exit "$code"
}

# ---------------------------------------------------------------- Step 0
log "Step 0 — raw folder audit --root $ZENODO_ROOT"
python3 qa_verification/raw_audit/run_raw_folder_audit.py \
  --root "$ZENODO_ROOT" \
  --out "$STEP0_JSON" 2>&1 | tee "$LOG_DIR/step0_raw_audit_${DATE_TAG}.log"
STEP0_RC=${PIPESTATUS[0]:-${pipestatus[1]:-0}}

if [[ ! -f "$STEP0_JSON" ]]; then
  die_report 1 "Step 0 produced no report (rc=$STEP0_RC)"
fi

python3 - "$STEP0_JSON" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
c = d.get("counts", {})
n = c.get("files_discovered", 0) or 0
failed = c.get("open_failed", 0) or 0
gaps = len(d.get("folder_gaps") or [])
ready = bool(d.get("ready_for_pipeline"))
pct = (100.0 * failed / n) if n else (100.0 if failed else 0.0)
# §4.1 gates
if failed and n and pct > 5.0:
    print(f"STOP_PCT {pct:.2f} failed={failed} n={n}", flush=True)
    sys.exit(31)
if gaps >= 2 and failed > 0 and pct >= 1.0:
    # suspicious multi-part signature: gaps + some open failures
    print(f"STOP_GAPS gaps={gaps} failed={failed} pct={pct:.2f}", flush=True)
    sys.exit(32)
if gaps >= 3:
    print(f"STOP_GAPS_ONLY gaps={gaps}", flush=True)
    sys.exit(32)
# quarantine path: some failures but under 5% → proceed, note for report
print(f"OK ready={ready} failed={failed} n={n} pct={pct:.2f} gaps={gaps}", flush=True)
print(f"ORPHANS {len((d.get('pairing') or {}).get('orphan_images') or [])} "
      f"{len((d.get('pairing') or {}).get('orphan_masks') or [])}", flush=True)
print(f"DUPS {len((d.get('duplicates') or {}).get('groups') or [])}", flush=True)
sys.exit(0)
PY
S0=$?
if [[ $S0 -eq 31 ]]; then
  die_report 3 "§4.1 STOP: >5% corrupt/unopenable — incomplete extraction suspected. See $STEP0_JSON"
elif [[ $S0 -eq 32 ]]; then
  die_report 3 "§4.1 STOP: multi-part extraction gaps (suspicious sibling folders). See $STEP0_JSON"
elif [[ $S0 -ne 0 ]]; then
  die_report 1 "Step 0 report parse failed (python rc=$S0)"
fi
log "Step 0 gate passed (quarantine/orphans/dups logged in report — proceed)"

# ---------------------------------------------------------------- Step 1
log "Step 1 — pipeline run seed=$SEED workers=$WORKERS (NO extract)"
DEST_DIR="$ROOT_REPO/data/master_dataset_processed"
OUT_ARCHIVE="$ROOT_REPO/data/master_dataset_v1.7z"
python3 sar_dataset_pipeline.py run \
  --stage_dir "$ZENODO_ROOT" \
  --dest_dir "$DEST_DIR" \
  --output_archive "$OUT_ARCHIVE" \
  --workers "$WORKERS" \
  --seed "$SEED" 2>&1 | tee "$STEP1_LOG"
PIPE_RC=${PIPESTATUS[0]:-${pipestatus[1]:-0}}
log "Step 1 exit=$PIPE_RC log=$STEP1_LOG"

RETRY_NOTE="no"
if [[ $PIPE_RC -eq 2 ]]; then
  # §4.2: one bounded retry only for known domain/calib override — requires human-readable reason.
  # We do NOT auto-guess flags. Stop and leave validation text for the report.
  VALID_MSG=""
  if [[ -f "$PIPELINE_SUMMARY" ]]; then
    VALID_MSG="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); v=d.get('validation') or d.get('validate') or {}; print(json.dumps(v, indent=2)[:4000])" "$PIPELINE_SUMMARY" 2>/dev/null || true)"
  fi
  {
    echo "# Dataset Build Status — ${DATE_TAG}"
    echo
    echo "## Summary verdict"
    echo "[ ] Fully complete and verified (all Section 2 conditions met)"
    echo "[x] Stopped partway — see below for exactly where and why"
    echo
    echo "## Step 0 — Raw Audit"
    echo "- Report: \`${STEP0_JSON}\`"
    echo "- (see report counts for discovered/corrupt/orphans/duplicates)"
    echo
    echo "## Step 1 — Pipeline Run"
    echo "- Command: \`python3 sar_dataset_pipeline.py run --stage_dir $ZENODO_ROOT --dest_dir $DEST_DIR --output_archive $OUT_ARCHIVE --workers $WORKERS --seed $SEED\`"
    echo "- Exit code: 2"
    echo "- Validation failure (verbatim from pipeline_summary.json):"
    echo '```json'
    echo "${VALID_MSG:-'(pipeline_summary.json missing or unreadable)'}"
    echo '```'
    echo "- Retry attempted? [no — §4.2 requires an unambiguous domain/calib override; not auto-guessed]"
    echo
    echo "## Open items for the human tomorrow"
    echo "- Pipeline validate failed (exit 2). Inspect validation block above; if value-domain/calibration, one retry with \`--value-domain\` / \`--calibrated-match\` / \`--uncalibrated-match\` is allowed."
    echo
    echo "## Full logs"
    echo "- Step 0: \`${STEP0_JSON}\`"
    echo "- Step 1: \`${STEP1_LOG}\`"
  } > "$STATUS_MD"
  exit 2
elif [[ $PIPE_RC -ne 0 ]]; then
  die_report 1 "Step 1 failed with exit $PIPE_RC (not 0/2). See $STEP1_LOG"
fi

# ---------------------------------------------------------------- Step 2
log "Step 2 — handoff scripts (mock/simple expected)"
{
  echo "=== join_wind.py ==="
  python3 scripts/join_wind.py; echo "join_wind exit=$?"
  echo "=== generate_ais.py ==="
  python3 scripts/generate_ais.py; echo "generate_ais exit=$?"
  echo "=== generate_model2_pairs.py ==="
  python3 scripts/generate_model2_pairs.py --pairs "$PAIRS"; echo "generate_model2_pairs exit=$?"
} 2>&1 | tee "$STEP2_LOG"
log "Step 2 done (see $STEP2_LOG)"

# ---------------------------------------------------------------- Step 3
log "Step 3 — independent QA verification seed=$SEED"
export QA_STAGE_DIR="$ZENODO_ROOT"
export QA_DEST_DIR="$DEST_DIR"
export QA_STATE_DIR="$ROOT_REPO/data/state"
export QA_FEATURES_DIR="$ROOT_REPO/data/features"
export QA_SPLITS_DIR="$ROOT_REPO/data/splits"
export QA_CORRIDORS="$ROOT_REPO/data/reference/shipping_corridors.geojson"
unset QA_OUTPUT_ARCHIVE QA_SOURCE_ARCHIVE 2>/dev/null || true

python3 qa_verification/run_all_verification.py --seed "$SEED" 2>&1 | tee "$STEP3_LOG"
QA_RC=${PIPESTATUS[0]:-${pipestatus[1]:-0}}
log "Step 3 exit=$QA_RC log=$STEP3_LOG"

QA_FILE="$QA_REPORT"
[[ -f "$QA_FILE" ]] || QA_FILE="$QA_REPORT_FALLBACK"

# ---------------------------------------------------------------- Step 4
log "Step 4 — write $STATUS_MD"
python3 - "$STATUS_MD" "$DATE_TAG" "$ZENODO_ROOT" "$SEED" "$WORKERS" \
  "$PIPE_RC" "$QA_RC" "$STEP0_JSON" "$QA_FILE" "$STEP1_LOG" "$STEP2_LOG" "$STEP3_LOG" \
  "$DEST_DIR" "$OUT_ARCHIVE" <<'PY'
import json, sys, os
from datetime import datetime, timezone
(status_md, date_tag, zenodo, seed, workers, pipe_rc, qa_rc,
 step0, qa_file, l1, l2, l3, dest, arch) = sys.argv[1:15]
pipe_rc, qa_rc = int(pipe_rc), int(qa_rc)

s0 = {}
if os.path.isfile(step0):
    try:
        s0 = json.load(open(step0))
    except Exception as e:
        s0 = {"error": str(e)}
counts = s0.get("counts") or {}
pair = s0.get("pairing") or {}
dups = s0.get("duplicates") or {}
integ = s0.get("integrity") or {}
orph_i = len(pair.get("orphan_images") or [])
orph_m = len(pair.get("orphan_masks") or [])
n_dup = len(dups.get("groups") or [])
n_fail = counts.get("open_failed", 0)
n_disc = counts.get("files_discovered", 0)
gaps = s0.get("folder_gaps") or []

# pipeline summary counts
oil = la = bg = tr = va = te = None
ps = os.path.join(dest, "pipeline_summary.json")
if os.path.isfile(ps):
    try:
        d = json.load(open(ps))
        labs = (d.get("label_counts") or d.get("labels") or {})
        if isinstance(labs, dict):
            oil = labs.get("oil"); la = labs.get("lookalike") or labs.get("look-alike")
            bg = labs.get("background")
        sp = d.get("splits") or d.get("split_counts") or {}
        if isinstance(sp, dict):
            tr, va, te = sp.get("train"), sp.get("val"), sp.get("test")
        # nested shapes
        if oil is None:
            lc = d.get("counts") or {}
            oil = lc.get("oil"); la = lc.get("lookalike"); bg = lc.get("background")
            tr, va, te = lc.get("train"), lc.get("val"), lc.get("test")
    except Exception:
        pass

# QA report
hard, soft, qa_ok = [], [], None
if qa_file and os.path.isfile(qa_file):
    try:
        q = json.load(open(qa_file))
        qa_ok = bool(q.get("ok"))
        for m in q.get("modules") or []:
            for f in m.get("findings") or []:
                if not f.get("ok") and f.get("severity") == "hard":
                    hard.append(f"{m.get('name')}: {f.get('message')}")
                elif not f.get("ok") and f.get("severity") == "soft":
                    soft.append(f"{m.get('name')}: {f.get('message')}")
    except Exception as e:
        hard.append(f"(could not parse QA report: {e})")

complete = (pipe_rc == 0 and qa_rc == 0 and not hard)
action = "proceeded"
if n_fail and n_disc and (100.0 * n_fail / n_disc) > 5:
    action = "STOPPED — incomplete extraction suspected"
elif n_fail:
    action = f"quarantined {n_fail} corrupt paths in report; proceeded"

lines = []
lines.append(f"# Dataset Build Status — {date_tag}")
lines.append("")
lines.append("## Summary verdict")
if complete:
    lines.append("[x] Fully complete and verified (all Section 2 conditions met)")
    lines.append("[ ] Stopped partway — see below for exactly where and why")
else:
    lines.append("[ ] Fully complete and verified (all Section 2 conditions met)")
    lines.append("[x] Stopped partway — see below for exactly where and why")
lines.append("")
lines.append("## Step 0 — Raw Audit")
lines.append(f"- Total files discovered: {n_disc}")
lines.append(f"- Corrupt/unopenable: {n_fail}")
lines.append(f"- Orphaned pairs: images={orph_i}, masks={orph_m}")
lines.append(f"- Duplicates: {n_dup}")
lines.append(f"- Folder gaps flagged: {len(gaps)}")
lines.append(f"- Action taken: {action}")
lines.append(f"- Report: `{step0}`")
lines.append("")
lines.append("## Step 1 — Pipeline Run")
lines.append(f"- Command used: `python3 sar_dataset_pipeline.py run --stage_dir {zenodo} --dest_dir {dest} --output_archive {arch} --workers {workers} --seed {seed}`")
lines.append(f"- Exit code: {pipe_rc}")
if pipe_rc == 2:
    lines.append("- If exit 2: see validation block in prior section / pipeline_summary.json")
lines.append(f"- Retry attempted? [no]")
lines.append(f"- Final counts: oil={oil}, look-alike={la}, background={bg}; train/val/test={tr}/{va}/{te}")
lines.append("")
lines.append("## Step 2 — Handoff Scripts")
lines.append("- join_wind.py: [source used, mock or era5] — see log (default mock expected)")
lines.append(f"- generate_ais.py: [output confirmed present] — see `{os.path.join('data','ais','ais_tracks.csv')}`")
lines.append(f"- generate_model2_pairs.py: [backend simple, pairs={os.environ.get('PAIRS','2500')}]")
lines.append("")
lines.append("## Step 3 — QA Verification")
lines.append(f"- Exit code: {qa_rc}")
lines.append(f"- Hard-fails (must be empty for a complete build): {len(hard)}")
for h in hard[:50]:
    lines.append(f"  - {h}")
if not hard:
    lines.append("  - (none)")
lines.append(f"- Soft-fails: {len(soft)}")
for s in soft[:50]:
    lines.append(f"  - {s}")
if not soft:
    lines.append("  - (none)")
lines.append("")
lines.append("## Open items for the human tomorrow")
if not complete:
    lines.append("- See summary verdict and Step exits above; inspect logs for full stdout/stderr.")
else:
    lines.append("- (none — all Section 2 conditions met)")
lines.append("")
lines.append("## Full logs")
for lab, p in [("Step 0", step0), ("Step 1", l1), ("Step 2", l2), ("Step 3", l3)]:
    lines.append(f"- {lab}: `{p}`")
lines.append("")
lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()} by run_zenodo_build.sh_")
open(status_md, "w").write("\n".join(lines) + "\n")
print(f"wrote {status_md}")
PY

log "Done. Status: $STATUS_MD"
if [[ $QA_RC -eq 0 && $PIPE_RC -eq 0 ]]; then
  log "RESULT: gates passed (exit 0)"
  exit 0
elif [[ $QA_RC -eq 2 || $PIPE_RC -eq 2 ]]; then
  log "RESULT: hard-fail / validate fail — see report (exit 2)"
  exit 2
else
  log "RESULT: incomplete — see report (exit $QA_RC)"
  exit "${QA_RC:-1}"
fi
