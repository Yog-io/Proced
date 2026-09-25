# Agent Orchestration Brief — Working Notes

> Living notes for the overnight Zenodo build. Update as steps complete.
> Source of truth: `AGENT_ORCHESTRATION_BRIEF.md` (read fully before starting).

---

## Environment / topology (confirmed)

| Machine | Role | Has |
|---|---|---|
| **This machine (builder PC)** | Code, tests, QA suite, orchestration prep | `proced/`, `qa_verification/`, `scripts/`, `tests/` — **no extracted Zenodo tree** |
| **Dataset PC** | Runs processing against real data | Extracted Zenodo multi-part `.7z` tree |

**Plan (confirmed):** clone/copy this `Proced` repo onto the dataset PC; the absolute path of the extracted tree is supplied as an argument when running (not hard-coded here).

```bash
# On dataset PC, after clone:
export ZENODO_ROOT=/path/to/extracted/zenodo/folder   # set when known
# Recommended output root (mirrors Zenodo layout inside a new folder):
export DEST_DIR=/path/to/Zenodo-Dataset_final
./run_zenodo_build.sh "$ZENODO_ROOT"
```

- Pipeline + raw_audit + handoff scripts + `run_all_verification.py` all execute **on the dataset PC**, pointed at `$ZENODO_ROOT`.
- Builder PC: develop/verify tools (111/111 green) and review `BUILD_STATUS_*.md` / logs after the run.
- Brief placeholder: `<path/to/extracted/zenodo/folder>` = `$ZENODO_ROOT` on the dataset PC.
- **Never run `extract`** — data is already extracted on the dataset PC.
- Scope: **Zenodo only** (no Kaggle/OSD this run).
- **Asymmetric layout supported:** sibling `*_images` / `*_mask` trees (with nested subfolders) pair across folders; mask-only trees are mirrored but not treated as scenes. Pass `--dest_dir …/Zenodo-Dataset_final` so the output tree mirrors the input under a new root.

---

## Execution order (gates — do not reorder)

### Step 0 — Raw folder audit (before anything else)
```bash
python qa_verification/raw_audit/run_raw_folder_audit.py \
  --root <path/to/extracted/zenodo/folder>
```
- Standalone; no pipeline state; writes `raw_folder_audit_report.json` under `--root` (or `--out`).
- Exit 0 = ready; 2 = not ready (e.g. open failures).
- **Read report fully before Step 1. Apply §4.1 decision rules.**

### Step 1 — Full pipeline run (skip `extract`)
```bash
python sar_dataset_pipeline.py run \
  --stage_dir <path/to/extracted/zenodo/folder> \
  --dest_dir /path/to/Zenodo-Dataset_final \
  --output_archive data/master_dataset_v1.7z \
  --workers 8 \
  --seed 42
```
- Order: `scan → geo → features → plan → convert → metadata → split → archive → validate`
- Exit **0** = pass; **2** = validate failed (apply §4.2: at most one bounded retry with documented override).
- Capture full stdout/stderr to a log file.

### Step 2 — Handoff scripts (repo root, this order)
```bash
python scripts/join_wind.py                 # default --source mock (expected)
python scripts/generate_ais.py
python scripts/generate_model2_pairs.py --pairs 2500   # default --backend simple
```
- Mock/simple defaults are **expected** (§4.3), not failures. Confirm outputs exist and are flagged synthetic/mock.

### Step 3 — Independent QA (against real output)
```bash
QA_STAGE_DIR=<path/to/extracted/zenodo/folder> \
QA_DEST_DIR=/path/to/Zenodo-Dataset_final \
QA_STATE_DIR=data/state \
QA_FEATURES_DIR=data/features \
QA_SPLITS_DIR=data/splits \
QA_CORRIDORS=data/reference/shipping_corridors.geojson \
python qa_verification/run_all_verification.py --seed 42
```
- Exit 0 pass / 2 hard-fail / 1 harness error. Read `verification_report.json` in full.
- Hard-fails: characterize only (§4.4) — do **not** fix data/code; list count + example path.

### Step 4 — Status report (ALWAYS write)
- File: `BUILD_STATUS_<date>.md` using brief §5 template.
- Never end a session without it, even on early halt.

---

## Definition of “done” (brief §2)

1. Step 0: `ready for pipeline: yes` **or** §4.1 quarantine documented.
2. Step 1: pipeline exit **0** (not 2).
3. Step 3: `run_all_verification.py` exit **0** (zero hard-fails). Soft-fails OK — list them.
4. Step 4: report written.

If 1–3 fail: stop per §4, write honest report. Clear stop > forced “done”.

---

## Guardrails (never violate — brief §3)

1. Never edit `proced/` or `qa_verification/` to make a failing check pass (except trivial path/env typo — log it).
2. Never delete/move/overwrite raw extracted source (quarantine = report paths only).
3. Never mark verification passed if any hard-fail in `verification_report.json`.
4. Never fabricate AIS/wind/currents — use `join_wind --source mock`, `generate_model2_pairs --backend simple`.
5. Max **one** auto-retry with adjusted param, only where §4 says so.
6. **Never run `extract`.**

---

## Decision rules (brief §4) — quick card

| Situation | Action |
|---|---|
| **4.1** <5% corrupt/unopenable | Quarantine paths in report; proceed Step 1 |
| **4.1** >~5% corrupt or empty sibling folders | **STOP** — incomplete multi-part extraction; wait for human |
| **4.1** Orphan image/mask | Proceed; log count (pipeline treats as unlabeled) |
| **4.1** Duplicates | Log + proceed; human decides dedupe later |
| **4.2** Pipeline exit 2, known domain/calib mismatch | **One** retry with documented `--calibrated-match` / `--uncalibrated-match` / `--value-domain` |
| **4.2** Anything else or retry fails | **STOP**; paste validation error verbatim |
| **4.3** Handoff mock/simple | Expected; confirm flags; proceed |
| **4.4** QA hard-fail | Characterize (check, count, example); optional “provisionally usable subset” note; **no repackage** |
| **4.4** QA soft-fail | List only; never blocks |
| **4.5** Unrecognized / ambiguous | **Do not guess.** Observe + record full error → report |

---

## Builder-PC facts (for report / readiness)

- Tests: **111/111** (86 pipeline + 25 QA), boundary clean (`qa_verification` never imports `proced/`).
- Asymmetric Zenodo layout: sibling `*_images` / `*_mask` trees + nested subfolders paired; folder-based mask detection; output mirrors structure under `--dest_dir` (use `Zenodo-Dataset_final`).
- `raw_audit/` modules: `discover_tree`, `check_integrity`, `check_pairing`, `check_duplicates`, `run_raw_folder_audit`.
- Env: Python 3.9.6, rasterio 1.4.3, py7zr 1.0.0, 8 CPUs; no system `7z` CLI (pipeline uses py7zr fallback).
- `data/` on builder PC is scaffolding only (`.gitkeep`, corridors, empty state) — **not** the dataset.
- **Progress bars everywhere**: `run` prints `▶ [3/9] features` / `✓ features 12.4s` + a master `pipeline 3/9` bar; stages/loops (pool folders, scenes, raw-audit files, QA modules) each have their own bar. Interactive terminal = live bar, `tee` logs = throttled `[progress]` lines, `PROGRESS=0` = silent.
- **CPU**: `--workers` defaults to every core (`WORKERS` env does the same in `run_zenodo_build.sh`); BLAS/OpenMP pinned to 1 thread per worker (processes do the parallelism), `--gdal-threads` per worker (default 2).
- **RAM**: full-scene arrays freed immediately after use (raw bands → dB, masks after plan/convert), pool results cleared as consumed, `gc.collect()` between pool chunks — memory does not stack up across scenes/folders.
- Related downloads (not the full extracted tree): mask `.7z`s + GEO CSVs under `~/Downloads` (2048² masks, Colab paths); notebook `Part1_Zenodo_SeaSentinel.ipynb` points at Zenodo record **8346860** (`01_Train_Val_Oil_Spill_images.7z`).

---

## Progress checklist

- [x] Understand brief §0–§6; notes + runner prepared on builder PC
- [x] Topology confirmed: clone Proced → dataset PC; path = run argument
- [ ] Copy/clone this repo onto dataset PC (`run_zenodo_build.sh` included)
- [ ] `pip install -r requirements.txt` on dataset PC
- [ ] Set `ZENODO_ROOT` / pass path → Step 0 raw audit + apply §4.1
- [ ] Step 1 pipeline run seed 42 → exit code + log
- [ ] Step 2 handoff scripts ×3 → confirm outputs
- [ ] Step 3 QA `run_all_verification` → exit + `verification_report.json`
- [ ] Step 4 `BUILD_STATUS_<date>.md` (runner always writes this)

---

## Open questions for human

1. ~~Exact absolute path of extracted tree on the dataset PC?~~ → supplied as `$ZENODO_ROOT` / CLI arg at run time.
2. ~~How is execution transferred?~~ → **clone Proced onto dataset PC**; run `./run_zenodo_build.sh <path>` there.
3. Expected worker count on dataset PC (brief says `--workers 8`)? → runner/CLI now default to **every CPU core**; `export WORKERS=8   # brief §1 value; omit to use every CPU core of the machine` restores the brief's literal value.
4. Disk headroom for `Zenodo-Dataset_final` + `master_dataset_v1.7z`?

---

## Dataset PC — ready commands (after `git clone` / copy)

```bash
cd Proced
pip install -r requirements.txt

export ZENODO_ROOT=/path/to/extracted/zenodo/folder   # ← set when known
export SEED=42
export WORKERS=8

# Full gated run (Steps 0→4): logs under logs/, report BUILD_STATUS_<date>.md
./run_zenodo_build.sh "$ZENODO_ROOT"

# Or step-by-step (same as brief §1):
python qa_verification/raw_audit/run_raw_folder_audit.py --root "$ZENODO_ROOT"
python sar_dataset_pipeline.py run \
  --stage_dir "$ZENODO_ROOT" \
  --dest_dir "${DEST_DIR:-Zenodo-Dataset_final}" \
  --output_archive data/master_dataset_v1.7z \
  --workers "$WORKERS" --seed "$SEED" 2>&1 | tee logs/step1_pipeline.log
python scripts/join_wind.py
python scripts/generate_ais.py
python scripts/generate_model2_pairs.py --pairs 2500
QA_STAGE_DIR="$ZENODO_ROOT" \
QA_DEST_DIR="${DEST_DIR:-Zenodo-Dataset_final}" \
QA_STATE_DIR=data/state \
QA_FEATURES_DIR=data/features \
QA_SPLITS_DIR=data/splits \
QA_CORRIDORS=data/reference/shipping_corridors.geojson \
python qa_verification/run_all_verification.py --seed "$SEED" 2>&1 | tee logs/step3_qa.log
# Then fill BUILD_STATUS_<date>.md from brief §5
```

`run_zenodo_build.sh` implements the gates: stops on §4.1 >5% corrupt / multi-part folder gaps (exit 3); on pipeline exit 2 it **does not auto-guess** a retry flag — it writes `BUILD_STATUS_*.md` with the validation reason so a human can apply the one allowed §4.2 override; always attempts Step 4 report.

---

## Log locations (fill during run)

- Step 0 report: ``
- Step 1 log: ``
- Step 1 exit: ``
- Step 2 outputs: ``
- Step 3 report: ``
- Step 3 exit: ``
- Final: `BUILD_STATUS_` ``
