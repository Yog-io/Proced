# HANDOVER REPORT — Block 1 SAR Oil-Spill Dataset System ("Proced")

> Purpose of this document: give another AI agent (or engineer) a complete,
> self-contained briefing of what exists in this repository, how the pieces
> fit, and exactly how to run it. Read this first, then the spec docs listed
> in §2 when you need normative detail.

---

## 1. What this system is

A production pipeline that turns raw Sentinel-1 / SAR oil-spill image trees
(Zenodo, Kaggle, etc.) into a **ready-to-train master dataset**:

* crops 256×256 PNG tiles (16-bit calibrated dB or native 8-bit) + binary masks
* full per-crop metadata (GEODATA.csv + root metadata.csv)
* full-extent shape + radiometric features (C.0) for the look-alike classifier
* group-safe train/val/test splits (no scene leakage)
* `.7z` archive, validation checklist, and an **independent QA suite** that
  re-computes everything from raw inputs without importing pipeline code.

**Repo**: `/Users/yogesh/Desktop/Proced` · git `origin = https://github.com/Yog-io/Proced.git`
· branch `main` · HEAD `3aab8e1` (3 commits, pushed, clean tree).

| commit | content |
|---|---|
| `1ffa075` | initial pipeline + QA suite (111 files) |
| `ce9c11c` | asymmetric Zenodo sibling `*_images`/`*_mask` tree support |
| `3aab8e1` | progress bars per step/stage, all-core CPU defaults, immediate RAM freeing (111 tests) |

Two-machine topology (important):
* **Builder PC** (this machine, macOS): code + tests only; `data/` is scaffolding, NOT the dataset.
* **Dataset PC**: has the extracted Zenodo tree; Proced is cloned there; the tree path is passed at run time (`$ZENODO_ROOT`).

---

## 2. Specification documents (source of truth — do not contradict)

| file | role |
|---|---|
| `archi.md` | architecture (stage order, memory/concurrency guardrails §2, validation checklist §5, QA layout §6) |
| `BLOCK1_DATA_TEAM_GUIDE.md` | data-team guide (Task 1.x, crop rules §2.2, splits Task 1.5) |
| `DATASET_FORMAT_REFINEMENT_REPORT.md` | format contract (C.0 features, C.2 dB formula, C.1/C.3 crop geometry, C.5 splits) |
| `DATASET_VERIFICATION_ARCHITECTURE.md` | QA spec — **user-edited**; §8 = raw folder audit |
| `AGENT_ORCHESTRATION_BRIEF.md` | **unmodifiable** overnight Zenodo build brief (Steps 0–4, §3 guardrails, §4 stop gates) |
| `AGENT_ORCHESTRATION_NOTES.md` | our working notes / ready-to-run commands for the dataset PC |
| `README.md` | developer-facing overview, issue log, optimisations, deliverables map |

First three are FINAL/unmodifiable. The brief (§3) forbids: editing source to
make checks pass, deleting/moving raw data, marking QA passed with hard-fails,
fabricating AIS/wind data, running `extract` on the dataset PC, more than one
auto-retry, and omitting `BUILD_STATUS_<date>.md`.

---

## 3. Architecture & data flow

```
raw extracted tree ──┐
                     ▼
 Stage 1  scan      catalog.json            (tree parse, VV/VH↔mask pairing,
                                             calibration + label + value-domain)
 Stage 3a geo       scene_geo.json          (real CRS/affine, else corridor-synthetic)
 Stage 2  features  instances.csv           (full-extent C.0 features per mask instance)
 Stage 3b plan      crop_plans.json         (where every 256² window goes, + negatives)
 Stage 4  convert   dest tree + crop_rows.json (windowed tiling → PNG + GEODATA.csv)
 Stage 5  metadata  metadata.csv + features/*.csv
 Stage 6a split     splits/{train,val,test,manifest}.csv   (grouped by source scene)
 Stage 6b archive   output .7z
 Stage 6c validate  checklist (archi §5) → exit 2 on failure
                     │
                     ▼
        independent QA (qa_verification/) → verification_report.json
```

* `extract` (Stage 0) is a **separate subcommand**; `run` never auto-extracts.
* Every stage is independently runnable; intermediate artifacts live under `--state_dir`
  (default `data/state/`: `run_config.json`, `catalog.json`, `scene_geo.json`,
  `instances.csv`, `crop_plans.json`, `crop_rows.json`).
* Code layout: `proced/` = pipeline (26 modules ≈ 4.5k LOC),
  `qa_verification/` = independent QA (≈ 5.2k LOC incl. tests),
  `tests/` = pytest suite (≈ 2.2k LOC), `scripts/` = handoff utilities,
  `lib/` = ERA5/Copernicus fetch wrappers.

**Hard boundary rule**: `qa_verification/` must NEVER import `proced/`
(enforced by regex `^\s*(?:from\s+proced[\s.]|import\s+proced\b)` in tests;
QA recomputes things independently on purpose). Verified `boundary: OK`.

---

## 4. Stage reference

| Stage | subcommand | module(s) | input → output |
|---|---|---|---|
| 0 | `extract` | `proced/archive.py` | `.7z` → `--stage_dir` (7z CLI or py7zr fallback) |
| 1 | `scan` | `proced/catalog.py`, `stage_scan.py` | raw tree → `catalog.json` + value domains + **full tree mirror** under `--dest_dir` |
| 3a | `geo` | `proced/geo.py`, `stage_geo.py` | catalog → `scene_geo.json` (real geo if present; else corridor-based synthetic + provenance) |
| 2 | `features` | `proced/features.py`, `stage_features.py` | masks + bands → `instances.csv` (C.0: area, perimeter, aspect, boundary complexity, damping ratio, GLCM, NDPI …) |
| 3b | `plan` | `proced/crops.py`, `stage_plan.py` | instances + masks → `crop_plans.json` (object tiles w/ jitter, overflow tiles, edge truncation, negatives) |
| 4 | `convert` | `proced/radiometry.py`, `raster_io.py`, `stage_convert.py` | plans → PNGs + per-folder `GEODATA.csv` + `crop_rows.json` (parallel, one worker owns a whole folder → no GEODATA races; resume unless `--force`) |
| 5 | `metadata` | `proced/metadata.py`, `stage_metadata.py` | state → root `metadata.csv`, `features/{shape,full_extent}_features.csv`, `synthetic_geo_assignments.csv` |
| 6a | `split` | `proced/splits.py` | crop rows → `splits/*.csv` (grouped by `crop_source_scene_id`, stratified by dominant label, 70/15/15, seed 42) |
| 6b | `archive` | `proced/archive.py` | dest tree → `.7z` |
| 6c | `validate` | `proced/validate.py` | dest tree → checklist report; `run` exits **2** if it fails |

`run` executes scan → geo → features → plan → convert → metadata → split →
archive → validate in order with per-stage progress (see §10).

---

## 5. Output dataset contract

```
<dest_dir>/                     # default: sibling "Zenodo-Dataset_final" of the input tree
├── <mirror of input tree>/     # every source folder re-created (directory-equivalence check)
│   └── …/crop_XXXX_VV.png      # 16-bit uint16 (calibrated) or 8-bit (native)
│       …/crop_XXXX_VH.png      # (VH only if dual-pol)
│       …/crop_XXXX_mask.png    # binary 0/255 PNG
│       …/GEODATA.csv           # one row per crop, GEODATA_COLUMNS first
├── metadata.csv                # master table = GEODATA_COLUMNS + MASTER_EXTRA_COLUMNS
├── pipeline_summary.json       # counts, splits, validation, config, elapsed
├── verification_report.json    # written by the independent QA suite (Step 3)
├── splits/{train,val,test,manifest}.csv   # image_id + group; no group spans splits
└── features/
    ├── shape_features.csv              # C.0 shape-only classifier rows
    ├── full_extent_features.csv        # + radiometric + geo-joined
    ├── synthetic_geo_assignments.csv   # provenance of synthetic coordinates
    └── lookalike_training_data.csv     # from scripts/join_wind.py (hand to Model Training)
```

**Radiometry (C.2)**: calibrated float scenes → `db = (px_uint16 / 65535) * 30 − 30`
(−30…0 dB), recorded as `db_conversion_formula_version = "v1_linear_neg30_to_0"`.
Uncalibrated scenes → native 8-bit, `"native_8bit_no_db"`. Domain is auto-detected
(`db` / `power` / `amplitude` / `native`), overridable with `--value-domain`.

**Labels**: default label comes from the folder path
(`*lookalike*/*look-alike*/*look_alike*` → `lookalike`; `*oil_spill*`/`oil-spill`/`oilspill` → `oil`;
otherwise → `oil`), overridable with `--label-override kaggle=oil` style flags.
Mask pixel value `1` = oil, `2` = look-alike. Instances below
`min_instance_area_px=20` ignored. Background crops come from negative placement —
scenes without masks contribute none unless `--negatives-from-unlabeled`.

**Crop geometry (C.1/C.3)**: 256×256, jitter ±25 px, overflow object → multiple
tiles (stride defaults to 50 % overlap = 128 px), negatives placed ≥300 px from
any positive (`neg_ratio 0.4`, ≤4 per clean scene, ≤64/scene), edge-touching
instances flagged `truncated_by_scene_edge` (kept for Model 1, excluded from the
classifier table).

**Key columns**: `crop_id`, `crop_source_scene_id` (split group), `is_calibrated`,
`has_dual_pol`, `is_synthetic_location`, `crop_bbox_corners`, `is_partial_object`,
`parent_group_id`, `truncated_by_scene_edge`, `db_conversion_formula_version`,
`label`, `full_extent_*`, `lat/lon/timestamp_utc`, `crop_xoff/crop_yoff`.

---

## 6. Input handling (incl. the Zenodo layout)

* Recognised rasters: `.tif .tiff .png .jp2 .img .jpg .jpeg`
  (`proced/config.py::RASTER_EXTENSIONS`).
* Mask detection: file-name tokens (`mask,label,labels,gt,lbl,ann`) **and**
  folder-based classification: any path under `*_images/` is an image, under
  `*_mask/` is a mask (asymmetric trees where an image has no same-folder mask).
* Pairing order (`catalog._find_mask`): same folder → **parallel sibling tree**
  (`…_images/x.tif` ↔ `…_mask/x.tif`, any depth) → conventional subdirs
  (`masks/`, `label/`, `ground_truth/`, …) → global stem index.
* Zenodo source layout (7 top-level folders, nested `.tif` subfolders):
  `01_Train_Val_{Lookalike,No_Oil,Oil_Spill}_{images,mask}`,
  `02_Test_images_and_ground_truth`. Output must mirror it — hence
  `--dest_dir …/Zenodo-Dataset_final` (the runner does this by default).

---

## 7. How to run

### A) Full gated build on the dataset PC (brief Steps 0–4)

Windows (PowerShell / cmd — no bash, no env vars needed; single-dash args):

```powershell
cd Proced
pip install -r requirements.txt
python run_zenodo_build.py -path D:\Zenodo-dataset -seed 42 -workers 8
#   optional: -dest D:\Zenodo-Dataset_final   (default: sibling folder of -path)
#             -pairs 2500                     (Step 2 synthetic Model-2 pairs)
```

macOS/Linux (same script, or the POSIX wrapper):

```bash
cd Proced
pip install -r requirements.txt

export ZENODO_ROOT=/path/to/extracted/zenodo/folder   # already extracted — never `extract`
export SEED=42
export WORKERS=8          # optional; default = every CPU core

./run_zenodo_build.sh "$ZENODO_ROOT"
# equivalent: python3 run_zenodo_build.py -path "$ZENODO_ROOT" -seed 42 -workers 8
# → logs/step{0..3}_*.log,  BUILD_STATUS_<date>.md,
#   $DEST_DIR/{pipeline_summary.json, verification_report.json}
# exit 0 = all gates passed · 2 = QA hard-fail / validate fail
#        · 3 = Step 0 STOP (incomplete extraction, §4.1) · 1 = harness error
```

`DEST_DIR`/`-dest` defaults to `$(dirname $ZENODO_ROOT)/Zenodo-Dataset_final`.

### B) Step-by-step (same thing, transparent)

```bash
python3 qa_verification/raw_audit/run_raw_folder_audit.py --root "$ZENODO_ROOT" --out audit.json
python3 sar_dataset_pipeline.py run \
  --stage_dir "$ZENODO_ROOT" --dest_dir "$DEST_DIR" \
  --output_archive data/master_dataset_v1.7z \
  --workers "$WORKERS" --seed "$SEED" 2>&1 | tee logs/step1_pipeline.log
python3 scripts/join_wind.py                # mock winds (flagged) — allowed, never fabricate unflagged
python3 scripts/generate_ais.py             # mock AIS
python3 scripts/generate_model2_pairs.py --pairs 2500   # backend: simple (flagged synthetic_*)
QA_STAGE_DIR="$ZENODO_ROOT" QA_DEST_DIR="$DEST_DIR" QA_STATE_DIR=data/state \
QA_FEATURES_DIR=data/features QA_SPLITS_DIR=data/splits \
QA_CORRIDORS=data/reference/shipping_corridors.geojson \
  python3 qa_verification/run_all_verification.py --seed "$SEED" 2>&1 | tee logs/step3_qa.log
```

### C) Single stages / development

```bash
python3 sar_dataset_pipeline.py scan --stage_dir <tree>     # … geo features plan
python3 sar_dataset_pipeline.py convert --stage_dir <tree>  # metadata split archive validate
python3 sar_dataset_pipeline.py validate                    # re-check an output tree
python3 sar_dataset_pipeline.py run --help                  # all flags
```

### D) Tests

```bash
pytest tests/ -q                 # 127 tests (86 pipeline + 16 runner + 25 QA)
pytest tests/ -q -m "not slow"    # skip end-to-end
```

---

## 8. CLI flags worth knowing (`add_common_args`)

| flag | default | meaning |
|---|---|---|
| `--stage_dir` | `data/scratch_unpacked` | extracted input tree |
| `--dest_dir` | `data/master_dataset_processed` | output tree (mirrors input) |
| `--state_dir` | `data/state` | intermediate artifacts |
| `--output_archive` | `data/master_dataset_v1.7z` | Stage 6b target |
| `--workers` | **every CPU core** | spawn-process pool |
| `--gdal-threads` | `2` | `GDAL_NUM_THREADS` per worker |
| `--gdal-cache-mb` | 256 | GDAL cache per worker (archi §2) |
| `--pool-chunk` | 16 | jobs per executor before recycling workers (memory guardrail) |
| `--tile-size/--jitter/--stride/--neg-buffer-px/--neg-ratio/--max-tiles-per-object` | 256 / 25 / 50 % / 300 / 0.4 / 48 | crop geometry |
| `--value-domain` | `auto` | `auto\|db\|power\|amplitude` |
| `--seed` | 42 | all stochastic choices are per-`(folder, scene_id)` stable |
| `--calibrated-match/--uncalibrated-match/--label-override` | – | path-substring overrides |
| `--force` | off | reprocess folders that already have `GEODATA.csv` |
| `--no-pack` / `--no-validate` | off | skip Stage 6b/6c |

All tunables live in **`proced/config.py::PipelineConfig`** (single source of truth).

---

## 9. Independent QA system (`qa_verification/`)

**Step 0 — raw folder audit** (runs *before* the pipeline, gates the run):
`raw_audit/{discover_tree,check_integrity,check_pairing,check_duplicates,run_raw_folder_audit}.py`
→ `raw_folder_audit_report.json`. Opens **every** raster, samples dtype/CRS/domain,
pairs images↔masks across sibling trees, hashes for byte-identical duplicates,
flags folder gaps/resolution outliers. §4.1 stop gates: >5 % unopenable → STOP;
multi-part gap signature → STOP; otherwise quarantine-and-proceed.
CLI: `--root <tree> [--out f.json] [--no-sample-values]`.

**Step 3 — verification suite** (`run_all_verification.py`):
* independent recomputes: `stage0_extract, stage1_scan, stage2_features,
  stage3a_geo, perpendicular_offset, stage3b_plan, stage4_db_roundtrip,
  stage6_splits_independent`
* structural: `stage5_schema_nulls, stage5_cross_file_joins, stage6_split_leakage`
* optional (on by default, `--skip-*` to disable): `visual_spotcheck,
  stats_histograms, stats_geo_scatter, adversarial_graceful_failure,
  adversarial_determinism`
* flags: `--sample-size 100 --seed 42 --only a,b --skip-dashboards --skip-visual
  --zenodo-expected --json`
* env: `QA_STAGE_DIR, QA_DEST_DIR, QA_STATE_DIR, QA_FEATURES_DIR, QA_SPLITS_DIR,
  QA_CORRIDORS, QA_OUTPUT_ARCHIVE, QA_SOURCE_ARCHIVE`
* exit codes: **0** pass · **2** ≥1 hard-fail · **1** harness error.
  Report: `verification_report.json` (dest dir, cwd fallback).

Tolerances (soft unless noted): dB round-trip `DB_TOL=0.05`, pixel tolerance
`PX_TOL=3`, label-count band `LABEL_COUNT_TOLERANCE=±50 %`, nulls in required
columns / split leakage / non-binary masks / bad bearing = hard fails.

---

## 10. Performance, CPU, memory, progress (as of `3aab8e1`)

* **CPU**: `--workers` (CLI *and* `PipelineConfig` default) = `os.cpu_count()`;
  `run_zenodo_build.py` (and the `.sh` wrapper) default `-workers`/`WORKERS`
  to all cores. Each worker pins
  BLAS/OpenMP (`OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_MAXIMUM_THREADS`) and OpenCV to
  **1 thread** so N processes don't oversubscribe; GDAL threads per worker are
  capped (`--gdal-threads`, default 2) with 256 MB cache each. The `run:` log
  prints `workers=… (cpu_count=…), gdal_threads=…, …` so you can verify.
* **Memory**: raw full-scene bands are dropped as soon as dB/linear copies
  exist; scene masks are nulled after plan/convert; pool results are cleared as
  they're consumed (no duplicate row lists); parent runs `gc.collect()` between
  pool chunks; executors recycle every `--pool-chunk` jobs so worker RSS stays
  bounded (archi §2).
* **Progress**: `proced/progress.py` (zero-dependency) — `run` prints
  `▶ [3/9] features` / `✓ features 12.4s` plus a master `pipeline 3/9` bar;
  every heavy loop has its own bar (pool folders/scenes, value-domain probes,
  geo scenes, metadata rows, validate folders, archive entries, raw-audit file
  open + hashing, QA modules). Interactive terminal → live bar; piped/`tee` logs
  → throttled `[progress] …` lines; `PROGRESS=0` → silent.
  `qa_verification/_progress.py` is an intentional copy (boundary rule).

---

## 11. Scripts / handoff outputs (brief Step 2)

| script | output | notes |
|---|---|---|
| `scripts/audit_datasets.py` | `data/raw/DATASET_AUDIT.md` | Task 1.1 |
| `scripts/unify_datasets.py` | `data/processed/*` | Task 1.3 supplementary sets |
| `scripts/assign_synthetic_geo.py` | provenance CSV | §2.2 |
| `scripts/join_wind.py` | `data/features/lookalike_training_data.csv` | Task 1.7 — **default `--source mock`, rows flagged**; `--source era5` needs CDS creds |
| `scripts/generate_ais.py` | `data/ais/ais_tracks.csv` | Task 1.8 — mock, allowed by brief |
| `scripts/generate_model2_pairs.py` | `data/synthetic/model2_pairs/` | Task 1.10 — default `--backend simple`, fields flagged `synthetic_*` |
| `scripts/validate_dataset.py` | checklist | standalone Stage 6c |
| `lib/fetch_winds.py`, `lib/fetch_currents.py` | ERA5 / Copernicus wrappers | optional deps commented in `requirements.txt` |

---

## 12. Guardrails & known limitations (state these honestly, never paper over)

* Never run `extract` on the dataset PC (tree is already extracted); never
  delete/move raw data; never edit code to make a check pass; never report QA
  as passed while hard-fails exist; never emit unflagged fabricated AIS/wind;
  at most one bounded retry (brief §4.2); always leave `BUILD_STATUS_<date>.md`.
* Synthetic coordinates are a deliberate approximation — `is_synthetic_location`
  is load-bearing on every row; they must never be presented as evidentiary
  spill locations.
* Mock winds / simple-physics pairs are **flagged** in their outputs; switch to
  `--source era5` / `--backend opendrift` when credentials & forcing data exist.
* No system `7z`/`gdalinfo` on this machine → py7zr + rasterio fallbacks (7z CLI
  is used when present, multi-threaded). `folium` not installed (dashboards
  degrade gracefully).
* Soft findings (class imbalance, visual spot-check flags, mock-wind rows) are
  reported but do not block handoff; hard findings do.

---

## 13. Test suite (127)

| file | # | covers |
|---|---|---|
| `test_catalog.py` | 15 | scan, pairing incl. asymmetric sibling trees, labels |
| `test_qa_verification.py` | 25 | raw audit + every QA module on fixtures (the "25 QA") |
| `test_runner.py` | 16 | `run_zenodo_build.py`: single-dash CLI, §4.1 gates, status writer, log streaming |
| `test_progress.py` | 12 | bars, stage helper, CPU defaults, BLAS pinning, boundary |
| `test_radiometry.py` | 10 | domain detection, dB round-trip |
| `test_geo.py` | 10 | corridors, perpendicular offset, timestamps |
| `test_features.py` | 7 | C.0 feature extraction |
| `test_pipeline_e2e.py` | 6 | full `run` + standalone stage chain (`slow`) |
| `test_crops.py` | 6 | crop planning/negatives |
| `test_metadata.py` / `test_splits.py` | 5 / 5 | metadata schema, split hygiene |
| `test_state.py` / `test_validate.py` | 4 / 4 | artifacts, checklist |
| `test_archive.py` | 2 | 7z round-trip |

Run: `pytest tests/ -q` → `127 passed`. Before changing anything, also run the
boundary regex check (§3) and `bash -n run_zenodo_build.sh`.

---

## 14. Environment

Python 3.9.6 (macOS, 8 cores) · rasterio 1.4.3 · numpy, pandas, opencv,
scikit-image, shapely, py7zr, Pillow, pyyaml, global-land-mask, pytest ·
tqdm present but **not required** (progress code is self-contained) ·
no `7z`/GDAL CLI · spawn-multiprocessing (keep `__main__` guards in scripts).

---

## 15. Orientation for the next agent

1. Read §2 spec docs for rules; `README.md` for the issue-by-issue history
   (each gotcha is documented with its fix).
2. Day-to-day dev: edit `proced/…`, run `pytest tests/ -q` (127 expected),
   keep `qa_verification` import-free of `proced`, keep
   `PipelineConfig` as the only place tunables are defined.
3. Before a dataset-PC run: verify `git log` matches this report, re-run tests,
   then follow §7A. Watch the Step-0 gate result before letting Step 1 start.
4. After a run: `BUILD_STATUS_<date>.md` + `logs/` are the deliverable evidence;
   `verification_report.json` must have zero hard-fails for a complete build.
5. If something fails: the stop reason is always recorded (exit code + status
   file + logs). Diagnose from those; do not delete or "clean up" raw data.
